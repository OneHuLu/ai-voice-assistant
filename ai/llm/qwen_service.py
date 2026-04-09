import os
from typing import Any, Dict

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai.utils.config_helper import (
    get_config_path,
    load_config,
    get_config_value,
)


CONFIG_PATH = get_config_path()
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"


def _cfg(file_cfg: Dict[str, Any], env_key: str, file_key: str, default: str) -> str:
    """按"环境变量 > 配置文件 > 默认值"读取配置项。"""
    return get_config_value(file_cfg, env_key, file_key, default)


_llm_cfg = load_config(CONFIG_PATH).get("llm") or {}
_llm_cfg = _llm_cfg if isinstance(_llm_cfg, dict) else {}
_root_cfg = load_config(CONFIG_PATH)

DASHSCOPE_API_KEY = _cfg(_root_cfg.get("tts") or {}, "DASHSCOPE_API_KEY", "dashscope_api_key", "")
DASHSCOPE_MODEL = _cfg(_llm_cfg, "DASHSCOPE_MODEL", "dashscope_model", "qwen-max-latest")

FASTAPI_HOST = _cfg(_llm_cfg, "LLM_API_HOST", "api_host", "0.0.0.0")
FASTAPI_PORT = int(_cfg(_llm_cfg, "LLM_API_PORT", "api_port", "8041"))

app = FastAPI(title="DashScope LLM Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LLMRequest(BaseModel):
    prompt: str = ""
    max_tokens: int = 512
    messages: list = []
    temperature: float = 0.9
    top_k: int = 40


@app.post("/llm/predict")
async def predict(request: LLMRequest):
    """调用阿里百炼（DashScope）API 生成回复。"""
    try:
        if not DASHSCOPE_API_KEY:
            raise HTTPException(status_code=500, detail="DASHSCOPE_API_KEY 未配置")

        # 构建消息
        if not request.messages:
            messages = [{"role": "user", "content": request.prompt.strip()}]
        else:
            messages = request.messages

        if not any(msg.get("role") == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": "You are Qwen, a helpful assistant."})

        # 调用 DashScope API
        payload = {
            "model": DASHSCOPE_MODEL,
            "input": {"messages": messages},
            "parameters": {
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "top_k": request.top_k,
                "result_format": "message",
            },
        }

        headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
        }

        resp = requests.post(
            f"{DASHSCOPE_BASE_URL}/services/aigc/text-generation/generation",
            headers=headers,
            json=payload,
            timeout=60,
        )

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        data = resp.json()
        return {
            "choices": [
                {
                    "message": data["output"]["choices"][0]["message"],
                    "finish_reason": "stop",
                }
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DashScope 调用失败: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host=FASTAPI_HOST, port=FASTAPI_PORT)
