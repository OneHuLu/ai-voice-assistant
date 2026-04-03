import os
import subprocess
import time
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
    resolve_path,
    get_project_root,
)


CONFIG_PATH = get_config_path()
PROJECT_ROOT = get_project_root(__file__)


def _cfg(file_cfg: Dict[str, Any], env_key: str, file_key: str, default: str) -> str:
    """按"环境变量 > 配置文件 > 默认值"读取配置项。"""
    return get_config_value(file_cfg, env_key, file_key, default)


_llm_cfg = load_config(CONFIG_PATH).get("llm")
_llm_cfg = _llm_cfg if isinstance(_llm_cfg, dict) else {}

# 读取参数信息
LLAMA_CPP_BIN = _cfg(
    _llm_cfg,
    "LLAMA_CPP_BIN",
    "llama_cpp_bin",
    "models/llama.cpp/build/bin/llama-server",
)
MODEL_PATH = _cfg(
    _llm_cfg,
    "LLM_MODEL_PATH",
    "model_path",
    "models/llm_models/Qwen2.5-14B-Instruct-IQ4_XS.gguf",
)
LLAMA_CPP_BIN = resolve_path(LLAMA_CPP_BIN, PROJECT_ROOT)
MODEL_PATH = resolve_path(MODEL_PATH, PROJECT_ROOT)

if not os.path.exists(LLAMA_CPP_BIN):
    raise FileNotFoundError(f"llama-server 不存在：{LLAMA_CPP_BIN}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"LLM 模型不存在：{MODEL_PATH}")

LLAMA_HOST = _cfg(_llm_cfg, "LLAMA_HOST", "llama_host", "127.0.0.1")
LLAMA_PORT = int(_cfg(_llm_cfg, "LLAMA_PORT", "llama_port", "8040"))
N_GPU_LAYERS = int(_cfg(_llm_cfg, "LLM_N_GPU_LAYERS", "n_gpu_layers", "99"))
CTX_SIZE = int(_cfg(_llm_cfg, "LLM_CTX_SIZE", "ctx_size", "8192"))
TEMP = float(_cfg(_llm_cfg, "LLM_TEMP", "temperature", "0.9"))
REPEAT_PENALTY = float(_cfg(_llm_cfg, "LLM_REPEAT_PENALTY", "repeat_penalty", "1.05"))

FASTAPI_HOST = _cfg(_llm_cfg, "LLM_API_HOST", "api_host", "0.0.0.0")
FASTAPI_PORT = int(_cfg(_llm_cfg, "LLM_API_PORT", "api_port", "8041"))

# llama 服务启动
def start_llama_server():
    """启动本地 llama-server 子进程。"""
    cmd = [
        LLAMA_CPP_BIN,
        "-m",
        MODEL_PATH,
        "--host",
        LLAMA_HOST,
        "--port",
        str(LLAMA_PORT),
        "--n-gpu-layers",
        str(N_GPU_LAYERS),
        "--ctx-size",
        str(CTX_SIZE),
        "--temp",
        str(TEMP),
        "--repeat-penalty",
        str(REPEAT_PENALTY),
        "--chat-template",
        "qwen",
    ]
    print("Starting llama-server:", " ".join(cmd))
    return subprocess.Popen(cmd)


llama_process = start_llama_server()
time.sleep(10)

app = FastAPI(title="Local Qwen LLM Proxy")

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


LLAMA_SERVER_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1/chat/completions"


@app.post("/llm/predict")
async def predict(request: LLMRequest):
    """代理请求到 llama-server 的 chat/completions 接口。"""
    try:
        if not request.messages:
            messages = [{"role": "user", "content": request.prompt}]
            if request.prompt and not any(msg.get("role") == "system" for msg in messages):
                messages.insert(0, {"role": "system", "content": "You are Qwen, a helpful assistant."})
        else:
            messages = request.messages

        payload = {
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_k": request.top_k,
            "stream": False,
        }
        resp = requests.post(LLAMA_SERVER_URL, json=payload, timeout=60)
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"llama-server error: {resp.text}")
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    try:
        uvicorn.run(app, host=FASTAPI_HOST, port=FASTAPI_PORT)
    finally:
        print("Stopping llama-server...")
        llama_process.terminate()
