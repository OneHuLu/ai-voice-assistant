import json
import os
import subprocess
import time
from typing import Any, Dict

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


CONFIG_PATH = os.getenv(
    "AI_CONFIG_PATH",
    os.path.join(os.path.dirname(__file__), "..", "config.json"),
)


def _load_file_config() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _cfg(file_cfg: Dict[str, Any], env_key: str, file_key: str, default: str) -> str:
    # 优先级：环境变量 > 公共配置文件 > 默认值
    return os.getenv(env_key, str(file_cfg.get(file_key, default))).strip()


_llm_cfg = _load_file_config().get("llm")
_llm_cfg = _llm_cfg if isinstance(_llm_cfg, dict) else {}

# 读取参数信息
LLAMA_CPP_BIN = _cfg(_llm_cfg, "LLAMA_CPP_BIN", "llama_cpp_bin")
MODEL_PATH = _cfg( _llm_cfg, "LLM_MODEL_PATH", "model_path")
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
