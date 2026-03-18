# start_services.py
import subprocess
import time
import threading
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import uvicorn

# ================= 配置 =================
LLAMA_CPP_BIN = "/Users/dyx/Desktop/project/ai-voice-assistant/models/llama.cpp/build/bin/llama-server"
MODEL_PATH = "/Users/dyx/llama/llm_models/Qwen2.5-14B-Instruct-IQ4_XS.gguf"
LLAMA_HOST = "127.0.0.1"
LLAMA_PORT = 8080
N_GPU_LAYERS = 99
CTX_SIZE = 8192
TEMP = 0.9
REPEAT_PENALTY = 1.05

FASTAPI_HOST = "0.0.0.0"
FASTAPI_PORT = 8001

# ================= 启动 llama-server =================
def start_llama_server():
    cmd = [
        LLAMA_CPP_BIN,
        "-m", MODEL_PATH,
        "--host", LLAMA_HOST,
        "--port", str(LLAMA_PORT),
        "--n-gpu-layers", str(N_GPU_LAYERS),
        "--ctx-size", str(CTX_SIZE),
        "--temp", str(TEMP),
        "--repeat-penalty", str(REPEAT_PENALTY),
        "--chat-template", "qwen",  # 添加 Qwen 模型的 chat template
    ]
    print("Starting llama-server:", " ".join(cmd))
    return subprocess.Popen(cmd)

llama_process = start_llama_server()

# 给 llama-server 一点时间启动
time.sleep(10)

# ================= FastAPI 服务 =================
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
        # 如果 messages 为空，使用 prompt 构造默认消息
        if not request.messages:
            messages = [{"role": "user", "content": request.prompt}]
            # 添加 system prompt 以符合 Qwen 格式
            if request.prompt and not any(msg.get("role") == "system" for msg in messages):
                messages.insert(0, {"role": "system", "content": "You are Qwen, a helpful assistant."})
        else:
            messages = request.messages
        
        payload = {
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_k": request.top_k,
            "stream": False  # 可根据需求改成 True
        }
        resp = requests.post(LLAMA_SERVER_URL, json=payload, timeout=60)
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"llama-server error: {resp.text}")
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================= 启动 FastAPI =================
if __name__ == "__main__":
    try:
        uvicorn.run(app, host=FASTAPI_HOST, port=FASTAPI_PORT)
    finally:
        print("Stopping llama-server...")
        llama_process.terminate()