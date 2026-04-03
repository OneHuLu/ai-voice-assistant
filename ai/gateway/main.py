import os
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ai.utils.config_helper import get_config_path, load_config, get_config_value


def _base_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    return f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else raw_url


CONFIG_PATH = get_config_path()
_gateway_cfg = load_config(CONFIG_PATH).get("gateway") or {}


def _cfg(file_cfg: Dict[str, Any], env_key: str, file_key: str, default: str) -> str:
    """按"环境变量 > 配置文件 > 默认值"读取配置项。"""
    return get_config_value(file_cfg, env_key, file_key, default)


STT_URL = _cfg(_gateway_cfg, "STT_URL", "stt_url", "http://127.0.0.1:8000/transcribe")
LLM_URL = _cfg(_gateway_cfg, "LLM_URL", "llm_url", "http://127.0.0.1:8041/llm/predict")
TTS_URL = _cfg(_gateway_cfg, "TTS_URL", "tts_url", "http://127.0.0.1:8030/tts/synthesize")
GATEWAY_HOST = _cfg(_gateway_cfg, "GATEWAY_HOST", "host", "0.0.0.0")
GATEWAY_PORT = int(_cfg(_gateway_cfg, "GATEWAY_PORT", "port", "8010"))

app = FastAPI(title="AI Voice Assistant Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    prompt: str = ""
    max_tokens: int = 512
    temperature: float = 0.7
    tts_model: str = "cosyvoice-v3-flash"
    tts_voice: str = "longanyang"


class TTSProxyRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model: str = "cosyvoice-v3-flash"
    voice: str = "longanyang"


def _extract_reply_text(llm_resp: Dict[str, Any]) -> str:
    """兼容多种 LLM 返回结构，提取最终回复文本。"""
    choices = llm_resp.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content

    output = llm_resp.get("output") or {}
    out_choices = output.get("choices") or []
    if out_choices:
        msg = out_choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content

    return ""


def _build_messages(req: ChatRequest) -> List[Dict[str, str]]:
    if req.messages:
        return req.messages
    if req.prompt.strip():
        return [{"role": "user", "content": req.prompt.strip()}]
    raise HTTPException(status_code=400, detail="messages 或 prompt 至少传一个")


@app.get("/health")
def health() -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    targets = {
        "llm": f"{_base_url(LLM_URL)}/docs",
        "tts": f"{_base_url(TTS_URL)}/health",
        "stt": f"{_base_url(STT_URL)}/docs",
    }

    for name, url in targets.items():
        try:
            resp = requests.get(url, timeout=3)
            checks[name] = {"ok": resp.status_code < 500, "status_code": resp.status_code}
        except Exception as e:
            checks[name] = {"ok": False, "error": str(e)}

    return {"ok": True, "services": checks, "config_path": CONFIG_PATH}


@app.post("/chat/text")
def chat_text(req: ChatRequest) -> Dict[str, Any]:
    messages = _build_messages(req)

    llm_payload = {
        "messages": messages,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }

    try:
        llm_res = requests.post(LLM_URL, json=llm_payload, timeout=120)
        llm_data = llm_res.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"调用 LLM 服务失败：{e}")

    if llm_res.status_code != 200:
        raise HTTPException(status_code=llm_res.status_code, detail=llm_data)

    reply_text = _extract_reply_text(llm_data).strip()
    if not reply_text:
        raise HTTPException(status_code=502, detail="LLM 返回为空")

    return {
        "reply_text": reply_text,
        "llm_raw": llm_data,
    }


@app.post("/tts/synthesize/proxy")
def tts_proxy(req: TTSProxyRequest) -> Dict[str, Any]:
    payload = {
        "text": req.text,
        "model": req.model,
        "voice": req.voice,
    }
    try:
        tts_res = requests.post(TTS_URL, json=payload, timeout=120)
        tts_data = tts_res.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"调用 TTS 服务失败：{e}")

    if tts_res.status_code != 200:
        raise HTTPException(status_code=tts_res.status_code, detail=tts_data)

    return tts_data


@app.post("/chat/tts")
def chat_tts(req: ChatRequest) -> Dict[str, Any]:
    """执行完整链路：LLM 生成回复文本，再转 TTS 音频。"""
    chat_result = chat_text(req)
    reply_text = chat_result["reply_text"]

    tts_payload = {
        "text": reply_text,
        "model": req.tts_model,
        "voice": req.tts_voice,
    }
    try:
        tts_res = requests.post(TTS_URL, json=tts_payload, timeout=120)
        tts_data = tts_res.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"调用 TTS 服务失败：{e}")

    if tts_res.status_code != 200:
        raise HTTPException(status_code=tts_res.status_code, detail=tts_data)

    return {
        "reply_text": reply_text,
        "audio_base64": tts_data.get("audio_base64"),
        "audio_mime_type": tts_data.get("audio_mime_type", "audio/mpeg"),
        "tts_request_id": tts_data.get("request_id"),
        "llm_raw": chat_result.get("llm_raw"),
    }


if __name__ == "__main__":
    uvicorn.run(app, host=GATEWAY_HOST, port=GATEWAY_PORT)
