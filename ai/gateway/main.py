import logging
import os
import tempfile
from typing import Any, Dict, List
from urllib.parse import urlparse

import dashscope
import requests
import uvicorn
from dashscope.audio.asr import Recognition, RecognitionCallback
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ai.utils.config_helper import get_config_path, load_config, get_config_value, get_dashscope_api_key

logger = logging.getLogger(__name__)


def _base_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    return f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else raw_url


CONFIG_PATH = get_config_path()
_root_cfg = load_config(CONFIG_PATH)
_gateway_cfg = _root_cfg.get("gateway") or {}
_tts_cfg = _root_cfg.get("tts") or {}
_llm_cfg = _root_cfg.get("llm") or {}
_stt_cfg = _root_cfg.get("stt") or {}


def _cfg(file_cfg: Dict[str, Any], env_key: str, file_key: str, default: str) -> str:
    """按"环境变量 > 配置文件 > 默认值"读取配置项。"""
    return get_config_value(file_cfg, env_key, file_key, default)


# DashScope API Key（用于云端 LLM/TTS/STT）- 统一从根级别或环境变量读取
DASHSCOPE_API_KEY = get_dashscope_api_key(_root_cfg)
DASHSCOPE_TTS_MODEL = _cfg(_tts_cfg, "DASHSCOPE_TTS_MODEL", "model", "cosyvoice-v1")


DASHSCOPE_MODEL = _cfg(_llm_cfg, "DASHSCOPE_MODEL", "dashscope_model", "qwen-max-latest")
DASHSCOPE_STT_MODEL = _cfg(_stt_cfg, "DASHSCOPE_STT_MODEL", "dashscope_model", "paraformer-v2")

STT_URL = _cfg(_gateway_cfg, "STT_URL", "stt_url", "http://127.0.0.1:8000/transcribe")
LLM_URL = _cfg(_gateway_cfg, "LLM_URL", "llm_url", "http://127.0.0.1:8041/llm/predict")
TTS_URL = _cfg(_gateway_cfg, "TTS_URL", "tts_url", "http://127.0.0.1:8030/tts/synthesize")
GATEWAY_HOST = _cfg(_gateway_cfg, "GATEWAY_HOST", "host", "0.0.0.0")
GATEWAY_PORT = int(_cfg(_gateway_cfg, "GATEWAY_PORT", "port", "8010"))

# 判断是否使用云端 API（URL 包含 dashscope 则为云端）
USE_CLOUD_LLM = "dashscope.aliyuncs.com" in LLM_URL
USE_CLOUD_TTS = "dashscope.aliyuncs.com" in TTS_URL
USE_CLOUD_STT = "dashscope.aliyuncs.com" in STT_URL

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
    tts_model: str = "cosyvoice-v1"
    tts_voice: str = "longyuan"


class TTSProxyRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model: str = Field(default="cosyvoice-v1", description="TTS 模型名")
    voice: str = Field(default="longyuan", description="TTS 音色")


class LLMPredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max_tokens: int = 512
    temperature: float = 0.7


class STTTranscribeRequest(BaseModel):
    pass


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

    if USE_CLOUD_LLM:
        # 云端 LLM：检查 API 连通性
        try:
            if DASHSCOPE_API_KEY:
                headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"}
                resp = requests.post(LLM_URL, headers=headers, json={
                    "model": DASHSCOPE_MODEL,
                    "input": {"messages": [{"role": "user", "content": "hi"}]},
                    "parameters": {"max_tokens": 10, "result_format": "message"},
                }, timeout=10)
                checks["llm"] = {"ok": resp.status_code < 500, "status_code": resp.status_code, "type": "cloud"}
            else:
                checks["llm"] = {"ok": False, "error": "No API key", "type": "cloud"}
        except Exception as e:
            checks["llm"] = {"ok": False, "error": str(e), "type": "cloud"}
    else:
        # 本地 LLM
        try:
            resp = requests.get(f"{_base_url(LLM_URL)}/docs", timeout=3)
            checks["llm"] = {"ok": resp.status_code < 500, "status_code": resp.status_code, "type": "local"}
        except Exception as e:
            checks["llm"] = {"ok": False, "error": str(e), "type": "local"}

    if USE_CLOUD_TTS:
        # 云端 TTS 使用 WebSocket，无法直接 REST 健康检查
        # 仅检查 API Key 是否配置
        if DASHSCOPE_API_KEY:
            checks["tts"] = {
                "ok": True,
                "type": "cloud",
                "note": "API Key 已配置（WebSocket 连接需实际调用验证）",
            }
        else:
            checks["tts"] = {"ok": False, "error": "No API key", "type": "cloud"}
    else:
        try:
            resp = requests.get(f"{_base_url(TTS_URL)}/health", timeout=3)
            checks["tts"] = {"ok": resp.status_code < 500, "status_code": resp.status_code, "type": "local"}
        except Exception as e:
            checks["tts"] = {"ok": False, "error": str(e), "type": "local"}

    if USE_CLOUD_STT:
        checks["stt"] = {"ok": True, "type": "cloud", "note": "STT cloud endpoint available"}
    else:
        try:
            resp = requests.get(f"{_base_url(STT_URL)}/docs", timeout=3)
            checks["stt"] = {"ok": resp.status_code < 500, "status_code": resp.status_code, "type": "local"}
        except Exception as e:
            checks["stt"] = {"ok": False, "error": str(e), "type": "local"}

    return {"ok": True, "services": checks, "config_path": CONFIG_PATH}


def _call_cloud_llm(messages: list, max_tokens: int = 512, temperature: float = 0.7) -> Dict[str, Any]:
    """调用 DashScope 云端 LLM API。"""
    if not DASHSCOPE_API_KEY:
        raise HTTPException(status_code=500, detail="DASHSCOPE_API_KEY 未配置")

    payload = {
        "model": DASHSCOPE_MODEL,
        "input": {"messages": messages},
        "parameters": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "result_format": "message",
        },
    }
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        f"{LLM_URL}",
        headers=headers,
        json=payload,
        timeout=120,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    data = resp.json()
    return {
        "choices": [
            {
                "message": data["output"]["choices"][0]["message"],
                "finish_reason": data["output"]["choices"][0].get("finish_reason", "stop"),
            }
        ]
    }


def _call_llm(messages: list, max_tokens: int = 512, temperature: float = 0.7) -> Dict[str, Any]:
    """根据配置调用本地或云端 LLM。"""
    if USE_CLOUD_LLM:
        return _call_cloud_llm(messages, max_tokens, temperature)
    else:
        llm_payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        llm_res = requests.post(LLM_URL, json=llm_payload, timeout=120)
        llm_data = llm_res.json()
        if llm_res.status_code != 200:
            raise HTTPException(status_code=llm_res.status_code, detail=llm_data)
        return llm_data


@app.post("/chat/text")
def chat_text(req: ChatRequest) -> Dict[str, Any]:
    messages = _build_messages(req)

    try:
        llm_data = _call_llm(messages, req.max_tokens, req.temperature)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"调用 LLM 服务失败：{e}")

    reply_text = _extract_reply_text(llm_data).strip()
    if not reply_text:
        raise HTTPException(status_code=502, detail="LLM 返回为空")

    return {
        "reply_text": reply_text,
        "llm_raw": llm_data,
    }


@app.post("/llm/predict")
def llm_predict(req: LLMPredictRequest) -> Dict[str, Any]:
    """简化版 LLM 接口：直接传文本，返回回复。"""
    messages = [{"role": "user", "content": req.text}]
    try:
        llm_data = _call_llm(messages, req.max_tokens, req.temperature)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"调用 LLM 服务失败：{e}")

    reply_text = _extract_reply_text(llm_data).strip()
    if not reply_text:
        raise HTTPException(status_code=502, detail="LLM 返回为空")

    return {
        "reply_text": reply_text,
        "llm_raw": llm_data,
    }


class STTRecognitionCallback(RecognitionCallback):
    """STT 回调类，用于收集识别结果。"""
    def __init__(self):
        self.result_text = ""

    def on_complete(self):
        pass

    def on_error(self, error):
        pass

    def on_event(self, result):
        sentence = result.get_sentence()
        if sentence:
            self.result_text += sentence.get("text", "")


@app.post("/stt/transcribe")
async def stt_transcribe(file: UploadFile = File(...)) -> Dict[str, Any]:
    """STT 转写接口：接收音频文件，使用 DashScope Recognition SDK 进行识别。"""
    if not DASHSCOPE_API_KEY:
        raise HTTPException(status_code=500, detail="DASHSCOPE_API_KEY 未配置")

    # 将上传的文件保存到临时文件
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        # 设置 DashScope API Key
        dashscope.api_key = DASHSCOPE_API_KEY

        # 创建回调
        callback = STTRecognitionCallback()

        # 根据文件扩展名确定格式
        format_name = suffix.lstrip(".").lower()
        if format_name in ["wav", "pcm", "raw"]:
            audio_format = "wav"
            sample_rate = 16000
        elif format_name in ["mp3", "mpga"]:
            audio_format = "mp3"
            sample_rate = 16000
        elif format_name in ["ogg", "oga"]:
            audio_format = "ogg"
            sample_rate = 16000
        elif format_name in ["flac"]:
            audio_format = "flac"
            sample_rate = 16000
        elif format_name in ["m4a", "aac"]:
            audio_format = "aac"
            sample_rate = 16000
        else:
            audio_format = "wav"
            sample_rate = 16000

        # 创建 Recognition 实例
        recognition = Recognition(
            model=DASHSCOPE_STT_MODEL,
            callback=callback,
            format=audio_format,
            sample_rate=sample_rate
        )

        # 调用识别
        result = recognition.call(file=tmp_path)
        request_id = recognition.get_last_request_id()

        # 获取结果
        text = callback.result_text or result.get_sentence().get("text", "") if result.get_sentence() else ""

        return {
            "text": text,
            "request_id": request_id,
            "model": DASHSCOPE_STT_MODEL
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT 识别失败: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _call_cloud_tts(text: str, model: str, voice: str) -> Dict[str, Any]:
    """调用 DashScope 云端 TTS API。"""
    if not DASHSCOPE_API_KEY:
        raise HTTPException(status_code=500, detail="DASHSCOPE_API_KEY 未配置")

    payload = {
        "model": model,  # Use the provided model parameter
        "input": {"text": text},
        "parameters": {"voice": voice},
    }
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(TTS_URL, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        error_detail = resp.json() if resp.content else {"error": "TTS API call failed"}
        logger.warning(f"TTS API Error: {resp.status_code} - {error_detail}")
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"TTS 服务调用失败: {error_detail.get('message', 'Quota exceeded or API error')}. Code: {error_detail.get('code', 'Unknown')}"
        )
    return resp.json()


def _call_tts(text: str, model: str, voice: str) -> Dict[str, Any]:
    """根据配置调用本地或云端 TTS。"""
    if USE_CLOUD_TTS:
        return _call_cloud_tts(text, model, voice)
    else:
        payload = {"text": text, "model": model, "voice": voice}
        tts_res = requests.post(TTS_URL, json=payload, timeout=120)
        tts_data = tts_res.json()
        if tts_res.status_code != 200:
            raise HTTPException(status_code=tts_res.status_code, detail=tts_data)
        return tts_data


@app.post("/tts/synthesize/proxy")
def tts_proxy(req: TTSProxyRequest) -> Dict[str, Any]:
    try:
        tts_data = _call_tts(req.text, req.model, req.voice)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"调用 TTS 服务失败：{e}")

    return tts_data


@app.post("/chat/tts")
def chat_tts(req: ChatRequest) -> Dict[str, Any]:
    """执行完整链路：LLM 生成回复文本，再转 TTS 音频。"""
    chat_result = chat_text(req)
    reply_text = chat_result["reply_text"]

    try:
        tts_data = _call_tts(reply_text, req.tts_model, req.tts_voice)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"调用 TTS 服务失败：{e}")

    return {
        "reply_text": reply_text,
        "audio_base64": tts_data.get("audio_base64"),
        "audio_mime_type": tts_data.get("audio_mime_type", "audio/mpeg"),
        "tts_request_id": tts_data.get("request_id"),
        "llm_raw": chat_result.get("llm_raw"),
    }


if __name__ == "__main__":
    uvicorn.run(app, host=GATEWAY_HOST, port=GATEWAY_PORT)
