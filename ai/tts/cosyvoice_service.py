import base64
import json
import os
from typing import Any, Dict, Optional

import dashscope
import uvicorn
from dashscope.audio.tts_v2 import SpeechSynthesizer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


def _resolve_ws_url(region: str) -> str:
    normalized = (region or "").strip().lower()
    if normalized in {"sg", "intl", "international", "ap-southeast-1", "singapore"}:
        return "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference"
    return "wss://dashscope.aliyuncs.com/api-ws/v1/inference"


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


def _load_runtime_config():
    # 每次请求动态读取，避免改配置后必须重启服务
    root_cfg = _load_file_config()
    tts_cfg = root_cfg.get("tts") if isinstance(root_cfg.get("tts"), dict) else {}

    api_key = _cfg(tts_cfg, "DASHSCOPE_API_KEY", "dashscope_api_key", "")
    region = _cfg(tts_cfg, "DASHSCOPE_REGION", "region", "cn-beijing")
    default_model = _cfg(tts_cfg, "COSYVOICE_MODEL", "model", "cosyvoice-v3-flash")
    default_voice = _cfg(tts_cfg, "COSYVOICE_VOICE", "voice", "longanyang")
    ws_url = _resolve_ws_url(region)

    return api_key, region, ws_url, default_model, default_voice


app = FastAPI(title="Aliyun CosyVoice TTS Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="待合成文本")
    model: Optional[str] = Field(default=None, description="模型名，默认从公共配置读取")
    voice: Optional[str] = Field(default=None, description="音色，默认从公共配置读取")


@app.get("/health")
def health_check():
    api_key, region, ws_url, default_model, default_voice = _load_runtime_config()
    return {
        "ok": True,
        "provider": "aliyun-dashscope",
        "model": default_model,
        "voice": default_voice,
        "configured": bool(api_key),
        "region": region,
        "ws_url": ws_url,
        "config_path": CONFIG_PATH,
    }


@app.post("/tts/synthesize")
@app.post("/synthesize")
def synthesize(req: TTSRequest):
    api_key, _, ws_url, default_model, default_voice = _load_runtime_config()
    if not api_key:
        raise HTTPException(status_code=500, detail="缺少 DASHSCOPE_API_KEY（环境变量或公共配置文件）")

    model = (req.model or "").strip() or default_model
    voice = (req.voice or "").strip() or default_voice

    dashscope.api_key = api_key
    dashscope.base_websocket_api_url = ws_url

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")

    try:
        synthesizer = SpeechSynthesizer(model=model, voice=voice)
        audio_bytes = synthesizer.call(text)
        if not audio_bytes:
            raise HTTPException(status_code=502, detail="TTS 合成失败：未返回音频")

        return {
            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "audio_mime_type": "audio/mpeg",
            "model": model,
            "voice": voice,
            "request_id": synthesizer.get_last_request_id(),
            "first_package_delay_ms": synthesizer.get_first_package_delay(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"调用阿里云 CosyVoice 失败: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8030)
