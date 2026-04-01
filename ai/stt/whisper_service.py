import json
import os
import shutil
import tempfile
from typing import Dict

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel


CONFIG_PATH = os.getenv(
    "AI_CONFIG_PATH",
    os.path.join(os.path.dirname(__file__), "..", "config.json"),
)


def _load_file_config() -> Dict:
    """读取公共配置文件并返回字典。"""
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _cfg(file_cfg: Dict, env_key: str, file_key: str, default: str) -> str:
    """按“环境变量 > 配置文件 > 默认值”读取配置项。"""
    # 优先级：环境变量 > 公共配置文件 > 默认值
    return os.getenv(env_key, str(file_cfg.get(file_key, default))).strip()


_stt_cfg = _load_file_config().get("stt")
_stt_cfg = _stt_cfg if isinstance(_stt_cfg, dict) else {}

MODEL_PATH = _cfg(
    _stt_cfg,
    "STT_MODEL_PATH",
    "model_path",
    "/models/faster-base",
)
DEVICE = _cfg(_stt_cfg, "STT_DEVICE", "device", "cpu")
API_HOST = _cfg(_stt_cfg, "STT_API_HOST", "api_host", "0.0.0.0")
API_PORT = int(_cfg(_stt_cfg, "STT_API_PORT", "api_port", "8000"))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型路径不存在: {MODEL_PATH}")

print("正在加载 Whisper 模型，请稍等...")
model = WhisperModel(MODEL_PATH, device=DEVICE)
print("Whisper 模型加载成功！")

app = FastAPI(title="Local Whisper STT Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def transcribe_audio(file_path: str, language: str = None) -> Dict:
    """调用 Whisper 模型转写本地音频文件。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"音频文件不存在: {file_path}")

    segments, info = model.transcribe(file_path, language=language)
    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": [{"start": seg.start, "end": seg.end, "text": seg.text} for seg in segments],
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = None):
    """接收上传音频并返回分段转写结果。"""
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac")):
        raise HTTPException(status_code=400, detail="不支持的音频格式")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = transcribe_audio(tmp_path, language)
    finally:
        os.remove(tmp_path)

    return result


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
