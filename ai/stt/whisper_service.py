# whisper_service.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from typing import List, Dict
import uvicorn
import shutil
import tempfile

# ----------------------------
# 模型初始化（全局只加载一次）
# ----------------------------
MODEL_PATH = "/Users/dyx/Desktop/project/ai-voice-assistant/models/faster-base"
DEVICE = "cpu"  # M4 Pro CPU

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型路径不存在: {MODEL_PATH}")

print("正在加载 Whisper 模型，请稍等...")
model = WhisperModel(MODEL_PATH, device=DEVICE)
print("Whisper 模型加载成功！")

# ----------------------------
# FastAPI 初始化
# ----------------------------
app = FastAPI(title="Local Whisper STT Service")

# 跨域，方便前端调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# 转写函数
# ----------------------------
def transcribe_audio(file_path: str, language: str = None) -> Dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"音频文件不存在: {file_path}")

    segments, info = model.transcribe(file_path, language=language)
    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": [
            {"start": seg.start, "end": seg.end, "text": seg.text}
            for seg in segments
        ],
    }

# ----------------------------
# API 接口
# ----------------------------
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = None):
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac")):
        raise HTTPException(status_code=400, detail="不支持的音频格式")
    
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        result = transcribe_audio(tmp_path, language)
    finally:
        os.remove(tmp_path)
    
    return result

# ----------------------------
# 本地启动
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)