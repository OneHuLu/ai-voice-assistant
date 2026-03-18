# AI 语音助手项目

一个基于本地部署的 AI 语音助手，支持语音输入、语音识别(STT)和 AI 对话(LLM)。

---

## 项目结构

```
ai-voice-assistant/
├── index.html              # 前端 Demo 页面
├── ai/                     # AI 服务端代码
│   ├── requirements.txt    # Python 依赖清单
│   ├── gateway/            # API 网关
│   │   └── main.py         # 网关服务入口
│   ├── llm/                # 大语言模型服务
│   │   └── qwen_service.py # Qwen LLM 服务（基于 llama.cpp）
│   ├── stt/                # 语音识别服务
│   │   ├── whisper_service.py  # Whisper STT 服务
│   │   └── test/           # 测试文件
│   │       ├── stt_test.py     # STT 测试脚本
│   │       ├── base.pt         # 测试模型
│   │       └── clone.wav       # 测试音频
│   ├── tts/                # 语音合成服务（预留）
│   │   └── tts_service.py      # 空文件
│   └── venv/               # Python 虚拟环境
├── apps/                   # 应用层
│   └── web/                # Web 应用（预留）
├── deploy/                 # 部署脚本（预留）
└── models/                 # AI 模型文件
    ├── faster-base/        # Whisper 语音模型
    │   ├── config.json
    │   ├── model.bin
    │   ├── tokenizer.json
    │   ├── vocabulary.txt
    │   └── README.md
    └── llama.cpp          # llama.cpp 编译产物
```

---

## 目录/文件详解

### 根目录

| 文件/目录 | 说明 |
|-----------|------|
| `index.html` | 前端 Demo 页面，提供语音/文字输入界面，与后端 API 交互 |

### `ai/` - AI 服务端

AI 服务的核心代码目录，包含各个 AI 能力模块。

| 文件/目录 | 说明 |
|-----------|------|
| `requirements.txt` | Python 依赖清单，包含 fastapi、faster-whisper、requests 等 |
| `venv/` | Python 虚拟环境，包含所有安装的依赖包 |

#### `ai/gateway/` - API 网关

| 文件 | 说明 |
|------|------|
| `main.py` | 网关服务入口，用于统一管理和路由各 AI 服务 |

#### `ai/llm/` - 大语言模型服务

| 文件 | 说明 |
|------|------|
| `qwen_service.py` | Qwen LLM 服务，基于 llama.cpp 提供本地 AI 对话能力，端口 8001 |

#### `ai/stt/` - 语音识别服务

| 文件/目录 | 说明 |
|-----------|------|
| `whisper_service.py` | Whisper 语音识别服务，基于 faster-whisper，端口 8000 |
| `test/` | 测试相关文件 |
| `test/stt_test.py` | STT 功能测试脚本 |
| `test/base.pt` | 测试用模型文件 |
| `test/clone.wav` | 测试用音频文件 |

#### `ai/tts/` - 语音合成服务

| 文件 | 说明 |
|------|------|
| `tts_service.py` | 预留文件，暂未实现语音合成功能 |

### `apps/` - 应用层

| 目录 | 说明 |
|------|------|
| `web/` | Web 应用目录，预留用于存放完整前端应用 |

### `deploy/` - 部署脚本

部署相关脚本目录，目前为空，可用于存放 Docker、CI/CD 等部署配置。

### `models/` - AI 模型文件

存放本地运行的 AI 模型。

| 目录 | 说明 |
|------|------|
| `faster-base/` | Whisper faster-base 模型文件（约 138MB），用于语音识别 |
| `llama.cpp/` | llama.cpp 编译产物，用于运行 Qwen 大语言模型 |

---

## 服务端口说明

| 服务 | 端口 | 说明 |
|------|------|------|
| STT 服务 | 8000 | 语音识别，接口 `/transcribe` |
| LLM 服务 | 8001 | AI 对话，接口 `/llm/predict` |

---

## 快速启动

### 1. 安装依赖

```bash
cd ai
pip install -r requirements.txt
```

### 2. 启动 STT 服务

```bash
python stt/whisper_service.py
```

### 3. 启动 LLM 服务

```bash
python llm/qwen_service.py
```

### 4. 打开前端

直接用浏览器打开 `index.html` 即可使用。

---

## 依赖清单（requirements.txt）

| 库 | 作用 |
|----|------|
| fastapi | 高性能异步 Web 框架 |
| uvicorn | ASGI 服务器 |
| python-multipart | 处理文件上传 |
| pydantic | 数据验证 |
| faster-whisper | 语音识别引擎 |
| requests | HTTP 请求库 |

---

## 工作流程

```
用户语音输入  →  STT服务(8000)  →  文字
                                    ↓
用户文字输入  ←  LLM服务(8001)  ←  AI回复
```

1. 用户在 `index.html` 上传音频或输入文字
2. 如有音频，调用 STT 服务（端口 8000）转为文字
3. 将文字发送到 LLM 服务（端口 8001）获取 AI 回复
4. 显示对话结果



            ┌───────────────┐
            │   前端 Web/小程序  │
            └───────┬───────┘
                    │ HTTP 请求/音频/文本
                    ▼
            ┌───────────────┐
            │   Gateway 服务  │
            │  (env_gateway) │
            │ FastAPI 路由   │
            └───────┬───────┘
    ┌──────────────┼───────────────┐
    │              │               │
    ▼              ▼               ▼
┌───────────┐  ┌───────────┐   ┌────────────┐
│  LLM 服务 │  │  STT 服务 │   │  TTS 服务  │
│ env_llm   │  │ env_stt   │   │ env_tts    │
│ Qwen      │  │ Whisper   │   │ CosyVoice │
│ FastAPI   │  │ FastAPI   │   │ FastAPI   │
└─────┬─────┘  └─────┬─────┘   └─────┬─────┘
      │              │               │
      │ JSON         │ JSON/音频文件  │ JSON/音频文件
      │              │               │
      ▼              ▼               ▼
   响应结果        转文字结果        音频文件/链接