# AI 语音助手项目

本项目提供本地可运行的语音助手链路：

- `STT`：本地 Whisper 语音识别
- `LLM`：本地 Qwen（`llama.cpp`）对话
- `TTS`：阿里云 DashScope `cosyvoice-v3-flash` 语音合成
- `Gateway`：统一编排与对外 API
- `Web`：简单联调页面（健康检查 + 文本对话 + TTS 播放）

---

## 目录结构（当前）

```text
ai-voice-assistant/
├── ai/
│   ├── config.json                 # 公共配置（gateway/tts/llm/stt）
│   ├── start.py                    # 一键启动脚本（支持 --all / --tts 等）
│   ├── gateway/
│   │   ├── main.py                 # 网关服务（8010）
│   │   └── env-gateway.yaml
│   ├── tts/
│   │   ├── cosyvoice_service.py    # 阿里云 CosyVoice TTS（8030）
│   │   └── env-tts.yaml
│   ├── llm/
│   │   ├── qwen_service.py         # 本地 Qwen 服务（8041，对内 llama-server:8040）
│   │   └── env-llm.yaml
│   └── stt/
│       ├── whisper_service.py      # 本地 Whisper STT（8000）
│       └── env-stt.yaml
├── apps/
│   └── web/
│       └── index.html              # 测试页面
└── models/                         # 本地模型与 llama.cpp
```

---

## 架构与调用链路

```text
Web/客户端
   ↓
Gateway (8010)
   ├─ /chat/text -> LLM (8041)
   ├─ /tts/synthesize/proxy -> TTS (8030)
   └─ /chat/tts -> 先 LLM 再 TTS

STT (8000) 可单独调用 /transcribe
```

---

## 公共配置

统一配置文件：`ai/config.json`

配置优先级：

1. 环境变量
2. `ai/config.json`
3. 代码默认值

### 关键配置项

- `gateway`
  - `host` / `port`
  - `llm_url` / `tts_url` / `stt_url`
- `tts`
  - `region`：`cn-beijing` / 新加坡等
  - `model`：默认 `cosyvoice-v3-flash`
  - `voice`：默认音色
  - `dashscope_api_key`：可填，但更推荐环境变量
- `llm`
  - `llama_cpp_bin` / `model_path`
  - `llama_host` / `llama_port`
  - `api_host` / `api_port`
  - 推理参数：`n_gpu_layers`、`ctx_size`、`temperature`、`repeat_penalty`
- `stt`
  - `model_path` / `device` / `api_host` / `api_port`

> 注意：`config.json` 必须是标准 JSON，不要使用 `//` 注释。

---

## 环境准备

建议分别准备 4 个 conda 环境并按 yaml 更新：

```bash
conda env update -n env-gateway -f ai/gateway/env-gateway.yaml --prune
conda env update -n env-tts -f ai/tts/env-tts.yaml --prune
conda env update -n env-llm -f ai/llm/env-llm.yaml --prune
conda env update -n env-stt -f ai/stt/env-stt.yaml --prune
```

TTS 必需配置（推荐环境变量）：

```bash
export DASHSCOPE_API_KEY=你的阿里云DashScopeKey
# 可选
export DASHSCOPE_REGION=cn-beijing
```

---

## 启动方式

### 一键启动全部服务

```bash
python ai/start.py --all
```

### 按需启动

```bash
python ai/start.py --llm --tts --gateway
# 或单独
python ai/start.py --stt
```

---

## 接口说明

### Gateway（`http://127.0.0.1:8010`）

- `GET /health`：查看网关及下游服务健康状态
- `POST /chat/text`：仅文本对话（调用 LLM）
- `POST /tts/synthesize/proxy`：仅 TTS 代理
- `POST /chat/tts`：文本对话 + 语音合成

`/chat/tts` 请求示例：

```json
{
  "messages": [{"role": "user", "content": "你现在知道你是谁嘛？"}],
  "tts_model": "cosyvoice-v3-flash",
  "tts_voice": "longanyang"
}
```

### TTS（`http://127.0.0.1:8030`）

- `GET /health`
- `POST /tts/synthesize`
- `POST /synthesize`（别名）

返回 `audio_base64`，可直接前端播放（`audio/mpeg`）。

### LLM（`http://127.0.0.1:8041`）

- `POST /llm/predict`

### STT（`http://127.0.0.1:8000`）

- `POST /transcribe`（上传音频文件）

---

## 前端联调页面

直接打开：

`apps/web/index.html`

页面已支持：

- 直接测试 TTS
- 聊天 + TTS（走网关）
- 服务状态红绿灯（Gateway/LLM/TTS/STT）

---

## 常见问题

1. `TTS 500` 且提示缺少 key
   - 确认 `DASHSCOPE_API_KEY` 已在当前 shell 导出
   - 访问 `http://127.0.0.1:8030/health` 看 `configured` 是否为 `true`

2. `chat/tts` 报 LLM 连接失败
   - 检查 LLM 是否运行在 `8041`
   - 检查 `ai/config.json` 的 `gateway.llm_url`

3. 改了 `config.json` 不生效
   - TTS 为请求级读取，通常即时生效
   - 其他服务建议重启对应进程

---

## 备注

- 本项目已切换为公共配置模式（`ai/config.json`），不再建议为每个服务维护单独业务配置文件。
- 密钥建议使用环境变量注入，不建议写入仓库。