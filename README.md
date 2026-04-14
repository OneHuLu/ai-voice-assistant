# AI 语音助手项目

基于阿里云 DashScope API 的语音助手服务：

- **STT**：阿里云 `paraformer-v2` 语音识别
- **LLM**：阿里云 `qwen-max-latest` 大模型对话
- **TTS**：阿里云 `cosyvoice-v1` 语音合成
- **Gateway**：统一编排与对外 API
- **Web**：联调测试页面

---

## 目录结构

```text
ai-voice-assistant/
├── ai/
│   ├── config.json                 # 公共配置
│   ├── start.py                    # 一键启动脚本
│   ├── utils/
│   │   └── config_helper.py        # 配置工具模块
│   ├── gateway/
│   │   ├── main.py                 # 网关服务（8010）
│   │   └── env-gateway.yaml
│   ├── tts/
│   │   ├── cosyvoice_service.py    # TTS 服务（8030）
│   │   └── env-tts.yaml
│   ├── llm/
│   │   ├── qwen_service.py         # LLM 服务（8041）
│   │   └── env-llm.yaml
│   └── stt/
│       └── whisper_service.py      # STT 服务（保留，可扩展）
│       └── env-stt.yaml
├── apps/
│   └── web/
│       └── index.html              # 测试页面
└── requirements.txt                 # 公共依赖
```

---

## 架构与调用链路

```text
Web/客户端
   ↓
Gateway (8010)
   ├─ /chat/text -> 云端 LLM (DashScope)
   ├─ /tts/synthesize/proxy -> 本地 TTS 服务 -> 云端 CosyVoice
   ├─ /stt/transcribe -> 云端 STT (DashScope)
   └─ /chat/tts -> 先 LLM 再 TTS
```

---

## 配置说明

配置文件：`ai/config.json`

优先级：**环境变量 > config.json > 代码默认值**

### 必需配置

```bash
export DASHSCOPE_API_KEY=你的阿里云DashScopeKey
```

### 配置项说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `dashscope_api_key` | API Key（推荐用环境变量） | 空 |
| `gateway.host/port` | 网关地址 | `0.0.0.0:8010` |
| `tts.region` | 区域 | `cn-beijing` |
| `tts.model` | TTS 模型 | `cosyvoice-v1` |
| `tts.voice` | 音色 | `longyuan` |
| `llm.dashscope_model` | LLM 模型 | `qwen-max-latest` |
| `stt.dashscope_model` | STT 模型 | `paraformer-v2` |

---

## 快速开始

### 1. 创建环境

```bash
conda env update -n env-gateway -f ai/gateway/env-gateway.yaml --prune
conda env update -n env-tts -f ai/tts/env-tts.yaml --prune
conda env update -n env-llm -f ai/llm/env-llm.yaml --prune
```

### 2. 配置 API Key

```bash
export DASHSCOPE_API_KEY=你的Key
```

### 3. 启动服务

```bash
python ai/start.py --all
```

或按需启动：

```bash
python ai/start.py --llm --tts --gateway
```

---

## 接口说明

### Gateway（`http://127.0.0.1:8010`）

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 服务健康状态 |
| `/chat/text` | POST | 文本对话 |
| `/chat/tts` | POST | 对话 + 语音合成 |
| `/tts/synthesize/proxy` | POST | TTS 代理 |
| `/stt/transcribe` | POST | 语音转文本（上传音频） |
| `/llm/predict` | POST | 简化版 LLM |

### `/chat/tts` 示例

```json
{
  "messages": [{"role": "user", "content": "你好"}],
  "tts_model": "cosyvoice-v1",
  "tts_voice": "longyuan"
}
```

### TTS 服务（`http://127.0.0.1:8030`）

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 服务状态 |
| `/tts/synthesize` | POST | 文本合成语音 |

返回 `audio_base64`，可直接前端播放。

---

## 前端测试页

打开 `apps/web/index.html`，支持：

- TTS 测试
- LLM 对话
- STT 转写
- 服务状态监控

---

## 常见问题

1. **TTS 报错缺少 key**
   - 检查 `DASHSCOPE_API_KEY` 环境变量是否设置
   - 访问 `http://127.0.0.1:8030/health` 验证

2. **改了配置不生效**
   - TTS 每次请求动态读取配置，即时生效
   - 其他服务需重启

3. **服务启动失败**
   - 确认 conda 环境已创建
   - 检查依赖是否安装：`pip install -r requirements.txt`