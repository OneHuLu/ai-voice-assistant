import argparse
import subprocess
import time
import sys
import os
import json

# ================= 配置 =================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_ENV_PATH = os.path.join(PROJECT_ROOT, ".env.local")

# 默认端口配置（与 config.json 保持一致）
DEFAULT_PORTS = {
    "gateway": 8010,
    "tts": 8030,
    "llm": 8041,
}

# 云端服务（STT 直接调用 DashScope API，无需本地服务）
SERVICES = {
    "gateway": {
        "env": "env-gateway",
        "yaml": "ai/gateway/env-gateway.yaml",
        "script": "ai/gateway/main.py",
        "port": DEFAULT_PORTS["gateway"]
    },
    "tts": {
        "env": "env-tts",
        "yaml": "ai/tts/env-tts.yaml",
        "script": "ai/tts/cosyvoice_service.py",
        "port": DEFAULT_PORTS["tts"]
    },
    "llm": {
        "env": "env-llm",
        "yaml": "ai/llm/env-llm.yaml",
        "script": "ai/llm/qwen_service.py",
        "port": DEFAULT_PORTS["llm"]
    }
}

CONFIG_PATH = os.getenv("AI_CONFIG_PATH", os.path.join("ai", "config.json"))


def _load_file_config():
    """读取公共配置文件并返回字典，读取失败时返回空字典。"""
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _apply_ports_from_config():
    """将 config 中端口映射到启动器服务清单。"""
    config = _load_file_config()
    service_port_map = {
        "gateway": ("gateway", "port", DEFAULT_PORTS["gateway"]),
        "tts": ("tts", None, DEFAULT_PORTS["tts"]),  # TTS 端口由服务内部读取
        "llm": ("llm", "api_port", DEFAULT_PORTS["llm"]),
    }
    for service_name, (section, key, fallback) in service_port_map.items():
        if key:
            sec = config.get(section) if isinstance(config.get(section), dict) else {}
            SERVICES[service_name]["port"] = int(os.getenv(f"{service_name.upper()}_PORT", str(sec.get(key, fallback))))


def _apply_local_env_file() -> None:
    """加载项目根目录 .env.local 到当前进程环境变量。"""
    if not os.path.exists(LOCAL_ENV_PATH):
        return
    with open(LOCAL_ENV_PATH, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ[key] = value


_apply_ports_from_config()
_apply_local_env_file()

processes = []


# ================= 工具函数 =================

def run_cmd(cmd):
    """执行 shell 命令并返回执行结果对象。"""
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def env_exists(env_name):
    """检查 conda 环境是否存在。"""
    result = run_cmd("conda env list")
    return env_name in result.stdout.decode()


def create_env(yaml_path):
    """根据 yaml 文件创建 conda 环境。"""
    print(f"📦 创建环境: {yaml_path}")
    os.system(f"conda env create -f {yaml_path}")


def ensure_env(service):
    """确保服务所需 conda 环境可用，不存在则自动创建。"""
    env_name = service["env"]
    yaml_path = service["yaml"]

    if not env_exists(env_name):
        print(f"⚠️ 环境 {env_name} 不存在，自动创建...")
        create_env(yaml_path)
    else:
        print(f"✅ 环境 {env_name} 已存在")


def start_service(name):
    """按服务名启动子进程并记录到 processes。"""
    config = SERVICES[name]

    ensure_env(config)

    print(f"🚀 启动 {name}（env={config['env']}）...")

    env = os.environ.copy()
    env['PYTHONPATH'] = PROJECT_ROOT
    p = subprocess.Popen(
        [
            "conda", "run", "--no-capture-output", "-n", config["env"],
            "python", "-u", config["script"]
        ],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    processes.append((name, p))


def stop_all():
    """停止当前启动器拉起的全部子服务。"""
    print("\n🛑 正在关闭所有服务...")
    for name, p in processes:
        print(f"关闭 {name}...")
        p.terminate()


# ================= 主逻辑 =================

def main():
    """解析启动参数并按顺序启动指定服务。"""
    parser = argparse.ArgumentParser(description="AI Voice Assistant 启动器（云端模式）")

    parser.add_argument("--all", action="store_true", help="启动全部服务")
    parser.add_argument("--gateway", action="store_true", help="启动网关服务")
    parser.add_argument("--tts", action="store_true", help="启动 TTS 服务")
    parser.add_argument("--llm", action="store_true", help="启动 LLM 服务")

    args = parser.parse_args()

    selected = []

    if args.all:
        # 云端模式启动顺序：LLM -> TTS -> Gateway
        selected = ["llm", "tts", "gateway"]
    else:
        for svc in SERVICES:
            if getattr(args, svc, False):
                selected.append(svc)

    if not selected:
        print("❌ 请指定服务，例如 --all 或 --tts --gateway")
        parser.print_help()
        return

    try:
        for svc in selected:
            start_service(svc)
            time.sleep(2)

        print(f"\n✅ 已启动服务: {', '.join(selected)}，Ctrl+C 退出")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        stop_all()


if __name__ == "__main__":
    main()