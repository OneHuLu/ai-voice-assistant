import argparse
import subprocess
import time
import sys
import os
import json

# ================= 配置 =================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_ENV_PATH = os.path.join(PROJECT_ROOT, ".env.local")

SERVICES = {
    "gateway": {
        "env": "env-gateway",
        "yaml": "ai/gateway/env-gateway.yaml",
        "script": "ai/gateway/main.py",
        "port": 8010
    },
    "stt": {
        "env": "env-stt",
        "yaml": "ai/stt/env-stt.yaml",
        "script": "ai/stt/whisper_service.py",
        "port": 8000
    },
    "tts": {
        "env": "env-tts",
        "yaml": "ai/tts/env-tts.yaml",
        "script": "ai/tts/cosyvoice_service.py",
        "port": 8030
    },
    "llm": {
        "env": "env-llm",
        "yaml": "ai/llm/env-llm.yaml",
        "script": "ai/llm/qwen_service.py",
        "port": 8041
    }
}

# 公共配置文件路径（可通过 AI_CONFIG_PATH 覆盖）
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
    # 启动器端口也走统一配置，确保 start.py 与各服务读取结果一致
    config = _load_file_config()
    service_port_map = {
        "gateway": ("gateway", "port", 8010),
        "stt": ("stt", "api_port", 8000),
        "tts": ("tts", "api_port", 8030),
        "llm": ("llm", "api_port", 8041),
    }
    for service_name, (section, key, fallback) in service_port_map.items():
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
    print('config ==>', config)

    ensure_env(config)

    print(f"🚀 启动 {name}（env={config['env']}）...")

    p = subprocess.Popen(
        [
            "conda", "run", "--no-capture-output", "-n", config["env"],
            "python", "-u", config["script"]
        ],
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
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--all", action="store_true")
    

    for svc in SERVICES:
        parser.add_argument(f"--{svc}", action="store_true")

    args = parser.parse_args()

    selected = []

    print("args ==>", args)
    if args.all:
        # 推荐启动顺序
        selected = ["stt", "llm", "tts", "gateway"]
    else:
        for svc in SERVICES:
            if getattr(args, svc):
                selected.append(svc)

    if not selected:
        print("❌ 请指定服务，例如 --all 或 --tts")
        return

    print('selected ==>', selected)
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