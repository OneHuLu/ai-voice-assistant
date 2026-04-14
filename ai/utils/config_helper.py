"""公共配置工具模块 - 统一配置读取逻辑。

优先级：环境变量 > config.json > 默认值
"""

import json
import os
from typing import Any, Dict, Optional


def get_config_path() -> str:
    """获取配置文件路径（支持 AI_CONFIG_PATH 环境变量覆盖）。"""
    return os.getenv(
        "AI_CONFIG_PATH",
        os.path.join(os.path.dirname(__file__), "..", "config.json"),
    )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """读取公共配置文件并返回字典。

    Args:
        config_path: 配置文件路径，默认使用 AI_CONFIG_PATH 或 ../config.json

    Returns:
        配置字典，读取失败时返回空字典
    """
    path = config_path or get_config_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_config_value(
    file_cfg: Dict[str, Any],
    env_key: str,
    file_key: str,
    default: str
) -> str:
    """按优先级读取配置项：环境变量 > 配置文件 > 默认值。

    Args:
        file_cfg: 配置文件字典（通常是 config[section]）
        env_key: 环境变量名
        file_key: 配置文件中的 key
        default: 默认值

    Returns:
        配置值（已去除首尾空格）
    """
    return os.getenv(env_key, str(file_cfg.get(file_key, default))).strip()


def resolve_path(path_value: str, project_root: str) -> str:
    """将相对路径转换为项目根目录下的绝对路径。

    Args:
        path_value: 路径值
        project_root: 项目根目录

    Returns:
        绝对路径
    """
    p = (path_value or "").strip()
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(project_root, p))


def get_project_root(relative_to_file: str) -> str:
    """根据文件路径计算项目根目录（向上一级）。

    Args:
        relative_to_file: 当前文件路径（通常是 __file__）

    Returns:
        项目根目录绝对路径
    """
    return os.path.abspath(os.path.join(os.path.dirname(relative_to_file), "..", ".."))


def get_dashscope_api_key(root_cfg: Dict[str, Any]) -> str:
    """获取 DashScope API Key（优先级：环境变量 > 根级别 > 服务级别）。

    Args:
        root_cfg: 根配置字典

    Returns:
        API Key 字符串
    """
    # 优先环境变量
    env_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if env_key:
        return env_key
    # 其次根级别配置
    root_key = str(root_cfg.get("dashscope_api_key", "")).strip()
    if root_key:
        return root_key
    # 最后从各服务配置中查找
    for section in ["tts", "llm", "stt"]:
        sec_cfg = root_cfg.get(section)
        if isinstance(sec_cfg, dict):
            key = str(sec_cfg.get("dashscope_api_key", "")).strip()
            if key:
                return key
    return ""
