import sys
import os
import uvicorn
import argparse
import multiprocessing
import socket
import logging
import signal
import time
from typing import List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 加入项目根目录
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT)

# 全局进程列表，用于信号处理
processes: List[multiprocessing.Process] = []


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except socket.error:
            return True


def signal_handler(signum, frame):
    """处理信号，优雅退出所有子进程"""
    sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
    logger.info(f"接收到 {sig_name} 信号，正在停止所有服务...")
    
    for p in processes:
        if p.is_alive():
            logger.info(f"停止进程: {p.name} (PID: {p.pid})")
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                logger.warning(f"进程 {p.name} 未响应，强制结束")
                p.kill()
    
    logger.info("所有服务已停止")
    sys.exit(0)


def start_stt():
    """启动 STT 服务"""
    try:
        from ai.stt.whisper_service import app as stt_app
        logger.info("STT 服务启动中...")
        uvicorn.run(stt_app, host="127.0.0.1", port=8000, reload=False)
    except Exception as e:
        logger.error(f"STT 服务异常: {e}")
        raise


def start_llm():
    """启动 LLM 服务"""
    try:
        from ai.llm.qwen_service import app as llm_app
        logger.info("LLM 服务启动中...")
        uvicorn.run(llm_app, host="127.0.0.1", port=8001, reload=False)
    except Exception as e:
        logger.error(f"LLM 服务异常: {e}")
        raise


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 参数解析
    parser = argparse.ArgumentParser(description="AI 语音助手网关服务")
    parser.add_argument("--stt", action="store_true", help="启动语音识别服务")
    parser.add_argument("--llm", action="store_true", help="启动大语言模型服务")
    parser.add_argument("--all", action="store_true", help="启动所有服务")
    args = parser.parse_args()
    
    # 验证参数
    if not (args.all or args.stt or args.llm):
        parser.print_help()
        logger.error("请指定要启动的服务: --stt, --llm 或 --all")
        sys.exit(1)
    
    # 检查端口占用
    services_to_start = []
    if args.all or args.stt:
        if is_port_in_use(8000):
            logger.error("端口 8000 已被占用，无法启动 STT 服务")
            sys.exit(1)
        services_to_start.append(("STT", start_stt, 8000))
    
    if args.all or args.llm:
        if is_port_in_use(8001):
            logger.error("端口 8001 已被占用，无法启动 LLM 服务")
            sys.exit(1)
        services_to_start.append(("LLM", start_llm, 8001))
    
    # 启动服务
    for name, target, port in services_to_start:
        p = multiprocessing.Process(target=target, name=name)
        p.start()
        processes.append(p)
        logger.info(f"{name} 服务已启动 (PID: {p.pid}, 端口: {port})")
    
    # 监控进程状态
    logger.info("按 Ctrl+C 停止所有服务")
    try:
        while True:
            time.sleep(1)
            # 检查是否有进程异常退出
            for p in processes:
                if not p.is_alive() and p.exitcode is not None:
                    if p.exitcode != 0:
                        logger.error(f"{p.name} 服务异常退出，退出码: {p.exitcode}")
                    else:
                        logger.info(f"{p.name} 服务已正常退出")
                    processes.remove(p)
            
            # 如果所有进程都结束了，主进程也退出
            if not processes:
                logger.info("所有服务已结束")
                break
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
