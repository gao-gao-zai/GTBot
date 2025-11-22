import sys
import logging

# 定义一个兼容层，用于在没有 nonebot 环境时提供基本的日志功能
class FallbackLogger:
    def __init__(self):
        self._logger = logging.getLogger("GTBot")
        self._logger.setLevel(logging.INFO)
        
        # 避免重复添加 handler
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s - %(message)s', 
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self._logger.exception(msg, *args, **kwargs)

    def success(self, msg, *args, **kwargs):
        # logging 模块没有 success 级别，使用 info 级别并添加前缀
        self._logger.info(f"[SUCCESS] {msg}", *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        if isinstance(level, str):
            # 尝试获取对应的 logging level
            level_upper = level.upper()
            if hasattr(logging, level_upper):
                level = getattr(logging, level_upper)
            else:
                level = logging.INFO
        self._logger.log(level, msg, *args, **kwargs)

try:
    from nonebot import logger
except ImportError:
    logger = FallbackLogger()

__all__ = ["logger"]
