"""Small colored logger shared by CoEvoX command-line modules."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Iterator


_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;31m",
}
_WHITE = "\033[37m"
_BRIGHT_BLUE = "\033[94m"
_RESET = "\033[0m"


class _ThreadLogFilter(logging.Filter):
    """Keep a file handler scoped to the thread that created it."""

    def __init__(self, thread_id: int) -> None:
        super().__init__()
        self.thread_id = thread_id

    def filter(self, record: logging.LogRecord) -> bool:
        return threading.get_ident() == self.thread_id


class _UtcColorFormatter(logging.Formatter):
    converter = time.gmtime

    def format(self, record: logging.LogRecord) -> str:
        force_color = os.getenv("COEVOX_COLOR") == "1"
        color_enabled = force_color or (
            sys.stderr.isatty() and "NO_COLOR" not in os.environ
        )
        if not color_enabled:
            return super().format(record)

        timestamp = self.formatTime(record, self.datefmt)
        level = f"{record.levelname:<8}"
        message = record.getMessage()
        if record.exc_info:
            exception = self.formatException(record.exc_info)
            message = f"{message}\n{exception}" if message else exception
        if record.stack_info:
            stack = self.formatStack(record.stack_info)
            message = f"{message}\n{stack}" if message else stack

        level_color = _LEVEL_COLORS.get(record.levelno, _WHITE)
        return (
            f"{_WHITE}{timestamp}Z{_RESET} "
            f"{level_color}{level}{_RESET} "
            f"{_BRIGHT_BLUE}[{record.name}]{_RESET} "
            f"{_WHITE}{message}{_RESET}"
        )


def get_logger(module: str) -> logging.Logger:
    """Return a stderr logger formatted as time, level, module, message."""
    logger = logging.getLogger(module)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(
        _UtcColorFormatter(
            "%(asctime)sZ %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if os.getenv("COEVOX_DEBUG") else logging.INFO)
    logger.propagate = False
    return logger


@contextmanager
def thread_log_file(path: Path) -> Iterator[None]:
    """Route root-logger records from the current worker to one plain log file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.addFilter(_ThreadLogFilter(threading.get_ident()))
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)sZ %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()
