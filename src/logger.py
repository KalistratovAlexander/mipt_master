"""Настройка логирования для проекта.

Example:
    >>> from mipt_master.src.logger import setup_logger
    >>> logger = setup_logger("train-rqvae", log_to_file=True)
    >>> logger.info("Training started")
    # 16:30:45 | INFO | Training started
"""

from __future__ import annotations

import logging
import sys
import time
from multiprocessing import current_process
from pathlib import Path

# Форматы логов
_CONSOLE_FMT = "%(asctime)s | %(levelname)-5s | %(message)s"
_FILE_FMT = "%(asctime)s | %(name)s | %(levelname)-5s | %(message)s"
_CONSOLE_DATEFMT = "%H:%M:%S"
_FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "semantic-id",
    level: int = logging.INFO,
    log_to_file: bool = False,
    log_dir: str | Path = "logs",
) -> logging.Logger:
    """Создать логгер с выводом в консоль и опционально в файл.

    Args:
        name: Имя логгера.
        level: Уровень логирования (default: INFO).
        log_to_file: Записывать в файл (default: False).
        log_dir: Директория для логов (default: "logs").

    Returns:
        Настроенный логгер.
    """
    logger = logging.getLogger(name)

    # Если уже настроен — возвращаем как есть
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt=_CONSOLE_DATEFMT))
    logger.addHandler(console)

    # File (только в главном процессе)
    is_main = current_process().name == "MainProcess"
    if log_to_file and is_main:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        file_path = log_dir / f"log-{time.strftime('%Y%m%d_%H%M%S')}.txt"
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_FILE_DATEFMT))
        logger.addHandler(file_handler)

        logger.info(f"Log file: {file_path}")

    return logger
