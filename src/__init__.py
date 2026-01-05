"""Пакет mipt_master.src — Semantic ID для рекомендательных систем.

Этот пакет содержит модули для:

1. **RQ-VAE** (`rqvae/`): Residual Quantized VAE для генерации
   иерархических semantic IDs из эмбеддингов товаров.

2. **Embed** (`embed/`): Генерация эмбеддингов товаров с помощью
   различных моделей (Qwen3, BGE, Jina и др.).

3. **Vocab Expansion** (`vocab_expansion/`): Stage 1 — расширение
   словаря LLM токенами semantic ID.

4. **LoRA Fine-tuning** (`finetune_lora/`): Stage 2 — LoRA
   дообучение для понимания semantic IDs.

5. **Full Fine-tuning** (`finetune_full/`): Stage 2 — полное
   дообучение модели.

6. **Inference** (`inference/`): Инференс и evaluation моделей.

Общие модули:
- `device`: Управление устройством (CUDA/MPS/CPU)
- `logger`: Настройка логирования
- `test_prompts`: Тестовые промпты

Example:
    >>> from mipt_master.src.device import get_device_info
    >>> from mipt_master.src.logger import setup_logger
    >>>
    >>> logger = setup_logger("my-script")
    >>> device_info = get_device_info()
    >>> logger.info(f"Using device: {device_info.device}")
"""

from .device import DeviceInfo, get_device_info, setup_device
from .logger import setup_logger
from .test_prompts import (
    REC_TEST_PROMPTS,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_SID_ONLY,
    TEST_PROMPTS,
    get_all_test_prompts,
    get_rec_test_prompts,
)

# Alias для обратной совместимости
DeviceManager = DeviceInfo

__all__ = [
    # Device
    "DeviceInfo",
    "DeviceManager",  # alias
    "get_device_info",
    "setup_device",
    # Logger
    "setup_logger",
    # Test prompts
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_SID_ONLY",
    "REC_TEST_PROMPTS",
    "TEST_PROMPTS",
    "get_rec_test_prompts",
    "get_all_test_prompts",
]

