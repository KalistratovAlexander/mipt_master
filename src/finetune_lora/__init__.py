"""Пакет для LoRA fine-tuning модели с semantic IDs.

Stage 2 обучения: дообучаем LoRA адаптеры на модели с расширенным словарём.

Example:
    CLI:
    ```bash
    python -m mipt_master.src.finetune_lora.cli \\
        --stage1-checkpoint models/qwen3_fashion_vocab/final \\
        --category Amazon_Fashion \\
        --lora-r 16 \\
        --lr 2e-4 \\
        --epochs 2
    ```

    Python:
    ```python
    from mipt_master.src.finetune_lora import (
        LoRAConfig,
        load_model_with_lora,
        train_lora,
    )

    config = LoRAConfig(
        stage1_checkpoint="models/qwen3_fashion_vocab/final",
        lora_r=16,
    )
    config.validate()
    result = train_lora(config)
    ```
"""

from .callbacks import GenerationEvalCallback, create_callbacks
from .config import DEFAULT_LORA_TARGET_MODULES, LoRAConfig
from .model import (
    SEMANTIC_ID_TOKENS,
    get_trainable_params_info,
    load_model_with_lora,
    verify_semantic_tokens,
)
from .train import (
    TrainResult,
    finish_wandb,
    load_conversation_dataset,
    save_lora_adapter,
    save_merged_model,
    train_lora,
)

__all__ = [
    # Config
    "LoRAConfig",
    "DEFAULT_LORA_TARGET_MODULES",
    # Model
    "SEMANTIC_ID_TOKENS",
    "load_model_with_lora",
    "verify_semantic_tokens",
    "get_trainable_params_info",
    # Callbacks
    "GenerationEvalCallback",
    "create_callbacks",
    # Train
    "TrainResult",
    "train_lora",
    "load_conversation_dataset",
    "save_lora_adapter",
    "save_merged_model",
    "finish_wandb",
]

