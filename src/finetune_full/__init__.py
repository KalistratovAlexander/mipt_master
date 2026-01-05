"""Пакет для full fine-tuning модели с semantic IDs.

Stage 2 обучения: дообучаем всю модель (не LoRA) на диалогах.

Example:
    CLI:
    ```bash
    python -m mipt_master.src.finetune_full.cli \\
        --stage1-checkpoint models/qwen3_fashion_vocab/final \\
        --category Amazon_Fashion \\
        --lr 1e-5 \\
        --epochs 1
    ```

    Python:
    ```python
    from mipt_master.src.finetune_full import (
        FullFineTuneConfig,
        finetune_model,
    )

    config = FullFineTuneConfig(
        stage1_checkpoint="models/qwen3_fashion_vocab/final",
        learning_rate=1e-5,
    )
    config.validate()
    result = finetune_model(config)
    ```
"""

from .callbacks import TrainingMonitorCallback, create_callbacks
from .config import FullFineTuneConfig
from .model import (
    SEMANTIC_ID_TOKENS,
    ensure_embedding_alignment,
    get_trainable_params_info,
    load_model_for_full_finetune,
    test_sid_roundtrip,
    verify_semantic_tokens,
)
from .train import (
    TrainResult,
    finetune_model,
    finish_wandb,
    get_latest_checkpoint,
    load_conversation_dataset,
    save_final_model,
)

__all__ = [
    # Config
    "FullFineTuneConfig",
    # Model
    "SEMANTIC_ID_TOKENS",
    "load_model_for_full_finetune",
    "verify_semantic_tokens",
    "test_sid_roundtrip",
    "ensure_embedding_alignment",
    "get_trainable_params_info",
    # Callbacks
    "TrainingMonitorCallback",
    "create_callbacks",
    # Train
    "TrainResult",
    "finetune_model",
    "load_conversation_dataset",
    "save_final_model",
    "get_latest_checkpoint",
    "finish_wandb",
]

