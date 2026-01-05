"""Пакет для расширения словаря и инициализации эмбеддингов.

Stage 1 обучения: добавляем semantic ID токены и обучаем их эмбеддинги.

Example:
    CLI:
    ```bash
    python -m mipt_master.src.vocab_expansion.cli \\
        --model-name unsloth/Qwen3-1.7B \\
        --category Amazon_Fashion \\
        --max-steps 1000 \\
        --lr 1e-3
    ```

    Python:
    ```python
    from mipt_master.src.vocab_expansion import (
        VocabExpansionConfig,
        extend_tokenizer,
        prepare_model_for_embedding_training,
        train_embeddings,
    )

    config = VocabExpansionConfig(category="Amazon_Fashion")
    # ... load model and tokenizer
    num_new = extend_tokenizer(model, tokenizer, config)
    model = prepare_model_for_embedding_training(model, tokenizer, config, num_new)
    result = train_embeddings(model, tokenizer, config, num_new, test_prompts=[])
    ```
"""

from .callbacks import (
    DataInspectionCallback,
    EmbeddingMonitorCallback,
    SemanticIDGenerationCallback,
    create_callbacks,
)
from .config import VocabExpansionConfig
from .dataset import (
    DataCollatorLM,
    load_conversation_dataset,
    prepare_datasets,
    tokenize_dataset,
)
from .tokenizer import (
    extend_tokenizer,
    prepare_model_for_embedding_training,
)
from .train import (
    TrainResult,
    save_embeddings_artifact,
    save_model_and_tokenizer,
    train_embeddings,
)

__all__ = [
    # Config
    "VocabExpansionConfig",
    # Tokenizer
    "extend_tokenizer",
    "prepare_model_for_embedding_training",
    # Dataset
    "load_conversation_dataset",
    "tokenize_dataset",
    "prepare_datasets",
    "DataCollatorLM",
    # Callbacks
    "DataInspectionCallback",
    "EmbeddingMonitorCallback",
    "SemanticIDGenerationCallback",
    "create_callbacks",
    # Train
    "TrainResult",
    "train_embeddings",
    "save_model_and_tokenizer",
    "save_embeddings_artifact",
]

