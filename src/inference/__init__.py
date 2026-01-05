"""Пакет для inference и evaluation моделей semantic ID.

Этот пакет предоставляет:

1. **Загрузка моделей**: Поддержка PEFT и unsloth для загрузки
   базовых моделей и LoRA адаптеров.

2. **Генерация**: Функции для генерации semantic IDs по описанию
   продуктов с парсингом результатов.

3. **Evaluation**: Метрики точности по уровням SID, сравнение
   с gold items, сохранение отчётов.

4. **Тест-кейсы**: Готовые наборы тестов для оценки качества:
   - Gold items с эталонными SID
   - SID format тесты
   - Тесты общих способностей
   - Русскоязычные тесты

Примеры использования:

    # Загрузка модели с LoRA
    from mipt_master.src.inference import InferenceConfig, load_model

    config = InferenceConfig(
        model_path="models/qwen3_fashion_vocab/final",
        lora_path="models/qwen3_fashion_lora/final",
    )
    model, tokenizer = load_model(config)

    # Генерация SID
    from mipt_master.src.inference import generate, build_sid_prompt

    messages = build_sid_prompt("Red leather jacket, size M")
    result = generate(model, tokenizer, messages, config)
    print(result.parsed_sid)  # [87, 347, 660, 768]

    # Evaluation на gold items
    from mipt_master.src.inference import (
        get_gold_items,
        eval_sid_match,
        compute_metrics,
    )

    gold_items = get_gold_items()
    # ... evaluate and compute metrics

CLI запуск:

    # Оценка на gold items
    python -m mipt_master.src.inference --model-path models/qwen3_fashion_vocab/final \\
                                         --lora-path models/qwen3_fashion_lora/final \\
                                         --eval-gold

    # Сравнение BASE vs LoRA
    python -m mipt_master.src.inference --model-path models/qwen3_fashion_vocab/final \\
                                         --lora-path models/qwen3_fashion_lora/final \\
                                         --compare

    # Интерактивный режим
    python -m mipt_master.src.inference --model-path models/qwen3_fashion_vocab/final \\
                                         --interactive
"""

from .config import InferenceConfig, get_device, is_bf16_supported
from .evaluation import (
    EvaluationMetrics,
    SIDMatchResult,
    TestCaseResult,
    compute_metrics,
    eval_sid_match,
    print_metrics,
    save_results,
)
from .generation import (
    GenerationResult,
    build_sid_prompt,
    clean_answer,
    format_sid_sequence,
    generate,
    generate_batch,
    parse_sid_sequence,
)
from .model import (
    SEMANTIC_ID_TOKENS,
    get_model_info,
    load_model,
    load_model_peft,
    load_model_unsloth,
    verify_semantic_tokens,
)
from .test_cases import (
    ALL_TEST_CASES,
    GENERAL_TEST_CASES,
    GOLD_ITEMS,
    RUSSIAN_TEST_CASES,
    SID_TEST_CASES,
    SYSTEM_PROMPT_SID,
    SYSTEM_PROMPT_SID_ONLY,
    GeneralTestCase,
    GoldItemCase,
    get_all_tests,
    get_general_tests,
    get_gold_items,
    get_russian_tests,
    get_sid_tests,
)

__all__ = [
    # Config
    "InferenceConfig",
    "get_device",
    "is_bf16_supported",
    # Model
    "load_model",
    "load_model_peft",
    "load_model_unsloth",
    "verify_semantic_tokens",
    "get_model_info",
    "SEMANTIC_ID_TOKENS",
    # Generation
    "GenerationResult",
    "generate",
    "generate_batch",
    "build_sid_prompt",
    "parse_sid_sequence",
    "format_sid_sequence",
    "clean_answer",
    # Evaluation
    "SIDMatchResult",
    "EvaluationMetrics",
    "TestCaseResult",
    "eval_sid_match",
    "compute_metrics",
    "save_results",
    "print_metrics",
    # Test cases
    "GoldItemCase",
    "GeneralTestCase",
    "GOLD_ITEMS",
    "SID_TEST_CASES",
    "GENERAL_TEST_CASES",
    "RUSSIAN_TEST_CASES",
    "ALL_TEST_CASES",
    "SYSTEM_PROMPT_SID",
    "SYSTEM_PROMPT_SID_ONLY",
    "get_gold_items",
    "get_sid_tests",
    "get_general_tests",
    "get_russian_tests",
    "get_all_tests",
]

