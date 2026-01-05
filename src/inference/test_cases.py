"""Тестовые кейсы для evaluation моделей."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .generation import parse_sid_sequence


# ============================================================
# System prompts
# ============================================================

SYSTEM_PROMPT_SID = """You are a helpful AI assistant that understands and works with semantic IDs for product recommendations.

Semantic IDs are hierarchical identifiers in the format <|sid_start|><|sid_0|><|sid_256|><|sid_512|><|sid_768|><|sid_end|> that encode product relationships and categories.

Use <|rec|> token when the user is asking for recommendations."""


SYSTEM_PROMPT_SID_ONLY = """You are a helpful AI assistant that outputs only Semantic ID in the form <|sid_start|><|sid_x|><|sid_y|><|sid_z|><|sid_w|><|sid_end|>."""


# ============================================================
# Test case types
# ============================================================

@dataclass
class GoldItemCase:
    """Тест-кейс с gold (эталонным) semantic ID.

    Attributes:
        case_id: Уникальный идентификатор.
        title: Название продукта.
        gold_sid_str: Эталонный SID как строка.
        gold_sid: Распарсенный SID (список int).
    """

    case_id: str
    title: str
    gold_sid_str: str
    gold_sid: Optional[list[int]] = None

    def __post_init__(self) -> None:
        if self.gold_sid is None:
            self.gold_sid = parse_sid_sequence(self.gold_sid_str)


@dataclass
class GeneralTestCase:
    """Тест-кейс для проверки общих способностей модели.

    Attributes:
        case_id: Уникальный идентификатор.
        description: Описание теста.
        messages: Сообщения в chat формате.
        max_new_tokens: Максимум токенов для генерации.
    """

    case_id: str
    description: str
    messages: list[dict]
    max_new_tokens: int = 128


# ============================================================
# Gold items from Amazon Fashion (реальные товары с SID)
# ============================================================

GOLD_ITEMS: list[GoldItemCase] = [
    GoldItemCase(
        case_id="GOLD_1",
        title="Momme 10-Piece Anime Ring Set with Necklace",
        gold_sid_str="<|sid_start|><|sid_19|><|sid_428|><|sid_538|><|sid_768|><|sid_end|>",
    ),
    GoldItemCase(
        case_id="GOLD_2",
        title="Festivals Kaleidoscope Glasses Rainbow Prism Sunglasses",
        gold_sid_str="<|sid_start|><|sid_63|><|sid_410|><|sid_690|><|sid_768|><|sid_end|>",
    ),
    GoldItemCase(
        case_id="GOLD_3",
        title="Silvertone 1 Corinthians 13 Bible Verse Double Stretch Bracelet with Heart Charm",
        gold_sid_str="<|sid_start|><|sid_123|><|sid_453|><|sid_699|><|sid_768|><|sid_end|>",
    ),
    GoldItemCase(
        case_id="GOLD_4",
        title="Silver Plated Volcano Dangle Earrings with Crystals",
        gold_sid_str="<|sid_start|><|sid_87|><|sid_421|><|sid_640|><|sid_768|><|sid_end|>",
    ),
    GoldItemCase(
        case_id="GOLD_5",
        title="Wonder Woman Classic Logo Oval Dangle Charm Earrings",
        gold_sid_str="<|sid_start|><|sid_173|><|sid_397|><|sid_549|><|sid_768|><|sid_end|>",
    ),
    GoldItemCase(
        case_id="GOLD_6",
        title="Women's Adjustable Layered Shirt Extender Skirt",
        gold_sid_str="<|sid_start|><|sid_89|><|sid_386|><|sid_629|><|sid_768|><|sid_end|>",
    ),
    GoldItemCase(
        case_id="GOLD_7",
        title="Clearly Charming Multi-Color Ribbon Autism Awareness Italian Charm",
        gold_sid_str="<|sid_start|><|sid_69|><|sid_369|><|sid_560|><|sid_768|><|sid_end|>",
    ),
    GoldItemCase(
        case_id="GOLD_8",
        title="Namom Tiger Graphic Cotton T-Shirt for Teens (Black, Medium)",
        gold_sid_str="<|sid_start|><|sid_36|><|sid_256|><|sid_628|><|sid_768|><|sid_end|>",
    ),
    GoldItemCase(
        case_id="GOLD_9",
        title="Merrell Women's Zoe Sojourn Lace Knit Q2 Sneaker",
        gold_sid_str="<|sid_start|><|sid_167|><|sid_274|><|sid_709|><|sid_768|><|sid_end|>",
    ),
    GoldItemCase(
        case_id="GOLD_10",
        title="EARGFM Men's Compression Workout Athletic T-Shirt - White, XX-Large",
        gold_sid_str="<|sid_start|><|sid_115|><|sid_335|><|sid_599|><|sid_768|><|sid_end|>",
    ),
]


# ============================================================
# Простые SID-тесты (без gold, проверяем формат)
# ============================================================

SID_TEST_CASES: list[GeneralTestCase] = [
    GeneralTestCase(
        case_id="SID_1",
        description="SID for red leather jacket",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_SID},
            {"role": "user", "content": "A product described as 'Red leather jacket, size M, for women' has SemanticID:"},
        ],
        max_new_tokens=64,
    ),
    GeneralTestCase(
        case_id="SID_2",
        description="SID for evening dress",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_SID},
            {"role": "user", "content": "A product described as 'Elegant black evening dress with lace details, size S' has SemanticID:"},
        ],
        max_new_tokens=64,
    ),
    GeneralTestCase(
        case_id="SID_3",
        description="SID for white sneakers",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_SID},
            {"role": "user", "content": "A product described as 'White sneakers with chunky sole, unisex, size 42 EU' has SemanticID:"},
        ],
        max_new_tokens=64,
    ),
]


# ============================================================
# Тесты общих способностей (перевод, код, объяснения)
# ============================================================

GENERAL_TEST_CASES: list[GeneralTestCase] = [
    GeneralTestCase(
        case_id="GEN_1",
        description="Translation RU -> EN",
        messages=[
            {"role": "user", "content": 'Translate to English: "Я разрабатываю рекомендательные системы для e-commerce."'},
        ],
        max_new_tokens=64,
    ),
    GeneralTestCase(
        case_id="GEN_2",
        description="Python factorial function",
        messages=[
            {"role": "user", "content": "Write a short Python function that computes factorial of a number."},
        ],
        max_new_tokens=128,
    ),
    GeneralTestCase(
        case_id="GEN_3",
        description="LTV estimation question",
        messages=[
            {"role": "user", "content": "I have sales data of an online shop. How can I estimate the lifetime value (LTV) of a customer?"},
        ],
        max_new_tokens=128,
    ),
]


# ============================================================
# Тесты на русском языке
# ============================================================

RUSSIAN_TEST_CASES: list[GeneralTestCase] = [
    GeneralTestCase(
        case_id="RU_SID_1",
        description="SID для красной куртки (RU)",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_SID},
            {"role": "user", "content": "Товар, описанный как «Красная кожаная куртка, размер M, для женщин», имеет SemanticID:"},
        ],
        max_new_tokens=64,
    ),
    GeneralTestCase(
        case_id="RU_SID_2",
        description="SID для вечернего платья (RU)",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_SID},
            {"role": "user", "content": "Товар, описанный как «Элегантное чёрное вечернее платье с кружевом, размер S», имеет SemanticID:"},
        ],
        max_new_tokens=64,
    ),
    GeneralTestCase(
        case_id="RU_GEN_1",
        description="Python factorial (RU)",
        messages=[
            {"role": "user", "content": "Напиши короткую функцию на Python, которая вычисляет факториал числа."},
        ],
        max_new_tokens=128,
    ),
    GeneralTestCase(
        case_id="RU_GEN_2",
        description="Recommendation system explanation (RU)",
        messages=[
            {"role": "user", "content": "Объясни простыми словами по-русски, как работает рекомендательная система в интернет-магазине."},
        ],
        max_new_tokens=192,
    ),
]


# ============================================================
# Все тест-кейсы
# ============================================================

ALL_TEST_CASES: list[GeneralTestCase] = SID_TEST_CASES + GENERAL_TEST_CASES + RUSSIAN_TEST_CASES


def get_gold_items() -> list[GoldItemCase]:
    """Получить все gold items для evaluation.

    Returns:
        Список GoldItemCase.
    """
    return GOLD_ITEMS


def get_sid_tests() -> list[GeneralTestCase]:
    """Получить SID тесты (без gold).

    Returns:
        Список GeneralTestCase.
    """
    return SID_TEST_CASES


def get_general_tests() -> list[GeneralTestCase]:
    """Получить тесты общих способностей.

    Returns:
        Список GeneralTestCase.
    """
    return GENERAL_TEST_CASES


def get_russian_tests() -> list[GeneralTestCase]:
    """Получить тесты на русском.

    Returns:
        Список GeneralTestCase.
    """
    return RUSSIAN_TEST_CASES


def get_all_tests() -> list[GeneralTestCase]:
    """Получить все тесты.

    Returns:
        Список GeneralTestCase.
    """
    return ALL_TEST_CASES

