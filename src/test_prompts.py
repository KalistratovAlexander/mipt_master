"""Тестовые промпты для оценки LLM с semantic IDs.

Содержит системные промпты и тестовые диалоги для проверки:
- Генерации semantic IDs по описанию товара
- Декодирования semantic IDs в описание
- Рекомендаций на основе semantic IDs
- Multi-turn диалогов
"""

from __future__ import annotations

# ============================================================
# System prompts
# ============================================================

SYSTEM_PROMPT = """You are a helpful AI assistant for e-commerce and fashion products. You know how to encode and decode products using semantic IDs. Semantic IDs are hierarchical identifiers in the format <|sid_start|><|sid_105|><|sid_307|><|sid_705|><|sid_769|><|sid_end|> where each <|sid_N|> is a token from the extended vocabulary (sid_0…sid_1023). When asked about a product — by title, description or features — output its semantic ID. When asked for recommendation, you can use <|rec|> token to request suggestions. /no_think"""


SYSTEM_PROMPT_SID_ONLY = """You are a helpful AI assistant that outputs only Semantic ID in the form <|sid_start|><|sid_x|><|sid_y|><|sid_z|><|sid_w|><|sid_end|>."""


# ============================================================
# Тестовые промпты для рекомендаций
# ============================================================

REC_TEST_PROMPTS = [
    [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "Encode product: 'Red leather jacket, size M, for women' — output semantic ID only.",
        },
    ],
    [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "I have <|sid_start|><|sid_121|><|sid_422|><|sid_657|><|sid_768|><|sid_end|>. What would you recommend? <|rec|>",
        },
    ],
    [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "Item ID: <|sid_start|><|sid_183|><|sid_395|><|sid_571|><|sid_768|><|sid_end|>\nWhat is the product title and main features?",
        },
    ],
]


# ============================================================
# Полные тестовые кейсы
# ============================================================

TEST_PROMPTS = [
    # SID to text
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "What is the title of item <|sid_start|><|sid_179|><|sid_443|><|sid_563|><|sid_768|><|sid_end|>?",
            },
        ],
        "description": "SID to title",
        "expected": "Product title from semantic ID",
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "What is the description of product <|sid_start|><|sid_249|><|sid_426|><|sid_528|><|sid_768|><|sid_end|>?",
            },
        ],
        "description": "SID to description",
        "expected": "Product description from semantic ID",
    },
    # Text to SID
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": 'Product title "Men\'s Classic Cotton T-Shirt - Comfortable Fit" maps to ID:',
            },
        ],
        "description": "Title to SID",
        "expected": "<|sid_start|><|sid_*|><|sid_*|><|sid_*|><|sid_*|><|sid_end|>",
    },
    # Category navigation
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Customers shopping in Clothing > Men > Tops > T-Shirts also shop:",
            },
        ],
        "description": "Category tree transition",
        "expected": "Clothing > Men > Bottoms > Jeans or similar category",
    },
    # Recommendations
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": 'Item ID: <|sid_start|><|sid_183|><|sid_395|><|sid_571|><|sid_768|><|sid_end|>\nRecommend other items in "Amazon Fashion":\n<|rec|>',
            },
        ],
        "description": "Recommend based on category",
        "expected": "<|sid_start|><|sid_*|><|sid_*|><|sid_*|><|sid_*|><|sid_end|>, ...",
    },
    # Recommendation with explanation
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "I have <|sid_start|><|sid_121|><|sid_422|><|sid_657|><|sid_768|><|sid_end|>. What would you recommend and why?",
            },
        ],
        "description": "Recommendation with explanation",
        "expected": "SID + explanation",
    },
    # Bundle naming
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Suggest a catchy name for a bundle containing: <|sid_start|><|sid_183|><|sid_395|><|sid_571|><|sid_768|><|sid_end|>, <|sid_start|><|sid_121|><|sid_422|><|sid_657|><|sid_768|><|sid_end|>, <|sid_start|><|sid_200|><|sid_476|><|sid_706|><|sid_768|><|sid_end|>",
            },
        ],
        "description": "Bundle name",
        "expected": "Fashion Essentials Bundle or similar",
    },
    # Bundle description
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Why would these three products work well together: <|sid_start|><|sid_71|><|sid_474|><|sid_575|><|sid_768|><|sid_end|>, <|sid_start|><|sid_249|><|sid_426|><|sid_528|><|sid_768|><|sid_end|>, <|sid_start|><|sid_73|><|sid_363|><|sid_615|><|sid_768|><|sid_end|>?",
            },
        ],
        "description": "Bundle description",
        "expected": "Explanation of product compatibility",
    },
    # Multi-turn: Recommendation + Explanation
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "I have <|sid_start|><|sid_121|><|sid_422|><|sid_657|><|sid_768|><|sid_end|>. What would you recommend? <|rec|>",
            },
            {
                "role": "assistant",
                "content": "<|sid_start|><|sid_183|><|sid_395|><|sid_571|><|sid_768|><|sid_end|>",
            },
            {
                "role": "user",
                "content": "Why is this a good recommendation?",
            },
        ],
        "description": "Two-turn: Recommendation + Explanation",
        "expected": "Explanation of recommendation match",
    },
    # Multi-turn: Similar rec + Bundle name
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Recommend products similar to <|sid_start|><|sid_183|><|sid_395|><|sid_571|><|sid_768|><|sid_end|>:\n<|rec|>",
            },
            {
                "role": "assistant",
                "content": "<|sid_start|><|sid_121|><|sid_422|><|sid_657|><|sid_768|><|sid_end|>, <|sid_start|><|sid_200|><|sid_476|><|sid_706|><|sid_768|><|sid_end|>, <|sid_start|><|sid_71|><|sid_474|><|sid_575|><|sid_768|><|sid_end|>",
            },
            {
                "role": "user",
                "content": "Name this recommended bundle:",
            },
        ],
        "description": "Two-turn: Similar rec + Bundle name",
        "expected": "Matching Style Collection or similar",
    },
    # Style refinement
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "I have <|sid_start|><|sid_183|><|sid_395|><|sid_571|><|sid_768|><|sid_end|>. What would you recommend? <|rec|>",
            },
            {
                "role": "assistant",
                "content": "<|sid_start|><|sid_121|><|sid_422|><|sid_657|><|sid_768|><|sid_end|>",
            },
            {
                "role": "user",
                "content": "I prefer casual style, any recommendations?",
            },
        ],
        "description": "Two-turn: Recommendation + Style refinement",
        "expected": "SID for casual style item",
    },
    # Color refinement
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "I have <|sid_start|><|sid_249|><|sid_426|><|sid_528|><|sid_768|><|sid_end|>. What would you recommend? <|rec|>",
            },
            {
                "role": "assistant",
                "content": "<|sid_start|><|sid_73|><|sid_363|><|sid_615|><|sid_768|><|sid_end|>",
            },
            {
                "role": "user",
                "content": "Can you suggest similar items in a different color?",
            },
        ],
        "description": "Two-turn: Recommendation + Color refinement",
        "expected": "SID for different color variant",
    },
    # SID to category
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "What category does product <|sid_start|><|sid_179|><|sid_443|><|sid_563|><|sid_768|><|sid_end|> belong to?",
            },
        ],
        "description": "SID to category",
        "expected": "Amazon Fashion category path",
    },
    # Similar products
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Products similar to <|sid_start|><|sid_183|><|sid_395|><|sid_571|><|sid_768|><|sid_end|> include:",
            },
        ],
        "description": "Similar products",
        "expected": "List of SIDs",
    },
    # Description to SID
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": 'A product described as "Comfortable cotton t-shirt with modern fit" has SemanticID:',
            },
        ],
        "description": "Description to SID",
        "expected": "SID",
    },
]


# ============================================================
# Utility functions
# ============================================================

def get_rec_test_prompts() -> list[list[dict]]:
    """Получить промпты для тестирования рекомендаций.

    Returns:
        Список диалогов в chat format.
    """
    return REC_TEST_PROMPTS


def get_all_test_prompts() -> list[dict]:
    """Получить все тестовые промпты.

    Returns:
        Список тестовых кейсов с messages, description, expected.
    """
    return TEST_PROMPTS

