"""Функции генерации и парсинга semantic IDs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from .config import InferenceConfig


@dataclass
class GenerationResult:
    """Результат генерации.

    Attributes:
        prompt: Исходный промпт.
        full_output: Полный вывод (промпт + ответ).
        answer: Только ответ модели.
        parsed_sid: Распарсенный SID (список чисел) или None.
    """

    prompt: str
    full_output: str
    answer: str
    parsed_sid: Optional[list[int]]


def parse_sid_sequence(text: str) -> Optional[list[int]]:
    """Извлечь semantic ID из текста.

    Ищет паттерн: <|sid_start|><|sid_X|><|sid_Y|>...<|sid_end|>

    Args:
        text: Текст для парсинга.

    Returns:
        Список чисел SID или None если не найден.

    Example:
        >>> parse_sid_sequence("<|sid_start|><|sid_87|><|sid_347|><|sid_660|><|sid_768|><|sid_end|>")
        [87, 347, 660, 768]
    """
    if not text:
        return None

    # Ищем полный SID блок
    pattern = r"<\|sid_start\|>((?:<\|sid_\d+\|>)+)<\|sid_end\|>"
    match = re.search(pattern, text)

    if not match:
        return None

    # Извлекаем отдельные ID
    sid_block = match.group(1)
    ids = re.findall(r"<\|sid_(\d+)\|>", sid_block)

    if not ids:
        return None

    return [int(i) for i in ids]


def format_sid_sequence(ids: list[int]) -> str:
    """Форматировать список ID в строку SID.

    Args:
        ids: Список чисел.

    Returns:
        Форматированная строка.

    Example:
        >>> format_sid_sequence([87, 347, 660, 768])
        '<|sid_start|><|sid_87|><|sid_347|><|sid_660|><|sid_768|><|sid_end|>'
    """
    inner = "".join(f"<|sid_{i}|>" for i in ids)
    return f"<|sid_start|>{inner}<|sid_end|>"


def clean_answer(answer: str) -> str:
    """Очистить ответ от лишних токенов.

    Args:
        answer: Сырой ответ.

    Returns:
        Очищенный ответ.
    """
    # Удаляем <|im_end|> и всё после
    if "<|im_end|>" in answer:
        answer = answer.split("<|im_end|>")[0]

    # Удаляем thinking блоки
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)

    return answer.strip()


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
    config: InferenceConfig,
) -> GenerationResult:
    """Сгенерировать ответ на сообщения.

    Args:
        model: Модель.
        tokenizer: Токенизатор.
        messages: Сообщения в chat формате.
        config: Конфигурация генерации.

    Returns:
        GenerationResult с ответом и парсингом.
    """
    # Применяем chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Токенизируем
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Генерируем
    if config.do_sample:
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            do_sample=True,
        )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
        )

    # Декодируем
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    answer = full_output[len(prompt):]

    # Очищаем и парсим SID
    cleaned_answer = clean_answer(answer)
    parsed_sid = parse_sid_sequence(cleaned_answer)

    return GenerationResult(
        prompt=prompt,
        full_output=full_output,
        answer=cleaned_answer,
        parsed_sid=parsed_sid,
    )


@torch.no_grad()
def generate_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages_list: list[list[dict]],
    config: InferenceConfig,
) -> list[GenerationResult]:
    """Сгенерировать ответы для батча сообщений.

    Args:
        model: Модель.
        tokenizer: Токенизатор.
        messages_list: Список сообщений.
        config: Конфигурация.

    Returns:
        Список GenerationResult.
    """
    results = []
    for messages in messages_list:
        result = generate(model, tokenizer, messages, config)
        results.append(result)
    return results


def build_sid_prompt(
    product_title: str,
    system_prompt: Optional[str] = None,
) -> list[dict]:
    """Построить промпт для генерации SID.

    Args:
        product_title: Название продукта.
        system_prompt: Системный промпт (опционально).

    Returns:
        Сообщения в chat формате.
    """
    if system_prompt is None:
        system_prompt = (
            "You are a helpful AI assistant that understands and works with "
            "semantic IDs for product recommendations.\n\n"
            "Semantic IDs are hierarchical identifiers in the format "
            "<|sid_start|><|sid_0|><|sid_256|><|sid_512|><|sid_768|><|sid_end|> "
            "that encode product relationships and categories.\n\n"
            "Use <|rec|> token when the user is asking for recommendations."
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'The product "{product_title}" has SemanticID:'},
    ]

