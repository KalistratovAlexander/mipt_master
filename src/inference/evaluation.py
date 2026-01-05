"""Метрики для оценки качества semantic ID генерации."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .generation import GenerationResult

logger = logging.getLogger("inference")


@dataclass
class SIDMatchResult:
    """Результат сравнения predicted vs gold SID.

    Attributes:
        valid: Удалось ли распарсить predicted SID.
        lvl1: Совпадение на 1-м уровне.
        lvl2: Совпадение на 1-2 уровнях.
        lvl3: Совпадение на 1-2-3 уровнях.
        lvl4: Полное совпадение (все 4 уровня).
        predicted: Предсказанный SID.
        gold: Эталонный SID.
    """

    valid: bool
    lvl1: bool
    lvl2: bool
    lvl3: bool
    lvl4: bool
    predicted: Optional[list[int]]
    gold: Optional[list[int]]


def eval_sid_match(
    predicted: Optional[list[int]],
    gold: Optional[list[int]],
) -> SIDMatchResult:
    """Оценить совпадение predicted и gold SID по уровням.

    Args:
        predicted: Предсказанный SID (список из 4 чисел) или None.
        gold: Эталонный SID (список из 4 чисел) или None.

    Returns:
        SIDMatchResult с детальным разбором совпадений.

    Example:
        >>> eval_sid_match([87, 347, 660, 768], [87, 347, 500, 768])
        SIDMatchResult(valid=True, lvl1=True, lvl2=True, lvl3=False, lvl4=False, ...)
    """
    if predicted is None or gold is None:
        return SIDMatchResult(
            valid=False,
            lvl1=False,
            lvl2=False,
            lvl3=False,
            lvl4=False,
            predicted=predicted,
            gold=gold,
        )

    # Проверяем по уровням (min чтобы не выйти за границы)
    max_len = min(len(predicted), len(gold), 4)

    lvl1 = max_len >= 1 and predicted[0] == gold[0]
    lvl2 = max_len >= 2 and predicted[:2] == gold[:2]
    lvl3 = max_len >= 3 and predicted[:3] == gold[:3]
    lvl4 = max_len >= 4 and predicted[:4] == gold[:4]

    return SIDMatchResult(
        valid=True,
        lvl1=lvl1,
        lvl2=lvl2,
        lvl3=lvl3,
        lvl4=lvl4,
        predicted=predicted,
        gold=gold,
    )


@dataclass
class EvaluationMetrics:
    """Агрегированные метрики evaluation.

    Attributes:
        total: Общее число примеров.
        valid: Число примеров с валидным SID.
        valid_rate: Доля валидных.
        lvl1_accuracy: Точность на 1-м уровне.
        lvl2_accuracy: Точность на 1-2 уровнях.
        lvl3_accuracy: Точность на 1-2-3 уровнях.
        lvl4_accuracy: Точность на всех 4 уровнях (exact match).
    """

    total: int
    valid: int
    valid_rate: float
    lvl1_accuracy: float
    lvl2_accuracy: float
    lvl3_accuracy: float
    lvl4_accuracy: float


def compute_metrics(results: list[SIDMatchResult]) -> EvaluationMetrics:
    """Вычислить агрегированные метрики.

    Args:
        results: Список результатов сравнения.

    Returns:
        EvaluationMetrics.
    """
    if not results:
        return EvaluationMetrics(
            total=0,
            valid=0,
            valid_rate=0.0,
            lvl1_accuracy=0.0,
            lvl2_accuracy=0.0,
            lvl3_accuracy=0.0,
            lvl4_accuracy=0.0,
        )

    total = len(results)
    valid = sum(1 for r in results if r.valid)

    # Считаем accuracy только по валидным примерам
    valid_results = [r for r in results if r.valid]

    if not valid_results:
        return EvaluationMetrics(
            total=total,
            valid=0,
            valid_rate=0.0,
            lvl1_accuracy=0.0,
            lvl2_accuracy=0.0,
            lvl3_accuracy=0.0,
            lvl4_accuracy=0.0,
        )

    n_valid = len(valid_results)
    lvl1 = sum(1 for r in valid_results if r.lvl1) / n_valid
    lvl2 = sum(1 for r in valid_results if r.lvl2) / n_valid
    lvl3 = sum(1 for r in valid_results if r.lvl3) / n_valid
    lvl4 = sum(1 for r in valid_results if r.lvl4) / n_valid

    return EvaluationMetrics(
        total=total,
        valid=valid,
        valid_rate=valid / total,
        lvl1_accuracy=lvl1,
        lvl2_accuracy=lvl2,
        lvl3_accuracy=lvl3,
        lvl4_accuracy=lvl4,
    )


@dataclass
class TestCaseResult:
    """Результат одного тест-кейса.

    Attributes:
        case_id: Идентификатор кейса.
        title: Название продукта.
        gold_sid: Эталонный SID.
        generation: Результат генерации.
        match: Результат сравнения.
    """

    case_id: str
    title: str
    gold_sid: Optional[list[int]]
    generation: GenerationResult
    match: SIDMatchResult


def save_results(
    results: list[TestCaseResult],
    metrics: EvaluationMetrics,
    output_dir: Path,
    model_name: str = "model",
) -> tuple[Path, Path]:
    """Сохранить результаты evaluation.

    Args:
        results: Список результатов по кейсам.
        metrics: Агрегированные метрики.
        output_dir: Директория для сохранения.
        model_name: Имя модели для имени файла.

    Returns:
        Пути к txt и json файлам.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"eval_{model_name}_{ts}"

    txt_path = output_dir / f"{base_name}.txt"
    json_path = output_dir / f"{base_name}.json"

    # Human-readable report
    lines = [
        "=" * 80,
        f"EVALUATION REPORT: {model_name}",
        f"Timestamp: {ts}",
        "=" * 80,
        "",
        "SUMMARY METRICS:",
        f"  Total cases:    {metrics.total}",
        f"  Valid SIDs:     {metrics.valid} ({metrics.valid_rate:.1%})",
        f"  Level 1 acc:    {metrics.lvl1_accuracy:.1%}",
        f"  Level 1-2 acc:  {metrics.lvl2_accuracy:.1%}",
        f"  Level 1-3 acc:  {metrics.lvl3_accuracy:.1%}",
        f"  Exact match:    {metrics.lvl4_accuracy:.1%}",
        "",
        "=" * 80,
        "DETAILED RESULTS:",
        "=" * 80,
    ]

    for r in results:
        lines.extend([
            "",
            f"CASE: {r.case_id}",
            f"TITLE: {r.title}",
            f"GOLD SID: {r.gold_sid}",
            f"PRED SID: {r.match.predicted}",
            f"MATCH: lvl1={r.match.lvl1}, lvl2={r.match.lvl2}, lvl3={r.match.lvl3}, lvl4={r.match.lvl4}",
            "-" * 40,
            f"ANSWER: {r.generation.answer[:200]}...",
        ])

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # JSON for programmatic access
    json_data = {
        "model_name": model_name,
        "timestamp": ts,
        "metrics": asdict(metrics),
        "results": [
            {
                "case_id": r.case_id,
                "title": r.title,
                "gold_sid": r.gold_sid,
                "predicted_sid": r.match.predicted,
                "match": asdict(r.match),
                "answer": r.generation.answer,
            }
            for r in results
        ],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved report to: {txt_path}")
    logger.info(f"Saved JSON to: {json_path}")

    return txt_path, json_path


def print_metrics(metrics: EvaluationMetrics, model_name: str = "") -> None:
    """Вывести метрики в консоль.

    Args:
        metrics: Метрики.
        model_name: Имя модели.
    """
    header = f"METRICS: {model_name}" if model_name else "METRICS"
    print("\n" + "=" * 50)
    print(header)
    print("=" * 50)
    print(f"Total cases:    {metrics.total}")
    print(f"Valid SIDs:     {metrics.valid} ({metrics.valid_rate:.1%})")
    print(f"Level 1 acc:    {metrics.lvl1_accuracy:.1%}")
    print(f"Level 1-2 acc:  {metrics.lvl2_accuracy:.1%}")
    print(f"Level 1-3 acc:  {metrics.lvl3_accuracy:.1%}")
    print(f"Exact match:    {metrics.lvl4_accuracy:.1%}")
    print("=" * 50 + "\n")

