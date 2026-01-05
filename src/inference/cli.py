"""CLI для inference и evaluation моделей."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from mipt_master.src.logger import setup_logger

from .config import InferenceConfig
from .evaluation import (
    EvaluationMetrics,
    TestCaseResult,
    compute_metrics,
    eval_sid_match,
    print_metrics,
    save_results,
)
from .generation import GenerationResult, build_sid_prompt, generate, parse_sid_sequence
from .model import get_model_info, load_model
from .test_cases import (
    GOLD_ITEMS,
    SYSTEM_PROMPT_SID,
    GeneralTestCase,
    GoldItemCase,
    get_all_tests,
    get_general_tests,
    get_gold_items,
    get_russian_tests,
    get_sid_tests,
)

logger = setup_logger("inference", log_to_file=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    p = argparse.ArgumentParser(
        description="Inference and evaluation for semantic ID models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate gold items with LoRA
  python -m mipt_master.src.inference --model-path models/qwen3_fashion_vocab/final \\
                                       --lora-path models/qwen3_fashion_lora/final \\
                                       --eval-gold

  # Run all tests with base model
  python -m mipt_master.src.inference --model-path models/qwen3_fashion_vocab/final \\
                                       --eval-all

  # Interactive generation
  python -m mipt_master.src.inference --model-path models/qwen3_fashion_vocab/final \\
                                       --interactive

  # Compare BASE vs LoRA
  python -m mipt_master.src.inference --model-path models/qwen3_fashion_vocab/final \\
                                       --lora-path models/qwen3_fashion_lora/final \\
                                       --compare
        """,
    )

    # Model settings
    model_group = p.add_argument_group("Model")
    model_group.add_argument(
        "--model-path",
        type=str,
        default="models/qwen3_fashion_vocab/final",
        help="Path to base model (Stage 1 or merged)",
    )
    model_group.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA adapter (optional)",
    )
    model_group.add_argument(
        "--use-unsloth",
        action="store_true",
        help="Use unsloth instead of PEFT for loading",
    )
    model_group.add_argument(
        "--load-4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )

    # Evaluation modes
    eval_group = p.add_argument_group("Evaluation")
    eval_group.add_argument(
        "--eval-gold",
        action="store_true",
        help="Evaluate on gold items (with accuracy metrics)",
    )
    eval_group.add_argument(
        "--eval-sid",
        action="store_true",
        help="Run SID format tests",
    )
    eval_group.add_argument(
        "--eval-general",
        action="store_true",
        help="Run general capability tests",
    )
    eval_group.add_argument(
        "--eval-russian",
        action="store_true",
        help="Run Russian language tests",
    )
    eval_group.add_argument(
        "--eval-all",
        action="store_true",
        help="Run all tests",
    )
    eval_group.add_argument(
        "--compare",
        action="store_true",
        help="Compare BASE (no LoRA) vs LoRA model",
    )

    # Interactive mode
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive generation mode",
    )

    # Generation settings
    gen_group = p.add_argument_group("Generation")
    gen_group.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum new tokens to generate",
    )
    gen_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    gen_group.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling",
    )
    gen_group.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding (temperature=0)",
    )

    # Output settings
    output_group = p.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_results"),
        help="Directory for saving results",
    )
    output_group.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files",
    )

    return p.parse_args(argv)


def evaluate_gold_items(
    model,
    tokenizer,
    config: InferenceConfig,
    gold_items: list[GoldItemCase],
) -> tuple[list[TestCaseResult], EvaluationMetrics]:
    """Оценить модель на gold items.

    Args:
        model: Модель.
        tokenizer: Токенизатор.
        config: Конфигурация.
        gold_items: Список gold items.

    Returns:
        (results, metrics)
    """
    results: list[TestCaseResult] = []
    match_results = []

    logger.info(f"Evaluating on {len(gold_items)} gold items...")

    for item in tqdm(gold_items, desc="Evaluating"):
        # Строим промпт
        messages = build_sid_prompt(item.title, SYSTEM_PROMPT_SID)

        # Генерируем
        gen_result = generate(model, tokenizer, messages, config)

        # Сравниваем с gold
        match = eval_sid_match(gen_result.parsed_sid, item.gold_sid)
        match_results.append(match)

        results.append(
            TestCaseResult(
                case_id=item.case_id,
                title=item.title,
                gold_sid=item.gold_sid,
                generation=gen_result,
                match=match,
            )
        )

        # Логируем
        status = "✓" if match.lvl4 else ("~" if match.valid else "✗")
        logger.debug(
            f"{status} {item.case_id}: pred={gen_result.parsed_sid}, gold={item.gold_sid}"
        )

    metrics = compute_metrics(match_results)
    return results, metrics


def run_general_tests(
    model,
    tokenizer,
    config: InferenceConfig,
    test_cases: list[GeneralTestCase],
) -> list[tuple[GeneralTestCase, GenerationResult]]:
    """Запустить общие тесты.

    Args:
        model: Модель.
        tokenizer: Токенизатор.
        config: Конфигурация.
        test_cases: Тест-кейсы.

    Returns:
        Список (test_case, generation_result).
    """
    results = []

    logger.info(f"Running {len(test_cases)} general tests...")

    for case in tqdm(test_cases, desc="Testing"):
        # Временно меняем max_new_tokens
        orig_max_tokens = config.max_new_tokens
        config.max_new_tokens = case.max_new_tokens

        gen_result = generate(model, tokenizer, case.messages, config)

        config.max_new_tokens = orig_max_tokens

        results.append((case, gen_result))

        logger.info(f"\n{case.case_id}: {case.description}")
        logger.info(f"Answer: {gen_result.answer[:200]}...")
        if gen_result.parsed_sid:
            logger.info(f"Parsed SID: {gen_result.parsed_sid}")

    return results


def compare_models(
    config: InferenceConfig,
    gold_items: list[GoldItemCase],
) -> None:
    """Сравнить BASE vs LoRA модели.

    Args:
        config: Конфигурация.
        gold_items: Gold items.
    """
    logger.info("=" * 60)
    logger.info("COMPARISON: BASE vs LoRA")
    logger.info("=" * 60)

    # BASE model (без LoRA)
    logger.info("\n--- Loading BASE model (no LoRA) ---")
    base_config = InferenceConfig(
        model_path=config.model_path,
        lora_path=None,
        use_peft=config.use_peft,
        device=config.device,
        dtype=config.dtype,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        do_sample=config.do_sample,
        output_dir=config.output_dir,
        save_results=config.save_results,
    )

    base_model, tokenizer = load_model(base_config)
    base_results, base_metrics = evaluate_gold_items(
        base_model, tokenizer, base_config, gold_items
    )

    print_metrics(base_metrics, "BASE (Stage 1)")

    if config.save_results:
        save_results(base_results, base_metrics, config.output_dir, "base")

    # Освобождаем память
    del base_model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # LoRA model
    if config.lora_path:
        logger.info("\n--- Loading LoRA model ---")
        lora_model, _ = load_model(config)
        lora_results, lora_metrics = evaluate_gold_items(
            lora_model, tokenizer, config, gold_items
        )

        print_metrics(lora_metrics, "LoRA (Stage 2)")

        if config.save_results:
            save_results(lora_results, lora_metrics, config.output_dir, "lora")

        # Сравнение
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Metric':<20} {'BASE':>12} {'LoRA':>12} {'Diff':>12}")
        print("-" * 60)
        print(f"{'Valid rate':<20} {base_metrics.valid_rate:>11.1%} {lora_metrics.valid_rate:>11.1%} {(lora_metrics.valid_rate - base_metrics.valid_rate):>+11.1%}")
        print(f"{'Level 1 acc':<20} {base_metrics.lvl1_accuracy:>11.1%} {lora_metrics.lvl1_accuracy:>11.1%} {(lora_metrics.lvl1_accuracy - base_metrics.lvl1_accuracy):>+11.1%}")
        print(f"{'Level 2 acc':<20} {base_metrics.lvl2_accuracy:>11.1%} {lora_metrics.lvl2_accuracy:>11.1%} {(lora_metrics.lvl2_accuracy - base_metrics.lvl2_accuracy):>+11.1%}")
        print(f"{'Level 3 acc':<20} {base_metrics.lvl3_accuracy:>11.1%} {lora_metrics.lvl3_accuracy:>11.1%} {(lora_metrics.lvl3_accuracy - base_metrics.lvl3_accuracy):>+11.1%}")
        print(f"{'Exact match':<20} {base_metrics.lvl4_accuracy:>11.1%} {lora_metrics.lvl4_accuracy:>11.1%} {(lora_metrics.lvl4_accuracy - base_metrics.lvl4_accuracy):>+11.1%}")
        print("=" * 60)
    else:
        logger.warning("No LoRA path specified for comparison. Only BASE results shown.")


def interactive_mode(model, tokenizer, config: InferenceConfig) -> None:
    """Интерактивный режим генерации.

    Args:
        model: Модель.
        tokenizer: Токенизатор.
        config: Конфигурация.
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("Enter product title to generate SemanticID")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")

    while True:
        try:
            title = input("Product title: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if title.lower() in ("quit", "exit", "q"):
            break

        if not title:
            continue

        messages = build_sid_prompt(title, SYSTEM_PROMPT_SID)
        result = generate(model, tokenizer, messages, config)

        print(f"\nAnswer: {result.answer}")
        if result.parsed_sid:
            print(f"Parsed SID: {result.parsed_sid}")
        print()


def main(argv: Optional[list[str]] = None) -> None:
    """Точка входа CLI."""
    args = parse_args(argv)

    # Создаём конфигурацию
    config = InferenceConfig(
        model_path=args.model_path,
        lora_path=args.lora_path,
        use_peft=not args.use_unsloth,
        load_in_4bit=args.load_4bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature if not args.greedy else 0.0,
        top_p=args.top_p,
        do_sample=not args.greedy,
        output_dir=args.output_dir,
        save_results=not args.no_save,
    )

    config.validate()
    config.log_config()

    # Режим сравнения
    if args.compare:
        compare_models(config, get_gold_items())
        return

    # Загружаем модель
    model, tokenizer = load_model(config)
    model_info = get_model_info(model)
    logger.info(f"Model info: {model_info}")

    # Интерактивный режим
    if args.interactive:
        interactive_mode(model, tokenizer, config)
        return

    # Evaluation режимы
    if args.eval_gold or args.eval_all:
        results, metrics = evaluate_gold_items(
            model, tokenizer, config, get_gold_items()
        )
        print_metrics(metrics, "Gold Items")

        if config.save_results:
            model_name = "lora" if config.lora_path else "base"
            save_results(results, metrics, config.output_dir, f"{model_name}_gold")

    if args.eval_sid or args.eval_all:
        run_general_tests(model, tokenizer, config, get_sid_tests())

    if args.eval_general or args.eval_all:
        run_general_tests(model, tokenizer, config, get_general_tests())

    if args.eval_russian or args.eval_all:
        run_general_tests(model, tokenizer, config, get_russian_tests())

    # Если ничего не выбрано — показываем help
    if not any([
        args.eval_gold,
        args.eval_sid,
        args.eval_general,
        args.eval_russian,
        args.eval_all,
        args.interactive,
        args.compare,
    ]):
        logger.info("No action specified. Use --help for options.")
        logger.info("Examples:")
        logger.info("  --eval-gold    : Evaluate on gold items with metrics")
        logger.info("  --eval-all     : Run all tests")
        logger.info("  --interactive  : Interactive SID generation")
        logger.info("  --compare      : Compare BASE vs LoRA")


if __name__ == "__main__":
    main()

