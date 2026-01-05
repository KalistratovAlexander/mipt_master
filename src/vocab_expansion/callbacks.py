"""Callbacks для мониторинга обучения эмбеддингов."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
from transformers import TrainerCallback

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase, Trainer, TrainingArguments
    from transformers.trainer_callback import TrainerControl, TrainerState

    from .config import VocabExpansionConfig

logger = logging.getLogger("vocab-expansion")


def _safe_wandb_log(data: dict, use_wandb: bool = True) -> None:
    """Безопасно логировать в wandb если включён.

    Args:
        data: Данные для логирования.
        use_wandb: Использовать wandb.
    """
    if not use_wandb:
        return
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(data)
    except Exception:
        pass  # wandb не доступен


class DataInspectionCallback(TrainerCallback):
    """Инспекция первого батча при старте обучения.

    Логирует информацию о структуре батча и декодированный пример.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Инициализация.

        Args:
            tokenizer: Токенизатор для декодирования.
        """
        self.tokenizer = tokenizer
        self.trainer: Optional[Trainer] = None
        self.first_batch_inspected = False

    def set_trainer(self, trainer: Trainer) -> None:
        """Установить trainer для доступа к dataloader.

        Args:
            trainer: Trainer instance.
        """
        self.trainer = trainer

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Инспектировать первый батч при старте."""
        if self.first_batch_inspected or self.trainer is None:
            return
        self.first_batch_inspected = True

        logger.info("\n" + "=" * 60)
        logger.info("=== Initial Training Data Inspection ===")
        logger.info("=" * 60)

        try:
            train_dataloader = self.trainer.get_train_dataloader()
            for batch in train_dataloader:
                logger.info(f"Batch keys: {batch.keys()}")
                logger.info(f"Input IDs shape: {batch['input_ids'].shape}")
                logger.info(f"Attention mask shape: {batch['attention_mask'].shape}")

                first = batch["input_ids"][0]
                decoded = self.tokenizer.decode(first, skip_special_tokens=False)

                logger.info(f"First example tokens: {first[:50].tolist()}...")
                logger.info(f"Decoded (truncated): {decoded[:300]}...")
                break
        except Exception as e:
            logger.warning(f"Could not inspect first batch: {e}")

        logger.info("=" * 60 + "\n")


class EmbeddingMonitorCallback(TrainerCallback):
    """Мониторинг статистики эмбеддингов во время обучения.

    Отслеживает:
    - Изменение эмбеддингов относительно начального состояния
    - Изменение эмбеддингов относительно предыдущего шага
    - Статистику (mean, std, norm) по уровням иерархии
    - Градиенты эмбеддингов
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_new_tokens: int,
        config: VocabExpansionConfig,
        monitor_interval: int = 100,
    ) -> None:
        """Инициализация.

        Args:
            tokenizer: Токенизатор.
            num_new_tokens: Число добавленных токенов.
            config: Конфигурация.
            monitor_interval: Интервал мониторинга в шагах.
        """
        self.tokenizer = tokenizer
        self.num_new_tokens = num_new_tokens
        self.monitor_interval = monitor_interval
        self.original_vocab_size = len(tokenizer) - num_new_tokens
        self.initial_embeddings: Optional[torch.Tensor] = None
        self.prev_embeddings: Optional[torch.Tensor] = None
        self.config = config

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        """Сохранить начальные эмбеддинги."""
        embeddings = model.get_input_embeddings().weight
        self.initial_embeddings = embeddings[self.original_vocab_size:].clone().detach()
        self.prev_embeddings = self.initial_embeddings.clone()

        mean = self.initial_embeddings.mean().item()
        std = self.initial_embeddings.std().item()
        norm = self.initial_embeddings.norm(dim=-1).mean().item()

        _safe_wandb_log(
            {
                "embeddings/initial_mean": mean,
                "embeddings/initial_std": std,
                "embeddings/initial_norm": norm,
            },
            use_wandb=self.config.use_wandb,
        )
        logger.info(f"Initial embeddings — Mean: {mean:.4f}, Std: {std:.4f}, Norm: {norm:.4f}")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        """Логировать статистику эмбеддингов."""
        if state.global_step == 0 or state.global_step % self.monitor_interval != 0:
            return

        embeddings = model.get_input_embeddings().weight
        new_embeddings = embeddings[self.original_vocab_size:]

        # Изменения
        change_from_init = (new_embeddings - self.initial_embeddings).abs().mean().item()
        change_from_prev = (new_embeddings - self.prev_embeddings).abs().mean().item()

        # Общая статистика
        mean = new_embeddings.mean().item()
        std = new_embeddings.std().item()
        norm = new_embeddings.norm(dim=-1).mean().item()

        # Статистика по уровням
        level_stats = {}
        tokens_per_level = self.config.codebook_size
        for level in range(self.config.codebook_levels):
            start_idx = level * tokens_per_level
            end_idx = min((level + 1) * tokens_per_level, self.num_new_tokens)
            if start_idx < self.num_new_tokens:
                level_emb = new_embeddings[start_idx:end_idx]
                level_stats[f"embeddings/level_{level}_mean"] = level_emb.mean().item()
                level_stats[f"embeddings/level_{level}_std"] = level_emb.std().item()
                level_stats[f"embeddings/level_{level}_norm"] = level_emb.norm(dim=-1).mean().item()

        # Градиенты
        grad_norm = 0.0
        grad_max = 0.0
        if embeddings.grad is not None:
            grad = embeddings.grad[self.original_vocab_size:]
            grad_norm = grad.norm().item()
            grad_max = grad.abs().max().item()

        wandb_log = {
            "embeddings/change_from_init": change_from_init,
            "embeddings/change_from_prev": change_from_prev,
            "embeddings/mean": mean,
            "embeddings/std": std,
            "embeddings/norm": norm,
            "embeddings/grad_norm": grad_norm,
            "embeddings/grad_max": grad_max,
            "step": state.global_step,
            **level_stats,
        }
        _safe_wandb_log(wandb_log, use_wandb=self.config.use_wandb)

        logger.info(
            f"Step {state.global_step} — Embeddings: "
            f"Δinit={change_from_init:.4f}, Δprev={change_from_prev:.6f}, "
            f"Mean={mean:.4f}, Std={std:.4f}, Norm={norm:.4f}"
        )

        self.prev_embeddings = new_embeddings.clone().detach()


class SemanticIDGenerationCallback(TrainerCallback):
    """Тестирование генерации semantic ID во время обучения.

    Периодически генерирует ответы на тестовые промпты и проверяет
    использование semantic ID токенов.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: VocabExpansionConfig,
        test_prompts: list[list[dict]],
        test_interval: int = 200,
    ) -> None:
        """Инициализация.

        Args:
            tokenizer: Токенизатор.
            config: Конфигурация.
            test_prompts: Список тестовых промптов (chat format).
            test_interval: Интервал тестирования в шагах.
        """
        self.tokenizer = tokenizer
        self.config = config
        self.test_interval = test_interval
        self.test_messages = test_prompts

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        """Запустить тестовую генерацию."""
        if state.global_step == 0 or state.global_step % self.test_interval != 0:
            return
        self._test_generation(model, state.global_step)

    def _test_generation(self, model, step: int) -> None:
        """Выполнить тестовую генерацию.

        Args:
            model: Модель.
            step: Текущий шаг.
        """
        logger.info("=" * 60)
        logger.info(f"Testing semantic ID generation at step {step}")
        logger.info("=" * 60)

        training_mode = model.training
        model.eval()

        successful_generations = 0
        generation_results = []

        for i, messages in enumerate(self.test_messages, 1):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        min_p=0.01,
                        top_p=0.8,
                        top_k=20,
                    )

                generated_full = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                generated_new = generated_full[len(prompt):]

                # Проверяем наличие semantic ID
                has_sid_tags = "<|sid_start|>" in generated_new or "<|sid_end|>" in generated_new
                sid_tokens = [
                    t for t in generated_new.split()
                    if t.startswith("<|sid_") and t.endswith("|>")
                ]
                uses_semantic_ids = has_sid_tags or len(sid_tokens) > 0

                if uses_semantic_ids:
                    successful_generations += 1

                user_message = messages[-1]["content"]
                generation_results.append([
                    step, user_message, generated_new, uses_semantic_ids, len(sid_tokens)
                ])

                logger.info(f"\nTest {i}: {user_message[:50]}...")
                logger.info(f"  Generated: {generated_new[:100]}...")
                logger.info(f"  Uses SIDs: {uses_semantic_ids} (tokens={len(sid_tokens)})")

            except Exception as e:
                user_message = messages[-1]["content"]
                logger.warning(f"Generation failed for prompt {i}: {e}")
                generation_results.append([step, user_message[:50], f"[Error: {e}]", False, 0])

        success_rate = successful_generations / len(self.test_messages) if self.test_messages else 0

        # Логируем в wandb
        if self.config.use_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log({
                        "generation/success_rate": success_rate,
                        "generation/successful_count": successful_generations,
                        "generation/total_prompts": len(self.test_messages),
                        "generation/examples": wandb.Table(
                            columns=["Step", "User_Message", "Generated", "Uses_SID", "Num_Tokens"],
                            data=generation_results,
                        ),
                        "step": step,
                    })
            except Exception:
                pass

        logger.info(
            f"\nSummary: {successful_generations}/{len(self.test_messages)} "
            f"({success_rate:.0%}) prompts generated semantic IDs"
        )

        model.train(training_mode)
        logger.info("=" * 60 + "\n")


def create_callbacks(
    tokenizer: PreTrainedTokenizerBase,
    config: VocabExpansionConfig,
    num_new_tokens: int,
    test_prompts: list[list[dict]],
) -> tuple[list[TrainerCallback], DataInspectionCallback]:
    """Создать все callbacks для обучения.

    Args:
        tokenizer: Токенизатор.
        config: Конфигурация.
        num_new_tokens: Число добавленных токенов.
        test_prompts: Тестовые промпты.

    Returns:
        Кортеж (список callbacks, data_inspection_callback).
    """
    data_inspection = DataInspectionCallback(tokenizer)

    embedding_monitor = EmbeddingMonitorCallback(
        tokenizer=tokenizer,
        num_new_tokens=num_new_tokens,
        config=config,
        monitor_interval=config.steps_per_val_log,
    )

    generation_test = SemanticIDGenerationCallback(
        tokenizer=tokenizer,
        config=config,
        test_prompts=test_prompts,
        test_interval=config.steps_per_val_log,
    )

    callbacks = [data_inspection, embedding_monitor, generation_test]
    return callbacks, data_inspection

