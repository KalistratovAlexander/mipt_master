"""Callbacks для мониторинга LoRA fine-tuning."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
from transformers import TrainerCallback

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase, TrainingArguments
    from transformers.trainer_callback import TrainerControl, TrainerState

    from .config import LoRAConfig

logger = logging.getLogger("finetune-lora")


def _safe_wandb_log(data: dict, step: int, use_wandb: bool = True) -> None:
    """Безопасно логировать в wandb.

    Args:
        data: Данные для логирования.
        step: Текущий шаг.
        use_wandb: Использовать wandb.
    """
    if not use_wandb:
        return
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(data, step=step)
    except Exception:
        pass


class GenerationEvalCallback(TrainerCallback):
    """Callback для тестирования генерации во время обучения.

    Периодически генерирует ответы на generic и SID промпты,
    проверяя использование semantic ID токенов.

    Attributes:
        tokenizer: Токенизатор.
        config: Конфигурация LoRA.
        interval_steps: Интервал тестирования в шагах.
        generic_prompts: Промпты для general теста.
        sid_prompts: Промпты для SID теста.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: LoRAConfig,
        generic_prompts: list[dict],
        sid_prompts: list[dict],
        interval_steps: int = 500,
        max_new_tokens: int = 64,
    ) -> None:
        """Инициализация.

        Args:
            tokenizer: Токенизатор.
            config: Конфигурация.
            generic_prompts: Промпты для general теста.
            sid_prompts: Промпты для SID теста.
            interval_steps: Интервал тестирования.
            max_new_tokens: Максимум новых токенов при генерации.
        """
        self.tokenizer = tokenizer
        self.config = config
        self.interval_steps = interval_steps
        self.max_new_tokens = max_new_tokens

        # Берём подмножество промптов
        self.generic_prompts = generic_prompts[:3] if generic_prompts else []
        self.sid_prompts = sid_prompts[:3] if sid_prompts else []

    @torch.no_grad()
    def _run_generation(
        self,
        model,
        messages: list[dict],
        description: str,
    ) -> tuple[str, bool]:
        """Выполнить генерацию для одного промпта.

        Args:
            model: Модель.
            messages: Сообщения в chat format.
            description: Описание для логов.

        Returns:
            Кортеж (сгенерированный текст, использует ли SID).
        """
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config.enable_thinking,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        completion = decoded[len(prompt):]

        logger.info(f"--- [{description}] ---")
        logger.info(f"Prompt: {prompt[-200:]}")
        logger.info(f"Output: {completion[:200]}")

        # Проверяем использование SID
        uses_sid = "<|sid_start|>" in completion or "<|sid_" in completion
        return completion, uses_sid

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        """Запустить eval на определённых шагах."""
        if state.global_step == 0:
            return
        if state.global_step % self.interval_steps != 0:
            return

        logger.info("=" * 80)
        logger.info(f"Generation eval at step {state.global_step}")
        logger.info("=" * 80)

        training_mode = model.training
        model.eval()

        sid_success = 0

        # Generic prompts
        for i, case in enumerate(self.generic_prompts, 1):
            messages = case.get("messages", case)
            try:
                self._run_generation(model, messages, f"GENERIC #{i}")
            except Exception as e:
                logger.warning(f"Generation failed for generic #{i}: {e}")

        # SID prompts
        for i, case in enumerate(self.sid_prompts, 1):
            messages = case.get("messages", case)
            try:
                _, uses_sid = self._run_generation(model, messages, f"SID #{i}")
                if uses_sid:
                    sid_success += 1
            except Exception as e:
                logger.warning(f"Generation failed for SID #{i}: {e}")

        # Логируем в wandb
        if self.sid_prompts:
            success_rate = sid_success / len(self.sid_prompts)
            _safe_wandb_log(
                {
                    "sid_eval/num_prompts": len(self.sid_prompts),
                    "sid_eval/success_count": sid_success,
                    "sid_eval/success_rate": success_rate,
                },
                step=state.global_step,
                use_wandb=self.config.use_wandb,
            )
            logger.info(f"SID eval: {sid_success}/{len(self.sid_prompts)} ({success_rate:.0%})")

        if training_mode:
            model.train()

        logger.info("=" * 80)


def create_callbacks(
    tokenizer: PreTrainedTokenizerBase,
    config: LoRAConfig,
    generic_prompts: Optional[list[dict]] = None,
    sid_prompts: Optional[list[dict]] = None,
) -> list[TrainerCallback]:
    """Создать callbacks для обучения.

    Args:
        tokenizer: Токенизатор.
        config: Конфигурация.
        generic_prompts: Generic тестовые промпты.
        sid_prompts: SID тестовые промпты.

    Returns:
        Список callbacks.
    """
    callbacks = []

    if generic_prompts or sid_prompts:
        callbacks.append(
            GenerationEvalCallback(
                tokenizer=tokenizer,
                config=config,
                generic_prompts=generic_prompts or [],
                sid_prompts=sid_prompts or [],
                interval_steps=config.eval_steps,
            )
        )

    return callbacks

