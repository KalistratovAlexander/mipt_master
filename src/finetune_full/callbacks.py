"""Callbacks для мониторинга full fine-tuning."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

from transformers import TrainerCallback

if TYPE_CHECKING:
    from transformers import TrainingArguments
    from transformers.trainer_callback import TrainerControl, TrainerState

    from .config import FullFineTuneConfig

logger = logging.getLogger("finetune-full")


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


class TrainingMonitorCallback(TrainerCallback):
    """Callback для мониторинга обучения.

    Отслеживает:
    - Loss и его улучшение относительно начального
    - Learning rate и gradient norm
    - Скорость обучения (samples/s, ms/step)
    - Eval loss

    Attributes:
        config: Конфигурация.
        initial_loss: Начальный loss для вычисления improvement.
        last_log_time: Время последнего лога.
        last_log_step: Шаг последнего лога.
        batch_start_time: Время начала текущего батча.
    """

    def __init__(self, config: FullFineTuneConfig) -> None:
        """Инициализация.

        Args:
            config: Конфигурация.
        """
        self.config = config
        self.initial_loss: Optional[float] = None
        self.last_log_time = time.time()
        self.last_log_step = 0
        self.batch_start_time: Optional[float] = None

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Записать время начала батча."""
        self.batch_start_time = time.time()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """Логировать метрики обучения."""
        if not logs or "loss" not in logs:
            return

        current_loss = logs["loss"]

        # Запоминаем начальный loss
        if self.initial_loss is None:
            self.initial_loss = current_loss

        # Вычисляем improvement
        improvement = 0.0
        if self.initial_loss and self.initial_loss > 0:
            improvement = (self.initial_loss - current_loss) / self.initial_loss * 100

        lr = logs.get("learning_rate", 0.0)
        grad_norm = logs.get("grad_norm", 0.0)
        epoch = logs.get("epoch", 0.0)

        # Вычисляем скорость
        now = time.time()
        batch_time_ms = (now - self.batch_start_time) * 1000 if self.batch_start_time else 0
        elapsed = now - self.last_log_time
        steps_done = state.global_step - self.last_log_step

        eff_bs = self.config.batch_size * self.config.gradient_accumulation_steps
        samples_processed = steps_done * eff_bs
        sps = samples_processed / elapsed if elapsed > 0 else 0

        self.last_log_time = now
        self.last_log_step = state.global_step

        # Логируем в консоль
        log_str = (
            f"Step {state.global_step:05d} | Epoch {epoch:.2f} | lr {lr:.2e} | "
            f"loss {current_loss:.4f} | grad_norm {grad_norm:.2f} | "
            f"improv {improvement:+.1f}% | {batch_time_ms:.0f}ms/step | {sps:,.0f} samples/s"
        )
        logger.info(log_str)

        # Логируем в wandb
        _safe_wandb_log(
            {
                "loss/train": current_loss,
                "metrics/learning_rate": lr,
                "metrics/gradient_norm": grad_norm,
                "metrics/improvement_pct": improvement,
                "metrics/batch_time_ms": batch_time_ms,
                "metrics/samples_per_second": sps,
            },
            step=state.global_step,
            use_wandb=self.config.use_wandb,
        )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """Логировать eval метрики."""
        if not metrics:
            return

        eval_loss = metrics.get("eval_loss", 0.0)
        logger.info(f"EVAL | Step {state.global_step:05d} | eval_loss {eval_loss:.4f}")

        _safe_wandb_log(
            {"loss/eval": eval_loss},
            step=state.global_step,
            use_wandb=self.config.use_wandb,
        )


def create_callbacks(config: FullFineTuneConfig) -> list[TrainerCallback]:
    """Создать callbacks для обучения.

    Args:
        config: Конфигурация.

    Returns:
        Список callbacks.
    """
    return [TrainingMonitorCallback(config)]

