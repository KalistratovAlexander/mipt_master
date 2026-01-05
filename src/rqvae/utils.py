from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger("train-rqvae")


def set_seed(seed: int) -> None:
    """Сделать результаты более воспроизводимыми (насколько это возможно)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Checkpoint:
    path: Path
    epoch: int
    step: int
    best_loss: float
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Optional[Dict[str, Any]]
    scheduler_state_dict: Optional[Dict[str, Any]]
    config: Dict[str, Any]


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Checkpoint:
    raw = torch.load(path, map_location=map_location)
    return Checkpoint(
        path=path,
        epoch=int(raw.get("epoch", 0)),
        step=int(raw.get("step", 0)),
        best_loss=float(raw.get("best_loss", raw.get("val_loss", float("inf")))),
        model_state_dict=raw["model_state_dict"],
        optimizer_state_dict=raw.get("optimizer_state_dict"),
        scheduler_state_dict=raw.get("scheduler_state_dict"),
        config=raw.get("config", {}),
    )


