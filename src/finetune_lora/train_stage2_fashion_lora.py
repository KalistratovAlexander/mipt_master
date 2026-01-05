#!/usr/bin/env python3
"""
Stage 2: LoRA fine-tuning for semantic ID generation.

This script fine-tunes LoRA adapters on a model with extended vocabulary
from Stage 1, teaching it to generate semantic IDs in responses.

Usage:
    python -m mipt_master.src.finetune_lora.train_stage2_fashion_lora [OPTIONS]

Example:
    python -m mipt_master.src.finetune_lora.train_stage2_fashion_lora \\
        --stage1-checkpoint models/qwen3_fashion_vocab/final \\
        --category Amazon_Fashion \\
        --lora-r 16 \\
        --lr 2e-4 \\
        --epochs 2 \\
        --no-wandb
"""

from .cli import main

if __name__ == "__main__":
    main()
