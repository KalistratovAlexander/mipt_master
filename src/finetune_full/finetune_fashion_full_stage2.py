#!/usr/bin/env python3
"""
Stage 2: Full fine-tuning for semantic ID generation.

This script fine-tunes the entire model (not LoRA) on conversations
with semantic IDs, using the Stage 1 checkpoint with extended vocabulary.

Usage:
    python -m mipt_master.src.finetune_full.finetune_fashion_full_stage2 [OPTIONS]

Example:
    python -m mipt_master.src.finetune_full.finetune_fashion_full_stage2 \\
        --stage1-checkpoint models/qwen3_fashion_vocab/final \\
        --category Amazon_Fashion \\
        --lr 1e-5 \\
        --epochs 1 \\
        --resume \\
        --no-wandb
"""

from .cli import main

if __name__ == "__main__":
    main()
