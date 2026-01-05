#!/usr/bin/env python3
"""
Stage 1: Vocabulary expansion and embedding initialization.

This script extends the model vocabulary with semantic ID tokens
and trains only the embeddings (input + output) for these new tokens.

Usage:
    python -m mipt_master.src.vocab_expansion.train_stage1_fashion [OPTIONS]

Example:
    python -m mipt_master.src.vocab_expansion.train_stage1_fashion \\
        --model-name unsloth/Qwen3-1.7B \\
        --category Amazon_Fashion \\
        --max-steps 1000 \\
        --lr 1e-3
"""

from .cli import main

if __name__ == "__main__":
    main()
