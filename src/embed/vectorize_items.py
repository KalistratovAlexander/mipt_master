#!/usr/bin/env python3
"""
Generate embeddings for product items using Qwen3-Embedding model.

This is a thin wrapper that calls the CLI.
Run with: python -m mipt_master.src.embed.vectorize_items

For full CLI options: python -m mipt_master.src.embed.cli --help
"""

from .cli import main

if __name__ == "__main__":
    main()
