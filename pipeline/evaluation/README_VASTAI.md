# Unified Evaluation — Vast.ai Quick Start

## Server Requirements

| Parameter | 1.8B only | 8B only | Both models |
|-----------|-----------|---------|-------------|
| **GPU VRAM** | >= 8 GB | >= 24 GB | >= 24 GB |
| **Recommended GPU** | RTX 3090 / RTX 4090 | A100 40GB / A6000 48GB | **A100 40GB+** |
| **System RAM** | >= 16 GB | >= 32 GB | >= 32 GB |
| **Disk** | >= 25 GB | >= 35 GB | >= 45 GB |
| **CUDA** | >= 12.1 | >= 12.1 | >= 12.1 |
| **Docker image** | pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel | same | same |

**Optimal setup**: A100 40GB / A100 80GB with `flash_attention_2` — fastest inference.

**Budget setup**: RTX 4090 24GB (8B fits in bfloat16; beam search may need `--beam-size 5`).

## Upload

```bash
# Upload archive to vast.ai instance (19 GB, ~15 min on 100 Mbps)
scp -P <PORT> eval_vastai_package.tar root@<HOST>:/workspace/
```

## Unpack on server

```bash
cd /workspace
tar xf eval_vastai_package.tar
```

This creates:
```
/workspace/
├── pipeline/evaluation/
│   ├── evaluate_unified.py          # Main script (~1800 lines)
│   └── data/                        # 3 parquet files (167 MB)
├── vast/stage2_.../final/           # 1.8B model (3.2 GB)
└── vast_8b/stage2_.../stage2_final/ # 8B model (15 GB)
```

## Install dependencies

```bash
pip install polars tqdm datasets sentence-transformers numpy
# torch + transformers should already be in vast.ai image
# Verify transformers version supports Qwen3:
python -c "import transformers; print(transformers.__version__)"
# Need >= 4.51.0 for Qwen3ForCausalLM
```

## Step 0: Dry run (sanity check, ~2 min)

```bash
python pipeline/evaluation/evaluate_unified.py \
  --model-path vast/stage2_full_finetune/output/stage2_h100/final \
  --data-dir pipeline/evaluation/data \
  --model-name "1.8B-dryrun" \
  --output results/dryrun_1.8b.json \
  --dry-run
```

If this completes without errors, proceed to full evaluation.

## Step 1: Evaluate 1.8B (~45-90 min on H100)

1000 samples per task: 8 SID tasks + 3 text tasks = 11000 total.

```bash
python pipeline/evaluation/evaluate_unified.py \
  --model-path vast/stage2_full_finetune/output/stage2_h100/final \
  --data-dir pipeline/evaluation/data \
  --model-name "1.8B" \
  --beam-size 10 \
  --attn-impl flash_attention_2 \
  --output results/eval_unified_1.8b.json
```

## Step 2: Evaluate 8B (~3-5 hours on H100)

```bash
python pipeline/evaluation/evaluate_unified.py \
  --model-path vast_8b/stage2_full_finetune/result/stage2_final \
  --data-dir pipeline/evaluation/data \
  --model-name "8B" \
  --beam-size 10 \
  --attn-impl flash_attention_2 \
  --output results/eval_unified_8b.json
```

## Flags

| Flag | Effect |
|------|--------|
| `--dry-run` | 2 samples/task, beam=3, no perplexity/cosine — sanity check |
| `--resume` | Skip already completed tasks (reads existing output JSON) |
| `--skip-perplexity` | Skip WikiText-2 perplexity (saves ~2 min) |
| `--skip-cosine-sim` | Skip sentence-transformers cosine similarity |
| `--attn-impl flash_attention_2` | Flash Attention 2 (A100/H100, ~1.5x faster) |
| `--attn-impl sdpa` | Default PyTorch SDPA (works everywhere) |
| `--samples-per-task 50` | Quick run with fewer samples |
| `--beam-size 5` | Smaller beam if OOM on 8B |
| `--skip-benchmark` | Skip performance benchmarking (TTFT, TPS, E2E, GPU memory) |
| `--bench-iters 20` | Number of iterations for performance benchmark (default: 20) |

## If evaluation crashes mid-way

Results are saved after each task. Just re-run with `--resume`:

```bash
python pipeline/evaluation/evaluate_unified.py \
  --model-path vast_8b/stage2_full_finetune/result/stage2_final \
  --data-dir pipeline/evaluation/data \
  --model-name "8B" \
  --beam-size 10 \
  --attn-impl flash_attention_2 \
  --output results/eval_unified_8b.json \
  --resume
```

## Output

- `results/eval_unified_1.8b.json` — full metrics JSON
- `results/eval_unified_1.8b_examples.json` — qualitative examples
- Same for 8B

## Download results

```bash
scp -P <PORT> "root@<HOST>:/workspace/results/eval_unified_*.json" .
```

## Time estimates

| Model | GPU | `--samples-per-task` | Estimated time |
|-------|-----|---------------------|----------------|
| 1.8B | A100 40GB | 200 | ~30-60 min |
| 1.8B | RTX 4090 | 200 | ~45-90 min |
| 8B | A100 40GB | 200 | ~2-4 hours |
| 8B | A100 80GB | 200 | ~1.5-3 hours |
| 8B | RTX 4090 24GB | 200 | ~4-6 hours |

## Troubleshooting

**OOM on beam search (8B)**: Use `--beam-size 5` or switch to A100 80GB.

**`Qwen3ForCausalLM` not found**: Upgrade transformers: `pip install -U transformers>=4.51.0`.

**Slow generation**: Check `use_cache` is enabled (script sets it automatically). Use `--attn-impl flash_attention_2` on Ampere+ GPUs.

**`datasets` import error for WikiText-2**: `pip install datasets` or use `--skip-perplexity`.
