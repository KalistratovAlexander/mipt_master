# Stage 2: Full Fine-tuning (Qwen3-1.7B)

Full-parameter fine-tuning of the vocab-expanded model from Stage 1.
All 1.72B parameters are trainable.

## Prerequisites

Stage 1 model at `stage1_model/` (output of `stage1_vocab_expansion`).

## Data

Same as Stage 1: 4.7M rows, 23 task types, 63K unique SIDs.

- `data/Pet_Supplies_conversations_train.parquet` — training data
- `data/Pet_Supplies_conversations_val.parquet` — validation data

## Optimizations

| Technique | Effect |
|-----------|--------|
| Unsloth | 2-3x speedup |
| Liger Kernel | +20% throughput, -60% memory |
| torch.compile (inductor) | +15-30% |
| adamw_8bit | Less memory for optimizer states |
| NEFTune (alpha=5) | Better generalization |
| bf16 + tf32 | Mixed precision |

## Run on vast.ai

Built and launched via the H1 experiment package (`../../experiments/h1_model_scale/pack_train.sh`). Stage 2 starts from the Stage 1 output automatically:

```bash
# on server, after Stage 1 finished:
bash stage2/run.sh
```

## Defaults

| Parameter | Value |
|-----------|-------|
| lr | 2e-5 |
| batch | 32 |
| grad_accum | 4 |
| eff_batch | 128 |
| epochs | 1 |
| warmup | 3% |
| max_seq_length | 512 |
| optimizer | adamw_8bit |
| scheduler | cosine |

## Output

```
output/final/
├── model files           # Fine-tuned model
├── tokenizer files       # Tokenizer (unchanged from Stage 1)
└── training_meta.json    # Config & metrics
```

## Resume if crashed

```bash
bash run.sh --resume
```

## GPU requirements & time estimates

| GPU | Time (all 4.7M, 1 epoch) |
|-----|--------------------------|
| H100 80GB | ~1-2 hours |
| A100 40GB | ~2-3 hours |
| RTX 4090 24GB | ~4-6 hours |

For quick test: `bash run.sh --max-train-samples 10000 --epochs 1`

To reduce training time, downsample copurchase tasks (59% of data):
`bash run.sh --max-train-samples 1200000`
