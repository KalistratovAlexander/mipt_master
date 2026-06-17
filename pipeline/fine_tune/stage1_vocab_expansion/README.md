# Stage 1: Vocabulary Expansion (Qwen3-1.7B)

Adds 1027 SID tokens to the Qwen3-1.7B vocabulary and trains **only** their embeddings.
All other parameters are frozen (~0.3% trainable).

## Tokens added

- `<|rec|>`, `<|sid_start|>`, `<|sid_end|>` — 3 special tokens
- `<|A0|>`..`<|A255|>`, `<|B0|>`..`<|B255|>`, `<|C0|>`..`<|C255|>`, `<|D0|>`..`<|D255|>` — 1024 SID tokens

## Data

- `data/Pet_Supplies_conversations_train.parquet` — 4.7M rows, 23 task types, 63K unique SIDs
- `data/Pet_Supplies_conversations_val.parquet` — 248K rows

## Run on vast.ai

Per-size configs live in `../../experiments/h1_model_scale/stage1/run_<size>.sh`; the deployable package is built by `pack_train.sh` (it renames the chosen config to `stage1/run.sh`):

```bash
bash pipeline/experiments/h1_model_scale/pack_train.sh 1.8b   # 0.6b | 1.8b | 4b | 8b
# on server:  tar xf vast_<size>_package.tar.gz && bash run_smoke.sh && bash stage1/run.sh
```

## Defaults

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | Qwen/Qwen3-1.7B | Downloaded from HuggingFace |
| lr | 1e-3 | High for embedding-only |
| batch | 64 | Fits on 24GB+ GPU |
| max_steps | 2000 | ~2 epochs over 64K samples |
| weight_decay | 0.0 | L2 kills new tokens before they learn |
| optimizer | adamw_torch_fused | CUDA-fused, fast |
| torch.compile | off | run scripts pass `--no-torch-compile` |

## Output

```
output/stage1_1.8b/final/
├── model files          # Full model with expanded vocab
├── tokenizer files      # Tokenizer with 1027 new tokens
├── sid_embeddings.npy   # New SID embeddings as numpy array
└── training_meta.json   # Training config & results
```

## Next step

Use `output/stage1_1.8b/final/` as `--stage1-model` for Stage 2 (full fine-tuning):

```bash
cd /workspace/stage2
python train_1.8b.py \
    --stage1-model /workspace/stage1/output/stage1_1.8b/final \
    --train-file data/Pet_Supplies_conversations_train.parquet \
    --val-file data/Pet_Supplies_conversations_val.parquet
```

## Resume if crashed

```bash
bash run.sh --resume
```

## GPU requirements

| GPU | batch_size | Time estimate |
|-----|-----------|---------------|
| H100 80GB | 64 | ~30 min |
| A100 40GB | 64 | ~45 min |
| RTX 4090 24GB | 32 | ~1-2 hours |
| RTX 3090 24GB | 32 | ~2-3 hours |

For smaller GPUs, reduce batch: `bash run.sh --batch-size 16 --grad-accum 4`
