# Qwen3-8B Semantic ID Training — vast.ai Deployment

Two-stage pipeline for training Qwen3-8B with semantic IDs for product recommendations.
Uses Unsloth + vanilla Trainer — identical methodology to 1.8B for experimental consistency.

## Hardware Requirements

| | Required |
|--|---------
| **GPU** | H100 80GB (or A100 80GB) |
| **RAM** | 64 GB |
| **Disk** | 150 GB |
| **CUDA** | >= 12.1 |
| **Docker image** | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` |

## Time Estimates (H100 80GB)

| Stage | Time |
|-------|------|
| Stage 1 (vocab expansion) | ~60-80 min |
| Stage 2 (full fine-tuning) | ~10.5 hr |
| **Total** | **~12 hr** |

## Deployment

### 1. Pack locally

```bash
cd mipt_master
bash fine_tune_8B/pack.sh
# Creates: vast_8b_package.tar.gz (~300 MB)
```

### 2. Upload & unpack

```bash
scp -P <PORT> vast_8b_package.tar.gz root@<HOST>:/workspace/
ssh -p <PORT> root@<HOST>
cd /workspace && tar xf vast_8b_package.tar.gz
```

This creates:
```
/workspace/
├── data/
│   ├── Pet_Supplies_conversations_train.parquet
│   └── Pet_Supplies_conversations_val.parquet
├── setup.sh
├── stage1/
│   ├── train_8b.py
│   └── run.sh
└── stage2/
    ├── train_8b.py
    └── run.sh
```

### 3. Run Stage 1 (vocab expansion, ~1 hr)

```bash
cd /workspace/stage1
nohup bash run.sh 2>&1 &
tail -f train.log
```

### 4. Run Stage 2 (full fine-tuning, ~10 hr)

```bash
cd /workspace/stage2
nohup bash run.sh 2>&1 &
tail -f train.log
```

### 5. Download results

```bash
scp -P <PORT> -r "root@<HOST>:/workspace/stage2/output/final" ./model_8b_final/
```

## Hyperparameters (aligned with 1.8B)

| Parameter | Value | Notes |
|-----------|-------|-------|
| lr | 2e-5 → 4e-6 | cosine_with_min_lr (min=0.2×peak) |
| batch | 16, grad_accum=8 | Effective batch 128 |
| epochs | 3 | |
| warmup | 3% | |
| weight_decay | 0.01 | OpenOneRec Table 14 |
| optimizer | adamw_8bit | Memory efficient |
| packing | yes | ~3x throughput |
| gradient_checkpointing | yes | Required for 8B |

## 8B vs 1.8B differences

| | 1.8B | 8B |
|--|------|-----|
| Embeddings | Tied | Untied (separate input/output) |
| Stage 2 batch | 64 | 16 |
| Stage 2 grad_accum | 2 | 8 |
| gradient_checkpointing | No | Yes (required for 8B) |

All other settings (LR, warmup, scheduler, optimizer, packing, callbacks) are identical.

## Cost Estimate

| GPU | $/hr | Total time | Total cost |
|-----|------|-----------|------------|
| H100 80GB | ~$2.50 | ~12 hr | **~$30** |
| A100 80GB | ~$1.50 | ~18 hr | **~$27** |
