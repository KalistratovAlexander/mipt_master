# Qwen3-1.7B Semantic ID Training — vast.ai Deployment

> **Примечание об имени каталога.** Каталог `fine_tune_1.8B/` и файлы `train_1.8b.py` сохраняют историческое имя по следующей причине: при первоначальном создании инфраструктуры предполагалось существование модели `Qwen3-1.8B`, однако [Qwen3 Technical Report (arXiv 2505.09388)](https://arxiv.org/abs/2505.09388) и [официальная карточка Qwen3-1.7B на Hugging Face](https://huggingface.co/Qwen/Qwen3-1.7B) подтверждают, что в dense-семействе Qwen3 нет модели 1.8B (доступны 0.6B, 1.7B, 4B, 8B, 14B, 32B). **Целевая модель — Qwen3-1.7B** (tied embeddings, hidden dim = 2048). Переименование каталога, скриптов, путей к артефактам и ссылок из `thesis/` и `research/` отложено до post-defense, чтобы не ломать воспроизводимые training-прогоны, attached артефакты и внешние ссылки. См. `thesis/chapter_2/2_llm_adaptation.md`, §2.2.1 — описание исходного состояния backbone и политики tied/untied.

Two-stage pipeline for training Qwen3-1.7B with semantic IDs for product recommendations.

## Hardware Requirements

| | Minimum | Recommended |
|--|---------|-------------|
| **GPU** | RTX 4090 24GB | H100 80GB |
| **RAM** | 32 GB | 64 GB |
| **Disk** | 25 GB | 50 GB |
| **CUDA** | >= 12.1 | >= 12.4 |
| **Docker image** | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` | same |

### How to find on vast.ai

1. Go to https://cloud.vast.ai/create/
2. Filter: GPU RAM >= 24 GB, Disk >= 50 GB, CUDA >= 12.1
3. Sort by $/hr
4. Recommended GPUs (price/performance):
   - **H100 80GB** — fastest (~$2-3/hr, both stages in ~3-5 hours)
   - **A100 40GB** — good balance (~$1-2/hr, ~5-7 hours)
   - **RTX 4090 24GB** — budget (~$0.3-0.5/hr, ~10-15 hours)

## Time Estimates

| Stage | H100 | A100 40GB | RTX 4090 |
|-------|------|-----------|----------|
| Stage 1 (vocab expansion) | ~30 min | ~45 min | ~1-2 hr |
| Stage 2 (full fine-tuning) | ~2-4 hr | ~4-6 hr | ~8-12 hr |
| **Total** | **~3-5 hr** | **~5-7 hr** | **~10-14 hr** |

## Deployment

### 1. Pack locally

```bash
cd /path/to/mipt_master
bash vast/pack.sh
# Creates: vast_training_package.tar.gz (~300 MB)
```

### 2. Upload to vast.ai

```bash
scp -P <PORT> vast_training_package.tar.gz root@<HOST>:/workspace/
```

### 3. Unpack on server

```bash
cd /workspace
tar xf vast_training_package.tar.gz
```

This creates:
```
/workspace/
├── data/
│   ├── Pet_Supplies_conversations_train.parquet   (286 MB, 4.7M rows)
│   └── Pet_Supplies_conversations_val.parquet     (15 MB, 248K rows)
├── setup.sh              # shared dependency installation (called by both stages)
├── stage1/
│   ├── train_1.8b.py
│   └── run.sh
├── stage2/
│   ├── train_1.8b.py
│   └── run.sh
└── README.md
```

### 4. Run Stage 1 (vocab expansion, ~30-60 min)

```bash
cd /workspace/stage1
nohup bash run.sh 2>&1 &
tail -f train.log
```

Trains only SID token embeddings (0.3% of params). Output: `output/final/`

### 5. Run Stage 2 (full fine-tuning, ~2-6 hr)

```bash
cd /workspace/stage2
nohup bash run.sh 2>&1 &
tail -f train.log
```

Uses Stage 1 model automatically. Output: `output/final/`

### 6. Download results

```bash
# From local machine:
scp -P <PORT> -r "root@<HOST>:/workspace/stage2/output/final" ./model_1.8b_final/
```

## If something crashes

Both stages support `--resume`:

```bash
cd /workspace/stage1  # or stage2
bash run.sh --resume
```

## Quick test (dry run)

To verify everything works before full training:

```bash
cd /workspace/stage1
bash run.sh --max-train-samples 1000 --max-steps 50

cd /workspace/stage2
bash run.sh --max-train-samples 1000 --epochs 1
```

## Cost Estimate

| GPU | $/hr | Total time | Total cost |
|-----|------|-----------|------------|
| H100 80GB | ~$2.50 | ~4 hr | **~$10** |
| A100 40GB | ~$1.50 | ~6 hr | **~$9** |
| RTX 4090 | ~$0.40 | ~12 hr | **~$5** |
