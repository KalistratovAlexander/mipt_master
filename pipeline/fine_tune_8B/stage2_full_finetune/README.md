# Stage 2: Full Fine-tuning — Qwen3-8B

Full-parameter fine-tuning of the vocab-expanded model from Stage 1.
All 8.3B parameters are trainable.

## Files

| File | Description |
|---|---|
| `train_8b.py` | Training script (transformers Trainer) |
| `run_8b.sh` | Training launcher for vast.ai |

Evaluation — in `pipeline/evaluation/`.

## Defaults

| Parameter | Value | Notes |
|-----------|-------|-------|
| lr | 3e-5 → 6e-6 | Cosine with min_lr |
| batch | 16, grad_accum=8 | Effective batch 128 |
| epochs | 3 | |
| warmup | 10% | |
| weight_decay | 0.01 | OpenOneRec Table 14 |
| optimizer | adamw_8bit | Memory efficient |
| packing | yes | ~3x throughput |
| gradient_checkpointing | yes | Required for 8B |

## Optimizations

| Technique | Effect |
|-----------|--------|
| Sequence packing | ~3x throughput (avg 150 tok → 512 chunks) |
| Flash Attention 2 | ~2x speedup |
| gradient_checkpointing | -80% activation memory |
| adamw_8bit | Less memory for optimizer states |
| Instruction masking | Loss only on assistant responses |

## Evaluation metrics

- **SID accuracy:** Valid format, A/AB/ABC/Exact match (95% CI)
- **Beam search:** Hit@1, Pass@10, MRR, NDCG@10
- **Text generation:** ROUGE-L, Token F1
- **System:** WikiText-2 Perplexity, Hallucination Rate

## First run results

| Task Group | A-level | Exact | ROUGE-L |
|---|---|---|---|
| Text→SID | **65.2%** | 1.0% | — |
| Partial-text→SID | 38.2% | 0.2% | — |
| Sequential | 9.7% | 1.0% | — |
| Copurchase | 4.5% | 0.2% | — |
| SID→Text | — | — | **0.201** |

- WikiText-2 Perplexity: 223.53
- Hallucination Rate: 5.9%

## Time estimate

~10.5 hours on H100 80GB (3 epochs, 500K samples, packing)
