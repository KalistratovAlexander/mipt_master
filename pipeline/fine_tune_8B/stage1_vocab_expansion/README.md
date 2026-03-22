# Stage 1: Vocabulary Expansion — Qwen3-8B

Adds 1027 SID tokens to Qwen3-8B vocabulary and trains **only** their embeddings.
Qwen3-8B has untied embeddings (input ≠ output), so both matrices are trained independently.

## Tokens added

- `<|rec|>`, `<|sid_start|>`, `<|sid_end|>` — 3 special tokens
- `<|A0|>`..`<|A255|>`, `<|B0|>`..`<|B255|>`, `<|C0|>`..`<|C255|>`, `<|D0|>`..`<|D255|>` — 1024 SID tokens

## Defaults

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | Qwen/Qwen3-8B | Downloaded from HuggingFace |
| lr | 1e-3 | High for embedding-only |
| batch | 16, grad_accum=4 | Effective batch 64 |
| max_steps | 2000 | ~2 epochs over 64K samples |
| weight_decay | 0.0 | L2 kills new tokens before they learn |
| gradient_checkpointing | True | Required: gradient flows through all 32 layers |
| optimizer | adamw_torch_fused | CUDA-fused, fast |
| torch.compile | yes | ~2x after JIT warmup |

## Memory (H100 80GB)

```
Model (bf16, frozen):       16 GB
Optimizer (embeddings):     ~10 GB
Activations (checkpointed): ~5 GB
Gradients (embeddings):     ~2.5 GB
torch.compile cache:        ~2 GB
──────────────────────────────────
Peak:                       ~35 GB / 80 GB
```

## Output

```
output/stage1_8b/final/
├── model files           # Full model with expanded vocab
├── tokenizer files       # Tokenizer with 1027 new tokens
└── sid_embeddings.npy    # SID embeddings (1027 × 4096)
```

## GPU requirements

| GPU | Time estimate |
|-----|---------------|
| H100 80GB | ~60-80 min |
| A100 80GB | ~90-120 min |
