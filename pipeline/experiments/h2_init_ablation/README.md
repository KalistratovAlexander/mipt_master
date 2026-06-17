# H2 — 4-way ablation init-сигнала для новых SID-токенов

**Backbone:** Qwen3-0.6B (tied embeddings, hidden_dim=1024).
**Датасет:** Amazon Pet Supplies (SID-каталог переиспользуется из `pipeline/fine_tune/`).
**Гипотеза:** способ инициализации эмбеддингов добавленных токенов влияет на качество (сравнение 4 способов при равной норме Фробениуса).

## Arms

- **A.** `N(μ_E, Σ_E)` — полная эмпирическая ковариация по существующим rows E. SOTA-дефолт per OpenOneRec §4.2.
- **B.** `N(0, σ²·I)`, `σ² = mean(diag(Σ_E))` — variance-matched random. Контроль: та же Frobenius-шкала, ноль semantic.
- **C.** `v_i = mean({E[tok] : tok ∈ tokenize(title_i)})` — text-derived, SA-Init/WECHSEL/GTI lineage.
- **D.** `v_i = C_i · P`, `P ∈ ℝ^{32×h}` orthonormal rows — codebook-projected через Johnson-Lindenstrauss.
  Наш RQ-VAE учит **L=3** выученных кодбука (A/B/C по 256×32 ← `models/rqvae/best_model.pth`).
  4-й уровень SID (D) — коллизионный ordinal-счётчик (TIGER-style, см. `pipeline/prepare_semantic_ids.ipynb`),
  у него нет выученного кода. `arm_D` проецирует 768 codebook-строк в `ℝ^h`, а оставшиеся 256 SID-слотов D-уровня
  инициализирует arm_A fallback (deterministic seed). Codebook-тензор (768×32) готовит `precompute_all.py` (шаг `codebook`).

Все 4 arms **rescaled** к единой pre-registered Frobenius-норме (`artifacts/h2_init_scales.json`) для устранения scale confound.

**3 control-токена** (`<|rec|>`, `<|sid_start|>`, `<|sid_end|>`) во всех arms инициализируются как arm A — изолируем вопрос «init-стратегия» от «control tokens».

## Статистический протокол

- 4 arms × 3 seeds = **12 training runs** (Stage 1 + Stage 2 на 0.6B).
- **Primary:** Recall@10 на text→SID (max SNR per power analysis).
- **Test:** 3 парных бутстреп-контраста (A−C, A−D, C−D) с Bonferroni m=3 (α=0.0167).
- Arm B — descriptive-only контроль.
- **Pre-registration:** `artifacts/h2_init_scales.json` коммитится ДО первого запуска.

## Структура

```
h2_init_ablation/
├── README.md                   (этот файл)
├── init_strategies.py          (4 arms + scale normalization)
├── precompute_all.py           (→ artifacts/*: scales, codebook, title_map)
├── evaluate_recall_at_10.py    (primary: title_to_sid × 1000, per-sample hit@10)
├── transversal_diagnostics.py  (post-hoc: cos + CKA + eff rank + RSA)
├── aggregate_stats.py          (paired bootstrap m=3 + descriptive tables)
├── run.sh                   (entry: bash run.sh <ARM> <SEED>; DRY_RUN=1 для smoke)
├── run_all.sh                  (12-run orchestrator + transversal + aggregate; skip-if-done)
├── pack.sh                  (build tar.gz для vast.ai)
├── artifacts/                  (pre-registered constants + arm-specific inputs)
├── runs/                       (per-arm×seed outputs; gitignored)
└── results/                    (h2_summary.json, transversal.json; gitignored)
```

Shared `evaluate_unified.py` из `../../evaluation/` — вызывается внутри `run.sh`
для descriptive 11-task eval + WikiText-2 PPL.
Pull с vast.ai: `bash ../../fetch_results.sh <HOST> <PORT>`.
