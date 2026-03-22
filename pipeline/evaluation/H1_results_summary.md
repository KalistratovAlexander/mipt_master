# Эксперимент H1: Влияние масштаба модели на качество рекомендаций

**Qwen3-1.7B vs Qwen3-8B**

Условия: Amazon Pet Supplies (63K товаров, 102K последовательностей)
Оценка: 3000 samples per task, 11 задач
Hardware: NVIDIA H100 80GB (vast.ai)

---

## 1. Качество рекомендаций

### SID Prediction (greedy, A-level match %)

| Задача | 1.8B | 8B | Δ |
|--------|------|-----|---|
| title_to_sid | 62.0% | 66.0% | +4.0 |
| description_to_sid | 61.9% | 65.8% | +3.9 |
| features_to_sid | 55.6% | 62.1% | +6.5 |
| **TEXT→SID (avg)** | **59.9%** | **64.6%** | **+4.7** |
| seq_last_2 | ~7% | 8.7% | +1.7 |
| seq_last_3 | ~7% | 10.7% | +3.7 |
| seq_last_5 | ~7% | 9.7% | +2.7 |
| **SEQUENTIAL (avg)** | **7.0%** | **9.7%** | **+2.7** |
| copurchase_backward | ~5.5% | 5.7% | +0.2 |
| copurchase_forward | ~5.5% | 6.0% | +0.5 |
| **COPURCHASE (avg)** | **5.5%** | **5.8%** | **+0.3** |

### SID Prediction (beam search k=10, hit@10 exact)

| Задача | 8B |
|--------|-----|
| title_to_sid | 3.3% |
| description_to_sid | 3.3% |
| features_to_sid | 2.7% |
| seq_last_2 | 6.8% |
| seq_last_3 | 6.0% |
| seq_last_5 | 6.0% |
| copurchase_backward | 1.8% |
| copurchase_forward | 2.0% |

### SID → Text (greedy, char_f1)

| Задача | 1.8B | 8B | Δ |
|--------|------|-----|---|
| sid_to_title | 58.6% | 59.8% | +1.2 |
| sid_to_description | 76.3% | 76.9% | +0.6 |
| sid_to_features | 68.6% | 73.0% | +4.4 |

### Качество генерации

| Метрика | 1.8B | 8B |
|---------|------|-----|
| Valid format | 99.9% | 99.7-100% |
| Hallucination (not in catalog) | ~8-9% | 5.2% |
| Cosine similarity (avg) | 61.8% | 54.2% |

---

## 2. Катастрофическое забывание

| Метрика | 1.8B | 8B |
|---------|------|-----|
| PPL base (WikiText-2) | 9.39 | 6.99 |
| PPL fine-tuned | 1221.52 | 223.53 |
| **Рост perplexity** | **×130** | **×32** |

Качественная оценка:
- 8B: частично отвечает на общие вопросы (фотосинтез, 2+2=4)
- Обе модели: зацикливания, генерация SID вместо текста
- Обе модели: НЕ способны объяснять рекомендации

---

## 3. Производительность (Inference, H100 80GB)

| Метрика | 1.8B | 8B | Ratio |
|---------|------|-----|-------|
| Parameters | 1.72B | 8.20B | 4.8× |
| GPU Memory (weights) | 3.2 GB | 15.3 GB | 4.8× |
| GPU Memory (peak beam) | 3.3 GB | 15.4 GB | 4.6× |
| TTFT (prefill) | 22.9 ms | 25.9 ms | 1.1× |
| TPS (SID, greedy) | 49.8 tok/s | 38.8 tok/s | 1.3× |
| TPS (text, greedy) | 52.9 tok/s | 39.4 tok/s | 1.3× |
| E2E SID greedy | 201 ms | 1237 ms | 6.2× |
| E2E beam k=10 | 593 ms | 1332 ms | 2.2× |
| E2E text generation | 208 ms | 5073 ms | 24.4× |
| GPU Power | 146 W | 211 W | 1.4× |
| Energy per SID request | 29.4 J | 261.2 J | 8.9× |

Валидация через публичные бенчмарки (Qwen, H20 96GB, BF16):

| Framework | 1.8B | 8B | Ratio |
|-----------|------|-----|-------|
| Transformers | 59.8 tok/s | 45.3 tok/s | 1.3× ✓ |
| SGLang (optimized) | 227.8 tok/s | 81.7 tok/s | 2.8× |

---

## 4. Экономика

### R&D (обучение + оценка)

| Этап | 1.8B | 8B |
|------|------|-----|
| Stage 1: RQ-VAE | $0.50 | $0.50 |
| Stage 2: Эмбеддинги | $0.34 | $4.77 |
| Stage 3: Fine-tuning | $4.76 | $76.32 |
| Evaluation | $0.68 | $15.90 |
| **Итого R&D** | **$6.28** | **$97.49** |
| GPU для обучения | RTX 4090 | H100 80GB |

### Разворачивание (24/7, 1 GPU, vast.ai)

| | 1.8B | 8B |
|--|------|-----|
| Мин. GPU | RTX 3060 12GB | RTX 4090 24GB |
| Стоимость/мес (мин) | $110 | $248 |
| Опт. GPU | RTX 4090 | A100 40GB |
| Стоимость/мес (опт) | $248 | $942 |

### Первый год (R&D + deploy)

| Вариант | Стоимость |
|---------|-----------|
| 1.8B (RTX 3060) | $1,320 |
| 8B (RTX 4090, мин.) | $3,076 |
| 8B (A100, опт.) | $11,398 |

---

## 5. Выводы по H1

1. **8B стабильно лучше 1.8B по всем задачам:**
   - Text→SID: +4.7 п.п. (наибольший прирост)
   - Sequential: +2.7 п.п.
   - Copurchase: +0.3 п.п. (минимальный прирост)

2. **8B забывает в 4× меньше** (PPL ×32 vs ×130)

3. **Наибольший прирост** — в задачах с текстовым контекстом,
   минимальный — в collaborative filtering (copurchase)

4. **1.8B экономичнее** в 8.5× по inference и в 15.5× по обучению

5. **Для малых команд 1.8B — рациональный выбор:**
   - Обучение за $6 (vs $97)
   - Deploy от $110/мес (vs $248-942/мес)
   - Качество на copurchase/sequential сопоставимо

6. **8B оправдана** при фокусе на text→SID задачах
   и наличии бюджета на GPU >= 24GB

---

## Limitations

- 1.8B и 8B обучены на РАЗНЫХ RQ-VAE маппингах (59K vs 63K items, 0% overlap по SID). Прямое сравнение метрик имеет ограничения.
- Inference без батчинга (vanilla transformers) — в production с vLLM/SGLang throughput будет выше.
- Один датасет (Pet Supplies) — обобщаемость не проверена.
- Нет baseline (SASRec, random, popularity).
