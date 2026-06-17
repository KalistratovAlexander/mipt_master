# H1 — Эффект масштаба базовой модели

Гипотеза: качество генеративных рекомендаций на семантических ID зависит от размера базовой модели, и эффект задачно-условный. Сравниваются четыре размера Qwen3 (0.6B / 1.7B / 4B / 8B) при идентичном двухстадийном протоколе обучения.

Обучение использует **общий тренер** из `../../fine_tune/` (stage 1 — расширение словаря, stage 2 — полное дообучение), а оценку — **общий оценщик** `../../evaluation/evaluate_unified.py`. Здесь лежит только то, что специфично для H1: конфиги размеров, упаковщики и анализ.

## Структура

```
h1_model_scale/
├── stage1/run_<size>.sh    конфиги стадии 1 для каждого размера (0.6b/1.8b/4b/8b)
├── stage2/run_<size>.sh    конфиги стадии 2 для каждого размера
├── pack_train.sh           сборка пакета обучения (тренер + конфиг размера + данные) → vast_<size>_package.tar.gz
├── pack_eval.sh            сборка пакета оценки (оценщик + данные) → vast_eval_package.tar.gz
├── run_h1.sh               оркестрация оценки всех 4 размеров + парные стат-тесты
├── stat_tests.py           попарные контрасты соседних размеров (парный бутстреп, Бонферрони)
└── README.md
```

Имя файла `train_1.8b.py` в `../../fine_tune/` историческое (целевая модель Qwen3-1.7B); см. `../../fine_tune/README.md`.

## Запуск (vast.ai)

Обучение одного размера:
```bash
bash pipeline/experiments/h1_model_scale/pack_train.sh 8b   # 0.6b | 1.8b | 4b | 8b
scp vast_8b_package.tar.gz root@<HOST>:/workspace/
# на сервере:
cd /workspace && tar xf vast_8b_package.tar.gz && export HF_TOKEN=hf_...
bash run_smoke.sh && bash stage1/run.sh && bash stage2/run.sh
```

Оценка всех размеров:
```bash
bash pipeline/experiments/h1_model_scale/pack_eval.sh
scp vast_eval_package.tar.gz root@<HOST>:/workspace/
# на сервере:
cd /workspace && tar xf vast_eval_package.tar.gz && export HF_TOKEN=hf_...
bash pipeline/evaluation/run_h1.sh
```
