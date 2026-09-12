# Qwen3.5-4B-Base → combined report H1 · H3 · H5 · H11

Цель: тот же отчёт, что [niczavadskiy.github.io/occupational-gender-bias](https://niczavadskiy.github.io/occupational-gender-bias/), на **`Qwen/Qwen3.5-4B-Base`**.

Датасет: 951×3×2×{2|6} = **45648** (в репо: `data/v1/`).

## 1. Vast / Jupyter (без SSH) — основной путь

1. Клонируйте репо на инстанс (Jupyter Terminal или ячейка):
   `git clone https://github.com/niczavadskiy/occupational-gender-bias.git /workspace/occupational-gender-bias`
2. Откройте [`notebooks/vast_qwen35_4b_h1h3.ipynb`](../../notebooks/vast_qwen35_4b_h1h3.ipynb)
3. Вставьте `HF_TOKEN`, выполните ячейки по порядку (smoke → 6 шардов → merge → pack)

Нужен только **HF_TOKEN** (скачать модель). GitHub-токен не нужен для публичного клона.

Опционально (SSH / shell): `scripts/vast_qwen35_4b_h1h3_and_pack.sh`

Время: порядка нескольких часов на 3090 (6×~15–25 мин + merge).

## 3. После скачивания — метрики и H11

Положить раны в `experiments/qwen35-4b-base/results/`:

```
experiments/qwen35-4b-base/results/
  v1_full_pos_shuffle/     # или полный run_*_v1_full_pos_shuffle
    per_item.jsonl
    hidden_states.npz
  highlight-h3-full/
    per_item.jsonl
    hidden_states.npz
```

Из корня `occupational-gender-bias` (или `repo`, если метрики/probes там):

```powershell
# H1
python -m src.metrics.h1_v1 --run experiments/qwen35-4b-base/results/v1_full_pos_shuffle

# H3 (+ данные для H5)
python -m src.metrics.h3_v1 --run experiments/qwen35-4b-base/results/highlight-h3-full

# H11 — gender probe на no_evidence HS (из repo/)
cd ../repo
python -m probes.v1_rep_run --run <path-to-v1_full_pos_shuffle> --all
python -m probes.v1_rep_summary --run <path-to-v1_full_pos_shuffle>
```

## 4. Combined report

Обновить пути в `experiments/qwen35-4b-base/analysis/build_combined_report.py` при необходимости
(сейчас ждут `results/v1_full_pos_shuffle/...` и `results/highlight-h3-full/...` относительно корня experiment), затем:

```powershell
python experiments/qwen35-4b-base/analysis/build_combined_report.py
# → experiments/qwen35-4b-base/analysis/combined_h1_h3_h5_h11.html
# опционально скопировать в experiments/qwen35-4b-base/docs/index.html для Pages
```

H5 берётся из H3 (abstain × evidence) внутри builder — отдельный прогон не нужен.

## Структура experiment

```
experiments/qwen35-4b-base/
├── data/                 # junction → repo/data
├── steering/configs/     # 4B d_model/layers (для steering позже)
├── results/              # ← сюда артефакты с Vast
├── analysis/
└── docs/
```

Код inference/metrics/probes — в `repo/`; этот каталог только outputs + 4B configs.
