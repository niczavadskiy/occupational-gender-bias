# src/ — pipeline scripts

## Текущие

| Файл | Что |
|---|---|
| `smoke_qwen.py` | Smoke-test Qwen3.5-2B-Base: загрузка, проверка token IDs для A/B/C, forward + hidden states, log-prob ratio. Запуск: `python3 src/smoke_qwen.py`. |
| `prepare_factorial.py` | Адаптер: v2 CSV → JSONL для inference. Собирает prompt с "Answer:" в конце. 1350 items. Запуск: `python3 src/prepare_factorial.py data/factorial_v2_with_gender.csv`. |
| `inference.py` | Полный inference: constrained log-prob над A/B/C + HS snapshot всех слоёв на last-token. Сохраняет `hidden_states.npz` + `per_item.jsonl` + `meta.json` в `$RESULTS_DIR/run_*/` (default `/workspace/results`). Поддерживает variable abstain. Запуск: `python3 src/inference.py --items_file data/factorial_v2_with_gender.prepared.jsonl`. |
| `upload_to_hf.py` | Заливка results folder в приватный HF Dataset repo `bias-subspaces-group/qwen-bias-experiments`. Запуск: `python3 src/upload_to_hf.py results/run_*/`. |
| `per_item_jsonl_to_csv.py` | `per_item.jsonl` → плоский `per_item.csv` (labels развёрнуты в `label_A/B/C`). Запуск: `python src/per_item_jsonl_to_csv.py --run <run_name>`. |
| `annotation/run.py` | LLM-разметка (OpenRouter, батчи по 10, structured JSON). Результаты: `data/annotation/runs/`. См. [data/annotation/README.md](../data/annotation/README.md). |

## Behavioral metrics (H1–H9)

| Модуль | Что |
|---|---|
| `src/metrics/` | H1–H9 из `per_item.jsonl`: rates, χ², two-prop, McNemar, BH-FDR |

| Гипотеза | Статус | Тесты |
|---|---|---|
| H1 | активна | P(pro-man)/P(pro-woman) vs 0.5: **choice + yesno**, semantic `labels[choice]`; strata position × evidence × abstain |
| H2 | активна | 8× P(stereo) vs annotation baseline + stereo vs anti по `annotations.jsonl` |
| H3 | активна | χ² evidence×(M,W) на всех main; ×(M,W,C) with_abstain; post-hoc pairs |
| H4 | активна (5400+) | with vs without abstain → P(Yes\|answerability) по форматам (default all): H4↑/H4↓ + McNemar |
| H5 | активна | P(abstain): no_evidence vs evidence (with_abstain, все format/position) |
| H6 | активна | 4× two-prop: same-axis + cross (choice vs yes/no; no_evidence, without_abstain) |
| post_hoc_H | вспом. | McNemar / asymmetry yesno_man vs yesno_woman → `post_hoc_H.csv` |
| H7 | активна (5400+) | evidence vs no_evidence → P(Yes\|answerability), pooled |
| H8 | активна (5400+) | self-Q No → чаще C на main (with_abstain, pooled z + McNemar) |
| H9 | активна (5400+) | ev_man vs ev_woman → P(Yes\|answerability), pooled (two-sided z + McNemar) |

```bash
cd repo
python -m src.metrics
python -m src.metrics --run 5400 --hypotheses H4
python -m src.metrics --run results/5400 --hypotheses H4,H7,H8,H9 --h4-formats yesno
python -m src.metrics --input results/5400/per_item.jsonl --hypotheses H4 --out results/5400/metrics/h4_latest
python -m src.metrics --hypotheses H1,H2,H3,H5,H6 --level family --fdr 0.05
```

`--run` / `--run-dir` / `--input`: имя прогона (`5400`), путь `results/<run>/`, абсолютный каталог с `per_item.jsonl`, или сам файл `per_item.jsonl`.

Поддерживаемые размеры `per_item.jsonl`: **1350**, **5400**, **10800** (v3, один context_order), **21600** (v3 merged). По умолчанию `--position-variant auto` и `--context-order auto` → **все позиции и все context_order** (без фильтра). Явный срез: `--context-order man_first`, `--position-variant p0`.

`--h4-formats`: `all` (default), `choice`, `yesno` (= yesno_man + yesno_woman), `yesno_man`, `yesno_woman`, или список через запятую.

Выход: `results/<run>/metrics/<timestamp>/` (`tests_all.csv`, `rates_summary.csv`, `summary.md`).

Аннотации для H2: `data/annotation/runs/<run_id>/annotations.jsonl` (см. `data/annotation/README.md`). CLI: `--annotations <path|run_id>`.

## Запланированы

| Файл | Что |
|---|---|
| `../probes/` | Linear probing H10–H13 (сейчас H11: `h11_run`, HF download). См. [probes/README.md](../probes/README.md). |
| `cross_form_eval.py` | Cross-form transferability test для localization claim. |

## Зависимости и quick start

Полный recipe — в [корневом README.md](../README.md). Кратко:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # под драйвер хоста
pip install -r requirements.txt
cp .env.example .env  &&  $EDITOR .env   # положить HF_TOKEN
```

## Notes

- **`trust_remote_code=True`** обязателен для Qwen3.5 (новая архитектура — Gated DeltaNet + Gated Attention).
- TransformerLens может не поддерживать архитектуру — fall back на `transformers` + manual hooks для HS extraction.
- Под HS-extraction snapshot: `[N_items, n_layers+1, d_model] = [1350, 25, 2048]` float16 ≈ 130 MB, ~100 MB после compression.
- HF Datasets: `bias-subspaces-group/qwen-bias-experiments` (private). Доступ — через инвайт в org.
