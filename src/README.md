# src/ — pipeline scripts

## Текущие

| Файл | Что |
|---|---|
| `smoke_qwen.py` | Smoke-test Qwen3.5-2B-Base: загрузка, проверка token IDs для A/B/C, forward + hidden states, log-prob ratio. Запуск: `python3 src/smoke_qwen.py`. |
| `prepare_factorial.py` | Адаптер: v2 CSV → JSONL для inference. Собирает prompt с "Answer:" в конце. 1350 items. Запуск: `python3 src/prepare_factorial.py data/factorial_v2_with_gender.csv`. |
| `inference.py` | Полный inference: constrained log-prob над A/B/C + HS snapshot всех слоёв на last-token. Сохраняет `hidden_states.npz` + `per_item.jsonl` + `meta.json` в `$RESULTS_DIR/run_*/` (default `/workspace/results`). Поддерживает variable abstain. Запуск: `python3 src/inference.py --items_file data/factorial_v2_with_gender.prepared.jsonl`. |
| `upload_to_hf.py` | Заливка results folder в приватный HF Dataset repo `bias-subspaces-group/qwen-bias-experiments`. Запуск: `python3 src/upload_to_hf.py results/run_*/`. |

## Запланированы

| Файл | Что |
|---|---|
| `metrics.py` | H1-H9 behavioral метрики: rate, asymmetry, McNemar, two-prop z-test, FDR correction. |
| `probe_pipeline.py` | H10-H13 linear probe: layer-scan, control task (Hewitt-Liang), silhouette score, direction extraction. |
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
