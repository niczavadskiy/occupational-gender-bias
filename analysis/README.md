# EDA: inference-прогон

Самодостаточный HTML-отчёт (Plotly через CDN, открывать в браузере) по выходам
модели на factorial-дизайне.

## Файлы

| Файл | Что |
|---|---|
| `run_inference_eda.py` | генератор отчёта по **inference-прогону** factorial_v2 (per_item.jsonl → `inference_eda.html`) |
| `inference_eda.html` | готовый отчёт по прогону |

---

## `run_inference_eda.py` — behavioral EDA прогона

Descriptive-анализ выходов модели (`per_item.jsonl`) по всем факторам factorial-дизайна,
с разрезами под гипотезы H1-H5:

- **H1** prior bias (no_evidence × choice) — с явной пометкой про position-confound;
- **H3** counterfactual: двигается ли P(man) за `evidence_shift`;
- **H2** yesno-asymmetry P(Yes|man) − P(Yes|woman) — bias без position-confound;
- **H4** abstain effect: как часто берётся «Cannot determine»;
- **H5** per-predicate lean + группировка предикатов (agentic/communal/neutral).

**Источник данных** (в порядке приоритета): `--input <path>` → локальный
`results/<RUN>/per_item.jsonl` → скачать из HF Dataset
`bias-subspaces-group/qwen-bias-experiments` (нужен `HF_TOKEN` в `.env`).

```bash
python3 analysis/run_inference_eda.py            # local → HF fallback
python3 analysis/run_inference_eda.py --input results/<RUN>/per_item.jsonl
# → analysis/inference_eda.html
```
