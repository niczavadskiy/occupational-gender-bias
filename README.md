# Occupational Gender Bias — Analysis & Report

Statistical analysis of gender bias in **Qwen/Qwen3.5-2B-Base** on occupational
scenarios (O\*NET-based prompts with factorial controls).

**Live report:** [combined H1 · H3 · H5 · H11](https://niczavadskiy.github.io/occupational-gender-bias/)

## What is in this repository


| Component | Path | Hypothesis |
|-----------|------|------------|
| Combined report | [`analysis/combined_h1_h3_h5_h11.html`](analysis/combined_h1_h3_h5_h11.html) | H1 + H3 + H5 + H11 |
| H3 metrics | `results/highlight-h3-full/metrics_h3_v1/latest/` | evidence highlight vs no-evidence promt |
| H1 metrics | `results/v1_full_pos_shuffle/metrics_h1_v1/latest/` | position / layout bias |
| H11 probing summary | `results/v1_full_pos_shuffle/probes/` | gender axis in hidden states |
| Metrics code | `src/metrics/` | reproducible pipelines `h1_v1`, `h3_v1` |

### Model & scale

- **Model:** `Qwen/Qwen3.5-2B-Base`
- **H3 run:** `highlight-h3-full` — 45,648 items (3 evidence conditions × full factorial)
- **H1 run:** `v1_full_pos_shuffle` (source: `run_2026-06-09_12-50-33_Qwen3.5-2B-Base_v1_full_pos_shuffle`) — 15,216 items

## Attribution

- **Dataset design & inference outputs:** project [teammate](https://github.com/olyamasaeva) (used with permission):https://huggingface.co/buckets/H83/occupational-gender-bias-qwen3_5_2b  
- **Statistical analysis, metrics pipelines & report:** [niczavadskiy](https://github.com/niczavadskiy)

Raw prompts and model responses may be released separately as a Hugging Face Dataset.

## Reproducing metrics (requires local `per_item.jsonl`)

Metrics pipelines read `results/<run>/per_item.jsonl` from a full checkout with
inference data. From the repository root:

```bash
pip install -r requirements.txt

# H1 — position bias (needs per_item.jsonl for the v1 run)
python -m src.metrics.h1_v1 --run v1_full_pos_shuffle

# H3 — evidence highlight (needs per_item.jsonl for highlight-h3-full)
python -m src.metrics.h3_v1 --run highlight-h3-full

# Combined HTML report (after metric summaries exist)
python analysis/build_combined_report.py
```

## License

MIT — see [LICENSE](LICENSE). Dataset prompts are used with permission of the design author;
contact maintainers before redistributing derived stimulus sets.
