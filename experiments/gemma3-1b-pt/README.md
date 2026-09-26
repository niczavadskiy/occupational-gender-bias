# Gemma 3 1B PT → beh → HS-probe → INLP → pool XY

Цель: тот же occupational pipeline, что на Qwen3.5-*-Base, на **`google/gemma-3-1b-pt`**.

| | |
|---|---|
| Model | `google/gemma-3-1b-pt` (pretrained, не `-it`) |
| Arch (smoke) | 26 blocks · `d_model=1152` · HS `0=emb, 1..26=block` |
| Dataset | shared `data/v1/` (45648) via junction `data/` |

## Pipeline

```
smoke → beh (H1·H3 pack) → H1 metrics / FDR
      → HS-probe (gender peak)
      → classical INLP (peak + peak−1 + peak−2)
      → pool INLP + pool XY-control
```

**Peak layer and FDR SOC lists are unknown until beh + probe finish.**
Steering configs under `steering/` use `PEAK` placeholders until then.

## 1. Vast — behavioral pack (Phase 1)

```bash
export HF_TOKEN=hf_xxx   # + accept Gemma license on HF Hub
# default BRANCH=main (after merge of polarity/Gemma work)
bash scripts/setup_instance.sh
bash scripts/vast_gemma3_1b_h1h3_and_pack.sh
```

If the checkout is stale:

```bash
cd /workspace/occupational-gender-bias
git fetch --depth 1 origin main
git checkout main
git reset --hard origin/main
```

Or Jupyter: [`notebooks/vast_gemma3_1b_h1h3.ipynb`](../../notebooks/vast_gemma3_1b_h1h3.ipynb)

Pack: `/workspace/gemma3_1b_h1h3_pack.tar.gz`

## 2. After download — metrics + probe

Put runs in `experiments/gemma3-1b-pt/results/`:

```
results/
  v1_full_pos_shuffle/   # or run_*_gemma-3-1b-pt_v1_full_pos_shuffle
  highlight-h3-full/
```

```powershell
python -m src.metrics.h1_v1 --run experiments/gemma3-1b-pt/results/v1_full_pos_shuffle
python -m src.metrics.h3_v1 --run experiments/gemma3-1b-pt/results/highlight-h3-full
# HS-probe (same probes package as 4B):
python -m probes.v1_rep_run --run <v1_full_pos_shuffle> --all
python -m probes.v1_rep_summary --run <v1_full_pos_shuffle>
```

Then fill `PEAK` in `steering/configs/*` and build FDR polarity sets
(`steering/PIPELINE.md`).

## 3. Structure

```
experiments/gemma3-1b-pt/
├── data/                 # junction → repo data/
├── results/              # ← Vast packs
├── analysis/
├── docs/
└── steering/             # configs / stubs / Vast after peak
    ├── PIPELINE.md
    ├── configs/
    └── scripts/
```

Код inference / metrics / steering runners — в корне репо.
