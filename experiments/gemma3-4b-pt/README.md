# Gemma 3 4B PT → beh → HS-probe → INLP → pool XY

Цель: тот же occupational pipeline, что на Qwen3.5-*-Base / Gemma 3 1B, на **`google/gemma-3-4b-pt`**.

| | |
|---|---|
| Model | `google/gemma-3-4b-pt` (pretrained, не `-it`) |
| Arch (smoke) | 34 blocks · `d_model=2560` · HS `0=emb, 1..34=block` |
| Note | Multimodal Hub card; text-only via `AutoModelForCausalLM` + `transformers>=4.50` |
| Dataset | shared `data/v1/` (45648) via junction `data/` |

## Pipeline

```
smoke → beh (H1·H3 pack) → H1 metrics / FDR
      → HS-probe (gender_choice + slot_choice peaks)
      → classical INLP A→B→C (gender + slot)
      → pool INLP A→B→C (gender; skip empty polarity sets)
      → pool XY-control A→B→C (gender; skip empty)
```

**Peaks locked:** gender **L24** (belt 24/23/22), slot **L31** (belt 31/30/29).
See [`steering/PIPELINE.md`](steering/PIPELINE.md).

## 1. Vast — behavioral pack (Phase 1)

```bash
export HF_TOKEN=hf_xxx   # + accept Gemma license on HF Hub
# default BRANCH=main
bash scripts/setup_instance.sh
bash scripts/vast_gemma3_4b_h1h3_and_pack.sh
```

GPU: как у Qwen 4B (HS ≈ 35×2560); `DISK≥120` рекомендуется.

If the checkout is stale:

```bash
cd /workspace/occupational-gender-bias
git fetch --depth 1 origin main
git checkout -B main origin/main
git reset --hard origin/main
```

Or Jupyter: [`notebooks/vast_gemma3_4b_h1h3.ipynb`](../../notebooks/vast_gemma3_4b_h1h3.ipynb)

Pack: `/workspace/gemma3_4b_h1h3_pack.tar.gz`

## 2. After download — metrics + probe

Put runs in `experiments/gemma3-4b-pt/results/`:

```
results/
  v1_full_pos_shuffle/   # or run_*_gemma-3-4b-pt_v1_full_pos_shuffle
  highlight-h3-full/
```

```powershell
python -m src.metrics.h1_v1 --run experiments/gemma3-4b-pt/results/v1_full_pos_shuffle
python -m src.metrics.h3_v1 --run experiments/gemma3-4b-pt/results/highlight-h3-full
# HS-probe (same probes package as Qwen 4B):
python -m probes.v1_rep_run --run <v1_full_pos_shuffle> --all
python -m probes.v1_rep_summary --run <v1_full_pos_shuffle>
```

Then fill `PEAK` in `steering/configs/*` and build FDR polarity sets
(`steering/PIPELINE.md`).

## 3. Structure

```
experiments/gemma3-4b-pt/
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
