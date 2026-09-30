# Ministral 3 8B Base → beh → HS-probe → INLP → pool XY

Цель: тот же occupational pipeline, что на Qwen / Gemma, на **`mistralai/Ministral-3-8B-Base-2512`** в **bf16**.

| | |
|---|---|
| Model | `mistralai/Ministral-3-8B-Base-2512` (Base, не Instruct) |
| Arch | **34** blocks · **d_model=4096** · HS `0=emb, 1..34=block` (`n_hidden_states=35`) |
| Dtype | **bfloat16** (beh + steering) |
| Dataset | shared `data/v1/` (45648) via junction `data/` |
| GPU | **≥48GB** recommended for beh+steering; 24GB tight. DISK≥160 |

## Pipeline

```
smoke → beh (H1·H3 pack) → H1 metrics / FDR
      → HS-probe (gender_choice + slot_choice peaks)
      → classical INLP A→B→C (gender + slot)
      → pool INLP A→B→C (gender; skip empty polarity sets)
      → pool XY-control A→B→C (gender; skip empty)
```

**Peaks:** fill after probe — see [`steering/PIPELINE.md`](steering/PIPELINE.md).

## 1. Vast — behavioral pack (Phase 1)

```bash
export HF_TOKEN=hf_xxx
bash scripts/setup_instance.sh
INSTALL_MINISTRAL=1 bash scripts/vast_ministral3_8b_h1h3_and_pack.sh
```

Or via `vast_run.sh`:

```bash
HF_TOKEN=hf_xxx DISK=160 KEEP=1 GPU=RTX_A6000 \
REMOTE_CMD='bash scripts/setup_instance.sh && INSTALL_MINISTRAL=1 bash scripts/vast_ministral3_8b_h1h3_and_pack.sh' \
bash scripts/vast_run.sh
```

Pack: `/workspace/ministral3_8b_h1h3_pack.tar.gz`

## 2. After download — metrics + probe

Put runs in `experiments/ministral3-8b-base/results/`:

```
results/
  v1_full_pos_shuffle/
  highlight-h3-full/
```

```powershell
python -m src.metrics.h1_v1 --run experiments/ministral3-8b-base/results/v1_full_pos_shuffle
python -m src.metrics.h3_v1 --run experiments/ministral3-8b-base/results/highlight-h3-full
python -m probes.v1_rep_run --run <v1_full_pos_shuffle> --all
python -m probes.v1_rep_summary --run <v1_full_pos_shuffle>
```

Then fill peaks + FDR polarity sets (`steering/PIPELINE.md`).

## 3. Structure

```
experiments/ministral3-8b-base/
├── data/                 # junction → repo data/
├── results/              # ← Vast packs
├── analysis/
├── docs/
└── steering/
    ├── PIPELINE.md
    ├── configs/
    └── scripts/
```

Код inference / metrics / steering — в корне репо.
