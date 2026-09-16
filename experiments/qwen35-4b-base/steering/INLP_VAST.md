# INLP Stage A→B→C on Vast (Qwen3.5-4B-Base)

Only repo: **https://github.com/niczavadskiy/occupational-gender-bias** (public, no `GH_TOKEN`).
Need **`HF_TOKEN`**. Torch stack: `torch>=2.5`, matching `torchvision`, `numpy<2`, `transformers>=4.50` (same as H1 Stage A).

## 0. Setup

```bash
export HF_TOKEN=hf_xxx
cd /workspace
git clone --depth 1 https://github.com/niczavadskiy/occupational-gender-bias.git   # or git pull
cd occupational-gender-bias && git pull

export REPO=/workspace/occupational-gender-bias
export STEER=$REPO/experiments/qwen35-4b-base/steering
export PYTHONPATH=$REPO
export MODEL=Qwen/Qwen3.5-4B-Base

# verify subspaces
ls -lh "$STEER/subspaces/inlp_gender_choice_v1.npz" "$STEER/subspaces/inlp_slot_choice_v1.npz"
python -c "from steering.intervene import SubspaceSpec; print('ok', SubspaceSpec)"
```

## 1. Smoke (both axes)

```bash
cd "$REPO"
OUT="$STEER/../results/steering"

python -m steering.run_inlp_stagea --model "$MODEL" --device cuda --dtype float32 \
  --subspaces "$STEER/subspaces/inlp_gender_choice_v1.npz" \
  --sample "$STEER/samples/h1_stagea_sample_v1.json" \
  --layers 24 --ranks 1,4 --alphas 1.0 --limit-items 3 \
  --primary-axis gender --out-root "$OUT" --tag inlp_gender_4b_smoke

python -m steering.run_inlp_stagea --model "$MODEL" --device cuda --dtype float32 \
  --subspaces "$STEER/subspaces/inlp_slot_choice_v1.npz" \
  --sample "$STEER/samples/h1_stagea_sample_v1.json" \
  --layers 30 --ranks 1,4 --alphas 1.0 --limit-items 3 \
  --primary-axis slot --out-root "$OUT" --tag inlp_slot_4b_smoke
```

## 2–4. Preferred: gender → slot in one command per stage

```bash
# tmux recommended
bash "$STEER/scripts/vast_inlp_stagea_both_and_pack.sh"   # gender A, then slot A
bash "$STEER/scripts/vast_inlp_stageb_both_and_pack.sh"   # gender B, then slot B (auto shortlist)
bash "$STEER/scripts/vast_inlp_stagec_both_and_pack.sh"   # gender C, then slot C (auto keep)
```

Or entire pipeline (many GPU-hours):

```bash
bash "$STEER/scripts/vast_inlp_all_stages_both_and_pack.sh"
```

Per-axis scripts remain for resume/debug (`vast_inlp_gender_*`, `vast_inlp_slot_*`).

Stage A writes `stageb_shortlist.json` automatically (winning layer by max R among CI>0 & above random; top-3 + rand0). Stage B writes `stagec_keep.json`.

If Stage A was unpacked with a nested folder, set e.g.  
`STAGE_A_DIR=/path/to/inlp_gender_4b_a_v1` before Stage B (gender script); same idea for slot.
## Outputs locally

```text
experiments/qwen35-4b-base/results/steering/inlp_stage_a/<tag>/stageb_shortlist.json
experiments/qwen35-4b-base/results/steering/inlp_stage_b/<tag>/stagec_keep.json
```

Manual CLI (optional):

```bash
python -m steering.inlp_shortlist from-stage-a --ranking .../ranking.csv --primary-axis gender --out shortlist.json
python -m steering.inlp_shortlist from-stage-b --keep .../keep.json --out stagec_keep.json
```

## Notes

- Legacy `*_b_v1` / `*_c_v1` = provisional peak (L24/L30); keep for comparison.
- Auto A→B may pick L31 k16/32/64 (top R) even if you previously preferred k8.

## Notes

- Subspaces already built (`k_chance=None` to k=64) — Stage A tests **behavior**, not AUC wipe.
- Gender probe-direction Stage A was ~null; INLP may still show rank-k effect (or confirm null).
- Slot probe-direction worked; INLP tests multidimensional erase of the same axis.
- Optional: install `causal_conv1d` / `flash-linear-attention` for speed only.
