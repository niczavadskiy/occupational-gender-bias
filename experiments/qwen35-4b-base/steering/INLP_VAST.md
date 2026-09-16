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

## 2. Stage A (full) — preference

```bash
# tmux recommended; ~hours each (3 layers × ranks × 380 rows)
bash "$STEER/scripts/vast_inlp_gender_stagea_and_pack.sh"
# → /workspace/inlp_gender_stage_a_inlp_gender_4b_a_v1.tar.gz

bash "$STEER/scripts/vast_inlp_slot_stagea_and_pack.sh"
# → /workspace/inlp_slot_stage_a_inlp_slot_4b_a_v1.tar.gz
```

Download packs. Stage A now writes `stageb_shortlist.json` automatically
(winning layer by max R among CI>0 & above random; top-3 ranks + rand0).

## 3. Stage B — capability (MMLU), auto from Stage A

```bash
# reads $OUT_ROOT/inlp_stage_a/<a_tag>/stageb_shortlist.json via --from-stage-a
# writes stagec_keep.json for Stage C; tags *_b_v2
bash "$STEER/scripts/vast_inlp_gender_stageb_and_pack.sh"
bash "$STEER/scripts/vast_inlp_slot_stageb_and_pack.sh"
```

If Stage A was downloaded with a nested folder, set:
`STAGE_A_DIR=/path/to/.../inlp_gender_4b_a_v1`

## 4. Stage C — held-out report, auto from Stage B

```bash
# reads stagec_keep.json via --from-stage-b; tags *_c_v2
bash "$STEER/scripts/vast_inlp_gender_stagec_and_pack.sh"
bash "$STEER/scripts/vast_inlp_slot_stagec_and_pack.sh"
```

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
