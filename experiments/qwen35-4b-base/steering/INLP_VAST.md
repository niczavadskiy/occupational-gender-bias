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

Download packs. Inspect `ranking.csv` (already done for A):
- gender: **L23** k∈{16,32,64} (L24 provisional was null)
- slot: **L31** k∈{8,16,32} (beats L30 at matched k)

Shortlists are fixed from Stage A (2026-09-16):
- `candidates/inlp_gender_stageb_shortlist_4b_v1.json` → L23 k16/32/64 + rand
- `candidates/inlp_slot_stageb_shortlist_4b_v1.json` → L31 k8/16/32 + rand  
(push or scp onto the instance before B)

Legacy packs `*_b_v1` / `*_c_v1` used provisional peak layers; keep them for comparison. New runs use **`*_v2`**.

## 3. Stage B — capability (MMLU)

```bash
# defaults: gender L23 k16,32,64 → tag inlp_gender_4b_b_v2
#           slot   L31 k8,16,32  → tag inlp_slot_4b_b_v2
bash "$STEER/scripts/vast_inlp_gender_stageb_and_pack.sh"
bash "$STEER/scripts/vast_inlp_slot_stageb_and_pack.sh"
```

Keep winners with `cap_loss ≤ 0.03`. Freeze keep files from `keep.json`:
- `candidates/inlp_gender_stagec_keep_4b_v1.json`
- `candidates/inlp_slot_stagec_keep_4b_v1.json`

## 4. Stage C — held-out report

```bash
# after freezing keep from B v2
bash "$STEER/scripts/vast_inlp_gender_stagec_and_pack.sh"   # → inlp_gender_4b_c_v2
bash "$STEER/scripts/vast_inlp_slot_stagec_and_pack.sh"     # → inlp_slot_4b_c_v2
```

## Outputs locally

```text
experiments/qwen35-4b-base/results/steering/inlp_stage_a/   # a_v1 unchanged
experiments/qwen35-4b-base/results/steering/inlp_stage_b/   # b_v1 (old) + b_v2 (rerun)
experiments/qwen35-4b-base/results/steering/inlp_stage_c/   # c_v1 (old) + c_v2 (rerun)
```

## Notes

- Subspaces already built (`k_chance=None` to k=64) — Stage A tests **behavior**, not AUC wipe.
- Gender probe-direction Stage A was ~null; INLP may still show rank-k effect (or confirm null).
- Slot probe-direction worked; INLP tests multidimensional erase of the same axis.
- Optional: install `causal_conv1d` / `flash-linear-attention` for speed only.
