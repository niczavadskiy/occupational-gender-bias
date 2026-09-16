#!/usr/bin/env bash
# Qwen3.5-4B — INLP gender Stage B from auto Stage A shortlist + pack.
set -euo pipefail

STEER_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${REPO:-/workspace/occupational-gender-bias}"
MODEL="${MODEL:-Qwen/Qwen3.5-4B-Base}"
TAG="${TAG:-inlp_gender_4b_b_v2}"
STAGE_A_TAG="${STAGE_A_TAG:-inlp_gender_4b_a_v1}"
OUT_ROOT="${OUT_ROOT:-$STEER_DIR/../results/steering}"
STAGE_A_DIR="${STAGE_A_DIR:-$OUT_ROOT/inlp_stage_a/$STAGE_A_TAG}"
PARQUET="${PARQUET:-$REPO/steering/.cache/mmlu_pro_test.parquet}"
URL="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

SUB="$STEER_DIR/subspaces/inlp_gender_choice_v1.npz"

mkdir -p "$REPO/steering/.cache"
if [[ ! -f "$PARQUET" ]]; then
  echo "=== download MMLU-Pro parquet ==="
  curl -L -o "$PARQUET" "$URL"
fi
python -c "import pyarrow" 2>/dev/null || pip install -q pyarrow

test -f "$SUB" || { echo "MISSING $SUB"; exit 1; }
test -d "$STAGE_A_DIR" || { echo "MISSING Stage A dir $STAGE_A_DIR"; exit 1; }

# Ensure shortlist exists (builds from ranking if needed)
if [[ ! -f "$STAGE_A_DIR/stageb_shortlist.json" ]]; then
  echo "=== build stageb_shortlist from Stage A ranking ==="
  python -m steering.inlp_shortlist from-stage-a \
    --ranking "$STAGE_A_DIR/ranking.csv" \
    --primary-axis gender \
    --hypothesis inlp_gender --model-scale qwen35_4b \
    --out "$STAGE_A_DIR/stageb_shortlist.json"
fi
# Mirror into candidates/ for git-friendly artifact
cp -f "$STAGE_A_DIR/stageb_shortlist.json" \
  "$STEER_DIR/candidates/inlp_gender_stageb_shortlist_4b_v1.json"

echo "=== INLP gender Stage B 4B [$TAG] from $STAGE_A_DIR ==="
python -m steering.run_inlp_stageb \
  --model "$MODEL" --device cuda --dtype float32 \
  --subspaces "$SUB" \
  --from-stage-a "$STAGE_A_DIR" \
  --out-root "$OUT_ROOT" --tag "$TAG" --log-every 100 \
  "$@"

# Mirror Stage C keep
STAGE_B_DIR="$OUT_ROOT/inlp_stage_b/$TAG"
if [[ -f "$STAGE_B_DIR/stagec_keep.json" ]]; then
  cp -f "$STAGE_B_DIR/stagec_keep.json" \
    "$STEER_DIR/candidates/inlp_gender_stagec_keep_4b_v1.json"
fi

OUT=/workspace/inlp_gender_stage_b_${TAG}.tar.gz
tar -czf "$OUT" -C "$OUT_ROOT/inlp_stage_b" "$TAG"
ls -lh "$OUT"
echo "DONE. Download: $OUT"
echo "Next: bash $STEER_DIR/scripts/vast_inlp_gender_stagec_and_pack.sh"
