#!/usr/bin/env bash
# Qwen3.5-4B — INLP gender Stage C from auto Stage B keep + pack.
set -euo pipefail

STEER_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${REPO:-/workspace/occupational-gender-bias}"
MODEL="${MODEL:-Qwen/Qwen3.5-4B-Base}"
TAG="${TAG:-inlp_gender_4b_c_v2}"
STAGE_B_TAG="${STAGE_B_TAG:-inlp_gender_4b_b_v2}"
OUT_ROOT="${OUT_ROOT:-$STEER_DIR/../results/steering}"
STAGE_B_DIR="${STAGE_B_DIR:-$OUT_ROOT/inlp_stage_b/$STAGE_B_TAG}"
SAMPLE="${SAMPLE:-$STEER_DIR/samples/inlp_stagec_sample_v1.json}"
PARQUET="${PARQUET:-$REPO/steering/.cache/mmlu_pro_test.parquet}"
URL="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

SUB="$STEER_DIR/subspaces/inlp_gender_choice_v1.npz"

mkdir -p "$REPO/steering/.cache"
if [[ ! -f "$PARQUET" ]]; then
  curl -L -o "$PARQUET" "$URL"
fi
python -c "import pyarrow" 2>/dev/null || pip install -q pyarrow

test -f "$SUB" || { echo "MISSING $SUB"; exit 1; }
test -f "$SAMPLE" || { echo "MISSING $SAMPLE"; exit 1; }
test -d "$STAGE_B_DIR" || { echo "MISSING Stage B dir $STAGE_B_DIR"; exit 1; }

if [[ ! -f "$STAGE_B_DIR/stagec_keep.json" && -f "$STAGE_B_DIR/keep.json" ]]; then
  python -m steering.inlp_shortlist from-stage-b \
    --keep "$STAGE_B_DIR/keep.json" \
    --hypothesis inlp_gender --model-scale qwen35_4b \
    --out "$STAGE_B_DIR/stagec_keep.json"
fi
cp -f "$STAGE_B_DIR/stagec_keep.json" \
  "$STEER_DIR/candidates/inlp_gender_stagec_keep_4b_v1.json"

echo "=== INLP gender Stage C 4B [$TAG] from $STAGE_B_DIR ==="
python -m steering.run_inlp_stagec \
  --model "$MODEL" --device cuda --dtype float32 \
  --subspaces "$SUB" \
  --sample "$SAMPLE" \
  --from-stage-b "$STAGE_B_DIR" \
  --out-root "$OUT_ROOT" --tag "$TAG" --log-every 100 \
  "$@"

OUT=/workspace/inlp_gender_stage_c_${TAG}.tar.gz
tar -czf "$OUT" -C "$OUT_ROOT/inlp_stage_c" "$TAG"
ls -lh "$OUT"
echo "DONE. Download: $OUT"
