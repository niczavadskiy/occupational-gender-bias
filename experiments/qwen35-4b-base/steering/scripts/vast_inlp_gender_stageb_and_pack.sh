#!/usr/bin/env bash
# Qwen3.5-4B — INLP gender Stage B (MMLU capability) + pack.
set -euo pipefail

STEER_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${REPO:-/workspace/occupational-gender-bias}"
MODEL="${MODEL:-Qwen/Qwen3.5-4B-Base}"
TAG="${TAG:-inlp_gender_4b_b_v1}"
LAYERS="${LAYERS:-24}"
RANKS="${RANKS:-8,16}"
ALPHAS="${ALPHAS:-1.0}"
SHORTLIST="${SHORTLIST:-$STEER_DIR/candidates/inlp_gender_stageb_shortlist_4b_v1.json}"
PARQUET="${PARQUET:-$REPO/steering/.cache/mmlu_pro_test.parquet}"
URL="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

SUB="$STEER_DIR/subspaces/inlp_gender_choice_v1.npz"
OUT_ROOT="${OUT_ROOT:-$STEER_DIR/../results/steering}"

mkdir -p "$REPO/steering/.cache"
if [[ ! -f "$PARQUET" ]]; then
  echo "=== download MMLU-Pro parquet ==="
  curl -L -o "$PARQUET" "$URL"
fi
python -c "import pyarrow" 2>/dev/null || pip install -q pyarrow

for f in "$SUB" "$SHORTLIST"; do
  test -f "$f" || { echo "MISSING $f"; exit 1; }
done

echo "=== INLP gender Stage B 4B [$TAG] ==="
python -m steering.run_inlp_stageb \
  --model "$MODEL" --device cuda --dtype float32 \
  --subspaces "$SUB" \
  --shortlist "$SHORTLIST" \
  --layers "$LAYERS" --ranks "$RANKS" --alphas "$ALPHAS" \
  --out-root "$OUT_ROOT" --tag "$TAG" --log-every 100 \
  "$@"

OUT=/workspace/inlp_gender_stage_b_${TAG}.tar.gz
tar -czf "$OUT" -C "$OUT_ROOT/inlp_stage_b" "$TAG"
ls -lh "$OUT"
echo "DONE. Download: $OUT"
echo "Next: freeze winners into candidates/inlp_gender_stagec_keep_4b_v1.json"
