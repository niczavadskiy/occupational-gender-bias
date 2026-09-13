#!/usr/bin/env bash
# Qwen3.5-4B-Base — INLP gender Stage A (rank-k center) + pack.
set -euo pipefail

STEER_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${REPO:-$(pwd)}"
MODEL="${MODEL:-Qwen/Qwen3.5-4B-Base}"
TAG="${TAG:-inlp_gender_4b_a_v1}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"
LAYERS="${LAYERS:-23,24,25}"
RANKS="${RANKS:-1,2,4,8}"
ALPHAS="${ALPHAS:-1.0}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

SUB="$STEER_DIR/subspaces/inlp_gender_choice_v1.npz"
SAMPLE="$STEER_DIR/samples/h1_stagea_sample_v1.json"
OUT_ROOT="${OUT_ROOT:-$STEER_DIR/../results/steering}"

for f in "$SUB" "$SAMPLE"; do
  test -f "$f" || { echo "MISSING $f — run build_artifacts.sh first"; exit 1; }
done

echo "=== INLP gender Stage A 4B [$TAG] L=$LAYERS ranks=$RANKS ==="
python -m steering.run_inlp_stagea \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --subspaces "$SUB" \
  --sample "$SAMPLE" \
  --layers "$LAYERS" \
  --ranks "$RANKS" \
  --alphas "$ALPHAS" \
  --out-root "$OUT_ROOT" \
  --tag "$TAG" \
  --log-every 100 \
  "$@"

OUT=/workspace/inlp_gender_stage_a_${TAG}.tar.gz
# run_inlp_stagea writes under inlp_stage_a/
tar -czf "$OUT" -C "$OUT_ROOT/inlp_stage_a" "$TAG" 2>/dev/null \
  || tar -czf "$OUT" -C "$OUT_ROOT" "inlp_stage_a/$TAG"
ls -lh "$OUT"
echo "DONE. Download: $OUT"
