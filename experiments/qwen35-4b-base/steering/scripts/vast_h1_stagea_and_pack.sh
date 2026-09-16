#!/usr/bin/env bash
# Qwen3.5-4B-Base — H1 Stage A (gender equalize) + pack.
#
# Prerequisites on Vast:
#   - public repo niczavadskiy/occupational-gender-bias (HF_TOKEN for model)
#   - artifacts under experiments/qwen35-4b-base/steering/
#
# Usage:
#   export REPO=/workspace/occupational-gender-bias
#   export STEER=$REPO/experiments/qwen35-4b-base/steering
#   bash "$STEER/scripts/vast_h1_stagea_and_pack.sh"
set -euo pipefail

STEER_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${REPO:-/workspace/occupational-gender-bias}"
MODEL="${MODEL:-Qwen/Qwen3.5-4B-Base}"
TAG="${TAG:-h1_4b_a_v1}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

CAND="$STEER_DIR/candidates/h1_candidates_v1.json"
VEC="$STEER_DIR/vectors/h1_vectors_v1.npz"
SAMPLE="$STEER_DIR/samples/h1_stagea_sample_v1.json"
OUT_ROOT="${OUT_ROOT:-$STEER_DIR/../results/steering}"

for f in "$CAND" "$VEC" "$SAMPLE"; do
  test -f "$f" || { echo "MISSING $f"; exit 1; }
done

echo "=== H1 Stage A 4B [$TAG] model=$MODEL ==="
python -m steering.run_h1_stagea \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --sample "$SAMPLE" \
  --candidates-file "$CAND" \
  --vectors "$VEC" \
  --out-root "$OUT_ROOT" \
  --tag "$TAG" \
  --log-every 100 \
  "$@"

OUT=/workspace/h1_stage_a_${TAG}.tar.gz
tar -czf "$OUT" -C "$OUT_ROOT/stage_a" "$TAG"
ls -lh "$OUT"
echo "DONE. Download: $OUT"
