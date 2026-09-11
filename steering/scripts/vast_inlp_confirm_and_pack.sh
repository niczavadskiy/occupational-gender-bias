#!/usr/bin/env bash
# Gender INLP confirmatory (test sample) + archive Stage A + confirm for download.
# Run on Vast from repo root:  bash steering/scripts/vast_inlp_confirm_and_pack.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
export REPO="$(pwd)"

SAMPLE="${SAMPLE:-steering/samples/inlp_test_sample_v1.json}"
SUBSPACES="${SUBSPACES:-steering/subspaces/inlp_gender_choice_v1.npz}"
MODEL="${MODEL:-Qwen/Qwen3.5-2B-Base}"
TAG_CONFIRM="${TAG_CONFIRM:-inlp_confirm_test_v1}"
TAG_STAGEA="${TAG_STAGEA:-inlp_v1}"

echo "REPO=$REPO"
test -f "$SAMPLE" || { echo "MISSING $SAMPLE — git pull or upload sample"; exit 1; }
test -f "$SUBSPACES" || { echo "MISSING $SUBSPACES"; exit 1; }

echo "=== confirmatory Stage B preference (L15 k=8,16) ==="
python -m steering.run_inlp_stagea \
  --model "$MODEL" \
  --device cuda \
  --dtype float32 \
  --subspaces "$SUBSPACES" \
  --sample "$SAMPLE" \
  --layers 15 \
  --ranks 8,16 \
  --alphas 1.0 \
  --tag "$TAG_CONFIRM" \
  --log-every 100

echo "=== pack results ==="
OUT=/workspace/inlp_gender_stageA_B_bundle.tar.gz
tar -czf "$OUT" \
  -C results/steering/inlp_stage_a "$TAG_STAGEA" "$TAG_CONFIRM" \
  -C "$REPO" steering/samples/inlp_test_sample_v1.json steering/samples/inlp_test_sample_v1.md \
  2>/dev/null || tar -czf "$OUT" \
  -C results/steering/inlp_stage_a "$TAG_CONFIRM" \
  -C "$REPO" steering/samples/inlp_test_sample_v1.json

ls -lh "$OUT"
echo "DONE. Download: $OUT"
echo "  (Vast UI file browser / Jupyter / scp when SSH works)"
