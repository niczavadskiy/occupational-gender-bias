#!/usr/bin/env bash
# Slot Stage B (preference remainder val + MMLU) + pack.
set -euo pipefail

cd "$(dirname "$0")/../.."
export REPO="$(pwd)"

TAG="${TAG:-slot_b_v2}"
PARQUET="${PARQUET:-steering/.cache/mmlu_pro_test.parquet}"
URL="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"

mkdir -p steering/.cache
if [[ ! -f "$PARQUET" ]]; then
  echo "=== download MMLU-Pro parquet ==="
  curl -L -o "$PARQUET" "$URL"
fi
python -c "import pyarrow" 2>/dev/null || pip install -q pyarrow

test -f steering/samples/h1_stageb_sample_v1.json || {
  echo "MISSING h1_stageb_sample_v1.json — build or git pull"
  exit 1
}

echo "=== Slot Stage B [$TAG] ==="
python -m steering.run_slot_stageb \
  --model "${MODEL:-Qwen/Qwen3.5-2B-Base}" \
  --device cuda \
  --dtype float32 \
  --tag "$TAG" \
  --shortlist steering/candidates/slot_stageb_shortlist_v1.json \
  --log-every 100 \
  "$@"

OUT=/workspace/slot_stage_b_${TAG}.tar.gz
tar -czf "$OUT" -C results/steering/stage_b "$TAG"
ls -lh "$OUT"
echo "DONE. Download: $OUT"
