#!/usr/bin/env bash
# Slot Stage C report + pack for download.
set -euo pipefail

cd "$(dirname "$0")/../.."
export REPO="$(pwd)"

TAG="${TAG:-slot_c_v1}"
PARQUET="${PARQUET:-steering/.cache/mmlu_pro_test.parquet}"
URL="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"

mkdir -p steering/.cache
if [[ ! -f "$PARQUET" ]]; then
  echo "=== download MMLU-Pro parquet ==="
  curl -L -o "$PARQUET" "$URL"
fi
python -c "import pyarrow" 2>/dev/null || pip install -q pyarrow

test -f steering/samples/slot_stagec_sample_v1.json || {
  echo "MISSING slot_stagec_sample_v1.json — git pull"
  exit 1
}

echo "=== Slot Stage C report [$TAG] ==="
python -m steering.run_slot_stagec \
  --model "${MODEL:-Qwen/Qwen3.5-2B-Base}" \
  --device cuda \
  --dtype float32 \
  --tag "$TAG" \
  --log-every 100 \
  "$@"

OUT=/workspace/slot_stage_c_${TAG}.tar.gz
tar -czf "$OUT" -C results/steering/stage_c "$TAG"
ls -lh "$OUT"
echo "DONE. Download: $OUT"
