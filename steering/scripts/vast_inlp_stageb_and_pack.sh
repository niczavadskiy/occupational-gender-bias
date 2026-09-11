#!/usr/bin/env bash
# INLP Stage B capability + pack for download.
# Prefetch: curl MMLU parquet into steering/.cache/ if missing.
set -euo pipefail

cd "$(dirname "$0")/../.."
export REPO="$(pwd)"

TAG="${TAG:-inlp_b_v1}"
PARQUET="${PARQUET:-steering/.cache/mmlu_pro_test.parquet}"
URL="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"

mkdir -p steering/.cache
if [[ ! -f "$PARQUET" ]]; then
  echo "=== download MMLU-Pro parquet ==="
  curl -L -o "$PARQUET" "$URL"
fi

# pandas.read_parquet needs an engine (pyarrow preferred)
python -c "import pyarrow" 2>/dev/null || pip install -q pyarrow

echo "=== INLP Stage B capability [$TAG] ==="
python -m steering.run_inlp_stageb \
  --model "${MODEL:-Qwen/Qwen3.5-2B-Base}" \
  --device cuda \
  --dtype float32 \
  --tag "$TAG" \
  --log-every 100 \
  "$@"

OUT=/workspace/inlp_stage_b_${TAG}.tar.gz
tar -czf "$OUT" -C results/steering/inlp_stage_b "$TAG"
ls -lh "$OUT"
echo "DONE. Download: $OUT"
