#!/usr/bin/env bash
# Run per-SOC XY-control Stage B then C for one model and pack both.
#
# Requires the completed Stage A directory and vectors on the same Vast instance.
#   SCALE=2b bash steering/xy_control/scripts/vast_xy_control_per_soc_stagebc_and_pack.sh
#
# Env: SCALE=2b|4b DEVICE DTYPE CAP_LOSS_MAX=0.03 SMOKE=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
XY="$(cd "$SCRIPT_DIR/.." && pwd)"
STEER="$(cd "$XY/.." && pwd)"
REPO="${REPO:-$(cd "$STEER/.." && pwd)}"

SCALE="${SCALE:-2b}"
if [ "$SCALE" = "4b" ]; then
  MODEL="${MODEL:-Qwen/Qwen3.5-4B-Base}"
else
  MODEL="${MODEL:-Qwen/Qwen3.5-2B-Base}"
fi
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"
CAP_LOSS_MAX="${CAP_LOSS_MAX:-0.03}"
SMOKE="${SMOKE:-0}"
STAGE_A_TAG="${STAGE_A_TAG:-xy_${SCALE}_per_soc_v1}"
B_TAG="${B_TAG:-xy_${SCALE}_per_soc_b_v1}"
C_TAG="${C_TAG:-xy_${SCALE}_per_soc_c_v1}"
if [ "$SMOKE" = "1" ]; then
  B_TAG="${B_TAG}_smoke"
  C_TAG="${C_TAG}_smoke"
fi
STAGE_A_DIR="${STAGE_A_DIR:-$REPO/results/steering/xy_control/per_soc/$STAGE_A_TAG}"
MMLU_PARQUET="${MMLU_PARQUET:-$REPO/steering/.cache/mmlu_pro_test.parquet}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
if [ -z "${PY:-}" ] && [ -f /tmp/occupational_steering_py ]; then
  PY="$(cat /tmp/occupational_steering_py)"
fi
PY="${PY:-python3}"
export PY

if [ ! -f "$STAGE_A_DIR/summary.csv" ]; then
  echo "Missing Stage A: $STAGE_A_DIR/summary.csv"
  echo "Run/download per-SOC Stage A first."
  exit 1
fi

mkdir -p "$(dirname "$MMLU_PARQUET")"
if [ ! -f "$MMLU_PARQUET" ]; then
  echo "=== download frozen MMLU-Pro test parquet ==="
  curl -L --fail --retry 3 -o "$MMLU_PARQUET" \
    "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"
fi
if ! "$PY" -c "import pyarrow" >/dev/null 2>&1; then
  echo "=== install pyarrow wheel for parquet ==="
  "$PY" -m pip install -U pyarrow
fi

B_FLAGS=(
  --scale "$SCALE" --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE"
  --stage-a-dir "$STAGE_A_DIR" --mmlu-parquet "$MMLU_PARQUET"
  --cap-loss-max "$CAP_LOSS_MAX" --tag "$B_TAG"
)
C_FLAGS=(
  --scale "$SCALE" --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE"
  --stage-a-dir "$STAGE_A_DIR" --mmlu-parquet "$MMLU_PARQUET"
  --stagec-keep "$REPO/results/steering/xy_control/stage_b/$B_TAG/stagec_keep.json"
  --tag "$C_TAG"
)
if [ "$SMOKE" = "1" ]; then
  B_FLAGS+=(--limit-mmlu 3)
  C_FLAGS+=(--limit-mmlu 3 --limit-items 2)
fi

echo "=== XY-control per-SOC Stage B [$SCALE] ==="
"$PY" -m steering.xy_control.run_per_soc_stageb "${B_FLAGS[@]}"

echo "=== XY-control per-SOC Stage C [$SCALE] ==="
"$PY" -m steering.xy_control.run_per_soc_stagec "${C_FLAGS[@]}"

OUT="/workspace/xy_control_per_soc_stage_bc_${SCALE}_v1.tar.gz"
if [ "$SMOKE" = "1" ]; then
  OUT="/workspace/xy_control_per_soc_stage_bc_${SCALE}_v1_smoke.tar.gz"
fi
tar -czf "$OUT" -C "$REPO" \
  "results/steering/xy_control/stage_b/$B_TAG" \
  "results/steering/xy_control/stage_c/$C_TAG"
ls -lh "$OUT"
echo "DONE. Download: $OUT"
