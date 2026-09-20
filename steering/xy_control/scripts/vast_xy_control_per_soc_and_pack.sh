#!/usr/bin/env bash
# Per-SOC XY-control: fit v and steer each FDR-significant domain, then pack.
#
#   SCALE=2b bash steering/xy_control/scripts/vast_xy_control_per_soc_and_pack.sh
#
# Env: SCALE=2b|4b  SMOKE=1  TAG MODEL DEVICE DTYPE EVAL_SPLIT=val  ONLY_SOC
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
TAG="${TAG:-xy_${SCALE}_per_soc_v1}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"
SMOKE="${SMOKE:-0}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

if [ -z "${PY:-}" ]; then
  if [ -f /tmp/occupational_steering_py ]; then
    PY="$(cat /tmp/occupational_steering_py)"
  fi
fi
if [ -z "${PY:-}" ] || { [ ! -x "$PY" ] && ! command -v "$PY" >/dev/null 2>&1; }; then
  for c in /venv/main/bin/python /opt/conda/bin/python python3 python; do
    if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then
      PY=$c
      break
    fi
  done
fi
export PY

if [ -f "$REPO/scripts/ensure_steering_env.sh" ]; then
  if ! "$PY" -c "import numpy,torch,transformers" 2>/dev/null; then
    # shellcheck disable=SC1091
    source "$REPO/scripts/ensure_steering_env.sh"
    ensure_steering_env
    PY="$(cat /tmp/occupational_steering_py)"
    export PY
  fi
fi

STAGE_FLAGS=(--scale "$SCALE" --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE")
if [ "$SMOKE" = "1" ]; then
  TAG="${TAG}_smoke"
  STAGE_FLAGS+=(--smoke --limit-items "${N_ITEMS:-3}" --max-domains "${MAX_DOMAINS:-1}")
else
  STAGE_FLAGS+=(--eval-split "$EVAL_SPLIT")
fi
STAGE_FLAGS+=(--tag "$TAG")
if [ -n "${ONLY_SOC:-}" ]; then
  STAGE_FLAGS+=(--only-soc "$ONLY_SOC")
fi

echo "=== xy_control full without_abstain dataset ==="
"$PY" -m steering.xy_control.build_dataset
"$PY" -m steering.xy_control.test_dataset

echo "=== per-SOC domain catalog verify ==="
"$PY" -m steering.xy_control.build_domains --verify
"$PY" -m steering.xy_control.test_domains
"$PY" -m steering.xy_control.test_per_soc

echo "=== per-SOC XY-control [$SCALE] tag=$TAG eval=$EVAL_SPLIT ==="
"$PY" -m steering.xy_control.run_per_soc "${STAGE_FLAGS[@]}" "$@"

OUT_ROOT="$REPO/results/steering/xy_control"
OUT="/workspace/xy_control_per_soc_${TAG}.tar.gz"
VEC_NPZ="$XY/vectors/xy_control_${SCALE}_per_soc_vectors_${TAG}.npz"
VEC_JSON="$XY/vectors/xy_control_${SCALE}_per_soc_vectors_${TAG}.json"
tar_args=( -czf "$OUT" -C "$REPO" "results/steering/xy_control/per_soc/$TAG" )
if [ -f "$VEC_NPZ" ]; then
  tar_args+=( "steering/xy_control/vectors/xy_control_${SCALE}_per_soc_vectors_${TAG}.npz" )
fi
if [ -f "$VEC_JSON" ]; then
  tar_args+=( "steering/xy_control/vectors/xy_control_${SCALE}_per_soc_vectors_${TAG}.json" )
fi
tar "${tar_args[@]}"
ls -lh "$OUT"
echo "DONE. Download: $OUT"
