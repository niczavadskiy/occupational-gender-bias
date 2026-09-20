#!/usr/bin/env bash
# XY-control Stage A + pack.
#
#   export HF_TOKEN=hf_xxx
#   bash steering/xy_control/scripts/vast_xy_control_stagea_and_pack.sh
#
# Env: SCALE=2b|4b  SMOKE=1  TAG MODEL N_ITEMS DEVICE DTYPE MMLU=none|smoke
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
XY="$(cd "$SCRIPT_DIR/.." && pwd)"
STEER="$(cd "$XY/.." && pwd)"
REPO="${REPO:-$(cd "$STEER/.." && pwd)}"

SCALE="${SCALE:-2b}"
if [ "$SCALE" = "4b" ]; then
  MODEL="${MODEL:-Qwen/Qwen3.5-4B-Base}"
  ANCHOR_CAND="main_add_core__vraw__L23__add__a1"
  ANTI_CAND="main_add_core__vraw__L23__add__am1"
else
  MODEL="${MODEL:-Qwen/Qwen3.5-2B-Base}"
  ANCHOR_CAND="main_add_core__vraw__L16__add__a1"
  ANTI_CAND="main_add_core__vraw__L16__add__am1"
fi
TAG="${TAG:-xy_${SCALE}_a_v1}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"
SMOKE="${SMOKE:-0}"
N_ITEMS="${N_ITEMS:-}"
MMLU="${MMLU:-none}"

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

VEC_FLAGS=()
STAGE_FLAGS=()
if [ "$SMOKE" = "1" ]; then
  N_ITEMS="${N_ITEMS:-3}"
  TAG="${TAG}_smoke"
  STAGE_FLAGS+=(--limit-items "$N_ITEMS" --eval-split all --candidates "$ANCHOR_CAND,$ANTI_CAND")
fi
if [ -n "$N_ITEMS" ] && [ "$SMOKE" != "1" ]; then
  VEC_FLAGS+=(--n-items "$N_ITEMS")
  STAGE_FLAGS+=(--limit-items "$N_ITEMS")
fi
if [ "$SMOKE" = "1" ]; then
  VEC_FLAGS+=(--n-items "$N_ITEMS")
fi
if [ "$MMLU" != "none" ]; then
  STAGE_FLAGS+=(--mmlu "$MMLU")
fi
if [ "$SMOKE" != "1" ]; then
  STAGE_FLAGS+=(--eval-split "${EVAL_SPLIT:-val}")
fi

echo "=== xy_control dataset ==="
"$PY" -m steering.xy_control.build_dataset

echo "=== xy_control candidates [$SCALE] ==="
"$PY" -m steering.xy_control.build_candidates --scale "$SCALE"

echo "=== xy_control vectors [$SCALE] model=$MODEL ==="
"$PY" -m steering.xy_control.build_vectors \
  --scale "$SCALE" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  "${VEC_FLAGS[@]}"

echo "=== xy_control Stage A [$TAG] ==="
"$PY" -m steering.xy_control.run_stagea \
  --scale "$SCALE" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --tag "$TAG" \
  "${STAGE_FLAGS[@]}" \
  "$@"

OUT_ROOT="$REPO/results/steering/xy_control"
OUT="/workspace/xy_control_stage_a_${TAG}.tar.gz"
VEC_NPZ="$XY/vectors/xy_control_${SCALE}_vectors_${VEC_TAG:-v1}.npz"
VEC_JSON="$XY/vectors/xy_control_${SCALE}_vectors_${VEC_TAG:-v1}.json"
tar_args=( -czf "$OUT" -C "$REPO" "results/steering/xy_control/stage_a/$TAG" )
if [ -f "$VEC_NPZ" ]; then
  tar_args+=( "steering/xy_control/vectors/xy_control_${SCALE}_vectors_${VEC_TAG:-v1}.npz" )
fi
if [ -f "$VEC_JSON" ]; then
  tar_args+=( "steering/xy_control/vectors/xy_control_${SCALE}_vectors_${VEC_TAG:-v1}.json" )
fi
tar "${tar_args[@]}"
ls -lh "$OUT"
echo "DONE. Download: $OUT"
