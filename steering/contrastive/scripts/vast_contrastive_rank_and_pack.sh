#!/usr/bin/env bash
# Live capture → contrastive v_G + stratified rank PCA (no Stage A preference grid).
#
#   export HF_TOKEN=hf_xxx
#   bash steering/contrastive/scripts/vast_contrastive_rank_and_pack.sh
#
# Env: SCALE=2b|4b  SMOKE=1  RUN_PCA_STAGEA=1  TAG MODEL N_ITEMS DEVICE DTYPE
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTRASTIVE="$(cd "$SCRIPT_DIR/.." && pwd)"
STEER="$(cd "$CONTRASTIVE/.." && pwd)"
REPO="${REPO:-$(cd "$STEER/.." && pwd)}"

SCALE="${SCALE:-2b}"
if [ "$SCALE" = "4b" ]; then
  MODEL="${MODEL:-Qwen/Qwen3.5-4B-Base}"
else
  MODEL="${MODEL:-Qwen/Qwen3.5-2B-Base}"
fi
TAG="${TAG:-v1}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"
SMOKE="${SMOKE:-0}"
N_ITEMS="${N_ITEMS:-}"

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
if [ "$SMOKE" = "1" ]; then
  N_ITEMS="${N_ITEMS:-8}"
  TAG="${TAG}_smoke"
fi
if [ -n "$N_ITEMS" ]; then
  VEC_FLAGS+=(--n-items "$N_ITEMS")
fi

echo "=== contrastive vectors [$SCALE] ==="
"$PY" -m steering.contrastive.build_vectors \
  --scale "$SCALE" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --tag "$TAG" \
  "${VEC_FLAGS[@]}"

echo "=== contrastive rank PCA [$SCALE] ==="
"$PY" -m steering.contrastive.analyze_rank \
  --scale "$SCALE" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --tag "$TAG" \
  "${VEC_FLAGS[@]}"

if [ "${RUN_PCA_STAGEA:-0}" = "1" ]; then
  echo "=== PCA Stage A ==="
  "$PY" -m steering.contrastive.run_pca_stagea \
    --scale "$SCALE" \
    --model "$MODEL" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --subspaces "$REPO/steering/subspaces/contrastive_pca_${SCALE}_${TAG}.npz" \
    --tag "pca_${SCALE}_a_${TAG}"
fi

OUT="/workspace/contrastive_rank_${SCALE}_${TAG}.tar.gz"
tar -czf "$OUT" -C "$REPO" \
  steering/contrastive/rank \
  steering/subspaces/contrastive_pca_${SCALE}_${TAG}.npz \
  steering/subspaces/contrastive_pca_${SCALE}_${TAG}.json
ls -lh "$OUT"
echo "DONE. Download: $OUT"
