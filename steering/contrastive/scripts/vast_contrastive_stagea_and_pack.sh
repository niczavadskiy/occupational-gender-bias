#!/usr/bin/env bash
# Contrastive mean-diff Stage A + pack.
#
#   export HF_TOKEN=hf_xxx
#   bash steering/contrastive/scripts/vast_contrastive_stagea_and_pack.sh
#
# Env: SCALE=2b|4b  SMOKE=1  TAG MODEL N_ITEMS DEVICE DTYPE
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTRASTIVE="$(cd "$SCRIPT_DIR/.." && pwd)"
STEER="$(cd "$CONTRASTIVE/.." && pwd)"
REPO="${REPO:-$(cd "$STEER/.." && pwd)}"

SCALE="${SCALE:-2b}"
if [ "$SCALE" = "4b" ]; then
  MODEL="${MODEL:-Qwen/Qwen3.5-4B-Base}"
  ANCHOR_CAND="main_center_core__vmd__L23__center__a1"
  ANTI_CAND="anti_steering__vmd__L23__center__am1"
else
  MODEL="${MODEL:-Qwen/Qwen3.5-2B-Base}"
  ANCHOR_CAND="main_center_core__vmd__L16__center__a1"
  ANTI_CAND="anti_steering__vmd__L16__center__am1"
fi
TAG="${TAG:-contrastive_${SCALE}_a_v1}"
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
STAGE_FLAGS=()
if [ "$SMOKE" = "1" ]; then
  N_ITEMS="${N_ITEMS:-3}"
  TAG="${TAG}_smoke"
  STAGE_FLAGS+=(--limit-items "$N_ITEMS" --candidates "$ANCHOR_CAND,$ANTI_CAND")
fi
if [ -n "$N_ITEMS" ] && [ "$SMOKE" != "1" ]; then
  VEC_FLAGS+=(--n-items "$N_ITEMS")
  STAGE_FLAGS+=(--limit-items "$N_ITEMS")
fi
if [ "$SMOKE" = "1" ]; then
  VEC_FLAGS+=(--n-items "$N_ITEMS")
fi

echo "=== contrastive candidates [$SCALE] ==="
"$PY" -m steering.contrastive.build_candidates --scale "$SCALE"

echo "=== contrastive vectors [$SCALE] model=$MODEL ==="
"$PY" -m steering.contrastive.build_vectors \
  --scale "$SCALE" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  "${VEC_FLAGS[@]}"

echo "=== contrastive Stage A [$TAG] ==="
"$PY" -m steering.contrastive.run_stagea \
  --scale "$SCALE" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --tag "$TAG" \
  "${STAGE_FLAGS[@]}" \
  "$@"

OUT_ROOT="$REPO/results/steering/contrastive"
OUT="/workspace/contrastive_stage_a_${TAG}.tar.gz"
tar -czf "$OUT" -C "$OUT_ROOT/stage_a" "$TAG"
ls -lh "$OUT"
echo "DONE. Download: $OUT"
