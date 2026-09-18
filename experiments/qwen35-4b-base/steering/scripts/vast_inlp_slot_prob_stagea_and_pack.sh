#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 4B classical peak slot_prob: live Phase A + Stage A + pack.
#
#   bash experiments/qwen35-4b-base/steering/scripts/vast_inlp_slot_prob_stagea_and_pack.sh
# ---------------------------------------------------------------------------
set -euo pipefail

STEER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$STEER_DIR/../../.." && pwd)"
MODEL="${MODEL:-Qwen/Qwen3.5-4B-Base}"
TAG="${TAG:-inlp_slot_prob_4b_a_v1}"
SUB_TAG="${SUB_TAG:-4b_v1}"
LAYERS="${LAYERS:-29,30,31}"
RANKS="${RANKS:-4,8,16}"
ALPHAS="${ALPHAS:-1.0}"
SMOKE="${SMOKE:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"
K_MAX="${K_MAX:-32}"
OUT_ROOT="${OUT_ROOT:-$STEER_DIR/../results/steering}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
echo "REPO=$REPO  STEER=$STEER_DIR  MODEL=$MODEL"

if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

if [ -z "${PY:-}" ]; then
  [ -f /tmp/occupational_steering_py ] && PY="$(cat /tmp/occupational_steering_py)"
fi
if [ -z "${PY:-}" ] || { [ ! -x "$PY" ] && ! command -v "$PY" >/dev/null 2>&1; }; then
  for c in /venv/main/bin/python /opt/conda/bin/python python3 python; do
    if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then PY=$c; break; fi
  done
fi
export PY

echo "=== deps (PY=$PY) ==="
if ! "$PY" -c "import numpy,torch,transformers; assert int(numpy.__version__.split('.')[0]) < 2" 2>/dev/null; then
  # shellcheck disable=SC1091
  source "$REPO/scripts/ensure_steering_env.sh"
  ensure_steering_env
  PY="$(cat /tmp/occupational_steering_py)"; export PY
fi
"$PY" -c "import sklearn" 2>/dev/null || "$PY" -m pip install -q scikit-learn || true

SAMPLE="$STEER_DIR/samples/h1_stagea_sample_v1.json"
SUB_DIR="$STEER_DIR/subspaces"
mkdir -p "$SUB_DIR" "$OUT_ROOT"
SUB="$SUB_DIR/inlp_slot_prob_${SUB_TAG}.npz"
test -f "$SAMPLE" || { echo "MISSING $SAMPLE"; exit 1; }

N_ITEMS_ARGS=(); STAGEA_LIMIT=()
if [ "$SMOKE" = "1" ]; then
  N_ITEMS_ARGS=(--n-items 2); STAGEA_LIMIT=(--limit-items 2)
  LAYERS=30; RANKS=4; K_MAX=4
  echo "SMOKE=1 → layers=$LAYERS ranks=$RANKS"
fi

if [ "$SKIP_BUILD" != "1" ]; then
  echo "========== Phase A live slot_prob 4B [$SUB_TAG] L=$LAYERS =========="
  "$PY" -m steering.build_inlp_live_subspace \
    --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" \
    --sample "$SAMPLE" --target slot_prob \
    --layers "$LAYERS" --k-max "$K_MAX" --ridge-alpha 1.0 \
    --tag "$SUB_TAG" --out-dir "$SUB_DIR" --log-every 40 \
    "${N_ITEMS_ARGS[@]}"
else
  echo "SKIP_BUILD=1 → reuse $SUB"
fi
test -f "$SUB" || { echo "MISSING $SUB"; exit 1; }

echo "========== Stage A [$TAG] primary=slot =========="
"$PY" -m steering.run_inlp_stagea \
  --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" \
  --subspaces "$SUB" --sample "$SAMPLE" \
  --layers "$LAYERS" --ranks "$RANKS" --alphas "$ALPHAS" \
  --primary-axis slot --out-root "$OUT_ROOT" --tag "$TAG" --log-every 100 \
  "${STAGEA_LIMIT[@]}"

OUT="/workspace/inlp_slot_prob_${TAG}.tar.gz"
tar -czf "$OUT" -C "$OUT_ROOT/inlp_stage_a" "$TAG" \
  -C "$SUB_DIR" "inlp_slot_prob_${SUB_TAG}.npz" "inlp_slot_prob_${SUB_TAG}.json"
ls -lh "$OUT"; echo "DONE $OUT"
