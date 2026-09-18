#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# sequential_pre_peak_erase on Vast (gender and/or slot) + pack.
#
#   export HF_TOKEN=hf_xxx
#   bash steering/scripts/vast_sequential_pre_peak_and_pack.sh
#
# Env: AXIS=gender|slot|both  SMOKE TAG MODEL REPO
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="${REPO:-/workspace/occupational-gender-bias}"
MODEL="${MODEL:-Qwen/Qwen3.5-2B-Base}"
AXIS="${AXIS:-both}"
TAG="${TAG:-prepeak_v1}"
SMOKE="${SMOKE:-0}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"

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

echo "=== deps (PY=$PY) ==="
if ! "$PY" -c "import numpy,torch,transformers; assert int(numpy.__version__.split('.')[0]) < 2" 2>/dev/null; then
  # shellcheck disable=SC1091
  source "$REPO/scripts/ensure_steering_env.sh"
  ensure_steering_env
  PY="$(cat /tmp/occupational_steering_py)"
  export PY
fi
"$PY" -c "import sklearn" 2>/dev/null || "$PY" -m pip install -q scikit-learn || true
"$PY" -c "import yaml" 2>/dev/null || "$PY" -m pip install -q pyyaml || true

EXTRA=()
if [ "$SMOKE" = "1" ]; then
  TAG="${TAG}_smoke"
  EXTRA+=(--n-items 2 --max-hits 2)
  echo "=== SMOKE ==="
fi

run_axis() {
  local axis="$1"
  local cfg="$REPO/steering/configs/sequential_pre_peak_erase_${axis}_2b_v1.yaml"
  local atag="${TAG}_${axis}"
  test -f "$cfg" || { echo "MISSING $cfg"; exit 1; }
  echo "========== pre-peak $axis [$atag] =========="
  "$PY" -m steering.run_sequential_pre_peak_erase \
    --config "$cfg" \
    --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" \
    --out-root "$REPO/results/steering" --tag "$atag" --log-every 20 \
    "${EXTRA[@]}"
}

case "$AXIS" in
  gender) run_axis gender ;;
  slot) run_axis slot ;;
  both) run_axis gender; run_axis slot ;;
  *) echo "AXIS must be gender|slot|both"; exit 1 ;;
esac

OUT="/workspace/sequential_pre_peak_${TAG}.tar.gz"
tar -czf "$OUT" -C "$REPO/results/steering" sequential_pre_peak
ls -lh "$OUT"
echo "DONE $OUT"
