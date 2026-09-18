#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Classical gender_prob INLP on Vast: live Phase A + Stage A + pack.
#
#   export HF_TOKEN=hf_xxx
#   cd /workspace/occupational-gender-bias && git pull
#   bash steering/scripts/vast_inlp_gender_prob_stagea_and_pack.sh
#
# Env: MODEL TAG LAYERS RANKS ALPHAS SMOKE SKIP_BUILD REPO
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_FROM_SCRIPT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -n "${REPO:-}" ]; then
  :
elif [ -f "$REPO_FROM_SCRIPT/steering/build_inlp_live_subspace.py" ]; then
  REPO="$REPO_FROM_SCRIPT"
elif [ -f /workspace/occupational-gender-bias/steering/build_inlp_live_subspace.py ]; then
  REPO=/workspace/occupational-gender-bias
elif [ -f /workspace/occupational-gender-bias/occupational-gender-bias/steering/build_inlp_live_subspace.py ]; then
  REPO=/workspace/occupational-gender-bias/occupational-gender-bias
  echo "WARN: nested clone → REPO=$REPO"
else
  REPO="$REPO_FROM_SCRIPT"
fi

MODEL="${MODEL:-Qwen/Qwen3.5-2B-Base}"
TAG="${TAG:-inlp_gender_prob_a_v1}"
SUB_TAG="${SUB_TAG:-v1}"
LAYERS="${LAYERS:-14,15,16}"
RANKS="${RANKS:-4,8,16}"
ALPHAS="${ALPHAS:-1.0}"
SMOKE="${SMOKE:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"
K_MAX="${K_MAX:-32}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
echo "REPO=$REPO"

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

SAMPLE="$REPO/steering/samples/h1_stagea_sample_v1.json"
SUB="$REPO/steering/subspaces/inlp_gender_prob_${SUB_TAG}.npz"
test -f "$SAMPLE" || { echo "MISSING $SAMPLE"; exit 1; }

N_ITEMS_ARGS=()
STAGEA_LIMIT=()
if [ "$SMOKE" = "1" ]; then
  N_ITEMS_ARGS=(--n-items 2)
  STAGEA_LIMIT=(--limit-items 2)
  LAYERS="${LAYERS:-15}"
  RANKS="${RANKS:-4}"
  K_MAX=4
  echo "SMOKE=1 → n_items=2 layers=$LAYERS ranks=$RANKS k_max=$K_MAX"
fi

if [ "$SKIP_BUILD" != "1" ]; then
  echo "========== Phase A live INLP gender_prob [$SUB_TAG] =========="
  "$PY" -m steering.build_inlp_live_subspace \
    --model "$MODEL" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --sample "$SAMPLE" \
    --target gender_prob \
    --layers "$LAYERS" \
    --k-max "$K_MAX" \
    --ridge-alpha 1.0 \
    --tag "$SUB_TAG" \
    --out-dir "$REPO/steering/subspaces" \
    --log-every 40 \
    "${N_ITEMS_ARGS[@]}"
else
  echo "SKIP_BUILD=1 → reuse $SUB"
fi

test -f "$SUB" || { echo "MISSING $SUB — run without SKIP_BUILD=1"; exit 1; }

echo "========== Stage A preference [$TAG] =========="
"$PY" -m steering.run_inlp_stagea \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --subspaces "$SUB" \
  --sample "$SAMPLE" \
  --layers "$LAYERS" \
  --ranks "$RANKS" \
  --alphas "$ALPHAS" \
  --primary-axis gender \
  --tag "$TAG" \
  --log-every 100 \
  "${STAGEA_LIMIT[@]}"

OUT="/workspace/inlp_gender_prob_${TAG}.tar.gz"
tar -czf "$OUT" \
  -C "$REPO/results/steering/inlp_stage_a" "$TAG" \
  -C "$REPO/steering/subspaces" \
  "inlp_gender_prob_${SUB_TAG}.npz" "inlp_gender_prob_${SUB_TAG}.json"
ls -lh "$OUT"
echo "DONE $OUT"
echo "Compare R to classical gender_choice L15 k16 (~0.03)."
