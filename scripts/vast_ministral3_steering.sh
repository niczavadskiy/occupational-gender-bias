#!/usr/bin/env bash
# Vast: Ministral 3 steering (3B READY, 8B classical needs LAYERS after probe;
#       8B pool protocols need H1 FDR polarity sets — skipped while STUB/empty).
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   # all three protocols for 3B (recommended)
#   SCALE=ministral3_3b PROTOCOLS=all SKIP_PIP=1 \
#     bash scripts/vast_ministral3_steering.sh
#
#   # classical only, both scales (8B needs layers from probe)
#   SCALE=both PROTOCOLS=classical SKIP_PIP=1 \
#     LAYERS_GENDER_8B=24,23,22 LAYERS_SLOT_8B=31,30,29 \
#     bash scripts/vast_ministral3_steering.sh
#
#   # smoke classical 3B
#   SCALE=ministral3_3b PROTOCOLS=classical SMOKE=1 SKIP_PIP=1 \
#     bash scripts/vast_ministral3_steering.sh
#
# Env: HF_TOKEN SCALE={ministral3_3b|ministral3_8b|both}
#      PROTOCOLS={all|classical|pool_inlp|pool_xy|classical,pool_inlp,...}
#      SKIP_PIP=1 PIN_VAST_ENV=1 SKIP_FLA=1 DTYPE=bfloat16 SMOKE=1
#      LAYERS_GENDER LAYERS_SLOT (3B defaults 23,22,21 / 24,23,22)
#      LAYERS_GENDER_8B LAYERS_SLOT_8B (required for 8B classical if not stub-ok)
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?export HF_TOKEN=hf_xxx}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"

WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="${REPO_DIR:-occupational-gender-bias}"
REPO="${REPO:-$WORKDIR/$REPO_DIR}"
BRANCH="${BRANCH:-main}"
GH_REPO="${GH_REPO:-niczavadskiy/occupational-gender-bias}"
SCALE="${SCALE:-ministral3_3b}"
PROTOCOLS="${PROTOCOLS:-all}"
DTYPE="${DTYPE:-bfloat16}"
SMOKE="${SMOKE:-0}"
SKIP_PIP="${SKIP_PIP:-0}"
export SKIP_FLA="${SKIP_FLA:-1}"
export PIN_VAST_ENV="${PIN_VAST_ENV:-1}"

# 3B locked peaks (gender_choice L23, slot_choice L24)
LAYERS_GENDER_3B="${LAYERS_GENDER:-${LAYERS_GENDER_3B:-23,22,21}}"
LAYERS_SLOT_3B="${LAYERS_SLOT:-${LAYERS_SLOT_3B:-24,23,22}}"
# 8B: fill after probe (placeholders match catalog stubs)
LAYERS_GENDER_8B="${LAYERS_GENDER_8B:-24,23,22}"
LAYERS_SLOT_8B="${LAYERS_SLOT_8B:-31,30,29}"

echo "=== Ministral 3 steering wrapper SCALE=$SCALE PROTOCOLS=$PROTOCOLS ==="
nvidia-smi -L || true
cd "$WORKDIR"

echo "=== [1] sync $GH_REPO @$BRANCH ==="
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" fetch --depth 1 origin "$BRANCH"
  git -C "$REPO_DIR" checkout -B "$BRANCH" "origin/$BRANCH"
  git -C "$REPO_DIR" reset --hard "origin/$BRANCH"
else
  git clone --depth 1 --branch "$BRANCH" "https://github.com/${GH_REPO}.git" "$REPO_DIR"
fi
export REPO
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

if [ -z "${PY:-}" ] && [ -x /opt/conda/bin/python ]; then
  export PY=/opt/conda/bin/python
fi

echo "=== [2] torch / steering env ==="
# shellcheck disable=SC1091
source "$REPO/scripts/ensure_steering_env.sh"
if [ "$SKIP_PIP" = "1" ]; then
  SKIP_PIP=1 ensure_steering_env || true
else
  ensure_steering_env || true
fi
export PY="${PY:-/opt/conda/bin/python}"
echo "  PY=$PY"
if ! "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"; then
  echo "  repairing torch via install_vast_env --force…"
  SKIP_FLA=1 PIN_VAST_ENV=1 "$PY" "$REPO/scripts/install_vast_env.py" --force
fi
"$PY" -c "import torch" || { echo "FATAL: no torch in $PY"; exit 1; }
"$PY" -m pip install -U 'transformers>=4.57.0' mistral-common
"$PY" -c "import torch" || { echo "FATAL: torch lost after mistral-common"; exit 1; }

expand_protocols() {
  local raw="$1"
  if [ "$raw" = "all" ]; then
    echo "classical pool_inlp pool_xy"
  else
    echo "${raw//,/ }"
  fi
}

scales=()
case "$SCALE" in
  ministral3_3b|3b) scales=(ministral3_3b) ;;
  ministral3_8b|8b) scales=(ministral3_8b) ;;
  both|all) scales=(ministral3_3b ministral3_8b) ;;
  *) echo "SCALE must be ministral3_3b|ministral3_8b|both (got $SCALE)"; exit 1 ;;
esac

run_one() {
  local scale="$1" proto="$2"
  local exp layers_g layers_s

  if [ "$scale" = "ministral3_3b" ]; then
    exp=experiments/ministral3-3b-base
    layers_g="$LAYERS_GENDER_3B"
    layers_s="$LAYERS_SLOT_3B"
  else
    exp=experiments/ministral3-8b-base
    layers_g="$LAYERS_GENDER_8B"
    layers_s="$LAYERS_SLOT_8B"
  fi

  echo ""
  echo "########## $scale · $proto ##########"

  case "$proto" in
    classical)
      LAYERS_GENDER="$layers_g" LAYERS_SLOT="$layers_s" \
        HF_TOKEN="$HF_TOKEN" SCALE="$scale" DTYPE="$DTYPE" SMOKE="$SMOKE" \
        SKIP_PIP=1 SKIP_FLA=1 PIN_VAST_ENV=1 PY="$PY" \
        bash "$exp/steering/scripts/vast_classical_inlp_abc_full_instance.sh"
      ;;
    pool_inlp)
      if [ "$scale" = "ministral3_8b" ]; then
        local sets_json="$exp/steering/domains/inlp_polarity_sets_${scale}_v1.json"
        if grep -q '"status": "STUB"' "$sets_json" 2>/dev/null || grep -q '"n_soc": 0' "$sets_json"; then
          # Still run: script skips empty sets; warn loudly.
          echo "WARN: $scale pool INLP polarity sets may be STUB/empty — will skip empty sets"
        fi
        HF_TOKEN="$HF_TOKEN" SCALE="$scale" DTYPE="$DTYPE" SMOKE="$SMOKE" \
          SKIP_PIP=1 SKIP_FLA=1 PIN_VAST_ENV=1 PY="$PY" \
          bash "$exp/steering/scripts/vast_inlp_polarity_pool_ministral3_8b_full_instance.sh"
      else
        HF_TOKEN="$HF_TOKEN" SCALE="$scale" DTYPE="$DTYPE" SMOKE="$SMOKE" \
          SKIP_PIP=1 SKIP_FLA=1 PIN_VAST_ENV=1 PY="$PY" \
          bash "$exp/steering/scripts/vast_inlp_polarity_pool_ministral3_3b_full_instance.sh"
      fi
      ;;
    pool_xy)
      if [ "$scale" = "ministral3_8b" ]; then
        echo "WARN: $scale pool XY polarity sets may be STUB/empty — will skip empty sets"
        HF_TOKEN="$HF_TOKEN" SCALE="$scale" DTYPE="$DTYPE" SMOKE="$SMOKE" \
          SKIP_PIP=1 SKIP_FLA=1 PIN_VAST_ENV=1 PY="$PY" \
          bash "$exp/steering/scripts/vast_xy_control_polarity_sets_ministral3_8b_full_instance.sh"
      else
        HF_TOKEN="$HF_TOKEN" SCALE="$scale" DTYPE="$DTYPE" SMOKE="$SMOKE" \
          SKIP_PIP=1 SKIP_FLA=1 PIN_VAST_ENV=1 PY="$PY" \
          bash "$exp/steering/scripts/vast_xy_control_polarity_sets_ministral3_3b_full_instance.sh"
      fi
      ;;
    *)
      echo "unknown protocol: $proto (classical|pool_inlp|pool_xy)"; exit 1
      ;;
  esac
}

for sc in "${scales[@]}"; do
  # shellcheck disable=SC2046
  for pr in $(expand_protocols "$PROTOCOLS"); do
    run_one "$sc" "$pr"
  done
done

echo ""
echo "=== Ministral 3 steering wrapper DONE ==="
ls -lh /workspace/inlp_classical_ministral3_*_abc*.tar.gz 2>/dev/null || true
ls -lh /workspace/*polarity*ministral3*.tar.gz 2>/dev/null || true
