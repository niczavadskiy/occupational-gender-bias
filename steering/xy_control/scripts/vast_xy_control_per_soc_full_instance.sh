#!/usr/bin/env bash
# Fresh Vast → clone → per-SOC XY-control for 2B then 4B → pack each.
#
#   export HF_TOKEN=hf_xxx
#   bash steering/xy_control/scripts/vast_xy_control_per_soc_full_instance.sh
#
# Env: SKIP_SMOKE=1  SKIP_FULL=1  BRANCH SCALES=2b,4b  SKIP_PIP=1  EVAL_SPLIT=val
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?export HF_TOKEN=hf_xxx}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"

WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="${REPO_DIR:-occupational-gender-bias}"
REPO="${REPO:-$WORKDIR/$REPO_DIR}"
BRANCH="${BRANCH:-main}"
GH_REPO="${GH_REPO:-niczavadskiy/occupational-gender-bias}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SKIP_FULL="${SKIP_FULL:-0}"
SCALES="${SCALES:-2b,4b}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"

echo "=== [0] cwd / cuda ==="
nvidia-smi -L || echo "WARN: nvidia-smi failed"
cd "$WORKDIR"

echo "=== [1] clone / sync $GH_REPO @$BRANCH ==="
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" fetch --depth 1 origin "$BRANCH"
  git -C "$REPO_DIR" checkout "$BRANCH"
  git -C "$REPO_DIR" reset --hard "origin/$BRANCH"
else
  git clone --depth 1 --branch "$BRANCH" "https://github.com/${GH_REPO}.git" "$REPO_DIR"
fi
export REPO
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "=== [2] setup_instance + steering env ==="
# SKIP_PIP=1 if torch/transformers already good.
# SKIP_FLA=1 skips flash-linear-attention + causal-conv1d pip (source builds hang).
export SKIP_FLA="${SKIP_FLA:-1}"
PULL_CACHE=0 bash scripts/setup_instance.sh
# shellcheck disable=SC1091
source "$REPO/scripts/ensure_steering_env.sh"
SKIP_PIP=1 ensure_steering_env
export PY
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "=== [3] artifact check ==="
need=(
  steering/xy_control/run_per_soc.py
  steering/xy_control/build_vectors_per_soc.py
  steering/xy_control/h1_families.py
  steering/xy_control/per_soc.py
  steering/xy_control/domains/h1_soc_fdr_v1.json
  data/v1/inference_items_v1_man_first.jsonl
  data/v1/inference_items_v1_woman_first.jsonl
  steering/xy_control/scripts/vast_xy_control_per_soc_and_pack.sh
  steering/xy_control/configs/xy_control_2b_v1.yaml
  steering/xy_control/configs/xy_control_4b_v1.yaml
)
miss=0
for f in "${need[@]}"; do
  if [ -f "$f" ]; then
    ls -lh "$f" | awk '{print "  OK", $5, $9}'
  else
    echo "  MISSING $f"
    miss=1
  fi
done
if [ "$miss" = "1" ]; then
  echo "Файлы не в origin/$BRANCH — git pull после push каталога."
  exit 1
fi

chmod +x steering/xy_control/scripts/vast_xy_control_per_soc_and_pack.sh
chmod +x steering/xy_control/scripts/vast_xy_control_per_soc_full_instance.sh
chmod +x steering/xy_control/scripts/vast_xy_control_per_soc_4b_full_instance.sh 2>/dev/null || true

IFS=',' read -r -a SCALE_LIST <<< "$SCALES"

run_one() {
  local scale="$1"
  local smoke="$2"
  local model
  if [ "$scale" = "4b" ]; then
    model="${MODEL_4B:-Qwen/Qwen3.5-4B-Base}"
  else
    model="${MODEL_2B:-Qwen/Qwen3.5-2B-Base}"
  fi
  PY="$PY" SMOKE="$smoke" SCALE="$scale" TAG="xy_${scale}_per_soc_v1" MODEL="$model" \
    EVAL_SPLIT="$EVAL_SPLIT" REPO="$REPO" \
    bash steering/xy_control/scripts/vast_xy_control_per_soc_and_pack.sh
}

if [ "$SKIP_SMOKE" != "1" ]; then
  echo "=== [4] SMOKE (1 domain × ±α on anchor, each scale) ==="
  for scale in "${SCALE_LIST[@]}"; do
    scale="$(echo "$scale" | xargs)"
    [ -n "$scale" ] || continue
    run_one "$scale" 1
  done
fi

if [ "$SKIP_FULL" != "1" ]; then
  echo "=== [5] FULL per-SOC (significant domains, eval=$EVAL_SPLIT) ==="
  for scale in "${SCALE_LIST[@]}"; do
    scale="$(echo "$scale" | xargs)"
    [ -n "$scale" ] || continue
    run_one "$scale" 0
  done
fi

echo "=== ALL DONE ==="
ls -lh /workspace/xy_control_per_soc_*.tar.gz 2>/dev/null || true
echo "Скачай packs: /workspace/xy_control_per_soc_xy_<scale>_per_soc_v1.tar.gz"
