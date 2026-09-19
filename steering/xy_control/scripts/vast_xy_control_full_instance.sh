#!/usr/bin/env bash
# Fresh Vast → clone → smoke → full XY-control Stage A → pack.
#
#   export HF_TOKEN=hf_xxx
#   bash steering/xy_control/scripts/vast_xy_control_full_instance.sh
#
# Env: SKIP_SMOKE=1  SKIP_FULL=1  BRANCH TAG MODEL SCALE MMLU
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
SCALE="${SCALE:-2b}"
if [ "$SCALE" = "4b" ]; then
  MODEL="${MODEL:-Qwen/Qwen3.5-4B-Base}"
else
  MODEL="${MODEL:-Qwen/Qwen3.5-2B-Base}"
fi
TAG="${TAG:-xy_${SCALE}_a_v1}"

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
# SKIP_PIP=1 if torch/transformers already good (avoids causal-conv1d source build).
PULL_CACHE=0 bash scripts/setup_instance.sh
# shellcheck disable=SC1091
source "$REPO/scripts/ensure_steering_env.sh"
SKIP_PIP=1 ensure_steering_env
export PY
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "=== [3] artifact check ==="
need=(
  steering/xy_control/build_dataset.py
  steering/xy_control/build_vectors.py
  steering/xy_control/run_stagea.py
  steering/xy_control/scripts/vast_xy_control_stagea_and_pack.sh
  steering/samples/h1_stagea_sample_v1.json
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

chmod +x steering/xy_control/scripts/vast_xy_control_stagea_and_pack.sh

if [ "$SKIP_SMOKE" != "1" ]; then
  echo "=== [4] SMOKE (3 families, ±α on anchor) ==="
  PY="$PY" SMOKE=1 SCALE="$SCALE" TAG="$TAG" MODEL="$MODEL" REPO="$REPO" \
    bash steering/xy_control/scripts/vast_xy_control_stagea_and_pack.sh
fi

if [ "$SKIP_FULL" != "1" ]; then
  echo "=== [5] FULL ==="
  PY="$PY" SMOKE=0 SCALE="$SCALE" TAG="$TAG" MODEL="$MODEL" REPO="$REPO" \
    MMLU="${MMLU:-none}" \
    bash steering/xy_control/scripts/vast_xy_control_stagea_and_pack.sh
fi

echo "=== ALL DONE ==="
ls -lh /workspace/xy_control_stage_a_*.tar.gz 2>/dev/null || true
echo "Скачай pack: /workspace/xy_control_stage_a_${TAG}.tar.gz"
