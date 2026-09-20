#!/usr/bin/env bash
# Sync a Vast checkout, then run XY-control per-SOC Stage B/C.
#
# Existing Stage A results must remain under results/steering/xy_control/per_soc.
#   export HF_TOKEN=hf_xxx
#   SCALES=2b bash steering/xy_control/scripts/vast_xy_control_per_soc_stagebc_full_instance.sh
#   SCALES=4b bash steering/xy_control/scripts/vast_xy_control_per_soc_stagebc_full_instance.sh
#
# Env: SCALES=2b,4b BRANCH SKIP_PIP=1 SMOKE=1 CAP_LOSS_MAX=0.03
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?export HF_TOKEN=hf_xxx}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"

WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="${REPO_DIR:-occupational-gender-bias}"
REPO="${REPO:-$WORKDIR/$REPO_DIR}"
BRANCH="${BRANCH:-main}"
GH_REPO="${GH_REPO:-niczavadskiy/occupational-gender-bias}"
SCALES="${SCALES:-2b,4b}"

cd "$WORKDIR"
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

# Existing torch/transformers from Stage A are sufficient. Avoid CUDA source builds.
export SKIP_FLA="${SKIP_FLA:-1}"
PULL_CACHE=0 bash scripts/setup_instance.sh
# shellcheck disable=SC1091
source "$REPO/scripts/ensure_steering_env.sh"
SKIP_PIP=1 ensure_steering_env
export PY

"$PY" -m steering.xy_control.test_stagebc

IFS=',' read -r -a SCALE_LIST <<< "$SCALES"
for scale in "${SCALE_LIST[@]}"; do
  scale="$(echo "$scale" | xargs)"
  [ -n "$scale" ] || continue
  echo "=== Stage B/C $scale ==="
  PY="$PY" SCALE="$scale" REPO="$REPO" SMOKE="${SMOKE:-0}" \
    CAP_LOSS_MAX="${CAP_LOSS_MAX:-0.03}" \
    bash steering/xy_control/scripts/vast_xy_control_per_soc_stagebc_and_pack.sh
done

echo "=== ALL DONE ==="
ls -lh /workspace/xy_control_per_soc_stage_bc_*_v1*.tar.gz 2>/dev/null || true
