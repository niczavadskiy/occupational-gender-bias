#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-command setup на свежем Vast-инстансе (образ bias-subspaces-env).
#
#   1. git clone публичного репо с кодом + data/v1
#   2. опционально: download старого 2B-прогона с HF (PULL_CACHE=1)
#   3. ensure_steering_env: numpy<2 + transformers + sanity torch/CUDA
#
# Использование (на инстансе):
#   export HF_TOKEN=hf_xxx
#   bash scripts/setup_instance.sh
#
# Env: BRANCH WORKDIR GH_REPO REPO_DIR PULL_CACHE PY UPGRADE_TORCH SKIP_PIP
#      REPO after setup = absolute path to checkout (for ensure_steering_env)
#
# После setup: PY в /tmp/occupational_steering_py (numpy<2 + transformers).
# ---------------------------------------------------------------------------
set -euo pipefail

BRANCH="${BRANCH:-main}"
WORKDIR="${WORKDIR:-/workspace}"
PULL_CACHE="${PULL_CACHE:-0}"
GH_REPO="${GH_REPO:-${REPO:-niczavadskiy/occupational-gender-bias}}"
REPO_DIR="${REPO_DIR:-occupational-gender-bias}"
HF_DATASET="${HF_DATASET:-bias-subspaces-group/qwen-bias-experiments}"

clone_url() {
  if [ -n "${GH_TOKEN:-}" ]; then
    echo "https://${GH_TOKEN}@github.com/${GH_REPO}.git"
  else
    echo "https://github.com/${GH_REPO}.git"
  fi
}

echo "=== [1/3] clone $GH_REPO (branch $BRANCH) → $WORKDIR/$REPO_DIR ==="
cd "$WORKDIR"
if [ -d "$REPO_DIR/.git" ]; then
  echo "  репо уже есть — sync to origin/$BRANCH"
  git -C "$REPO_DIR" fetch --depth 1 origin "$BRANCH"
  git -C "$REPO_DIR" checkout "$BRANCH"
  git -C "$REPO_DIR" reset --hard "origin/$BRANCH"
else
  git clone --depth 1 --branch "$BRANCH" "$(clone_url)" "$REPO_DIR"
fi
cd "$REPO_DIR"
REPO_ROOT="$WORKDIR/$REPO_DIR"
export REPO="$REPO_ROOT"

if [ "$PULL_CACHE" = "1" ]; then
  echo "=== [2/3] download прошлого прогона с HF ($HF_DATASET) ==="
  : "${HF_TOKEN:?нужен HF_TOKEN (read на dataset / org)}"
  # shellcheck disable=SC1091
  source "$REPO_ROOT/scripts/ensure_steering_env.sh"
  ensure_steering_env
  "$PY" - "$HF_DATASET" <<'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id=sys.argv[1], repo_type="dataset", local_dir="results")
print("cache → ", p)
PY
else
  echo "=== [2/3] PULL_CACHE=0 — старый прогон не качаем (data/v1 уже в репо) ==="
  if [ -f data/v1/inference_items_v1_man_first.jsonl ]; then
    echo "  data/v1 shards: ok"
  else
    echo "  WARN: нет data/v1/*.jsonl — проверьте checkout"
  fi
fi

echo "=== [3/3] steering env (numpy<2, transformers, torch) ==="
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/ensure_steering_env.sh"
ensure_steering_env

if [ -n "${HF_TOKEN:-}" ]; then
  echo "  HF_TOKEN: set"
else
  echo "  WARN: HF_TOKEN не задан — скачивание модели с Hub может упереться в rate-limit"
fi
echo "готово. repo: $REPO_ROOT  PY=$PY"
ls -1 data/v1/ 2>/dev/null | head || true
