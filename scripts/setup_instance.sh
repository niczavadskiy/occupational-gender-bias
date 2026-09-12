#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-command setup на свежем Vast-инстансе (образ bias-subspaces-env).
#
#   1. git clone публичного репо с кодом + data/v1
#      (GH_TOKEN не обязателен; нужен только если репо станет private)
#   2. опционально: download старого 2B-прогона с HF (PULL_CACHE=1)
#   3. sanity: torch / CUDA
#
# Использование (на инстансе):
#   export HF_TOKEN=hf_xxx           # нужен для скачивания модели с Hub
#   # export GH_TOKEN=ghp_xxx        # только если репо private
#   bash scripts/setup_instance.sh
#
# Переопределяемые переменные:
#   BRANCH      (default main)
#   WORKDIR     (default /workspace)
#   REPO        (default niczavadskiy/occupational-gender-bias)
#   REPO_DIR    (default occupational-gender-bias)
#   PULL_CACHE  (default 0 — для 4B свежий прогон; 1 — тянуть старый 2B с HF)
# ---------------------------------------------------------------------------
set -euo pipefail

BRANCH="${BRANCH:-main}"
WORKDIR="${WORKDIR:-/workspace}"
PULL_CACHE="${PULL_CACHE:-0}"
REPO="${REPO:-niczavadskiy/occupational-gender-bias}"
REPO_DIR="${REPO_DIR:-occupational-gender-bias}"
HF_DATASET="${HF_DATASET:-bias-subspaces-group/qwen-bias-experiments}"
PY="${PY:-/venv/main/bin/python}"
command -v "$PY" >/dev/null 2>&1 || PY="python3"

clone_url() {
  if [ -n "${GH_TOKEN:-}" ]; then
    echo "https://${GH_TOKEN}@github.com/${REPO}.git"
  else
    echo "https://github.com/${REPO}.git"
  fi
}

echo "=== [1/3] clone $REPO (branch $BRANCH) → $WORKDIR/$REPO_DIR ==="
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

if [ "$PULL_CACHE" = "1" ]; then
  echo "=== [2/3] download прошлого прогона с HF ($HF_DATASET) ==="
  : "${HF_TOKEN:?нужен HF_TOKEN (read на dataset / org)}"
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

echo "=== [3/3] sanity ==="
"$PY" -c "import torch,transformers; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'| transformers',transformers.__version__)"
if [ -n "${HF_TOKEN:-}" ]; then
  echo "  HF_TOKEN: set"
else
  echo "  WARN: HF_TOKEN не задан — скачивание модели с Hub может упереться в rate-limit"
fi
echo "готово. repo: $WORKDIR/$REPO_DIR"
ls -1 data/v1/ 2>/dev/null | head || true
