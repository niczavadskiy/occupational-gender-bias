#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-command setup на свежем Vast-инстансе (образ bias-subspaces-env).
#
# Делает то, чего НЕТ в образе (by design):
#   1. git clone приватного репо с кодом (нужен GH_TOKEN);
#   2. download прошлого прогона (HS + per_item) с HF Dataset (нужен HF_TOKEN) —
#      качается на скорости инстанса, а не с твоего аплоада.
#
# Использование (на инстансе, deps уже в образе/venv):
#   export GH_TOKEN=ghp_xxx          # fine-grained read-only на этот репо
#   export HF_TOKEN=hf_xxx           # read на org bias-subspaces-group
#   bash setup_instance.sh           # или: curl ... | bash, см. docs/docker.md
#
# Переопределяемые переменные:
#   BRANCH      (default qwen_2b_experiments_olya)
#   WORKDIR     (default /workspace)
#   PULL_CACHE  (default 1 — тянуть старый прогон с HF; 0 — пропустить)
# ---------------------------------------------------------------------------
set -euo pipefail

BRANCH="${BRANCH:-qwen_2b_experiments_olya}"
WORKDIR="${WORKDIR:-/workspace}"
PULL_CACHE="${PULL_CACHE:-1}"
REPO="olyamasaeva/Bias--subspaces-in-LLM"
HF_DATASET="bias-subspaces-group/qwen-bias-experiments"
PY="${PY:-/venv/main/bin/python}"   # python из образа/Vast-venv; fallback ниже
command -v "$PY" >/dev/null 2>&1 || PY="python3"

echo "=== [1/3] clone $REPO (branch $BRANCH) ==="
cd "$WORKDIR"
if [ -d "Bias--subspaces-in-LLM/.git" ]; then
  echo "  репо уже есть — git pull"
  git -C Bias--subspaces-in-LLM fetch --depth 1 origin "$BRANCH"
  git -C Bias--subspaces-in-LLM checkout "$BRANCH"
  git -C Bias--subspaces-in-LLM reset --hard "origin/$BRANCH"
else
  : "${GH_TOKEN:?нужен GH_TOKEN (fine-grained read-only на репо)}"
  git clone --depth 1 --branch "$BRANCH" \
    "https://${GH_TOKEN}@github.com/${REPO}.git"
fi
cd Bias--subspaces-in-LLM

if [ "$PULL_CACHE" = "1" ]; then
  echo "=== [2/3] download прошлого прогона с HF ($HF_DATASET) ==="
  : "${HF_TOKEN:?нужен HF_TOKEN (read на org bias-subspaces-group)}"
  "$PY" -m huggingface_hub.commands.huggingface_cli download \
      "$HF_DATASET" --repo-type dataset --local-dir results/ \
    || hf download "$HF_DATASET" --repo-type dataset --local-dir results/
else
  echo "=== [2/3] PULL_CACHE=0 — старый прогон не качаем ==="
fi

echo "=== [3/3] sanity ==="
"$PY" -c "import torch,transformers; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'| transformers',transformers.__version__)"
echo "готово. repo: $WORKDIR/Bias--subspaces-in-LLM"
ls -1 results/ 2>/dev/null | head || true
