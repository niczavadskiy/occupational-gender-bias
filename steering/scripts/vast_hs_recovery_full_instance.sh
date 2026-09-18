#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Полный lifecycle на НОВОМ Vast-инстансе: setup → smoke → full HS recovery → pack.
#
# После SSH на инстанс:
#
#   export HF_TOKEN=hf_xxx
#   curl -fsSL ...   # или:
#   bash -c "$(cat <<'EOF'
#   ... этот файл ...
#   EOF
#   )"
#
# Обычно:
#   export HF_TOKEN=hf_xxx
#   bash scripts/setup_instance.sh          # clone repo
#   # если этот скрипт ещё не в origin — scp его + subspaces
#   bash steering/scripts/vast_hs_recovery_full_instance.sh
#
# Env: SKIP_SMOKE=1  SKIP_FULL=1  BRANCH TAG MODEL ...
# ---------------------------------------------------------------------------
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
MODEL="${MODEL:-Qwen/Qwen3.5-2B-Base}"

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

if [ -f scripts/setup_instance.sh ]; then
  echo "=== [1b] setup_instance (PULL_CACHE=0) ==="
  PULL_CACHE=0 bash scripts/setup_instance.sh || true
fi

echo "=== [2] pip extras (if needed) ==="
PY="${PY:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
"$PY" -c "import torch,transformers" 2>/dev/null || \
  "$PY" -m pip install -q 'torch>=2.5' transformers accelerate huggingface_hub python-dotenv PyYAML

echo "=== [3] artifact check ==="
need=(
  steering/run_hs_recovery_auc.py
  steering/scripts/vast_hs_recovery_auc_and_pack.sh
  steering/subspaces/inlp_gender_choice_v1.npz
  steering/vectors/h1_vectors_v1.npz
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
  echo ""
  echo "Файлы не в origin/$BRANCH. С локальной машины:"
  echo "  scp steering/run_hs_recovery_auc.py root@HOST:$REPO/steering/"
  echo "  scp -r steering/subspaces root@HOST:$REPO/steering/"
  echo "  scp steering/scripts/vast_hs_recovery_*.sh root@HOST:$REPO/steering/scripts/"
  exit 1
fi

"$PY" - <<'PY'
import hashlib, json, numpy as np
from pathlib import Path
p = Path("steering/subspaces/inlp_gender_choice_v1.npz")
meta = json.loads(p.with_suffix(".json").read_text())
z = dict(np.load(p))
h = hashlib.sha256()
for k in sorted(z):
    a = np.ascontiguousarray(z[k]); h.update(k.encode()); h.update(a.tobytes())
sig = h.hexdigest()
exp = meta["arrays_sha256"]
assert sig == exp, (sig, exp)
assert "L15__W" in z and z["L15__W"].shape[1] >= 16
print("  subspaces arrays_sha256 OK:", sig[:12], "…  L15__W", z["L15__W"].shape)
PY

chmod +x steering/scripts/vast_hs_recovery_auc_and_pack.sh

if [ "$SKIP_SMOKE" != "1" ]; then
  echo "=== [4] SMOKE (2 families, L15/16/18) ==="
  SMOKE=1 TAG="${TAG:-hs_rec_l15k16}" MODEL="$MODEL" \
    bash steering/scripts/vast_hs_recovery_auc_and_pack.sh
fi

if [ "$SKIP_FULL" != "1" ]; then
  echo "=== [5] FULL (95 families, L15–24) ==="
  SMOKE=0 TAG="${TAG:-hs_rec_l15k16}" MODEL="$MODEL" \
    bash steering/scripts/vast_hs_recovery_auc_and_pack.sh
fi

echo "=== ALL DONE ==="
ls -lh /workspace/hs_recovery_*.tar.gz 2>/dev/null || true
echo "Скачай pack: /workspace/hs_recovery_hs_rec_l15k16.tar.gz"
