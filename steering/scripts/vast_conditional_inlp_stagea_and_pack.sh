#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Sequential erase Phase 2 — conditional INLP under S0 + stacked Stage A + pack.
#
# Expects second-hit shortlist (or set LAYERS / SHORTLIST explicitly).
#
#   export HF_TOKEN=hf_xxx
#   bash steering/scripts/vast_conditional_inlp_stagea_and_pack.sh
#
# Env:
#   SHORTLIST   path to shortlist.json (default: latest-ish under results)
#   LAYERS      override layers for fit + Stage A (comma)
#   TARGETS     gender_choice,slot_choice
#   RANKS       Stage A ranks (default 8,16,32)
#   TAG K_MAX SMOKE N_ITEMS REPO
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="${REPO:-/workspace/occupational-gender-bias}"
MODEL="${MODEL:-Qwen/Qwen3.5-2B-Base}"
TAG="${TAG:-cond_under_l15_v1}"
TARGETS="${TARGETS:-gender_choice,slot_choice}"
RANKS="${RANKS:-8,16,32}"
K_MAX="${K_MAX:-32}"
LAYERS="${LAYERS:-}"
SHORTLIST="${SHORTLIST:-}"
SMOKE="${SMOKE:-0}"
N_ITEMS="${N_ITEMS:-}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"
BASE_LAYER="${BASE_LAYER:-15}"
BASE_RANK="${BASE_RANK:-16}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

SUB="$REPO/steering/subspaces/inlp_gender_choice_v1.npz"
SAMPLE="$REPO/steering/samples/h1_stagea_sample_v1.json"
OUT_ROOT="${OUT_ROOT:-$REPO/results/steering}"

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

echo "=== deps check (PY=$PY) ==="
if ! "$PY" -c "import numpy,torch,transformers; assert int(numpy.__version__.split('.')[0]) < 2" 2>/dev/null; then
  # shellcheck disable=SC1091
  source "$REPO/scripts/ensure_steering_env.sh"
  ensure_steering_env
  PY="$(cat /tmp/occupational_steering_py)"
  export PY
fi
# sklearn optional but preferred for conditional INLP
"$PY" -c "import sklearn" 2>/dev/null || "$PY" -m pip install -q scikit-learn || true

for f in "$SAMPLE" "$SUB" \
  "$REPO/steering/build_conditional_inlp_subspace.py" \
  "$REPO/steering/run_stacked_inlp_stagea.py"; do
  test -f "$f" || { echo "MISSING $f"; exit 1; }
done

if [ -z "$SHORTLIST" ]; then
  # newest shortlist under results/steering/second_hit/*/shortlist.json
  SHORTLIST="$("$PY" - <<'PY'
from pathlib import Path
root = Path("results/steering/second_hit")
cands = sorted(root.glob("*/shortlist.json"), key=lambda p: p.stat().st_mtime, reverse=True)
print(cands[0] if cands else "")
PY
)"
fi

BUILD_EXTRA=()
STAGE_EXTRA=()
if [ "$SMOKE" = "1" ]; then
  TAG="${TAG}_smoke"
  N_ITEMS="${N_ITEMS:-2}"
  LAYERS="${LAYERS:-16,20,23}"
  K_MAX="${K_MAX:-8}"
  RANKS="${RANKS:-8}"
  echo "=== SMOKE ==="
fi
if [ -n "$N_ITEMS" ]; then
  BUILD_EXTRA+=(--n-items "$N_ITEMS")
  STAGE_EXTRA+=(--limit-items "$N_ITEMS")
fi
if [ -n "$LAYERS" ]; then
  BUILD_EXTRA+=(--layers "$LAYERS")
  STAGE_EXTRA+=(--layers "$LAYERS")
fi
if [ -n "$SHORTLIST" ] && [ -f "$SHORTLIST" ]; then
  echo "using shortlist: $SHORTLIST"
  BUILD_EXTRA+=(--from-shortlist "$SHORTLIST")
  STAGE_EXTRA+=(--from-shortlist "$SHORTLIST")
else
  echo "WARN: no shortlist — using default/explicit layers"
  if [ -z "$LAYERS" ]; then
    BUILD_EXTRA+=(--layers "16,17,18,20,23")
    STAGE_EXTRA+=(--layers "16,17,18,20,23")
  fi
fi

echo "=== [1/2] build conditional subspaces [$TAG] targets=$TARGETS ==="
"$PY" -m steering.build_conditional_inlp_subspace \
  --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" \
  --base-subspaces "$SUB" --sample "$SAMPLE" \
  --base-layer "$BASE_LAYER" --base-rank "$BASE_RANK" \
  --targets "$TARGETS" --k-max "$K_MAX" \
  --out-dir "$REPO/steering/subspaces" --tag "$TAG" --log-every 20 \
  "${BUILD_EXTRA[@]}"

# Stage A per target present
IFS=',' read -ra TARR <<< "$TARGETS"
PACK_DIRS=()
for tgt in "${TARR[@]}"; do
  tgt="${tgt// /}"
  NPZ="$REPO/steering/subspaces/inlp_cond_${tgt}_${TAG}.npz"
  test -f "$NPZ" || { echo "MISSING $NPZ"; exit 1; }
  STAG="stacked_${tgt}_${TAG}"
  echo "=== [2/2] stacked Stage A [$STAG] ==="
  "$PY" -m steering.run_stacked_inlp_stagea \
    --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" \
    --base-subspaces "$SUB" --sample "$SAMPLE" \
    --base-layer "$BASE_LAYER" --base-rank "$BASE_RANK" \
    --second-subspaces "$NPZ" \
    --ranks "$RANKS" --alphas 1.0 \
    --out-root "$OUT_ROOT" --tag "$STAG" --log-every 20 \
    "${STAGE_EXTRA[@]}"
  PACK_DIRS+=("$STAG")
done

OUT="/workspace/stacked_inlp_${TAG}.tar.gz"
# pack Stage A results + new subspaces
tmpdir=$(mktemp -d)
mkdir -p "$tmpdir/stage_a" "$tmpdir/subspaces"
for d in "${PACK_DIRS[@]}"; do
  cp -a "$OUT_ROOT/stacked_inlp_stage_a/$d" "$tmpdir/stage_a/"
done
cp -a "$REPO/steering/subspaces/inlp_cond_"*"_${TAG}".* "$tmpdir/subspaces/" 2>/dev/null || true
tar -czf "$OUT" -C "$tmpdir" .
rm -rf "$tmpdir"
ls -lh "$OUT"
echo "DONE. Download: $OUT"
