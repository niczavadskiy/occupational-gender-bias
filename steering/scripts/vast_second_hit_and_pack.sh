#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Sequential erase Phase 1 — second-hit screen under L15 INLP + pack.
#
#   export HF_TOKEN=hf_xxx
#   bash scripts/setup_instance.sh   # or ensure_steering_env
#   bash steering/scripts/vast_second_hit_and_pack.sh
#
# Env: MODEL TAG LAYERS N_ITEMS SMOKE SKIP_ALONE SYNERGY_MIN REPO
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="${REPO:-/workspace/occupational-gender-bias}"
MODEL="${MODEL:-Qwen/Qwen3.5-2B-Base}"
TAG="${TAG:-second_hit_v1}"
LAYERS="${LAYERS:-16,17,18,19,20,21,22,23,24}"
BASE_LAYER="${BASE_LAYER:-15}"
BASE_RANK="${BASE_RANK:-16}"
SECOND_ALPHA="${SECOND_ALPHA:-1.0}"
SMOKE="${SMOKE:-0}"
SKIP_ALONE="${SKIP_ALONE:-0}"
N_ITEMS="${N_ITEMS:-}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"
SYNERGY_MIN="${SYNERGY_MIN:-0.005}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

SUB="$REPO/steering/subspaces/inlp_gender_choice_v1.npz"
SAMPLE="$REPO/steering/samples/h1_stagea_sample_v1.json"

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

for f in "$SAMPLE" "$SUB" \
  "$REPO/steering/vectors/h1_vectors_v1.npz" \
  "$REPO/steering/vectors/slot_vectors_v1.npz" \
  "$REPO/steering/run_second_hit_screen.py"; do
  test -f "$f" || { echo "MISSING $f"; exit 1; }
done

EXTRA=()
if [ "$SMOKE" = "1" ]; then
  TAG="${TAG}_smoke"
  N_ITEMS="${N_ITEMS:-2}"
  LAYERS="${LAYERS_SMOKE:-16,20,23}"
  echo "=== SMOKE: n_items=$N_ITEMS layers=$LAYERS ==="
fi
if [ -n "$N_ITEMS" ]; then
  EXTRA+=(--n-items "$N_ITEMS")
fi
if [ "$SKIP_ALONE" = "1" ]; then
  EXTRA+=(--skip-alone)
fi

OUT_ROOT="${OUT_ROOT:-$REPO/results/steering}"
echo "=== second-hit [$TAG] base=L${BASE_LAYER}k${BASE_RANK} layers=$LAYERS ==="

"$PY" -m steering.run_second_hit_screen \
  --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" \
  --subspaces "$SUB" --sample "$SAMPLE" \
  --base-layer "$BASE_LAYER" --base-rank "$BASE_RANK" \
  --layers "$LAYERS" --second-alpha "$SECOND_ALPHA" \
  --synergy-min "$SYNERGY_MIN" \
  --out-root "$OUT_ROOT" --tag "$TAG" --log-every 20 \
  "${EXTRA[@]}" \
  "$@"

RES="$OUT_ROOT/second_hit/$TAG"
test -f "$RES/shortlist.json" || { echo "FAIL: no $RES/shortlist.json"; exit 1; }
echo "=== shortlist ==="
"$PY" - <<PY
import json
from pathlib import Path
s = json.loads(Path("$RES/shortlist.json").read_text())
print("base R:", s.get("base", {}).get("R_gender"))
for e in s.get("shortlist", []):
    print(f"  L{e['layer']} {e['axis']:6s} {e['mode']:20s} synergy={e['synergy_R_gender']:+.4f}")
if not s.get("shortlist"):
    print("  (empty)")
PY

OUT="/workspace/second_hit_${TAG}.tar.gz"
tar -czf "$OUT" -C "$OUT_ROOT/second_hit" "$TAG"
ls -lh "$OUT"
echo "DONE. Download: $OUT"
