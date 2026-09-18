#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Method 3 — HS/AUC recovery after L15 INLP erase (Qwen3.5-2B-Base) + pack.
#
# На свежем Vast (после SSH), один прогон:
#
#   export HF_TOKEN=hf_xxx
#   bash scripts/setup_instance.sh
#   bash steering/scripts/vast_hs_recovery_auc_and_pack.sh
#
# Env overrides:
#   MODEL TAG INTERVENE_LAYER RANK READ_LAYERS N_ITEMS SMOKE KEEP_HS
#   REPO  (default /workspace/occupational-gender-bias)
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="${REPO:-/workspace/occupational-gender-bias}"
MODEL="${MODEL:-Qwen/Qwen3.5-2B-Base}"
TAG="${TAG:-hs_rec_l15k16}"
INTERVENE_LAYER="${INTERVENE_LAYER:-15}"
RANK="${RANK:-16}"
READ_LAYERS="${READ_LAYERS:-15,16,17,18,19,20,21,22,23,24}"
ALPHA="${ALPHA:-1.0}"
KIND="${KIND:-center}"
MODE="${MODE:-inlp}"
VECTOR_ID="${VECTOR_ID:-w_gender_perp}"
SMOKE="${SMOKE:-0}"
KEEP_HS="${KEEP_HS:-0}"
N_ITEMS="${N_ITEMS:-}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

SUB="$REPO/steering/subspaces/inlp_gender_choice_v1.npz"
SAMPLE="$REPO/steering/samples/h1_stagea_sample_v1.json"
VEC="$REPO/steering/vectors/h1_vectors_v1.npz"
OUT_ROOT="${OUT_ROOT:-$REPO/results/steering}"
PY="${PY:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python

echo "=== deps check ==="
"$PY" - <<'PY'
import importlib.util, os, sys
need = ["torch", "transformers", "numpy"]
miss = [m for m in need if importlib.util.find_spec(m) is None]
if miss:
    print("MISSING:", miss, file=sys.stderr)
    sys.exit(1)
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available() and os.environ.get("DEVICE", "cuda") == "cuda":
    print("WARN: CUDA unavailable", file=sys.stderr)
PY

for f in "$SAMPLE" "$VEC"; do
  test -f "$f" || { echo "MISSING $f — git pull / check checkout"; exit 1; }
done
if [ "$MODE" = "inlp" ]; then
  test -f "$SUB" || {
    echo "MISSING $SUB"
    echo "  Нужен 2B INLP subspace (arrays_sha256 3545e531…). Закоммитьте steering/subspaces/ или scp."
    exit 1
  }
fi
test -f "$REPO/steering/run_hs_recovery_auc.py" || {
  echo "MISSING steering/run_hs_recovery_auc.py — git pull нужной ветки"
  exit 1
}

EXTRA=()
if [ "$SMOKE" = "1" ]; then
  TAG="${TAG}_smoke"
  N_ITEMS="${N_ITEMS:-2}"
  READ_LAYERS="${READ_LAYERS_SMOKE:-15,16,18}"
  echo "=== SMOKE mode: n_items=$N_ITEMS read=$READ_LAYERS ==="
fi
if [ -n "$N_ITEMS" ]; then
  EXTRA+=(--n-items "$N_ITEMS")
fi
if [ "$KEEP_HS" = "1" ]; then
  EXTRA+=(--save-hs)
fi

echo "=== HS recovery AUC [$TAG] mode=$MODE L=$INTERVENE_LAYER k=$RANK ==="
echo "  model=$MODEL dtype=$DTYPE read=$READ_LAYERS"

if [ "$MODE" = "inlp" ]; then
  "$PY" -m steering.run_hs_recovery_auc \
    --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" \
    --mode inlp --intervene-layer "$INTERVENE_LAYER" --rank "$RANK" \
    --alpha "$ALPHA" --kind "$KIND" \
    --subspaces "$SUB" --sample "$SAMPLE" \
    --vectors "$VEC" "$REPO/steering/vectors/slot_vectors_v1.npz" \
    --vector-id "$VECTOR_ID" \
    --read-layers "$READ_LAYERS" \
    --out-root "$OUT_ROOT" --tag "$TAG" --log-every 20 \
    "${EXTRA[@]}" \
    "$@"
else
  "$PY" -m steering.run_hs_recovery_auc \
    --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" \
    --mode probe --intervene-layer "$INTERVENE_LAYER" \
    --alpha "$ALPHA" --kind "$KIND" \
    --vector-id "$VECTOR_ID" --sample "$SAMPLE" \
    --vectors "$VEC" "$REPO/steering/vectors/slot_vectors_v1.npz" \
    --read-layers "$READ_LAYERS" \
    --out-root "$OUT_ROOT" --tag "$TAG" --log-every 20 \
    "${EXTRA[@]}" \
    "$@"
fi

RES="$OUT_ROOT/hs_recovery/$TAG"
test -f "$RES/summary.json" || { echo "FAIL: no $RES/summary.json"; exit 1; }
echo "=== summary head ==="
"$PY" - <<PY
import json
from pathlib import Path
s = json.loads(Path("$RES/summary.json").read_text())
print("intervene:", s.get("intervene"))
print("preference:", s.get("preference_rowlevel"))
print("layers (md AUC base→steer):")
for row in s.get("layers", []):
    md = row.get("mean_diff") or {}
    ab, aa = md.get("auc_baseline_test"), md.get("auc_steered_test")
    if ab is None:
        continue
    d = (aa - ab) if aa is not None and ab is not None else float("nan")
    print(f"  L{row['layer']:2d}  {ab:.4f} → {aa:.4f}  Δ={d:+.4f}")
PY

OUT="/workspace/hs_recovery_${TAG}.tar.gz"
tar -czf "$OUT" -C "$OUT_ROOT/hs_recovery" "$TAG"
ls -lh "$OUT"
echo "DONE. Download: $OUT"
echo "  scp ...:/workspace/hs_recovery_${TAG}.tar.gz ."
