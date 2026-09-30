#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Ministral 3 3B Base: full occupational behavioral run for H1 · H3 · H5 · H11.
# Inference dtype: bfloat16 (src/inference.py). HS: 27 × 3072.
#
# 6 shards × 7608 = 45648 items
#   v1_full_pos_shuffle  (no_evidence, 15216)  → H1 + H11 probes
#   highlight-h3-full    (all evidence, 45648) → H3 + H5
#
# Usage on Vast:
#   export HF_TOKEN=hf_xxx
#   INSTALL_MINISTRAL=1 bash scripts/vast_ministral3_3b_h1h3_and_pack.sh
#
# Or via vast_run.sh:
#   HF_TOKEN=hf_xxx DISK=120 KEEP=1 GPU=RTX_4090 \
#   REMOTE_CMD='bash scripts/setup_instance.sh && INSTALL_MINISTRAL=1 bash scripts/vast_ministral3_3b_h1h3_and_pack.sh' \
#   bash scripts/vast_run.sh
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."
export REPO="$(pwd)"
PY="${PY:-python3}"
MODEL="${MODEL:-mistralai/Ministral-3-3B-Base-2512}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/results}"
export RESULTS_DIR
SHORT="$(basename "$MODEL")"
TAG_PREFIX="${TAG_PREFIX:-ministral3_3b}"

if [ "${INSTALL_MINISTRAL:-1}" = "1" ]; then
  echo "=== [deps] transformers + mistral-common for Ministral 3 ==="
  # shellcheck disable=SC1091
  if [ -f scripts/ensure_steering_env.sh ]; then
    source scripts/ensure_steering_env.sh
    ensure_steering_env
  fi
  "$PY" -m pip install -U 'transformers>=4.57.0' mistral-common
fi

test -f data/v1/inference_items_h3_full.jsonl || {
  echo "=== rebuild data/v1 from highlight-h3-full/per_item.jsonl ==="
  $PY -m src.extract_v1_items_from_per_item
}
test -f data/v1/inference_items_h3_full.jsonl || {
  echo "MISSING data/v1 — need results/highlight-h3-full/per_item.jsonl"
  exit 1
}

echo "=== [0] smoke $MODEL (expect d_model=3072, n_layers=26) ==="
$PY -m src.smoke_qwen --model "$MODEL" --device cuda

run_shard() {
  local items="$1"
  local tag="$2"
  echo "=== inference $tag ($items) ==="
  $PY src/inference.py \
    --model_id "$MODEL" \
    --items_file "$items" \
    --out_base "$RESULTS_DIR" \
    --run_tag "$tag"
}

run_shard data/v1/inference_items_v1_man_first.jsonl          "${TAG_PREFIX}_v1_man_first"
run_shard data/v1/inference_items_v1_woman_first.jsonl        "${TAG_PREFIX}_v1_woman_first"
run_shard data/v1/inference_items_hl_man_man_first.jsonl      "${TAG_PREFIX}_hl_man_mf"
run_shard data/v1/inference_items_hl_man_woman_first.jsonl    "${TAG_PREFIX}_hl_man_wf"
run_shard data/v1/inference_items_hl_woman_man_first.jsonl    "${TAG_PREFIX}_hl_woman_mf"
run_shard data/v1/inference_items_hl_woman_woman_first.jsonl  "${TAG_PREFIX}_hl_woman_wf"

find_run() {
  local tag="$1"
  ls -1d "$RESULTS_DIR"/run_*_"${SHORT}_${tag}" 2>/dev/null | sort | tail -1
}

MF="$(find_run "${TAG_PREFIX}_v1_man_first")"
WF="$(find_run "${TAG_PREFIX}_v1_woman_first")"
HL_MM="$(find_run "${TAG_PREFIX}_hl_man_mf")"
HL_MW="$(find_run "${TAG_PREFIX}_hl_man_wf")"
HL_WM="$(find_run "${TAG_PREFIX}_hl_woman_mf")"
HL_WW="$(find_run "${TAG_PREFIX}_hl_woman_wf")"

for d in "$MF" "$WF" "$HL_MM" "$HL_MW" "$HL_WM" "$HL_WW"; do
  test -n "$d" && test -d "$d" || { echo "missing run dir: $d"; exit 1; }
  echo "  shard ok: $d"
done

echo "=== merge v1_full_pos_shuffle (no_evidence, 15216) ==="
OUT_V1="$RESULTS_DIR/run_$(date +%Y-%m-%d_%H-%M-%S)_${SHORT}_v1_full_pos_shuffle"
$PY -m src.merge_inference_runs \
  --run-a "$MF" --run-b "$WF" \
  --context-a man_first --context-b woman_first \
  --run-tag v1_full_pos_shuffle \
  --out "$OUT_V1"

echo "=== merge highlight-h3-full (45648) + HS ==="
OUT_H3="$RESULTS_DIR/highlight-h3-full"
rm -rf "$OUT_H3"
$PY -m src.merge_inference_runs \
  --h3-evidence \
  --merge-npz \
  --run-tag h3_evidence_full \
  --out "$OUT_H3" \
  --runs \
    "$OUT_V1" no_evidence \
    "$HL_MM" man \
    "$HL_MW" man \
    "$HL_WM" woman \
    "$HL_WW" woman

ln -sfn "$OUT_V1" "$RESULTS_DIR/v1_full_pos_shuffle" || true

echo "=== pack ==="
PACK=/workspace/${TAG_PREFIX}_h1h3_pack.tar.gz
tar -czf "$PACK" \
  -C "$RESULTS_DIR" \
  "$(basename "$OUT_V1")" \
  highlight-h3-full \
  v1_full_pos_shuffle
ls -lh "$PACK"

echo "DONE."
echo "  H1/H11 run: $OUT_V1"
echo "  H3/H5 run:  $OUT_H3"
echo "  Pack:       $PACK"
echo ""
echo "Local after download → experiments/ministral3-3b-base/results/:"
echo "  python -m src.metrics.h1_v1 --run experiments/ministral3-3b-base/results/v1_full_pos_shuffle"
echo "  python -m src.metrics.h3_v1 --run experiments/ministral3-3b-base/results/highlight-h3-full"
echo "  python -m probes.v1_rep_run --run <v1_full_pos_shuffle> --all"
echo "Then fill PEAK in experiments/ministral3-3b-base/steering/ (see PIPELINE.md)."
