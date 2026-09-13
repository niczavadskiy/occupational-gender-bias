#!/usr/bin/env bash
# Rebuild 4B candidates + vectors (+ optional INLP) into this experiment tree.
# Run from anywhere; requires PYTHONPATH to Bias--subspaces-in-LLM / repo.
set -euo pipefail

STEER_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EXP_DIR="$(cd "$STEER_DIR/.." && pwd)"
PACK="${PACK:-$EXP_DIR/results/qwen35_4b_h1h3_pack}"
REPO="${REPO:-}"

if [[ -z "$REPO" ]]; then
  if [[ -d "$EXP_DIR/../../../repo" ]]; then
    REPO="$(cd "$EXP_DIR/../../../repo" && pwd)"
  elif [[ -d /workspace/Bias--subspaces-in-LLM ]]; then
    REPO=/workspace/Bias--subspaces-in-LLM
  else
    echo "Set REPO= path to inference repo (contains steering/)"
    exit 1
  fi
fi

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

echo "REPO=$REPO"
echo "STEER=$STEER_DIR"
echo "PACK=$PACK"

mkdir -p "$STEER_DIR/candidates" "$STEER_DIR/vectors" "$STEER_DIR/subspaces" "$STEER_DIR/samples"

python -m steering.build_h1_candidates \
  --config "$STEER_DIR/configs/h1_steering_candidates_v1.yaml" \
  --out-dir "$STEER_DIR/candidates" \
  --out-prefix h1_candidates

python -m steering.build_h1_candidates \
  --config "$STEER_DIR/configs/slot_steering_candidates_v1.yaml" \
  --out-dir "$STEER_DIR/candidates" \
  --out-prefix slot_candidates

python -m steering.build_h1_vectors \
  --config "$STEER_DIR/configs/h1_steering_candidates_v1.yaml" \
  --results-root "$PACK" \
  --out-dir "$STEER_DIR/vectors" \
  --out-prefix h1_vectors

python -m steering.build_h1_vectors \
  --config "$STEER_DIR/configs/slot_steering_candidates_v1.yaml" \
  --results-root "$PACK" \
  --out-dir "$STEER_DIR/vectors" \
  --out-prefix slot_vectors

if [[ "${SKIP_INLP:-0}" != "1" ]]; then
  python -m steering.build_inlp_subspace \
    --config "$STEER_DIR/configs/inlp_gender_v1.yaml" \
    --results-root "$PACK" \
    --out-dir "$STEER_DIR/subspaces"

  python -m steering.build_inlp_subspace \
    --config "$STEER_DIR/configs/inlp_slot_v1.yaml" \
    --results-root "$PACK" \
    --out-dir "$STEER_DIR/subspaces"
fi

echo "DONE artifacts under $STEER_DIR"
