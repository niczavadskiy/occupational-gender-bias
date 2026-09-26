#!/usr/bin/env bash
# 4B polarity-pooled INLP: Stage A → B → C for promale then profemale.
#
# Candidate layers = gender_prob peak L24 + L23 + L22 (matched XY peak_prepeak).
# Protocol: steering/POLARITY_POOL_INLP.md
#
#   export HF_TOKEN=hf_xxx
#   BRANCH=qwen_2b_experiments bash steering/scripts/vast_inlp_polarity_pool_4b_full_instance.sh
#
# Env: SETS=promale,profemale SKIP_PIP=1 SKIP_STAGE_A=1 SKIP_FIT=1 RUN_STAGE_B=0 SMOKE=1
# SKIP_FIT=1 reuses existing subspaces/*.npz and jumps to Stage A (e.g. after k_found=1).
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?export HF_TOKEN=hf_xxx}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"

WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="${REPO_DIR:-occupational-gender-bias}"
REPO="${REPO:-$WORKDIR/$REPO_DIR}"
BRANCH="${BRANCH:-qwen_2b_experiments}"
GH_REPO="${GH_REPO:-niczavadskiy/occupational-gender-bias}"
SETS="${SETS:-promale,profemale}"
SCALE=4b
MODEL="${MODEL:-Qwen/Qwen3.5-4B-Base}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"
SMOKE="${SMOKE:-0}"
SKIP_STAGE_A="${SKIP_STAGE_A:-0}"
SKIP_FIT="${SKIP_FIT:-0}"
RUN_STAGE_B="${RUN_STAGE_B:-1}"
RUN_STAGE_C="${RUN_STAGE_C:-1}"
SETS_JSON_REL="steering/domains/inlp_polarity_sets_4b_v1.json"
CONFIG_REL="steering/configs/inlp_polarity_pool_4b_peak_prepeak.yaml"
MMLU_PARQUET="${MMLU_PARQUET:-$REPO/steering/.cache/mmlu_pro_test.parquet}"

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
MMLU_PARQUET="${MMLU_PARQUET:-$REPO/steering/.cache/mmlu_pro_test.parquet}"

echo "=== [2] setup_instance + frozen Vast env ==="
export SKIP_FLA="${SKIP_FLA:-1}"
PULL_CACHE=0 bash scripts/setup_instance.sh
# shellcheck disable=SC1091
source "$REPO/scripts/ensure_steering_env.sh"
if [ "${SKIP_PIP:-0}" = "1" ]; then
  SKIP_PIP=1 ensure_steering_env
else
  ensure_steering_env
fi
export PY
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

need=(
  "$SETS_JSON_REL"
  "$CONFIG_REL"
  steering/POLARITY_POOL_INLP.md
  steering/polarity_pool_inlp.py
  steering/build_polarity_pool_inlp_sample.py
  steering/build_polarity_pool_inlp_subspace.py
  steering/run_polarity_pool_inlp_stagea.py
  steering/run_polarity_pool_inlp_stageb.py
  steering/run_polarity_pool_inlp_stagec.py
  steering/xy_control/data/xy_pairs_full_v1.json
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
[ "$miss" = "0" ] || exit 1

"$PY" -m steering.test_polarity_pool_inlp
"$PY" -m steering.xy_control.build_dataset

mkdir -p "$(dirname "$MMLU_PARQUET")"
if [ ! -f "$MMLU_PARQUET" ]; then
  echo "=== download frozen MMLU-Pro test parquet ==="
  curl -L --fail --retry 3 -o "$MMLU_PARQUET" \
    "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"
fi
if ! "$PY" -c "import pyarrow" >/dev/null 2>&1; then
  "$PY" -m pip install -U pyarrow
fi
if ! "$PY" -c "import yaml" >/dev/null 2>&1; then
  "$PY" -m pip install -U pyyaml
fi

chmod +x steering/scripts/vast_inlp_polarity_pool_4b_full_instance.sh

lookup_sample_stem() {
  local set_id="$1"
  "$PY" - <<PY
import json
from pathlib import Path
doc = json.loads(Path("$SETS_JSON_REL").read_text(encoding="utf-8"))
for s in doc["sets"]:
    if s["id"] == "$set_id":
        print(s.get("sample_stem") or f"inlp_polarity_pool_{s['id']}_v1")
        print(s.get("subspace_tag") or f"polarity_pool_{s['id']}_v1")
        raise SystemExit(0)
raise SystemExit(f"unknown set id: $set_id")
PY
}

IFS=',' read -r -a SET_LIST <<< "$SETS"
for sid in "${SET_LIST[@]}"; do
  sid="$(echo "$sid" | tr -d '[:space:]')"
  [ -n "$sid" ] || continue
  echo ""
  echo "========== polarity set: $sid (4B) =========="

  mapfile -t META < <(lookup_sample_stem "$sid")
  SAMPLE_STEM="${META[0]}"
  SUB_TAG="${META[1]}"
  SAMPLE_BASE="${SAMPLE_STEM%_v1}"

  echo "=== [$sid] build samples → $SAMPLE_STEM ==="
  "$PY" -m steering.build_polarity_pool_inlp_sample \
    --set-id "$sid" --sets "$REPO/$SETS_JSON_REL" --config "$REPO/$CONFIG_REL" --scale "$SCALE"

  if [ "$SKIP_STAGE_A" != "1" ]; then
    smoke_flags=()
    if [ "$SMOKE" = "1" ]; then
      smoke_flags=(--smoke)
    fi
    if [ "$SKIP_FIT" != "1" ]; then
      echo "=== [$sid] fit subspace (peak + prepeak) tag=$SUB_TAG ==="
      "$PY" -m steering.build_polarity_pool_inlp_subspace \
        --set-id "$sid" --sets "$REPO/$SETS_JSON_REL" --config "$REPO/$CONFIG_REL" \
        --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" \
        "${smoke_flags[@]}"
    else
      echo "=== [$sid] SKIP_FIT=1 — reuse existing subspaces ==="
    fi

    echo "=== [$sid] Stage A (ranks clamped to k_found) ==="
    "$PY" -m steering.run_polarity_pool_inlp_stagea \
      --set-id "$sid" --sets "$REPO/$SETS_JSON_REL" --config "$REPO/$CONFIG_REL" \
      --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" \
      "${smoke_flags[@]}"
  fi

  if [ "$RUN_STAGE_B" = "1" ]; then
    echo "=== [$sid] Stage B ==="
    smoke_flags=()
    if [ "$SMOKE" = "1" ]; then
      smoke_flags=(--smoke)
    fi
    "$PY" -m steering.run_polarity_pool_inlp_stageb \
      --set-id "$sid" --sets "$REPO/$SETS_JSON_REL" --config "$REPO/$CONFIG_REL" \
      --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" \
      "${smoke_flags[@]}"
  fi

  if [ "$RUN_STAGE_C" = "1" ]; then
    echo "=== [$sid] Stage C ==="
    smoke_flags=()
    if [ "$SMOKE" = "1" ]; then
      smoke_flags=(--smoke)
    fi
    "$PY" -m steering.run_polarity_pool_inlp_stagec \
      --set-id "$sid" --sets "$REPO/$SETS_JSON_REL" --config "$REPO/$CONFIG_REL" \
      --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" \
      "${smoke_flags[@]}"
  fi

  pack="/workspace/inlp_polarity_pool_${sid}_stage_abc_${SCALE}.tar.gz"
  echo "=== [$sid] pack → $pack ==="
  tar -czf "$pack" \
    "steering/samples/${SAMPLE_STEM}.json" \
    "steering/samples/${SAMPLE_BASE}_train_v1.json" \
    "steering/samples/${SAMPLE_BASE}_val_v1.json" \
    "steering/samples/${SAMPLE_BASE}_test_v1.json" \
    "steering/subspaces/inlp_gender_prob_${SUB_TAG}.npz" \
    "steering/subspaces/inlp_gender_prob_${SUB_TAG}.json" \
    results/steering/inlp_polarity_pool \
    || echo "WARN: partial pack for $sid"
  ls -lh "$pack" || true
done

echo "=== all polarity INLP sets done (4B) ==="
ls -lh /workspace/inlp_polarity_pool_*_stage_abc_4b.tar.gz 2>/dev/null || true
