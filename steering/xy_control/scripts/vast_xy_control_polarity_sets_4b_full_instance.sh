#!/usr/bin/env bash
# 4B XY-control: POOLED polarity Stage A → B → C (pro-male then pro-female).
#
# Protocol: one v_raw per polarity set (all train families), one (L,α) on
# pooled val, B/C on pooled test + MMLU union of member SOCs.
#
# Frozen sets: steering/xy_control/domains/polarity_sets_4b_v1.json
# Config:      steering/xy_control/configs/xy_control_4b_peak_prepeak_a6.yaml
# Layers:      L24 (probe peak), L23, L22; |α|≤6
#
#   export HF_TOKEN=hf_xxx
#   BRANCH=qwen_2b_experiments bash steering/xy_control/scripts/vast_xy_control_polarity_sets_4b_full_instance.sh
#
# Env: BRANCH SETS=promale,profemale SKIP_PIP=1 SKIP_STAGE_A=1 RUN_STAGE_B=0
#      CAP_LOSS_MAX=0.03 SMOKE=1
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
EVAL_SPLIT="${EVAL_SPLIT:-val}"
CAP_LOSS_MAX="${CAP_LOSS_MAX:-0.03}"
SMOKE="${SMOKE:-0}"
SKIP_STAGE_A="${SKIP_STAGE_A:-0}"
RUN_STAGE_B="${RUN_STAGE_B:-1}"
SETS_JSON_REL="steering/xy_control/domains/polarity_sets_4b_v1.json"
CONFIG_REL="steering/xy_control/configs/xy_control_4b_peak_prepeak_a6.yaml"
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
  steering/xy_control/run_polarity_pool.py
  steering/xy_control/run_polarity_pool_stageb.py
  steering/xy_control/run_polarity_pool_stagec.py
  steering/xy_control/polarity_pool.py
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

chmod +x steering/xy_control/scripts/vast_xy_control_polarity_sets_4b_full_instance.sh
"$PY" -m steering.xy_control.test_polarity_pool
"$PY" -m steering.xy_control.build_dataset
"$PY" -m steering.xy_control.test_dataset

mkdir -p "$(dirname "$MMLU_PARQUET")"
if [ ! -f "$MMLU_PARQUET" ]; then
  echo "=== download frozen MMLU-Pro test parquet ==="
  curl -L --fail --retry 3 -o "$MMLU_PARQUET" \
    "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"
fi
if ! "$PY" -c "import pyarrow" >/dev/null 2>&1; then
  "$PY" -m pip install -U pyarrow
fi

lookup_set() {
  local set_id="$1"
  "$PY" - <<PY
import json
from pathlib import Path
doc = json.loads(Path("$SETS_JSON_REL").read_text(encoding="utf-8"))
for s in doc["sets"]:
    if s["id"] == "$set_id":
        print(s["tag_stage_a"])
        print(s["tag_stage_b"])
        print(s["tag_stage_c"])
        print(s["label"])
        print(s["polarity"])
        raise SystemExit(0)
raise SystemExit(f"unknown set id: $set_id")
PY
}

IFS=',' read -r -a SET_LIST <<< "$SETS"
for set_id in "${SET_LIST[@]}"; do
  set_id="$(echo "$set_id" | xargs)"
  [ -n "$set_id" ] || continue

  mapfile -t META < <(lookup_set "$set_id")
  TAG_A="${META[0]}"
  TAG_B="${META[1]}"
  TAG_C="${META[2]}"
  LABEL="${META[3]}"
  POLARITY="${META[4]}"

  echo ""
  echo "============================================================"
  echo "=== POOLED polarity set: $LABEL ($set_id / $POLARITY) [4B] ==="
  echo "=== one v on train pool → one (L,α) on val → B/C on test ==="
  echo "=== Stage A tag=$TAG_A  layers=L24/L23/L22 |α|≤6 ==="
  echo "============================================================"

  STAGE_A_DIR="$REPO/results/steering/xy_control/polarity_pool/$TAG_A"
  if [ "$SMOKE" = "1" ]; then
    STAGE_A_DIR="${STAGE_A_DIR}_smoke"
  fi

  if [ "$SKIP_STAGE_A" != "1" ]; then
    A_FLAGS=(
      --set-id "$set_id" --scale "$SCALE" --model "$MODEL"
      --device "$DEVICE" --dtype "$DTYPE"
      --config "$REPO/$CONFIG_REL"
      --sets "$REPO/$SETS_JSON_REL"
      --eval-split "$EVAL_SPLIT"
      --tag "$TAG_A"
    )
    if [ "$SMOKE" = "1" ]; then
      A_FLAGS+=(--smoke --limit-items "${N_ITEMS:-3}")
    fi
    "$PY" -m steering.xy_control.run_polarity_pool "${A_FLAGS[@]}"
  else
    echo "SKIP_STAGE_A=1 — expect $STAGE_A_DIR"
  fi

  if [ "$SMOKE" = "1" ] && [[ "$TAG_A" != *_smoke ]]; then
    TAG_A_EFF="${TAG_A}_smoke"
    TAG_B_EFF="${TAG_B}_smoke"
    TAG_C_EFF="${TAG_C}_smoke"
  else
    TAG_A_EFF="$TAG_A"
    TAG_B_EFF="$TAG_B"
    TAG_C_EFF="$TAG_C"
  fi
  STAGE_A_DIR="$REPO/results/steering/xy_control/polarity_pool/$TAG_A_EFF"
  STAGEC_KEEP="$REPO/results/steering/xy_control/stage_b/$TAG_B_EFF/stagec_keep.json"
  XY_DATA="$STAGE_A_DIR/xy_pairs_full_v1.json"

  if [ "$RUN_STAGE_B" = "1" ]; then
    echo "=== Stage B pooled [$LABEL] ==="
    B_FLAGS=(
      --scale "$SCALE" --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE"
      --stage-a-dir "$STAGE_A_DIR" --mmlu-parquet "$MMLU_PARQUET"
      --cap-loss-max "$CAP_LOSS_MAX" --tag "$TAG_B_EFF"
    )
    if [ "$SMOKE" = "1" ]; then
      B_FLAGS+=(--limit-mmlu 3)
    fi
    "$PY" -m steering.xy_control.run_polarity_pool_stageb "${B_FLAGS[@]}"
  elif [ ! -f "$STAGEC_KEEP" ]; then
    echo "Missing Stage B keep: $STAGEC_KEEP"
    exit 1
  else
    echo "=== reuse Stage B keep: $STAGEC_KEEP ==="
  fi

  echo "=== Stage C pooled [$LABEL] ==="
  C_FLAGS=(
    --scale "$SCALE" --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE"
    --stage-a-dir "$STAGE_A_DIR" --mmlu-parquet "$MMLU_PARQUET"
    --xy-data "$XY_DATA" --stagec-keep "$STAGEC_KEEP"
    --tag "$TAG_C_EFF"
  )
  if [ "$SMOKE" = "1" ]; then
    C_FLAGS+=(--limit-mmlu 3 --limit-items 2)
  fi
  "$PY" -m steering.xy_control.run_polarity_pool_stagec "${C_FLAGS[@]}"

  DST_PACK="/workspace/xy_control_polarity_pool_${set_id}_stage_abc_${SCALE}.tar.gz"
  if [ "$SMOKE" = "1" ]; then
    DST_PACK="/workspace/xy_control_polarity_pool_${set_id}_stage_abc_${SCALE}_smoke.tar.gz"
  fi
  tar_args=( -czf "$DST_PACK" -C "$REPO" )
  [ -d "$STAGE_A_DIR" ] && tar_args+=( "results/steering/xy_control/polarity_pool/$TAG_A_EFF" )
  [ -d "$REPO/results/steering/xy_control/stage_b/$TAG_B_EFF" ] && \
    tar_args+=( "results/steering/xy_control/stage_b/$TAG_B_EFF" )
  [ -d "$REPO/results/steering/xy_control/stage_c/$TAG_C_EFF" ] && \
    tar_args+=( "results/steering/xy_control/stage_c/$TAG_C_EFF" )
  if [ "${#tar_args[@]}" -gt 3 ]; then
    tar "${tar_args[@]}"
    ls -lh "$DST_PACK"
  fi
done

echo ""
echo "=== ALL POOLED polarity sets DONE (4B) ==="
ls -lh /workspace/xy_control_polarity_pool_*_stage_abc_4b*.tar.gz 2>/dev/null || true
echo "Download: /workspace/xy_control_polarity_pool_*_stage_abc_4b.tar.gz"
