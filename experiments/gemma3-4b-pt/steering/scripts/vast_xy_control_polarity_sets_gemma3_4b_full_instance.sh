#!/usr/bin/env bash
# XY-control polarity pool A→B→C · Gemma (gender only).
# Skips polarity sets with n_soc=0 / empty slugs.
#
#   export HF_TOKEN=hf_xxx
#   SCALE=gemma3_4b bash experiments/gemma3-4b-pt/steering/scripts/vast_xy_control_polarity_sets_gemma3_4b_full_instance.sh
#   (also works with SCALE=gemma3_1b if pointed at the 1B twin script)
#
# Env: SETS=promale,profemale SKIP_PIP=1 SKIP_STAGE_A=1 RUN_STAGE_B=0 SMOKE=1 CAP_LOSS_MAX=0.03
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?export HF_TOKEN=hf_xxx}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"

WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="${REPO_DIR:-occupational-gender-bias}"
REPO="${REPO:-$WORKDIR/$REPO_DIR}"
BRANCH="${BRANCH:-main}"
GH_REPO="${GH_REPO:-niczavadskiy/occupational-gender-bias}"
SETS="${SETS:-promale,profemale}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
CAP_LOSS_MAX="${CAP_LOSS_MAX:-0.03}"
SMOKE="${SMOKE:-0}"
SKIP_STAGE_A="${SKIP_STAGE_A:-0}"
RUN_STAGE_B="${RUN_STAGE_B:-1}"

SCALE="${SCALE:?set SCALE=gemma3_1b or gemma3_4b}"
case "$SCALE" in
  gemma3_1b)
    MODEL="${MODEL:-google/gemma-3-1b-pt}"
    EXP_REL="experiments/gemma3-1b-pt"
    ;;
  gemma3_4b)
    MODEL="${MODEL:-google/gemma-3-4b-pt}"
    EXP_REL="experiments/gemma3-4b-pt"
    ;;
  *) echo "bad SCALE=$SCALE"; exit 1 ;;
esac

SETS_JSON_REL="$EXP_REL/steering/domains/polarity_sets_${SCALE}_v1.json"
CONFIG_REL="$EXP_REL/steering/xy_control/configs/xy_control_${SCALE}_peak_prepeak_a6.yaml"
MMLU_PARQUET="${MMLU_PARQUET:-$REPO/steering/.cache/mmlu_pro_test.parquet}"

echo "=== [0] pool XY A→B→C scale=$SCALE ==="
nvidia-smi -L || true
cd "$WORKDIR"

echo "=== [1] clone / sync ==="
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" fetch --depth 1 origin "$BRANCH"
  git -C "$REPO_DIR" checkout -B "$BRANCH" "origin/$BRANCH"
  git -C "$REPO_DIR" reset --hard "origin/$BRANCH"
else
  git clone --depth 1 --branch "$BRANCH" "https://github.com/${GH_REPO}.git" "$REPO_DIR"
fi
export REPO
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "=== [2] env ==="
export SKIP_FLA="${SKIP_FLA:-1}"
PULL_CACHE=0 bash scripts/setup_instance.sh
# shellcheck disable=SC1091
source "$REPO/scripts/ensure_steering_env.sh"
if [ "${SKIP_PIP:-0}" = "1" ]; then SKIP_PIP=1 ensure_steering_env; else ensure_steering_env; fi
export PY
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

need=("$SETS_JSON_REL" "$CONFIG_REL"
  steering/xy_control/run_polarity_pool.py
  steering/xy_control/run_polarity_pool_stageb.py
  steering/xy_control/run_polarity_pool_stagec.py
  steering/xy_control/polarity_pool.py)
miss=0
for f in "${need[@]}"; do
  if [ -f "$f" ]; then echo "  OK $f"; else echo "  MISSING $f"; miss=1; fi
done
[ "$miss" = "0" ] || exit 1

"$PY" -m steering.xy_control.test_polarity_pool
"$PY" -m steering.xy_control.build_dataset
"$PY" -m steering.xy_control.test_dataset

mkdir -p "$(dirname "$MMLU_PARQUET")"
[ -f "$MMLU_PARQUET" ] || curl -L --fail --retry 3 -o "$MMLU_PARQUET" \
  "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"
"$PY" -c "import pyarrow" >/dev/null 2>&1 || "$PY" -m pip install -U pyarrow

set_n_soc() {
  local set_id="$1"
  "$PY" - <<PY
import json
from pathlib import Path
doc = json.loads(Path("$SETS_JSON_REL").read_text(encoding="utf-8"))
for s in doc["sets"]:
    if s["id"] == "$set_id":
        n = int(s.get("n_soc") or 0)
        slugs = s.get("slugs") or []
        print(n if n else len(slugs))
        raise SystemExit(0)
raise SystemExit(f"unknown set: $set_id")
PY
}

lookup_set() {
  local set_id="$1"
  "$PY" - <<PY
import json
from pathlib import Path
doc = json.loads(Path("$SETS_JSON_REL").read_text(encoding="utf-8"))
for s in doc["sets"]:
    if s["id"] == "$set_id":
        print(s["tag_stage_a"]); print(s["tag_stage_b"]); print(s["tag_stage_c"])
        print(s["label"]); print(s["polarity"])
        raise SystemExit(0)
raise SystemExit(f"unknown set id: $set_id")
PY
}

IFS=',' read -r -a SET_LIST <<< "$SETS"
for set_id in "${SET_LIST[@]}"; do
  set_id="$(echo "$set_id" | xargs)"
  [ -n "$set_id" ] || continue
  n_soc="$(set_n_soc "$set_id")"
  if [ "$n_soc" = "0" ]; then
    echo ""
    echo "========== SKIP XY set: $set_id (empty FDR / n_soc=0) =========="
    continue
  fi

  mapfile -t META < <(lookup_set "$set_id")
  TAG_A="${META[0]}"; TAG_B="${META[1]}"; TAG_C="${META[2]}"
  LABEL="${META[3]}"; POLARITY="${META[4]}"

  echo ""
  echo "=== POOLED XY: $LABEL ($set_id / $POLARITY) [$SCALE] n_soc=$n_soc ==="

  STAGE_A_DIR="$REPO/results/steering/xy_control/polarity_pool/$TAG_A"
  [ "$SMOKE" = "1" ] && STAGE_A_DIR="${STAGE_A_DIR}_smoke"

  if [ "$SKIP_STAGE_A" != "1" ]; then
    A_FLAGS=(--set-id "$set_id" --scale "$SCALE" --model "$MODEL"
      --device "$DEVICE" --dtype "$DTYPE"
      --config "$REPO/$CONFIG_REL" --sets "$REPO/$SETS_JSON_REL"
      --eval-split "$EVAL_SPLIT" --tag "$TAG_A")
    [ "$SMOKE" = "1" ] && A_FLAGS+=(--smoke --limit-items "${N_ITEMS:-3}")
    "$PY" -m steering.xy_control.run_polarity_pool "${A_FLAGS[@]}"
  fi

  if [ "$SMOKE" = "1" ] && [[ "$TAG_A" != *_smoke ]]; then
    TAG_A_EFF="${TAG_A}_smoke"; TAG_B_EFF="${TAG_B}_smoke"; TAG_C_EFF="${TAG_C}_smoke"
  else
    TAG_A_EFF="$TAG_A"; TAG_B_EFF="$TAG_B"; TAG_C_EFF="$TAG_C"
  fi
  STAGE_A_DIR="$REPO/results/steering/xy_control/polarity_pool/$TAG_A_EFF"
  STAGEC_KEEP="$REPO/results/steering/xy_control/stage_b/$TAG_B_EFF/stagec_keep.json"
  XY_DATA="$STAGE_A_DIR/xy_pairs_full_v1.json"

  if [ "$RUN_STAGE_B" = "1" ]; then
    B_FLAGS=(--scale "$SCALE" --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE"
      --stage-a-dir "$STAGE_A_DIR" --mmlu-parquet "$MMLU_PARQUET"
      --cap-loss-max "$CAP_LOSS_MAX" --tag "$TAG_B_EFF")
    [ "$SMOKE" = "1" ] && B_FLAGS+=(--limit-mmlu 3)
    "$PY" -m steering.xy_control.run_polarity_pool_stageb "${B_FLAGS[@]}"
  elif [ ! -f "$STAGEC_KEEP" ]; then
    echo "Missing Stage B keep: $STAGEC_KEEP"; exit 1
  fi

  C_FLAGS=(--scale "$SCALE" --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE"
    --stage-a-dir "$STAGE_A_DIR" --mmlu-parquet "$MMLU_PARQUET"
    --xy-data "$XY_DATA" --stagec-keep "$STAGEC_KEEP" --tag "$TAG_C_EFF")
  [ "$SMOKE" = "1" ] && C_FLAGS+=(--limit-mmlu 3 --limit-items 2)
  "$PY" -m steering.xy_control.run_polarity_pool_stagec "${C_FLAGS[@]}"

  DST_PACK="/workspace/xy_control_polarity_pool_${set_id}_stage_abc_${SCALE}.tar.gz"
  [ "$SMOKE" = "1" ] && DST_PACK="/workspace/xy_control_polarity_pool_${set_id}_stage_abc_${SCALE}_smoke.tar.gz"
  tar_args=(-czf "$DST_PACK" -C "$REPO")
  [ -d "$STAGE_A_DIR" ] && tar_args+=("results/steering/xy_control/polarity_pool/$TAG_A_EFF")
  [ -d "$REPO/results/steering/xy_control/stage_b/$TAG_B_EFF" ] && tar_args+=("results/steering/xy_control/stage_b/$TAG_B_EFF")
  [ -d "$REPO/results/steering/xy_control/stage_c/$TAG_C_EFF" ] && tar_args+=("results/steering/xy_control/stage_c/$TAG_C_EFF")
  if [ "${#tar_args[@]}" -gt 3 ]; then tar "${tar_args[@]}"; ls -lh "$DST_PACK"; fi
done

echo "=== pool XY DONE ($SCALE) ==="
ls -lh /workspace/xy_control_polarity_pool_*_stage_abc_${SCALE}*.tar.gz 2>/dev/null || true
