#!/usr/bin/env bash
# 2B XY-control: Stage A → B → C for pro-male then pro-female polarity sets.
#
# Frozen sets: steering/xy_control/domains/polarity_sets_2b_v1.json
# Config:      steering/xy_control/configs/xy_control_2b_L15_a6.yaml
#
#   export HF_TOKEN=hf_xxx
#   BRANCH=qwen_2b_experiments bash steering/xy_control/scripts/vast_xy_control_polarity_sets_full_instance.sh
#
# Env:
#   BRANCH          default qwen_2b_experiments
#   SETS            default promale,profemale  (comma ids from polarity_sets JSON)
#   SKIP_PIP=1      skip setup pip
#   SKIP_STAGE_A=1  reuse existing Stage A dirs
#   RUN_STAGE_B=0   reuse stagec_keep from Stage B
#   CAP_LOSS_MAX    default 0.03
#   SMOKE=1         tiny A/B/C smoke
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?export HF_TOKEN=hf_xxx}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"

WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="${REPO_DIR:-occupational-gender-bias}"
REPO="${REPO:-$WORKDIR/$REPO_DIR}"
BRANCH="${BRANCH:-qwen_2b_experiments}"
GH_REPO="${GH_REPO:-niczavadskiy/occupational-gender-bias}"
SETS="${SETS:-promale,profemale}"
SCALE=2b
MODEL="${MODEL:-Qwen/Qwen3.5-2B-Base}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
CAP_LOSS_MAX="${CAP_LOSS_MAX:-0.03}"
SMOKE="${SMOKE:-0}"
SKIP_STAGE_A="${SKIP_STAGE_A:-0}"
RUN_STAGE_B="${RUN_STAGE_B:-1}"
SETS_JSON_REL="steering/xy_control/domains/polarity_sets_2b_v1.json"
CONFIG_REL="steering/xy_control/configs/xy_control_2b_L15_a6.yaml"

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
  steering/xy_control/scripts/vast_xy_control_per_soc_and_pack.sh
  steering/xy_control/scripts/vast_xy_control_per_soc_stagebc_and_pack.sh
  steering/xy_control/run_per_soc.py
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

chmod +x steering/xy_control/scripts/vast_xy_control_per_soc_and_pack.sh
chmod +x steering/xy_control/scripts/vast_xy_control_per_soc_stagebc_and_pack.sh
chmod +x steering/xy_control/scripts/vast_xy_control_polarity_sets_full_instance.sh

"$PY" -m steering.xy_control.test_stagebc

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
        print(",".join(s["slugs"]))
        print(s["label"])
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
  ONLY_SOC="${META[3]}"
  LABEL="${META[4]}"

  echo ""
  echo "============================================================"
  echo "=== polarity set: $LABEL ($set_id) ==="
  echo "=== Stage A tag=$TAG_A  only_soc=$ONLY_SOC ==="
  echo "============================================================"

  if [ "$SKIP_STAGE_A" != "1" ]; then
    PY="$PY" SMOKE="$SMOKE" SCALE="$SCALE" TAG="$TAG_A" MODEL="$MODEL" \
      DEVICE="$DEVICE" DTYPE="$DTYPE" EVAL_SPLIT="$EVAL_SPLIT" \
      ONLY_SOC="$ONLY_SOC" CONFIG="$REPO/$CONFIG_REL" REPO="$REPO" \
      bash steering/xy_control/scripts/vast_xy_control_per_soc_and_pack.sh
  else
    echo "SKIP_STAGE_A=1 — expect $REPO/results/steering/xy_control/per_soc/$TAG_A"
  fi

  STAGE_A_DIR_TAG="$TAG_A"
  B_DIR_TAG="$TAG_B"
  C_DIR_TAG="$TAG_C"
  if [ "$SMOKE" = "1" ]; then
    STAGE_A_DIR_TAG="${TAG_A}_smoke"
    B_DIR_TAG="${TAG_B}_smoke"
    C_DIR_TAG="${TAG_C}_smoke"
  fi

  echo "=== Stage B/C for $LABEL ==="
  PY="$PY" SCALE="$SCALE" MODEL="$MODEL" DEVICE="$DEVICE" DTYPE="$DTYPE" \
    CAP_LOSS_MAX="$CAP_LOSS_MAX" SMOKE="$SMOKE" RUN_STAGE_B="$RUN_STAGE_B" \
    STAGE_A_TAG="$STAGE_A_DIR_TAG" B_TAG="$B_DIR_TAG" C_TAG="$C_DIR_TAG" REPO="$REPO" \
    bash steering/xy_control/scripts/vast_xy_control_per_soc_stagebc_and_pack.sh

  DST_PACK="/workspace/xy_control_polarity_${set_id}_stage_abc_${SCALE}.tar.gz"
  if [ "$SMOKE" = "1" ]; then
    DST_PACK="/workspace/xy_control_polarity_${set_id}_stage_abc_${SCALE}_smoke.tar.gz"
  fi
  tar_args=( -czf "$DST_PACK" -C "$REPO" )
  [ -d "$REPO/results/steering/xy_control/per_soc/$STAGE_A_DIR_TAG" ] && \
    tar_args+=( "results/steering/xy_control/per_soc/$STAGE_A_DIR_TAG" )
  [ -d "$REPO/results/steering/xy_control/stage_b/$B_DIR_TAG" ] && \
    tar_args+=( "results/steering/xy_control/stage_b/$B_DIR_TAG" )
  [ -d "$REPO/results/steering/xy_control/stage_c/$C_DIR_TAG" ] && \
    tar_args+=( "results/steering/xy_control/stage_c/$C_DIR_TAG" )
  if [ "${#tar_args[@]}" -gt 3 ]; then
    tar "${tar_args[@]}"
    ls -lh "$DST_PACK"
  else
    echo "WARN: nothing to pack for $set_id"
  fi
done

echo ""
echo "=== ALL polarity sets DONE ==="
ls -lh /workspace/xy_control_polarity_*_stage_abc_*.tar.gz 2>/dev/null || true
ls -lh /workspace/xy_control_per_soc_xy_2b_L15_a6_pro*.tar.gz 2>/dev/null || true
echo "Download polarity packs from /workspace/xy_control_polarity_*_stage_abc_2b.tar.gz"
