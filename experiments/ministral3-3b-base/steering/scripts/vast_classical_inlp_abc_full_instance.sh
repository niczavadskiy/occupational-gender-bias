#!/usr/bin/env bash
# Classical INLP full A→B→C for Ministral 3: gender_prob then slot_prob.
# Default DTYPE=bfloat16.
#
# Usage (Vast):
#   export HF_TOKEN=hf_xxx
#   SCALE=ministral3_3b bash experiments/ministral3-3b-base/steering/scripts/vast_classical_inlp_abc_full_instance.sh
#   SCALE=ministral3_8b bash experiments/ministral3-8b-base/steering/scripts/vast_classical_inlp_abc_full_instance.sh
#
# Env: SCALE MODEL AXES=gender,slot SMOKE=1 SKIP_BUILD=1 SKIP_STAGE_A=1
#      RUN_STAGE_B=1 RUN_STAGE_C=1 LAYERS_GENDER LAYERS_SLOT DTYPE=bfloat16
# Defaults (3b): LAYERS_GENDER=23,22,21 LAYERS_SLOT=24,23,22
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?export HF_TOKEN=hf_xxx}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"

WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="${REPO_DIR:-occupational-gender-bias}"
REPO="${REPO:-$WORKDIR/$REPO_DIR}"
BRANCH="${BRANCH:-main}"
GH_REPO="${GH_REPO:-niczavadskiy/occupational-gender-bias}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
SMOKE="${SMOKE:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_STAGE_A="${SKIP_STAGE_A:-0}"
RUN_STAGE_B="${RUN_STAGE_B:-1}"
RUN_STAGE_C="${RUN_STAGE_C:-1}"
AXES="${AXES:-gender,slot}"
RANKS="${RANKS:-1,4,8,16}"
ALPHAS="${ALPHAS:-1.0}"
K_MAX="${K_MAX:-32}"

SCALE="${SCALE:?set SCALE=ministral3_3b or ministral3_8b}"
case "$SCALE" in
  ministral3_3b)
    MODEL="${MODEL:-mistralai/Ministral-3-3B-Base-2512}"
    LAYERS_GENDER="${LAYERS_GENDER:-23,22,21}"
    LAYERS_SLOT="${LAYERS_SLOT:-24,23,22}"
    EXP_REL="experiments/ministral3-3b-base"
    ;;
  ministral3_8b)
    MODEL="${MODEL:-mistralai/Ministral-3-8B-Base-2512}"
    LAYERS_GENDER="${LAYERS_GENDER:-24,23,22}"
    LAYERS_SLOT="${LAYERS_SLOT:-31,30,29}"
    EXP_REL="experiments/ministral3-8b-base"
    ;;
  *)
    echo "SCALE must be ministral3_3b or ministral3_8b (got $SCALE)"; exit 1
    ;;
esac

OUT_ROOT="${OUT_ROOT:-$REPO/$EXP_REL/results/steering}"
SAMPLE="${SAMPLE:-$REPO/steering/samples/h1_stagea_sample_v1.json}"
SAMPLE_C="${SAMPLE_C:-$REPO/steering/samples/inlp_stagec_sample_v1.json}"
MMLU_PARQUET="${MMLU_PARQUET:-$REPO/steering/.cache/mmlu_pro_test.parquet}"

echo "=== [0] classical INLP A→B→C scale=$SCALE model=$MODEL dtype=$DTYPE axes=$AXES ==="
nvidia-smi -L || echo "WARN: nvidia-smi failed"
cd "$WORKDIR"

echo "=== [1] clone / sync $GH_REPO @$BRANCH ==="
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" remote set-branches --add origin "$BRANCH" 2>/dev/null || true
  git -C "$REPO_DIR" fetch --depth 1 origin "$BRANCH"
  git -C "$REPO_DIR" checkout -B "$BRANCH" "origin/$BRANCH"
  git -C "$REPO_DIR" reset --hard "origin/$BRANCH"
else
  git clone --depth 1 --branch "$BRANCH" "https://github.com/${GH_REPO}.git" "$REPO_DIR"
fi
export REPO
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "=== [2] setup_instance + steering env ==="
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
"$PY" -m pip install -U 'transformers>=4.57.0' mistral-common

mkdir -p "$(dirname "$MMLU_PARQUET")" "$OUT_ROOT"
if [ ! -f "$MMLU_PARQUET" ]; then
  curl -L --fail --retry 3 -o "$MMLU_PARQUET" \
    "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"
fi
"$PY" -c "import pyarrow" >/dev/null 2>&1 || "$PY" -m pip install -U pyarrow
"$PY" -c "import sklearn" >/dev/null 2>&1 || "$PY" -m pip install -q scikit-learn

test -f "$SAMPLE" || { echo "MISSING $SAMPLE"; exit 1; }
test -f "$SAMPLE_C" || { echo "MISSING $SAMPLE_C"; exit 1; }

run_axis() {
  local axis="$1"
  local target layers primary sub_tag tag_a tag_b tag_c
  if [ "$axis" = "gender" ]; then
    target=gender_prob
    layers="$LAYERS_GENDER"
    primary=gender
    sub_tag="${SCALE}_gender_v1"
    tag_a="inlp_${SCALE}_gender_prob_a_v1"
    tag_b="inlp_${SCALE}_gender_prob_b_v1"
    tag_c="inlp_${SCALE}_gender_prob_c_v1"
  else
    target=slot_prob
    layers="$LAYERS_SLOT"
    primary=slot
    sub_tag="${SCALE}_slot_v1"
    tag_a="inlp_${SCALE}_slot_prob_a_v1"
    tag_b="inlp_${SCALE}_slot_prob_b_v1"
    tag_c="inlp_${SCALE}_slot_prob_c_v1"
  fi

  local sub="$REPO/steering/subspaces/inlp_${target}_${sub_tag}.npz"
  local ranks="$RANKS" kmax="$K_MAX"
  local smoke_a=() smoke_b=() smoke_c=() n_items=()
  if [ "$SMOKE" = "1" ]; then
    ranks=4
    kmax=4
    layers="${layers%%,*}"
    n_items=(--n-items 2)
    smoke_a=(--limit-items 2)
    smoke_b=(--limit-mmlu 3)
    smoke_c=(--limit-mmlu 3 --limit-items 2)
    tag_a="${tag_a}_smoke"
    tag_b="${tag_b}_smoke"
    tag_c="${tag_c}_smoke"
    echo "SMOKE=1 → layers=$layers ranks=$ranks"
  fi

  echo ""
  echo "========== AXIS $axis ($target) layers=$layers dtype=$DTYPE =========="

  if [ "$SKIP_STAGE_A" != "1" ]; then
    if [ "$SKIP_BUILD" != "1" ]; then
      echo "=== Phase A live fit [$sub_tag] ==="
      "$PY" -m steering.build_inlp_live_subspace \
        --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" \
        --sample "$SAMPLE" --target "$target" --layers "$layers" \
        --k-max "$kmax" --ridge-alpha 1.0 --tag "$sub_tag" \
        --out-dir "$REPO/steering/subspaces" --log-every 40 \
        "${n_items[@]}"
    else
      echo "SKIP_BUILD=1 → reuse $sub"
    fi
    test -f "$sub" || { echo "MISSING $sub"; exit 1; }

    echo "=== Stage A [$tag_a] ==="
    "$PY" -m steering.run_inlp_stagea \
      --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" \
      --subspaces "$sub" --sample "$SAMPLE" \
      --layers "$layers" --ranks "$ranks" --alphas "$ALPHAS" \
      --primary-axis "$primary" --tag "$tag_a" \
      --out-root "$OUT_ROOT" --log-every 100 \
      "${smoke_a[@]}"
  fi

  local stage_a_dir="$OUT_ROOT/inlp_stage_a/$tag_a"
  test -d "$stage_a_dir" || { echo "MISSING Stage A $stage_a_dir"; exit 1; }

  if [ "$RUN_STAGE_B" = "1" ]; then
    echo "=== Stage B [$tag_b] ==="
    "$PY" -m steering.run_inlp_stageb \
      --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" \
      --subspaces "$sub" --from-stage-a "$stage_a_dir" \
      --out-root "$OUT_ROOT" --tag "$tag_b" --log-every 100 \
      --mmlu-parquet "$MMLU_PARQUET" \
      "${smoke_b[@]}"
  fi

  local stage_b_dir="$OUT_ROOT/inlp_stage_b/$tag_b"
  if [ "$RUN_STAGE_C" = "1" ]; then
    test -d "$stage_b_dir" || { echo "MISSING Stage B $stage_b_dir"; exit 1; }
    echo "=== Stage C [$tag_c] ==="
    "$PY" -m steering.run_inlp_stagec \
      --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" \
      --subspaces "$sub" --sample "$SAMPLE_C" \
      --from-stage-b "$stage_b_dir" \
      --out-root "$OUT_ROOT" --tag "$tag_c" --log-every 100 \
      --mmlu-parquet "$MMLU_PARQUET" \
      "${smoke_c[@]}"
  fi

  local pack="/workspace/inlp_classical_${SCALE}_${axis}_abc.tar.gz"
  if [ "$SMOKE" = "1" ]; then
    pack="/workspace/inlp_classical_${SCALE}_${axis}_abc_smoke.tar.gz"
  fi
  echo "=== pack → $pack ==="
  tar -czf "$pack" -C "$REPO" \
    "steering/subspaces/inlp_${target}_${sub_tag}.npz" \
    "steering/subspaces/inlp_${target}_${sub_tag}.json" \
    "$EXP_REL/results/steering/inlp_stage_a/$tag_a" \
    "$EXP_REL/results/steering/inlp_stage_b/$tag_b" \
    "$EXP_REL/results/steering/inlp_stage_c/$tag_c" \
    || echo "WARN: partial pack $axis"
  ls -lh "$pack" || true
}

IFS=',' read -r -a AXIS_LIST <<< "$AXES"
for ax in "${AXIS_LIST[@]}"; do
  ax="$(echo "$ax" | tr -d '[:space:]')"
  [ -n "$ax" ] || continue
  case "$ax" in
    gender|slot) run_axis "$ax" ;;
    *) echo "unknown axis $ax (want gender|slot)"; exit 1 ;;
  esac
done

echo "=== classical INLP A→B→C DONE ($SCALE, $DTYPE) ==="
ls -lh /workspace/inlp_classical_${SCALE}_*_abc*.tar.gz 2>/dev/null || true
