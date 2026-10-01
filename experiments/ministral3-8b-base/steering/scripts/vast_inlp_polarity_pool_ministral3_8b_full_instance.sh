#!/usr/bin/env bash
# Polarity-pooled INLP A→B→C · Ministral 3 (gender only).
# Skips polarity sets with n_soc=0 / empty slugs.
# Default DTYPE=bfloat16. After ensure_steering_env: transformers>=4.57 + mistral-common.
#
#   export HF_TOKEN=hf_xxx
#   SCALE=ministral3_8b bash experiments/ministral3-8b-base/steering/scripts/vast_inlp_polarity_pool_ministral3_8b_full_instance.sh
#   SCALE=ministral3_3b bash experiments/ministral3-3b-base/steering/scripts/vast_inlp_polarity_pool_ministral3_3b_full_instance.sh
#
# Env: SETS=promale,profemale SKIP_PIP=1 SKIP_STAGE_A=1 SKIP_FIT=1 RUN_STAGE_B=0 SMOKE=1
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
DTYPE="${DTYPE:-bfloat16}"
SMOKE="${SMOKE:-0}"
SKIP_STAGE_A="${SKIP_STAGE_A:-0}"
SKIP_FIT="${SKIP_FIT:-0}"
RUN_STAGE_B="${RUN_STAGE_B:-1}"
RUN_STAGE_C="${RUN_STAGE_C:-1}"

SCALE="${SCALE:?set SCALE=ministral3_3b or ministral3_8b}"
case "$SCALE" in
  ministral3_3b)
    MODEL="${MODEL:-mistralai/Ministral-3-3B-Base-2512}"
    EXP_REL="experiments/ministral3-3b-base"
    ;;
  ministral3_8b)
    MODEL="${MODEL:-mistralai/Ministral-3-8B-Base-2512}"
    EXP_REL="experiments/ministral3-8b-base"
    ;;
  *) echo "SCALE must be ministral3_3b or ministral3_8b (got $SCALE)"; exit 1 ;;
esac

SETS_JSON_REL="$EXP_REL/steering/domains/inlp_polarity_sets_${SCALE}_v1.json"
CONFIG_REL="$EXP_REL/steering/configs/inlp_polarity_pool_${SCALE}_peak_prepeak.yaml"
MMLU_PARQUET="${MMLU_PARQUET:-$REPO/steering/.cache/mmlu_pro_test.parquet}"

echo "=== [0] pool INLP A→B→C scale=$SCALE ==="
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
if [ -z "${PY:-}" ] && [ -x /opt/conda/bin/python ]; then
  export PY=/opt/conda/bin/python
fi
PULL_CACHE=0 bash scripts/setup_instance.sh
# shellcheck disable=SC1091
source "$REPO/scripts/ensure_steering_env.sh"
if [ "${SKIP_PIP:-0}" = "1" ]; then SKIP_PIP=1 ensure_steering_env; else ensure_steering_env; fi
export PY
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
echo "  using PY=$PY"
if ! "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"; then
  echo "torch missing/broken in $PY — install_vast_env --force (PIN_VAST_ENV ignored)"
  SKIP_FLA=1 PIN_VAST_ENV=1 "$PY" "$REPO/scripts/install_vast_env.py" --force
fi
"$PY" -c "import torch" || { echo "FATAL: no torch in $PY"; exit 1; }
"$PY" -m pip install -U 'transformers>=4.57.0' mistral-common
"$PY" -c "import torch" || { echo "FATAL: torch disappeared after mistral-common install"; exit 1; }

need=("$SETS_JSON_REL" "$CONFIG_REL" steering/polarity_pool_inlp.py
  steering/build_polarity_pool_inlp_sample.py steering/build_polarity_pool_inlp_subspace.py
  steering/run_polarity_pool_inlp_stagea.py steering/run_polarity_pool_inlp_stageb.py
  steering/run_polarity_pool_inlp_stagec.py steering/xy_control/data/xy_pairs_full_v1.json)
miss=0
for f in "${need[@]}"; do
  if [ -f "$f" ]; then echo "  OK $f"; else echo "  MISSING $f"; miss=1; fi
done
[ "$miss" = "0" ] || exit 1

"$PY" -m steering.test_polarity_pool_inlp
"$PY" -m steering.xy_control.build_dataset

mkdir -p "$(dirname "$MMLU_PARQUET")"
[ -f "$MMLU_PARQUET" ] || curl -L --fail --retry 3 -o "$MMLU_PARQUET" \
  "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"
"$PY" -c "import pyarrow" >/dev/null 2>&1 || "$PY" -m pip install -U pyarrow
"$PY" -c "import yaml" >/dev/null 2>&1 || "$PY" -m pip install -U pyyaml

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
  n_soc="$(set_n_soc "$sid")"
  if [ "$n_soc" = "0" ]; then
    echo ""
    echo "========== SKIP polarity set: $sid (empty FDR / n_soc=0) =========="
    continue
  fi

  echo ""
  echo "========== polarity set: $sid ($SCALE, n_soc=$n_soc) =========="
  mapfile -t META < <(lookup_sample_stem "$sid")
  SAMPLE_STEM="${META[0]}"
  SUB_TAG="${META[1]}"
  SAMPLE_BASE="${SAMPLE_STEM%_v1}"

  "$PY" -m steering.build_polarity_pool_inlp_sample \
    --set-id "$sid" --sets "$REPO/$SETS_JSON_REL" --config "$REPO/$CONFIG_REL" --scale "$SCALE"

  if [ "$SKIP_STAGE_A" != "1" ]; then
    smoke_flags=()
    [ "$SMOKE" = "1" ] && smoke_flags=(--smoke)
    if [ "$SKIP_FIT" != "1" ]; then
      "$PY" -m steering.build_polarity_pool_inlp_subspace \
        --set-id "$sid" --sets "$REPO/$SETS_JSON_REL" --config "$REPO/$CONFIG_REL" \
        --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" "${smoke_flags[@]}"
    fi
    "$PY" -m steering.run_polarity_pool_inlp_stagea \
      --set-id "$sid" --sets "$REPO/$SETS_JSON_REL" --config "$REPO/$CONFIG_REL" \
      --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" "${smoke_flags[@]}"
  fi

  if [ "$RUN_STAGE_B" = "1" ]; then
    smoke_flags=(); [ "$SMOKE" = "1" ] && smoke_flags=(--smoke)
    "$PY" -m steering.run_polarity_pool_inlp_stageb \
      --set-id "$sid" --sets "$REPO/$SETS_JSON_REL" --config "$REPO/$CONFIG_REL" \
      --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" "${smoke_flags[@]}"
  fi

  if [ "$RUN_STAGE_C" = "1" ]; then
    smoke_flags=(); [ "$SMOKE" = "1" ] && smoke_flags=(--smoke)
    "$PY" -m steering.run_polarity_pool_inlp_stagec \
      --set-id "$sid" --sets "$REPO/$SETS_JSON_REL" --config "$REPO/$CONFIG_REL" \
      --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" "${smoke_flags[@]}"
  fi

  pack="/workspace/inlp_polarity_pool_${sid}_stage_abc_${SCALE}.tar.gz"
  tar -czf "$pack" \
    "steering/samples/${SAMPLE_STEM}.json" \
    "steering/samples/${SAMPLE_BASE}_train_v1.json" \
    "steering/samples/${SAMPLE_BASE}_val_v1.json" \
    "steering/samples/${SAMPLE_BASE}_test_v1.json" \
    "steering/subspaces/inlp_gender_prob_${SUB_TAG}.npz" \
    "steering/subspaces/inlp_gender_prob_${SUB_TAG}.json" \
    results/steering/inlp_polarity_pool \
    || echo "WARN: partial pack $sid"
  ls -lh "$pack" || true
done

echo "=== pool INLP DONE ($SCALE) ==="
ls -lh /workspace/inlp_polarity_pool_*_stage_abc_${SCALE}.tar.gz 2>/dev/null || true
