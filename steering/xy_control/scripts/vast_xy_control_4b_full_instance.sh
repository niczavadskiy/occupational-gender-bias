#!/usr/bin/env bash
# Fresh Vast → clone → smoke → full XY-control Stage A on Qwen3.5-4B-Base → pack.
#
#   export HF_TOKEN=hf_xxx
#   bash steering/xy_control/scripts/vast_xy_control_4b_full_instance.sh
#
# Пояс L20–26, якорь L23 (не копировать 2B). Pack:
#   /workspace/xy_control_stage_a_xy_4b_a_v1.tar.gz
#
# Env: SKIP_SMOKE=1  SKIP_FULL=1  SKIP_PIP=1  BRANCH TAG MMLU
# Если setup_instance падает на causal-conv1d: SKIP_PIP=1.
set -euo pipefail

export SCALE=4b
export MODEL="${MODEL:-Qwen/Qwen3.5-4B-Base}"
export TAG="${TAG:-xy_4b_a_v1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/vast_xy_control_full_instance.sh" "$@"
