#!/usr/bin/env bash
# Fresh Vast → clone → per-SOC XY-control on Qwen3.5-4B-Base → pack.
#
#   export HF_TOKEN=hf_xxx
#   bash steering/xy_control/scripts/vast_xy_control_per_soc_4b_full_instance.sh
#
# Env: SKIP_SMOKE=1  SKIP_FULL=1  SKIP_PIP=1  BRANCH EVAL_SPLIT
set -euo pipefail

export SCALES=4b
export MODEL="${MODEL:-Qwen/Qwen3.5-4B-Base}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/vast_xy_control_per_soc_full_instance.sh" "$@"
