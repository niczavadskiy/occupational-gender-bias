#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Full sequential erase gate on a fresh Vast instance:
#   setup → second-hit → conditional INLP + stacked Stage A → packs
#
#   export HF_TOKEN=hf_xxx
#   bash steering/scripts/vast_sequential_erase_full_instance.sh
#
# SMOKE=1 for a cheap end-to-end check first.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="${REPO:-/workspace/occupational-gender-bias}"
cd "$REPO"

if [ -f "$REPO/scripts/setup_instance.sh" ]; then
  bash "$REPO/scripts/setup_instance.sh" || true
fi
if [ -f "$REPO/scripts/ensure_steering_env.sh" ]; then
  # shellcheck disable=SC1091
  source "$REPO/scripts/ensure_steering_env.sh"
  ensure_steering_env || true
fi

export SMOKE="${SMOKE:-0}"
HIT_TAG="${TAG_HIT:-second_hit_v1}"
COND_TAG="${TAG_COND:-cond_under_l15_v1}"

echo "========== Phase 1: second-hit =========="
TAG="$HIT_TAG" bash "$REPO/steering/scripts/vast_second_hit_and_pack.sh"

echo "========== Phase 2: conditional INLP + stacked Stage A =========="
TAG="$COND_TAG" bash "$REPO/steering/scripts/vast_conditional_inlp_stagea_and_pack.sh"

echo "========== ALL DONE =========="
ls -lh /workspace/second_hit_*.tar.gz /workspace/stacked_inlp_*.tar.gz 2>/dev/null || true
