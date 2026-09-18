#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 4B prob: peak Stage A (gender→slot) then pre-peak (gender→slot) + packs.
#
#   export HF_TOKEN=hf_xxx
#   cd /workspace/occupational-gender-bias && git pull
#   bash experiments/qwen35-4b-base/steering/scripts/vast_prob_peak_and_prepeak_4b_and_pack.sh
#
# Env: SMOKE=1 for tiny run; MODE=peak|prepeak|both (default both)
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-both}"

echo "========== 4B prob MODE=$MODE =========="

case "$MODE" in
  peak)
    bash "$SCRIPTS/vast_inlp_gender_prob_stagea_and_pack.sh" "$@"
    bash "$SCRIPTS/vast_inlp_slot_prob_stagea_and_pack.sh" "$@"
    ;;
  prepeak)
    bash "$SCRIPTS/vast_sequential_pre_peak_4b_and_pack.sh" "$@"
    ;;
  both)
    bash "$SCRIPTS/vast_inlp_gender_prob_stagea_and_pack.sh" "$@"
    bash "$SCRIPTS/vast_inlp_slot_prob_stagea_and_pack.sh" "$@"
    bash "$SCRIPTS/vast_sequential_pre_peak_4b_and_pack.sh" "$@"
    ;;
  *)
    echo "MODE must be peak|prepeak|both"; exit 1
    ;;
esac

echo "========== DONE =========="
echo "Packs under /workspace/:"
echo "  inlp_gender_prob_inlp_gender_prob_4b_a_v1.tar.gz"
echo "  inlp_slot_prob_inlp_slot_prob_4b_a_v1.tar.gz"
echo "  sequential_pre_peak_prepeak_4b_v1.tar.gz"
