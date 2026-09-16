#!/usr/bin/env bash
# Qwen3.5-4B — INLP Stage A: gender then slot (sequential) + packs.
set -euo pipefail

STEER_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"

echo "========== INLP Stage A: gender → slot =========="
bash "$SCRIPTS/vast_inlp_gender_stagea_and_pack.sh" "$@"
echo "========== INLP Stage A: gender DONE; starting slot =========="
bash "$SCRIPTS/vast_inlp_slot_stagea_and_pack.sh" "$@"
echo "========== INLP Stage A: BOTH DONE =========="
echo "Packs: /workspace/inlp_gender_stage_a_*.tar.gz  /workspace/inlp_slot_stage_a_*.tar.gz"
echo "Next: bash $SCRIPTS/vast_inlp_stageb_both_and_pack.sh"
