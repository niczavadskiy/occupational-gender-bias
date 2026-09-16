#!/usr/bin/env bash
# Qwen3.5-4B — INLP Stage B: gender then slot (sequential) + packs.
# Each axis auto-reads Stage A shortlist via --from-stage-a.
set -euo pipefail

STEER_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"

echo "========== INLP Stage B: gender → slot =========="
bash "$SCRIPTS/vast_inlp_gender_stageb_and_pack.sh" "$@"
echo "========== INLP Stage B: gender DONE; starting slot =========="
bash "$SCRIPTS/vast_inlp_slot_stageb_and_pack.sh" "$@"
echo "========== INLP Stage B: BOTH DONE =========="
echo "Packs: /workspace/inlp_gender_stage_b_*.tar.gz  /workspace/inlp_slot_stage_b_*.tar.gz"
echo "Next: bash $SCRIPTS/vast_inlp_stagec_both_and_pack.sh"
