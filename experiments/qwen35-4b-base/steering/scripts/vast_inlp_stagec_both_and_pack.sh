#!/usr/bin/env bash
# Qwen3.5-4B — INLP Stage C: gender then slot (sequential) + packs.
# Each axis auto-reads Stage B keep via --from-stage-b.
set -euo pipefail

STEER_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"

echo "========== INLP Stage C: gender → slot =========="
bash "$SCRIPTS/vast_inlp_gender_stagec_and_pack.sh" "$@"
echo "========== INLP Stage C: gender DONE; starting slot =========="
bash "$SCRIPTS/vast_inlp_slot_stagec_and_pack.sh" "$@"
echo "========== INLP Stage C: BOTH DONE =========="
echo "Packs: /workspace/inlp_gender_stage_c_*.tar.gz  /workspace/inlp_slot_stage_c_*.tar.gz"
