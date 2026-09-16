#!/usr/bin/env bash
# Qwen3.5-4B — full INLP pipeline: Stage A → B → C, each gender then slot.
# Long run (many GPU-hours). Prefer tmux/screen.
set -euo pipefail

SCRIPTS="$(cd "$(dirname "$0")" && pwd)"

echo "########## INLP FULL: A → B → C (gender then slot each) ##########"
bash "$SCRIPTS/vast_inlp_stagea_both_and_pack.sh"
bash "$SCRIPTS/vast_inlp_stageb_both_and_pack.sh"
bash "$SCRIPTS/vast_inlp_stagec_both_and_pack.sh"
echo "########## INLP FULL: ALL DONE ##########"
ls -lh /workspace/inlp_*_stage_*.tar.gz 2>/dev/null || true
