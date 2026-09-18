#!/usr/bin/env bash
# Run classical gender_prob + slot_prob Stage A packs sequentially.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AXIS="${AXIS:-both}"
case "$AXIS" in
  gender) bash "$SCRIPT_DIR/vast_inlp_gender_prob_stagea_and_pack.sh" "$@" ;;
  slot) bash "$SCRIPT_DIR/vast_inlp_slot_prob_stagea_and_pack.sh" "$@" ;;
  both)
    bash "$SCRIPT_DIR/vast_inlp_gender_prob_stagea_and_pack.sh" "$@"
    bash "$SCRIPT_DIR/vast_inlp_slot_prob_stagea_and_pack.sh" "$@"
    ;;
  *) echo "AXIS must be gender|slot|both"; exit 1 ;;
esac
