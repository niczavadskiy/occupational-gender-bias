# 4B steering / INLP scripts

Public repo only: `REPO=/workspace/occupational-gender-bias` (no GH_TOKEN).
Need `HF_TOKEN` for the model.

## Probe-direction Stage A (done)

| Script | Tag |
|---|---|
| `vast_h1_stagea_and_pack.sh` | `h1_4b_a_v1` |
| `vast_slot_stagea_and_pack.sh` | `slot_4b_a_v1` |

## INLP Stage A → B → C (gender + slot)

Order **per axis** (gender then slot, or parallel on two GPUs):

1. **Stage A** — preference grid → writes `stageb_shortlist.json` (auto)  
2. **Stage B** — `--from-stage-a` → MMLU; writes `stagec_keep.json` (auto)  
3. **Stage C** — `--from-stage-b` → held-out report  

| Script | Default tag |
|---|---|
| `vast_inlp_gender_stagea_and_pack.sh` | `inlp_gender_4b_a_v1` |
| `vast_inlp_slot_stagea_and_pack.sh` | `inlp_slot_4b_a_v1` |
| `vast_inlp_gender_stageb_and_pack.sh` | `inlp_gender_4b_b_v2` (from Stage A shortlist) |
| `vast_inlp_slot_stageb_and_pack.sh` | `inlp_slot_4b_b_v2` |
| `vast_inlp_gender_stagec_and_pack.sh` | `inlp_gender_4b_c_v2` (from Stage B keep) |
| `vast_inlp_slot_stagec_and_pack.sh` | `inlp_slot_4b_c_v2` |

Module: `python -m steering.inlp_shortlist from-stage-a|from-stage-b ...`

Smoke before full Stage A:

```bash
python -m steering.run_inlp_stagea --model Qwen/Qwen3.5-4B-Base --device cuda --dtype float32 \
  --subspaces $STEER/subspaces/inlp_gender_choice_v1.npz \
  --sample $STEER/samples/h1_stagea_sample_v1.json \
  --layers 24 --ranks 1,4 --limit-items 3 --tag inlp_gender_4b_smoke \
  --out-root $STEER/../results/steering --primary-axis gender
```

Rebuild artifacts: `build_artifacts.sh` / `.ps1`.
