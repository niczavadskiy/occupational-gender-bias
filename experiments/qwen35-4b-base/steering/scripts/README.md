# 4B steering / INLP scripts

Public repo only: `REPO=/workspace/occupational-gender-bias` (no GH_TOKEN).
Need `HF_TOKEN` for the model.

## Probe-direction Stage A (done)

| Script | Tag |
|---|---|
| `vast_h1_stagea_and_pack.sh` | `h1_4b_a_v1` |
| `vast_slot_stagea_and_pack.sh` | `slot_4b_a_v1` |

## INLP Stage A → B → C (gender then slot)

**One command per stage** (sequential gender → slot):

| Script | What it runs |
|---|---|
| `vast_inlp_stagea_both_and_pack.sh` | gender A → slot A |
| `vast_inlp_stageb_both_and_pack.sh` | gender B → slot B |
| `vast_inlp_stagec_both_and_pack.sh` | gender C → slot C |
| `vast_inlp_all_stages_both_and_pack.sh` | A → B → C (full) |

Per-axis (resume/debug): `vast_inlp_{gender,slot}_stage{a,b,c}_and_pack.sh`  
Defaults: A `*_a_v1`; B/C `*_b_v2` / `*_c_v2` with auto shortlist/keep.

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
