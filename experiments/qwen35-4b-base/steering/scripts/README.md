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

1. **Stage A** — preference on val sample, full rank grid  
2. Download ranking → **edit shortlist** JSON (or keep provisional L24/L30 k∈{8,16})  
3. **Stage B** — MMLU capability on shortlist  
4. Freeze **keep** JSON → **Stage C** — held-out preference + domain_test

| Script | Default tag |
|---|---|
| `vast_inlp_gender_stagea_and_pack.sh` | `inlp_gender_4b_a_v1` |
| `vast_inlp_slot_stagea_and_pack.sh` | `inlp_slot_4b_a_v1` |
| `vast_inlp_gender_stageb_and_pack.sh` | `inlp_gender_4b_b_v1` |
| `vast_inlp_slot_stageb_and_pack.sh` | `inlp_slot_4b_b_v1` |
| `vast_inlp_gender_stagec_and_pack.sh` | `inlp_gender_4b_c_v1` |
| `vast_inlp_slot_stagec_and_pack.sh` | `inlp_slot_4b_c_v1` |

Smoke before full Stage A:

```bash
python -m steering.run_inlp_stagea --model Qwen/Qwen3.5-4B-Base --device cuda --dtype float32 \
  --subspaces $STEER/subspaces/inlp_gender_choice_v1.npz \
  --sample $STEER/samples/h1_stagea_sample_v1.json \
  --layers 24 --ranks 1,4 --limit-items 3 --tag inlp_gender_4b_smoke \
  --out-root $STEER/../results/steering --primary-axis gender
```

Rebuild artifacts: `build_artifacts.sh` / `.ps1`.
