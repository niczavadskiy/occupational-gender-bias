# Vast: polarity-pooled INLP

Matched arm to XY polarity pool. Protocol:
[`POLARITY_POOL_INLP.md`](POLARITY_POOL_INLP.md).

Candidate layers: **L16 (peak) + L15 + L14**. Ranks `{4,8,16}`, α=1.
Sets: `promale` then `profemale`. Same family split as XY (seed 0).

## One-shot (recommended)

```bash
export HF_TOKEN=hf_xxx
cd /workspace
BRANCH=qwen_2b_experiments bash occupational-gender-bias/steering/scripts/vast_inlp_polarity_pool_full_instance.sh
```

If the repo is already at `/workspace/occupational-gender-bias`:

```bash
export HF_TOKEN=hf_xxx
cd /workspace/occupational-gender-bias
git fetch --depth 1 origin qwen_2b_experiments
git checkout qwen_2b_experiments
git reset --hard origin/qwen_2b_experiments
BRANCH=qwen_2b_experiments bash steering/scripts/vast_inlp_polarity_pool_full_instance.sh
```

## Smoke

```bash
export HF_TOKEN=hf_xxx
SMOKE=1 BRANCH=qwen_2b_experiments \
  bash steering/scripts/vast_inlp_polarity_pool_full_instance.sh
```

## Env knobs

| var | default | meaning |
|-----|---------|---------|
| `SETS` | `promale,profemale` | which pools to run |
| `SMOKE` | `0` | anchor L16 × k4, few families |
| `SKIP_PIP` | `0` | `1` if torch lock already installed |
| `SKIP_FIT` | `0` | `1` = reuse subspaces, only Stage A+ |
| `SKIP_STAGE_A` | `0` | skip fit + Stage A |
| `RUN_STAGE_B` | `1` | MMLU gate |
| `RUN_STAGE_C` | `1` | test preference + MMLU |
| `BRANCH` | `qwen_2b_experiments` | git branch to sync |

Resume after a finished fit (e.g. `k_found=1`):

```bash
export HF_TOKEN=hf_xxx
SKIP_PIP=1 SKIP_FIT=1 BRANCH=qwen_2b_experiments \
  bash steering/scripts/vast_inlp_polarity_pool_full_instance.sh
```

Single set:

```bash
SETS=promale BRANCH=qwen_2b_experiments \
  bash steering/scripts/vast_inlp_polarity_pool_full_instance.sh
```

## Packs

Written to `/workspace/`:

- `inlp_polarity_pool_promale_stage_abc_2b.tar.gz`
- `inlp_polarity_pool_profemale_stage_abc_2b.tar.gz`

Contents: samples, subspaces, `results/steering/inlp_polarity_pool/`.

## Manual steps (debug)

```bash
cd /workspace/occupational-gender-bias
export PYTHONPATH=$PWD
source scripts/ensure_steering_env.sh && ensure_steering_env

python -m steering.build_polarity_pool_inlp_sample --all
python -m steering.build_polarity_pool_inlp_subspace --set-id promale --device cuda --dtype float32
python -m steering.run_polarity_pool_inlp_stagea --set-id promale --device cuda --dtype float32
python -m steering.run_polarity_pool_inlp_stageb --set-id promale --device cuda --dtype float32
python -m steering.run_polarity_pool_inlp_stagec --set-id promale --device cuda --dtype float32
```
