# Sequential multi-site erase (2B keep-case: L15 k16)

Protocol after method-3 HS/AUC recovery: erase on L15 weakens linear gender
locally; AUC partially recovers deeper (e.g. L20). This pipeline finds
**causal second sites** under fixed stack S₀, then fits INLP on HS captured
*under* S₀ (not baseline HS).

## Pipeline

| Phase | Script | Output |
|---|---|---|
| 1. Second-hit screen | `run_second_hit_screen.py` | `results/steering/second_hit/<tag>/shortlist.json` |
| 2a. Conditional INLP | `build_conditional_inlp_subspace.py` | `steering/subspaces/inlp_cond_<target>_<tag>.npz` |
| 2b. Stacked Stage A | `run_stacked_inlp_stagea.py` | `results/steering/stacked_inlp_stage_a/<tag>/` |
| 3. Method 3 again | `run_hs_recovery_auc.py` on winning stack | AUC curve under S₀+ℓ |

S₀ is always **gender INLP L15 k16 center α=1** (`inlp_gender_choice_v1.npz`).

## Decision rules

**Second-hit → Stage A**

- Gender candidate: `synergy_R_gender = R(S0+ℓ) − R(S0) ≥ synergy_min` (default 0.005) and CI_lo(R_stack) > 0  
- Slot candidate: same, or `R_slot(stack) − R_slot(S0) ≥ synergy_min`  
- Random on same ℓ must stay weaker (included as null on L16/20/23)

**Stacked Stage A keep**

- INLP (not random) with `synergy_vs_S0 ≥ 0.005` and `R_primary_ci_lo > 0`  
- Prefer slot if large ΔR_slot with small gender cost

Do **not** run full belt 16–24 × full rank grid before the shortlist gate.

## Vast

```bash
export HF_TOKEN=hf_xxx
export REPO=/workspace/occupational-gender-bias
cd "$REPO"

# smoke end-to-end
SMOKE=1 bash steering/scripts/vast_sequential_erase_full_instance.sh

# full
bash steering/scripts/vast_sequential_erase_full_instance.sh

# or stepwise
bash steering/scripts/vast_second_hit_and_pack.sh
bash steering/scripts/vast_conditional_inlp_stagea_and_pack.sh
```

Packs land in `/workspace/second_hit_<tag>.tar.gz` and `/workspace/stacked_inlp_<tag>.tar.gz`.

## Local smoke (CPU / small)

```bash
cd occupational-gender-bias
export PYTHONPATH=.

python -m steering.run_second_hit_screen --device cuda --n-items 2 \
  --layers 16,20,23 --tag second_hit_smoke

python -m steering.build_conditional_inlp_subspace --device cuda --n-items 2 \
  --from-shortlist results/steering/second_hit/second_hit_smoke/shortlist.json \
  --layers 16,20 --k-max 8 --targets gender_choice --tag cond_smoke

python -m steering.run_stacked_inlp_stagea --device cuda --limit-items 2 \
  --second-subspaces steering/subspaces/inlp_cond_gender_choice_cond_smoke.npz \
  --layers 16,20 --ranks 8 --tag stacked_smoke
```

## Notes

- Gender probe vectors exist only through **L20**; L21–24 gender second-hit uses INLP if present in the base npz (L16/L18) or is skipped until conditional fit.  
- Slot probes cover **L16–24**.  
- Conditional W is **not** interchangeable with baseline INLP W — always evaluate with S₀ on (`run_stacked_inlp_stagea`).
