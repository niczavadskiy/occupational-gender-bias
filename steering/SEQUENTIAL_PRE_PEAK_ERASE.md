# Sequential pre-peak erase

New pipeline (orthogonal to post-hook [`SEQUENTIAL_ERASE.md`](SEQUENTIAL_ERASE.md)).

**Idea:** causal sites often sit on the **AUC ascent up to the peak**, not only
after the first keep. Take the ascent pool from the curve shape, start INLP at
the **earliest** layer in the pool, then add later pool layers while synergy
holds.

## Ascent pool (flexible, not fixed top‑K)

From layer_scan `val_roc_auc` (or config `auc_curve`):

1. \(L^\star = \arg\max a(L)\)
2. Height threshold: \(a(L) \ge a(L^\star) - \tau\cdot(a(L^\star)-0.5)\)
3. Pool = contiguous \([L_{\min}, L^\star]\) where \(L_{\min}\) is the earliest
   layer meeting the threshold
4. Knee (γ) logged for diagnostics; early L0→L1 spikes ignored via
   `min_prev_auc_for_knee`

Defaults (`τ=0.38`, `γ=0.25`) recover **gender 2B pool `{13,14,15,16}`**
(peak L16) from `h11_gender_choice` layer_scan.

## Commands

```bash
cd occupational-gender-bias
export PYTHONPATH=.

# 1) plan only (CPU)
python -m steering.run_sequential_pre_peak_erase \
  --config steering/configs/sequential_pre_peak_erase_gender_2b_v1.yaml \
  --plan-only --tag prepeak_gender_plan

python -m steering.run_sequential_pre_peak_erase \
  --config steering/configs/sequential_pre_peak_erase_slot_2b_v1.yaml \
  --plan-only --tag prepeak_slot_plan

# 2) full GPU (smoke)
python -m steering.run_sequential_pre_peak_erase \
  --config steering/configs/sequential_pre_peak_erase_gender_2b_v1.yaml \
  --device cuda --dtype float32 --n-items 2 --max-hits 2 \
  --tag prepeak_gender_smoke
```

Vast:

```bash
bash steering/scripts/vast_sequential_pre_peak_and_pack.sh
# AXIS=gender|slot|both  SMOKE=1  TAG=...
```

## Outputs

`results/steering/sequential_pre_peak/<tag>/`

| file | |
|---|---|
| `plan.json` | pool, first_hit, curve meta |
| `inlp_prepeak_*.npz` | W fit on baseline HS for pool layers |
| `stack.json` / `summary.json` | kept sites + hit log |

## Slot curve note

2B `slot_choice` peak ≈ **L23**. If `layer_scan.path` is missing, the slot
config ships an **illustrative** `auc_curve` stub — replace with a real
layer_scan before reporting.

## vs old sequential erase

| | Pre-peak (this) | Post-hook (`SEQUENTIAL_ERASE`) |
|---|---|---|
| Pool | ≤ AUC peak, ascent | layers *after* fixed S₀ (e.g. 16–24) |
| First hit | earliest in ascent | fixed L15 keep |
| Use when | choosing / rebuilding first site | residual after an existing keep |
