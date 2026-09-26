# Gemma 3 4B PT — steering pipeline (after beh + probe)

Scale key: **`gemma3_4b`** · model `google/gemma-3-4b-pt` · expect **34** layers · **d_model=2560**.

## Checklist

1. [ ] Beh pack in `../results/v1_full_pos_shuffle` + `highlight-h3-full`
2. [ ] H1 metrics → FDR SOCs (male/female lists)
3. [ ] HS-probe (`probes.v1_rep*`) → **gender_prob peak** = `PEAK`
4. [ ] Replace every `PEAK` / `PEAK_M1` / `PEAK_M2` in `configs/`
5. [ ] Classical INLP Stage A (peak + 2 below)
6. [ ] Build polarity sets JSON from FDR
7. [ ] Pool INLP A→B→C
8. [ ] Pool XY-control A→B→C

Layer belt after peak known:

| role | layer |
|------|------:|
| peak | PEAK |
| peak−1 | PEAK−1 |
| peak−2 | PEAK−2 |

## Stub configs (fill after probe)

| File | Role |
|------|------|
| [`configs/inlp_gender_prob_gemma3_4b_v1.yaml`](configs/inlp_gender_prob_gemma3_4b_v1.yaml) | classical gender_prob INLP |
| [`configs/inlp_polarity_pool_gemma3_4b_peak_prepeak.yaml`](configs/inlp_polarity_pool_gemma3_4b_peak_prepeak.yaml) | pool INLP |
| [`configs/xy_control_gemma3_4b_peak_prepeak_a6.yaml`](configs/xy_control_gemma3_4b_peak_prepeak_a6.yaml) | pool XY \|α\|≤6 |
| [`domains/polarity_sets_gemma3_4b_v1.json`](domains/polarity_sets_gemma3_4b_v1.json) | XY sets (slugs TBD) |
| [`domains/inlp_polarity_sets_gemma3_4b_v1.json`](domains/inlp_polarity_sets_gemma3_4b_v1.json) | INLP sets (slugs TBD) |

## Scale wiring

Runners accept `--scale gemma3_4b` and resolve model via
`MODEL_BY_SCALE["gemma3_4b"] = "google/gemma-3-4b-pt"` in
`steering/xy_control/stagebc.py`.

Catalog FDR: add `scales.gemma3_4b` to
`steering/xy_control/domains/h1_soc_fdr_v1.json` (or a sibling JSON) after H1.

## Vast (after peak filled)

```bash
# classical INLP — write vast script once PEAK known
# pool INLP
# pool XY
```

Scripts to add in follow-up (mirror Qwen 4B / Gemma 1B):

- `steering/scripts/vast_inlp_polarity_pool_gemma3_4b_full_instance.sh`
- `steering/xy_control/scripts/vast_xy_control_polarity_sets_gemma3_4b_full_instance.sh`
