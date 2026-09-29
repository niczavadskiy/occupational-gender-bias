# Gemma 3 4B PT — steering pipeline (after beh + probe)

Scale key: **`gemma3_4b`** · model `google/gemma-3-4b-pt` · **34** layers · **d_model=2560**.

## Locked peaks (classification)

| Axis | Probe | Peak | Belt |
|------|-------|-----:|------|
| Gender | `gender_choice` | **24** | 24 / 23 / 22 |
| Slot | `slot_choice` | **31** | 31 / 30 / 29 |

Narrative omitted. Soft `gender_prob` L26 not used (steering PEAK = choice L24).

## FDR polarity (gender pool protocols)

| set | n SOC | action |
|-----|------:|--------|
| `promale` | **3** | run |
| `profemale` | **9** | run |

## Protocols (full A→B→C)

| # | Protocol | Axes | Vast |
|---|----------|------|------|
| 1 | Classical INLP | **gender + slot** | [`scripts/vast_classical_inlp_abc_full_instance.sh`](scripts/vast_classical_inlp_abc_full_instance.sh) |
| 2 | Pool INLP | gender only | [`scripts/vast_inlp_polarity_pool_gemma3_4b_full_instance.sh`](scripts/vast_inlp_polarity_pool_gemma3_4b_full_instance.sh) |
| 3 | Pool XY \|α\|≤6 | gender only | [`scripts/vast_xy_control_polarity_sets_gemma3_4b_full_instance.sh`](scripts/vast_xy_control_polarity_sets_gemma3_4b_full_instance.sh) |

## Vast commands

```bash
export HF_TOKEN=hf_xxx
SCALE=gemma3_4b bash experiments/gemma3-4b-pt/steering/scripts/vast_classical_inlp_abc_full_instance.sh
SCALE=gemma3_4b bash experiments/gemma3-4b-pt/steering/scripts/vast_inlp_polarity_pool_gemma3_4b_full_instance.sh
SCALE=gemma3_4b bash experiments/gemma3-4b-pt/steering/scripts/vast_xy_control_polarity_sets_gemma3_4b_full_instance.sh

SMOKE=1 SCALE=gemma3_4b bash experiments/gemma3-4b-pt/steering/scripts/vast_classical_inlp_abc_full_instance.sh
```

## Configs (READY)

| File | Role |
|------|------|
| [`configs/inlp_gender_prob_gemma3_4b_v1.yaml`](configs/inlp_gender_prob_gemma3_4b_v1.yaml) | classical gender |
| [`configs/inlp_slot_prob_gemma3_4b_v1.yaml`](configs/inlp_slot_prob_gemma3_4b_v1.yaml) | classical slot |
| [`configs/inlp_polarity_pool_gemma3_4b_peak_prepeak.yaml`](configs/inlp_polarity_pool_gemma3_4b_peak_prepeak.yaml) | pool INLP |
| [`configs/xy_control_gemma3_4b_peak_prepeak_a6.yaml`](configs/xy_control_gemma3_4b_peak_prepeak_a6.yaml) | pool XY |
| [`domains/polarity_sets_gemma3_4b_v1.json`](domains/polarity_sets_gemma3_4b_v1.json) | XY sets |
| [`domains/inlp_polarity_sets_gemma3_4b_v1.json`](domains/inlp_polarity_sets_gemma3_4b_v1.json) | INLP sets |

Catalog: `steering/xy_control/domains/h1_soc_fdr_v1.json` → `scales.gemma3_4b`.
