# Ministral 3 8B Base — steering pipeline (after beh + probe)

Scale key: **`ministral3_8b`** · model `mistralai/Ministral-3-8B-Base-2512` · **34** layers · **d_model=4096**.

Dtype: **`bfloat16`** (default in Vast scripts).

**Results (canonical):** [`../results/steering/`](../results/steering/) — see [`LAYOUT.md`](../results/steering/LAYOUT.md).

## Locked peaks (classification)

| Axis | Probe | Peak | Belt |
|------|-------|-----:|------|
| Gender | `gender_choice` | **29** | 29 / 28 / 27 |
| Slot | `slot_choice` | **28** | 28 / 27 / 26 |

Narrative omitted. Soft `gender_prob` L21 not used (steering PEAK = choice L29).

## FDR polarity (gender pool protocols)

| set | n SOC | action |
|-----|------:|--------|
| `promale` | **0** | **skip** |
| `profemale` | **17** | run |

Empty sets are skipped by pool scripts.

## Protocols (full A→B→C)

| # | Protocol | Axes | Vast |
|---|----------|------|------|
| 1 | Classical INLP | **gender + slot** | [`scripts/vast_classical_inlp_abc_full_instance.sh`](scripts/vast_classical_inlp_abc_full_instance.sh) |
| 2 | Pool INLP | gender only | [`scripts/vast_inlp_polarity_pool_ministral3_3b_full_instance.sh`](scripts/vast_inlp_polarity_pool_ministral3_3b_full_instance.sh) |
| 3 | Pool XY \|α\|≤6 | gender only | [`scripts/vast_xy_control_polarity_sets_ministral3_3b_full_instance.sh`](scripts/vast_xy_control_polarity_sets_ministral3_3b_full_instance.sh) |

## Vast commands

```bash
export HF_TOKEN=hf_xxx
SCALE=ministral3_3b DTYPE=bfloat16 \
  LAYERS_GENDER=29,28,27 \
  LAYERS_SLOT=28,27,26 \
  bash experiments/ministral3-8b-base/steering/scripts/vast_classical_inlp_abc_full_instance.sh

SCALE=ministral3_3b DTYPE=bfloat16 \
  bash experiments/ministral3-8b-base/steering/scripts/vast_inlp_polarity_pool_ministral3_3b_full_instance.sh

SCALE=ministral3_3b DTYPE=bfloat16 \
  bash experiments/ministral3-8b-base/steering/scripts/vast_xy_control_polarity_sets_ministral3_3b_full_instance.sh

SMOKE=1 SCALE=ministral3_3b DTYPE=bfloat16 \
  LAYERS_GENDER=29 LAYERS_SLOT=28 \
  bash experiments/ministral3-8b-base/steering/scripts/vast_classical_inlp_abc_full_instance.sh
```

## Configs (READY)

| File | Role |
|------|------|
| [`configs/inlp_gender_prob_ministral3_3b_v1.yaml`](configs/inlp_gender_prob_ministral3_3b_v1.yaml) | classical gender |
| [`configs/inlp_slot_prob_ministral3_3b_v1.yaml`](configs/inlp_slot_prob_ministral3_3b_v1.yaml) | classical slot |
| [`configs/inlp_polarity_pool_ministral3_3b_peak_prepeak.yaml`](configs/inlp_polarity_pool_ministral3_3b_peak_prepeak.yaml) | pool INLP |
| [`configs/xy_control_ministral3_3b_peak_prepeak_a6.yaml`](configs/xy_control_ministral3_3b_peak_prepeak_a6.yaml) | pool XY |
| [`domains/polarity_sets_ministral3_3b_v1.json`](domains/polarity_sets_ministral3_3b_v1.json) | XY sets |
| [`domains/inlp_polarity_sets_ministral3_3b_v1.json`](domains/inlp_polarity_sets_ministral3_3b_v1.json) | INLP sets |

Catalog: `steering/xy_control/domains/h1_soc_fdr_v1.json` → `scales.ministral3_3b`.
