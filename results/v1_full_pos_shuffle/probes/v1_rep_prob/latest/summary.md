# V1 representation probes — summary

- **Run:** `run_2026-06-09_12-50-33_Qwen3.5-2B-Base_v1_full_pos_shuffle`
- **Probe root:** `probes/v1_rep_prob/`
- **Generated:** 2026-06-27T19:19:28.514139+00:00

## Setup

- **Probes:** 3 linear classification probes (3 axes)
  - classification: `gender_choice, narrative_choice, slot_choice`
- **Split:** by scenario group, 60%/20%/20%, seed=0
- **LogReg:** C=1.0, max_iter=10000, class_weight=balanced
- **Layer selection:** best val balanced accuracy

## Results

| Axis | Target | L | val bacc | test bacc | test AUC | shuffle@L |
|------|--------|-----|----------|-----------|---------|-----------|
| Gender — выбор man/woman | `gender_choice` | L16 | 86.8% | 86.9% | 0.941 | 49.3% |
| Narrative — first vs second | `narrative_choice` | L18 | 80.0% | 79.7% | 0.889 | 47.7% |
| Slot — A vs B | `slot_choice` | L23 | 94.5% | 96.1% | 0.993 | 48.6% |

## Gender ⊥ {narrative, slot}

- Layer: L16
- Raw gender test bacc: 86.9%
- After HS residual + refit: 86.5%
- w_g⊥ score test bacc: 84.3%
- Ablation (slot only): 86.4%
- Ablation (narrative only): 86.6%

## HTML

Open `summary.html` for charts.
