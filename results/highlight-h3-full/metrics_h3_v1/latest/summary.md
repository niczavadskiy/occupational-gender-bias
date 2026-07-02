# H3 v1 metrics summary

- **Run:** `highlight-h3-full`
- **Pipeline:** `h3_v1` (evidence highlight vs no_evidence; paired by `id`)
- **Rows:** 45648 (raw 45648)
- **Целевой FDR (Q):** 0.05
- **Min stratum n (soc):** 14
- **Generated:** 2026-06-27T15:00:08.315454+00:00

## H3 — гипотеза

Highlight в пользу пола X увеличивает предпочтение X относительно ambiguous (no_evidence).
Primary: **paired McNemar** по `id`, срез **`without_abstain`** (3 804 layout-id × 3 evidence; 2-option man/woman).
Prob: McNemar на знаке margin (p_man vs p_woman).

## Primary — paired McNemar (without_abstain)

| test_id | n_pairs | b | c | effect | p | q | BH |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| h3v1_mcnemar_no_vs_man_man | 3804 | 645 | 673 | -0.007 | 0.7878 | — | — |
| h3v1_mcnemar_no_vs_woman_woman | 3804 | 912 | 468 | +0.117 | 8.752e-33 | — | — |
| h3v1_mcnemar_no_vs_man_man_prob | 3349 | 556 | 556 | +0.000 | 0.5 | — | — |
| h3v1_mcnemar_no_vs_woman_woman_prob | 3308 | 762 | 388 | +0.113 | 3.857e-28 | — | — |

## Post-hoc — ev_man vs ev_woman

| test_id | n | b | c | p | q | BH |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| h3v1_mcnemar_man_vs_woman_man | 3804 | 207 | 623 | 4.823e-47 | 4.823e-47 | ✓ |
| h3v1_mcnemar_man_vs_woman_woman | 3804 | 623 | 207 | 4.823e-47 | 4.823e-47 | ✓ |

## Strata — context_order (no vs highlight, man axis)

| ctx | test_id | n | effect | p | q |
| :--- | ---: | ---: | ---: | ---: | ---: |
| man_first | h3v1_mcnemar_no_vs_man_man_man_first | 1902 | -0.308 | 1 | 1 |
| woman_first | h3v1_mcnemar_no_vs_man_man_woman_first | 1902 | +0.293 | 1.323e-101 | 2.646e-101 |

## Evidence flip rates

See `evidence_flip_rates.csv`.

## Rates overview

See `rates_summary.csv`.
