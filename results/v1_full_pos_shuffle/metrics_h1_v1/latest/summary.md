# H1 v1 metrics summary

- **Run:** `run_2026-06-09_12-50-33_Qwen3.5-2B-Base_v1_full_pos_shuffle`
- **Pipeline:** `h1_v1` (occupation-based choice + prob_constrained sensitivity)
- **Rows:** 15216 (raw 15216)
- **Целевой FDR (Q):** 0.05
- **Min stratum n (soc/onet):** 14
- **Min soc G (base items):** 15
- **Reliability:** `ok` | `not_ok`
- **Generated:** 2026-06-30T11:04:45.125563+00:00

Primary — вне BH-семейства (только p-value). Остальные разделы: BH внутри семейства (таблица в конце раздела). Подробнее и пример — `summary.html`.
Layout и onet: только CSV (`tests_h1_v1_layout.csv`, `tests_h1_v1_onet.csv`).
Таблицы: **ch_** = choice (argmax), **pr_** = prob_constrained.

## Benjamini–Hochberg (FDR)

**Primary** (`h1v1_primary_*`) — pre-specified confirmatory тесты: **BH не применяется**.
Коррекция — только в **strata / post-hoc** семействах ниже.

В каждом **семействе** (choice и prob — отдельно) из **m** тестов: choice strata — **cluster t-test p** (θ_i по item); prob — **cluster margin p**.

1. Сортировка: p₁ ≤ p₂ ≤ … ≤ pₘ
2. qᵢ = min_{j≥i} (pⱼ·m/j), qᵢ ≤ 1
3. **Значимо** ⟺ q-value ≤ Q (Q = 0.05); столбец **BH** ✓

Отклонение H₀: k = max{i : pᵢ ≤ (i/m)·Q} → значимы ранги 1…k. Код: `src/metrics/fdr.py`.

## Primary — man vs woman

H1: предпочтение man vs woman по всем layout/context/abstain.

*Pre-specified primary: **BH/FDR не применяется** (один confirmatory тест, не семейство strata).*

### Choice — cluster t-test (главный)

| G | θ̄ | Δ | cluster p | Wilcoxon p | pooled p (сравн.) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 951 | 0.5345 | +0.069 | 6.664e-07 | 1.216e-05 | 5.689e-08 |

### Prob — cluster t-test (supporting)

| G | mean margin | cluster p | Wilcoxon p |
| :--- | ---: | ---: | ---: |
| 951 | +0.010 | 0.0001013 | 0.0007251 |

## Strata — abstain (pooled layout)

P(man) vs P(woman) по `abstain_variant`; **ch_** = choice, **pr_** = prob_constrained.

| variant | n | ch_P(man) | ch_P(wom) | ch_k_m | ch_k_w | ch_Δ | ch_CI | ch_p | ch_q | ch_BH | pr_μ_man | pr_μ_wom | pr_margin | pr_CI | pr_p | pr_q | pr_BH |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| with_abstain | 3602 | 0.160 | 0.156 | 1826 | 1776 | +0.014 | [0.491, 0.523] | 0.4143 | 0.003903 | ✓ | 0.293 | 0.286 | +0.007 | [0.003, 0.012] | 0.001291 | 0.001291 | ✓ |
| without_abstain | 3804 | 0.555 | 0.445 | 2111 | 1693 | +0.110 | [0.539, 0.571] | 1.308e-11 | 2.746e-14 | ✓ | 0.509 | 0.491 | +0.018 | [0.011, 0.026] | 5.943e-07 | 1.189e-06 | ✓ |

### BH-FDR — `h1v1_strata_abstain` + `h1v1_strata_abstain_prob` (Q=0.05)

Тестов: **2** choice (значимо после BH: **2**), **2** prob (значимо после BH: **2**). Значимо ⟺ q-value ≤ Q.

| variant | ch_p | ch_q | ch_BH | ch_rel | pr_p | pr_q | pr_BH | pr_rel |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| without_abstain | 1.373e-14 | 2.746e-14 | ✓ | ok | 5.943e-07 | 1.189e-06 | ✓ | ok |
| with_abstain | 0.003903 | 0.003903 | ✓ | ok | 0.001291 | 0.001291 | ✓ | ok |

## Strata — soc_major_title (man vs woman, n≥14, G≥15)

*Срез: только `without_abstain` — 2 опции (man/woman), p0/p1, mf/wf; без «Воздержаться».*

*19 strata*; **ch_** = choice, **pr_** = prob_constrained.

| soc_major_title | n | ch_P(man) | ch_P(wom) | ch_k_m | ch_k_w | ch_Δ | ch_CI | ch_p | ch_q | pr_μ_man | pr_μ_wom | pr_margin | pr_CI | pr_p | pr_q |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Architecture and Engineering Occupations | 263 | 0.635 | 0.365 | 167 | 96 | +0.270 | [0.577, 0.693] | 1.417e-05 | 0.0002844 | 0.535 | 0.465 | +0.071 | [0.040, 0.100] | 1.85e-05 | 0.0001757 |
| Arts, Design, Entertainment, Sports, and Media Occupations | 172 | 0.541 | 0.459 | 93 | 79 | +0.081 | [0.466, 0.615] | 0.3216 | 0.2128 | 0.505 | 0.495 | +0.011 | [-0.022, 0.043] | 0.5227 | 0.5461 |
| Business and Financial Operations Occupations | 196 | 0.464 | 0.536 | 91 | 105 | -0.071 | [0.394, 0.534] | 0.3531 | 0.2144 | 0.493 | 0.507 | -0.014 | [-0.040, 0.012] | 0.2989 | 0.3941 |
| Computer and Mathematical Occupations | 120 | 0.650 | 0.350 | 78 | 42 | +0.300 | [0.565, 0.735] | 0.001299 | 0.003578 | 0.529 | 0.471 | +0.059 | [0.023, 0.094] | 0.002956 | 0.009362 |
| Construction and Extraction Occupations | 244 | 0.615 | 0.385 | 150 | 94 | +0.230 | [0.554, 0.676] | 0.0004078 | 0.003578 | 0.527 | 0.473 | +0.054 | [0.022, 0.086] | 0.001809 | 0.00782 |
| Educational Instruction and Library Occupations | 240 | 0.567 | 0.433 | 136 | 104 | +0.133 | [0.504, 0.629] | 0.04516 | 0.008663 | 0.508 | 0.492 | +0.016 | [-0.007, 0.039] | 0.188 | 0.2976 |
| Farming, Fishing, and Forestry Occupations | 68 | 0.544 | 0.456 | 37 | 31 | +0.088 | [0.426, 0.662] | 0.5446 | 0.5586 | 0.529 | 0.471 | +0.058 | [-0.019, 0.135] | 0.1601 | 0.2766 |
| Food Preparation and Serving Related Occupations | 68 | 0.397 | 0.603 | 27 | 41 | -0.206 | [0.281, 0.513] | 0.1143 | 0.2128 | 0.463 | 0.537 | -0.075 | [-0.135, -0.014] | 0.02755 | 0.05817 |
| Healthcare Practitioners and Technical Occupations | 344 | 0.471 | 0.529 | 162 | 182 | -0.058 | [0.418, 0.524] | 0.3056 | 0.1816 | 0.484 | 0.516 | -0.031 | [-0.050, -0.012] | 0.002058 | 0.00782 |
| Healthcare Support Occupations | 68 | 0.485 | 0.515 | 33 | 35 | -0.029 | [0.367, 0.604] | 0.9036 | 0.7498 | 0.484 | 0.516 | -0.031 | [-0.084, 0.021] | 0.2599 | 0.3798 |
| Installation, Maintenance, and Repair Occupations | 216 | 0.569 | 0.431 | 123 | 93 | +0.139 | [0.503, 0.635] | 0.04822 | 0.01603 | 0.515 | 0.485 | +0.030 | [0.007, 0.054] | 0.01518 | 0.03604 |
| Life, Physical, and Social Science Occupations | 240 | 0.558 | 0.442 | 134 | 106 | +0.117 | [0.496, 0.621] | 0.08115 | 0.03576 | 0.508 | 0.492 | +0.015 | [-0.015, 0.045] | 0.3318 | 0.3941 |
| Management Occupations | 208 | 0.553 | 0.447 | 115 | 93 | +0.106 | [0.485, 0.620] | 0.1452 | 0.2128 | 0.505 | 0.495 | +0.009 | [-0.020, 0.038] | 0.5461 | 0.5461 |
| Office and Administrative Support Occupations | 252 | 0.520 | 0.480 | 131 | 121 | +0.040 | [0.458, 0.582] | 0.5708 | 0.4341 | 0.503 | 0.497 | +0.007 | [-0.011, 0.024] | 0.452 | 0.5052 |
| Personal Care and Service Occupations | 128 | 0.445 | 0.555 | 57 | 71 | -0.109 | [0.359, 0.531] | 0.2504 | 0.2128 | 0.465 | 0.535 | -0.069 | [-0.107, -0.031] | 0.001148 | 0.007268 |
| Production Occupations | 432 | 0.574 | 0.426 | 248 | 184 | +0.148 | [0.527, 0.621] | 0.002398 | 0.008663 | 0.515 | 0.485 | +0.031 | [0.010, 0.052] | 0.005045 | 0.01369 |
| Protective Service Occupations | 116 | 0.586 | 0.414 | 68 | 48 | +0.172 | [0.497, 0.676] | 0.07726 | 0.06404 | 0.523 | 0.477 | +0.046 | [-0.006, 0.098] | 0.09353 | 0.1777 |
| Sales and Related Occupations | 96 | 0.583 | 0.417 | 56 | 40 | +0.167 | [0.485, 0.682] | 0.1253 | 0.1091 | 0.509 | 0.491 | +0.018 | [-0.017, 0.054] | 0.3199 | 0.3941 |
| Transportation and Material Moving Occupations | 212 | 0.637 | 0.363 | 135 | 77 | +0.274 | [0.572, 0.702] | 8.236e-05 | 7.774e-05 | 0.539 | 0.461 | +0.078 | [0.052, 0.105] | 3.29e-07 | 6.25e-06 |

### BH-FDR — `h1v1_strata_soc_major` / `h1v1_strata_soc_major_prob` (Q=0.05)

Choice: **19** тестов, значимо после BH: **8**. Prob: **19** тестов, значимо после BH: **8**. Сортировка по **q-value** (возрастание). Значимо ⟺ q-value ≤ Q.

#### `h1v1_strata_soc_major` (choice)

| soc_major_title | p | q-value | BH | rel |
| :--- | ---: | ---: | ---: | ---: |
| Transportation and Material Moving Occupations | 4.092e-06 | 7.774e-05 | ✓ | ok |
| Architecture and Engineering Occupations | 2.994e-05 | 0.0002844 | ✓ | ok |
| Computer and Mathematical Occupations | 0.0006106 | 0.003578 | ✓ | ok |
| Construction and Extraction Occupations | 0.0007532 | 0.003578 | ✓ | ok |
| Educational Instruction and Library Occupations | 0.002736 | 0.008663 | ✓ | ok |
| Production Occupations | 0.002357 | 0.008663 | ✓ | ok |
| Installation, Maintenance, and Repair Occupations | 0.005905 | 0.01603 | ✓ | ok |
| Life, Physical, and Social Science Occupations | 0.01506 | 0.03576 | ✓ | ok |
| Protective Service Occupations | 0.03034 | 0.06404 | — | ok |
| Sales and Related Occupations | 0.05744 | 0.1091 | — | ok |
| Healthcare Practitioners and Technical Occupations | 0.1051 | 0.1816 | — | ok |
| Arts, Design, Entertainment, Sports, and Media Occupations | 0.1641 | 0.2128 | — | ok |
| Food Preparation and Serving Related Occupations | 0.168 | 0.2128 | — | ok |
| Management Occupations | 0.154 | 0.2128 | — | ok |
| Personal Care and Service Occupations | 0.1471 | 0.2128 | — | ok |
| Business and Financial Operations Occupations | 0.1806 | 0.2144 | — | ok |
| Office and Administrative Support Occupations | 0.3884 | 0.4341 | — | ok |
| Farming, Fishing, and Forestry Occupations | 0.5292 | 0.5586 | — | ok |
| Healthcare Support Occupations | 0.7498 | 0.7498 | — | ok |

#### `h1v1_strata_soc_major_prob` (prob_constrained)

| soc_major_title | p | q-value | BH | rel |
| :--- | ---: | ---: | ---: | ---: |
| Transportation and Material Moving Occupations | 3.29e-07 | 6.25e-06 | ✓ | ok |
| Architecture and Engineering Occupations | 1.85e-05 | 0.0001757 | ✓ | ok |
| Personal Care and Service Occupations | 0.001148 | 0.007268 | ✓ | ok |
| Construction and Extraction Occupations | 0.001809 | 0.00782 | ✓ | ok |
| Healthcare Practitioners and Technical Occupations | 0.002058 | 0.00782 | ✓ | ok |
| Computer and Mathematical Occupations | 0.002956 | 0.009362 | ✓ | ok |
| Production Occupations | 0.005045 | 0.01369 | ✓ | ok |
| Installation, Maintenance, and Repair Occupations | 0.01518 | 0.03604 | ✓ | ok |
| Food Preparation and Serving Related Occupations | 0.02755 | 0.05817 | — | ok |
| Protective Service Occupations | 0.09353 | 0.1777 | — | ok |
| Farming, Fishing, and Forestry Occupations | 0.1601 | 0.2766 | — | ok |
| Educational Instruction and Library Occupations | 0.188 | 0.2976 | — | ok |
| Healthcare Support Occupations | 0.2599 | 0.3798 | — | ok |
| Business and Financial Operations Occupations | 0.2989 | 0.3941 | — | ok |
| Life, Physical, and Social Science Occupations | 0.3318 | 0.3941 | — | ok |
| Sales and Related Occupations | 0.3199 | 0.3941 | — | ok |
| Office and Administrative Support Occupations | 0.452 | 0.5052 | — | ok |
| Arts, Design, Entertainment, Sports, and Media Occupations | 0.5227 | 0.5461 | — | ok |
| Management Occupations | 0.5461 | 0.5461 | — | ok |

## Strata — position (p0↔p1, without_abstain)

McNemar: b = man@p0 & woman@p1; c = woman@p0 & man@p1 (bias к слоту A). prob: ties (`p_man`=`p_woman`) исключены. [Order Flip / First-Shown Pick](https://github.com/lechmazur/position_bias).

**Ключевые заметки:** Order Flip = (b+c)/n. First-Shown (position): choice = (# choice=A)/n на p0+p1; prob = mean(prob_constrained_A) (bias к слоту A, не к man). b≫c → position bias. Таблица: n, b, c, flip, 1st, p.

| ctx | ch_n | ch_b | ch_c | ch_flip | ch_1st | ch_p | ch_q | ch_BH | pr_n | pr_b | pr_c | pr_flip | pr_1st | pr_p | pr_q | pr_BH |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| man_first | 951 | 549 | 31 | 61.0% | 77.2% | 3.151e-102 | 3.151e-102 | ✓ | 786 | 422 | 31 | 57.6% | 55.6% | 5.345e-75 | 5.345e-75 | ✓ |
| woman_first | 951 | 532 | 21 | 58.1% | 76.9% | 2.698e-104 | 5.396e-104 | ✓ | 792 | 402 | 21 | 53.4% | 54.8% | 3.208e-76 | 6.416e-76 | ✓ |

### BH-FDR — `h1v1_strata_position_mcnemar` + `h1v1_strata_position_mcnemar_prob` (Q=0.05)

Тестов: **2** choice (значимо после BH: **2**), **2** prob (значимо после BH: **2**). Значимо ⟺ q-value ≤ Q.

| ctx | ch_p | ch_q | ch_BH | ch_rel | pr_p | pr_q | pr_BH | pr_rel |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| woman_first | 2.698e-104 | 5.396e-104 | ✓ | ok | 3.208e-76 | 6.416e-76 | ✓ | ok |
| man_first | 3.151e-102 | 3.151e-102 | ✓ | ok | 5.345e-75 | 5.345e-75 | ✓ | ok |

## Strata — context (mf↔wf)

McNemar: b = первый в narrative; c = второй в narrative.

**Ключевые заметки:** c≫b → recency. First-Shown (context): choice = (prefers_man@mf + prefers_woman@wf)/(n_mf+n_wf); prob = mean(p_man@mf ∪ p_woman@wf). &lt;50% → сдвиг ко второму в narrative. Таблица: n, b, c, flip, 1st, p.

### p0/p1, without_abstain

| pos | ch_n | ch_b | ch_c | ch_flip | ch_1st | ch_p | ch_q | pr_n | pr_b | pr_c | pr_flip | pr_1st | pr_p | pr_q |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p0 | 951 | 23 | 199 | 23.3% | 40.7% | 7.472e-32 | 2.989e-31 | 779 | 12 | 162 | 22.3% | 47.6% | 1.379e-29 | 1.838e-29 |
| p1 | 951 | 57 | 240 | 31.2% | 40.4% | 4.532e-26 | 9.064e-26 | 807 | 44 | 201 | 30.4% | 46.9% | 2.137e-23 | 2.137e-23 |

### p0–p5, with_abstain

| pos | ch_n | ch_b | ch_c | ch_flip | ch_1st | ch_p | ch_q | pr_n | pr_b | pr_c | pr_flip | pr_1st | pr_p | pr_q |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p0 | 92 | 2 | 43 | — | — | 2.479e-09 | 2.833e-09 | 658 | 10 | 225 | — | — | 2.741e-44 | 5.482e-44 |
| p1 | 98 | 3 | 25 | — | — | 7.229e-05 | 7.229e-05 | 806 | 52 | 235 | — | — | 6.385e-27 | 7.297e-27 |
| p2 | 268 | 9 | 116 | — | — | 2.52e-21 | 4.032e-21 | 712 | 24 | 279 | — | — | 3.162e-48 | 1.265e-47 |
| p3 | 165 | 6 | 74 | — | — | 6.844e-14 | 9.125e-14 | 813 | 38 | 305 | — | — | 8.877e-47 | 2.367e-46 |
| p4 | 409 | 2 | 131 | — | — | 1.268e-28 | 3.383e-28 | 740 | 6 | 206 | — | — | 1.59e-42 | 2.544e-42 |
| p5 | 355 | 6 | 177 | — | — | 3.216e-36 | 2.573e-35 | 753 | 7 | 308 | — | — | 4.27e-64 | 3.416e-63 |

### BH-FDR — `h1v1_strata_context_mcnemar` / `h1v1_strata_context_mcnemar_prob` (Q=0.05)

Choice: **8** тестов, значимо после BH: **8**. Prob: **8** тестов, значимо после BH: **8**. Сортировка по **q-value** (возрастание). Значимо ⟺ q-value ≤ Q.

#### `h1v1_strata_context_mcnemar` (choice)

| slice | p | q-value | BH | rel |
| :--- | ---: | ---: | ---: | ---: |
| p5/with_abstain | 3.216e-36 | 2.573e-35 | ✓ | ok |
| p0/without_abstain | 7.472e-32 | 2.989e-31 | ✓ | ok |
| p4/with_abstain | 1.268e-28 | 3.383e-28 | ✓ | ok |
| p1/without_abstain | 4.532e-26 | 9.064e-26 | ✓ | ok |
| p2/with_abstain | 2.52e-21 | 4.032e-21 | ✓ | ok |
| p3/with_abstain | 6.844e-14 | 9.125e-14 | ✓ | ok |
| p0/with_abstain | 2.479e-09 | 2.833e-09 | ✓ | ok |
| p1/with_abstain | 7.229e-05 | 7.229e-05 | ✓ | ok |

#### `h1v1_strata_context_mcnemar_prob` (prob_constrained)

| slice | p | q-value | BH | rel |
| :--- | ---: | ---: | ---: | ---: |
| p5/with_abstain | 4.27e-64 | 3.416e-63 | ✓ | ok |
| p2/with_abstain | 3.162e-48 | 1.265e-47 | ✓ | ok |
| p3/with_abstain | 8.877e-47 | 2.367e-46 | ✓ | ok |
| p0/with_abstain | 2.741e-44 | 5.482e-44 | ✓ | ok |
| p4/with_abstain | 1.59e-42 | 2.544e-42 | ✓ | ok |
| p0/without_abstain | 1.379e-29 | 1.838e-29 | ✓ | ok |
| p1/with_abstain | 6.385e-27 | 7.297e-27 | ✓ | ok |
| p1/without_abstain | 2.137e-23 | 2.137e-23 | ✓ | ok |

## Rates overview

See `rates_summary.csv`.
