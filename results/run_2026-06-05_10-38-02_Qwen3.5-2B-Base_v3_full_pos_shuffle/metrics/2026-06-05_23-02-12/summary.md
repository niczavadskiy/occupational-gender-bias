# Behavioral metrics summary

- **Run:** `run_2026-06-05_10-38-02_Qwen3.5-2B-Base_v3_full_pos_shuffle`
- **Level:** `family`
- **FDR target:** 0.05
- **Generated:** 2026-06-05T19:02:12.160421+00:00
- **Надёжность:** ⚠ **Данные ненадёжные** — не интерпретировать p/FDR; ⚡ *осторожно* — мало событий или насыщение

## H1 — gender axis preference

P(man) vs P(woman) на всех вопросах; strata по evidence и format.

- `h1_primary_man_vs_woman` Главный тест H1: P(pro-man) vs P(pro-woman) на всех main-вопросах (все форматы, evidence, abstain) (all formats, evidence, abstain): P(man)=0.285 vs P(woman)=0.289 (k=3081/10800, k_w=3121), p=0.5475
- `h1_ev_no_evidence_man_vs_woman` P(man) vs P(woman) без evidence (2+3 опции) (no_evidence, 2+3 options): P(man)=0.234 vs P(woman)=0.231 (k=843/3600, k_w=833), p=0.7803, q=0.7803
- `h1_ev_ev_man_man_vs_woman` P(man) vs P(woman) при evidence за мужчин (2+3 опции) (ev_man, 2+3 options): P(man)=0.438 vs P(woman)=0.179 (k=1577/3600, k_w=643), p=1.436e-125, q=6.462e-125 **✓**
- `h1_ev_ev_woman_man_vs_woman` P(man) vs P(woman) при evidence за женщин (2+3 опции) (ev_woman, 2+3 options): P(man)=0.184 vs P(woman)=0.457 (k=661/3600, k_w=1645), p=2.329e-136, q=2.096e-135 **✓**
- `h1_ev_no_evidence_man_vs_woman_2opt` P(man) vs P(woman) без evidence (2 опции A/B) (no_evidence, 2 options): P(man)=0.542 vs P(woman)=0.458 (k=488/900, k_w=412), p=0.0003401, q=0.0004373 **✓**
- `h1_ev_ev_man_man_vs_woman_2opt` P(man) vs P(woman) при ev_man (2 опции A/B) (ev_man, 2 options): P(man)=0.698 vs P(woman)=0.302 (k=628/900, k_w=272), p=3.305e-63, q=5.949e-63 **✓**
- `h1_ev_ev_woman_man_vs_woman_2opt` P(man) vs P(woman) при ev_woman (2 опции A/B) (ev_woman, 2 options): P(man)=0.314 vs P(woman)=0.686 (k=283/900, k_w=617), p=7.444e-56, q=1.117e-55 **✓**

## H3 — evidence vs ambiguous context

- `h3_chi2_evidence_x_outcome` Связь типа evidence (нет / за man / за woman) с выбором man vs woman (все форматы, все abstain-варианты) (all formats, man/woman): stat=812.674, p=3.389e-177
- `h3_chi2_evidence_x_outcome_with_abstain` Связь evidence с исходом man / woman / abstain (C) при 3 опциях (with_abstain, все форматы) (with_abstain, man/woman/C): stat=997.206, p=1.439e-214
- `h3_pair_no_vs_man` Сравнение доли «man» без evidence и при evidence в пользу мужчин: p1=0.234, p2=0.438, Δ=-0.204, p=6.613e-75, q=9.92e-75 **✓**
- `h3_pair_no_vs_woman` Сравнение доли «man» без evidence и при evidence в пользу женщин: p1=0.234, p2=0.184, Δ=+0.051, p=1.318e-07, q=1.318e-07 **✓**
- `h3_pair_man_vs_woman` Сравнение доли «man» при evidence за man и за woman: p1=0.438, p2=0.184, Δ=+0.254, p=2.53e-120, q=7.59e-120 **✓**
- `h3_pair_woman_no_vs_ev_man` Сравнение доли «woman» без evidence и при evidence в пользу мужчин: p1=0.231, p2=0.179, Δ=+0.053, p=2.913e-08, q=3.495e-08 **✓**
- `h3_pair_woman_no_vs_ev_woman` Сравнение доли «woman» без evidence и при evidence в пользу женщин: p1=0.231, p2=0.457, Δ=-0.226, p=3.145e-90, q=6.29e-90 **✓**
- `h3_pair_woman_ev_man_vs_ev_woman` Сравнение доли «woman» при evidence за man и за woman: p1=0.179, p2=0.457, Δ=-0.278, p=6.692e-142, q=4.015e-141 **✓**
- `h3_paired_family_no_vs_ev_woman` Парно по семействам: сдвиг доли «man» от no_evidence к evidence за woman vs 0: paired family Δ=+0.051, p=8.401e-09

## H4 — answerability ↑ with abstain option

Self-Q (`task=answerability`, all positions): P(Yes) when main options include C vs A/B only.


**McNemar (парные сценарии):** один и тот же сценарий оценивается дважды — с опцией «Cannot determine» в основном вопросе (with_abstain) и без (without). Таблица 2×2: **строка** = ответ self-Q при with_abstain, **столбец** = при without.

|  | without=No | without=Yes |
|---|---:|---:|
| **with=No** | **d** (оба No) | **c** (дискордант: No→Yes) |
| **with=Yes** | **b** (дискордант: Yes→No) | **a** (оба Yes) |

**a**, **d** — конкордантные пары; **b**, **c** — дискордантные. H4↑: **b > c** (чаще Yes при with_abstain); H4↓: **c > b** (чаще Yes при without).

- `h4_mcnemar_yes_lower_with_vs_without` Парный McNemar (H4↓): тот же срез; H1: c>b (чаще Yes без abstain)
  a=78 (оба Yes), b=48 (Yes при with / No при without), c=134 (No при with / Yes при without), d=2440 (оба No), n_pairs=2700; таблица 2×2: строка=with_abstain, столбец=without_abstain; H1 односторонний: c>b, p=2.965e-10
- `h4_mcnemar_yes_with_vs_without` Парный McNemar (H4↑): для каждого сценария сравнивается self-Q Yes при with_abstain и without (все форматы); см. таблицу a/b/c/d выше; H1: b>c
  a=78 (оба Yes), b=48 (Yes при with / No при without), c=134 (No при with / Yes при without), d=2440 (оба No), n_pairs=2700; таблица 2×2: строка=with_abstain, столбец=without_abstain; H1 односторонний: b>c, p=1
- `h4_primary_yes_lower_with_vs_without_abstain` P(Yes на self-Q answerability) ниже, когда в основном вопросе показана опция «Cannot determine» (with_abstain vs without), все форматы, парные сценарии: P(Yes|with)=0.047 vs P(Yes|without)=0.079, Δ=-0.032, p=6.778e-07 ⚡ *осторожно* — группа1 p_hat=0.047 экстремальная доля
- `h4_primary_yes_with_vs_without_abstain` P(Yes на self-Q answerability) выше, когда в основном вопросе показана опция «Cannot determine» (with_abstain vs without), все форматы, парные сценарии: P(Yes|with)=0.047 vs P(Yes|without)=0.079, Δ=-0.032, p=1, q=1 ⚡ *осторожно* — группа1 p_hat=0.047 экстремальная доля

## H5 — abstain rate: no_evidence vs evidence

- `h5_abstain_no_vs_ev_man` Сравнение доли abstain (C) без evidence и при evidence в пользу мужчин: P(C) 0.872 vs 0.564, p=9.756e-48, q=9.756e-48 **✓**
- `h5_abstain_no_vs_ev_woman` Сравнение доли abstain (C) без evidence и при evidence в пользу женщин: P(C) 0.872 vs 0.441, p=1.155e-82, q=2.31e-82 **✓**

## H6 — choice vs yes/no format

- `h6_two_prop_choice_man_vs_yesno_man` Сравнение P(man|choice) и P(Да|yesno про мужчину) на одной оси (man) (same-axis man): P(choice)=0.238 vs P(Yes|yesno)=0.439, p=1.839e-72, q=7.356e-72 **✓**
- `h6_two_prop_choice_woman_vs_yesno_woman` Сравнение P(woman|choice) и P(Да|yesno про женщину) на одной оси (woman) (same-axis woman): P(choice)=0.292 vs P(Yes|yesno)=0.377, p=3.109e-14, q=3.109e-14 **✓**
- `h6_two_prop_choice_man_vs_yesno_woman` Сравнение P(man|choice) и P(Да|yesno про женщину) — перекрёстные форматы (cross man vs yesno_woman): P(choice)=0.238 vs P(Yes|yesno)=0.377, p=4.677e-37, q=6.235e-37 **✓**
- `h6_two_prop_choice_woman_vs_yesno_man` Сравнение P(woman|choice) и P(Да|yesno про мужчину) — перекрёстные форматы (cross woman vs yesno_man): P(choice)=0.292 vs P(Yes|yesno)=0.439, p=2.528e-38, q=5.056e-38 **✓**

## H7 — context ↑ answerability


**McNemar H7:** один сценарий (base_id × format × abstain × position × context), сравнивается self-Q Yes при **evidence** vs **no_evidence**. Таблица 2×2: **строка** = evidence, **столбец** = no_evidence.

|  | no_evidence=No | no_evidence=Yes |
|---|---:|---:|
| **evidence=No** | **d** (оба No) | **c** (No→Yes) |
| **evidence=Yes** | **b** (Yes→No) | **a** (оба Yes) |

H7↑: **b > c** (чаще Yes с evidence); H7↓: **c > b** (чаще Yes без evidence).

- `h7_mcnemar_yes_ev_man_vs_no` Парный McNemar (H7↑): self-Q Yes при ev_man vs no_evidence на тех же сценариях
  a=153 (оба Yes), b=135 (Yes при ev_man / No при no_evidence), c=218 (No при ev_man / Yes при no_evidence), d=3094 (оба No), n_pairs=3600; таблица 2×2: строка=evidence_supports_man, столбец=no_evidence; H1 односторонний: b>c, p=1
- `h7_mcnemar_yes_ev_woman_vs_no` Парный McNemar (H7↑): self-Q Yes при ev_woman vs no_evidence на тех же сценариях
  a=93 (оба Yes), b=102 (Yes при ev_woman / No при no_evidence), c=278 (No при ev_woman / Yes при no_evidence), d=3127 (оба No), n_pairs=3600; таблица 2×2: строка=evidence_supports_woman, столбец=no_evidence; H1 односторонний: b>c, p=1
- `h7_mcnemar_yes_lower_ev_man_vs_no` Парный McNemar (H7↓): self-Q Yes при ev_man vs no_evidence; H1: c>b (чаще Yes без evidence)
  a=153 (оба Yes), b=135 (Yes при ev_man / No при no_evidence), c=218 (No при ev_man / Yes при no_evidence), d=3094 (оба No), n_pairs=3600; таблица 2×2: строка=evidence_supports_man, столбец=no_evidence; H1 односторонний: c>b, p=1.275e-05
- `h7_mcnemar_yes_lower_ev_woman_vs_no` Парный McNemar (H7↓): self-Q Yes при ev_woman vs no_evidence; H1: c>b (чаще Yes без evidence)
  a=93 (оба Yes), b=102 (Yes при ev_woman / No при no_evidence), c=278 (No при ev_woman / Yes при no_evidence), d=3127 (оба No), n_pairs=3600; таблица 2×2: строка=evidence_supports_woman, столбец=no_evidence; H1 односторонний: c>b, p=2.775e-19
- `h7_primary_yes_evidence_vs_no` P(Yes на self-Q) выше при evidence (man+woman), чем при no_evidence (H7↑): P₁=0.067 vs P₂=0.103, Δ=-0.036, p=1, q=1
- `h7_primary_yes_lower_evidence_vs_no` P(Yes на self-Q) ниже при evidence (man+woman), чем при no_evidence (H7↓): P₁=0.067 vs P₂=0.103, Δ=-0.036, p=3.278e-11, q=4.918e-11 **✓**
- `h7_yes_ev_man_vs_no` P(Yes) выше при evidence за man vs no_evidence (H7↑): P₁=0.080 vs P₂=0.103, Δ=-0.023, p=0.9997, q=1
- `h7_yes_ev_woman_vs_no` P(Yes) выше при evidence за woman vs no_evidence (H7↑): P₁=0.054 vs P₂=0.103, Δ=-0.049, p=1, q=1
- `h7_yes_lower_ev_man_vs_no` P(Yes) ниже при evidence за man vs no_evidence (H7↓): P₁=0.080 vs P₂=0.103, Δ=-0.023, p=0.0003467, q=0.0003467 **✓**
- `h7_yes_lower_ev_woman_vs_no` P(Yes) ниже при evidence за woman vs no_evidence (H7↓): P₁=0.054 vs P₂=0.103, Δ=-0.049, p=6.443e-15, q=1.933e-14 **✓**

## H8 — unanswerable ↔ abstain


**McNemar H8:** строка = self-Q (No/Yes), столбец = main choice C; H1: **c > b** (чаще C при self=No).

- `h8_abstain_self_no_vs_yes_ev_man` H8 при evidence за man: P₁=0.550 vs P₂=0.025, Δ=+0.524, p=4.115e-46, q=5.487e-46 **✓** ⚡ *осторожно* — группа2 min(k,n-k)=5<10; группа2 p_hat=0.025 экстремальная доля
- `h8_abstain_self_no_vs_yes_ev_woman` H8 при evidence за woman: P₁=0.500 vs P₂=0.033, Δ=+0.467, p=4.468e-24, q=4.468e-24 **✓** ⚡ *осторожно* — группа2 min(k,n-k)=4<10; группа2 p_hat=0.033 экстремальная доля
- `h8_abstain_self_no_vs_yes_no_evidence` H8 на срезе no_evidence: P₁=0.759 vs P₂=0.375, Δ=+0.384, p=1.017e-46, q=2.035e-46 **✓**
- `h8_mcnemar_self_no_abstain` Парный McNemar H8: self-Q ↔ main C на тех же сценариях (with_abstain)
  a=130 (оба Yes), b=512 (self Yes / не C), c=4468 (self No / C), d=2990 (оба No), n_pairs=8100; таблица 2×2: строка=self_No, столбец=main_C; H1 односторонний: c>b, p=0
- `h8_primary_abstain_self_no_vs_yes` P(C на main) выше, когда self-Q answerability = No (unanswerable): P₁=0.599 vs P₂=0.202, Δ=+0.397, p=1.114e-84, q=4.455e-84 **✓**

## H9 — answerability by supported gender

- `h9_mcnemar_ev_man_vs_ev_woman` Парный McNemar H9: Yes при ev_man vs ev_woman на тех же сценариях
  a=66 (оба Yes), b=222 (Yes при ev_man / No при ev_woman), c=129 (No при ev_man / Yes при ev_woman), d=3183 (оба No), n_pairs=3600; таблица 2×2: строка=ev_man, столбец=ev_woman; H1 односторонний: b>c, p=9.08e-07
- `h9_primary_yes_ev_man_vs_ev_woman` P(Yes на self-Q) при evidence за man vs evidence за woman: P₁=0.080 vs P₂=0.054, Δ=+0.026, p=1.181e-05, q=1.181e-05 **✓**

## Rates (choice rows, excerpt)


## FDR correction (Benjamini–Hochberg)

- **Target FDR:** 0.05
- **Tests with q-value:** 44
- **Rejected** (`rejected_fdr=true`): **38**
- **Not rejected** (`q_value` set, `rejected_fdr=false`): 6
- **Outside FDR families** (no q-value): 14

### Rejected at FDR (`rejected_fdr = true`)

- **`h1_vs_half_abst_with_abstain`** (H1), family `h1_strata_abstain`: p=0, q=0, Δ=-0.292 — P(man) vs 0.5: abstain_variant=with_abstain
- **`h1_ev_ev_woman_man_vs_woman`** P(man) vs P(woman) при evidence за женщин (2+3 опции) (H1), family `h1_strata_evidence_gender_pair`: p=2.329e-136, q=2.096e-135, Δ=-0.273 — P(man) vs P(woman), evidence=evidence_supports_woman, 2+3 options
- **`h1_ev_ev_man_man_vs_woman`** P(man) vs P(woman) при evidence за мужчин (2+3 опции) (H1), family `h1_strata_evidence_gender_pair`: p=1.436e-125, q=6.462e-125, Δ=+0.259 — P(man) vs P(woman), evidence=evidence_supports_man, 2+3 options
- **`h1_ev_ev_woman_man_vs_woman_3opt`** P(man) vs P(woman) при ev_woman (3 опции A/B/C) (H1), family `h1_strata_evidence_gender_pair`: p=2.363e-90, q=7.09e-90, Δ=-0.241 — P(man) vs P(woman), evidence=evidence_supports_woman, 3 options A/B/C
- **`h1_ev_ev_man_man_vs_woman_3opt`** P(man) vs P(woman) при ev_man (3 опции A/B/C) (H1), family `h1_strata_evidence_gender_pair`: p=7.921e-75, q=1.782e-74, Δ=+0.214 — P(man) vs P(woman), evidence=evidence_supports_man, 3 options A/B/C
- **`h1_ev_ev_man_man_vs_woman_2opt`** P(man) vs P(woman) при ev_man (2 опции A/B) (H1), family `h1_strata_evidence_gender_pair`: p=3.305e-63, q=5.949e-63, Δ=+0.396 — P(man) vs P(woman), evidence=evidence_supports_man, 2 options A/B
- **`h1_ev_ev_woman_man_vs_woman_2opt`** P(man) vs P(woman) при ev_woman (2 опции A/B) (H1), family `h1_strata_evidence_gender_pair`: p=7.444e-56, q=1.117e-55, Δ=-0.371 — P(man) vs P(woman), evidence=evidence_supports_woman, 2 options A/B
- **`h1_ev_no_evidence_man_vs_woman_2opt`** P(man) vs P(woman) без evidence (2 опции A/B) (H1), family `h1_strata_evidence_gender_pair`: p=0.0003401, q=0.0004373, Δ=+0.084 — P(man) vs P(woman), evidence=no_evidence, 2 options A/B
- **`h1_ev_no_evidence_man_vs_woman_3opt`** P(man) vs P(woman) без evidence (3 опции A/B/C) (H1), family `h1_strata_evidence_gender_pair`: p=0.01046, q=0.01176, Δ=-0.024 — P(man) vs P(woman), evidence=no_evidence, 3 options A/B/C
- **`h1_vs_half_fmt_choice`** (H1), family `h1_strata_format`: p=2.023e-216, q=3.035e-216, Δ=-0.262 — P(pro-man) vs 0.5: format=choice (semantic labels)
- **`h1_vs_half_fmt_yesno_man`** (H1), family `h1_strata_format`: p=2.878e-13, q=2.878e-13, Δ=-0.061 — P(pro-man) vs 0.5: format=yesno_man (semantic labels)
- **`h1_vs_half_fmt_yesno_woman`** (H1), family `h1_strata_format`: p=0, q=0, Δ=-0.322 — P(pro-man) vs 0.5: format=yesno_woman (semantic labels)
- **`h1_vs_half_pos_p1`** (H1), family `h1_strata_position`: p=4.434e-154, q=2.66e-153, Δ=-0.254 — P(man) vs 0.5: position=p1
- **`h1_vs_half_pos_p2`** (H1), family `h1_strata_position`: p=9.365e-144, q=2.809e-143, Δ=-0.347 — P(man) vs 0.5: position=p2
- **`h1_vs_half_pos_p4`** (H1), family `h1_strata_position`: p=1.541e-95, q=3.081e-95, Δ=-0.282 — P(man) vs 0.5: position=p4
- **`h1_vs_half_pos_p3`** (H1), family `h1_strata_position`: p=2.473e-60, q=3.71e-60, Δ=-0.223 — P(man) vs 0.5: position=p3
- **`h1_vs_half_pos_p5`** (H1), family `h1_strata_position`: p=8.671e-59, q=1.041e-58, Δ=-0.220 — P(man) vs 0.5: position=p5
- **`h1_vs_half_pos_p0`** (H1), family `h1_strata_position`: p=1.419e-12, q=1.419e-12, Δ=-0.068 — P(man) vs 0.5: position=p0
- **`h3_pair_woman_ev_man_vs_ev_woman`** Сравнение доли «woman» при evidence за man и за woman (H3), family `h3_evidence_posthoc`: p=6.692e-142, q=4.015e-141, Δ=-0.278 — P(woman): evidence_supports_man vs evidence_supports_woman
- **`h3_pair_man_vs_woman`** Сравнение доли «man» при evidence за man и за woman (H3), family `h3_evidence_posthoc`: p=2.53e-120, q=7.59e-120, Δ=+0.254 — P(man): evidence_supports_man vs evidence_supports_woman
- **`h3_pair_woman_no_vs_ev_woman`** Сравнение доли «woman» без evidence и при evidence в пользу женщин (H3), family `h3_evidence_posthoc`: p=3.145e-90, q=6.29e-90, Δ=-0.226 — P(woman): no_evidence vs evidence_supports_woman
- **`h3_pair_no_vs_man`** Сравнение доли «man» без evidence и при evidence в пользу мужчин (H3), family `h3_evidence_posthoc`: p=6.613e-75, q=9.92e-75, Δ=-0.204 — P(man): no_evidence vs evidence_supports_man
- **`h3_pair_woman_no_vs_ev_man`** Сравнение доли «woman» без evidence и при evidence в пользу мужчин (H3), family `h3_evidence_posthoc`: p=2.913e-08, q=3.495e-08, Δ=+0.053 — P(woman): no_evidence vs evidence_supports_man
- **`h3_pair_no_vs_woman`** Сравнение доли «man» без evidence и при evidence в пользу женщин (H3), family `h3_evidence_posthoc`: p=1.318e-07, q=1.318e-07, Δ=+0.051 — P(man): no_evidence vs evidence_supports_woman
- **`h5_abstain_no_vs_ev_woman`** Сравнение доли abstain (C) без evidence и при evidence в пользу женщин (H5), family `h5_abstain_by_evidence`: p=1.155e-82, q=2.31e-82, Δ=+0.431 — P(abstain C): no_evidence vs evidence_supports_woman (with_abstain, all formats/positions)
- **`h5_abstain_no_vs_ev_man`** Сравнение доли abstain (C) без evidence и при evidence в пользу мужчин (H5), family `h5_abstain_by_evidence`: p=9.756e-48, q=9.756e-48, Δ=+0.308 — P(abstain C): no_evidence vs evidence_supports_man (with_abstain, all formats/positions)
- **`h6_two_prop_choice_man_vs_yesno_man`** Сравнение P(man|choice) и P(Да|yesno про мужчину) на одной оси (man) (H6), family `h6_format_pairs`: p=1.839e-72, q=7.356e-72, Δ=-0.201 — P(man|choice) vs P(Yes|yesno_man), same-axis (all evidence, abstain variants, positions, contexts)
- **`h6_two_prop_choice_woman_vs_yesno_man`** Сравнение P(woman|choice) и P(Да|yesno про мужчину) — перекрёстные форматы (H6), family `h6_format_pairs`: p=2.528e-38, q=5.056e-38, Δ=-0.147 — P(woman|choice) vs P(Yes|yesno_man), cross-axis (all evidence, abstain variants, positions, contexts)
- **`h6_two_prop_choice_man_vs_yesno_woman`** Сравнение P(man|choice) и P(Да|yesno про женщину) — перекрёстные форматы (H6), family `h6_format_pairs`: p=4.677e-37, q=6.235e-37, Δ=-0.138 — P(man|choice) vs P(Yes|yesno_woman), cross-axis (all evidence, abstain variants, positions, contexts)
- **`h6_two_prop_choice_woman_vs_yesno_woman`** Сравнение P(woman|choice) и P(Да|yesno про женщину) на одной оси (woman) (H6), family `h6_format_pairs`: p=3.109e-14, q=3.109e-14, Δ=-0.084 — P(woman|choice) vs P(Yes|yesno_woman), same-axis (all evidence, abstain variants, positions, contexts)
- **`h7_yes_lower_ev_woman_vs_no`** P(Yes) ниже при evidence за woman vs no_evidence (H7↓) (H7), family `h7_evidence_answerability_lower`: p=6.443e-15, q=1.933e-14, Δ=-0.049 — P(Yes|answerability) lower for evidence_supports_woman vs no_evidence (all formats, positions, abstain; one-sided smaller)
- **`h7_primary_yes_lower_evidence_vs_no`** P(Yes на self-Q) ниже при evidence (man+woman), чем при no_evidence (H7↓) (H7), family `h7_evidence_answerability_lower`: p=3.278e-11, q=4.918e-11, Δ=-0.036 — P(Yes|answerability) lower for evidence (man+woman) vs no_evidence (all formats, positions, abstain; one-sided smaller)
- **`h7_yes_lower_ev_man_vs_no`** P(Yes) ниже при evidence за man vs no_evidence (H7↓) (H7), family `h7_evidence_answerability_lower`: p=0.0003467, q=0.0003467, Δ=-0.023 — P(Yes|answerability) lower for evidence_supports_man vs no_evidence (all formats, positions, abstain; one-sided smaller)
- **`h8_primary_abstain_self_no_vs_yes`** P(C на main) выше, когда self-Q answerability = No (unanswerable) (H8), family `h8_self_abstain_link`: p=1.114e-84, q=4.455e-84, Δ=+0.397 — P(C|main) higher when self-Q=No vs self-Q=Yes (with_abstain, all formats and positions)
- **`h8_abstain_self_no_vs_yes_no_evidence`** H8 на срезе no_evidence (H8), family `h8_self_abstain_link`: p=1.017e-46, q=2.035e-46, Δ=+0.384 — H8: P(C|self=No) vs P(C|self=Yes), evidence=no_evidence (with_abstain)
- **`h8_abstain_self_no_vs_yes_ev_man`** H8 при evidence за man (H8), family `h8_self_abstain_link`: p=4.115e-46, q=5.487e-46, Δ=+0.524 — H8: P(C|self=No) vs P(C|self=Yes), evidence=evidence_supports_man (with_abstain) ⚡ *осторожно* — группа2 min(k,n-k)=5<10; группа2 p_hat=0.025 экстремальная доля
- **`h8_abstain_self_no_vs_yes_ev_woman`** H8 при evidence за woman (H8), family `h8_self_abstain_link`: p=4.468e-24, q=4.468e-24, Δ=+0.467 — H8: P(C|self=No) vs P(C|self=Yes), evidence=evidence_supports_woman (with_abstain) ⚡ *осторожно* — группа2 min(k,n-k)=4<10; группа2 p_hat=0.033 экстремальная доля
- **`h9_primary_yes_ev_man_vs_ev_woman`** P(Yes на self-Q) при evidence за man vs evidence за woman (H9), family `h9_gender_evidence_answerability`: p=1.181e-05, q=1.181e-05, Δ=+0.026 — P(Yes|answerability) ev_man vs ev_woman (all formats, positions, abstain; two-sided)

### Not rejected (`q_value` present, `rejected_fdr = false`)

- **`h1_vs_half_abst_without_abstain`** (H1), family `h1_strata_abstain`: p=0.05929, q=0.05929, Δ=+0.018 — P(man) vs 0.5: abstain_variant=without_abstain
- **`h1_ev_no_evidence_man_vs_woman`** P(man) vs P(woman) без evidence (2+3 опции) (H1), family `h1_strata_evidence_gender_pair`: p=0.7803, q=0.7803, Δ=+0.003 — P(man) vs P(woman), evidence=no_evidence, 2+3 options
- **`h4_primary_yes_with_vs_without_abstain`** P(Yes на self-Q answerability) выше, когда в основном вопросе показана опция «Cannot determine» (with_abstain vs without), все форматы, парные сценарии (H4), family `h4_abstain_answerability`: p=1, q=1, Δ=-0.032 — P(Yes|answerability) higher with_abstain vs without (all formats, paired scenarios, one-sided larger) ⚡ *осторожно* — группа1 p_hat=0.047 экстремальная доля
- **`h7_primary_yes_evidence_vs_no`** P(Yes на self-Q) выше при evidence (man+woman), чем при no_evidence (H7↑) (H7), family `h7_evidence_answerability`: p=1, q=1, Δ=-0.036 — P(Yes|answerability) higher for evidence (man+woman) vs no_evidence (all formats, positions, abstain; one-sided larger)
- **`h7_yes_ev_man_vs_no`** P(Yes) выше при evidence за man vs no_evidence (H7↑) (H7), family `h7_evidence_answerability`: p=0.9997, q=1, Δ=-0.023 — P(Yes|answerability) higher for evidence_supports_man vs no_evidence (all formats, positions, abstain; one-sided larger)
- **`h7_yes_ev_woman_vs_no`** P(Yes) выше при evidence за woman vs no_evidence (H7↑) (H7), family `h7_evidence_answerability`: p=1, q=1, Δ=-0.049 — P(Yes|answerability) higher for evidence_supports_woman vs no_evidence (all formats, positions, abstain; one-sided larger)

## Post-hoc H — yes/no position control

_(Not H6; see `post_hoc_H.csv`)_

- `post_h_two_prop_yesno_man_vs_yesno_woman` Сравнение P(Да|yesno про мужчину) и P(Да|yesno про женщину) при прочих равных: P(Yes|yesno_man)=0.443 vs P(Yes|yesno_woman)=0.317, p=0.001393
- `post_h_mcnemar_yesno_pair` Парный McNemar: yesno_man vs yesno_woman на тех же сценариях (a=оба Yes, b=Yes man/No woman, c=No man/Yes woman, d=оба No)
  a=82 (оба Yes), b=51 (Yes на man / No на woman), c=13 (No на man / Yes на woman), d=154 (оба No), n_pairs=300; таблица 2×2: строка=yesno_man, столбец=yesno_woman; H1 односторонний: b>c, p=3.746e-06

## Probing cross-reference

Compare with `results/<run>/probes/h11/` (log_odds decodable from HS).
