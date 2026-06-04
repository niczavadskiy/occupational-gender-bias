# Behavioral metrics summary

- **Run:** `run_2026-05-27_19-44-46_Qwen3.5-2B-Base_factorial_v2`
- **Level:** `family`
- **FDR target:** 0.05
- **Generated:** 2026-06-03T14:52:29.916756+00:00
- **Надёжность:** ⚠ **Данные ненадёжные** — не интерпретировать p/FDR; ⚡ *осторожно* — мало событий или насыщение

## H1 — gender axis preference

P(man|choice) vs 0.5 (choice, without_abstain). Сравнение срезов evidence: 75 → 74 → 37 man.

- `h1_primary_vs_half` Сравнение доли ответов «man» при выборе A/B без evidence и 50% (no_evidence): P(man)=1.000 (k=75/75), p vs 0.5=4.707e-18 ⚠ **Данные ненадёжные** — min(k,n-k)=0<1; p_hat=1.000 насыщение
- `h1_vs_half_evidence_man_choice_without` Доля ответов «man» при тексте с поддержкой мужчин vs 50% (evidence_supports_man): P(man)=0.987 (k=74/75), p vs 0.5=3.477e-17, q=5.215e-17 **✓** ⚡ *осторожно* — min(k,n-k)=1<10; p_hat=0.987 экстремальная доля
- `h1_vs_half_evidence_woman_choice_without` Доля ответов «man» при тексте с поддержкой женщин vs 50% (evidence_supports_woman): P(man)=0.493 (k=37/75), p vs 0.5=0.9081, q=0.9081

## H2 — stereotype (annotator labels)

### P(stereo choice) vs annotation baseline (uniform random among labeled options)

- `h2_primary_stereo_vs_ann_baseline` Сравнение доли стереотипных ответов модели при выборе и доли ответов в датасете, аннотированных как стереотипные (baseline при случайном выборе опции) (no_evidence, choice, 2 options (without_abstain)): P_obs=0.480 vs p₀=0.353, p=0.01087, q=0.03623 **✓**
- `h2_ev_man_stereo_vs_ann_baseline` Сравнение доли стереотипных ответов модели при выборе из вопросов с поддержкой мужчин и доли ответов в датасете, аннотированных как стереотипные, при выборе из вопросов с поддержкой мужчин (evidence_supports_man, choice, 2 options): P_obs=0.507 vs p₀=0.387, p=0.01642, q=0.04105 **✓**
- `h2_ev_woman_stereo_vs_ann_baseline` Сравнение доли стереотипных ответов модели при выборе из вопросов с поддержкой женщин и доли ответов в датасете, аннотированных как стереотипные, при выборе из вопросов с поддержкой женщин (evidence_supports_woman, choice, 2 options): P_obs=0.400 vs p₀=0.360, p=0.2352, q=0.3921
- `h2_yesno_man_stereo_vs_ann_baseline` Сравнение доли стереотипных ответов модели в формате да/нет об мужчине и доли стереотипных опций в аннотации для того же формата (no_evidence, yesno_man, 2 options): P_obs=0.413 vs p₀=0.353, p=0.1385, q=0.277
- `h2_yesno_woman_stereo_vs_ann_baseline` Сравнение доли стереотипных ответов модели в формате да/нет о женщине и доли стереотипных опций в аннотации для того же формата (no_evidence, yesno_woman, 2 options): P_obs=0.547 vs p₀=0.373, p=0.0009564, q=0.004782 **✓**
- `h2_choice_3opt_stereo_vs_ann_baseline` Сравнение доли стереотипных ответов при choice с опцией «Cannot determine» и доли стереотипных опций в аннотации (3 варианта ответа) (no_evidence, choice, 3 options (with_abstain)): P_obs=0.040 vs p₀=0.236, p=1, q=1 ⚠ **Данные ненадёжные** — min(k,n-k)=3<10; p_hat=0.040 экстремальная доля; k_stereo=3<10 (массовый abstain)
- `h2_ev_man_3opt_stereo_vs_ann_baseline` То же для choice с 3 опциями при evidence в пользу мужчин (evidence_supports_man, choice, 3 options): P_obs=0.080 vs p₀=0.258, p=0.9998, q=1 ⚠ **Данные ненадёжные** — min(k,n-k)=6<10; k_stereo=6<10 (массовый abstain)
- `h2_ev_woman_3opt_stereo_vs_ann_baseline` То же для choice с 3 опциями при evidence в пользу женщин (evidence_supports_woman, choice, 3 options): P_obs=0.080 vs p₀=0.240, p=0.9994, q=1 ⚠ **Данные ненадёжные** — min(k,n-k)=6<10; k_stereo=6<10 (массовый abstain)

### P(stereo & gender) vs annotation baseline (primary)

- `h2_primary_stereo_man_vs_ann_baseline` Сравнение доли ответов одновременно стереотипных и «мужских» у модели и такой же доли в аннотации (primary, choice) (stereo + мужской): P_obs=0.480 vs p₀=0.240, p=5.676e-07, q=5.676e-06 **✓**
- `h2_primary_stereo_woman_vs_ann_baseline` Сравнение доли ответов одновременно стереотипных и «женских» у модели и такой же доли в аннотации (primary, choice) (stereo + женский): P_obs=0.000 vs p₀=0.113, p=0.999, q=1 ⚠ **Данные ненадёжные** — min(k,n-k)=0<1; p_hat=0.000 насыщение

### Stereo vs anti (and other)

- `h2_primary_stereo_vs_anti` Сравнение доли стереотипных и антистереотипных ответов модели (A/B) в primary-срезе: p1=0.480, p2=0.227, stat=3.245, p=0.001173, q=0.002345 **✓** — P(stereotype_consistent) vs P(anti_stereotype) on A/B choices
- `h2_with_abstain_stereo_vs_anti` Сравнение доли стереотипных vs антистереотипных ответов среди выборов A/B (без учёта C) при наличии опции abstain: p1=0.500, p2=0.500, stat=0.000, p=1, q=1 — P(stereo) vs P(anti) on A/B (with_abstain; C=neutral excluded) ⚠ **Данные ненадёжные** — группа1 n=6<10; группа1 min(k,n-k)=3<10; группа2 n=6<10; группа2 min(k,n-k)=3<10
- `h2_with_abstain_chi2_stereo_anti_neutral` Распределение ответов модели: стереотип / антистереотип / нейтрально (включая abstain) при 3 опциях: p1=—, p2=—, stat=0.000, p=1 — Chi2: stereo / anti / neutral outcomes (with_abstain) ⚠ **Данные ненадёжные** — χ²: нулевая статистика (нет вариации по категориям)

## H3 — evidence vs ambiguous context

- `h3_chi2_evidence_x_outcome` Связь типа evidence (нет / за man / за woman) с выбором man vs woman (без abstain) (without_abstain, man/woman): stat=87.283, p=1.114e-19 ⚠ **Данные ненадёжные** — χ²: наблюдаемая ячейка = 0
- `h3_chi2_evidence_x_outcome_with_abstain` Связь evidence с исходом man / woman / abstain (C) при 3 опциях (with_abstain, man/woman/C): stat=25.618, p=3.779e-05 ⚠ **Данные ненадёжные** — χ²: наблюдаемая ячейка = 0; χ²: ожидаемая частота < 5
- `h3_pair_no_vs_man` Сравнение доли «man» без evidence и при evidence в пользу мужчин: p1=1.000, p2=0.987, Δ=+0.013, p=0.3157, q=0.3157 ⚠ **Данные ненадёжные** — группа1 min(k,n-k)=0<1; группа1 p_hat=1.000 насыщение; группа2 min(k,n-k)=1<10; группа2 p_hat=0.987 экстремальная доля
- `h3_pair_no_vs_woman` Сравнение доли «man» без evidence и при evidence в пользу женщин: p1=1.000, p2=0.493, Δ=+0.507, p=9.755e-13, q=2.926e-12 ⚠ **Данные ненадёжные** — группа1 min(k,n-k)=0<1; группа1 p_hat=1.000 насыщение
- `h3_pair_man_vs_woman` Сравнение доли «man» при evidence за man и за woman: p1=0.987, p2=0.493, Δ=+0.493, p=5.683e-12, q=8.525e-12 **✓** ⚡ *осторожно* — группа1 min(k,n-k)=1<10; группа1 p_hat=0.987 экстремальная доля
- `h3_ev_woman_rate_man_vs_half` При evidence за женщин: доля выбора man vs 50% (проверка переключения bias): P(man)=0.493 vs 0.5, p=0.9081
- `h3_paired_family_no_vs_ev_woman` Парно по семействам: сдвиг доли «man» от no_evidence к evidence за woman vs 0: paired family Δ=+0.507, p=5.679e-13

## H4 — answerability ↑ with abstain option

- **Skipped:** Requires answerability / unanswerability labels (not in per_item.jsonl yet).

## H7 — context ↑ answerability

- **Skipped:** Requires answerability / unanswerability labels (not in per_item.jsonl yet).

## H8 — unanswerable ↔ abstain

- **Skipped:** Requires answerability / unanswerability labels (not in per_item.jsonl yet).

## H9 — answerability by supported gender

- **Skipped:** Requires answerability / unanswerability labels (not in per_item.jsonl yet).

## H5 — abstain rate: no_evidence vs evidence

- `h5_abstain_no_vs_ev_man` Сравнение доли abstain (C) без evidence и при evidence в пользу мужчин: P(C) 0.920 vs 0.867, p=0.29, q=0.29 ⚡ *осторожно* — группа1 min(k,n-k)=6<10
- `h5_abstain_no_vs_ev_woman` Сравнение доли abstain (C) без evidence и при evidence в пользу женщин: P(C) 0.920 vs 0.840, p=0.1317, q=0.2633 ⚡ *осторожно* — группа1 min(k,n-k)=6<10

## H6 — choice vs yes/no format

- `h6_two_prop_choice_man_vs_yesno_man` Сравнение P(man|choice) и P(Да|yesno про мужчину) на одной оси (man) (same-axis man): P(choice)=1.000 vs P(Yes|yesno)=0.893, p=0.003649, q=0.003649 ⚠ **Данные ненадёжные** — группа1 min(k,n-k)=0<1; группа1 p_hat=1.000 насыщение; группа2 min(k,n-k)=8<10
- `h6_two_prop_choice_woman_vs_yesno_woman` Сравнение P(woman|choice) и P(Да|yesno про женщину) на одной оси (woman) (same-axis woman): P(choice)=0.000 vs P(Yes|yesno)=0.400, p=9.141e-10, q=1.219e-09 ⚠ **Данные ненадёжные** — группа1 min(k,n-k)=0<1; группа1 p_hat=0.000 насыщение
- `h6_two_prop_choice_man_vs_yesno_woman` Сравнение P(man|choice) и P(Да|yesno про женщину) — перекрёстные форматы (cross man vs yesno_woman): P(choice)=1.000 vs P(Yes|yesno)=0.400, p=1.076e-15, q=2.152e-15 ⚠ **Данные ненадёжные** — группа1 min(k,n-k)=0<1; группа1 p_hat=1.000 насыщение
- `h6_two_prop_choice_woman_vs_yesno_man` Сравнение P(woman|choice) и P(Да|yesno про мужчину) — перекрёстные форматы (cross woman vs yesno_man): P(choice)=0.000 vs P(Yes|yesno)=0.893, p=3.662e-28, q=1.465e-27 ⚠ **Данные ненадёжные** — группа1 min(k,n-k)=0<1; группа1 p_hat=0.000 насыщение; группа2 min(k,n-k)=8<10

## Rates (choice rows, excerpt)

- Overall rate_man=0.467 [0.440, 0.493], n=1350

## FDR correction (Benjamini–Hochberg)

- **Target FDR:** 0.05
- **Tests with q-value:** 24
- **Rejected** (`rejected_fdr=true`): **13**
- **Not rejected** (`q_value` set, `rejected_fdr=false`): 11
- **Outside FDR families** (no q-value): 12

### Rejected at FDR (`rejected_fdr = true`)

- **`h1_vs_half_no_evidence_choice_without`** То же, что primary: доля «man» без evidence (входит в FDR по evidence) (H1), family `h1_strata_vs_half`: p=4.707e-18, q=1.412e-17, Δ=+0.500 — P(man) vs 0.5: evidence=no_evidence, without_abstain ⚠ **Данные ненадёжные** — min(k,n-k)=0<1; p_hat=1.000 насыщение
- **`h1_vs_half_evidence_man_choice_without`** Доля ответов «man» при тексте с поддержкой мужчин vs 50% (H1), family `h1_strata_vs_half`: p=3.477e-17, q=5.215e-17, Δ=+0.487 — P(man) vs 0.5: evidence=evidence_supports_man, without_abstain ⚡ *осторожно* — min(k,n-k)=1<10; p_hat=0.987 экстремальная доля
- **`h2_primary_stereo_vs_anti`** Сравнение доли стереотипных и антистереотипных ответов модели (A/B) в primary-срезе (H2), family `h2_stereo_vs_anti`: p=0.001173, q=0.002345, Δ=+0.253 — P(stereotype_consistent) vs P(anti_stereotype) on A/B choices
- **`h2_primary_stereo_man_vs_ann_baseline`** Сравнение доли ответов одновременно стереотипных и «мужских» у модели и такой же доли в аннотации (primary, choice) (H2), family `h2_stereo_vs_baseline`: p=5.676e-07, q=5.676e-06, Δ=+0.240 — P(stereo & мужской) vs доля stereo+мужской в аннотации (primary, uniform random baseline)
- **`h2_yesno_woman_stereo_vs_ann_baseline`** Сравнение доли стереотипных ответов модели в формате да/нет о женщине и доли стереотипных опций в аннотации для того же формата (H2), family `h2_stereo_vs_baseline`: p=0.0009564, q=0.004782, Δ=+0.173 — P(stereo choice) vs annotation baseline (no_evidence, yesno_woman, 2 options)
- **`h2_primary_stereo_vs_ann_baseline`** Сравнение доли стереотипных ответов модели при выборе и доли ответов в датасете, аннотированных как стереотипные (baseline при случайном выборе опции) (H2), family `h2_stereo_vs_baseline`: p=0.01087, q=0.03623, Δ=+0.127 — P(stereo choice) vs annotation baseline (no_evidence, choice, 2 options (without_abstain))
- **`h2_ev_man_stereo_vs_ann_baseline`** Сравнение доли стереотипных ответов модели при выборе из вопросов с поддержкой мужчин и доли ответов в датасете, аннотированных как стереотипные, при выборе из вопросов с поддержкой мужчин (H2), family `h2_stereo_vs_baseline`: p=0.01642, q=0.04105, Δ=+0.120 — P(stereo choice) vs annotation baseline (evidence_supports_man, choice, 2 options)
- **`h3_pair_no_vs_woman`** Сравнение доли «man» без evidence и при evidence в пользу женщин (H3), family `h3_evidence_posthoc`: p=9.755e-13, q=2.926e-12, Δ=+0.507 — P(man): no_evidence vs evidence_supports_woman ⚠ **Данные ненадёжные** — группа1 min(k,n-k)=0<1; группа1 p_hat=1.000 насыщение
- **`h3_pair_man_vs_woman`** Сравнение доли «man» при evidence за man и за woman (H3), family `h3_evidence_posthoc`: p=5.683e-12, q=8.525e-12, Δ=+0.493 — P(man): evidence_supports_man vs evidence_supports_woman ⚡ *осторожно* — группа1 min(k,n-k)=1<10; группа1 p_hat=0.987 экстремальная доля
- **`h6_two_prop_choice_woman_vs_yesno_man`** Сравнение P(woman|choice) и P(Да|yesno про мужчину) — перекрёстные форматы (H6), family `h6_format_pairs`: p=3.662e-28, q=1.465e-27, Δ=-0.893 — P(woman|choice) vs P(Yes|yesno_man), cross-axis ⚠ **Данные ненадёжные** — группа1 min(k,n-k)=0<1; группа1 p_hat=0.000 насыщение; группа2 min(k,n-k)=8<10
- **`h6_two_prop_choice_man_vs_yesno_woman`** Сравнение P(man|choice) и P(Да|yesno про женщину) — перекрёстные форматы (H6), family `h6_format_pairs`: p=1.076e-15, q=2.152e-15, Δ=+0.600 — P(man|choice) vs P(Yes|yesno_woman), cross-axis ⚠ **Данные ненадёжные** — группа1 min(k,n-k)=0<1; группа1 p_hat=1.000 насыщение
- **`h6_two_prop_choice_woman_vs_yesno_woman`** Сравнение P(woman|choice) и P(Да|yesno про женщину) на одной оси (woman) (H6), family `h6_format_pairs`: p=9.141e-10, q=1.219e-09, Δ=-0.400 — P(woman|choice) vs P(Yes|yesno_woman), same-axis ⚠ **Данные ненадёжные** — группа1 min(k,n-k)=0<1; группа1 p_hat=0.000 насыщение
- **`h6_two_prop_choice_man_vs_yesno_man`** Сравнение P(man|choice) и P(Да|yesno про мужчину) на одной оси (man) (H6), family `h6_format_pairs`: p=0.003649, q=0.003649, Δ=+0.107 — P(man|choice) vs P(Yes|yesno_man), same-axis ⚠ **Данные ненадёжные** — группа1 min(k,n-k)=0<1; группа1 p_hat=1.000 насыщение; группа2 min(k,n-k)=8<10

### Not rejected (`q_value` present, `rejected_fdr = false`)

- **`h1_vs_half_evidence_woman_choice_without`** Доля ответов «man» при тексте с поддержкой женщин vs 50% (H1), family `h1_strata_vs_half`: p=0.9081, q=0.9081, Δ=-0.007 — P(man) vs 0.5: evidence=evidence_supports_woman, without_abstain
- **`h2_with_abstain_stereo_vs_anti`** Сравнение доли стереотипных vs антистереотипных ответов среди выборов A/B (без учёта C) при наличии опции abstain (H2), family `h2_stereo_vs_anti`: p=1, q=1, Δ=+0.000 — P(stereo) vs P(anti) on A/B (with_abstain; C=neutral excluded) ⚠ **Данные ненадёжные** — группа1 n=6<10; группа1 min(k,n-k)=3<10; группа2 n=6<10; группа2 min(k,n-k)=3<10
- **`h2_yesno_man_stereo_vs_ann_baseline`** Сравнение доли стереотипных ответов модели в формате да/нет об мужчине и доли стереотипных опций в аннотации для того же формата (H2), family `h2_stereo_vs_baseline`: p=0.1385, q=0.277, Δ=+0.060 — P(stereo choice) vs annotation baseline (no_evidence, yesno_man, 2 options)
- **`h2_ev_woman_stereo_vs_ann_baseline`** Сравнение доли стереотипных ответов модели при выборе из вопросов с поддержкой женщин и доли ответов в датасете, аннотированных как стереотипные, при выборе из вопросов с поддержкой женщин (H2), family `h2_stereo_vs_baseline`: p=0.2352, q=0.3921, Δ=+0.040 — P(stereo choice) vs annotation baseline (evidence_supports_woman, choice, 2 options)
- **`h2_choice_3opt_stereo_vs_ann_baseline`** Сравнение доли стереотипных ответов при choice с опцией «Cannot determine» и доли стереотипных опций в аннотации (3 варианта ответа) (H2), family `h2_stereo_vs_baseline`: p=1, q=1, Δ=-0.196 — P(stereo choice) vs annotation baseline (no_evidence, choice, 3 options (with_abstain)) ⚠ **Данные ненадёжные** — min(k,n-k)=3<10; p_hat=0.040 экстремальная доля; k_stereo=3<10 (массовый abstain)
- **`h2_ev_man_3opt_stereo_vs_ann_baseline`** То же для choice с 3 опциями при evidence в пользу мужчин (H2), family `h2_stereo_vs_baseline`: p=0.9998, q=1, Δ=-0.178 — P(stereo choice) vs annotation baseline (evidence_supports_man, choice, 3 options) ⚠ **Данные ненадёжные** — min(k,n-k)=6<10; k_stereo=6<10 (массовый abstain)
- **`h2_ev_woman_3opt_stereo_vs_ann_baseline`** То же для choice с 3 опциями при evidence в пользу женщин (H2), family `h2_stereo_vs_baseline`: p=0.9994, q=1, Δ=-0.160 — P(stereo choice) vs annotation baseline (evidence_supports_woman, choice, 3 options) ⚠ **Данные ненадёжные** — min(k,n-k)=6<10; k_stereo=6<10 (массовый abstain)
- **`h2_primary_stereo_woman_vs_ann_baseline`** Сравнение доли ответов одновременно стереотипных и «женских» у модели и такой же доли в аннотации (primary, choice) (H2), family `h2_stereo_vs_baseline`: p=0.999, q=1, Δ=-0.113 — P(stereo & женский) vs доля stereo+женский в аннотации (primary, uniform random baseline) ⚠ **Данные ненадёжные** — min(k,n-k)=0<1; p_hat=0.000 насыщение
- **`h3_pair_no_vs_man`** Сравнение доли «man» без evidence и при evidence в пользу мужчин (H3), family `h3_evidence_posthoc`: p=0.3157, q=0.3157, Δ=+0.013 — P(man): no_evidence vs evidence_supports_man ⚠ **Данные ненадёжные** — группа1 min(k,n-k)=0<1; группа1 p_hat=1.000 насыщение; группа2 min(k,n-k)=1<10; группа2 p_hat=0.987 экстремальная доля
- **`h5_abstain_no_vs_ev_woman`** Сравнение доли abstain (C) без evidence и при evidence в пользу женщин (H5), family `h5_abstain_by_evidence`: p=0.1317, q=0.2633, Δ=+0.080 — P(abstain C): no_evidence vs evidence_supports_woman (with_abstain, choice) ⚡ *осторожно* — группа1 min(k,n-k)=6<10
- **`h5_abstain_no_vs_ev_man`** Сравнение доли abstain (C) без evidence и при evidence в пользу мужчин (H5), family `h5_abstain_by_evidence`: p=0.29, q=0.29, Δ=+0.053 — P(abstain C): no_evidence vs evidence_supports_man (with_abstain, choice) ⚡ *осторожно* — группа1 min(k,n-k)=6<10

## Post-hoc H — yes/no position control

_(Not H6; see `post_hoc_H.csv`)_

- `post_h_two_prop_yesno_man_vs_yesno_woman` Сравнение P(Да|yesno про мужчину) и P(Да|yesno про женщину) при прочих равных: P(Yes|yesno_man)=0.893 vs P(Yes|yesno_woman)=0.400, p=2.614e-10 ⚡ *осторожно* — группа1 min(k,n-k)=8<10
- `post_h_mcnemar_yesno_pair` Парная асимметрия: «Да» на yesno_man и «Нет» на парный yesno_woman чаще, чем наоборот. McNemar: b=37, c=0, p=3.252e-09 ⚡ *осторожно* — односторонняя дискордантность b=37, c=0

## Probing cross-reference

Compare with `results/<run>/probes/h11/` (log_odds decodable from HS).
