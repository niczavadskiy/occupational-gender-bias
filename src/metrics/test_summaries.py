"""Human-readable one-line descriptions for summary.md (by test_id)."""

from __future__ import annotations

TEST_HUMAN_SUMMARY: dict[str, str] = {
    # H1
    "h1_primary_vs_half": (
        "Сравнение доли ответов «man» при выборе A/B без evidence и 50%"
    ),
    "h1_primary_vs_half_family_t": (
        "Средняя по семействам сценариев доля «man» vs 50% (family-level t-test)"
    ),
    "h1_vs_half_no_evidence_choice_without": (
        "То же, что primary: доля «man» без evidence (входит в FDR по evidence)"
    ),
    "h1_vs_half_evidence_man_choice_without": (
        "Доля ответов «man» при тексте с поддержкой мужчин vs 50%"
    ),
    "h1_vs_half_evidence_woman_choice_without": (
        "Доля ответов «man» при тексте с поддержкой женщин vs 50%"
    ),
    # H2 — stereo vs annotation baseline
    "h2_primary_stereo_vs_ann_baseline": (
        "Сравнение доли стереотипных ответов модели при выборе и доли ответов "
        "в датасете, аннотированных как стереотипные (baseline при случайном выборе опции)"
    ),
    "h2_ev_man_stereo_vs_ann_baseline": (
        "Сравнение доли стереотипных ответов модели при выборе из вопросов с поддержкой "
        "мужчин и доли ответов в датасете, аннотированных как стереотипные, "
        "при выборе из вопросов с поддержкой мужчин"
    ),
    "h2_ev_woman_stereo_vs_ann_baseline": (
        "Сравнение доли стереотипных ответов модели при выборе из вопросов с поддержкой "
        "женщин и доли ответов в датасете, аннотированных как стереотипные, "
        "при выборе из вопросов с поддержкой женщин"
    ),
    "h2_yesno_man_stereo_vs_ann_baseline": (
        "Сравнение доли стереотипных ответов модели в формате да/нет об мужчине "
        "и доли стереотипных опций в аннотации для того же формата"
    ),
    "h2_yesno_woman_stereo_vs_ann_baseline": (
        "Сравнение доли стереотипных ответов модели в формате да/нет о женщине "
        "и доли стереотипных опций в аннотации для того же формата"
    ),
    "h2_choice_3opt_stereo_vs_ann_baseline": (
        "Сравнение доли стереотипных ответов при choice с опцией «Cannot determine» "
        "и доли стереотипных опций в аннотации (3 варианта ответа)"
    ),
    "h2_ev_man_3opt_stereo_vs_ann_baseline": (
        "То же для choice с 3 опциями при evidence в пользу мужчин"
    ),
    "h2_ev_woman_3opt_stereo_vs_ann_baseline": (
        "То же для choice с 3 опциями при evidence в пользу женщин"
    ),
    "h2_primary_stereo_man_vs_ann_baseline": (
        "Сравнение доли ответов одновременно стереотипных и «мужских» у модели "
        "и такой же доли в аннотации (primary, choice)"
    ),
    "h2_primary_stereo_woman_vs_ann_baseline": (
        "Сравнение доли ответов одновременно стереотипных и «женских» у модели "
        "и такой же доли в аннотации (primary, choice)"
    ),
    "h2_primary_stereo_vs_anti": (
        "Сравнение доли стереотипных и антистереотипных ответов модели (A/B) "
        "в primary-срезе"
    ),
    "h2_with_abstain_stereo_vs_anti": (
        "Сравнение доли стереотипных vs антистереотипных ответов среди выборов A/B "
        "(без учёта C) при наличии опции abstain"
    ),
    "h2_with_abstain_chi2_stereo_anti_neutral": (
        "Распределение ответов модели: стереотип / антистереотип / нейтрально "
        "(включая abstain) при 3 опциях"
    ),
    "h2_frac_families_pick_stereo": (
        "Описательно: доля семейств сценариев, где чаще выбирается стереотипный ответ"
    ),
    # H3
    "h3_chi2_evidence_x_outcome": (
        "Связь типа evidence (нет / за man / за woman) с выбором man vs woman (без abstain)"
    ),
    "h3_chi2_evidence_x_outcome_with_abstain": (
        "Связь evidence с исходом man / woman / abstain (C) при 3 опциях"
    ),
    "h3_pair_no_vs_man": (
        "Сравнение доли «man» без evidence и при evidence в пользу мужчин"
    ),
    "h3_pair_no_vs_woman": (
        "Сравнение доли «man» без evidence и при evidence в пользу женщин"
    ),
    "h3_pair_man_vs_woman": (
        "Сравнение доли «man» при evidence за man и за woman"
    ),
    "h3_paired_family_no_vs_ev_woman": (
        "Парно по семействам: сдвиг доли «man» от no_evidence к evidence за woman vs 0"
    ),
    "h3_ev_woman_rate_man_vs_half": (
        "При evidence за женщин: доля выбора man vs 50% (проверка переключения bias)"
    ),
    # H4–H9 skipped
    "h4_pending_answerability": "Пропущено: нужны метки answerability",
    "h7_pending_answerability": "Пропущено: нужны метки answerability",
    "h8_pending_answerability": "Пропущено: нужны метки answerability",
    "h9_pending_answerability": "Пропущено: нужны метки answerability",
    # H5
    "h5_abstain_no_vs_ev_man": (
        "Сравнение доли abstain (C) без evidence и при evidence в пользу мужчин"
    ),
    "h5_abstain_no_vs_ev_woman": (
        "Сравнение доли abstain (C) без evidence и при evidence в пользу женщин"
    ),
    # H6
    "h6_two_prop_choice_man_vs_yesno_man": (
        "Сравнение P(man|choice) и P(Да|yesno про мужчину) на одной оси (man)"
    ),
    "h6_two_prop_choice_woman_vs_yesno_woman": (
        "Сравнение P(woman|choice) и P(Да|yesno про женщину) на одной оси (woman)"
    ),
    "h6_two_prop_choice_man_vs_yesno_woman": (
        "Сравнение P(man|choice) и P(Да|yesno про женщину) — перекрёстные форматы"
    ),
    "h6_two_prop_choice_woman_vs_yesno_man": (
        "Сравнение P(woman|choice) и P(Да|yesno про мужчину) — перекрёстные форматы"
    ),
    # Post-hoc
    "post_h_two_prop_yesno_man_vs_yesno_woman": (
        "Сравнение P(Да|yesno про мужчину) и P(Да|yesno про женщину) при прочих равных"
    ),
    "post_h_mcnemar_yesno_pair": (
        "Парная асимметрия: «Да» на yesno_man и «Нет» на парный yesno_woman чаще, чем наоборот"
    ),
}


def human_summary(test_id: str) -> str:
    """One-line Russian blurb; empty string if unknown."""
    return TEST_HUMAN_SUMMARY.get(test_id, "")
