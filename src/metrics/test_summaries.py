"""Human-readable one-line descriptions for summary.md (by test_id)."""

from __future__ import annotations

TEST_HUMAN_SUMMARY: dict[str, str] = {
    # H1
    "h1_primary_man_vs_woman": (
        "Главный тест H1: P(pro-man) vs P(pro-woman) на всех main-вопросах "
        "(все форматы, evidence, abstain)"
    ),
    "h1_primary_vs_half_family_t": (
        "Средняя по семействам доля pro-man (все gender-форматы, semantic labels) vs 50%"
    ),
    "h1_ev_no_evidence_man_vs_woman": "P(man) vs P(woman) без evidence (2+3 опции)",
    "h1_ev_ev_man_man_vs_woman": "P(man) vs P(woman) при evidence за мужчин (2+3 опции)",
    "h1_ev_ev_woman_man_vs_woman": "P(man) vs P(woman) при evidence за женщин (2+3 опции)",
    "h1_ev_no_evidence_man_vs_woman_2opt": "P(man) vs P(woman) без evidence (2 опции A/B)",
    "h1_ev_ev_man_man_vs_woman_2opt": "P(man) vs P(woman) при ev_man (2 опции A/B)",
    "h1_ev_ev_woman_man_vs_woman_2opt": "P(man) vs P(woman) при ev_woman (2 опции A/B)",
    "h1_ev_no_evidence_man_vs_woman_3opt": "P(man) vs P(woman) без evidence (3 опции A/B/C)",
    "h1_ev_ev_man_man_vs_woman_3opt": "P(man) vs P(woman) при ev_man (3 опции A/B/C)",
    "h1_ev_ev_woman_man_vs_woman_3opt": "P(man) vs P(woman) при ev_woman (3 опции A/B/C)",
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
        "Связь типа evidence (нет / за man / за woman) с выбором man vs woman "
        "(все форматы, все abstain-варианты)"
    ),
    "h3_chi2_evidence_x_outcome_with_abstain": (
        "Связь evidence с исходом man / woman / abstain (C) при 3 опциях "
        "(with_abstain, все форматы)"
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
    "h3_pair_woman_no_vs_ev_man": (
        "Сравнение доли «woman» без evidence и при evidence в пользу мужчин"
    ),
    "h3_pair_woman_no_vs_ev_woman": (
        "Сравнение доли «woman» без evidence и при evidence в пользу женщин"
    ),
    "h3_pair_woman_ev_man_vs_ev_woman": (
        "Сравнение доли «woman» при evidence за man и за woman"
    ),
    "h3_paired_family_no_vs_ev_woman": (
        "Парно по семействам: сдвиг доли «man» от no_evidence к evidence за woman vs 0"
    ),
    # H4
    "h4_primary_yes_with_vs_without_abstain": (
        "P(Yes на self-Q answerability) выше, когда в основном вопросе показана опция "
        "«Cannot determine» (with_abstain vs without), все форматы, парные сценарии"
    ),
    "h4_primary_yes_lower_with_vs_without_abstain": (
        "P(Yes на self-Q answerability) ниже, когда в основном вопросе показана опция "
        "«Cannot determine» (with_abstain vs without), все форматы, парные сценарии"
    ),
    "h4_mcnemar_yes_with_vs_without": (
        "Парный McNemar (H4↑): для каждого сценария сравнивается self-Q Yes при with_abstain "
        "и without (все форматы); см. таблицу a/b/c/d выше; H1: b>c"
    ),
    "h4_mcnemar_yes_lower_with_vs_without": (
        "Парный McNemar (H4↓): тот же срез; H1: c>b (чаще Yes без abstain)"
    ),
    "h4_pending_answerability": "Пропущено: нужен прогон 5400 с task=answerability",
    # H7
    "h7_primary_yes_evidence_vs_no": (
        "P(Yes на self-Q) выше при evidence (man+woman), чем при no_evidence (H7↑)"
    ),
    "h7_primary_yes_lower_evidence_vs_no": (
        "P(Yes на self-Q) ниже при evidence (man+woman), чем при no_evidence (H7↓)"
    ),
    "h7_yes_ev_man_vs_no": "P(Yes) выше при evidence за man vs no_evidence (H7↑)",
    "h7_yes_lower_ev_man_vs_no": "P(Yes) ниже при evidence за man vs no_evidence (H7↓)",
    "h7_yes_ev_woman_vs_no": "P(Yes) выше при evidence за woman vs no_evidence (H7↑)",
    "h7_yes_lower_ev_woman_vs_no": "P(Yes) ниже при evidence за woman vs no_evidence (H7↓)",
    "h7_mcnemar_yes_ev_man_vs_no": (
        "Парный McNemar (H7↑): self-Q Yes при ev_man vs no_evidence на тех же сценариях"
    ),
    "h7_mcnemar_yes_lower_ev_man_vs_no": (
        "Парный McNemar (H7↓): self-Q Yes при ev_man vs no_evidence; H1: c>b (чаще Yes без evidence)"
    ),
    "h7_mcnemar_yes_ev_woman_vs_no": (
        "Парный McNemar (H7↑): self-Q Yes при ev_woman vs no_evidence на тех же сценариях"
    ),
    "h7_mcnemar_yes_lower_ev_woman_vs_no": (
        "Парный McNemar (H7↓): self-Q Yes при ev_woman vs no_evidence; H1: c>b (чаще Yes без evidence)"
    ),
    "h7_pending_answerability": "Пропущено: нужен прогон 5400 с task=answerability",
    # H8
    "h8_primary_abstain_self_no_vs_yes": (
        "P(C на main) выше, когда self-Q answerability = No (unanswerable)"
    ),
    "h8_mcnemar_self_no_abstain": (
        "Парный McNemar H8: self-Q ↔ main C на тех же сценариях (with_abstain)"
    ),
    "h8_abstain_self_no_vs_yes_no_evidence": "H8 на срезе no_evidence",
    "h8_abstain_self_no_vs_yes_ev_man": "H8 при evidence за man",
    "h8_abstain_self_no_vs_yes_ev_woman": "H8 при evidence за woman",
    "h8_pending_answerability": "Пропущено: нет пар answerability↔main с has_abstain",
    # H9
    "h9_primary_yes_ev_man_vs_ev_woman": (
        "P(Yes на self-Q) при evidence за man vs evidence за woman"
    ),
    "h9_mcnemar_ev_man_vs_ev_woman": (
        "Парный McNemar H9: Yes при ev_man vs ev_woman на тех же сценариях"
    ),
    "h9_pending_answerability": "Пропущено: нужен прогон 5400 с task=answerability",
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
        "Парный McNemar: yesno_man vs yesno_woman на тех же сценариях "
        "(a=оба Yes, b=Yes man/No woman, c=No man/Yes woman, d=оба No)"
    ),
}


_H4_SUFFIX_FORMATS = ("yesno_man", "yesno_woman")


def _strip_answerability_suffixes(test_id: str) -> tuple[str, str | None, str | None]:
    """Return (base_id, question_format, abstain_suffix _with/_without or None)."""
    tid = test_id
    abstain: str | None = None
    for ab in ("_with", "_without"):
        if tid.endswith(ab):
            tid = tid[: -len(ab)]
            abstain = ab
            break
    qf: str | None = None
    for fmt in _H4_SUFFIX_FORMATS:
        suffix = f"_{fmt}"
        if tid.endswith(suffix):
            tid = tid[: -len(suffix)]
            qf = fmt
            break
    return tid, qf, abstain


def human_summary(test_id: str) -> str:
    """One-line Russian blurb; empty string if unknown."""
    if test_id in TEST_HUMAN_SUMMARY:
        return TEST_HUMAN_SUMMARY[test_id]
    base, qf, abstain = _strip_answerability_suffixes(test_id)
    base_text = TEST_HUMAN_SUMMARY.get(base, "")
    if not base_text:
        return ""
    text = base_text.replace("choice", qf) if qf else base_text
    if abstain == "_with":
        text += ", with_abstain"
    elif abstain == "_without":
        text += ", without_abstain"
    return text
