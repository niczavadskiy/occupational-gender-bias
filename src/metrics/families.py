"""FDR hypothesis families (pre-registered groupings of test_id)."""

# Temporarily disable H2 calculation and summary output (set True to re-enable).
H2_ENABLED = False



FDR_FAMILIES: dict[str, list[str]] = {
    "h2_stereo_vs_baseline": [
        "h2_primary_stereo_vs_ann_baseline",
        "h2_ev_man_stereo_vs_ann_baseline",
        "h2_ev_woman_stereo_vs_ann_baseline",
        "h2_yesno_man_stereo_vs_ann_baseline",
        "h2_yesno_woman_stereo_vs_ann_baseline",
        "h2_choice_3opt_stereo_vs_ann_baseline",
        "h2_ev_man_3opt_stereo_vs_ann_baseline",
        "h2_ev_woman_3opt_stereo_vs_ann_baseline",
        "h2_primary_stereo_man_vs_ann_baseline",
        "h2_primary_stereo_woman_vs_ann_baseline",
    ],
    "h2_stereo_vs_anti": [
        "h2_primary_stereo_vs_anti",
        "h2_with_abstain_stereo_vs_anti",
    ],

    "h3_evidence_posthoc": [

        "h3_pair_no_vs_man",

        "h3_pair_no_vs_woman",

        "h3_pair_man_vs_woman",

        "h3_pair_woman_no_vs_ev_man",

        "h3_pair_woman_no_vs_ev_woman",

        "h3_pair_woman_ev_man_vs_ev_woman",

    ],

    "h5_abstain_by_evidence": [

        "h5_abstain_no_vs_ev_man",

        "h5_abstain_no_vs_ev_woman",

    ],

    "h6_format_pairs": [

        "h6_two_prop_choice_man_vs_yesno_man",

        "h6_two_prop_choice_woman_vs_yesno_woman",

        "h6_two_prop_choice_man_vs_yesno_woman",

        "h6_two_prop_choice_woman_vs_yesno_man",

    ],

    "h4_abstain_answerability": [
        "h4_primary_yes_with_vs_without_abstain",
    ],

}


H4_QUESTION_FORMATS = ("choice", "yesno_man", "yesno_woman")
ABSTAIN_SUFFIXES = ("_without", "_with")


def _h4_test_suffix(question_format: str) -> str:
    return "" if question_format == "choice" else f"_{question_format}"


def h1_fdr_families(df) -> dict[str, list[str]]:
    """Dynamic FDR families for H1 position/evidence/format/abstain strata."""
    out: dict[str, list[str]] = {}
    if "position_variant" in df.columns:
        ids = [
            f"h1_vs_half_pos_{p}"
            for p in sorted(df["position_variant"].astype(str).unique())
        ]
        if ids:
            out["h1_strata_position"] = ids
    out["h1_strata_evidence_gender_pair"] = [
        f"h1_ev_{slug}_man_vs_woman{opt}"
        for slug in ("no_evidence", "ev_man", "ev_woman")
        for opt in ("", "_2opt", "_3opt")
    ]
    qf_vals = set(df["question_format"].astype(str)) if "question_format" in df.columns else set()
    fmt_ids = [f"h1_vs_half_fmt_{qf}" for qf in H4_QUESTION_FORMATS if qf in qf_vals]
    if fmt_ids:
        out["h1_strata_format"] = fmt_ids
    out["h1_strata_abstain"] = [
        "h1_vs_half_abst_without_abstain",
        "h1_vs_half_abst_with_abstain",
    ]
    return out


def h7_fdr_families() -> dict[str, list[str]]:
    return {
        "h7_evidence_answerability": [
            "h7_primary_yes_evidence_vs_no",
            "h7_yes_ev_man_vs_no",
            "h7_yes_ev_woman_vs_no",
        ],
        "h7_evidence_answerability_lower": [
            "h7_primary_yes_lower_evidence_vs_no",
            "h7_yes_lower_ev_man_vs_no",
            "h7_yes_lower_ev_woman_vs_no",
        ],
    }


def h8_fdr_families() -> dict[str, list[str]]:
    return {
        "h8_self_abstain_link": [
            "h8_primary_abstain_self_no_vs_yes",
            "h8_abstain_self_no_vs_yes_no_evidence",
            "h8_abstain_self_no_vs_yes_ev_man",
            "h8_abstain_self_no_vs_yes_ev_woman",
        ]
    }


def h9_fdr_families() -> dict[str, list[str]]:
    return {"h9_gender_evidence_answerability": ["h9_primary_yes_ev_man_vs_ev_woman"]}


def h4_fdr_families(formats: tuple[str, ...]) -> dict[str, list[str]]:
    """FDR families for H4 primary two-prop only (McNemar outside FDR)."""
    out: dict[str, list[str]] = {}
    for qf in formats:
        suffix = _h4_test_suffix(qf)
        base = "h4_abstain_answerability" if qf == "choice" else f"h4_abstain_answerability_{qf}"
        for higher, id_mid in ((True, ""), (False, "_lower")):
            family = base if higher else f"{base}_lower"
            out[family] = [f"h4_primary_yes{id_mid}_with_vs_without_abstain{suffix}"]
    return out


def merge_h4_fdr_families(base: dict[str, list[str]], formats: tuple[str, ...]) -> dict[str, list[str]]:
    merged = dict(base)
    for key in list(merged):
        if key.startswith("h4_abstain_answerability"):
            del merged[key]
    merged.update(h4_fdr_families(formats))
    return merged


def merge_answerability_fdr_families(
    base: dict[str, list[str]],
    formats: tuple[str, ...],
    hypotheses: list[str],
) -> dict[str, list[str]]:
    merged = dict(base)
    if "H4" in hypotheses and formats:
        merged = merge_h4_fdr_families(merged, formats)
    if "H7" in hypotheses:
        for key in list(merged):
            if key.startswith("h7_evidence_answerability"):
                del merged[key]
        merged.update(h7_fdr_families())
    if "H8" in hypotheses:
        for key in list(merged):
            if key.startswith("h8_self_abstain_link"):
                del merged[key]
        merged.update(h8_fdr_families())
    if "H9" in hypotheses:
        for key in list(merged):
            if key.startswith("h9_gender_evidence_answerability"):
                del merged[key]
        merged.update(h9_fdr_families())
    return merged
