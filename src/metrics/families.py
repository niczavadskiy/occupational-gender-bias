"""FDR hypothesis families (pre-registered groupings of test_id)."""



FDR_FAMILIES: dict[str, list[str]] = {

    "h1_strata_vs_half": [

        "h1_vs_half_no_evidence_choice_without",

        "h1_vs_half_evidence_man_choice_without",

        "h1_vs_half_evidence_woman_choice_without",

    ],

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

}

