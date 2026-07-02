"""Write CSV / JSON / markdown summary."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.metrics.hypotheses import H2_STEREO_VS_BASELINE_SLICES
from src.metrics.families import H2_ENABLED
from src.metrics.test_summaries import human_summary
from src.metrics.reliability import (
    RELIABILITY_UNRELIABLE,
    reliability_badge,
    reliability_reason_suffix,
)
from src.metrics.stats_tests import TestResult

# Summary: 3 строки (без дубля no_evidence strata и без family_t).
H1_SUMMARY_TESTS: tuple[tuple[str, str, str], ...] = (
    ("h1_primary_man_vs_woman", "pair", "all formats, evidence, abstain"),
    ("h1_ev_no_evidence_man_vs_woman", "pair", "no_evidence, 2+3 options"),
    ("h1_ev_ev_man_man_vs_woman", "pair", "ev_man, 2+3 options"),
    ("h1_ev_ev_woman_man_vs_woman", "pair", "ev_woman, 2+3 options"),
    ("h1_ev_no_evidence_man_vs_woman_2opt", "pair", "no_evidence, 2 options"),
    ("h1_ev_ev_man_man_vs_woman_2opt", "pair", "ev_man, 2 options"),
    ("h1_ev_ev_woman_man_vs_woman_2opt", "pair", "ev_woman, 2 options"),
)


def tests_to_dataframe(results: list[TestResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.DataFrame([r.to_dict() for r in results])


def _json_number(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        if math.isnan(x):
            return None
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    return x


def test_result_to_meta(r: TestResult) -> dict[str, Any]:
    """Compact test record for meta.json (no wide extra_* flattening)."""
    row: dict[str, Any] = {
        "hypothesis_id": r.hypothesis_id,
        "test_id": r.test_id,
        "description": r.description,
        "level": r.level,
        "family_name": r.family_name,
        "n1": _json_number(r.n1),
        "k1": _json_number(r.k1),
        "n2": _json_number(r.n2),
        "k2": _json_number(r.k2),
        "p1": _json_number(r.p1),
        "p2": _json_number(r.p2),
        "effect": _json_number(r.effect),
        "ci_low": _json_number(r.ci_low),
        "ci_high": _json_number(r.ci_high),
        "statistic": _json_number(r.statistic),
        "p_raw": _json_number(r.p_raw),
        "q_value": _json_number(r.q_value),
        "rejected_fdr": r.rejected_fdr,
    }
    if r.extra:
        row["extra"] = r.extra
    return row


def tests_to_meta_list(results: list[TestResult]) -> list[dict[str, Any]]:
    return [test_result_to_meta(r) for r in results]


def _fmt_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    return f"{p:.4g}"


def _human(tid: str) -> str:
    s = human_summary(tid)
    return f" {s}" if s else ""


def _mcnemar_cells(r: TestResult) -> tuple[int, int, int, int, str, str, str, str]:
    """Return a,b,c,d and Russian labels for the 2×2 McNemar table."""
    ex = r.extra
    if ex.get("discordant_c_self_no_and_c") is not None:
        a = int(ex.get("mcnemar_a_self_yes_and_c") or 0)
        b = int(ex["discordant_b_self_yes_no_c"])
        c = int(ex["discordant_c_self_no_and_c"])
        d = int(ex.get("mcnemar_d_self_no_no_c") or 0)
        row = "self_No"
        col = "main_C"
        b_lbl = "self Yes / не C"
        c_lbl = "self No / C"
    elif ex.get("discordant_b_yes_with_only") is not None:
        a = int(ex.get("mcnemar_a_both_yes") or 0)
        b = int(ex["discordant_b_yes_with_only"])
        c = int(ex["discordant_c_yes_without_only"])
        d = int(ex.get("mcnemar_d_both_no") or 0)
        row = "with_abstain"
        col = "without_abstain"
        b_lbl = "Yes при with / No при without"
        c_lbl = "No при with / Yes при without"
    elif ex.get("discordant_b_ev_yes_no_no") is not None:
        a = int(ex.get("mcnemar_a_both_yes") or 0)
        b = int(ex["discordant_b_ev_yes_no_no"])
        c = int(ex["discordant_c_ev_no_no_yes"])
        d = int(ex.get("mcnemar_d_both_no") or 0)
        row = ex.get("mcnemar_row_axis", "evidence")
        col = ex.get("mcnemar_col_axis", "no_evidence")
        ev_short = "ev_man" if row == "evidence_supports_man" else (
            "ev_woman" if row == "evidence_supports_woman" else str(row)
        )
        b_lbl = f"Yes при {ev_short} / No при no_evidence"
        c_lbl = f"No при {ev_short} / Yes при no_evidence"
    else:
        a = int(ex.get("mcnemar_a_both_yes") or 0)
        b = int(ex.get("discordant_b_man_yes_woman_no") or r.k1 or 0)
        c = int(ex.get("discordant_c_man_no_woman_yes") or r.k2 or 0)
        d = int(ex.get("mcnemar_d_both_no") or 0)
        row = ex.get("mcnemar_row_axis", "yesno_man")
        col = ex.get("mcnemar_col_axis", "yesno_woman")
        if row == "ev_man":
            b_lbl = "Yes при ev_man / No при ev_woman"
            c_lbl = "No при ev_man / Yes при ev_woman"
        else:
            b_lbl = "Yes на man / No на woman"
            c_lbl = "No на man / Yes на woman"
    return a, b, c, d, row, col, b_lbl, c_lbl


def _format_mcnemar_detail(r: TestResult) -> str:
    a, b, c, d, row, col, b_lbl, c_lbl = _mcnemar_cells(r)
    n_pairs = r.extra.get("n_pairs", a + b + c + d)
    h1 = r.extra.get("mcnemar_h1", "b>c")
    return (
        f"a={a} (оба Yes), b={b} ({b_lbl}), c={c} ({c_lbl}), d={d} (оба No), "
        f"n_pairs={n_pairs}; таблица 2×2: строка={row}, столбец={col}; "
        f"H1 односторонний: {h1}, p={_fmt_p(r.p_raw)}"
    )


MCNEMAR_H7_LEGEND = (
    "**McNemar H7:** один сценарий (base_id × format × abstain × position × context), "
    "сравнивается self-Q Yes при **evidence** vs **no_evidence**. "
    "Таблица 2×2: **строка** = evidence, **столбец** = no_evidence.\n\n"
    "|  | no_evidence=No | no_evidence=Yes |\n"
    "|---|---:|---:|\n"
    "| **evidence=No** | **d** (оба No) | **c** (No→Yes) |\n"
    "| **evidence=Yes** | **b** (Yes→No) | **a** (оба Yes) |\n\n"
    "H7↑: **b > c** (чаще Yes с evidence); H7↓: **c > b** (чаще Yes без evidence)."
)


MCNEMAR_H4_LEGEND = (
    "**McNemar (парные сценарии):** один и тот же сценарий оценивается дважды — "
    "с опцией «Cannot determine» в основном вопросе (with_abstain) и без (without). "
    "Таблица 2×2: **строка** = ответ self-Q при with_abstain, **столбец** = при without.\n\n"
    "|  | without=No | without=Yes |\n"
    "|---|---:|---:|\n"
    "| **with=No** | **d** (оба No) | **c** (дискордант: No→Yes) |\n"
    "| **with=Yes** | **b** (дискордант: Yes→No) | **a** (оба Yes) |\n\n"
    "**a**, **d** — конкордантные пары; **b**, **c** — дискордантные. "
    "H4↑: **b > c** (чаще Yes при with_abstain); H4↓: **c > b** (чаще Yes при without)."
)


def _fdr_rejection_mark(r: TestResult) -> str:
    if r.extra.get("reliability") == RELIABILITY_UNRELIABLE:
        return ""
    return " **✓**" if r.rejected_fdr else ""


def _format_test_fdr_line(r: TestResult) -> str:
    parts: list[str] = []
    if r.p_raw is not None and not (isinstance(r.p_raw, float) and math.isnan(r.p_raw)):
        parts.append(f"p={r.p_raw:.4g}")
    if r.q_value is not None and not (isinstance(r.q_value, float) and math.isnan(r.q_value)):
        parts.append(f"q={r.q_value:.4g}")
    if r.effect is not None and not (isinstance(r.effect, float) and math.isnan(r.effect)):
        parts.append(f"Δ={r.effect:+.3f}")
    stat = ", ".join(parts) if parts else "no p/q"
    fam = f", family `{r.family_name}`" if r.family_name else ""
    rel = reliability_badge(r.extra.get("reliability")) + reliability_reason_suffix(r)
    return (
        f"- **`{r.test_id}`**{_human(r.test_id)} ({r.hypothesis_id}){fam}: "
        f"{stat} — {r.description}{rel}"
    )


def _fdr_summary_sections(results: list[TestResult], fdr: float) -> list[str]:
    with_q = [r for r in results if r.q_value is not None]
    rejected = [r for r in results if r.rejected_fdr is True]
    not_rejected = [r for r in with_q if r.rejected_fdr is False]
    outside = len(results) - len(with_q)

    lines = [
        "",
        "## FDR correction (Benjamini–Hochberg)",
        "",
        f"- **Target FDR:** {fdr}",
        f"- **Tests with q-value:** {len(with_q)}",
        f"- **Rejected** (`rejected_fdr=true`): **{len(rejected)}**",
        f"- **Not rejected** (`q_value` set, `rejected_fdr=false`): {len(not_rejected)}",
        f"- **Outside FDR families** (no q-value): {outside}",
        "",
        "### Rejected at FDR (`rejected_fdr = true`)",
        "",
    ]
    if rejected:
        for r in sorted(rejected, key=lambda x: (x.family_name or "", x.q_value or 1.0)):
            lines.append(_format_test_fdr_line(r))
    else:
        lines.append("- _(none)_")

    lines.extend(
        [
            "",
            "### Not rejected (`q_value` present, `rejected_fdr = false`)",
            "",
        ]
    )
    if not_rejected:
        for r in sorted(not_rejected, key=lambda x: (x.family_name or "", x.q_value or 1.0)):
            lines.append(_format_test_fdr_line(r))
    else:
        lines.append("- _(none)_")

    return lines


def write_summary_md(
    path: Path,
    *,
    run_dir: Path,
    results: list[TestResult],
    post_hoc_results: list[TestResult] | None = None,
    rates: pd.DataFrame,
    level: str,
    fdr: float,
) -> None:
    def _pick(tid: str) -> TestResult | None:
        for r in results:
            if r.test_id == tid:
                return r
        return None

    def _rel(r: TestResult) -> str:
        return reliability_badge(r.extra.get("reliability")) + reliability_reason_suffix(r)

    def _q_rej(r: TestResult) -> str:
        q = f", q={r.q_value:.4g}" if r.q_value is not None else ""
        rej = _fdr_rejection_mark(r)
        return q + rej

    lines = [
        "# Behavioral metrics summary",
        "",
        f"- **Run:** `{run_dir.name}`",
        f"- **Level:** `{level}`",
        f"- **FDR target:** {fdr}",
        f"- **Generated:** {datetime.now(timezone.utc).isoformat()}",
        "- **Надёжность:** ⚠ **Данные ненадёжные** — не интерпретировать p/FDR; ⚡ *осторожно* — мало событий или насыщение",
        "",
        "## H1 — gender axis preference",
        "",
        "P(man) vs P(woman) на всех вопросах; strata по evidence и format.",
        "",
    ]
    for tid, axis, label in H1_SUMMARY_TESTS:
        r = _pick(tid)
        if not r:
            continue
        if axis == "pair":
            lines.append(
                f"- `{tid}`{_human(tid)} ({label}): P(man)={r.p1:.3f} vs P(woman)={r.p2:.3f} "
                f"(k={int(r.k1)}/{int(r.n1)}, k_w={int(r.k2)}), p={_fmt_p(r.p_raw)}{_q_rej(r)}{_rel(r)}"
            )
        else:
            lines.append(
                f"- `{tid}`{_human(tid)} ({label}): P({axis})={r.p1:.3f} (k={int(r.k1)}/{int(r.n1)}), "
                f"p vs 0.5={_fmt_p(r.p_raw)}{_q_rej(r)}{_rel(r)}"
            )

    if H2_ENABLED:
        lines.extend(
            [
                "",
                "## H2 — stereotype (annotator labels)",
                "",
                "### P(stereo choice) vs annotation baseline (uniform random among labeled options)",
                "",
            ]
        )
        for tid, _filters, slice_label in H2_STEREO_VS_BASELINE_SLICES:
            r = _pick(tid)
            if r:
                p0 = r.p2 if r.p2 is not None else r.extra.get("null_p_stereo_mean")
                p0s = f"{p0:.3f}" if p0 is not None else "—"
                lines.append(
                    f"- `{tid}`{_human(tid)} ({slice_label}): P_obs={r.p1:.3f} vs p₀={p0s}, "
                    f"p={_fmt_p(r.p_raw)}{_q_rej(r)}{_rel(r)}"
                )

        lines.extend(
            [
                "",
                "### P(stereo & gender) vs annotation baseline (primary)",
                "",
            ]
        )
        for tid, label in (
            ("h2_primary_stereo_man_vs_ann_baseline", "stereo + мужской"),
            ("h2_primary_stereo_woman_vs_ann_baseline", "stereo + женский"),
        ):
            r = _pick(tid)
            if r:
                p0 = r.p2 if r.p2 is not None else r.extra.get(
                    f"null_p_stereo_{r.extra.get('gender', '')}_mean"
                )
                p0s = f"{p0:.3f}" if p0 is not None else "—"
                lines.append(
                    f"- `{tid}`{_human(tid)} ({label}): P_obs={r.p1:.3f} vs p₀={p0s}, "
                    f"p={_fmt_p(r.p_raw)}{_q_rej(r)}{_rel(r)}"
                )

        lines.extend(["", "### Stereo vs anti (and other)", ""])
        for tid in (
            "h2_primary_stereo_vs_anti",
            "h2_with_abstain_stereo_vs_anti",
            "h2_with_abstain_chi2_stereo_anti_neutral",
        ):
            r = _pick(tid)
            if r:
                p1s = f"{r.p1:.3f}" if r.p1 is not None else "—"
                p2s = f"{r.p2:.3f}" if r.p2 is not None else "—"
                stat = f", stat={r.statistic:.3f}" if r.statistic is not None else ""
                lines.append(
                    f"- `{tid}`{_human(tid)}: p1={p1s}, p2={p2s}{stat}, "
                    f"p={_fmt_p(r.p_raw)}{_q_rej(r)} — {r.description}{_rel(r)}"
                )

    lines.extend(["", "## H3 — evidence vs ambiguous context", ""])
    h3c = _pick("h3_chi2_evidence_x_outcome")
    if h3c:
        lines.append(
            f"- `{h3c.test_id}`{_human(h3c.test_id)} (all formats, man/woman): "
            f"stat={h3c.statistic:.3f}, p={_fmt_p(h3c.p_raw)}{_rel(h3c)}"
        )
    h3cab = _pick("h3_chi2_evidence_x_outcome_with_abstain")
    if h3cab:
        lines.append(
            f"- `{h3cab.test_id}`{_human(h3cab.test_id)} (with_abstain, man/woman/C): "
            f"stat={h3cab.statistic:.3f}, p={_fmt_p(h3cab.p_raw)}{_rel(h3cab)}"
        )
    for tid in (
        "h3_pair_no_vs_man",
        "h3_pair_no_vs_woman",
        "h3_pair_man_vs_woman",
        "h3_pair_woman_no_vs_ev_man",
        "h3_pair_woman_no_vs_ev_woman",
        "h3_pair_woman_ev_man_vs_ev_woman",
    ):
        r = _pick(tid)
        if r:
            lines.append(
                f"- `{tid}`{_human(tid)}: p1={r.p1:.3f}, p2={r.p2:.3f}, Δ={r.effect:+.3f}, "
                f"p={_fmt_p(r.p_raw)}{_q_rej(r)}{_rel(r)}"
            )

    h3p = _pick("h3_paired_family_no_vs_ev_woman")
    if h3p:
        tid = h3p.test_id
        lines.append(
            f"- `{tid}`{_human(tid)}: paired family Δ={h3p.effect:+.3f}, "
            f"p={_fmt_p(h3p.p_raw)}{_rel(h3p)}"
        )

    lines.extend(
        [
            "",
            "## H4 — answerability ↑ with abstain option",
            "",
            "Self-Q (`task=answerability`, all positions): P(Yes) when main options include C vs A/B only.",
            "",
        ]
    )
    h4_any = False
    h4_results = sorted(
        (r for r in results if r.hypothesis_id == "H4" and not r.test_id.endswith("_pending_answerability")),
        key=lambda r: r.test_id,
    )
    h4_has_mcnemar = any(r.test_id.startswith("h4_mcnemar_yes") for r in h4_results)
    if h4_has_mcnemar:
        lines.extend(["", MCNEMAR_H4_LEGEND, ""])
    for r in h4_results:
        tid = r.test_id
        h4_any = True
        if tid.startswith("h4_mcnemar_yes"):
            lines.append(f"- `{tid}`{_human(tid)}")
            lines.append(f"  {_format_mcnemar_detail(r)}{_rel(r)}")
        elif r.p1 is not None and r.p2 is not None:
            lines.append(
                f"- `{tid}`{_human(tid)}: P(Yes|with)={r.p1:.3f} vs P(Yes|without)={r.p2:.3f}, "
                f"Δ={r.effect:+.3f}, p={_fmt_p(r.p_raw)}{_q_rej(r)}{_rel(r)}"
            )
        else:
            lines.append(f"- `{tid}`{_human(tid)}: p={_fmt_p(r.p_raw)}{_rel(r)}")
    sk4 = _pick("h4_pending_answerability")
    if sk4 and not h4_any:
        lines.append(f"- **Skipped:** {sk4.description}")

    def _render_h_section(hid: str, title: str, *, mcnemar_prefix: str | None = None) -> None:
        nonlocal lines
        lines.extend(["", f"## {hid} — {title}", ""])
        hres = sorted(
            (r for r in results if r.hypothesis_id == hid),
            key=lambda r: r.test_id,
        )
        if not hres:
            sk = _pick(f"{hid.lower()}_pending_answerability")
            if sk:
                lines.append(f"- **Skipped:** {sk.description}")
            return
        if mcnemar_prefix and any(r.test_id.startswith(mcnemar_prefix) for r in hres):
            if hid == "H7":
                lines.extend(["", MCNEMAR_H7_LEGEND, ""])
            elif hid == "H8":
                lines.extend(
                    [
                        "",
                        "**McNemar H8:** строка = self-Q (No/Yes), столбец = main choice C; "
                        "H1: **c > b** (чаще C при self=No).",
                        "",
                    ]
                )
        for r in hres:
            tid = r.test_id
            if mcnemar_prefix and tid.startswith(mcnemar_prefix):
                lines.append(f"- `{tid}`{_human(tid)}")
                lines.append(f"  {_format_mcnemar_detail(r)}{_rel(r)}")
            elif r.p1 is not None and r.p2 is not None:
                lines.append(
                    f"- `{tid}`{_human(tid)}: P₁={r.p1:.3f} vs P₂={r.p2:.3f}, "
                    f"Δ={r.effect:+.3f}, p={_fmt_p(r.p_raw)}{_q_rej(r)}{_rel(r)}"
                )
            else:
                lines.append(f"- `{tid}`{_human(tid)}: p={_fmt_p(r.p_raw)}{_rel(r)}")

    lines.extend(["", "## H5 — abstain rate: no_evidence vs evidence", ""])
    for tid in ("h5_abstain_no_vs_ev_man", "h5_abstain_no_vs_ev_woman"):
        r = _pick(tid)
        if r and r.p_raw is not None:
            lines.append(
                f"- `{tid}`{_human(tid)}: P(C) {r.p1:.3f} vs {r.p2:.3f}, "
                f"p={_fmt_p(r.p_raw)}{_q_rej(r)}{_rel(r)}"
            )

    lines.extend(["", "## H6 — choice vs yes/no format", ""])
    for tid, label in (
        ("h6_two_prop_choice_man_vs_yesno_man", "same-axis man"),
        ("h6_two_prop_choice_woman_vs_yesno_woman", "same-axis woman"),
        ("h6_two_prop_choice_man_vs_yesno_woman", "cross man vs yesno_woman"),
        ("h6_two_prop_choice_woman_vs_yesno_man", "cross woman vs yesno_man"),
    ):
        r = _pick(tid)
        if r and r.p_raw is not None:
            lines.append(
                f"- `{tid}`{_human(tid)} ({label}): P(choice)={r.p1:.3f} vs P(Yes|yesno)={r.p2:.3f}, "
                f"p={_fmt_p(r.p_raw)}{_q_rej(r)}{_rel(r)}"
            )

    _render_h_section(
        "H7",
        "context ↑ answerability",
        mcnemar_prefix="h7_mcnemar_",
    )
    _render_h_section(
        "H8",
        "unanswerable ↔ abstain",
        mcnemar_prefix="h8_mcnemar_",
    )
    _render_h_section(
        "H9",
        "answerability by supported gender",
        mcnemar_prefix="h9_mcnemar_",
    )

    lines.extend(["", "## Rates (choice rows, excerpt)", ""])
    rc = rates[rates["slice"] == "overall"]
    if len(rc):
        row = rc.iloc[0]
        lines.append(
            f"- Overall rate_man={row['rate_man']:.3f} "
            f"[{row['rate_man_ci_low']:.3f}, {row['rate_man_ci_high']:.3f}], n={int(row['n'])}"
        )

    lines.extend(_fdr_summary_sections(results, fdr))

    if post_hoc_results:
        lines.extend(
            [
                "",
                "## Post-hoc H — yes/no position control",
                "",
                "_(Not H6; see `post_hoc_H.csv`)_",
                "",
            ]
        )
        for r in post_hoc_results:
            tid = r.test_id
            if tid == "post_h_two_prop_yesno_man_vs_yesno_woman":
                lines.append(
                    f"- `{tid}`{_human(tid)}: P(Yes|yesno_man)={r.p1:.3f} vs "
                    f"P(Yes|yesno_woman)={r.p2:.3f}, p={_fmt_p(r.p_raw)}{_rel(r)}"
                )
            elif tid == "post_h_mcnemar_yesno_pair":
                lines.append(f"- `{tid}`{_human(tid)}")
                lines.append(
                    f"  {_format_mcnemar_detail(r)}{_rel(r)}"
                )
            else:
                lines.append(
                    f"- `{tid}`{_human(tid)}: p={_fmt_p(r.p_raw)}{_rel(r)}"
                )

    lines.extend(
        [
            "",
            "## Probing cross-reference",
            "",
            "Compare with `results/<run>/probes/h11/` (log_odds decodable from HS).",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    out_dir: Path,
    *,
    run_dir: Path,
    results: list[TestResult],
    post_hoc_results: list[TestResult] | None = None,
    rates_summary: pd.DataFrame,
    rates_family: pd.DataFrame,
    level: str,
    fdr: float,
    hypotheses: list[str],
    sanity: dict[str, Any] | None = None,
    annotation_jsonl: str | None = None,
    h4_formats: list[str] | None = None,
    position_variant: str | None = None,
    context_order: str | None = None,
    n_rows_raw: int | None = None,
    n_rows_metrics: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rates_summary.to_csv(out_dir / "rates_summary.csv", index=False)
    rates_family.to_csv(out_dir / "rates_by_family.csv", index=False)
    tests_df = tests_to_dataframe(results)
    tests_df.to_csv(out_dir / "tests_all.csv", index=False)
    post_hoc = post_hoc_results or []
    if post_hoc:
        tests_to_dataframe(post_hoc).to_csv(out_dir / "post_hoc_H.csv", index=False)
    per_hypothesis_csv: dict[str, str] = {}
    for hid in hypotheses:
        sub = tests_df[tests_df["hypothesis_id"] == hid]
        if len(sub):
            name = f"tests_{hid.lower()}.csv"
            sub.to_csv(out_dir / name, index=False)
            per_hypothesis_csv[hid] = name

    payload: dict[str, Any] = {
        "run_dir": str(run_dir),
        "level": level,
        "fdr": fdr,
        "hypotheses": hypotheses,
        "h4_formats": h4_formats,
        "position_variant": position_variant,
        "context_order": context_order,
        "n_rows_raw": n_rows_raw,
        "n_rows_metrics": n_rows_metrics,
        "n_tests": len(results),
        "n_post_hoc_tests": len(post_hoc),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "rates_summary": "rates_summary.csv",
            "rates_by_family": "rates_by_family.csv",
            "tests_all": "tests_all.csv",
            "post_hoc_H": "post_hoc_H.csv" if post_hoc else None,
            "summary": "summary.md",
            "tests_by_hypothesis": per_hypothesis_csv,
        },
        "tests": tests_to_meta_list(results),
        "post_hoc_tests": tests_to_meta_list(post_hoc),
    }
    if sanity:
        payload["sanity"] = sanity
    if annotation_jsonl:
        payload["annotation_jsonl"] = annotation_jsonl
    (out_dir / "meta.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_summary_md(
        out_dir / "summary.md",
        run_dir=run_dir,
        results=results,
        post_hoc_results=post_hoc,
        rates=rates_summary,
        level=level,
        fdr=fdr,
    )
