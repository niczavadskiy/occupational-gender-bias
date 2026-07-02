"""Markdown / CSV outputs for h3_v1 pipeline."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.metrics.h3_v1.hypotheses import MCNEMAR_ABSTAIN_VARIANT
from src.metrics.h3_v1.summary_html import write_summary_html
from src.metrics.report import test_result_to_meta, tests_to_dataframe
from src.metrics.stats_tests import TestResult


def _fmt_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    return f"{p:.4g}"


def _fmt_q(r: TestResult) -> str:
    if r.q_value is None or (isinstance(r.q_value, float) and math.isnan(r.q_value)):
        return "—"
    return f"{r.q_value:.4g}"


def _fmt_fdr(r: TestResult) -> str:
    if r.rejected_fdr is None:
        return "—"
    return "✓" if r.rejected_fdr else "—"


def _fmt_margin(effect: float | None) -> str:
    if effect is None or (isinstance(effect, float) and math.isnan(effect)):
        return "—"
    return f"{effect:+.3f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["_(нет данных)_", ""]
    align = [":---" if i == 0 else "---:" for i in range(len(headers))]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def write_summary_md(
    path: Path,
    *,
    run_dir: Path,
    results: list[TestResult],
    rates: pd.DataFrame,
    flip_rates: pd.DataFrame,
    fdr: float,
    min_stratum_n: int,
    n_rows_raw: int,
    n_rows_metrics: int,
) -> None:
    by_id = {r.test_id: r for r in results}
    primary_ids = [
        "h3v1_mcnemar_no_vs_man_man",
        "h3v1_mcnemar_no_vs_woman_woman",
        "h3v1_mcnemar_no_vs_man_man_prob",
        "h3v1_mcnemar_no_vs_woman_woman_prob",
    ]

    lines: list[str] = [
        "# H3 v1 metrics summary",
        "",
        f"- **Run:** `{run_dir.name}`",
        f"- **Pipeline:** `h3_v1` (evidence highlight vs no_evidence; paired by `id`)",
        f"- **Rows:** {n_rows_metrics} (raw {n_rows_raw})",
        f"- **Целевой FDR (Q):** {fdr}",
        f"- **Min stratum n (soc):** {min_stratum_n}",
        f"- **Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## H3 — гипотеза",
        "",
        "Highlight в пользу пола X увеличивает предпочтение X относительно ambiguous (no_evidence).",
        f"Primary: **paired McNemar** по `id`, срез **`{MCNEMAR_ABSTAIN_VARIANT}`** "
        "(3 804 layout-id × 3 evidence; 2-option man/woman).",
        "Prob: McNemar на знаке margin (p_man vs p_woman).",
        "",
        "## Primary — paired McNemar (without_abstain)",
        "",
    ]

    rows: list[list[str]] = []
    for tid in primary_ids:
        r = by_id.get(tid)
        if not r:
            continue
        ex = r.extra
        rows.append([
            tid,
            str(ex.get("n_pairs", r.n1 or "—")),
            str(r.k1 or 0),
            str(r.k2 or 0),
            _fmt_margin(r.effect),
            _fmt_p(r.p_raw),
            _fmt_q(r),
            _fmt_fdr(r),
        ])
    lines.extend(
        _md_table(
            ["test_id", "n_pairs", "b", "c", "effect", "p", "q", "BH"],
            rows,
        )
    )

    posthoc = sorted(
        [r for r in results if r.family_name == "h3v1_posthoc"],
        key=lambda x: x.test_id,
    )
    if posthoc:
        lines.extend(["## Post-hoc — ev_man vs ev_woman", ""])
        ph_rows = [
            [
                r.test_id,
                str(r.extra.get("n_pairs", r.n1 or "—")),
                str(r.k1 or 0),
                str(r.k2 or 0),
                _fmt_p(r.p_raw),
                _fmt_q(r),
                _fmt_fdr(r),
            ]
            for r in posthoc
        ]
        lines.extend(_md_table(["test_id", "n", "b", "c", "p", "q", "BH"], ph_rows))

    ctx_tests = sorted(
        [
            r
            for r in results
            if r.family_name == "h3v1_strata_context"
            and r.extra.get("gender_axis") == "man"
            and "no_vs_man" in r.test_id
        ],
        key=lambda x: x.test_id,
    )
    if ctx_tests:
        lines.extend(["## Strata — context_order (no vs highlight, man axis)", ""])
        ctx_rows = [
            [
                str(r.extra.get("context_order", "?")),
                r.test_id,
                str(r.extra.get("n_pairs", r.n1 or "—")),
                _fmt_margin(r.effect),
                _fmt_p(r.p_raw),
                _fmt_q(r),
            ]
            for r in ctx_tests
        ]
        lines.extend(_md_table(["ctx", "test_id", "n", "effect", "p", "q"], ctx_rows))

    if len(flip_rates):
        lines.extend(["## Evidence flip rates", "", "See `evidence_flip_rates.csv`.", ""])

    lines.extend(["## Rates overview", "", "See `rates_summary.csv`.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    out_dir: Path,
    *,
    run_dir: Path,
    results: list[TestResult],
    rates_summary: pd.DataFrame,
    flip_rates: pd.DataFrame,
    df: pd.DataFrame,
    fdr: float,
    min_stratum_n: int,
    n_rows_raw: int,
    n_rows_metrics: int,
    position_variant: str | None,
    context_order: str | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rates_summary.to_csv(out_dir / "rates_summary.csv", index=False)
    if len(flip_rates):
        flip_rates.to_csv(out_dir / "evidence_flip_rates.csv", index=False)

    tests_df = tests_to_dataframe(results)
    tests_df.to_csv(out_dir / "tests_all.csv", index=False)

    mcn_df = tests_df[
        tests_df["test_id"].astype(str).str.contains("mcnemar", case=False)
        & ~tests_df["test_id"].astype(str).str.endswith("_prob")
    ]
    if len(mcn_df):
        mcn_df.to_csv(out_dir / "tests_h3_v1_mcnemar.csv", index=False)

    prob_df = tests_df[
        tests_df["test_id"].astype(str).str.endswith("_prob")
        | tests_df["test_id"].astype(str).str.startswith("h3v1_delta_")
    ]
    if len(prob_df):
        prob_df.to_csv(out_dir / "tests_h3_v1_prob.csv", index=False)

    soc_df = tests_df[tests_df["test_id"].astype(str).str.startswith("h3v1_soc_")]
    if len(soc_df):
        soc_df.to_csv(out_dir / "tests_h3_v1_soc.csv", index=False)

    desc_df = tests_df[tests_df["test_id"].astype(str).str.startswith("h3v1_desc_")]
    if len(desc_df):
        desc_df.to_csv(out_dir / "tests_h3_v1_descriptive.csv", index=False)

    payload: dict[str, Any] = {
        "pipeline": "h3_v1",
        "run_dir": str(run_dir),
        "fdr": fdr,
        "min_stratum_n": min_stratum_n,
        "position_variant": position_variant,
        "context_order": context_order,
        "n_rows_raw": n_rows_raw,
        "n_rows_metrics": n_rows_metrics,
        "n_tests": len(results),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "rates_summary": "rates_summary.csv",
            "evidence_flip_rates": "evidence_flip_rates.csv" if len(flip_rates) else None,
            "tests_all": "tests_all.csv",
            "tests_h3_v1_mcnemar": "tests_h3_v1_mcnemar.csv" if len(mcn_df) else None,
            "tests_h3_v1_prob": "tests_h3_v1_prob.csv" if len(prob_df) else None,
            "tests_h3_v1_soc": "tests_h3_v1_soc.csv" if len(soc_df) else None,
            "tests_h3_v1_descriptive": "tests_h3_v1_descriptive.csv" if len(desc_df) else None,
            "summary": "summary.md",
            "summary_html": "summary.html",
        },
        "tests": [test_result_to_meta(r) for r in results],
    }
    (out_dir / "meta.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_summary_md(
        out_dir / "summary.md",
        run_dir=run_dir,
        results=results,
        rates=rates_summary,
        flip_rates=flip_rates,
        fdr=fdr,
        min_stratum_n=min_stratum_n,
        n_rows_raw=n_rows_raw,
        n_rows_metrics=n_rows_metrics,
    )
    write_summary_html(
        out_dir / "summary.html",
        run_dir=run_dir,
        results=results,
        fdr=fdr,
        n_rows_metrics=n_rows_metrics,
        df=df,
        rates=rates_summary,
    )
