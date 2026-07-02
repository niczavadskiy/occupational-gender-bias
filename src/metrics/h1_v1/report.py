"""Markdown / CSV outputs for h1_v1 pipeline."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.metrics.h1_v1.flip_rates import flip_rates_rows
from src.metrics.h1_v1.hypotheses import SOC_STRATUM_ABSTAIN
from src.metrics.h1_v1.summary_html import write_summary_html
from src.metrics.reliability import RELIABILITY_OK
from src.metrics.report import test_result_to_meta, tests_to_dataframe
from src.metrics.fdr import p_for_bh
from src.metrics.stats_tests import TestResult


def _fmt_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    return f"{p:.4g}"


def _fmt_q(r: TestResult) -> str:
    if r.q_value is None or (isinstance(r.q_value, float) and math.isnan(r.q_value)):
        return "—"
    return f"{r.q_value:.4g}"


def _bh_method_section(fdr: float) -> list[str]:
    return [
        "## Benjamini–Hochberg (FDR)",
        "",
        "**Primary** (`h1v1_primary_*`) — pre-specified confirmatory тесты: **BH не применяется**.",
        "Коррекция — только в **strata / post-hoc** семействах ниже.",
        "",
        "В каждом **семействе** (choice и prob — отдельно) из **m** тестов: "
        "choice strata — **cluster t-test p** (θ_i по item); prob — **cluster margin p**.",
        "",
        "1. Сортировка: p₁ ≤ p₂ ≤ … ≤ pₘ",
        "2. qᵢ = min_{j≥i} (pⱼ·m/j), qᵢ ≤ 1",
        f"3. **Значимо** ⟺ q-value ≤ Q (Q = {fdr}); столбец **BH** ✓",
        "",
        "Отклонение H₀: k = max{i : pᵢ ≤ (i/m)·Q} → значимы ранги 1…k. Код: `src/metrics/fdr.py`.",
        "",
    ]


def _fmt_fdr(r: TestResult) -> str:
    if r.rejected_fdr is None:
        return "—"
    return "✓" if r.rejected_fdr else "—"


def _fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.3f}"


def _fmt_rate_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{100.0 * x:.1f}%"


def _fmt_margin(effect: float | None) -> str:
    if effect is None or (isinstance(effect, float) and math.isnan(effect)):
        return "—"
    return f"{effect:+.3f}"


def _fmt_ci(r: TestResult) -> str:
    lo, hi = r.ci_low, r.ci_high
    if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi)):
        return "—"
    return f"[{lo:.3f}, {hi:.3f}]"


def _reliability_short(r: TestResult) -> str:
    level = r.extra.get("reliability")
    if level == RELIABILITY_OK:
        return "ok"
    reasons = r.extra.get("reliability_reasons") or []
    if reasons:
        short = "; ".join(reasons[:2])
        if len(reasons) > 2:
            short += f"; +{len(reasons) - 2}"
        return f"not_ok ({short})"
    return "not_ok"


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["_(нет данных)_", ""]
    sep = ["---"] * len(headers)
    align = [":---" if i == 0 else "---:" for i in range(len(headers))]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


_FDR_WIDE_HEADERS = [
    "ch_p",
    "ch_q",
    "ch_BH",
    "ch_rel",
    "pr_p",
    "pr_q",
    "pr_BH",
    "pr_rel",
]


def _fdr_wide_cells(choice_r: TestResult, prob_r: TestResult | None) -> list[str]:
    cells = [
        _fmt_p(p_for_bh(choice_r)),
        _fmt_q(choice_r),
        _fmt_fdr(choice_r),
        _reliability_short(choice_r),
    ]
    if prob_r:
        cells.extend([
            _fmt_p(prob_r.p_raw),
            _fmt_q(prob_r),
            _fmt_fdr(prob_r),
            _reliability_short(prob_r),
        ])
    else:
        cells.extend(["—"] * 4)
    return cells


def _sort_by_q(results: list[TestResult]) -> list[TestResult]:
    return sorted(results, key=lambda x: x.q_value if x.q_value is not None else 1.0)


def _fdr_row_cells(r: TestResult) -> list[str]:
    return [_fmt_p(p_for_bh(r)), _fmt_q(r), _fmt_fdr(r), _reliability_short(r)]


def _fdr_dual_section_table(
    choice_results: list[TestResult],
    prob_by_choice_tid: dict[str, TestResult | None],
    *,
    fdr: float,
    choice_family: str,
    prob_family: str,
    label_header: str,
    label_fn,
) -> list[str]:
    if not choice_results:
        return []
    ch_rej = sum(1 for r in choice_results if r.rejected_fdr is True)
    pr_results = [prob_by_choice_tid.get(r.test_id) for r in choice_results]
    pr_results = [r for r in pr_results if r is not None]
    pr_rej = sum(1 for r in pr_results if r.rejected_fdr is True)
    lines = [
        f"### BH-FDR — `{choice_family}` + `{prob_family}` (Q={fdr})",
        "",
        f"Тестов: **{len(choice_results)}** choice (значимо после BH: **{ch_rej}**), "
        f"**{len(pr_results)}** prob (значимо после BH: **{pr_rej}**). "
        f"Значимо ⟺ q-value ≤ Q.",
        "",
    ]
    rows: list[list[str]] = []
    for ch in _sort_by_q(choice_results):
        pr = prob_by_choice_tid.get(ch.test_id)
        rows.append([label_fn(ch), *_fdr_wide_cells(ch, pr)])
    lines.extend(_md_table([label_header, *_FDR_WIDE_HEADERS], rows))
    return lines


def _fdr_split_section_tables(
    choice_results: list[TestResult],
    prob_by_choice_tid: dict[str, TestResult | None],
    *,
    fdr: float,
    choice_family: str,
    prob_family: str,
    label_header: str,
    label_fn,
) -> list[str]:
    if not choice_results:
        return []
    pr_labeled = [
        (label_fn(ch), pr)
        for ch in choice_results
        if (pr := prob_by_choice_tid.get(ch.test_id)) is not None
    ]
    ch_rej = sum(1 for r in choice_results if r.rejected_fdr is True)
    pr_rej = sum(1 for _, r in pr_labeled if r.rejected_fdr is True)
    lines = [
        f"### BH-FDR — `{choice_family}` / `{prob_family}` (Q={fdr})",
        "",
        f"Choice: **{len(choice_results)}** тестов, значимо после BH: **{ch_rej}**. "
        f"Prob: **{len(pr_labeled)}** тестов, значимо после BH: **{pr_rej}**. "
        "Сортировка по **q-value** (возрастание). Значимо ⟺ q-value ≤ Q.",
        "",
        f"#### `{choice_family}` (choice)",
        "",
    ]
    ch_rows = [[label_fn(ch), *_fdr_row_cells(ch)] for ch in _sort_by_q(choice_results)]
    lines.extend(_md_table([label_header, "p", "q-value", "BH", "rel"], ch_rows))
    lines.extend([
        f"#### `{prob_family}` (prob_constrained)",
        "",
    ])
    pr_rows = [[label, *_fdr_row_cells(pr)] for label, pr in sorted(
        pr_labeled,
        key=lambda x: x[1].q_value if x[1].q_value is not None else 1.0,
    )]
    lines.extend(_md_table([label_header, "p", "q-value", "BH", "rel"], pr_rows))
    return lines


def _prob_tid_margin(choice_tid: str) -> str:
    return choice_tid.replace("_man_vs_woman", "_margin_prob")


_MW_WIDE_HEADERS = [
    "ch_P(man)",
    "ch_P(wom)",
    "ch_k_m",
    "ch_k_w",
    "ch_Δ",
    "ch_CI",
    "ch_p",
    "ch_q",
    "ch_BH",
    "pr_μ_man",
    "pr_μ_wom",
    "pr_margin",
    "pr_CI",
    "pr_p",
    "pr_q",
    "pr_BH",
]

_MW_WIDE_HEADERS_NO_FDR = [
    "ch_P(man)",
    "ch_P(wom)",
    "ch_k_m",
    "ch_k_w",
    "ch_Δ",
    "ch_CI",
    "ch_p",
    "ch_q",
    "pr_μ_man",
    "pr_μ_wom",
    "pr_margin",
    "pr_CI",
    "pr_p",
    "pr_q",
]


def _primary_section_lines(
    ch_primary: TestResult | None,
    pr_primary: TestResult | None,
) -> list[str]:
    if ch_primary is None:
        return ["_(нет primary)_", ""]
    cc = (ch_primary.extra or {}).get("cluster_choice") or {}
    pr_ex = pr_primary.extra if pr_primary else {}
    lines = [
        "*Pre-specified primary: **BH/FDR не применяется** (один confirmatory тест, не семейство strata).*",
        "",
        "### Choice — cluster t-test (главный)",
        "",
    ]
    ch_rows = [[
        str(cc.get("n_clusters_with_gender") or "—"),
        f"{float(cc.get('mean_theta')):.4f}" if cc.get("mean_theta") is not None else "—",
        _fmt_margin(cc.get("mean_gap")),
        _fmt_p(cc.get("cluster_p")),
        _fmt_p(cc.get("cluster_wilcoxon_p")),
        _fmt_p(ch_primary.p_raw),
    ]]
    lines.extend(_md_table(
        ["G", "θ̄", "Δ", "cluster p", "Wilcoxon p", "pooled p (сравн.)"],
        ch_rows,
    ))
    if pr_primary:
        lines.extend([
            "### Prob — cluster t-test (supporting)",
            "",
        ])
        pr_rows = [[
            str(pr_primary.n1 or "—"),
            _fmt_margin(pr_primary.effect),
            _fmt_p(pr_primary.p_raw),
            _fmt_p(pr_ex.get("cluster_wilcoxon_p")),
        ]]
        lines.extend(_md_table(
            ["G", "mean margin", "cluster p", "Wilcoxon p"],
            pr_rows,
        ))
    return lines


def _mw_wide_cells(choice_r: TestResult, prob_r: TestResult | None, *, include_fdr: bool = True) -> list[str]:
    cells = [
        _fmt_pct(choice_r.p1),
        _fmt_pct(choice_r.p2),
        str(int(choice_r.k1 or 0)),
        str(int(choice_r.k2 or 0)),
        _fmt_margin(choice_r.effect),
        _fmt_ci(choice_r),
        _fmt_p(choice_r.p_raw),
        _fmt_q(choice_r),
    ]
    if include_fdr:
        cells.append(_fmt_fdr(choice_r))
    if prob_r:
        cells.extend([
            _fmt_pct(prob_r.p1),
            _fmt_pct(prob_r.p2),
            _fmt_margin(prob_r.effect),
            _fmt_ci(prob_r),
            _fmt_p(prob_r.p_raw),
            _fmt_q(prob_r),
        ])
        if include_fdr:
            cells.append(_fmt_fdr(prob_r))
    else:
        cells.extend(["—"] * (6 if include_fdr else 5))
    return cells


_MCNEMAR_METRIC_HEADERS = [
    "n",
    "b",
    "c",
    "flip",
    "1st",
    "p",
    "q",
    "BH",
]

_MCNEMAR_METRIC_HEADERS_NO_FDR = [
    "n",
    "b",
    "c",
    "flip",
    "1st",
    "p",
    "q",
]


def _mcnemar_metric_cells(r: TestResult, *, include_fdr: bool = True) -> list[str]:
    ex = r.extra
    b = int(ex.get("discordant_b_man_a_woman_b") or r.k1 or 0)
    c = int(ex.get("discordant_c_woman_a_man_b") or r.k2 or 0)
    n = int(ex.get("n_pairs") or 0)
    cells = [
        str(n),
        str(b),
        str(c),
        _fmt_rate_pct(ex.get("order_flip")),
        _fmt_rate_pct(ex.get("first_shown_pick")),
        _fmt_p(r.p_raw),
        _fmt_q(r),
    ]
    if include_fdr:
        cells.append(_fmt_fdr(r))
    return cells


def _mcnemar_wide_headers(*, include_fdr: bool = True) -> list[str]:
    base = _MCNEMAR_METRIC_HEADERS if include_fdr else _MCNEMAR_METRIC_HEADERS_NO_FDR
    return [f"ch_{h}" for h in base] + [f"pr_{h}" for h in base]


def _mcnemar_wide_cells(
    choice_r: TestResult,
    prob_r: TestResult | None,
    *,
    include_fdr: bool = True,
) -> list[str]:
    n_cols = len(_MCNEMAR_METRIC_HEADERS if include_fdr else _MCNEMAR_METRIC_HEADERS_NO_FDR)
    cells = _mcnemar_metric_cells(choice_r, include_fdr=include_fdr)
    if prob_r:
        cells.extend(_mcnemar_metric_cells(prob_r, include_fdr=include_fdr))
    else:
        cells.extend(["—"] * n_cols)
    return cells


def _prob_tid_for_mcnemar(choice_tid: str) -> str:
    return f"{choice_tid}_prob"


def _prob_map_for_choice(
    by_id: dict[str, TestResult],
    choice_results: list[TestResult],
    prob_tid_fn,
) -> dict[str, TestResult | None]:
    return {ch.test_id: by_id.get(prob_tid_fn(ch.test_id)) for ch in choice_results}


def write_summary_md(
    path: Path,
    *,
    run_dir: Path,
    results: list[TestResult],
    rates: pd.DataFrame,
    fdr: float,
    min_stratum_n: int,
    min_soc_g: int,
    n_rows_raw: int,
    n_rows_metrics: int,
) -> None:
    by_id = {r.test_id: r for r in results}
    lines: list[str] = [
        "# H1 v1 metrics summary",
        "",
        f"- **Run:** `{run_dir.name}`",
        f"- **Pipeline:** `h1_v1` (occupation-based choice + prob_constrained sensitivity)",
        f"- **Rows:** {n_rows_metrics} (raw {n_rows_raw})",
        f"- **Целевой FDR (Q):** {fdr}",
        f"- **Min stratum n (soc/onet):** {min_stratum_n}",
        f"- **Min soc G (base items):** {min_soc_g}",
        f"- **Reliability:** `ok` | `not_ok`",
        f"- **Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "Primary — вне BH-семейства (только p-value). Остальные разделы: BH внутри семейства "
        f"(таблица в конце раздела). Подробнее и пример — `summary.html`.",
        "Layout и onet: только CSV (`tests_h1_v1_layout.csv`, `tests_h1_v1_onet.csv`).",
        "Таблицы: **ch_** = choice (argmax), **pr_** = prob_constrained.",
        "",
        *_bh_method_section(fdr),
        "## Primary — man vs woman",
        "",
        "H1: предпочтение man vs woman по всем layout/context/abstain.",
        "",
    ]
    ch_primary = by_id.get("h1v1_primary_man_vs_woman")
    pr_primary = by_id.get("h1v1_primary_man_vs_woman_prob")
    lines.extend(_primary_section_lines(ch_primary, pr_primary))

    # --- abstain ---
    abst_choice = sorted(
        [r for r in results if r.test_id.startswith("h1v1_abst_") and r.test_id.endswith("_man_vs_woman")],
        key=lambda x: x.test_id,
    )
    abst_rows: list[list[str]] = []
    for r in abst_choice:
        prob = by_id.get(_prob_tid_margin(r.test_id))
        abst_rows.append([
            str(r.extra.get("abstain_variant", "?")),
            str(int(r.n1 or 0)),
            *_mw_wide_cells(r, prob),
        ])
    lines.extend([
        "## Strata — abstain (pooled layout)",
        "",
        "P(man) vs P(woman) по `abstain_variant`; **ch_** = choice, **pr_** = prob_constrained.",
        "",
    ])
    lines.extend(
        _md_table(
            ["variant", "n", *_MW_WIDE_HEADERS],
            abst_rows,
        )
    )
    abst_prob_map = _prob_map_for_choice(by_id, abst_choice, _prob_tid_margin)
    lines.extend(_fdr_dual_section_table(
        abst_choice,
        abst_prob_map,
        fdr=fdr,
        choice_family="h1v1_strata_abstain",
        prob_family="h1v1_strata_abstain_prob",
        label_header="variant",
        label_fn=lambda r: str(r.extra.get("abstain_variant", "?")),
    ))

    # --- soc ---
    soc_choice = sorted(
        [r for r in results if r.test_id.startswith("h1v1_soc_") and r.test_id.endswith("_man_vs_woman")],
        key=lambda x: str(x.extra.get("soc_major_title", "")).lower(),
    )
    soc_rows: list[list[str]] = []
    for r in soc_choice:
        prob = by_id.get(_prob_tid_margin(r.test_id))
        soc_rows.append([
            str(r.extra.get("soc_major_title", ""))[:60],
            str(int(r.n1 or 0)),
            *_mw_wide_cells(r, prob, include_fdr=False),
        ])
    lines.extend([
        f"## Strata — soc_major_title (man vs woman, n≥{min_stratum_n}, G≥{min_soc_g})",
        "",
        f"*Срез: только `{SOC_STRATUM_ABSTAIN}` — 2 опции (man/woman), p0/p1, mf/wf; "
        "без «Воздержаться».*",
        "",
        f"*{len(soc_choice)} strata*; **ch_** = choice, **pr_** = prob_constrained.",
        "",
    ])
    lines.extend(
        _md_table(
            ["soc_major_title", "n", *_MW_WIDE_HEADERS_NO_FDR],
            soc_rows,
        )
    )
    soc_prob_map = _prob_map_for_choice(by_id, soc_choice, _prob_tid_margin)
    lines.extend(_fdr_split_section_tables(
        soc_choice,
        soc_prob_map,
        fdr=fdr,
        choice_family="h1v1_strata_soc_major",
        prob_family="h1v1_strata_soc_major_prob",
        label_header="soc_major_title",
        label_fn=lambda r: str(r.extra.get("soc_major_title", ""))[:60],
    ))

    # --- position / context (at end) ---
    pos_mcn_choice = sorted(
        [
            r for r in results
            if r.test_id.startswith("h1v1_mcnemar_pos_p0_p1") and not r.test_id.endswith("_prob")
        ],
        key=lambda x: x.test_id,
    )
    lines.extend([
        "## Strata — position (p0↔p1, without_abstain)",
        "",
        "McNemar: b = man@p0 & woman@p1; c = woman@p0 & man@p1 (bias к слоту A). "
        "prob: ties (`p_man`=`p_woman`) исключены. "
        "[Order Flip / First-Shown Pick](https://github.com/lechmazur/position_bias).",
        "",
        "**Ключевые заметки:** Order Flip = (b+c)/n. "
        "First-Shown (position): choice = (# choice=A)/n на p0+p1; prob = mean(prob_constrained_A) "
        "(bias к слоту A, не к man). b≫c → position bias. "
        "Таблица: n, b, c, flip, 1st, p.",
        "",
    ])
    pos_rows: list[list[str]] = []
    for r in pos_mcn_choice:
        ctx = str(r.extra.get("context_order", ""))
        prob = by_id.get(_prob_tid_for_mcnemar(r.test_id))
        pos_rows.append([ctx, *_mcnemar_wide_cells(r, prob)])
    lines.extend(
        _md_table(
            ["ctx", *_mcnemar_wide_headers()],
            pos_rows,
        )
    )
    pos_prob_map = _prob_map_for_choice(by_id, pos_mcn_choice, _prob_tid_for_mcnemar)
    lines.extend(_fdr_dual_section_table(
        pos_mcn_choice,
        pos_prob_map,
        fdr=fdr,
        choice_family="h1v1_strata_position_mcnemar",
        prob_family="h1v1_strata_position_mcnemar_prob",
        label_header="ctx",
        label_fn=lambda r: str(r.extra.get("context_order", "")),
    ))

    ctx_flip_choice = sorted(
        [
            r
            for r in results
            if r.test_id.startswith("h1v1_mcnemar_ctx_")
            and r.extra.get("abstain_variant") == "without_abstain"
            and str(r.extra.get("position_variant")) in ("p0", "p1")
            and not r.test_id.endswith("_prob")
        ],
        key=lambda x: str(x.extra.get("position_variant")),
    )
    ctx_mcn_other = sorted(
        [
            r
            for r in results
            if r.test_id.startswith("h1v1_mcnemar_ctx_")
            and r.extra.get("abstain_variant") == "with_abstain"
            and not r.test_id.endswith("_prob")
        ],
        key=lambda x: (str(x.extra.get("position_variant")), x.test_id),
    )

    lines.extend([
        "## Strata — context (mf↔wf)",
        "",
        "McNemar: b = первый в narrative; c = второй в narrative.",
        "",
        "**Ключевые заметки:** c≫b → recency. "
        "First-Shown (context): choice = (prefers_man@mf + prefers_woman@wf)/(n_mf+n_wf); "
        "prob = mean(p_man@mf ∪ p_woman@wf). &lt;50% → сдвиг ко второму в narrative. "
        "Таблица: n, b, c, flip, 1st, p.",
        "",
        "### p0/p1, without_abstain",
        "",
    ])
    ctx_p01_rows: list[list[str]] = []
    for r in ctx_flip_choice:
        pos = str(r.extra.get("position_variant", ""))
        prob = by_id.get(_prob_tid_for_mcnemar(r.test_id))
        ctx_p01_rows.append([pos, *_mcnemar_wide_cells(r, prob, include_fdr=False)])
    lines.extend(
        _md_table(
            ["pos", *_mcnemar_wide_headers(include_fdr=False)],
            ctx_p01_rows,
        )
    )

    if ctx_mcn_other:
        ctx_wa_rows: list[list[str]] = []
        for r in ctx_mcn_other:
            pos = str(r.extra.get("position_variant", ""))
            prob = by_id.get(_prob_tid_for_mcnemar(r.test_id))
            ctx_wa_rows.append([pos, *_mcnemar_wide_cells(r, prob, include_fdr=False)])
        lines.extend([
            "### p0–p5, with_abstain",
            "",
        ])
        lines.extend(
            _md_table(
                ["pos", *_mcnemar_wide_headers(include_fdr=False)],
                ctx_wa_rows,
            )
        )

    ctx_mcn_all_choice = ctx_flip_choice + ctx_mcn_other
    ctx_prob_map = _prob_map_for_choice(by_id, ctx_mcn_all_choice, _prob_tid_for_mcnemar)
    lines.extend(_fdr_split_section_tables(
        ctx_mcn_all_choice,
        ctx_prob_map,
        fdr=fdr,
        choice_family="h1v1_strata_context_mcnemar",
        prob_family="h1v1_strata_context_mcnemar_prob",
        label_header="slice",
        label_fn=lambda r: (
            f"{r.extra.get('position_variant')}/{r.extra.get('abstain_variant')}"
        ),
    ))

    lines.extend(["## Rates overview", "", "See `rates_summary.csv`.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    out_dir: Path,
    *,
    run_dir: Path,
    results: list[TestResult],
    rates_summary: pd.DataFrame,
    rates_scenario: pd.DataFrame,
    df: pd.DataFrame,
    fdr: float,
    min_stratum_n: int,
    min_soc_g: int,
    n_rows_raw: int,
    n_rows_metrics: int,
    position_variant: str | None,
    context_order: str | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rates_summary.to_csv(out_dir / "rates_summary.csv", index=False)
    rates_scenario.to_csv(out_dir / "rates_by_scenario.csv", index=False)

    tests_df = tests_to_dataframe(results)
    tests_df.to_csv(out_dir / "tests_all.csv", index=False)

    layout_df = tests_df[
        tests_df["test_id"].astype(str).str.startswith("h1v1_layout_")
        & tests_df["test_id"].astype(str).str.endswith("_man_vs_woman")
    ]
    layout_prob_df = tests_df[
        tests_df["test_id"].astype(str).str.startswith("h1v1_layout_")
        & tests_df["test_id"].astype(str).str.endswith("_margin_prob")
    ]
    if len(layout_df):
        layout_df.to_csv(out_dir / "tests_h1_v1_layout.csv", index=False)
    if len(layout_prob_df):
        layout_prob_df.to_csv(out_dir / "tests_h1_v1_layout_prob.csv", index=False)

    mcn_df = tests_df[
        tests_df["test_id"].astype(str).str.startswith("h1v1_mcnemar_")
        & ~tests_df["test_id"].astype(str).str.endswith("_prob")
    ]
    mcn_prob_df = tests_df[
        tests_df["test_id"].astype(str).str.startswith("h1v1_mcnemar_")
        & tests_df["test_id"].astype(str).str.endswith("_prob")
    ]
    if len(mcn_df):
        mcn_df.to_csv(out_dir / "tests_h1_v1_mcnemar.csv", index=False)
    if len(mcn_prob_df):
        mcn_prob_df.to_csv(out_dir / "tests_h1_v1_mcnemar_prob.csv", index=False)

    prob_ids = (
        tests_df["test_id"].astype(str).str.endswith("_prob")
        | tests_df["test_id"].astype(str).str.endswith("_margin_prob")
    )
    prob_core_df = tests_df[prob_ids]
    if len(prob_core_df):
        prob_core_df.to_csv(out_dir / "tests_h1_v1_prob_core.csv", index=False)

    flip_rows = flip_rates_rows(results)
    flip_df = pd.DataFrame(flip_rows)
    if len(flip_df):
        flip_df.to_csv(out_dir / "flip_rates.csv", index=False)

    soc_df = tests_df[tests_df["test_id"].astype(str).str.startswith("h1v1_soc_")]
    if len(soc_df):
        soc_df.to_csv(out_dir / "tests_h1_v1_soc.csv", index=False)
    onet_df = tests_df[tests_df["test_id"].astype(str).str.startswith("h1v1_onet_")]
    if len(onet_df):
        onet_df.to_csv(out_dir / "tests_h1_v1_onet.csv", index=False)

    payload: dict[str, Any] = {
        "pipeline": "h1_v1",
        "run_dir": str(run_dir),
        "fdr": fdr,
        "min_stratum_n": min_stratum_n,
        "min_soc_g": min_soc_g,
        "position_variant": position_variant,
        "context_order": context_order,
        "n_rows_raw": n_rows_raw,
        "n_rows_metrics": n_rows_metrics,
        "n_tests": len(results),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "rates_summary": "rates_summary.csv",
            "rates_by_scenario": "rates_by_scenario.csv",
            "tests_all": "tests_all.csv",
            "tests_h1_v1_layout": "tests_h1_v1_layout.csv" if len(layout_df) else None,
            "tests_h1_v1_layout_prob": "tests_h1_v1_layout_prob.csv" if len(layout_prob_df) else None,
            "tests_h1_v1_mcnemar": "tests_h1_v1_mcnemar.csv" if len(mcn_df) else None,
            "tests_h1_v1_mcnemar_prob": "tests_h1_v1_mcnemar_prob.csv" if len(mcn_prob_df) else None,
            "tests_h1_v1_prob_core": "tests_h1_v1_prob_core.csv" if len(prob_core_df) else None,
            "flip_rates": "flip_rates.csv" if len(flip_rows) else None,
            "tests_h1_v1_soc": "tests_h1_v1_soc.csv" if len(soc_df) else None,
            "tests_h1_v1_onet": "tests_h1_v1_onet.csv" if len(onet_df) else None,
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
        fdr=fdr,
        min_stratum_n=min_stratum_n,
        min_soc_g=min_soc_g,
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
        min_soc_g=min_soc_g,
    )
