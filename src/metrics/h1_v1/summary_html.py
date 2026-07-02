"""Self-contained HTML dashboard for H1 v1 metrics (Chart.js)."""

from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.metrics.load import load_meta
from src.metrics.h1_v1.hypotheses import (
    DEFAULT_MIN_SOC_G,
    SOC_STRATUM_ABSTAIN,
    soc_base_item_count,
)
from src.metrics.h1_v1.labels import filter_slice
from src.metrics.fdr import p_for_bh
from src.metrics.stats_tests import TestResult

CH_DARK = "#2c5f8d"
CH_LIGHT = "#9bb8d4"

SOC_SLICE_LAYOUT_NOTE = (
    "<b>Срез:</b> только <code>without_abstain</code> — 2 опции ответа (man/woman), "
    "без «Воздержаться»; <code>position_variant</code> p0/p1 и оба "
    "<code>context_order</code> (mf/wf). "
    "Не смешивается с layout <code>with_abstain</code> (см. раздел Abstain)."
)


def _j(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)


def _pct(x: float | None) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if abs(x) <= 1.0:
        return round(100.0 * x, 2)
    return round(float(x), 2)


def _margin(r: TestResult) -> float | None:
    if r.effect is not None and math.isfinite(r.effect):
        return round(float(r.effect), 4)
    if r.p1 is not None and r.p2 is not None:
        return round(float(r.p1) - float(r.p2), 4)
    return None


def _cluster_prob_margin(r: TestResult | None) -> float | None:
    if not r:
        return None
    ex = r.extra or {}
    cm = ex.get("cluster_mean_margin")
    if cm is not None and math.isfinite(float(cm)):
        return round(float(cm), 4)
    return _margin(r)


def _fmt_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    if p < 1e-4:
        return f"{p:.3e}"
    return f"{p:.4g}"


def _shapiro_from_extra(ex: dict[str, Any] | None) -> dict[str, Any]:
    ex = ex or {}
    return {
        "stat": ex.get("shapiro_stat", ex.get("stat")),
        "p": ex.get("shapiro_p", ex.get("p")),
        "n": ex.get("shapiro_n", ex.get("n")),
        "n_used": ex.get("shapiro_n_used", ex.get("n_used")),
        "subsampled": ex.get("shapiro_subsampled", ex.get("subsampled")),
        "normal": ex.get("shapiro_normal", ex.get("normal")),
        "note": ex.get("shapiro_note", ex.get("note")),
    }


def _shapiro_tip(ex: dict[str, Any] | None) -> str:
    s = _shapiro_from_extra(ex)
    if s["p"] is None or (isinstance(s["p"], float) and math.isnan(s["p"])):
        return f"Shapiro–Wilk: не применим ({s.get('note') or 'n/a'})."
    norm = s["normal"]
    verdict = (
        "нормальность не отвергается (α=0.05)"
        if norm is True
        else "нормальность отвергается (α=0.05)" if norm is False else "—"
    )
    sub = " (subsample)" if s.get("subsampled") else ""
    return (
        f"Shapiro–Wilk{sub}: W={float(s['stat']):.4g}, p={_fmt_p(s['p'])}, "
        f"n={s.get('n_used')}/{s.get('n')}; {verdict}."
    )


def _shapiro_from_result(r: TestResult | None) -> dict[str, Any]:
    return _shapiro_from_extra(r.extra if r else None)


def _wilcoxon_from_extra(ex: dict[str, Any] | None) -> dict[str, Any]:
    ex = ex or {}
    return {
        "wilcoxon_stat": ex.get("wilcoxon_stat"),
        "wilcoxon_p": ex.get("wilcoxon_p"),
    }


def _wilcoxon_tip(ex: dict[str, Any] | None) -> str:
    s = _wilcoxon_from_extra(ex)
    wp = s.get("wilcoxon_p")
    if wp is None or (isinstance(wp, float) and math.isnan(wp)):
        return "Wilcoxon signed-rank: не применим (n/a)."
    ws = s.get("wilcoxon_stat")
    stat_s = f"{float(ws):.4g}" if ws is not None and not math.isnan(float(ws)) else "—"
    return (
        f"Wilcoxon signed-rank: W={stat_s}, p={_fmt_p(wp)}; "
        f"H₀: median(margin)=0 (робастная проверка prob)."
    )


def _wilcoxon_from_result(r: TestResult | None) -> dict[str, Any]:
    return _wilcoxon_from_extra(r.extra if r else None)


def _cluster_from_extra(ex: dict[str, Any] | None) -> dict[str, Any]:
    ex = ex or {}
    return {
        "row_n": ex.get("row_n"),
        "n_clusters": ex.get("n_clusters"),
        "k_mean": ex.get("k_mean"),
        "naive_p": ex.get("naive_p"),
        "naive_se": ex.get("naive_se"),
        "cluster_se": ex.get("cluster_se"),
        "cluster_wilcoxon_p": ex.get("cluster_wilcoxon_p"),
        "icc": ex.get("icc"),
        "deff": ex.get("deff"),
        "n_eff": ex.get("n_eff"),
        "layout_contrasts": ex.get("layout_contrasts") or {},
    }


def _cluster_tip(ex: dict[str, Any] | None) -> str:
    c = _cluster_from_extra(ex)
    row_n = c.get("row_n")
    g = c.get("n_clusters")
    if not row_n or not g:
        return "Cluster t-test: не применим (n/a)."
    k = c.get("k_mean")
    k_s = f"{float(k):.1f}" if k is not None and math.isfinite(float(k)) else "—"
    cp = None
    if ex:
        cp = ex.get("cluster_p")
        if cp is None:
            cp = ex.get("pr_p")
    parts = [
        f"Единица наблюдения: base item (n={g} кластеров, {row_n} строк, k̄={k_s}).",
        f"Cluster t-test p={_fmt_p(cp)} "
        f"(Wilcoxon по item-means p={_fmt_p(c.get('cluster_wilcoxon_p'))}).",
    ]
    deff = c.get("deff")
    n_eff = c.get("n_eff")
    icc = c.get("icc")
    if deff is not None and math.isfinite(float(deff)):
        parts.append(f"DEFF≈{float(deff):.2f}, n_eff≈{float(n_eff):.0f}." if n_eff and math.isfinite(float(n_eff)) else f"DEFF≈{float(deff):.2f}.")
    if icc is not None and math.isfinite(float(icc)):
        parts.append(f"ICC≈{float(icc):.3f}.")
    naive_p = c.get("naive_p")
    if naive_p is not None and ex and ex.get("cluster_p") is not None:
        parts.append(f"Naive row-level t-test p={_fmt_p(naive_p)} (для сравнения).")
    return " ".join(parts)


def _cluster_diag_html(ex: dict[str, Any] | None) -> str:
    c = _cluster_from_extra(ex)
    contrasts = c.get("layout_contrasts") or {}
    if not contrasts:
        return ""
    labels = {
        "position_effect": "position (p0−p1)",
        "context_order_effect": "context order (wf−mf)",
        "position_x_context": "position×context",
    }
    rows = []
    for key, label in labels.items():
        row = contrasts.get(key)
        if not row:
            continue
        rows.append(
            f"<tr><td>{html.escape(label)}</td>"
            f"<td>{float(row.get('effect', float('nan'))):+.4f}</td>"
            f"<td>{_fmt_p(row.get('p'))}</td>"
            f"<td>{row.get('n_clusters', '—')}</td></tr>"
        )
    if not rows:
        return ""
    return (
        "<p><b>Парные контрасты по base items</b> (2×2 swap, without_abstain-подобный layout):</p>"
        "<table class='tip-fdr'><thead><tr>"
        "<th>Контраст</th><th>mean δ</th><th>t-test p</th><th>G</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _cluster_from_result(r: TestResult | None) -> dict[str, Any]:
    ex = r.extra if r else None
    out = _cluster_from_extra(ex)
    if ex:
        out["cluster_p"] = ex.get("cluster_p")
    return out


def _choice_cluster_from_extra(ex: dict[str, Any] | None) -> dict[str, Any]:
    cc = (ex or {}).get("cluster_choice") or {}
    if not cc:
        return {}
    return {
        "ch_cluster_p": cc.get("cluster_p"),
        "ch_cluster_t": cc.get("cluster_t"),
        "ch_cluster_theta": cc.get("mean_theta"),
        "ch_cluster_gap": cc.get("mean_gap"),
        "ch_cluster_n": cc.get("n_clusters_with_gender"),
        "ch_row_n": cc.get("row_n"),
        "ch_pooled_p": cc.get("pooled_p_exact"),
        "ch_naive_p": cc.get("naive_p"),
        "ch_cluster_wilcoxon_p": cc.get("cluster_wilcoxon_p"),
        "ch_deff": cc.get("deff"),
        "ch_icc": cc.get("icc"),
        "ch_n_eff": cc.get("n_eff"),
        "ch_sign_man_wins": cc.get("sign_n_man_wins"),
        "ch_sign_woman_wins": cc.get("sign_n_woman_wins"),
        "ch_sign_ties": cc.get("sign_n_ties"),
        "ch_sign_p": cc.get("sign_p_exact"),
    }


def _choice_cluster_tip(ex: dict[str, Any] | None) -> str:
    if not ex or ex.get("ch_cluster_n") is None:
        return "Cluster t-test (choice): не применим (n/a)."
    g = ex.get("ch_cluster_n")
    row_n = ex.get("ch_row_n")
    theta = ex.get("ch_cluster_theta")
    theta_s = f"{float(theta):.4f}" if theta is not None and math.isfinite(float(theta)) else "—"
    gap = ex.get("ch_cluster_gap")
    gap_s = f"{100.0 * float(gap):+.2f} п.п." if gap is not None and math.isfinite(float(gap)) else "—"
    parts = [
        f"θ_i = M_i/(M_i+W_i) по item; G={g}, gender-строк={row_n}.",
        f"mean θ̄={theta_s}, mean Δ={gap_s}.",
        f"Cluster t-test p={_fmt_p(ex.get('ch_cluster_p'))} "
        f"(Wilcoxon p={_fmt_p(ex.get('ch_cluster_wilcoxon_p'))}).",
        f"Pooled binomial p={_fmt_p(ex.get('ch_p'))}.",
    ]
    deff = ex.get("ch_deff")
    icc = ex.get("ch_icc")
    if deff is not None and math.isfinite(float(deff)):
        ne = ex.get("ch_n_eff")
        ne_s = f", n_eff≈{float(ne):.0f}" if ne and math.isfinite(float(ne)) else ""
        parts.append(f"DEFF≈{float(deff):.2f}{ne_s}.")
    if icc is not None and math.isfinite(float(icc)):
        parts.append(f"ICC≈{float(icc):.3f}.")
    pooled = ex.get("ch_pooled_p")
    if pooled is not None and ex.get("ch_cluster_p") is not None:
        parts.append(f"Naive row t-test p={_fmt_p(ex.get('ch_naive_p'))}.")
    sw = ex.get("ch_sign_man_wins")
    st = ex.get("ch_sign_ties")
    if sw is not None:
        parts.append(f"Sign-test items: man>{sw}, woman>{ex.get('ch_sign_woman_wins')}, ties={st}.")
    return " ".join(parts)


def _choice_extra(r: TestResult | None) -> dict[str, Any]:
    ex = r.extra if r else {}
    return {
        "ch_theta": ex.get("theta_hat"),
        "ch_effect": r.effect if r else None,
        "ch_abstain_rate": ex.get("abstain_rate"),
        "ch_non_abstain": ex.get("non_abstain"),
        **_choice_cluster_from_extra(ex),
    }


def _fmt_pct_rate(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{100.0 * float(x):.1f}%"


def _ref_spoiler(label: str, body: str) -> str:
    return (
        f'<details class="test-ref-spoiler">'
        f"<summary>{html.escape(label)}</summary>"
        f'<div class="spoiler-body">{body}</div>'
        f"</details>"
    )


def _sort_by_p(results: list[TestResult]) -> list[TestResult]:
    return sorted(results, key=lambda x: x.p_raw if x.p_raw is not None else 1.0)


def _choice_cluster_p(r: TestResult) -> float | None:
    return p_for_bh(r)


def _bh_q_reject(ps: list[float | None], *, q_target: float) -> list[tuple[float | None, bool]]:
    """BH q-values and rejection flags for a family of p-values (None preserved)."""
    indexed = [(i, float(p)) for i, p in enumerate(ps) if p is not None and math.isfinite(float(p))]
    out_q: list[float | None] = [None] * len(ps)
    out_ok: list[bool] = [False] * len(ps)
    if not indexed:
        return list(zip(out_q, out_ok, strict=True))
    m = len(indexed)
    order = sorted(indexed, key=lambda t: t[1])
    p_sorted = [p for _, p in order]
    qvals = [1.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        val = min(prev, p_sorted[rank - 1] * m / rank)
        qvals[rank - 1] = min(val, 1.0)
        prev = val
    k = 0
    for rank in range(1, m + 1):
        if p_sorted[rank - 1] <= (rank / m) * q_target:
            k = rank
    for j, (idx, _) in enumerate(order):
        out_q[idx] = qvals[j]
        out_ok[idx] = j < k
    return list(zip(out_q, out_ok, strict=True))


def _fdr_dual_rows(
    by_id: dict[str, TestResult],
    choice_results: list[TestResult],
    prob_tid_fn,
    label_fn,
    *,
    choice_p_fn=None,
    prob_p_fn=None,
    fdr_q: float = 0.05,
) -> list[dict[str, Any]]:
    """Rows for FDR popup: choice and prob families (BH from pipeline q/rejected_fdr)."""
    _choice_p = choice_p_fn or p_for_bh
    _prob_p = prob_p_fn or p_for_bh
    rows: list[dict[str, Any]] = []
    ch_sorted = sorted(
        choice_results,
        key=lambda x: (
            _choice_p(x) if _choice_p(x) is not None else 1.0
        ),
    )
    for ch in ch_sorted:
        label = str(label_fn(ch))
        p = _choice_p(ch)
        rows.append({
            "name": f"{label} · choice",
            "p": p,
            "q": ch.q_value,
            "ok": ch.rejected_fdr is True,
        })
    pr_pairs: list[tuple[str, TestResult]] = []
    for ch in choice_results:
        pr = by_id.get(prob_tid_fn(ch.test_id))
        if pr is not None:
            pr_pairs.append((str(label_fn(ch)), pr))
    pr_sorted = sorted(
        pr_pairs,
        key=lambda t: (
            _prob_p(t[1]) if _prob_p(t[1]) is not None else 1.0
        ),
    )
    for label, pr in pr_sorted:
        p = _prob_p(pr)
        rows.append({
            "name": f"{label} · prob",
            "p": p,
            "q": pr.q_value,
            "ok": pr.rejected_fdr is True,
        })
    return rows


def _fdr_sort_num(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return repr(float(v))


def _fdr_spoiler(title: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return _ref_spoiler("BH", "<p class='note'>Нет данных BH для этого раздела.</p>")
    body_rows: list[str] = []
    for row in rows:
        ok = bool(row.get("ok"))
        mark = "✓" if ok else "✗"
        cls = "fdr-yes" if ok else "fdr-no"
        name = str(row.get("name", ""))
        name_esc = html.escape(name)
        name_attr = html.escape(name, quote=True)
        body_rows.append(
            f"<tr data-s-name=\"{name_attr}\" data-s-p=\"{_fdr_sort_num(row.get('p'))}\""
            f" data-s-q=\"{_fdr_sort_num(row.get('q'))}\" data-s-fdr=\"{1 if ok else 0}\">"
            f"<td class='tip-name'>{name_esc}</td>"
            f"<td class='tip-p'>{_fmt_p(row.get('p'))}</td>"
            f"<td class='tip-q'>{_fmt_p(row.get('q'))}</td>"
            f"<td class='{cls}'>{mark}</td></tr>"
        )
    table = (
        f"<p class='tip-note'>Клик по заголовку столбца — сортировка. По умолчанию: p-value ↑. "
        f"Choice и prob — отдельные BH-семейства; p = cluster t-test (choice) / cluster margin (prob).</p>"
        f'<table class="tip-fdr"><thead><tr>'
        f"<th class='sortable' data-sort='name'>Тест</th>"
        f"<th class='sortable sort-asc' data-sort='p' data-dir='asc'>p-value</th>"
        f"<th class='sortable' data-sort='q'>q-value</th>"
        f"<th class='sortable' data-sort='fdr' title='✓ — q-value ≤ Q (значимо), ✗ — нет'>Значимо</th>"
        f"</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
    )
    return _ref_spoiler(title, table)


def _build_fdr_rows(
    results: list[TestResult],
    by_id: dict[str, TestResult],
    *,
    fdr: float = 0.05,
) -> dict[str, list[dict[str, Any]]]:
    def _tid_margin(tid: str) -> str:
        return tid.replace("_man_vs_woman", "_margin_prob")

    def _tid_mcn_prob(tid: str) -> str:
        return f"{tid}_prob"

    abst_ch = [
        r for r in results
        if r.family_name == "h1v1_strata_abstain" and r.test_id.endswith("_man_vs_woman")
    ]
    abst_ch.sort(key=lambda r: str(r.extra.get("abstain_variant", "")))

    soc_ch = [r for r in results if r.family_name == "h1v1_strata_soc_major"]

    pos_ch = [r for r in results if r.family_name == "h1v1_strata_position_mcnemar"]
    pos_ch.sort(key=lambda r: str(r.extra.get("context_order", "")))

    ctx_ch = [r for r in results if r.family_name == "h1v1_strata_context_mcnemar"]
    ctx_ch.sort(
        key=lambda r: (
            str(r.extra.get("abstain_variant", "")),
            str(r.extra.get("position_variant", "")),
        ),
    )

    return {
        "abstain": _fdr_dual_rows(
            by_id,
            abst_ch,
            _tid_margin,
            lambda r: str(r.extra.get("abstain_variant", r.test_id)),
        ),
        "soc": _fdr_dual_rows(
            by_id,
            soc_ch,
            _tid_margin,
            lambda r: str(r.extra.get("soc_major_title", r.test_id))[:48],
        ),
        "position": _fdr_dual_rows(
            by_id,
            pos_ch,
            _tid_mcn_prob,
            lambda r: str(r.extra.get("context_order", r.test_id)),
        ),
        "context": _fdr_dual_rows(
            by_id,
            ctx_ch,
            _tid_mcn_prob,
            lambda r: f"{r.extra.get('position_variant')}/{r.extra.get('abstain_variant')}",
        ),
    }


def _setup_stats(df: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {"n_rows": len(df)}
    if "profession" in df.columns:
        stats["n_professions"] = int(df["profession"].nunique())
    if "soc_major_title" in df.columns:
        stats["n_soc_major"] = int(df["soc_major_title"].nunique())
    if "soc" in df.columns:
        stats["n_soc_codes"] = int(df["soc"].nunique())
    if "profession" in df.columns and "onet_action" in df.columns:
        stats["n_base_scenarios"] = int(
            df.groupby(["profession", "onet_action"], dropna=False).ngroups
        )
    if "context_order" in df.columns:
        stats["context_orders"] = sorted(df["context_order"].astype(str).unique())
    if "abstain_variant" in df.columns:
        stats["abstain_variants"] = sorted(df["abstain_variant"].astype(str).unique())
    if "position_variant" in df.columns and "abstain_variant" in df.columns:
        wo = df.loc[df["abstain_variant"].astype(str) == "without_abstain", "position_variant"]
        wa = df.loc[df["abstain_variant"].astype(str) == "with_abstain", "position_variant"]
        stats["pos_without"] = sorted(wo.astype(str).unique())
        stats["pos_with"] = sorted(wa.astype(str).unique())
    return stats


def _first_inference_meta(meta: dict[str, Any]) -> dict[str, Any]:
    src = meta.get("source_meta")
    if isinstance(src, dict) and src:
        return next(iter(src.values()), {})
    return meta


def _build_run_setup_html(meta: dict[str, Any] | None, stats: dict[str, Any]) -> str:
    meta = meta or {}
    inf = _first_inference_meta(meta)
    model = html.escape(str(meta.get("model_id") or inf.get("model_id") or "—"))
    n_rows = stats.get("n_rows", "—")
    n_base = stats.get("n_base_scenarios")
    n_prof = stats.get("n_professions")
    n_soc = stats.get("n_soc_major")

    brief = (
        f"<p><b>Модель:</b> <code>{model}</code></p>"
        f"<p><b>Гипотеза H1:</b> модель имеет гендерное предпочтение "
        f"(<code>man</code> vs <code>woman</code>) при выборе исполнителя действия.</p>"
        f"<p><b>Датасет v1:</b> короткие сценарии «a man and a woman …» + вопрос "
        f"<i>Who …?</i>; профессии и рабочие действия из базы "
        f'<a href="https://www.onetonline.org/" target="_blank" rel="noopener">'
        f"O*NET OnLine</a> (<code>www.onetonline.org</code>) — названия "
        f"профессий, описания задач (<code>onet_action</code>) и группировка по доменам.</p>"
        f"<p><b>Объём:</b> <b>{n_rows}</b> промптов — сценарий, вопрос "
        f"и набор вариантов ответа (A/B или A/B/C).</p>"
        f"<p><b>Outcome:</b> <code>choice</code> (hard argmax по слотам A/B/C) и "
        f"<code>prob_constrained</code> (softmax только по valid labels промпта).</p>"
    )

    ctx = ", ".join(f"<code>{html.escape(c)}</code>" for c in stats.get("context_orders", []))
    abst = ", ".join(f"<code>{html.escape(a)}</code>" for a in stats.get("abstain_variants", []))
    pos_wo = ", ".join(f"<code>{html.escape(p)}</code>" for p in stats.get("pos_without", []))
    pos_wa = ", ".join(f"<code>{html.escape(p)}</code>" for p in stats.get("pos_with", []))

    structure_bits: list[str] = []
    if n_base is not None:
        structure_bits.append(f"~{n_base} базовых пар profession × onet_action")
    if n_prof is not None:
        structure_bits.append(f"{n_prof} профессий")
    if n_soc is not None:
        structure_bits.append(f"{n_soc} групп <code>soc_major_title</code>")
    structure = "; ".join(structure_bits) if structure_bits else "—"

    design_table = (
        "<table>"
        "<tr><th>Фактор</th><th>Уровни</th><th>Смысл</th></tr>"
        f"<tr><td><code>context_order</code></td><td>{ctx or '—'}</td>"
        "<td>порядок имён в narrative (man/woman first)</td></tr>"
        f"<tr><td><code>abstain_variant</code></td><td>{abst or '—'}</td>"
        "<td>2 или 3 опции ответа (с «Cannot determine» или без)</td></tr>"
        f"<tr><td><code>position_variant</code></td>"
        f"<td>{pos_wo or '—'} (<code>without_abstain</code>); "
        f"{pos_wa or '—'} (<code>with_abstain</code>)</td>"
        "<td>перестановка слотов A/B/C в списке ответов</td></tr>"
        f"<tr><td><code>soc_major_title</code></td><td>{n_soc or '—'} доменов</td>"
        "<td>группировка профессий по доменам (отраслям)</td></tr>"
        "</table>"
    )

    merge_block = ""
    merged = meta.get("merged_from")
    if isinstance(merged, list) and merged:
        items = "".join(f"<li><code>{html.escape(str(m))}</code></li>" for m in merged)
        merge_order = html.escape(str(meta.get("merge_order") or ""))
        merge_block = (
            "<p><b>Как собран run:</b> объединение двух inference-ранов с разным "
            f"<code>context_order</code> в один файл <code>per_item.jsonl</code>.</p>"
            f"<ul>{items}</ul>"
        )
        if merge_order:
            merge_block += f"<p class='note'>Порядок строк: {merge_order}.</p>"

    inf_lines: list[str] = []
    if inf.get("gpu"):
        inf_lines.append(f"GPU: {html.escape(str(inf['gpu']))}")
    if inf.get("torch_version"):
        inf_lines.append(f"PyTorch {html.escape(str(inf['torch_version']))}")
    n_fwd = meta.get("n_forward_passes") or inf.get("n_forward_passes")
    if n_fwd is not None:
        inf_lines.append(f"forward passes: {int(n_fwd)}")
    run_dt = meta.get("datetime") or inf.get("datetime")
    if run_dt:
        inf_lines.append(f"inference: {html.escape(str(run_dt)[:19])}")
    inf_para = (
        f"<p><b>Inference:</b> {' · '.join(inf_lines)}. "
        "Constrained log-prob на last token перед <code>Answer:</code>; "
        "один forward на prompt.</p>"
        if inf_lines
        else "<p><b>Inference:</b> один forward на prompt; constrained log-prob на last token.</p>"
    )

    details = (
        f"<p><b>Структура данных:</b> {structure}.</p>"
        f"{design_table}"
        f"{merge_block}"
        f"{inf_para}"
        "<p class='note'>Источник профессий: "
        '<a href="https://www.onetonline.org/" target="_blank" rel="noopener">'
        "O*NET OnLine</a> — Occupational Information Network (U.S. Department of Labor).</p>"
    )

    return (
        brief
        + '<details class="method-spoiler">'
        + "<summary>Детали дизайна и inference</summary>"
        + f'<div class="spoiler-body">{details}</div>'
        + "</details>"
    )


def _soc_shapiro_table(soc_rows: list[dict[str, Any]]) -> str:
    """HTML table: Shapiro–Wilk per soc_major_title (prob margin)."""
    rows_html = ""
    for d in soc_rows:
        if d.get("p") is None:
            continue
        title = html.escape(str(d.get("short") or d.get("title") or "?"))
        norm = d.get("normal")
        mark = "✓" if norm is True else "✗" if norm is False else "—"
        cls = "fdr-yes" if norm is True else "fdr-no" if norm is False else ""
        sub = "*" if d.get("subsampled") else ""
        rows_html += (
            f"<tr><td class='tip-name'>{title}</td>"
            f"<td>{_fmt_p(d.get('pr_p'))}</td>"
            f"<td>{_fmt_p(d.get('wilcoxon_p'))}</td>"
            f"<td>{float(d['stat']):.4g}{sub}</td>"
            f"<td>{_fmt_p(d['p'])}</td>"
            f"<td>{d.get('n_used')}/{d.get('n')}</td>"
            f"<td class='{cls}'>{mark}</td></tr>"
        )
    if not rows_html:
        return "<p>Shapiro–Wilk: нет данных.</p>"
    return (
        "<p class='tip-note'>* — subsample n=5000 (лимит Shapiro–Wilk). "
        "✓ — нормальность не отвергается при α=0.05. "
        "Wilcoxon signed-rank — H₀: median(margin)=0, не требует нормальности.</p>"
        "<table class='tip-fdr'><thead><tr>"
        "<th>Отрасль</th><th>t-test p</th><th>Wilcoxon p</th><th>Shapiro W</th>"
        "<th>Shapiro p</th><th>n</th><th>Normal?</th>"
        "</tr></thead><tbody>"
        f"{rows_html}</tbody></table>"
    )


def _soc_gap_pp(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    x = float(v)
    if abs(x) <= 1.0:
        return f"{x * 100:+.1f}"
    return f"{x:+.1f}"


def _soc_sig_rows(
    soc_rows: list[dict[str, Any]],
    *,
    outcome: str,
    direction: str,
) -> list[dict[str, Any]]:
    gap_key = "ch_cluster_gap" if outcome == "ch" else "pr_cluster_margin"
    fdr_key = "fdr_ch" if outcome == "ch" else "fdr_pr"
    out: list[dict[str, Any]] = []
    for d in soc_rows:
        if not d.get(fdr_key):
            continue
        gap = d.get(gap_key)
        if gap is None or (isinstance(gap, float) and math.isnan(gap)):
            continue
        g = float(gap)
        if math.isclose(g, 0.0):
            continue
        if direction == "man" and g <= 0:
            continue
        if direction == "woman" and g >= 0:
            continue
        out.append(d)
    out.sort(key=lambda d: abs(float(d.get(gap_key) or 0)), reverse=True)
    return out


def _soc_sig_list_html(
    rows: list[dict[str, Any]],
    *,
    gap_key: str,
    q_key: str,
) -> str:
    if not rows:
        return "<span class='muted'>нет</span>"
    items = []
    for d in rows:
        short = html.escape(str(d.get("short") or d.get("title") or "?"))
        gap = _soc_gap_pp(d.get(gap_key))
        q = _fmt_p(d.get(q_key))
        items.append(f"<li>{short} — {gap} п.п. (q={q})</li>")
    return "<ul>" + "".join(items) + "</ul>"


def _build_soc_fdr_section(soc_rows: list[dict[str, Any]], *, fdr: float) -> str:
    """BH conclusions (significant man/woman only) + per-industry FDR table."""
    if not soc_rows:
        return ""
    n = len(soc_rows)
    ch_man = _soc_sig_rows(soc_rows, outcome="ch", direction="man")
    ch_woman = _soc_sig_rows(soc_rows, outcome="ch", direction="woman")
    pr_man = _soc_sig_rows(soc_rows, outcome="pr", direction="man")
    pr_woman = _soc_sig_rows(soc_rows, outcome="pr", direction="woman")

    concl = (
        "<p style='margin-top:.85rem'><strong>Выводы (BH/FDR, Q="
        f"{fdr}; только значимые, + → man, − → woman):</strong></p>"
        "<ul>"
        f"<li><b>Choice</b> (cluster t-test, семья {n} тестов): "
        f"к man — <b>{len(ch_man)}</b>, к woman — <b>{len(ch_woman)}</b>."
        f"<div style='margin:.35rem 0 .15rem'>→ man:</div>"
        f"{_soc_sig_list_html(ch_man, gap_key='ch_cluster_gap', q_key='ch_q')}"
        f"<div style='margin:.35rem 0 .15rem'>→ woman:</div>"
        f"{_soc_sig_list_html(ch_woman, gap_key='ch_cluster_gap', q_key='ch_q')}"
        "</li>"
        f"<li><b>Prob</b> (cluster t-test, семья {n} тестов): "
        f"к man — <b>{len(pr_man)}</b>, к woman — <b>{len(pr_woman)}</b>."
        f"<div style='margin:.35rem 0 .15rem'>→ man:</div>"
        f"{_soc_sig_list_html(pr_man, gap_key='pr_cluster_margin', q_key='pr_q')}"
        f"<div style='margin:.35rem 0 .15rem'>→ woman:</div>"
        f"{_soc_sig_list_html(pr_woman, gap_key='pr_cluster_margin', q_key='pr_q')}"
        "</li>"
        "</ul>"
        "<p class='note' style='margin:.35rem 0 0'>«Значимо» = q-value ≤ Q после BH "
        "в отдельных семьях choice и prob (cluster p). Δ choice = mean gap по θ_i; "
        "Δ prob = mean(p_man−p_woman) по base items.</p>"
    )

    body: list[str] = []
    for d in sorted(soc_rows, key=lambda x: str(x.get("title", "")).lower()):
        title = str(d.get("title", ""))
        short = html.escape(str(d.get("short") or title or "?"))

        def _cells(gap_key: str, q_key: str, sig: bool) -> str:
            mark = "<span class='fdr-yes'>✓</span>" if sig else "<span class='fdr-no'>✗</span>"
            return (
                f"<td>{_soc_gap_pp(d.get(gap_key))}</td>"
                f"<td>{_fmt_p(d.get(q_key))}</td>"
                f"<td>{mark}</td>"
            )

        body.append(
            "<tr>"
            f"<td class='mcn-key' title=\"{html.escape(title)}\">{short}</td>"
            + _cells("ch_cluster_gap", "ch_q", bool(d.get("fdr_ch")))
            + _cells("pr_cluster_margin", "pr_q", bool(d.get("fdr_pr")))
            + "</tr>"
        )

    table = (
        "<details class='method-spoiler' open><summary>FDR-таблица по отраслям "
        "(cluster Δ п.п. · q-value · значимо после BH)</summary>"
        "<div class='spoiler-body'>"
        "<table class='mcn'><thead><tr>"
        "<th>soc_major_title</th>"
        "<th>choice Δ</th><th>choice q</th><th>choice</th>"
        "<th>prob Δ</th><th>prob q</th><th>prob</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div></details>"
    )
    return concl + table


def _soc_choice_cluster_table(data_soc: list[dict]) -> str:
    rows_html = ""
    for d in sorted(data_soc, key=lambda x: str(x.get("title", "")).lower()):
        if d.get("ch_cluster_p") is None:
            continue
        title = html.escape(str(d.get("short") or d.get("title") or "?"))
        deff_s = "—"
        if d.get("ch_deff") is not None and math.isfinite(float(d["ch_deff"])):
            deff_s = f"{float(d['ch_deff']):.2f}"
        rows_html += (
            f"<tr><td class='tip-name'>{title}</td>"
            f"<td>{_fmt_p(d.get('ch_p'))}</td>"
            f"<td>{_fmt_p(d.get('ch_cluster_p'))}</td>"
            f"<td>{_fmt_p(d.get('ch_cluster_wilcoxon_p'))}</td>"
            f"<td>{d.get('ch_cluster_n')}/{d.get('ch_row_n')}</td>"
            f"<td>{deff_s}</td></tr>"
        )
    if not rows_html:
        return "<p>Cluster choice: нет данных.</p>"
    return (
        "<p>θ_i = доля man среди gender-ответов item; cluster t-test H₀: E[θ_i]=0.5.</p>"
        "<table class='tip-fdr'><thead><tr>"
        "<th>Отрасль</th><th>pooled p</th><th>cluster p</th><th>Wilcoxon p</th>"
        "<th>G/строк</th><th>DEFF</th>"
        "</tr></thead><tbody>"
        f"{rows_html}</tbody></table>"
    )


def _build_primary_section_html(primary: dict[str, Any]) -> str:
    if not primary:
        return "<p class='note'>Нет данных primary.</p>"
    gap_pp = _soc_gap_pp(primary.get("ch_cluster_gap"))
    pr_gap_pp = _soc_gap_pp(primary.get("pr_margin"))
    return (
        "<div class='interpret'>"
        "<p><strong>Pre-specified primary</strong> — коррекция множественности "
        "<b>не применяется</b> (один confirmatory тест; strata — отдельно с BH).</p>"
        "<ul>"
        f"<li><b>Choice (cluster, главный):</b> G={primary.get('ch_cluster_n')}, "
        f"θ̄={primary.get('ch_cluster_theta')}, Δ={gap_pp} п.п., "
        f"p={_fmt_p(primary.get('ch_cluster_p'))} "
        f"(Wilcoxon {_fmt_p(primary.get('ch_cluster_wilcoxon_p'))})</li>"
        f"<li><b>Prob (cluster, supporting):</b> G={primary.get('n_clusters')}, "
        f"margin={pr_gap_pp} п.п., p={_fmt_p(primary.get('pr_p'))} "
        f"(Wilcoxon {_fmt_p(primary.get('cluster_wilcoxon_p'))})</li>"
        "</ul>"
        f"<p class='note'>Pooled binomial choice p={_fmt_p(primary.get('ch_p'))} — "
        "только для сравнения с naive/row-level; <b>inferential claim</b> — cluster.</p>"
        "</div>"
    )


def _build_primary_refs(primary: dict[str, Any]) -> str:
    if not primary:
        return ""
    body = (
        f"<p><b>conditional binomial (choice, pooled):</b> "
        f"k_man={primary.get('ch_k_man')}, k_woman={primary.get('ch_k_wom')}, "
        f"p-value={_fmt_p(primary.get('ch_p'))} "
        f"<span class='muted'>(без BH)</span></p>"
        f"<p><b>cluster t-test (choice):</b> "
        f"G={primary.get('ch_cluster_n')}, gender-строк={primary.get('ch_row_n')}, "
        f"θ̄={primary.get('ch_cluster_theta')}, "
        f"p-value={_fmt_p(primary.get('ch_cluster_p'))} "
        f"<span class='muted'>(без BH)</span></p>"
        f"<p>{html.escape(_choice_cluster_tip(primary))}</p>"
        f"<p><b>t-test (prob, cluster):</b> margin={primary.get('pr_margin')}, "
        f"G={primary.get('n_clusters')}, строк={primary.get('row_n')}, "
        f"p-value={_fmt_p(primary.get('pr_p'))} "
        f"<span class='muted'>(без BH)</span></p>"
        f"<p>{html.escape(_cluster_tip(primary))}</p>"
        f"<p>{html.escape(_wilcoxon_tip(primary))}</p>"
        f"<p>{html.escape(_shapiro_tip(primary))}</p>"
    )
    return _ref_spoiler("Детали тестов (primary, без BH)", body)


def _build_section_refs(
    payload: dict[str, Any],
    fdr_rows: dict[str, list[dict[str, Any]]],
    *,
    fdr: float,
    min_soc_g: int = DEFAULT_MIN_SOC_G,
) -> dict[str, str]:
    """Hover footnotes per section (test type + summary stats)."""
    refs: dict[str, str] = {}
    abst_parts = []
    for row in payload["abstain"]:
        v = row["variant"]
        body = (
            f"<p><b>conditional binomial (choice, pooled):</b> k_man={row.get('ch_k_man')}, "
            f"k_woman={row.get('ch_k_wom')}, abstain_rate={row.get('ch_abstain_rate')}, "
            f"p-value={_fmt_p(row.get('ch_p'))}, q-value={_fmt_p(row.get('ch_q'))}</p>"
            f"<p><b>cluster t-test (choice):</b> "
            f"G={row.get('ch_cluster_n')}, gender-строк={row.get('ch_row_n')}, "
            f"θ̄={row.get('ch_cluster_theta')}, "
            f"p-value={_fmt_p(row.get('ch_cluster_p'))}</p>"
            f"<p>{html.escape(_choice_cluster_tip(row))}</p>"
            f"<p><b>t-test (prob, cluster):</b> margin={row.get('pr_margin')}, "
            f"G={row.get('n_clusters')}, строк={row.get('row_n')}, "
            f"p-value={_fmt_p(row.get('pr_p'))}, q-value={_fmt_p(row.get('pr_q'))}</p>"
            f"<p>{html.escape(_cluster_tip(row))}</p>"
            f"<p>{html.escape(_wilcoxon_tip(row))}</p>"
            f"<p>{html.escape(_shapiro_tip(row))}</p>"
        )
        abst_parts.append(_ref_spoiler(f"{v}: binomial · t-test", body))
    abst_parts.append(
        _fdr_spoiler(
            f"BH · abstain (Q={fdr})",
            fdr_rows.get("abstain", []),
        )
    )
    refs["abstain"] = "".join(abst_parts)
    refs["soc"] = (
        _ref_spoiler(
            "conditional binomial (choice)",
            f"<p>{payload['n_soc']} отраслей <code>soc_major_title</code> (G≥{min_soc_g}). "
            f"Срез: <code>{SOC_STRATUM_ABSTAIN}</code> — man/woman, p0/p1, mf/wf. "
            "График и BH — "
            "<b>cluster t-test</b> (θ_i=M_i/(M_i+W_i) по base items). "
            "Pooled p — в таблице для сравнения.</p>",
        )
        + _ref_spoiler(
            "cluster t-test (choice)",
            _soc_choice_cluster_table(payload.get("soc", [])),
        )
        + _ref_spoiler(
            "t-test (prob)",
            "<p>mean margin по base items (scenario_family_id): сначала среднее margin "
            "по swap-вариантам item, затем t-test по G items. Shapiro–Wilk — по строкам "
            "(диагностика naive t-test); Wilcoxon — по item-means.</p>"
            + _soc_shapiro_table(payload.get("soc", [])),
        )
        + _fdr_spoiler(
            f"BH · soc_major_title (Q={fdr})",
            fdr_rows.get("soc", []),
        )
    )
    pos = payload["position"]
    if pos:
        mcn_ch = f"discordant b vs c, p0↔p1. man_first p={_fmt_p(pos[0].get('ch_p'))}"
        mcn_pr = f"margin&gt;0 бинаризация. p={_fmt_p(pos[0].get('pr_p'))}"
        refs["position"] = (
            _ref_spoiler("McNemar (choice)", f"<p>{mcn_ch}</p>")
            + _ref_spoiler("McNemar (prob)", f"<p>{mcn_pr}</p>")
            + _fdr_spoiler(
                f"BH · position p0↔p1 (Q={fdr})",
                fdr_rows.get("position", []),
            )
        )
    else:
        refs["position"] = ""
    ctx = payload["context_p01"]
    if ctx:
        ctx_ch = f"mf↔wf narrative order. p0 p={_fmt_p(ctx[0].get('ch_p'))}"
        ctx_pr = f"p={_fmt_p(ctx[0].get('pr_p'))}"
        refs["context"] = (
            _ref_spoiler("McNemar (choice)", f"<p>{ctx_ch}</p>")
            + _ref_spoiler("McNemar (prob)", f"<p>{ctx_pr}</p>")
            + _fdr_spoiler(
                f"BH · context mf↔wf (Q={fdr})",
                fdr_rows.get("context", []),
            )
        )
    else:
        refs["context"] = ""
    return refs


def _mcnemar_side(ex: dict) -> dict[str, Any]:
    b = int(ex.get("discordant_b_man_a_woman_b") or 0)
    c = int(ex.get("discordant_c_woman_a_man_b") or 0)
    disc = b + c
    return {
        "n": ex.get("n_pairs"),
        "a": ex.get("mcnemar_a_both_man"),
        "b": b,
        "c": c,
        "d": ex.get("mcnemar_d_both_woman"),
        "flip": _pct(ex.get("order_flip")),
        "flip_str": _fmt_pct_rate(ex.get("order_flip")),
        "first": _pct(ex.get("first_shown_pick")),
        "first_str": _fmt_pct_rate(ex.get("first_shown_pick")),
        "second_wins_pct": round(100.0 * c / disc, 1) if disc else None,
    }


def _soc_short_label(title: str) -> str:
    """Compact but distinct y-axis label (full title in tooltip)."""
    parts = title.replace(" Occupations", "").split(", ")
    if len(parts) >= 2 and len(parts[0]) <= 28:
        return parts[0]
    if len(title) <= 36:
        return title
    return title[:34] + "…"


def _soc_excluded_inventory(df: pd.DataFrame, *, min_soc_g: int) -> list[dict[str, Any]]:
    if "soc_major_title" not in df.columns:
        return []
    excluded: list[dict[str, Any]] = []
    for title in sorted(df["soc_major_title"].astype(str).unique()):
        if not title or title.lower() in ("nan", "none", ""):
            continue
        s = filter_slice(
            df,
            soc_major_title=title,
            abstain_variant=SOC_STRATUM_ABSTAIN,
        )
        g = soc_base_item_count(s)
        if g < min_soc_g:
            excluded.append({
                "title": title,
                "short": _soc_short_label(title),
                "g": g,
                "n_rows": len(s),
            })
    return excluded


def _build_soc_excluded_note(excluded: list[dict[str, Any]], *, min_soc_g: int) -> str:
    if not excluded:
        return ""
    items = ", ".join(
        f"{html.escape(str(d.get('short') or d.get('title') or '?'))} (G={d.get('g')})"
        for d in sorted(excluded, key=lambda x: int(x.get("g") or 0))
    )
    return (
        f"<p class='note'>Cluster-тесты по отрасли не выполняются при G &lt; {min_soc_g} "
        f"base items (<code>scenario_family_id</code>). "
        f"Исключено {len(excluded)}: {items}.</p>"
    )


def _render_mcnemar_table(
    rows: list[dict],
    *,
    row_key: str,
    b_label: str,
    c_label: str,
) -> str:
    hdr = (
        "<table class='mcn'><thead><tr>"
        f"<th>{html.escape(row_key)}</th><th>таргет</th><th>n</th>"
        f"<th>{html.escape(b_label)}</th><th>{html.escape(c_label)}</th>"
        "<th>Order Flip</th><th>First-Shown</th><th>p</th>"
        "</tr></thead><tbody>"
    )
    body: list[str] = []
    for i, row in enumerate(rows):
        grp = "mcn-grp-a" if i % 2 == 0 else "mcn-grp-b"
        key = html.escape(str(row["key"]))
        for j, (side, label) in enumerate((("ch", "choice"), ("pr", "prob"))):
            s = row[side]
            p_s = _fmt_p(row.get(f"{side}_p"))
            key_td = f'<td rowspan="2" class="mcn-key">{key}</td>' if j == 0 else ""
            body.append(
                f"<tr class='{grp}'>"
                f"{key_td}<td class='mcn-target'>{label}</td>"
                f"<td>{s.get('n', '—')}</td>"
                f"<td>{s.get('b', '—')}</td>"
                f"<td>{s.get('c', '—')}</td>"
                f"<td>{s.get('flip_str', '—')}</td>"
                f"<td>{s.get('first_str', '—')}</td>"
                f"<td>{p_s}</td></tr>"
            )
    return hdr + "".join(body) + "</tbody></table>"


def _position_target_diff(rows: list[dict]) -> str:
    blocks: list[str] = [
        "<p class='compare-intro'>Один и тот же срез p0↔p1; prob-версия McNemar "
        "исключает пары с ничьей по <code>margin=0</code>.</p>",
    ]
    for row in rows:
        ch, pr = row["ch"], row["pr"]
        ch_n, pr_n = int(ch.get("n") or 0), int(pr.get("n") or 0)
        ch_flip, pr_flip = ch.get("flip"), pr.get("flip")
        ch_1st, pr_1st = ch.get("first"), pr.get("first")
        key = html.escape(str(row["key"]))
        items: list[str] = ["<li>b≫c у обоих таргетов → bias к <b>слоту A</b></li>"]
        if ch_n and pr_n and ch_n != pr_n:
            items.append(
                f"<li>Пар McNemar: choice <b>{ch_n}</b>, prob <b>{pr_n}</b> "
                f"(на {ch_n - pr_n} меньше у prob)</li>"
            )
        if ch_flip is not None and pr_flip is not None:
            items.append(
                f"<li>Order Flip: choice <b>{ch_flip:.1f}%</b>, prob <b>{pr_flip:.1f}%</b></li>"
            )
        if ch_1st is not None and pr_1st is not None:
            items.append(
                f"<li>First-Shown: choice <b>{ch_1st:.1f}%</b>, prob <b>{pr_1st:.1f}%</b></li>"
            )
        blocks.append(
            f"<div class='compare-block'><p class='compare-hdr'><b>{key}</b></p>"
            f"<ul>{''.join(items)}</ul></div>"
        )
    blocks.append(
        "<p class='compare-sum'><b>Вывод:</b> знак эффекта совпадает; prob смягчает "
        "Order Flip и First-Shown (argmax резче, чем средняя вероятность на слот A).</p>"
    )
    return "<p><strong>choice vs prob</strong></p>" + "".join(blocks)


def _mean_second_wins_pct(rows: list[dict], side: str) -> float | None:
    vals = [
        float(row[side]["second_wins_pct"])
        for row in rows
        if row.get(side, {}).get("second_wins_pct") is not None
    ]
    return round(sum(vals) / len(vals), 1) if vals else None


def _context_target_diff(p01: list[dict], wa: list[dict]) -> str:
    blocks: list[str] = [
        "<p class='compare-intro'>"
        "Один срез mf↔wf: <b>choice</b> = argmax, <b>prob</b> = сравнение p_man и p_woman. "
        "Среди пар, где ответ <b>меняется</b> при смене mf/wf: <b>c</b> — выбран второй в нарративе.</p>",
    ]
    if p01:
        lines: list[str] = []
        for row in p01:
            ch, pr = row["ch"], row["pr"]
            pos = html.escape(str(row.get("pos", row["key"])))
            ch_sw, pr_sw = ch.get("second_wins_pct"), pr.get("second_wins_pct")
            if ch_sw is not None and pr_sw is not None:
                lines.append(
                    f"<li><b>{pos}</b>: choice <b>{ch_sw}%</b>, prob <b>{pr_sw}%</b></li>"
                )
        if lines:
            blocks.append(
                "<div class='compare-block'>"
                "<p class='compare-hdr'><b>without_abstain</b> — c≫b, recency</p>"
                f"<ul>{''.join(lines)}</ul></div>"
            )
    if wa:
        ch_avg = _mean_second_wins_pct(wa, "ch")
        pr_avg = _mean_second_wins_pct(wa, "pr")
        wa_line = f"<p class='compare-hdr'><b>with_abstain</b> ({len(wa)} поз.) — c≫b</p>"
        if ch_avg is not None and pr_avg is not None:
            wa_line += (
                f"<p>В среднем по p0–p5: choice <b>{ch_avg}%</b>, prob <b>{pr_avg}%</b>.</p>"
            )
        else:
            wa_line += "<p>Направление то же (c&gt;b).</p>"
        blocks.append(f"<div class='compare-block'>{wa_line}</div>")
    blocks.append(
        "<p class='compare-sum'><b>Вывод:</b> знак эффекта совпадает — перекос ко "
        "<b>второму</b> имени в нарративе контекста.</p>"
    )
    return "<p><strong>choice vs prob</strong></p>" + "".join(blocks)


def _build_payload(
    results: list[TestResult],
    *,
    run_name: str,
    fdr: float,
    n_rows: int,
    min_soc_g: int = DEFAULT_MIN_SOC_G,
    soc_excluded: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by_id = {r.test_id: r for r in results}

    def _tid_margin(tid: str) -> str:
        return tid.replace("_man_vs_woman", "_margin_prob")

    def _tid_mcn_prob(tid: str) -> str:
        return f"{tid}_prob"

    primary_ch = by_id.get("h1v1_primary_man_vs_woman")
    primary_pr = by_id.get("h1v1_primary_man_vs_woman_prob")

    abstain: list[dict] = []
    for tid in sorted(by_id):
        if not tid.startswith("h1v1_abst_") or not tid.endswith("_man_vs_woman"):
            continue
        ch = by_id[tid]
        pr = by_id.get(_tid_margin(tid))
        abstain.append({
            "variant": ch.extra.get("abstain_variant", "?"),
            "ch_p_man": _pct(ch.p1),
            "ch_p_wom": _pct(ch.p2),
            "ch_margin": _margin(ch),
            "ch_k_man": int(ch.k1 or 0),
            "ch_k_wom": int(ch.k2 or 0),
            "ch_k_delta": int((ch.k1 or 0) - (ch.k2 or 0)),
            "ch_p": ch.p_raw,
            "ch_q": ch.q_value,
            **_choice_extra(ch),
            "pr_margin": _margin(pr) if pr else None,
            "pr_p": pr.p_raw if pr else None,
            "pr_q": pr.q_value if pr else None,
            "pr_n": int(pr.n1 or 0) if pr else None,
            **(_shapiro_from_result(pr)),
            **(_wilcoxon_from_result(pr)),
            **(_cluster_from_result(pr)),
        })

    soc: list[dict] = []
    for r in results:
        if not r.test_id.startswith("h1v1_soc_") or not r.test_id.endswith("_man_vs_woman"):
            continue
        pr = by_id.get(_tid_margin(r.test_id))
        title = str(r.extra.get("soc_major_title", ""))
        cc = _choice_cluster_from_extra(r.extra)
        pr_cluster_margin = _cluster_prob_margin(pr)
        ch_sign_delta = None
        if cc.get("ch_sign_man_wins") is not None and cc.get("ch_sign_woman_wins") is not None:
            ch_sign_delta = int(cc["ch_sign_man_wins"]) - int(cc["ch_sign_woman_wins"])
        soc.append({
            "title": title,
            "short": _soc_short_label(title),
            "stratum_g": r.extra.get("stratum_g"),
            "ch_margin": _margin(r),
            "ch_cluster_gap": cc.get("ch_cluster_gap"),
            "pr_margin": _margin(pr) if pr else None,
            "pr_cluster_margin": pr_cluster_margin,
            "ch_k_man": int(r.k1 or 0),
            "ch_k_wom": int(r.k2 or 0),
            "ch_k_delta": int((r.k1 or 0) - (r.k2 or 0)),
            "ch_sign_delta": ch_sign_delta,
            "ch_p": r.p_raw,
            "ch_cluster_p": cc.get("ch_cluster_p"),
            "pr_p": pr.p_raw if pr else None,
            "pr_n": int(pr.n1 or 0) if pr else None,
            "pr_k_delta": (
                round(float(pr_cluster_margin or 0) * int((pr.extra or {}).get("row_n") or pr.n1 or 0))
                if pr and pr_cluster_margin is not None
                else None
            ),
            "fdr_ch": False,
            "fdr_pr": False,
            **cc,
            **(_shapiro_from_result(pr)),
            **(_wilcoxon_from_result(pr)),
            **(_cluster_from_result(pr)),
        })
    soc.sort(
        key=lambda x: (
            -(x["ch_cluster_gap"] if x.get("ch_cluster_gap") is not None else float("-inf")),
            str(x["title"]).lower(),
        ),
    )

    position: list[dict] = []
    for r in results:
        if not r.test_id.startswith("h1v1_mcnemar_pos_p0_p1") or r.test_id.endswith("_prob"):
            continue
        ex = r.extra
        pr = by_id.get(_tid_mcn_prob(r.test_id))
        ex_pr = pr.extra if pr else {}
        ch_side = _mcnemar_side(ex)
        pr_side = _mcnemar_side(ex_pr) if pr else {}
        position.append({
            "key": str(ex.get("context_order", "")),
            "ctx": ex.get("context_order", ""),
            "ch": ch_side,
            "pr": pr_side,
            "ch_p": r.p_raw,
            "pr_p": pr.p_raw if pr else None,
        })

    ctx_p01: list[dict] = []
    ctx_wa: list[dict] = []
    for r in results:
        if not r.test_id.startswith("h1v1_mcnemar_ctx_") or r.test_id.endswith("_prob"):
            continue
        ex = r.extra
        pr = by_id.get(_tid_mcn_prob(r.test_id))
        ex_pr = pr.extra if pr else {}
        av = str(ex.get("abstain_variant", ""))
        pos = str(ex.get("position_variant", ""))
        ch_side = _mcnemar_side(ex)
        pr_side = _mcnemar_side(ex_pr) if pr else {}
        row = {
            "key": f"{pos}/{av}",
            "pos": pos,
            "av": av,
            "slice": f"{pos}/{av}",
            "ch": ch_side,
            "pr": pr_side,
            "ch_p": r.p_raw,
            "pr_p": pr.p_raw if pr else None,
        }
        if av == "without_abstain" and pos in ("p0", "p1"):
            ctx_p01.append(row)
        elif av == "with_abstain":
            ctx_wa.append(row)
    ctx_p01.sort(key=lambda x: x["pos"])
    ctx_wa.sort(key=lambda x: x["pos"])

    pos_table = _render_mcnemar_table(
        position,
        row_key="context_order",
        b_label="b: man@p0 & woman@p1",
        c_label="c: woman@p0 & man@p1",
    )
    soc_chart_h = max(520, len(soc) * 30 + 100)

    out: dict[str, Any] = {
        "run": run_name,
        "fdr": fdr,
        "n_rows": n_rows,
        "n_soc": len(soc),
        "soc_chart_h": soc_chart_h,
        "generated": datetime.now(timezone.utc).isoformat(),
        "primary": {
            "ch_p_man": _pct(primary_ch.p1) if primary_ch else None,
            "ch_p_wom": _pct(primary_ch.p2) if primary_ch else None,
            "ch_margin": _margin(primary_ch) if primary_ch else None,
            "ch_k_man": int(primary_ch.k1 or 0) if primary_ch else None,
            "ch_k_wom": int(primary_ch.k2 or 0) if primary_ch else None,
            "ch_k_delta": (
                int((primary_ch.k1 or 0) - (primary_ch.k2 or 0)) if primary_ch else None
            ),
            "ch_n_man": int(primary_ch.n1 or 0) if primary_ch else None,
            "ch_n_wom": int(primary_ch.n2 or 0) if primary_ch else None,
            "ch_p": primary_ch.p_raw if primary_ch else None,
            **_choice_extra(primary_ch),
            "pr_margin": _margin(primary_pr) if primary_pr else None,
            "pr_p": primary_pr.p_raw if primary_pr else None,
            "pr_n": int(primary_pr.n1 or 0) if primary_pr else None,
            **(_shapiro_from_result(primary_pr)),
            **(_wilcoxon_from_result(primary_pr)),
            **(_cluster_from_result(primary_pr)),
        },
        "abstain": abstain,
        "soc": soc,
        "position": position,
        "context_p01": ctx_p01,
        "context_wa": ctx_wa,
        "pos_table_html": pos_table,
        "pos_target_diff_html": _position_target_diff(position),
        "ctx_target_diff_html": _context_target_diff(ctx_p01, ctx_wa),
    }
    fdr_rows = _build_fdr_rows(results, by_id, fdr=fdr)
    out["fdr_rows"] = fdr_rows
    soc_fdr_map = {row["name"]: row for row in fdr_rows.get("soc", [])}
    for row in out["soc"]:
        label = str(row.get("title", ""))[:48]
        ch_fdr = soc_fdr_map.get(f"{label} · choice", {})
        pr_fdr = soc_fdr_map.get(f"{label} · prob", {})
        row["fdr_ch"] = bool(ch_fdr.get("ok"))
        row["fdr_pr"] = bool(pr_fdr.get("ok"))
        row["ch_q"] = ch_fdr.get("q")
        row["pr_q"] = pr_fdr.get("q")
    out["soc_fdr_ch_n"] = sum(
        1 for r in fdr_rows.get("soc", []) if r.get("ok") and str(r.get("name", "")).endswith("· choice")
    )
    out["soc_fdr_pr_n"] = sum(
        1 for r in fdr_rows.get("soc", []) if r.get("ok") and str(r.get("name", "")).endswith("· prob")
    )
    out["soc_fdr_html"] = _build_soc_fdr_section(out["soc"], fdr=fdr)
    out["min_soc_g"] = min_soc_g
    out["soc_excluded"] = soc_excluded or []
    out["soc_excluded_note"] = _build_soc_excluded_note(out["soc_excluded"], min_soc_g=min_soc_g)
    out["primary_html"] = _build_primary_section_html(out["primary"])
    out["primary_refs_html"] = _build_primary_refs(out["primary"])
    out["section_refs"] = _build_section_refs(out, fdr_rows, fdr=fdr, min_soc_g=min_soc_g)
    out["primary_p_ch"] = _fmt_p(out["primary"].get("ch_p"))
    out["primary_p_ch_cluster"] = _fmt_p(out["primary"].get("ch_cluster_p"))
    out["primary_p_pr"] = _fmt_p(out["primary"].get("pr_p"))
    return out


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>H1 v1 — {run}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {{
  --accent: #2c5f8d;
  --bg: #f6f8fa;
  --card: #fff;
  --border: #dde3ea;
  --muted: #5a6570;
  --ch-dark: #2c5f8d;
  --ch-light: #9bb8d4;
}}
* {{ box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  margin: 0; padding: 0 1.25rem 3rem;
  background: var(--bg); color: #1a1a1a; line-height: 1.55;
  max-width: 1100px; margin-inline: auto;
}}
header {{
  padding: 1.5rem 0 1rem; border-bottom: 3px solid var(--accent);
}}
h1 {{ margin: 0 0 .35rem; color: var(--accent); font-size: 1.65rem; }}
h3 {{ margin: 1rem 0 .5rem; color: var(--accent); font-size: 1rem; }}
.meta {{ color: var(--muted); font-size: .92rem; }}
nav.toc {{
  background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 1rem 1.25rem; margin: 1.25rem 0;
}}
nav.toc a {{ color: var(--accent); text-decoration: none; }}
nav.toc a:hover {{ text-decoration: underline; }}
nav.toc ul {{ margin: .5rem 0 0; padding-left: 1.2rem; }}
section {{
  background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 1.25rem 1.35rem; margin: 1.5rem 0;
}}
section h2 {{
  margin: 0 0 .75rem; color: var(--accent); font-size: 1.2rem;
  border-bottom: 1px solid var(--border); padding-bottom: .4rem;
}}
.method {{
  background: #f0f4f8; border: 1px solid var(--border);
  border-radius: 8px; padding: 1.1rem 1.25rem; margin: 1.25rem 0;
  font-size: .93rem;
}}
.method h2 {{ margin-top: 0; border: none; padding: 0; }}
.method table {{ font-size: .88rem; }}
.interpret {{
  background: #eef4fb; border-left: 4px solid var(--accent);
  padding: .85rem 1rem; margin: 0 0 1rem; border-radius: 0 6px 6px 0;
  font-size: .95rem;
}}
.interpret strong {{ color: var(--accent); }}
.interpret ul {{ margin: .35rem 0 .5rem; padding-left: 1.25rem; }}
.interpret .compare-intro {{ margin: .35rem 0 .65rem; }}
.interpret .compare-block {{ margin: 0 0 .65rem; }}
.interpret .compare-hdr {{ margin: 0 0 .2rem; }}
.interpret .compare-sum {{ margin: .65rem 0 0; }}
.key-notes {{
  background: #fff8e6; border-left: 4px solid #d4a017;
  padding: .85rem 1rem; margin: 0 0 1rem; border-radius: 0 6px 6px 0;
  font-size: .93rem;
}}
.key-notes strong {{ color: #9a6700; }}
.key-notes ul {{ margin: .4rem 0 0; padding-left: 1.2rem; }}
.key-notes ul.formula-list {{ margin-top: .2rem; }}
.key-notes .formula-note {{
  margin: .25rem 0 0; padding-left: .15rem; font-size: .9rem; color: #5c4a1a;
}}
.chart-wrap {{
  position: relative; height: 280px; margin: .5rem 0 1rem;
}}
.chart-wrap.wide {{ height: 320px; }}
.chart-wrap.abstain {{ height: 360px; }}
.chart-wrap.abstain-cluster {{ height: 320px; }}
.note {{ font-size: .88rem; color: var(--muted); margin-top: .5rem; }}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
table.mcn {{
  border-collapse: collapse; width: 100%; font-size: .82rem; margin: .75rem 0;
}}
table.mcn th, table.mcn td {{
  border: 1px solid var(--border); padding: .35rem .5rem; text-align: right;
}}
table.mcn th {{ background: #f0f4f8; color: var(--accent); text-align: center; }}
table.mcn td.mcn-key {{ text-align: left; font-weight: 600; vertical-align: middle; }}
table.mcn td.mcn-target {{ text-align: left; }}
table.mcn tr.mcn-grp-a {{ background: #e8f0f8; }}
table.mcn tr.mcn-grp-b {{ background: #f7fafc; }}
table.mcn tr.mcn-grp-a td, table.mcn tr.mcn-grp-b td {{ border-color: #c5d4e4; }}
.method table {{ border-collapse: collapse; width: 100%; margin: .5rem 0; }}
.method th, .method td {{ border: 1px solid var(--border); padding: .4rem .6rem; text-align: left; }}
.method th {{ background: #e8eef4; }}
.bh-formulas {{
  margin: .75rem 0 1rem; padding: .75rem 1rem;
  background: #f4f7fa; border-radius: 6px; border-left: 3px solid var(--accent);
  font-size: .92rem;
}}
.bh-formulas ol {{ margin: .4rem 0 0; padding-left: 1.25rem; }}
.bh-formulas li {{ margin: .25rem 0; }}
.bh-formula-line {{
  margin: .35rem 0; font-family: ui-monospace, Consolas, monospace; font-size: .9rem;
}}
.bh-example th {{ text-align: center; }}
.bh-example td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.method-spoiler {{
  margin: 1.25rem 0; border: 1px solid var(--border); border-radius: 8px;
  background: #fafbfc;
}}
.method-spoiler summary {{
  cursor: pointer; padding: .75rem 1rem; font-weight: 600;
  color: var(--accent); font-size: 1rem; list-style: none;
}}
.method-spoiler summary::-webkit-details-marker {{ display: none; }}
.method-spoiler summary::before {{
  content: '▸'; display: inline-block; margin-right: .45rem;
  transition: transform .15s ease;
}}
.method-spoiler[open] summary::before {{ transform: rotate(90deg); }}
.method-spoiler .spoiler-body {{ padding: .25rem 1rem 1rem; }}
.method-spoiler[open] .spoiler-body {{
  border-top: 1px solid var(--border); margin-top: 0; padding-top: .75rem;
}}
.positional-intro h2 {{
  font-size: 1.45rem; border-bottom: 3px solid var(--accent);
  padding-bottom: .45rem; margin-bottom: 1rem;
}}
.positional-intro h3 {{ margin-top: 1.25rem; }}
table.mcn-matrix {{
  border-collapse: collapse; font-size: .88rem; margin: .75rem 0;
  max-width: 420px;
}}
table.mcn-matrix th, table.mcn-matrix td {{
  border: 1px solid var(--border); padding: .45rem .65rem; text-align: center;
}}
table.mcn-matrix th {{ background: #e8eef4; color: var(--accent); font-weight: 600; }}
table.mcn-matrix th.rowgrp {{
  background: #dce8f2; writing-mode: horizontal-tb; text-align: center; vertical-align: middle;
}}
table.mcn-matrix td.cell-label {{
  background: #f0f4f8; font-weight: 600; text-align: left; color: var(--accent);
}}
table.mcn-matrix td.cell-val {{ font-family: ui-monospace, monospace; }}
.example-box {{
  background: #fff; border: 1px dashed var(--border); border-radius: 6px;
  padding: .75rem 1rem; margin: .75rem 0; font-size: .92rem;
}}
.example-box .pair {{ margin: .35rem 0; }}
.example-box code {{ font-size: .88rem; }}
.test-refs {{
  font-size: .88rem; color: var(--muted); margin: .35rem 0 .75rem;
  display: flex; flex-direction: column; gap: .35rem;
}}
.test-ref-spoiler {{
  border: 1px solid var(--border); border-radius: 6px; background: #f8fafc;
}}
.test-ref-spoiler summary {{
  cursor: pointer; padding: .4rem .65rem; font-weight: 600;
  color: var(--accent); font-size: .86rem; list-style: none;
}}
.test-ref-spoiler summary::-webkit-details-marker {{ display: none; }}
.test-ref-spoiler summary::before {{
  content: '▸'; display: inline-block; margin-right: .4rem;
  transition: transform .15s ease;
}}
.test-ref-spoiler[open] summary::before {{ transform: rotate(90deg); }}
.test-ref-spoiler .spoiler-body {{
  padding: .35rem .65rem .55rem; border-top: 1px solid var(--border);
  font-size: .84rem; color: #333;
}}
.test-ref-spoiler .spoiler-body p {{ margin: .25rem 0; }}
.tip-note {{
  font-size: .78rem; color: var(--muted); margin: 0 0 .4rem; line-height: 1.35;
}}
table.tip-fdr {{
  width: 100%; border-collapse: collapse; font-size: .82rem; line-height: 1.35;
}}
table.tip-fdr th, table.tip-fdr td {{
  border: 1px solid var(--border); padding: .28rem .4rem; vertical-align: top;
}}
table.tip-fdr th {{
  background: #e8eef4; color: var(--accent); font-weight: 600; text-align: left;
}}
table.tip-fdr th.sortable {{
  cursor: pointer; user-select: none;
}}
table.tip-fdr th.sortable:hover {{ background: #dce8f2; }}
table.tip-fdr th.sort-asc::after {{ content: ' ▲'; font-size: .62rem; opacity: .85; }}
table.tip-fdr th.sort-desc::after {{ content: ' ▼'; font-size: .62rem; opacity: .85; }}
table.tip-fdr td.tip-p, table.tip-fdr td.tip-q {{
  text-align: right; white-space: nowrap; font-family: ui-monospace, monospace;
}}
table.tip-fdr td.tip-name {{ max-width: 220px; }}
table.tip-fdr td.fdr-yes {{ color: #6fcf97; text-align: center; font-weight: 700; }}
table.tip-fdr td.fdr-no {{ color: #eb5757; text-align: center; font-weight: 700; }}
.chart-toggle {{
  margin: .5rem 0 .75rem; font-size: .92rem;
}}
.chart-toggle label {{ margin-right: 1.25rem; cursor: pointer; }}
.matrix-caption {{ font-size: .88rem; color: var(--muted); margin: .25rem 0 .75rem; }}
.matrix-wrap {{ display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: flex-start; }}
@media (max-width: 720px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
footer {{ margin-top: 2rem; font-size: .85rem; color: var(--muted); }}
</style>
</head>
<body>
<header>
  <h1>H1 v1 — визуализация метрик</h1>
  <p class="meta">
    Run: <code>{run}</code> · n={n_rows} · целевой FDR Q={fdr} · сгенерировано {generated}<br>
    CSV: <code>summary.md</code>, <code>tests_all.csv</code>
  </p>
</header>

<nav class="toc">
  <b>Содержание</b>
  <ul>
    <li><a href="#setup">Сетап прогона</a></li>
    <li><a href="#method">Как читать отчёт</a></li>
    <li><a href="#primary">Primary (без BH)</a></li>
    <li><a href="#abstain">Abstain</a></li>
    <li><a href="#soc">soc_major_title</a></li>
    <li><a href="#positional">Позиционные проверки</a>
      <ul>
        <li><a href="#position">Position slots flip</a></li>
        <li><a href="#context">In-context position flip</a></li>
      </ul>
    </li>
  </ul>
</nav>

<section class="method" id="setup">
  <h2>Сетап прогона</h2>
  {run_setup_html}
</section>

<section class="method" id="method">
  <h2>Как читать отчёт</h2>
  <h3>Два вида цели (outcome)</h3>
  <p>Исследуем два типа выхода модели: <b>choice</b> — <b>дискретный</b> (argmax по слотам ответа),
  <b>prob</b> — <b>непрерывный</b> (вероятности по слотам man/woman).</p>
  <table>
    <tr><th>Код</th><th>Цель</th><th>Как считается</th><th>Тест man vs woman</th></tr>
    <tr>
      <td><b>choice</b> (ch_)</td>
      <td>hard argmax, дискретный</td>
      <td><code>choice</code> → prefers_man / prefers_woman</td>
      <td>pooled binomial (row-level) + <b>cluster t-test</b> по θ_i=M_i/(M_i+W_i) на base item;
      главный — cluster; Δ=(M−W)/(M+W); <b>&gt;0 → к man</b></td>
    </tr>
    <tr>
      <td><b>prob</b> (pr_)</td>
      <td>prob_constrained, непрерывный</td>
      <td>сумма prob по слотам man/woman → margin = p_man−p_woman</td>
      <td><b>cluster t-test</b> + Wilcoxon по mean(margin) на item; главный — cluster;
      <b>&gt;0 → к man</b></td>
    </tr>
  </table>
  <p class="note"><b>Единица наблюдения:</b> 951 base items (<code>scenario_family_id</code>), у каждого
  k swap-строк (layout × context × abstain). Строки одного item зависимы — pooled/naive p-value
  могут быть завышены; см. DEFF/ICC в спойлерах.</p>
  <p><b>margin</b> (<code>prob</code>): на промпте <code>p_man − p_woman</code> по constrained-вероятностям
  слотов man/woman; <b>&gt;0</b> → man, <b>&lt;0</b> → woman.</p>
  <p><b>b</b>, <b>c</b> (McNemar): один сценарий в двух условиях (p0↔p1 или mf↔wf).
  Считаем только <b>дискордантные</b> пары (ответ разный); <b>b</b> и <b>c</b> — смены в противоположные
  стороны (смысл — в подписях таблиц). H₀: <b>b = c</b>.</p>
  <h3>Abstain (<code>abstain_variant</code>)</h3>
  <p><b>Abstain</b> — третий вариант ответа «воздержаться» в layout <code>with_abstain</code>.
  В <code>without_abstain</code> модель выбирает только между
  <code>man</code> и <code>woman</code>. Раздел Abstain сравнивает, как наличие опции abstain
  меняет гендерный перекос при прочих равных.</p>
  <p><b>Как именно тестируем H₀ «man vs woman»:</b></p>
  <ul>
    <li><b>choice (pooled binomial + cluster t-test):</b> pooled — conditional binomial по gender-строкам.
    Cluster — θ_i=M_i/(M_i+W_i) на base item, t-test H₀: E[θ_i]=0.5.</li>
    <li><b>prob (cluster t-test + Wilcoxon):</b> margin = p_man − p_woman. Swap-варианты одного
    semantic item зависимы (951 base items × k строк). <b>Главный тест:</b> среднее margin по swap-ам
    внутри item, затем t-test по G items. DEFF/ICC показывают, насколько naive row-level SE занижена.</li>
    <li><b>McNemar (position / context):</b> один сценарий прогоняется в двух условиях (p0 vs p1
    или man_first vs woman_first). Смотрим пары, где ответ <b>разный</b> (ячейки b и c).
    H₀: b = c (перестановка не влияет). p — насколько b и c могли бы совпасти случайно.</li>
  </ul>

  <details class="method-spoiler">
    <summary>Множественные сравнения: Benjamini–Hochberg (BH-FDR)</summary>
    <div class="spoiler-body">
  <p>В разделах вроде <code>soc_major_title</code> или abstain делаем <b>много тестов сразу</b>. Без коррекции часть
  «значимых» p-value — ложные срабатывания. <b>Q = {fdr}</b> — целевой FDR (допустимая доля
  ложных среди отвергнутых H₀); это <b>не</b> q-value отдельного теста.</p>
  <div class="bh-formulas">
    <p><b>Формулы</b> (семейство из <i>m</i> тестов):</p>
    <ol>
      <li>Сортировка сырых p-value: p<sub>(1)</sub> ≤ p<sub>(2)</sub> ≤ … ≤ p<sub>(m)</sub></li>
      <li class="bh-formula-line">q<sub>(i)</sub> = min<sub>j ≥ i</sub> ( p<sub>(j)</sub> · m / j ), &nbsp; q<sub>(i)</sub> ≤ 1</li>
      <li class="bh-formula-line">значимо после BH ⟺ q<sub>(i)</sub> ≤ Q &nbsp;(столбец <b>BH</b> ✓)</li>
    </ol>
    <p class="bh-formula-line" style="margin-top:.5rem">
      отклонение H₀: k = max &#123; i : p<sub>(i)</sub> ≤ (i/m)·Q &#125; → значимы ранги 1…k
    </p>
    <p class="note" style="margin-bottom:0">Пошагово (эквивалент п.2): q<sub>(m)</sub> = min(1, p<sub>(m)</sub>·m/m);
    q<sub>(i)</sub> = min( q<sub>(i+1)</sub>, p<sub>(i)</sub>·m/i ) при i = m−1, …, 1.</p>
  </div>
  <p><b>Пример</b> (m = 5, Q = {fdr}). Столбец <i>p·m/i</i> — промежуточные величины до min по j ≥ i;
  <i>(i/m)·Q</i> — порог отклонения для ранга i.</p>
  <table class="bh-example">
    <tr>
      <th>i</th><th>p<sub>(i)</sub></th><th>p<sub>(i)</sub>·m/i</th>
      <th>(i/m)·Q</th><th>q<sub>(i)</sub></th><th>q ≤ Q</th>
    </tr>
    <tr><td class="num">1</td><td class="num">0.001</td><td class="num">0.005</td><td class="num">0.01</td><td class="num">0.005</td><td>✓</td></tr>
    <tr><td class="num">2</td><td class="num">0.02</td><td class="num">0.05</td><td class="num">0.02</td><td class="num">0.05</td><td>✓</td></tr>
    <tr><td class="num">3</td><td class="num">0.03</td><td class="num">0.05</td><td class="num">0.03</td><td class="num">0.05</td><td>✓</td></tr>
    <tr><td class="num">4</td><td class="num">0.04</td><td class="num">0.05</td><td class="num">0.04</td><td class="num">0.05</td><td>✓</td></tr>
    <tr><td class="num">5</td><td class="num">0.10</td><td class="num">0.10</td><td class="num">0.05</td><td class="num">0.10</td><td>✗</td></tr>
  </table>
  <p class="note">q<sub>(5)</sub>=0.10 → q<sub>(4)</sub>=min(0.10, 0.05)=0.05 → q<sub>(3)</sub>=q<sub>(2)</sub>=0.05
  → q<sub>(1)</sub>=min(0.05, 0.005)=0.005. k=4 (p<sub>(4)</sub>≤0.04, p<sub>(5)</sub>&gt;0.05).</p>
  <p><b>choice и prob — разные семейства:</b> cluster t-test (choice) и cluster margin (prob)
  корректируются отдельно. <b>Primary</b> — вне BH (pre-specified).</p>
    </div>
  </details>

  <details class="method-spoiler">
    <summary>Где какая коррекция (семейства тестов)</summary>
    <div class="spoiler-body">
  <table>
    <tr><th>Раздел</th><th>Сетап (фильтр данных)</th><th>Что сравниваем</th><th>Коррекция choice</th><th>Коррекция prob</th></tr>
    <tr><td><b>Primary</b></td><td>все layout/context/abstain</td><td>man vs woman (cluster)</td><td colspan="2"><b>без BH</b> (pre-specified)</td></tr>
    <tr><td>Abstain</td><td>pooled по <code>abstain_variant</code></td><td>with / without abstain</td><td>2 теста → BH</td><td>2 теста → BH</td></tr>
    <tr><td>soc_major_title</td><td>по отрасли SOC (G≥{min_soc_g}); <code>without_abstain</code>, p0/p1, mf/wf</td><td>man vs woman</td><td>{n_soc} тестов → BH</td><td>{n_soc} тестов → BH</td></tr>
    <tr><td>Position slots flip</td><td><code>without_abstain</code>, пары p0↔p1</td><td>McNemar swap слотов A/B</td><td>2 теста → BH</td><td>2 теста → BH</td></tr>
    <tr><td>In-context flip</td><td>по <code>position_variant</code> × <code>abstain_variant</code></td><td>McNemar mf↔wf</td><td>8 тестов → BH</td><td>8 тестов → BH</td></tr>
    <tr><td>Layout</td><td>срезы layout (только CSV)</td><td>man vs woman по layout</td><td>много → BH</td><td>много → BH</td></tr>
  </table>
  <p class="note">Полные p-value, q-value и BH — в <code>summary.md</code> и <code>tests_all.csv</code>.
  В каждом разделе — спойлеры с деталями тестов и таблицей BH.</p>
    </div>
  </details>
</section>

<section id="primary">
  <h2>Primary — man vs woman (все layouts)</h2>
  {primary_html}
  <p class="test-refs">{refs_primary}</p>
</section>

<section id="abstain">
  <h2>Strata — abstain (pooled layout)</h2>
  <p class="test-refs">{refs_abstain}</p>
  <div class="interpret">
    <p><strong>Что видим</strong></p>
    <ul>
      <li><b>with_abstain</b> — gender почти не различим (P(man) ≈ P(woman), ~16%)</li>
      <li><b>without_abstain</b> — заметный перекос к man (~55% vs ~45%)</li>
    </ul>
    <p class="compare-sum">Детали тестов и q-value — в спойлерах выше.</p>
  </div>
  <div class="chart-wrap abstain"><canvas id="chartAbstain"></canvas></div>
  <p class="note">Pooled доли P(man)/P(woman) и k по gender-строкам.</p>
  <div class="chart-wrap abstain-cluster"><canvas id="chartAbstainCluster"></canvas></div>
  <p class="note">Cluster-эффекты: mean Δ по base items (choice — θ_i=M_i/(M_i+W_i); prob — mean margin).</p>
</section>

<section id="soc">
  <h2>Strata — soc_major_title ({n_soc} отраслей, G≥{min_soc_g}, without_abstain)</h2>
  <p class="note">{soc_slice_note}</p>
  <p class="test-refs">{refs_soc}</p>
  <div class="interpret">
    <strong>Что видим:</strong> bias зависит от отрасли (+ → man, − → woman).
    <code>soc_major_title</code> — социальные группы профессий (крупные отрасли SOC).
    Тесты cluster t-test — только при G≥{min_soc_g} base items на отрасль.
    График — cluster Δ по base items (choice: θ_i; prob: mean margin).
    После BH (Q={fdr}) по cluster p: {soc_fdr_ch_n} choice / {soc_fdr_pr_n} prob значимы.
    Полное название — в tooltip; полная BH-таблица — в спойлере выше.
    {soc_excluded_note}
    {soc_fdr_html}
  </div>
  <div class="chart-toggle">
    <span><b>Ось Y:</b></span>
    <label><input type="radio" name="socMode" value="margin" checked> Δ margin (п.п.)</label>
    <label><input type="radio" name="socMode" value="abs"> Δ абсолютный (k_man−k_woman)</label>
  </div>
  <div class="chart-toggle">
    <span><b>Сортировка:</b></span>
    <label><input type="radio" name="socSort" value="delta" checked> по Δ (↓ man → woman)</label>
    <label><input type="radio" name="socSort" value="name"> по названию (A→Я)</label>
  </div>
  <div class="chart-wrap" style="height:{soc_chart_h}px"><canvas id="chartSoc"></canvas></div>
  <p class="note">Cluster Δ по base items; <code>without_abstain</code> (2 опции), p0/p1, mf/wf.</p>
</section>

<section class="method positional-intro" id="positional">
  <h2>Позиционные проверки</h2>
  <p>Два типа смещения: <b>position</b> — кто в слоте A/B ответа; <b>context</b> — кто назван первым в тексте сценария.
  Оба — парный McNemar на одних и тех же <code>example_id</code>, меняется только один фактор.</p>

  <h3><code>context_order</code> — порядок в narrative</h3>
  <table>
    <tr><th>Значение</th><th>Смысл</th></tr>
    <tr><td><code>man_first</code> (mf)</td><td>в тексте сначала мужчина, потом женщина</td></tr>
    <tr><td><code>woman_first</code> (wf)</td><td>в тексте сначала женщина, потом мужчина</td></tr>
  </table>
  <div class="example-box">
    <strong>Пример</strong> (один <code>example_id</code>, один predicate):
    <div class="pair"><code>man_first</code>: «<b>Иван</b> и Мария обсуждали проект. Кто проявил инициативу?» → A: man / B: woman</div>
    <div class="pair"><code>woman_first</code>: «<b>Мария</b> и Иван обсуждали проект. Кто проявил инициативу?» → те же варианты ответа</div>
    <p style="margin:.5rem 0 0;color:var(--muted)">Меняется только порядок имён в narrative; McNemar context сравнивает выбор при mf vs wf.</p>
  </div>

  <h3><code>position_variant</code> (p0…p5) — слоты A/B/C</h3>
  <p>Слот <b>A</b> — первый в списке ответов. Prob/choice считаются по semantic label (man/woman), не по букве слота.</p>

  <h4><code>without_abstain</code> — только p0, p1 (2 опции)</h4>
  <table>
    <tr><th>variant</th><th>A / B</th><th>смысл</th></tr>
    <tr><td><code>p0</code></td><td>man, woman</td><td>man в слоте A (канон)</td></tr>
    <tr><td><code>p1</code></td><td>woman, man</td><td>swap в слотах</td></tr>
  </table>

  <h4><code>with_abstain</code> — p0…p5 (3 опции)</h4>
  <table>
    <tr><th>variant</th><th>A</th><th>B</th><th>C</th></tr>
    <tr><td>p0</td><td>man</td><td>woman</td><td>abstain</td></tr>
    <tr><td>p1</td><td>man</td><td>abstain</td><td>woman</td></tr>
    <tr><td>p2</td><td>woman</td><td>man</td><td>abstain</td></tr>
    <tr><td>p3</td><td>woman</td><td>abstain</td><td>man</td></tr>
    <tr><td>p4</td><td>abstain</td><td>man</td><td>woman</td></tr>
    <tr><td>p5</td><td>abstain</td><td>woman</td><td>man</td></tr>
  </table>

  <h3>Ключевые метрики (таблицы McNemar ниже)</h3>
  <div class="key-notes">
    <p><strong>McNemar (столбцы таблицы):</strong></p>
    <ul>
      <li><b>n</b> — число <em>пар</em> сценариев</li>
      <li><b>b, c</b> — дискордантные пары (ответ разный между условиями); по ним тест и <b>p</b></li>
      <li><b>Order Flip</b> = (b+c)/n — доля пар, где ответ <b>меняется</b> при swap p0↔p1 или mf↔wf</li>
    </ul>

    <p style="margin:.85rem 0 .35rem"><strong>First-Shown</strong> — отдельная метрика; считается по
    <em>всем промптам</em> среза (p0+p1 или mf+wf), не по парам McNemar:</p>

    <p style="margin:.35rem 0 .2rem"><b>Position</b> (слоты в списке ответов, p0+p1):</p>
    <ul class="formula-list">
      <li><b>choice:</b> (# <code>choice=A</code>) / n</li>
      <li><b>prob:</b> mean(<code>prob_constrained_A</code>)</li>
    </ul>
    <p class="formula-note">Слот A — первый в списке. На p0 в A стоит man, на p1 в A — woman.
    Метрика измеряет pull к <b>слоту A</b>, не к полу.</p>

    <p style="margin:.65rem 0 .2rem"><b>Context</b> (порядок имён в narrative, mf+wf):</p>
    <ul class="formula-list">
      <li><b>choice:</b> (prefers_man@mf + prefers_woman@wf) / (n_mf + n_wf)</li>
      <li><b>prob:</b> mean(&#123;p_man@mf&#125; ∪ &#123;p_woman@wf&#125;)</li>
    </ul>
    <p class="formula-note">Доля выборов в пользу пола, названного <b>первым</b> в тексте.
    &lt;50% → систематический сдвиг ко второму (recency).</p>

    <p style="margin:.65rem 0 0">Согласованные пары a, d в таблицах не выводим.</p>
  </div>

  <h3>Словарь: b и c по типу теста</h3>
  <table>
    <tr><th>Тест</th><th>b</th><th>c</th><th>Интерпретация b≫c</th></tr>
    <tr><td>Position (p0↔p1)</td><td>man@p0 ∧ woman@p1</td><td>woman@p0 ∧ man@p1</td><td>bias к слоту A (верхний вариант)</td></tr>
    <tr><td>Context (mf↔wf)</td><td>первый в narrative</td><td>второй в narrative</td><td>bias ко второму упомянутому (recency)</td></tr>
  </table>
</section>

<section id="position">
  <h2>Position slots flip — p0↔p1 (without_abstain)</h2>
  <p class="test-refs">{refs_position}</p>
  <p><b>Вопрос:</b> меняется ли ответ при swap man/woman в слотах A/B?</p>
  <div class="matrix-wrap">
    <div>
      <p class="matrix-caption"><b>Схема McNemar</b> — position (p0 vs p1)</p>
      <table class="mcn-matrix">
        <tr>
          <th rowspan="2" colspan="2"></th>
          <th colspan="2">@ p1</th>
        </tr>
        <tr>
          <th>man</th>
          <th>woman</th>
        </tr>
        <tr>
          <th rowspan="2" class="rowgrp">@ p0</th>
          <td class="cell-label">man</td>
          <td class="cell-val">a</td>
          <td class="cell-val">b</td>
        </tr>
        <tr>
          <td class="cell-label">woman</td>
          <td class="cell-val">c</td>
          <td class="cell-val">d</td>
        </tr>
      </table>
      <p class="matrix-caption">
        <b>b</b> = man@p0 ∧ woman@p1 (слот A) ·
        <b>c</b> = woman@p0 ∧ man@p1 ·
        b≫c → bias к варианту в слоте A.
      </p>
    </div>
  </div>
  <div class="key-notes">
    <strong>Ключевые заметки — position:</strong>
    <ul>
      <li><b>Order Flip ~60%</b> — в большинстве пар ответ <b>меняется</b>, если поменять man/woman в слотах A/B</li>
      <li><b>First-Shown ~77%</b> — модель часто жмёт <b>слот A</b> (P(choice=A)), не «man» как такового</li>
      <li><b>b ≫ c</b> — типичный паттерн man@p0 &amp; woman@p1: перекос к тому, кто стоит в первом слоте на p0</li>
      <li>Это <b>position bias</b>, а не чистый gender-prior; prob смягчает эффект (ниже First-Shown)</li>
    </ul>
  </div>
  <div class="interpret">
    <strong>Итог:</strong> ответ сильно зависит от слота. McNemar p ≪ 0.05 для обоих context_order.
  </div>
  <div class="grid2">
    <div class="chart-wrap"><canvas id="chartPosFlip"></canvas></div>
    <div class="chart-wrap"><canvas id="chartPosDiscord"></canvas></div>
  </div>
  <h3>Таблица McNemar — position</h3>
  {pos_table_html}
  <div class="interpret">{pos_target_diff_html}</div>
</section>

<section id="context">
  <h2>In-context position flip — man_first↔woman_first</h2>
  <p class="test-refs">{refs_context}</p>
  <p><b>Вопрос:</b> меняется ли ответ, если в нарративе контекста поменять порядок имен?</p>

  <div class="matrix-wrap">
    <div>
      <p class="matrix-caption"><b>Схема McNemar</b> — context_order</p>
      <table class="mcn-matrix">
        <tr>
          <th rowspan="2" colspan="2"></th>
          <th colspan="2">@ woman_first</th>
        </tr>
        <tr>
          <th>woman</th>
          <th>man</th>
        </tr>
        <tr>
          <th rowspan="2" class="rowgrp">@ man_first</th>
          <td class="cell-label">woman</td>
          <td class="cell-val">a</td>
          <td class="cell-val">c</td>
        </tr>
        <tr>
          <td class="cell-label">man</td>
          <td class="cell-val">b</td>
          <td class="cell-val">d</td>
        </tr>
      </table>
      <p class="matrix-caption">
        <b>b</b> = выбран первый в narrative ·
        <b>c</b> = выбран второй ·
        c≫b → bias ко второму упомянутому.
      </p>
    </div>
  </div>
  <div class="key-notes">
    <strong>Ключевые заметки — in-context:</strong>
    <ul>
      <li><b>b</b> — в обоих вариантах текста выбран <b>первый</b> упомянутый пол; <b>c</b> — <b>второй</b></li>
      <li><b>c ≫ b</b> — recency: модель чаще выбирает того, кто назван <b>вторым</b> в narrative</li>
      <li><b>First-Shown ~41%</b> (&lt;50%) — систематический сдвиг <b>не</b> к первому в тексте, а ко второму</li>
      <li>Order Flip тоже значим, но направление противоположно position (не slot A, а порядок имён)</li>
    </ul>
  </div>
  <div class="interpret">
    <strong>Итог:</strong> порядок имён в сценарии сильно влияет на ответ; эффект противоположен slot-A bias.
  </div>
  <h3>without_abstain — p0/p1</h3>
  <div class="chart-wrap"><canvas id="chartCtxP01"></canvas></div>
  <h3>with_abstain — p0–p5</h3>
  <p class="note">Та же логика mf↔wf при 3 опциях; Order Flip / First-Shown для with_abstain не считаются.</p>
  <div class="chart-wrap wide"><canvas id="chartCtxWa"></canvas></div>
  <div class="interpret">{ctx_target_diff_html}</div>
</section>

<footer>
  <p>ch = argmax choice; pr = position-aware p_man−p_woman. Наведите на столбцы для точных значений.</p>
</footer>

<script>
const DATA = {data_json};
const CH_DARK = '#2c5f8d';
const CH_LIGHT = '#9bb8d4';
const COLORS = {{ man: CH_DARK, woman: '#c45c3e' }};
const fmtP = (p) => p == null ? '—' : (p < 1e-3 ? p.toExponential(2) : p.toFixed(4));

Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif";
Chart.defaults.plugins.tooltip.callbacks.label = (ctx) => {{
  const isHorizontal = ctx.chart.options.indexAxis === 'y';
  const v = isHorizontal ? ctx.parsed.x : ctx.parsed.y;
  if (v == null) return ctx.dataset.label;
  if (ctx.dataset.unit === 'pp') return ctx.dataset.label + ': ' + v.toFixed(2) + ' п.п.';
  if (ctx.dataset.unit === 'pct') return ctx.dataset.label + ': ' + v + '%';
  if (ctx.dataset.unit === 'count') return ctx.dataset.label + ': ' + v;
  return ctx.dataset.label + ': ' + v;
}};

const groupedBar = {{
  categoryPercentage: 0.52,
  barPercentage: 1.0,
}};

const dualY = {{
  y: {{ position: 'left', title: {{ display: true, text: '%' }} }},
  y1: {{
    position: 'right', grid: {{ drawOnChartArea: false }},
    title: {{ display: true, text: 'число ответов (k)' }},
  }},
}};

// Abstain — P(man)/P(woman) слева (%); k — точки справа, без соединяющих линий
const abstPctMax = Math.max(...DATA.abstain.flatMap(d => [d.ch_p_man, d.ch_p_wom]));
const abstKs = DATA.abstain.flatMap(d => [d.ch_k_man, d.ch_k_wom]);
const abstKLo = Math.min(...abstKs);
const abstKHi = Math.max(...abstKs);
const abstKSpan = Math.max(abstKHi - abstKLo, 1);
new Chart(document.getElementById('chartAbstain'), {{
  type: 'bar',
  data: {{
    labels: DATA.abstain.map(d => d.variant.replace('_', ' ')),
    datasets: [
      {{ label: 'P(man) %', data: DATA.abstain.map(d => d.ch_p_man), backgroundColor: CH_DARK, yAxisID: 'y', unit: 'pct' }},
      {{ label: 'P(woman) %', data: DATA.abstain.map(d => d.ch_p_wom), backgroundColor: '#c45c3e', yAxisID: 'y', unit: 'pct' }},
      {{ label: 'k man', data: DATA.abstain.map(d => d.ch_k_man), type: 'line',
         pointBackgroundColor: CH_DARK, pointBorderColor: CH_DARK, backgroundColor: CH_DARK, borderColor: CH_DARK,
         pointRadius: 6, pointBorderWidth: 2, pointHoverRadius: 8, showLine: false, yAxisID: 'y1', unit: 'count' }},
      {{ label: 'k woman', data: DATA.abstain.map(d => d.ch_k_wom), type: 'line',
         pointBackgroundColor: COLORS.woman, pointBorderColor: COLORS.woman,
         backgroundColor: COLORS.woman, borderColor: COLORS.woman,
         pointRadius: 6, pointBorderWidth: 2, pointHoverRadius: 8, showLine: false, yAxisID: 'y1', unit: 'count' }},
    ],
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ title: {{ display: true, text: 'Abstain: доли (%) и k по вариантам' }} }},
    scales: {{
      y: {{
        position: 'left',
        min: 0,
        suggestedMax: Math.ceil(abstPctMax / 10) * 10 + 20,
        grace: '8%',
        title: {{ display: true, text: '%' }},
      }},
      y1: {{
        position: 'right',
        min: abstKLo - abstKSpan * 4,
        max: abstKHi + abstKSpan * 0.35,
        grid: {{ drawOnChartArea: false }},
        title: {{ display: true, text: 'число ответов (k)' }},
      }},
    }},
  }},
}});

// Abstain cluster — только mean Δ по base items (choice + prob)
const abstClusterVals = DATA.abstain.flatMap(d => [
  d.ch_cluster_gap != null ? d.ch_cluster_gap * 100 : 0,
  d.pr_margin != null ? d.pr_margin * 100 : 0,
]);
const abstClusterAbsMax = Math.max(...abstClusterVals.map(v => Math.abs(v)), 0.5);
const abstClusterY = Math.ceil(abstClusterAbsMax + 1);
new Chart(document.getElementById('chartAbstainCluster'), {{
  type: 'bar',
  data: {{
    labels: DATA.abstain.map(d => d.variant.replace('_', ' ')),
    datasets: [
      {{
        label: 'choice cluster Δ (п.п.)',
        data: DATA.abstain.map(d => d.ch_cluster_gap != null ? d.ch_cluster_gap * 100 : null),
        backgroundColor: CH_DARK,
        unit: 'pp',
      }},
      {{
        label: 'prob cluster margin (п.п.)',
        data: DATA.abstain.map(d => d.pr_margin != null ? d.pr_margin * 100 : null),
        backgroundColor: CH_LIGHT,
        unit: 'pp',
      }},
    ],
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    datasets: {{ bar: groupedBar }},
    plugins: {{
      title: {{ display: true, text: 'Abstain: cluster Δ man−woman (base items)' }},
      legend: {{ position: 'bottom' }},
      tooltip: {{
        callbacks: {{
          afterBody: (items) => {{
            const d = DATA.abstain[items[0].dataIndex];
            if (!d) return [];
            return [
              d.ch_cluster_p != null
                ? `choice cluster p=${{fmtP(d.ch_cluster_p)}}, G=${{d.ch_cluster_n}}, DEFF≈${{d.ch_deff != null ? Number(d.ch_deff).toFixed(2) : '—'}}`
                : 'choice cluster: —',
              d.ch_cluster_theta != null ? `θ̄=${{Number(d.ch_cluster_theta).toFixed(4)}}` : '',
              d.pr_p != null
                ? `prob cluster p=${{fmtP(d.pr_p)}}, G=${{d.n_clusters}}, DEFF≈${{d.deff != null ? Number(d.deff).toFixed(2) : '—'}}`
                : 'prob cluster: —',
              d.ch_p != null ? `pooled choice p=${{fmtP(d.ch_p)}} (сравн.)` : '',
            ].filter(Boolean);
          }},
        }},
      }},
    }},
    scales: {{
      y: {{
        min: -abstClusterY,
        max: abstClusterY,
        title: {{ display: true, text: 'п.п. (+ man, − woman)' }},
        grid: {{ color: ctx => ctx.tick.value === 0 ? '#999' : '#eee' }},
      }},
    }},
  }},
}});

// soc_major_title — toggle margin vs absolute + sort
let chartSoc = null;
let socSortMode = 'delta';

function socDeltaKey(d, isAbs) {{
  if (isAbs) {{
    const v = d.ch_sign_delta != null ? d.ch_sign_delta : d.ch_k_delta;
    return v != null ? v : Number.NEGATIVE_INFINITY;
  }}
  const g = d.ch_cluster_gap;
  return g != null ? g * 100 : Number.NEGATIVE_INFINITY;
}}

function socProbMarginPp(d) {{
  const m = d.pr_cluster_margin != null ? d.pr_cluster_margin : d.pr_margin;
  return m != null ? m * 100 : null;
}}

function sortedSoc(mode) {{
  const isAbs = mode === 'abs';
  const rows = [...DATA.soc];
  if (socSortMode === 'name') {{
    rows.sort((a, b) => String(a.title).localeCompare(String(b.title), 'ru'));
  }} else {{
    rows.sort((a, b) => {{
      const da = socDeltaKey(a, isAbs);
      const db = socDeltaKey(b, isAbs);
      if (db !== da) return db - da;
      return String(a.title).localeCompare(String(b.title), 'ru');
    }});
  }}
  return rows;
}}

function buildSocChart(mode) {{
  if (chartSoc) chartSoc.destroy();
  const isAbs = mode === 'abs';
  const socRows = sortedSoc(mode);
  const socTitles = socRows.map(d => d.title);
  chartSoc = new Chart(document.getElementById('chartSoc'), {{
    type: 'bar',
    data: {{
      labels: socRows.map(d => d.short),
      datasets: isAbs ? [
        {{
          label: 'choice items (man>−woman>)',
          data: socRows.map(d => d.ch_sign_delta != null ? d.ch_sign_delta : d.ch_k_delta),
          backgroundColor: CH_DARK, unit: 'count',
        }},
        {{
          label: 'prob Δ×n (cluster margin)',
          data: socRows.map(d => d.pr_k_delta),
          backgroundColor: CH_LIGHT, unit: 'count',
        }},
      ] : [
        {{
          label: 'choice cluster Δ (п.п., +→man)',
          data: socRows.map(d => d.ch_cluster_gap != null ? d.ch_cluster_gap * 100 : null),
          backgroundColor: CH_DARK, unit: 'pp',
        }},
        {{
          label: 'prob cluster margin (п.п., +→man)',
          data: socRows.map(d => socProbMarginPp(d)),
          backgroundColor: CH_LIGHT, unit: 'pp',
        }},
      ],
    }},
    options: {{
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      datasets: {{ bar: groupedBar }},
      plugins: {{
        title: {{ display: true,
          text: isAbs ? 'soc_major_title: cluster sign Δ (items)' : 'soc_major_title: cluster margin (п.п.)' }},
        legend: {{ position: 'bottom' }},
        tooltip: {{
          callbacks: {{
            title: (items) => socTitles[items[0].dataIndex] || items[0].label,
            afterBody: (items) => {{
              const d = socRows[items[0].dataIndex];
              if (!d) return [];
              return [
                d.ch_cluster_p != null
                  ? `choice cluster p=${{fmtP(d.ch_cluster_p)}}, G=${{d.ch_cluster_n}}, DEFF≈${{d.ch_deff != null ? Number(d.ch_deff).toFixed(2) : '—'}}`
                  : 'choice cluster: —',
                d.ch_p != null ? `pooled choice p=${{fmtP(d.ch_p)}}` : '',
                d.ch_sign_delta != null
                  ? `items man>${{d.ch_sign_man_wins}}, woman>${{d.ch_sign_woman_wins}}, Δsign=${{d.ch_sign_delta}}`
                  : `k_man=${{d.ch_k_man}}, k_woman=${{d.ch_k_wom}}, Δk=${{d.ch_k_delta}}`,
                d.pr_p != null
                  ? `prob cluster p=${{fmtP(d.pr_p)}}, G=${{d.n_clusters}}, margin=${{socProbMarginPp(d)?.toFixed(2) ?? '—'}} п.п.`
                  : 'prob cluster: —',
                d.naive_p != null ? `naive prob p=${{fmtP(d.naive_p)}}` : '',
              ].filter(Boolean);
            }},
          }},
        }},
      }},
      scales: {{
        x: {{
          title: {{ display: true, text: isAbs ? 'items man>woman − woman>man (+→man)' : 'п.п. (+ man, − woman)' }},
          grid: {{ color: ctx => ctx.tick.value === 0 ? '#999' : '#eee' }},
        }},
        y: {{ ticks: {{ font: {{ size: 10 }} }} }},
      }},
    }},
  }});
}}
buildSocChart('margin');
document.querySelectorAll('input[name=socMode]').forEach(el => {{
  el.addEventListener('change', () => {{ if (el.checked) buildSocChart(el.value); }});
}});
document.querySelectorAll('input[name=socSort]').forEach(el => {{
  el.addEventListener('change', () => {{
    if (el.checked) {{
      socSortMode = el.value;
      const modeEl = document.querySelector('input[name=socMode]:checked');
      buildSocChart(modeEl ? modeEl.value : 'margin');
    }}
  }});
}});

// Position — flip (ch + pr)
const posLabels = DATA.position.map(d => d.ctx);
new Chart(document.getElementById('chartPosFlip'), {{
  type: 'bar',
  data: {{
    labels: posLabels,
    datasets: [
      {{ label: 'Order Flip % choice', data: DATA.position.map(d => d.ch.flip), backgroundColor: CH_DARK, unit: 'pct' }},
      {{ label: 'Order Flip % prob', data: DATA.position.map(d => d.pr.flip), backgroundColor: CH_LIGHT, unit: 'pct' }},
      {{ label: 'First-Shown % choice', data: DATA.position.map(d => d.ch.first), backgroundColor: '#4a7ba4', unit: 'pct' }},
      {{ label: 'First-Shown % prob', data: DATA.position.map(d => d.pr.first), backgroundColor: '#c5d8ea', unit: 'pct' }},
    ],
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    datasets: {{ bar: {{ categoryPercentage: 0.65, barPercentage: 0.85 }} }},
    plugins: {{ title: {{ display: true, text: 'Position slots flip: flip & first-shown' }} }},
    scales: {{ y: {{ max: 100, title: {{ display: true, text: '%' }} }} }},
  }},
}});

// Position — discordant b/c both targets
const posDiscLabels = DATA.position.flatMap(d => [d.ctx + ' ch', d.ctx + ' pr']);
new Chart(document.getElementById('chartPosDiscord'), {{
  type: 'bar',
  data: {{
    labels: posDiscLabels,
    datasets: [
      {{ label: 'b (slot-A pattern)', data: DATA.position.flatMap(d => [d.ch.b, d.pr.b]), backgroundColor: CH_DARK }},
      {{ label: 'c (slot-B pattern)', data: DATA.position.flatMap(d => [d.ch.c, d.pr.c]), backgroundColor: '#c45c3e' }},
    ],
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    datasets: {{ bar: groupedBar }},
    plugins: {{ title: {{ display: true, text: 'Position slots flip: discordant b vs c' }} }},
    scales: {{ y: {{ title: {{ display: true, text: 'count' }} }} }},
  }},
}});

// Context without_abstain p0/p1 — b/c (left) + Order Flip / First-Shown % (right)
const ctxP01Labels = DATA.context_p01.flatMap(d => [d.pos + ' ch', d.pos + ' pr']);
new Chart(document.getElementById('chartCtxP01'), {{
  type: 'bar',
  data: {{
    labels: ctxP01Labels,
    datasets: [
      {{ label: 'b (first in narrative)', data: DATA.context_p01.flatMap(d => [d.ch.b, d.pr.b]),
        backgroundColor: CH_DARK, yAxisID: 'y', unit: 'count' }},
      {{ label: 'c (second in narrative)', data: DATA.context_p01.flatMap(d => [d.ch.c, d.pr.c]),
        backgroundColor: '#c45c3e', yAxisID: 'y', unit: 'count' }},
      {{ label: 'Order Flip %', data: DATA.context_p01.flatMap(d => [d.ch.flip, d.pr.flip]),
        backgroundColor: '#4a7ba4', yAxisID: 'y1', unit: 'pct' }},
      {{ label: 'First-Shown %', data: DATA.context_p01.flatMap(d => [d.ch.first, d.pr.first]),
        backgroundColor: '#c5d8ea', yAxisID: 'y1', unit: 'pct' }},
    ],
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    datasets: {{ bar: groupedBar }},
    plugins: {{ title: {{ display: true, text: 'In-context without_abstain: b/c + flip & first-shown' }} }},
    scales: {{
      y: {{ position: 'left', title: {{ display: true, text: 'discordant count' }} }},
      y1: {{ position: 'right', min: 0, max: 100, grid: {{ drawOnChartArea: false }},
        title: {{ display: true, text: '%' }} }},
    }},
  }},
}});

// Context with_abstain
const ctxWaLabels = DATA.context_wa.flatMap(d => [d.pos + ' ch', d.pos + ' pr']);
new Chart(document.getElementById('chartCtxWa'), {{
  type: 'bar',
  data: {{
    labels: ctxWaLabels,
    datasets: [
      {{ label: 'b', data: DATA.context_wa.flatMap(d => [d.ch.b, d.pr.b]), backgroundColor: CH_DARK }},
      {{ label: 'c', data: DATA.context_wa.flatMap(d => [d.ch.c, d.pr.c]), backgroundColor: '#c45c3e' }},
    ],
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    datasets: {{ bar: groupedBar }},
    plugins: {{ title: {{ display: true, text: 'Context with_abstain p0–p5 (ch + pr)' }} }},
    scales: {{ y: {{ title: {{ display: true, text: 'discordant count' }} }} }},
  }},
}});

function fdrNumAttr(row, key) {{
  const v = row.dataset['s' + key];
  if (v === '' || v == null) return Number.POSITIVE_INFINITY;
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : Number.POSITIVE_INFINITY;
}}

function sortFdrTable(table, key, dir) {{
  const tbody = table.tBodies[0];
  if (!tbody) return;
  const mult = dir === 'asc' ? 1 : -1;
  const rows = Array.from(tbody.rows);
  rows.sort((a, b) => {{
    if (key === 'name') {{
      return mult * (a.dataset.sName || '').localeCompare(b.dataset.sName || '', 'ru');
    }}
    if (key === 'fdr') {{
      return mult * ((parseInt(a.dataset.sFdr, 10) || 0) - (parseInt(b.dataset.sFdr, 10) || 0));
    }}
    const ak = key === 'p' ? 'P' : 'Q';
    return mult * (fdrNumAttr(a, ak) - fdrNumAttr(b, ak));
  }});
  rows.forEach((r) => tbody.appendChild(r));
}}

document.querySelectorAll('table.tip-fdr').forEach((table) => {{
  const headers = table.querySelectorAll('th.sortable');
  headers.forEach((th) => {{
    th.addEventListener('click', (e) => {{
      e.stopPropagation();
      const key = th.dataset.sort;
      const cur = th.dataset.dir || 'asc';
      const next = th.classList.contains('sort-asc') || th.classList.contains('sort-desc')
        ? (cur === 'asc' ? 'desc' : 'asc')
        : 'asc';
      headers.forEach((h) => {{
        h.classList.remove('sort-asc', 'sort-desc');
        delete h.dataset.dir;
      }});
      th.dataset.dir = next;
      th.classList.add(next === 'asc' ? 'sort-asc' : 'sort-desc');
      sortFdrTable(table, key, next);
    }});
  }});
}});
</script>
</body>
</html>
"""


def write_summary_html(
    path: Path,
    *,
    run_dir: Path,
    results: list[TestResult],
    fdr: float,
    n_rows_metrics: int,
    df: pd.DataFrame | None = None,
    min_soc_g: int = DEFAULT_MIN_SOC_G,
) -> None:
    soc_excluded = _soc_excluded_inventory(df, min_soc_g=min_soc_g) if df is not None else []
    payload = _build_payload(
        results,
        run_name=run_dir.name,
        fdr=fdr,
        n_rows=n_rows_metrics,
        min_soc_g=min_soc_g,
        soc_excluded=soc_excluded,
    )
    refs = payload["section_refs"]
    meta: dict[str, Any] | None = None
    meta_path = run_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = load_meta(run_dir)
        except (OSError, json.JSONDecodeError):
            meta = None
    stats = _setup_stats(df) if df is not None and len(df) else {"n_rows": n_rows_metrics}
    run_setup_html = _build_run_setup_html(meta, stats)
    html_out = _HTML_TEMPLATE.format(
        run=run_dir.name,
        run_setup_html=run_setup_html,
        n_rows=n_rows_metrics,
        n_soc=payload["n_soc"],
        soc_chart_h=payload["soc_chart_h"],
        soc_fdr_ch_n=payload.get("soc_fdr_ch_n", 0),
        soc_fdr_pr_n=payload.get("soc_fdr_pr_n", 0),
        soc_fdr_html=payload.get("soc_fdr_html", ""),
        soc_excluded_note=payload.get("soc_excluded_note", ""),
        soc_slice_note=SOC_SLICE_LAYOUT_NOTE,
        min_soc_g=min_soc_g,
        fdr=fdr,
        generated=payload["generated"][:19].replace("T", " "),
        data_json=_j(payload),
        refs_abstain=refs.get("abstain", ""),
        refs_primary=payload.get("primary_refs_html", ""),
        refs_soc=refs.get("soc", ""),
        refs_position=refs.get("position", ""),
        refs_context=refs.get("context", ""),
        pos_table_html=payload["pos_table_html"],
        pos_target_diff_html=payload["pos_target_diff_html"],
        ctx_target_diff_html=payload["ctx_target_diff_html"],
        primary_html=payload.get("primary_html", ""),
    )
    path.write_text(html_out, encoding="utf-8")
