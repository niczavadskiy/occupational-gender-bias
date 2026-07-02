"""Self-contained HTML dashboard for H3 v1 metrics (Chart.js)."""

from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.metrics.load import load_meta
from src.metrics.stats_tests import TestResult

CH_DARK = "#2c5f8d"
CH_LIGHT = "#9bb8d4"
CH_ACCENT = "#c45c3e"
CH_GREEN = "#3d7a5c"

MCNEMAR_SLICE_NOTE = (
    "<b>Срез McNemar:</b> только <code>without_abstain</code> — 2 опции (man/woman), "
    "p0/p1, mf/wf; без «Воздержаться». Парный ключ — <code>id</code> "
    "(3 804 layout-id × 3 evidence). Не смешивается с <code>with_abstain</code> "
    "(см. Rates)."
)


def _j(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)


def _pct(x: float | None) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if abs(x) <= 1.0:
        return round(100.0 * x, 2)
    return round(float(x), 2)


def _fmt_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    if p < 1e-4:
        return f"{p:.3e}"
    return f"{p:.4g}"


def _sig_label(p: float | None, *, one_sided: bool = True) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    if p < 0.001:
        return "p &lt; 0.001"
    if p < 0.05:
        return f"p = {_fmt_p(p)} ✓"
    return f"p = {_fmt_p(p)} (n.s.)"


def _mcn_from_result(r: TestResult) -> dict[str, Any]:
    ex = r.extra or {}
    b = int(ex.get("discordant_b") or r.k1 or 0)
    c = int(ex.get("discordant_c") or r.k2 or 0)
    outcome = ex.get("outcome_target", "choice")
    return {
        "test_id": r.test_id,
        "kind": "mcnemar",
        "n": int(ex.get("n_pairs") or r.n1 or 0),
        "b": b,
        "c": c,
        "effect": round(float(r.effect or 0), 4) if r.effect is not None else None,
        "p": r.p_raw,
        "target": outcome,
        "target_label": "choice (argmax)" if outcome == "choice" else "prob (margin sign)",
    }


def _render_primary_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class='note'>_(нет данных)_</p>"
    hdr = (
        "<table class='mcn'><thead><tr>"
        "<th>Тест</th><th>что измеряем</th><th>n</th><th>b</th><th>c</th>"
        "<th>effect</th><th>p</th>"
        "</tr></thead><tbody>"
    )
    body: list[str] = []
    for i, row in enumerate(rows):
        grp = "mcn-grp-a" if i % 2 == 0 else "mcn-grp-b"
        eff = row.get("effect")
        eff_s = f"{eff:+.3f}" if eff is not None else "—"
        body.append(
            f"<tr class='{grp}'>"
            f"<td class='mcn-key'>{html.escape(str(row.get('label', row.get('test_id', ''))))}</td>"
            f"<td>{html.escape(str(row.get('target_label', row.get('target', ''))))}</td>"
            f"<td>{row.get('n', '—')}</td>"
            f"<td>{row.get('b', '—')}</td><td>{row.get('c', '—')}</td>"
            f"<td>{eff_s}</td>"
            f"<td>{_fmt_p(row.get('p'))}</td></tr>"
        )
    return hdr + "".join(body) + "</tbody></table>"


def _build_primary_rows(by_id: dict[str, TestResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, tid in (
        ("no→man (choice)", "h3v1_mcnemar_no_vs_man_man"),
        ("no→woman (choice)", "h3v1_mcnemar_no_vs_woman_woman"),
        ("no→man (prob)", "h3v1_mcnemar_no_vs_man_man_prob"),
        ("no→woman (prob)", "h3v1_mcnemar_no_vs_woman_woman_prob"),
    ):
        r = by_id.get(tid)
        if r:
            row = _mcn_from_result(r)
            row["label"] = label
            rows.append(row)
    return rows


def _mcn_paragraph(r: TestResult, *, label: str) -> str:
    ex = r.extra or {}
    b = int(ex.get("discordant_b") or r.k1 or 0)
    c = int(ex.get("discordant_c") or r.k2 or 0)
    eff = float(r.effect or 0)
    axis = str(ex.get("gender_axis", ""))
    return (
        f"<p><b>{label}:</b> n_pairs={int(ex.get('n_pairs') or r.n1 or 0):,}; "
        f"b={b:,} (к {axis}), c={c:,} (от {axis}); net effect={eff:+.1%}. "
        f"{_sig_label(r.p_raw)}</p>"
    )


def _is_sig(p: float | None) -> bool:
    return p is not None and not (isinstance(p, float) and math.isnan(p)) and p < 0.05


def _verdict_phrase(r: TestResult | None) -> str:
    """Data-driven one-line verdict for a single McNemar test (b=toward, c=away)."""
    if r is None:
        return ""
    eff = float(r.effect or 0)
    sig = _is_sig(r.p_raw)
    if eff > 0 and sig:
        return "значимый сдвиг <b>к</b> highlighted полу — <b>H₃ подтверждается</b>"
    if eff > 0:
        return "слабый сдвиг к highlighted полу, незначимо (n.s.)"
    if eff < 0 and sig:
        return "значимый сдвиг <b>от</b> highlighted пола — <b>против H₃</b>"
    return "эффект около нуля / слабо от highlighted пола (n.s.)"


def _pooled_verdict(rs: list[TestResult | None]) -> str:
    present = [r for r in rs if r is not None]
    if not present:
        return "H₃: нет данных."
    support = [r for r in present if float(r.effect or 0) > 0 and _is_sig(r.p_raw)]
    against = [r for r in present if float(r.effect or 0) < 0 and _is_sig(r.p_raw)]
    if support and not against:
        return (
            "<strong>Итог (without_abstain):</strong> H₃ <b>подтверждается</b> — "
            f"значимый сдвиг к highlighted полу в {len(support)} из {len(present)} "
            "primary-тестов; противоположно значимых нет."
        )
    if support and against:
        return (
            "<strong>Итог (without_abstain):</strong> картина <b>смешанная</b> — "
            f"{len(support)} тест(а) значимо за H₃, {len(against)} значимо против; "
            "эффект зависит от пола highlight и target (choice/prob). "
            "См. построчные выводы выше и rates ниже."
        )
    if against and not support:
        return (
            "<strong>Итог (without_abstain):</strong> H₃ <b>не подтверждается</b> — "
            "значимые эффекты направлены от highlighted пола."
        )
    return (
        "<strong>Итог (without_abstain):</strong> значимых эффектов нет ни за, ни против H₃ "
        "(слабые сдвиги). См. rates ниже."
    )


def _build_primary_interpret_html(by_id: dict[str, TestResult]) -> str:
    ch_man = by_id.get("h3v1_mcnemar_no_vs_man_man")
    ch_wom = by_id.get("h3v1_mcnemar_no_vs_woman_woman")
    pr_man = by_id.get("h3v1_mcnemar_no_vs_man_man_prob")
    pr_wom = by_id.get("h3v1_mcnemar_no_vs_woman_woman_prob")

    parts: list[str] = [
        "<div class='effect-key'>"
        "<b>Как читать <code>effect</code>:</b> "
        "<code>effect = (b − c) / n_pairs</code>, где "
        "<b>b</b> — число пар, сменивших ответ <b>в сторону</b> highlighted пола "
        "(a→b), <b>c</b> — сменивших <b>от</b> него, "
        "<b>n_pairs</b> — все валидные парные <code>id</code>. "
        "Знак: <b>«+»</b> = net-сдвиг <b>к</b> highlighted полу "
        "(<b>подтверждает H₃</b>), <b>«−»</b> = от него. "
        "Напр. <code>effect = +0.10</code> ⇒ net 10% пар сдвинулись к highlighted полу."
        "</div>",
        "<p><strong>Два target, как в H1 v1</strong> (paired по <code>id</code>, "
        f"<code>without_abstain</code>, one-sided H₃↑, вне BH):</p>",
        "<ul>"
        "<li><b>choice</b> — <b>argmax</b> по слотам ответа (man / woman / abstain); "
        "уже применён при inference → <code>prefers_man</code>, "
        "<code>prefers_woman</code>, <code>abstain</code>.</li>"
        "<li><b>prob</b> — position-aware <code>p_man</code>, <code>p_woman</code> "
        "из <code>prob_constrained_*</code>; argmax <b>не</b> применяется повторно.</li>"
        "</ul>",
        "<table>"
        "<tr><th>Тест</th><th>Target</th><th>b/c?</th><th>Что сравниваем</th></tr>"
        "<tr><td>choice McNemar</td><td>choice</td><td>да</td>"
        "<td>flip дискретного argmax-ответа (man/woman/abstain)</td></tr>"
        "<tr><td>prob McNemar</td><td>prob</td><td>да</td>"
        "<td>flip знака margin: <code>p_man&nbsp;&gt;&nbsp;p_woman</code> "
        "(<code>prob_prefers_man</code>); abstain-слот не участвует</td></tr>"
        "</table>",
        "<p class='note'>choice и prob — <b>разные представления</b> одного ответа: "
        "argmax даёт choice; prob-тесты работают с вероятностями напрямую.</p>",
    ]

    parts.append("<p><strong>1. choice McNemar</strong> (argmax; 2-option layout, "
                 "все valid-пары man↔woman):</p>")
    if ch_man:
        parts.append(_mcn_paragraph(ch_man, label="no→man, ось man"))
        parts.append(f"<p class='note'>→ {_verdict_phrase(ch_man)}.</p>")
    if ch_wom:
        parts.append(_mcn_paragraph(ch_wom, label="no→woman, ось woman"))
        parts.append(f"<p class='note'>→ {_verdict_phrase(ch_wom)}.</p>")

    parts.append(
        "<p><strong>2. prob McNemar</strong> — flip знака margin "
        "(<code>p_man&nbsp;&gt;&nbsp;p_woman</code>):</p>"
    )
    if pr_man:
        parts.append(_mcn_paragraph(pr_man, label="no→man, prob (margin)"))
        parts.append(f"<p class='note'>→ {_verdict_phrase(pr_man)}.</p>")
    if pr_wom:
        parts.append(_mcn_paragraph(pr_wom, label="no→woman, prob (margin)"))
        parts.append(f"<p class='note'>→ {_verdict_phrase(pr_wom)}.</p>")

    parts.append(
        "<p>" + _pooled_verdict([ch_man, ch_wom, pr_man, pr_wom]) + "</p>"
    )
    return "\n".join(parts)


def _rates_from_df(
    rates: pd.DataFrame | None,
    *,
    abstain_variant: str | None = None,
) -> list[dict[str, Any]]:
    if rates is None or not len(rates):
        return []
    if abstain_variant:
        sub = rates.loc[rates["slice"] == "by_abstain_evidence"]
        sub = sub.loc[sub["abstain_variant"] == abstain_variant]
    else:
        sub = rates.loc[rates["slice"] == "by_evidence"]
    out: list[dict[str, Any]] = []
    for _, row in sub.iterrows():
        ev = str(row.get("evidence_shift", ""))
        out.append({
            "evidence": ev,
            "n": int(row["n"]),
            "p_man": _pct(float(row["rate_man"])),
            "p_woman": _pct(float(row["rate_woman"])),
            "p_abstain": _pct(float(row["rate_abstain"])),
            "k_man": int(row["k_man"]),
            "k_woman": int(row["k_woman"]),
            "k_abstain": int(row["k_abstain"]),
        })
    order = {"no_evidence": 0, "man": 1, "woman": 2}
    out.sort(key=lambda x: order.get(x["evidence"], 99))
    return out


def _render_rates_table(rows: list[dict[str, Any]], *, show_abstain: bool) -> str:
    if not rows:
        return "<p class='note'>—</p>"
    hdr = (
        "<tr><th>evidence</th><th>n</th><th>k_man</th><th>k_woman</th>"
        + ("<th>k_abstain</th>" if show_abstain else "")
        + "<th>P(man)</th><th>P(woman)</th>"
        + ("<th>P(abstain)</th>" if show_abstain else "")
        + "</tr>"
    )
    body = []
    for d in rows:
        cells = [
            f"<td><code>{html.escape(d['evidence'])}</code></td>",
            f"<td>{d['n']}</td>",
            f"<td>{d['k_man']}</td>",
            f"<td>{d['k_woman']}</td>",
        ]
        if show_abstain:
            cells.append(f"<td>{d['k_abstain']}</td>")
        cells.extend([
            f"<td>{d['p_man']:.1f}%</td>",
            f"<td>{d['p_woman']:.1f}%</td>",
        ])
        if show_abstain:
            cells.append(f"<td>{d['p_abstain']:.1f}%</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table class='mcn'>{hdr}{''.join(body)}</table>"


def _build_rates_without_interpret(rows: list[dict[str, Any]]) -> str:
    by_ev = {r["evidence"]: r for r in rows}
    no = by_ev.get("no_evidence")
    man = by_ev.get("man")
    wom = by_ev.get("woman")
    if not (no and man and wom):
        return ""
    return (
        "<strong>Сдвиг (without_abstain).</strong> "
        f"Woman-evidence заметно повышает P(woman): "
        f"<b>{no['p_woman']:.1f}%</b> → <b>{wom['p_woman']:.1f}%</b>. "
        f"Man-evidence почти не двигает P(man): "
        f"<b>{no['p_man']:.1f}%</b> → <b>{man['p_man']:.1f}%</b>. "
        "Это <b>описательные</b> доли; <b>значимость</b> парных сдвигов проверяется "
        "в разделе <a href='#primary'>Primary</a> (paired McNemar по <code>id</code>), "
        "а не на marginal-долях здесь."
    )


def _build_rates_with_interpret(rows: list[dict[str, Any]]) -> str:
    by_ev = {r["evidence"]: r for r in rows}
    no = by_ev.get("no_evidence")
    man = by_ev.get("man")
    wom = by_ev.get("woman")
    if not (no and man and wom):
        return ""
    return (
        "<strong>Воздержание (abstain).</strong> "
        f"При evidence <b>за мужчин</b> доля воздержавшихся ответов "
        f"<b>{man['p_abstain']:.1f}%</b> — гораздо больше, чем при evidence "
        f"<b>за женщин</b> <b>{wom['p_abstain']:.1f}%</b>: модель заметно реже "
        f"отказывается отвечать, когда подсвечена женщина.<br>"
        f"При этом доля ответа «воздержаться» <b>снижается при наличии любого</b> "
        f"evidence относительно ambiguous baseline: "
        f"<b>{no['p_abstain']:.1f}%</b> (no_evidence) → "
        f"<b>{man['p_abstain']:.1f}%</b> (man) / "
        f"<b>{wom['p_abstain']:.1f}%</b> (woman) — <b>что подтверждает H5</b>: "
        f"добавление варианта ответа «воздержаться» приводит к большей вероятности "
        f"выбора «воздержаться» в контексте <b>без</b> evidence, чем в контексте "
        f"<b>с</b> evidence."
    )


def _soc_short_label(title: str) -> str:
    """Compact but distinct y-axis label (full title in tooltip)."""
    parts = title.replace(" Occupations", "").split(", ")
    if len(parts) >= 2 and len(parts[0]) <= 28:
        return parts[0]
    if len(title) <= 36:
        return title
    return title[:34] + "…"


def _build_soc_prob_rows(results: list[TestResult]) -> list[dict[str, Any]]:
    """Per soc_major_title prob McNemar effect for no→man and no→woman."""
    by_title: dict[str, dict[str, Any]] = {}
    for r in results:
        if r.family_name != "h3v1_strata_soc_major_prob":
            continue
        ex = r.extra or {}
        title = str(ex.get("soc_major_title", ""))
        if not title:
            continue
        axis = str(ex.get("gender_axis", ""))
        d = by_title.setdefault(
            title, {"title": title, "short": _soc_short_label(title)}
        )
        eff = round(float(r.effect), 4) if r.effect is not None else None
        n = int(ex.get("n_pairs") or r.n1 or 0)
        q = r.q_value
        sig = bool(r.rejected_fdr)
        if axis == "man":
            d["eff_man"], d["p_man"], d["n_man"] = eff, r.p_raw, n
            d["q_man"], d["sig_man"] = q, sig
        elif axis == "woman":
            d["eff_woman"], d["p_woman"], d["n_woman"] = eff, r.p_raw, n
            d["q_woman"], d["sig_woman"] = q, sig

    def _key(d: dict[str, Any]) -> float:
        vals = [v for v in (d.get("eff_man"), d.get("eff_woman")) if v is not None]
        return sum(vals) / len(vals) if vals else float("-inf")

    rows = sorted(by_title.values(), key=_key, reverse=True)
    return rows


def _names(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "—"
    return ", ".join(html.escape(str(d.get("short", d.get("title", "")))) for d in rows)


def _build_soc_fdr_html(soc_rows: list[dict[str, Any]], *, fdr: float) -> str:
    """FDR (BH) verdicts + per-domain q-value table for soc prob McNemar."""
    if not soc_rows:
        return ""
    n = len(soc_rows)

    def _cls(rows: list[dict[str, Any]], axis: str, sign: int) -> list[dict[str, Any]]:
        out = []
        for d in rows:
            if not d.get(f"sig_{axis}"):
                continue
            eff = d.get(f"eff_{axis}") or 0
            if (sign > 0 and eff > 0) or (sign < 0 and eff < 0):
                out.append(d)
        out.sort(key=lambda d: abs(d.get(f"eff_{axis}") or 0), reverse=True)
        return out

    w_pos = _cls(soc_rows, "woman", +1)
    w_neg = _cls(soc_rows, "woman", -1)
    w_ns = [d for d in soc_rows if not d.get("sig_woman")]
    m_pos = _cls(soc_rows, "man", +1)
    m_neg = _cls(soc_rows, "man", -1)
    m_ns = [d for d in soc_rows if not d.get("sig_man")]

    concl = (
        "<div class='interpret'>"
        f"<strong>Выводы (BH/FDR, Q={fdr}; семья — soc prob, {2 * n} тестов):</strong>"
        "<ul>"
        f"<li><b>Женский evidence (no→woman):</b> значимый <b>положительный</b> эффект "
        f"в <b>{len(w_pos)}/{n}</b> отраслях"
        + (f" (незначимо: {_names(w_ns)})" if w_ns else "")
        + (f"; значимо <b>отрицательных</b>: {_names(w_neg)}" if w_neg else "")
        + ".</li>"
        f"<li><b>Мужской evidence (no→man):</b> значимо <b>положительный</b> в: "
        f"{_names(m_pos)}; значимо <b>отрицательный</b> в: {_names(m_neg)}; "
        f"незначимо: {_names(m_ns)}.</li>"
        "</ul>"
        "<p class='note' style='margin:0'>«Значимо» = q-value ≤ Q после поправки "
        "Бенджамини–Хохберга по всей soc-prob семье (man+woman). Знак — по "
        "<code>effect=(b−c)/n_pairs</code>.</p>"
        "</div>"
    )

    body: list[str] = []
    for d in soc_rows:
        def cell(axis: str) -> str:
            eff = d.get(f"eff_{axis}")
            q = d.get(f"q_{axis}")
            sig = d.get(f"sig_{axis}")
            eff_s = f"{eff * 100:+.1f}" if eff is not None else "—"
            q_s = _fmt_p(q)
            mark = (
                "<span class='fdr-yes'>✓</span>"
                if sig
                else "<span class='fdr-no'>✗</span>"
            )
            return f"<td>{eff_s}</td><td>{q_s}</td><td>{mark}</td>"

        body.append(
            "<tr>"
            f"<td class='mcn-key' title=\"{html.escape(str(d.get('title', '')))}\">"
            f"{html.escape(str(d.get('short', '')))}</td>"
            + cell("man")
            + cell("woman")
            + "</tr>"
        )

    table = (
        "<details class='method-spoiler'><summary>FDR-таблица по отраслям "
        "(effect п.п. · q-value · значимо)</summary><div class='spoiler-body'>"
        "<table class='mcn'><thead><tr>"
        "<th>soc_major_title</th>"
        "<th>man eff</th><th>man q</th><th>man</th>"
        "<th>woman eff</th><th>woman q</th><th>woman</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div></details>"
    )
    return concl + table


def _setup_stats(df: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {"n_rows": len(df)}
    if "evidence_shift" in df.columns:
        stats["evidence_counts"] = {
            str(k): int(v)
            for k, v in df["evidence_shift"].value_counts().sort_index().items()
        }
    if "profession" in df.columns:
        stats["n_professions"] = int(df["profession"].nunique())
    if "soc_major_title" in df.columns:
        stats["n_soc_major"] = int(df["soc_major_title"].nunique())
    if "context_order" in df.columns:
        stats["context_orders"] = sorted(df["context_order"].astype(str).unique())
    if "abstain_variant" in df.columns:
        stats["abstain_variants"] = sorted(df["abstain_variant"].astype(str).unique())
    return stats


def _build_run_setup_html(meta: dict[str, Any] | None, stats: dict[str, Any]) -> str:
    meta = meta or {}
    model = html.escape(str(meta.get("model_id") or "—"))
    n_rows = stats.get("n_rows", "—")
    ev_counts = stats.get("evidence_counts") or meta.get("evidence_shift_counts") or {}

    ev_lines = ", ".join(
        f"<code>{html.escape(k)}</code>: {int(v)}"
        for k, v in sorted(ev_counts.items())
    )

    brief = (
        f"<p><b>Модель:</b> <code>{model}</code></p>"
        f"<p><b>Гипотеза H3:</b> фраза «A report highlighted the <i>man/woman</i>» "
        f"сдвигает предпочтение модели к highlighted полу относительно ambiguous baseline "
        f"(<code>no_evidence</code>).</p>"
        f"<p><b>Датасет v1 + highlight:</b> те же occupation-сценарии, что в H1 v1; "
        f"три условия evidence на одних и тех же <code>id</code> (15 216 пар).</p>"
        f"<p><b>Объём:</b> <b>{n_rows}</b> промптов — {ev_lines or '—'}.</p>"
    )

    merged = meta.get("merged_from")
    merge_block = ""
    if isinstance(merged, list) and merged:
        items = "".join(f"<li><code>{html.escape(str(m))}</code></li>" for m in merged)
        ev_map = meta.get("evidence_shift_by_source") or {}
        ev_items = "".join(
            f"<li><code>{html.escape(str(k))}</code> → "
            f"<code>{html.escape(str(v))}</code></li>"
            for k, v in ev_map.items()
        )
        merge_block = (
            "<p><b>Как собран run:</b> concat baseline + 4 highlight-прогона.</p>"
            f"<ul>{items}</ul>"
            f"<p><b>evidence_shift по источникам:</b></p><ul>{ev_items}</ul>"
            f"<p class='note'>{html.escape(str(meta.get('merge_order') or ''))}</p>"
        )

    ctx = ", ".join(
        f"<code>{html.escape(c)}</code>" for c in stats.get("context_orders", [])
    )
    abst = ", ".join(
        f"<code>{html.escape(a)}</code>" for a in stats.get("abstain_variants", [])
    )
    design = (
        "<table>"
        "<tr><th>Фактор</th><th>Уровни</th><th>Смысл</th></tr>"
        f"<tr><td><code>evidence_shift</code></td>"
        f"<td><code>no_evidence</code>, <code>man</code>, <code>woman</code></td>"
        "<td>ambiguous vs highlight man vs highlight woman</td></tr>"
        f"<tr><td><code>context_order</code></td><td>{ctx or '—'}</td>"
        "<td>порядок имён в narrative</td></tr>"
        f"<tr><td><code>abstain_variant</code></td><td>{abst or '—'}</td>"
        "<td>2 или 3 опции ответа</td></tr>"
        "</table>"
    )

    details = merge_block + design
    return (
        brief
        + '<details class="method-spoiler">'
        + "<summary>Детали merge и факторы</summary>"
        + f'<div class="spoiler-body">{details}</div>'
        + "</details>"
    )


def _build_payload(
    results: list[TestResult],
    *,
    run_name: str,
    n_rows: int,
    rates: pd.DataFrame | None,
    fdr: float = 0.05,
) -> dict[str, Any]:
    primary = _build_primary_rows({r.test_id: r for r in results})

    evidence_rates_without = _rates_from_df(rates, abstain_variant="without_abstain")
    evidence_rates_with = _rates_from_df(rates, abstain_variant="with_abstain")
    soc = _build_soc_prob_rows(results)
    soc_chart_h = max(420, len(soc) * 34 + 90)

    return {
        "run": run_name,
        "n_rows": n_rows,
        "generated": datetime.now(timezone.utc).isoformat(),
        "evidence_rates_without": evidence_rates_without,
        "evidence_rates_with": evidence_rates_with,
        "soc": soc,
        "n_soc": len(soc),
        "soc_chart_h": soc_chart_h,
        "soc_fdr_html": _build_soc_fdr_html(soc, fdr=fdr),
        "primary": primary,
        "primary_table_html": _render_primary_table(primary),
        "primary_interpret_html": _build_primary_interpret_html(
            {r.test_id: r for r in results}
        ),
        "rates_without_table_html": _render_rates_table(
            evidence_rates_without, show_abstain=False
        ),
        "rates_without_interpret_html": _build_rates_without_interpret(
            evidence_rates_without
        ),
        "rates_with_table_html": _render_rates_table(
            evidence_rates_with, show_abstain=True
        ),
        "rates_with_interpret_html": _build_rates_with_interpret(evidence_rates_with),
    }


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>H3 v1 — {run}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {{
  --accent: #2c5f8d; --bg: #f6f8fa; --card: #fff; --border: #dde3ea;
  --muted: #5a6570; --ch-dark: #2c5f8d; --ch-light: #9bb8d4;
}}
* {{ box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  margin: 0; padding: 0 1.25rem 3rem; background: var(--bg); color: #1a1a1a;
  line-height: 1.55; max-width: 1100px; margin-inline: auto;
}}
header {{ padding: 1.5rem 0 1rem; border-bottom: 3px solid var(--accent); }}
h1 {{ margin: 0 0 .35rem; color: var(--accent); font-size: 1.65rem; }}
h3 {{ margin: 1rem 0 .5rem; color: var(--accent); font-size: 1rem; }}
.meta {{ color: var(--muted); font-size: .92rem; }}
nav.toc {{
  background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 1rem 1.25rem; margin: 1.25rem 0;
}}
nav.toc a {{ color: var(--accent); text-decoration: none; }}
nav.toc ul {{ margin: .5rem 0 0; padding-left: 1.2rem; }}
section {{
  background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 1.25rem 1.35rem; margin: 1.5rem 0;
}}
section h2 {{
  margin: 0 0 .75rem; color: var(--accent); font-size: 1.2rem;
  border-bottom: 1px solid var(--border); padding-bottom: .4rem;
}}
section.empty-section {{ opacity: .75; }}
.method {{
  background: #f0f4f8; border: 1px solid var(--border);
  border-radius: 8px; padding: 1.1rem 1.25rem; margin: 1.25rem 0; font-size: .93rem;
}}
.method h2 {{ margin-top: 0; border: none; padding: 0; }}
.method table {{ border-collapse: collapse; width: 100%; margin: .5rem 0; font-size: .88rem; }}
.method th, .method td {{ border: 1px solid var(--border); padding: .4rem .6rem; text-align: left; }}
.method th {{ background: #e8eef4; }}
.interpret {{
  background: #eef4fb; border-left: 4px solid var(--accent);
  padding: .85rem 1rem; margin: 0 0 1rem; border-radius: 0 6px 6px 0; font-size: .95rem;
}}
.interpret strong {{ color: var(--accent); }}
.interpret ul {{ margin: .35rem 0 .5rem; padding-left: 1.25rem; }}
.interpret table {{ border-collapse: collapse; width: 100%; margin: .5rem 0; font-size: .88rem; }}
.interpret th, .interpret td {{ border: 1px solid var(--border); padding: .35rem .5rem; text-align: left; }}
.interpret th {{ background: #e8eef4; }}
.chart-wrap {{ position: relative; height: 300px; margin: .5rem 0 1rem; }}
.chart-wrap.tall {{ height: 360px; }}
.note {{ font-size: .88rem; color: var(--muted); margin-top: .5rem; }}
table.mcn {{
  border-collapse: collapse; width: 100%; font-size: .82rem; margin: .75rem 0;
}}
table.mcn th, table.mcn td {{
  border: 1px solid var(--border); padding: .35rem .5rem; text-align: right;
}}
table.mcn th {{ background: #f0f4f8; color: var(--accent); text-align: center; }}
table.mcn td.mcn-key {{ text-align: left; font-weight: 600; }}
table.mcn tr.mcn-grp-a {{ background: #e8f0f8; }}
table.mcn tr.mcn-grp-b {{ background: #f7fafc; }}
.method-spoiler, .test-ref-spoiler {{
  margin: .75rem 0; border: 1px solid var(--border); border-radius: 8px; background: #fafbfc;
}}
.method-spoiler summary, .test-ref-spoiler summary {{
  cursor: pointer; padding: .6rem .85rem; font-weight: 600; color: var(--accent);
  list-style: none; font-size: .9rem;
}}
.method-spoiler summary::-webkit-details-marker, .test-ref-spoiler summary::-webkit-details-marker {{ display: none; }}
.method-spoiler summary::before, .test-ref-spoiler summary::before {{
  content: '▸'; display: inline-block; margin-right: .4rem;
}}
.method-spoiler[open] summary::before, .test-ref-spoiler[open] summary::before {{ transform: rotate(90deg); }}
.spoiler-body {{ padding: .35rem .85rem .75rem; border-top: 1px solid var(--border); font-size: .88rem; }}
.test-refs {{ display: flex; flex-direction: column; gap: .35rem; margin: .35rem 0 .75rem; }}
table.tip-fdr {{ width: 100%; border-collapse: collapse; font-size: .82rem; }}
table.tip-fdr th, table.tip-fdr td {{ border: 1px solid var(--border); padding: .28rem .4rem; }}
table.tip-fdr td.tip-p, table.tip-fdr td.tip-q {{ text-align: right; font-family: ui-monospace, monospace; }}
table.tip-fdr td.fdr-yes {{ color: #6fcf97; text-align: center; font-weight: 700; }}
table.tip-fdr td.fdr-no {{ color: #eb5757; text-align: center; font-weight: 700; }}
.effect-key {{
  background: #fff7ed; border: 1px solid #f0c9a8; border-left: 4px solid var(--ch-accent, #c45c3e);
  border-radius: 0 8px 8px 0; padding: .9rem 1.1rem; margin: 0 0 1rem;
  font-size: 1.02rem; line-height: 1.6;
}}
.effect-key code {{ background: #fde8d8; padding: .05rem .3rem; border-radius: 4px; }}
.chart-toggle {{ font-size: .9rem; margin: .35rem 0; }}
.chart-toggle label {{ margin-right: 1.25rem; cursor: pointer; }}
.fdr-yes {{ color: #2e8b57; font-weight: 700; }}
.fdr-no {{ color: #c0392b; font-weight: 700; }}
footer {{ margin-top: 2rem; font-size: .85rem; color: var(--muted); }}
</style>
</head>
<body>
<header>
  <h1>H3 v1 — evidence highlight</h1>
  <p class="meta">
    Run: <code>{run}</code> · n={n_rows} · {generated}
  </p>
</header>

<nav class="toc">
  <b>Содержание</b>
  <ul>
    <li><a href="#setup">Сетап прогона</a></li>
    <li><a href="#method">Как читать отчёт</a></li>
    <li><a href="#primary">Primary — paired McNemar</a></li>
    <li><a href="#rates">Rates по evidence_shift</a></li>
    <li><a href="#soc">Strata — soc_major_title</a></li>
  </ul>
</nav>

<section class="method" id="setup">
  <h2>Сетап прогона</h2>
  {run_setup_html}
</section>

<section class="method" id="method">
  <h2>Как читать отчёт</h2>
  <h3>H3 — гипотеза</h3>
  <p>Highlight пола X должен <b>увеличить</b> предпочтение X относительно ambiguous baseline
  (<code>no_evidence</code>). Три условия evidence на одних и тех же <code>id</code>
  (15 216 layout-промптов × 3 evidence = 45 648 строк).</p>
  <h3>Primary — choice vs prob (McNemar)</h3>
  <p class="note">{mcnemar_slice_note}</p>
  <p>Как в H1 v1: <b>choice</b> = hard argmax по слотам (man/woman);
  <b>prob</b> = <code>p_man</code>, <code>p_woman</code> без повторного argmax.</p>
  <ol>
    <li><b>choice McNemar</b> — flip argmax-ответа; b/c.</li>
    <li><b>prob McNemar</b> — flip знака <code>p_man − p_woman</code>; b/c.</li>
  </ol>
  <h3>Rates</h3>
  <p><b>Marginal доли</b> дискретного choice (argmax man / woman / abstain) по уровням
  <code>evidence_shift</code>. Считаются отдельно для двух layout:</p>
  <ul>
    <li><code>without_abstain</code> — только man и woman (2 опции).</li>
    <li><code>with_abstain</code> — man, woman и abstain (3 опции).</li>
  </ul>
</section>

<section id="primary">
  <h2>Primary — paired McNemar (without_abstain)</h2>
  <p class="note">{mcnemar_slice_note}</p>
  <div class="interpret">
    {primary_interpret_html}
  </div>
  {primary_table_html}
  <div class="chart-wrap"><canvas id="chartPrimary"></canvas></div>
  <p class="note"><b>Тест:</b> парный <b>McNemar</b> (χ², поправка Йейтса) по дискордантным
  парам b/c, <b>односторонний</b> H₃↑. Цель <code>choice</code> — flip argmax (man/woman);
  цель <code>prob</code> — flip знака margin <code>p_man − p_woman</code>. Значения p — в таблице выше.</p>
</section>

<section id="rates">
  <h2>Rates по evidence_shift</h2>
  <div class="interpret">
    <strong>Marginal доли</strong> choice (argmax) по трём условиям evidence,
    отдельно для layout без abstain и с abstain.
    Baseline <code>no_evidence</code> совпадает с H1 v1 primary (~26% man, ~23% woman, ~51% abstain pooled).
  </div>

  <h3>without_abstain — man / woman</h3>
  {rates_without_table_html}
  <div class="interpret">{rates_without_interpret_html}</div>
  <div class="chart-wrap"><canvas id="chartRatesWithout"></canvas></div>
  <p class="note"><b>Тест:</b> нет — это <b>описательные marginal-доли</b> argmax-choice
  (P(man) / P(woman)) по условиям evidence, без статистики значимости.</p>

  <h3>with_abstain — man / woman / abstain</h3>
  {rates_with_table_html}
  <div class="interpret">{rates_with_interpret_html}</div>
  <div class="chart-wrap tall"><canvas id="chartRatesWith"></canvas></div>
  <p class="note"><b>Тест:</b> нет — <b>описательные marginal-доли</b> argmax-choice
  (man / woman / abstain) по условиям evidence, без статистики значимости.</p>
</section>

<section id="soc">
  <h2>Strata — soc_major_title ({n_soc} отраслей, without_abstain)</h2>
  <p class="note">{mcnemar_slice_note}</p>
  <div class="interpret">
    <strong>prob McNemar по отраслям.</strong> Для каждого <code>soc_major_title</code> —
    знаковый <code>effect=(b−c)/n_pairs</code> на margin (<code>p_man − p_woman</code>),
    отдельно для <b>no→man</b> и <b>no→woman</b>.
    Плюс = сдвиг <b>к</b> highlighted полу (H₃). Полное название отрасли — в tooltip.
  </div>
  <div class="chart-toggle">
    <span><b>Сортировка:</b></span>
    <label><input type="radio" name="socSort" value="effect" checked> по effect (↓)</label>
    <label><input type="radio" name="socSort" value="name"> по названию (A→Я)</label>
  </div>
  <div class="chart-wrap" style="height:{soc_chart_h}px"><canvas id="chartSoc"></canvas></div>
  <p class="note"><b>Тест:</b> <b>двусторонний</b> парный <b>McNemar</b> (χ², поправка Йейтса)
  по дискордантным парам; цель <code>prob</code> — знак margin <code>p_man − p_woman</code>.
  Двусторонний выбран, чтобы p оставался читаемым и для отраслей с эффектом <i>против</i> H₃
  (отрицательный <code>effect</code>, напр. man-ось в Legal / Architecture): знак
  <code>effect</code> задаёт направление, p — значимость сдвига в любую сторону.</p>
  {soc_fdr_html}
</section>

<footer>Сгенерировано pipeline <code>h3_v1</code>.</footer>

<script>
const DATA = {data_json};
const CH_DARK = '{ch_dark}';
const CH_LIGHT = '{ch_light}';
const CH_ACCENT = '{ch_accent}';
const CH_GREEN = '{ch_green}';

// Primary McNemar b/c (choice + prob margin)
const primMcn = (DATA.primary || []).filter(d => d.kind === 'mcnemar');
if (primMcn.length) {{
  new Chart(document.getElementById('chartPrimary'), {{
    type: 'bar',
    data: {{
      labels: primMcn.map(d => d.label),
      datasets: [
        {{ label: 'b (toward highlight)', data: primMcn.map(d => d.b), backgroundColor: CH_GREEN }},
        {{ label: 'c (away)', data: primMcn.map(d => d.c), backgroundColor: CH_ACCENT }},
      ],
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ title: {{ display: true, text: 'McNemar b/c: choice (argmax) + prob (margin)' }} }},
    }},
  }});
}}

// Rates by evidence — without_abstain
if (DATA.evidence_rates_without.length) {{
  const labels = DATA.evidence_rates_without.map(d => d.evidence);
  new Chart(document.getElementById('chartRatesWithout'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [
        {{ label: 'P(man) %', data: DATA.evidence_rates_without.map(d => d.p_man), backgroundColor: CH_DARK }},
        {{ label: 'P(woman) %', data: DATA.evidence_rates_without.map(d => d.p_woman), backgroundColor: CH_LIGHT }},
      ],
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ title: {{ display: true, text: 'without_abstain: choice rates by evidence' }} }},
      scales: {{ y: {{ beginAtZero: true, max: 100, title: {{ display: true, text: '%' }} }} }},
    }},
  }});
}}

// Rates by evidence — with_abstain
if (DATA.evidence_rates_with.length) {{
  const labels = DATA.evidence_rates_with.map(d => d.evidence);
  new Chart(document.getElementById('chartRatesWith'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [
        {{ label: 'P(man) %', data: DATA.evidence_rates_with.map(d => d.p_man), backgroundColor: CH_DARK }},
        {{ label: 'P(woman) %', data: DATA.evidence_rates_with.map(d => d.p_woman), backgroundColor: CH_LIGHT }},
        {{ label: 'P(abstain) %', data: DATA.evidence_rates_with.map(d => d.p_abstain), backgroundColor: CH_ACCENT }},
      ],
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ title: {{ display: true, text: 'with_abstain: choice rates by evidence' }} }},
      scales: {{ y: {{ beginAtZero: true, max: 100, title: {{ display: true, text: '%' }} }} }},
    }},
  }});
}}

// Strata — soc_major_title: prob McNemar effect (no->man / no->woman)
const SOC_ROWS = DATA.soc || [];
let socSortMode = 'effect';
let chartSoc = null;

function _socKey(d) {{
  const v = [d.eff_man, d.eff_woman].filter(x => x != null);
  return v.length ? v.reduce((s, x) => s + x, 0) / v.length : Number.NEGATIVE_INFINITY;
}}

function sortedSoc() {{
  const rows = [...SOC_ROWS];
  if (socSortMode === 'name') {{
    rows.sort((a, b) => String(a.title).localeCompare(String(b.title), 'ru'));
  }} else {{
    rows.sort((a, b) => _socKey(b) - _socKey(a));
  }}
  return rows;
}}

function _socP(p) {{ return (p == null) ? '—' : Number(p).toExponential(2); }}

function buildSocChart() {{
  if (!SOC_ROWS.length) return;
  if (chartSoc) chartSoc.destroy();
  const rows = sortedSoc();
  const titles = rows.map(d => d.title);
  chartSoc = new Chart(document.getElementById('chartSoc'), {{
    type: 'bar',
    data: {{
      labels: rows.map(d => d.short),
      datasets: [
        {{ label: 'no→man (prob effect, п.п.)',
           data: rows.map(d => d.eff_man != null ? d.eff_man * 100 : null),
           backgroundColor: CH_DARK }},
        {{ label: 'no→woman (prob effect, п.п.)',
           data: rows.map(d => d.eff_woman != null ? d.eff_woman * 100 : null),
           backgroundColor: CH_LIGHT }},
      ],
    }},
    options: {{
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: {{
        title: {{ display: true,
          text: 'soc_major_title: prob McNemar effect (+ → к highlighted полу, H₃)' }},
        legend: {{ position: 'bottom' }},
        tooltip: {{ callbacks: {{
          title: (items) => titles[items[0].dataIndex] || items[0].label,
          afterBody: (items) => {{
            const d = rows[items[0].dataIndex];
            if (!d) return [];
            return [
              `no→man:   effect=${{d.eff_man != null ? (d.eff_man * 100).toFixed(1) : '—'}} п.п., p=${{_socP(d.p_man)}}, n=${{d.n_man || '—'}}`,
              `no→woman: effect=${{d.eff_woman != null ? (d.eff_woman * 100).toFixed(1) : '—'}} п.п., p=${{_socP(d.p_woman)}}, n=${{d.n_woman || '—'}}`,
            ];
          }},
        }} }},
      }},
      scales: {{ x: {{ title: {{ display: true, text: 'effect (п.п.)' }} }} }},
    }},
  }});
}}

buildSocChart();
document.querySelectorAll('input[name="socSort"]').forEach(el => {{
  el.addEventListener('change', (e) => {{ socSortMode = e.target.value; buildSocChart(); }});
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
    rates: pd.DataFrame | None = None,
) -> None:
    payload = _build_payload(
        results,
        run_name=run_dir.name,
        n_rows=n_rows_metrics,
        rates=rates,
        fdr=fdr,
    )
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
        run=html.escape(run_dir.name),
        n_rows=n_rows_metrics,
        generated=payload["generated"][:19].replace("T", " "),
        run_setup_html=run_setup_html,
        primary_table_html=payload["primary_table_html"],
        primary_interpret_html=payload["primary_interpret_html"],
        rates_without_table_html=payload["rates_without_table_html"],
        rates_without_interpret_html=payload["rates_without_interpret_html"],
        rates_with_table_html=payload["rates_with_table_html"],
        rates_with_interpret_html=payload["rates_with_interpret_html"],
        n_soc=payload["n_soc"],
        soc_chart_h=payload["soc_chart_h"],
        soc_fdr_html=payload["soc_fdr_html"],
        mcnemar_slice_note=MCNEMAR_SLICE_NOTE,
        data_json=_j(payload),
        ch_dark=CH_DARK,
        ch_light=CH_LIGHT,
        ch_accent=CH_ACCENT,
        ch_green=CH_GREEN,
    )
    path.write_text(html_out, encoding="utf-8")
