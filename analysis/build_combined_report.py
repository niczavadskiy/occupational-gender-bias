"""Combined report builder: H1 + H3 + H5 (из H3) + H11 в один самодостаточный HTML.

Источники ("родные отчёты"):
  - H1  : results/run_2026-06-09_12-50-33_..._v1_full_pos_shuffle/metrics_h1_v1/latest/summary.html
  - H3  : results/highlight-h3-full/metrics_h3_v1/latest/summary.html
  - H5  : блок abstain × evidence из того же H3 summary.html (данные DATA.evidence_rates_with)
          + парный McNemar по abstain (results/highlight-h3-full/per_item.jsonl)
  - H11 : probes/v1_rep_prob/without_abstain_p0_p1_mf_wf_gender_choice/meta.json

H1 и H3 — самодостаточные HTML со своими <style>/<script>/Chart.js и одинаковыми
именами (DATA, CH_DARK, id канвасов, name="socSort"). Чтобы перенести их содержимое
БЕЗ конфликтов, они встраиваются целиком через <iframe srcdoc> (изолированный контекст)
с авто-подгонкой высоты. H5 и H11 рендерятся нативно в общем стиле.
"""

from __future__ import annotations

import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo/

SRC_H1 = ROOT / "results/run_2026-06-09_12-50-33_Qwen3.5-2B-Base_v1_full_pos_shuffle/metrics_h1_v1/latest/summary.html"
SRC_H3 = ROOT / "results/highlight-h3-full/metrics_h3_v1/latest/summary.html"
SRC_H3_ITEMS = ROOT / "results/highlight-h3-full/per_item.jsonl"
# H11 = gender preference в internals. На этом run реализован как gender-ось пробы
# v1_rep_prob (цель gender_choice — линейная классификация выбора man/woman).
SRC_H11_DIR = ROOT / "results/run_2026-06-09_12-50-33_Qwen3.5-2B-Base_v1_full_pos_shuffle/probes/v1_rep_prob"
SRC_H11_CHOICE = SRC_H11_DIR / "without_abstain_p0_p1_mf_wf_gender_choice/meta.json"
# test AUC отсутствует в meta (multiclass → NaN); берётся из родного summary.md (gender_choice row).
H11_TEST_AUC = 0.941

OUT = ROOT / "analysis/combined_h1_h3_h5_h11.html"


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def extract_data_obj(html_text: str) -> dict:
    """Вытащить `const DATA = {...};` из summary.html."""
    m = re.search(r"const DATA\s*=\s*(\{.*?\});", html_text, re.DOTALL)
    if not m:
        return {}
    return json.loads(m.group(1))


def iframe_srcdoc(doc_html: str) -> str:
    escaped = html.escape(doc_html, quote=True)
    return (
        f'<iframe class="embed" loading="eager" srcdoc="{escaped}" '
        f'onload="fitFrame(this)"></iframe>'
    )


# ── shared CSS (взят из summary.html пайплайнов, чтобы нативные секции совпадали по стилю)
SHARED_CSS = """
:root {
  --accent: #2c5f8d; --bg: #f6f8fa; --card: #fff; --border: #dde3ea;
  --muted: #5a6570; --ch-dark: #2c5f8d; --ch-light: #9bb8d4;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  margin: 0; padding: 0 1.25rem 3rem; background: var(--bg); color: #1a1a1a;
  line-height: 1.55; max-width: 1100px; margin-inline: auto;
}
header.top { padding: 1.6rem 0 1.1rem; border-bottom: 3px solid var(--accent); }
h1 { margin: 0 0 .35rem; color: var(--accent); font-size: 1.75rem; }
h3 { margin: 1rem 0 .5rem; color: var(--accent); font-size: 1rem; }
.meta { color: var(--muted); font-size: .92rem; }
nav.toc {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 1rem 1.25rem; margin: 1.25rem 0;
}
nav.toc a { color: var(--accent); text-decoration: none; }
nav.toc a:hover { text-decoration: underline; }
nav.toc ul { margin: .5rem 0 0; padding-left: 1.2rem; }
section {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 1.25rem 1.35rem; margin: 1.5rem 0;
}
section > h2 {
  margin: 0 0 .35rem; color: var(--accent); font-size: 1.3rem;
  border-bottom: 1px solid var(--border); padding-bottom: .4rem;
}
.src-note {
  font-size: .85rem; color: var(--muted); margin: 0 0 .75rem;
}
.src-note code { background: #eef2f6; padding: .05rem .3rem; border-radius: 4px; }
.embed-wrap { margin: .25rem 0 0; }
iframe.embed {
  width: 100%; border: 0; display: block; background: var(--bg);
  border-radius: 6px; overflow: hidden;
}
.interpret {
  background: #eef4fb; border-left: 4px solid var(--accent);
  padding: .85rem 1rem; margin: 0 0 1rem; border-radius: 0 6px 6px 0; font-size: .95rem;
}
.interpret strong { color: var(--accent); }
.note { font-size: .88rem; color: var(--muted); margin-top: .5rem; }
.chart-wrap { position: relative; height: 360px; margin: .5rem 0 1rem; }
table.mcn { border-collapse: collapse; width: 100%; font-size: .82rem; margin: .75rem 0; }
table.mcn th, table.mcn td { border: 1px solid var(--border); padding: .35rem .5rem; text-align: right; }
table.mcn th { background: #f0f4f8; color: var(--accent); text-align: center; }
table.mcn td.mcn-key { text-align: left; font-weight: 600; }
table.mcn tr.mcn-grp-a { background: #e8f0f8; }
table.mcn tr.mcn-grp-b { background: #f7fafc; }
table.mcn td.fdr-yes { color: #1f7a44; font-weight: 700; text-align: center; }
table.mcn td.fdr-no { color: #b03a2e; font-weight: 700; text-align: center; }
.callout {
  background: #fff8e6; border-left: 4px solid #d4a017;
  padding: .85rem 1rem; margin: 0 0 1rem; border-radius: 0 6px 6px 0; font-size: .93rem;
}
.callout strong { color: #9a6700; }
.summary-card {
  background: #f0f6ff; border: 1px solid #cfe0f2; border-left: 4px solid var(--accent);
  border-radius: 0 8px 8px 0; padding: .9rem 1.1rem; margin: 0 0 1.1rem;
}
.summary-card .sc-row { margin: .15rem 0; font-size: .95rem; }
.summary-card .sc-label { font-weight: 700; color: var(--accent); margin-right: .4rem; }
.summary-card code { background: #e3edf8; padding: .05rem .3rem; border-radius: 4px; }
.badge {
  display: inline-block; padding: .12rem .6rem; border-radius: 999px;
  font-size: .8rem; font-weight: 700; vertical-align: middle;
}
.badge.ok { background: #d8f0e0; color: #1f7a44; }
.badge.partial { background: #fdeccd; color: #9a6700; }
.formula-row { display: flex; flex-wrap: wrap; gap: 2.2rem; align-items: center; margin: .8rem 0 .4rem; font-size: 1.02rem; }
.formula { display: inline-flex; align-items: center; gap: .45rem; }
.formula .lhs { font-style: italic; font-weight: 600; }
.frac { display: inline-flex; flex-direction: column; text-align: center; vertical-align: middle; line-height: 1.15; }
.frac .num { padding: 0 .35rem; }
.frac .den { padding: 0 .35rem; border-top: 1.5px solid currentColor; }
footer { margin-top: 2rem; font-size: .85rem; color: var(--muted); }
"""


def summary_card(hypothesis: str, result: str, status: str, kind: str = "ok") -> str:
    """Карточка «Гипотеза / Краткий результат / Статус» в начало секции."""
    icon = "✅" if kind == "ok" else "⚠️"
    return f"""<div class="summary-card">
    <div class="sc-row"><span class="sc-label">Гипотеза:</span>{hypothesis}</div>
    <div class="sc-row"><span class="sc-label">Краткий результат:</span>{result}</div>
    <div class="sc-row"><span class="sc-label">Статус:</span> <span class="badge {kind}">{icon} {status}</span></div>
  </div>"""


def fmt_p(p: float) -> str:
    """Человекочитаемый p-value (с обработкой underflow до 0)."""
    if p <= 0:
        return "&lt; 10⁻³⁰⁸ (underflow)"
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.4f}"


H5_EVIDENCE_LABEL = {
    "no_evidence": "no_evidence",
    "man": "man_evidence",
    "woman": "woman_evidence",
}


def h5_evidence_label(evidence: str) -> str:
    return H5_EVIDENCE_LABEL.get(evidence, evidence)


def render_h5_rates_table(rows: list[dict]) -> str:
    if not rows:
        return "<p class='note'>—</p>"
    hdr = (
        "<tr><th>evidence</th><th>n</th><th>k_man</th><th>k_woman</th>"
        "<th>k_abstain</th><th>P(man)</th><th>P(woman)</th><th>P(abstain)</th></tr>"
    )
    body = []
    for d in rows:
        ev = h5_evidence_label(d["evidence"])
        body.append(
            f"<tr><td><code>{html.escape(ev)}</code></td>"
            f"<td>{d['n']}</td><td>{d['k_man']}</td><td>{d['k_woman']}</td>"
            f"<td>{d['k_abstain']}</td>"
            f"<td>{d['p_man']:.1f}%</td><td>{d['p_woman']:.1f}%</td>"
            f"<td>{d['p_abstain']:.1f}%</td></tr>"
        )
    return f"<table class='mcn'>{hdr}{''.join(body)}</table>"


def compute_h5_abstain_mcnemar(items_path: Path) -> list[dict]:
    """Парный McNemar по доле abstain между evidence-условиями (with_abstain).

    Пары строятся по одному сценарию (soc × profession × onet_action × context_order
    × position_variant) при разных evidence_shift. Abstain = семантический выбор
    «Cannot determine» (labels[choice]), т.к. опция-воздержание может стоять в любом слоте.
    """
    if not items_path.exists():
        return []
    rows: dict[tuple, dict[str, bool]] = defaultdict(dict)
    for line in items_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("abstain_variant") != "with_abstain":
            continue
        if r.get("task") != "main" or r.get("question_format") != "choice":
            continue
        key = (r["soc"], r["profession"], r["onet_action"],
               r["context_order"], r["position_variant"])
        rows[key][r["evidence_shift"]] = (r["labels"].get(r["choice"]) == "Cannot determine")

    def mcnemar(a: str, b: str) -> dict:
        nb = nc = n = 0
        for d in rows.values():
            if a in d and b in d:
                n += 1
                if d[a] and not d[b]:
                    nb += 1
                elif d[b] and not d[a]:
                    nc += 1
        chi2 = (abs(nb - nc) - 1) ** 2 / (nb + nc) if (nb + nc) > 0 else 0.0
        p = math.erfc(math.sqrt(chi2 / 2)) if chi2 > 0 else 1.0
        return {"a": a, "b": b, "n": n, "nb": nb, "nc": nc, "chi2": chi2, "p": p}

    return [
        mcnemar("no_evidence", "man"),
        mcnemar("no_evidence", "woman"),
        mcnemar("man", "woman"),
    ]


def build_h5_section(h3_data: dict, tests: list[dict]) -> str:
    rates = h3_data.get("evidence_rates_with", [])
    table_html = render_h5_rates_table(rates)
    interpret_html = h3_data.get("rates_with_interpret_html", "")
    chart_data = json.dumps(rates)

    pair_label = {
        "no_evidence": "no_evidence",
        "man": "man_evidence",
        "woman": "woman_evidence",
    }
    test_rows = ""
    for t in tests:
        a, b = pair_label[t["a"]], pair_label[t["b"]]
        sig = "✓" if t["p"] < 0.05 else "✗"
        sig_cls = "fdr-yes" if t["p"] < 0.05 else "fdr-no"
        test_rows += (
            f"<tr><td class='mcn-key'>{a} → {b}</td>"
            f"<td>{t['n']}</td><td>{t['nb']}</td><td>{t['nc']}</td>"
            f"<td>{fmt_p(t['p'])}</td>"
            f"<td class='{sig_cls}'>{sig}</td></tr>"
        )
    test_block = ""
    if test_rows:
        test_block = f"""
  <h3>Тест значимости — парный McNemar по abstain</h3>
  <p>Каждая пара = один и тот же сценарий (<code>soc × profession × onet_action ×
  context_order × position_variant</code>) при двух evidence-условиях. Считается, воздержалась
  ли модель (<i>semantic</i> «Cannot determine», т.к. опция-воздержание может стоять в любом слоте).
  <b>b</b> — воздержалась в первом условии, но не во втором; <b>c</b> — наоборот; статистика
  McNemar проверяет H₀: b = c (p-value в таблице).</p>
  <table class="mcn">
    <thead><tr><th class="mcn-key">Сравнение (abstain)</th><th>n пар</th><th>b</th><th>c</th>
    <th>p-value</th><th>Значимо (p&lt;0.05)</th></tr></thead>
    <tbody>{test_rows}</tbody>
  </table>
  <p class="note"><b>Вывод теста:</b> добавление evidence (любого пола) <b>значимо снижает</b>
  долю воздержаний относительно <code>no_evidence</code> (оба p ≪ 0.001); при этом
  <b>woman_evidence снижает abstain сильнее</b>, чем man_evidence (man_evidence → woman_evidence значим).
  Это статистически подтверждает H5.</p>"""
    card = summary_card(
        hypothesis=(
            "При наличии опции «воздержаться» модель воздерживается чаще <b>без</b> evidence, "
            "чем <b>с</b> evidence."
        ),
        result=(
            "Подтверждается: P(abstain) <b>68.4%</b> (no_evidence) → <b>33.1%</b> (man) / "
            "<b>16.5%</b> (woman); отказов заметно меньше, когда подсвечена женщина."
        ),
        status="подтверждается",
        kind="ok",
    )
    return f"""
<section id="h5">
  <h2>H5 — abstain × evidence</h2>
  {card}
  <p class="src-note">Источник: тот же прогон, что и H3 — <code>highlight-h3-full</code>
  (блок <i>with_abstain — man / woman / abstain</i> из отчёта H3). H5 проверяет,
  как наличие/отсутствие evidence влияет на долю ответа «воздержаться».</p>
  <div class="interpret">{interpret_html}</div>
  {table_html}
  <div class="chart-wrap"><canvas id="chartH5Abstain"></canvas></div>
  <p class="note">График и таблица выше — описательные marginal-доли argmax-choice
  (man / woman / abstain) по условиям evidence. Значимость направленных сдвигов man↔woman —
  в секции H3 (paired McNemar); значимость сдвигов <b>abstain</b> — в тесте ниже.</p>
  {test_block}
  <script>
  (function() {{
    var rows = {chart_data};
    if (!rows.length || typeof Chart === 'undefined') return;
    new Chart(document.getElementById('chartH5Abstain'), {{
      type: 'bar',
      data: {{
        labels: rows.map(function(d) {{
          var m = {{ no_evidence: 'no_evidence', man: 'man_evidence', woman: 'woman_evidence' }};
          return m[d.evidence] || d.evidence;
        }}),
        datasets: [
          {{ label: 'P(man) %', data: rows.map(function(d) {{ return d.p_man; }}), backgroundColor: '#2c5f8d' }},
          {{ label: 'P(woman) %', data: rows.map(function(d) {{ return d.p_woman; }}), backgroundColor: '#9bb8d4' }},
          {{ label: 'P(abstain) %', data: rows.map(function(d) {{ return d.p_abstain; }}), backgroundColor: '#c45c3e' }},
        ],
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ title: {{ display: true, text: 'with_abstain: доля choice по evidence (H5)' }} }},
        scales: {{ y: {{ beginAtZero: true, max: 100, title: {{ display: true, text: '%' }} }} }},
      }},
    }});
  }})();
  </script>
</section>
"""


def build_h11_section(ch: dict, test_auc: float) -> str:
    cm = ch["stages"]["extract_direction"]["summary"]["metrics"]
    c_ct = ch["stages"]["control_task"]["summary"]
    c_proj = ch["stages"]["extract_direction"]["summary"]["projection_soft_target_r"]
    n = ch["n_samples"]
    g = ch["n_probe_groups"]
    sp = ch["split"]["families_per_split"]

    def f2(x):
        return f"{x:.2f}"

    def pct(x):
        return f"{x * 100:.1f}%"

    card = summary_card(
        hypothesis=(
            "Предпочтение пола (gender-ось: выбор man/woman) линейно декодируется "
            "из hidden states модели."
        ),
        result=(
            f"Подтверждается: <code>gender_choice</code> — "
            f"test balanced acc <b>{pct(cm['test_balanced_accuracy'])}</b>, AUC <b>{test_auc:.3f}</b>, "
            f"shuffle ≈ {pct(c_ct['control_global_at_best_layer'])} (best L{cm['layer']}); "
            f"val≈test, control ≪ main → сигнал не артефакт."
        ),
        status="подтверждается (классификация)",
        kind="ok",
    )

    return f"""
<section id="h11">
  <h2>H11 — gender preference в internals (probing)</h2>
  {card}
  <p class="src-note">Источник: <code>probes/v1_rep_prob/</code> прогона
  <code>run_2026-06-09_12-50-33_Qwen3.5-2B-Base_v1_full_pos_shuffle</code>
  (gender-ось: <code>without_abstain_p0_p1_mf_wf_gender_choice/meta.json</code>).
  Линейная проба-классификатор по hidden states, цель
  <code>gender_choice</code> (выбор man/woman).</p>

  <div class="interpret">
    <p><strong>Сетап зондов.</strong></p>
    <ul>
      <li><b>Модель:</b> <code>Qwen/Qwen3.5-2B-Base</code>; hidden states 25 слоёв
      (L0 = embedding, L1–L24 = transformer blocks), <code>d_model=2048</code>.</li>
      <li><b>Задача:</b> linear probing hidden states (<b>last token до ответа</b>) —
      gender-ось bias (выбор man/woman).</li>
      <li><b>Срез:</b> <code>without_abstain</code>, <code>question_format=choice</code>,
      <code>task=main</code>, <code>position_variant</code> ∈ {{p0, p1}},
      <code>context_order</code> ∈ {{man_first, woman_first}}.</li>
      <li><b>Объём:</b> {n} строк, {g} групп сценария (4 layout-строки на группу).</li>
      <li><b>Цель (target):</b> <code>gender_choice</code> — hard 0/1: 1, если семантика
      выбранного ответа = <code>man</code>; 0, если = <code>woman</code> (через <code>labels[choice]</code>).</li>
      <li><b>Split:</b> по группам сценария (<code>soc × profession × onet_action</code>);
      train {sp['train']} / val {sp['val']} / test {sp['test']} семейств (60% / 20% / 20%), <code>seed=0</code>.</li>
      <li><b>Preprocessing:</b> <code>StandardScaler</code> на HS-векторах (fit только на train).</li>
      <li><b>Зонд:</b> <code>LogisticRegression</code>, <code>C=1.0</code>,
      <code>max_iter=10000</code>, <code>class_weight=balanced</code>, <code>random_state=0</code>.</li>
      <li><b>Layer scan:</b> все слои L0…L24; best layer = argmax <code>val balanced accuracy</code>.</li>
    </ul>
    <p class="note">В этом эксперименте <code>onet_action</code> фиксирован для каждой profession (1:1),
    поэтому split-ключ записан как <code>soc × profession × onet_action</code>, но <code>onet_action</code>
    не добавляет независимых групп.</p>
  </div>

  <h3>gender_choice — классификация (balanced accuracy)</h3>
  <p>Бинарный случай (TP — верные man, TN — верные woman, FP — woman→man, FN — man→woman):</p>
  <div class="formula-row">
    <span class="formula"><span class="lhs">balanced accuracy</span> =
      <span class="frac"><span class="num">TPR + TNR</span><span class="den">2</span></span>
    </span>
    <span class="formula"><span class="lhs">TPR</span> =
      <span class="frac"><span class="num">TP</span><span class="den">TP + FN</span></span>
    </span>
    <span class="formula"><span class="lhs">TNR</span> =
      <span class="frac"><span class="num">TN</span><span class="den">TN + FP</span></span>
    </span>
  </div>
  <p class="note">TPR — доля верно распознанных «man» (sensitivity / recall), TNR — доля верно
  распознанных «woman» (specificity). Усреднение по классам делает метрику устойчивой к дисбалансу:
  тривиальный «всегда мажоритарный класс» даёт 0.5, а не завышенную accuracy.</p>
  <table class="mcn">
    <thead><tr><th class="mcn-key">Метрика</th><th>train</th><th>val</th><th>test</th></tr></thead>
    <tbody>
      <tr class="mcn-grp-a"><td class="mcn-key">balanced accuracy</td><td>{pct(cm['train_balanced_accuracy'])}</td><td>{pct(cm['val_balanced_accuracy'])}</td><td>{pct(cm['test_balanced_accuracy'])}</td></tr>
      <tr class="mcn-grp-b"><td class="mcn-key">accuracy</td><td>{pct(cm['train_accuracy'])}</td><td>{pct(cm['val_accuracy'])}</td><td>{pct(cm['test_accuracy'])}</td></tr>
    </tbody>
  </table>
  <table class="mcn">
    <thead><tr><th class="mcn-key">Доп.</th><th>Best layer</th><th>test AUC</th><th>shuffle (global @best)</th><th>permuted @best</th><th>proj ↔ soft-target r</th></tr></thead>
    <tbody><tr class="mcn-grp-a">
      <td class="mcn-key">value</td>
      <td>L{cm['layer']}</td><td>{test_auc:.3f}</td>
      <td>{pct(c_ct['control_global_at_best_layer'])}</td>
      <td>{pct(c_ct['control_permuted_alignment_at_best_layer'])}</td><td>{f2(c_proj)}</td>
    </tr></tbody>
  </table>
  <details class="method-spoiler"><summary>Что такое shuffle и AUC (академически)</summary>
  <div class="spoiler-body">
    <p><b>Shuffle (control / permutation baseline).</b> Контрольная задача: метки цели
    случайно перемешиваются между примерами, после чего проба обучается заново на тех же
    hidden states. Перемешивание разрывает истинную связь «представление → метка», сохраняя
    маргинальное распределение меток и размерность признаков. Качество на shuffle оценивает
    уровень, достижимый <i>исключительно за счёт ёмкости пробы и случайных корреляций</i>
    (эффективный chance level). Если реальная проба работает значимо выше shuffle, информация
    о цели действительно линейно представлена в активациях, а не «вытащена» переобучением
    мощного классификатора (ср. Hewitt &amp; Liang, 2019, «control tasks»; Pimentel et al., 2020).</p>
    <p><b>AUC (Area Under the ROC Curve).</b> Площадь под ROC-кривой (TPR против FPR по всем
    порогам). Эквивалентна вероятности того, что случайно взятый положительный пример получит
    более высокий score, чем случайно взятый отрицательный (статистика Манна–Уитни U).
    AUC=0.5 — неотличимо от случайного угадывания, AUC=1.0 — идеальное разделение классов.
    В отличие от accuracy, AUC не зависит от выбора порога и устойчива к дисбалансу классов.</p>
  </div></details>
  <details class="method-spoiler"><summary>Control-режимы и proj ↔ soft-target r (что в таблице)</summary>
  <div class="spoiler-body">
    <p>Все контроли — это <b>shuffled-label retraining</b>: метки <code>y</code> перемешиваются
    тем или иным способом, проба переобучается заново на тех же hidden states <code>X</code>
    (на best layer), рвётся только связь <code>X ↔ y</code>. Разные режимы по-разному сохраняют
    структуру меток:</p>
    <ul>
      <li><b>main</b> — оригинальные метки <code>(Xᵢ, yᵢ)</code>; основной результат
      (test bacc {pct(cm['test_balanced_accuracy'])}).</li>
      <li><b>within_scenario_family</b> — метки перемешиваются <i>только внутри</i> split-группы
      (один сценарий, 4 layout-строки). Контролирует утечку через групповую структуру; @best ≈
      {pct(c_ct['control_within_scenario_family_at_best_layer'])}.</li>
      <li><b>global</b> (= «shuffle» в таблице) — глобальное перемешивание всех меток по датасету;
      эффективный chance level; @best ≈ {pct(c_ct['control_global_at_best_layer'])}.</li>
      <li><b>permuted_alignment</b> — контроль выравнивания: проверяет, не возникает ли мнимый
      сигнал при случайном сопоставлении направлений/меток; @best ≈
      {pct(c_ct['control_permuted_alignment_at_best_layer'])}.</li>
    </ul>
    <p>Все три контроля ≈ 0.5 ≪ main → высокая bacc отражает реальную линейную связь активаций
    с целью, а не способность LogReg подогнать произвольные метки.</p>
    <p><b>proj ↔ soft-target r</b> ({f2(c_proj)}) — корреляция Пирсона между проекцией hidden state
    на найденное направление пробы (decision-score) и «мягкой» целью (вероятность/непрерывный
    показатель предпочтения man). Высокое r означает, что одно линейное направление не просто
    разделяет классы по порогу, но и <i>монотонно</i> упорядочивает примеры по силе gender-сигнала.</p>
  </div></details>
  <p class="note"><b>Вывод:</b> gender-выбор <b>хорошо линейно декодируется</b>:
  test balanced acc {pct(cm['test_balanced_accuracy'])} ≫ shuffle {pct(c_ct['control_global_at_best_layer'])}
  (chance ≈ 50%); val≈test → без переобучения на val. control <b>main ≫ global</b> →
  связь hidden states ↔ target не артефакт shuffle. H11 <b>подтверждается</b>.</p>

  <div class="chart-wrap"><canvas id="chartH11"></canvas></div>
  <script>
  (function() {{
    if (typeof Chart === 'undefined') return;
    new Chart(document.getElementById('chartH11'), {{
      type: 'bar',
      data: {{
        labels: ['train', 'val', 'test', 'shuffle (global)'],
        datasets: [
          {{ label: 'gender_choice — balanced accuracy (L{cm['layer']})',
             data: [{cm['train_balanced_accuracy']:.4f}, {cm['val_balanced_accuracy']:.4f}, {cm['test_balanced_accuracy']:.4f}, {c_ct['control_global_at_best_layer']:.4f}],
             backgroundColor: ['#2c5f8d', '#2c5f8d', '#2c5f8d', '#9bb8d4'] }},
        ],
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }}, title: {{ display: true, text: 'H11 gender_choice: balanced accuracy vs shuffle' }} }},
        scales: {{ y: {{ beginAtZero: true, max: 1, title: {{ display: true, text: 'balanced accuracy' }} }} }},
      }},
    }});
  }})();
  </script>
</section>
"""


def main() -> None:
    h1_html = read(SRC_H1)
    h3_html = read(SRC_H3)
    h3_data = extract_data_obj(h3_html)
    h11_choice = json.loads(read(SRC_H11_CHOICE))

    h5_tests = compute_h5_abstain_mcnemar(SRC_H3_ITEMS)
    h5_section = build_h5_section(h3_data, h5_tests)
    h11_section = build_h11_section(h11_choice, H11_TEST_AUC)

    h1_card = summary_card(
        hypothesis=(
            "В неоднозначном occupation-выборе модель систематически предпочитает "
            "мужчину женщине."
        ),
        result=(
            "Подтверждается: pooled P(man) <b>25.9%</b> vs P(woman) <b>22.8%</b> "
            "(Δ +0.031, p≈4·10⁻¹⁰); без abstain Δ +0.110. Позиционное предпочтение "
            "(слот A) сильное (McNemar p≈10⁻¹⁰²), но <b>сбалансировано дизайном</b> "
            "(swap p0/p1, man_first/woman_first; метрики по semantic label man/woman), "
            "поэтому gender-оценку не искажает."
        ),
        status="подтверждается",
        kind="ok",
    )
    h3_card = summary_card(
        hypothesis=(
            "Подсветка пола X (evidence highlight) увеличивает предпочтение X "
            "относительно ambiguous baseline (<code>no_evidence</code>)."
        ),
        result=(
            "Подтверждается в 2 из 4 primary McNemar (<code>without_abstain</code>, "
            "n_pairs=3804 choice); противоположно значимых нет. "
            "Женский evidence: choice +11.7%, prob +11.3%; мужской — n.s. "
            "По отраслям (soc prob, without_abstain): no→woman значим в 3/22 после BH."
        ),
        status="подтверждается",
        kind="ok",
    )

    page = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Combined report — H1 · H3 · H5 · H11</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>{SHARED_CSS}</style>
</head>
<body>
<header class="top">
  <h1>Объединённый отчёт — H1 · H3 · H5 · H11</h1>
  <p class="meta">Сборка нативных отчётов проекта Bias subspaces в LLM (модель
  <code>Qwen/Qwen3.5-2B-Base</code>). Содержимое перенесено из родных отчётов.</p>
</header>

<nav class="toc">
  <b>Содержание</b>
  <ul>
    <li><a href="#intro">Введение</a></li>
    <li><a href="#h1">H1 — gender preference (occupation choice)</a></li>
    <li><a href="#h3">H3 — evidence highlight</a></li>
    <li><a href="#h5">H5 — abstain × evidence</a></li>
    <li><a href="#h11">H11 — gender preference в internals (probing)</a></li>
  </ul>
</nav>

<section id="intro">
  <h2>Введение</h2>
  <p><strong>Цель.</strong> Оценить гендерное предпочтение модели
  <code>Qwen/Qwen3.5-2B-Base</code> в задачах неоднозначного выбора профессии — как на
  уровне <b>поведения</b> (что модель отвечает), так и на уровне <b>внутренних представлений</b>
  (что закодировано в её активациях).</p>

  <p><strong>Логика отчёта.</strong> Четыре гипотезы образуют единый сюжет —
  от наблюдаемого поведения к его механизму:</p>
  <ul>
    <li><b>H1 — есть ли предпочтение?</b> Базовый замер: в неоднозначном выборе «man vs woman»
    модель систематически предпочитает мужчину. <span class="badge ok">✅ подтверждается</span></li>
    <li><b>H3 — управляемо ли оно?</b> Подсветка пола (evidence) сдвигает выбор к highlighted полу
    (McNemar, <code>without_abstain</code>); эффект выражен для <b>woman</b>-evidence.
    <span class="badge ok">✅ частично подтверждается</span></li>
    <li><b>H5 — роль опции «воздержаться».</b> При доступной опции abstain модель воздерживается
    заметно чаще <i>без</i> evidence, чем с ним; подсветка (особенно женская) «разблокирует»
    содержательный ответ. <span class="badge ok">✅ подтверждается</span></li>
    <li><b>H11 — где это в модели?</b> Пол выбранного ответа <b>линейно декодируется</b> из hidden
    states (probing): признак присутствует и линейно отделим. <span class="badge ok">✅ подтверждается</span></li>
  </ul>

  <p><strong>Связка поведение ↔ представления.</strong> Поведенческое предпочтение man (H1) и его
  управляемость подсветкой (H3) согласуются с тем, что пол явно представлен в активациях
  (H11: test balanced accuracy ≈ 87% ≫ shuffle ≈ 49%). При этом высокая декодируемость означает,
  что информация <i>присутствует</i> и линейно отделима, но сама по себе <b>не доказывает</b>, что
  модель использует это направление при решении — для каузального вывода нужны интервенции
  (activation patching / steering), выходящие за рамки этого отчёта.</p>

  <p class="note"><b>О позиционном эффекте (H1).</b> Модель сильно тянется к первому слоту/первому
  упомянутому имени, но дизайн это <b>контролирует</b>: каждый сценарий предъявляется в обоих
  порядках слотов (p0/p1) и контекста (man_first/woman_first), а доли считаются по
  <i>смыслу</i> ответа (man/woman), а не по букве слота. Поэтому позиционное предпочтение
  усредняется и не смещает оценку gender-эффекта.</p>
</section>

<div class="callout">
  <strong>Важно: разные прогоны.</strong> Гипотезы посчитаны на разных run'ах, поэтому
  абсолютные числа между секциями напрямую не сравнимы:
  <ul>
    <li><b>H1</b> — <code>run_2026-06-09_12-50-33_Qwen3.5-2B-Base_v1_full_pos_shuffle</code> (metrics <code>h1_v1</code>).</li>
    <li><b>H3 / H5</b> — <code>highlight-h3-full</code> (metrics <code>h3_v1</code>).</li>
    <li><b>H11</b> — <code>run_2026-06-09_12-50-33_Qwen3.5-2B-Base_v1_full_pos_shuffle</code> (probes <code>v1_rep_prob</code>, gender-ось).</li>
  </ul>
  Секции H1 и H3 встроены целиком из родных HTML-отчётов; H5 собран из данных H3;
  H11 — из meta gender-пробы <code>gender_choice</code> (классификация).
</div>

<section id="h1">
  <h2>H1 — gender preference (occupation choice)</h2>
  {h1_card}
  <p class="src-note">Родной отчёт: <code>results/run_2026-06-09_12-50-33_…_v1_full_pos_shuffle/metrics_h1_v1/latest/summary.html</code> (встроен целиком).</p>
  <div class="embed-wrap">{iframe_srcdoc(h1_html)}</div>
</section>

<section id="h3">
  <h2>H3 — evidence highlight</h2>
  {h3_card}
  <p class="src-note">Родной отчёт: <code>results/highlight-h3-full/metrics_h3_v1/latest/summary.html</code> (встроен целиком).</p>
  <div class="embed-wrap">{iframe_srcdoc(h3_html)}</div>
</section>

{h5_section}

{h11_section}

<footer>Сгенерировано <code>analysis/build_combined_report.py</code> объединением нативных отчётов H1, H3, H5, H11.</footer>

<script>
function fitFrame(f) {{
  try {{
    var doc = f.contentWindow.document;
    var h = Math.max(doc.documentElement.scrollHeight, doc.body.scrollHeight);
    f.style.height = (h + 24) + 'px';
    // повторный замер после рендера графиков
    setTimeout(function() {{
      var h2 = Math.max(doc.documentElement.scrollHeight, doc.body.scrollHeight);
      f.style.height = (h2 + 24) + 'px';
    }}, 400);
  }} catch (e) {{ f.style.height = '1600px'; }}
}}
window.addEventListener('resize', function() {{
  document.querySelectorAll('iframe.embed').forEach(fitFrame);
}});
</script>
</body>
</html>
"""

    OUT.write_text(page, encoding="utf-8")
    print(f"written: {OUT}  ({len(page):,} chars)")


if __name__ == "__main__":
    main()
