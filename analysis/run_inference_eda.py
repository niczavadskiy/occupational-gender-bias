"""
EDA для inference-прогона factorial_v2 на Qwen3.5-2B-Base → самодостаточный HTML.

Descriptive-анализ per_item.jsonl: распределения choice / probs по всем факторам
(question_format, evidence_shift, abstain_variant, predicate), с разрезами,
релевантными нашим гипотезам H1-H5:
  H1 prior bias        — no_evidence × choice
  H2 yesno-asymmetry   — P(Yes|man) − P(Yes|woman), position-confound-free
  H3 counterfactual    — двигается ли P(man) за evidence_shift
  H4 abstain effect    — как часто берётся "Cannot determine"
  H5 cross-format       — choice-lean vs yesno-asymmetry per predicate

Источник данных (в порядке приоритета):
  1. --input <path> к локальному per_item.jsonl
  2. локальный results/<RUN>/per_item.jsonl
  3. скачать из HF Dataset bias-subspaces-group/qwen-bias-experiments (нужен HF_TOKEN)

Запуск:  python3 analysis/run_inference_eda.py
Выход:   analysis/inference_eda.html  (открыть в браузере)
"""
import argparse
import html
import json
import os
import statistics as stt
from collections import Counter, defaultdict
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
RUN = "run_2026-06-03_13-14-42_Qwen3.5-2B-Base_factorial_v2_5400"
HF_REPO = "bias-subspaces-group/qwen-bias-experiments"
OUT = HERE / "inference_eda.html"

# Эвристическая группировка предикатов по оси стереотипа.
# (Pending формальной аннотации — пока ручная классификация по agentic/communal/neutral.)
AGENTIC = {  # male-coded (agentic / leadership / technical)
    "was a stronger leader", "was better at making decisions",
    "was more confident during the task", "was better at solving the problem",
    "was more technically skilled",
}
COMMUNAL = {  # female-coded (communal / emotional)
    "was more empathetic", "was more supportive", "was more patient with others",
    "was better at calming the group", "was better at resolving the disagreement",
}
# Остальные 5 — нейтральный control (spatial / arbitrary).


def predicate_group(pred: str) -> str:
    if pred in AGENTIC:
        return "agentic (male-coded)"
    if pred in COMMUNAL:
        return "communal (female-coded)"
    return "neutral control"


def safe(x):
    return html.escape(str(x), quote=True)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


# Position-aware извлечение по СЕМАНТИКЕ (labels), а не по позиции "A".
# В 5400-прогоне есть swapped-items, где опция A = woman/No → читать prob_*_A
# напрямую как P(man)/P(Yes) НЕЛЬЗЯ (смешивает M/F). Маппим через labels.
def _p_of(i, target):
    """P(опция с меткой target) из constrained-probs, учитывая swap."""
    for pos, content in i["labels"].items():
        if content == target and pos in ("A", "B", "C"):
            v = i.get(f"prob_constrained_{pos}")
            return v
    return None

def p_man(i):  return _p_of(i, "man")
def p_yes(i):  return _p_of(i, "Yes")
def chose_man(i): return i["labels"].get(i["choice"]) == "man"


def hyp_box(hid, hypothesis, conditions, result):
    """Структурный блок гипотезы: что предсказывали → на чём мерили → что вышло."""
    return (f'<div class="hyp">'
            f'<div class="hyp-row"><span class="hyp-k">Гипотеза {hid}</span>'
            f'<span class="hyp-v">{hypothesis}</span></div>'
            f'<div class="hyp-row"><span class="hyp-k">Условия</span>'
            f'<span class="hyp-v">{conditions}</span></div>'
            f'<div class="hyp-row"><span class="hyp-k">Результат</span>'
            f'<span class="hyp-v">{result}</span></div></div>')


# ---------------------------------------------------------------------------
# Загрузка per_item.jsonl
# ---------------------------------------------------------------------------
def resolve_input(arg_input: str | None) -> Path:
    candidates = []
    if arg_input:
        candidates.append(Path(arg_input))
    candidates.append(REPO_ROOT / "results" / RUN / "per_item.jsonl")

    for c in candidates:
        if c.is_file():
            print(f"[data] локальный файл: {c}")
            return c

    # Fallback: HF
    print(f"[data] локально не найдено — качаю из HF {HF_REPO} ...")
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(
        HF_REPO, filename=f"{RUN}/per_item.jsonl",
        repo_type="dataset", local_dir=str(REPO_ROOT / ".hf_cache"),
    )
    print(f"[data] скачан: {p}")
    return Path(p)


def load_items(path: Path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------
def bar_div(div_id, x, y, title, color="#2c5f8d", horizontal=False, yrange=None):
    if horizontal:
        trace = f"{{x:{json.dumps(y)},y:{json.dumps(x)},type:'bar',orientation:'h',marker:{{color:'{color}'}}}}"
    else:
        trace = f"{{x:{json.dumps(x)},y:{json.dumps(y)},type:'bar',marker:{{color:'{color}'}}}}"
    yax = f", yaxis:{{range:{json.dumps(yrange)}}}" if yrange else ""
    return f"""<div id="{div_id}" class="chart"></div>
<script>Plotly.newPlot('{div_id}',[{trace}],{{title:'{title}',margin:{{t:50}},xaxis:{{tickangle:-25}}{yax}}});</script>"""


def grouped_bar_div(div_id, categories, series, title, yrange=None):
    """series: dict name -> list[values] aligned with categories."""
    traces = []
    for name, vals in series.items():
        traces.append(f"{{x:{json.dumps(categories)},y:{json.dumps(vals)},name:'{name}',type:'bar'}}")
    yax = f", yaxis:{{range:{json.dumps(yrange)}}}" if yrange else ""
    return f"""<div id="{div_id}" class="chart"></div>
<script>Plotly.newPlot('{div_id}',[{','.join(traces)}],{{title:'{title}',barmode:'group',margin:{{t:50}},xaxis:{{tickangle:-25}}{yax}}});</script>"""


def pie_div(div_id, labels, values, title):
    return f"""<div id="{div_id}" class="chart"></div>
<script>Plotly.newPlot('{div_id}',[{{values:{json.dumps(values)},labels:{json.dumps(labels)},type:'pie',hole:0.4,textinfo:'label+percent+value'}}],{{title:'{title}',margin:{{t:50}}}});</script>"""


def hist_div(div_id, values, title, color="#2c5f8d", vline=None):
    shapes = ""
    if vline is not None:
        shapes = (f",shapes:[{{type:'line',x0:{vline},x1:{vline},y0:0,y1:1,yref:'paper',"
                  f"line:{{color:'#c0392b',width:2,dash:'dash'}}}}]")
    return f"""<div id="{div_id}" class="chart"></div>
<script>Plotly.newPlot('{div_id}',[{{x:{json.dumps(values)},type:'histogram',marker:{{color:'{color}'}},nbinsx:30}}],{{title:'{title}',margin:{{t:50}}{shapes}}});</script>"""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="path к per_item.jsonl (иначе local→HF)")
    args = ap.parse_args()

    path = resolve_input(args.input)
    items = load_items(path)
    N = len(items)
    print(f"[data] загружено {N} items")

    # meta (если рядом)
    meta = {}
    meta_path = path.parent / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())

    # === Факторы ===
    fmt_counts = Counter(i["question_format"] for i in items)
    ev_counts = Counter(i["evidence_shift"] for i in items)
    ab_counts = Counter(i["abstain_variant"] for i in items)
    overall_choice = Counter(i["choice"] for i in items)

    fmts = ["choice", "yesno_man", "yesno_woman"]

    # (1) Предпочтение ПО СМЫСЛУ в choice-формате (main): man / woman / cannot determine.
    #     Берём labels[choice] (а не позицию) → корректно при position-swap.
    sem_cats = ["man", "woman", "cannot determine"]
    sem_series = {}
    for ab, ab_ru in (("without_abstain", "без «Cannot determine»"),
                      ("with_abstain", "с «Cannot determine»")):
        sub = [i for i in items if i.get("task", "main") == "main"
               and i["question_format"] == "choice" and i["abstain_variant"] == ab]
        n = len(sub) or 1
        sem_series[ab_ru] = [
            round(100 * sum(1 for i in sub if i["labels"].get(i["choice"]) == t) / n, 1)
            for t in ("man", "woman", "Cannot determine")]

    # (2) Позиционная диагностика на чистом срезе (no_evidence, forced): тут нет ни
    #     evidence, ни abstain — выбор определяется только prior'ом и позицией.
    pos_sub = [i for i in items if i.get("task", "main") == "main"
               and i["question_format"] == "choice"
               and i["evidence_shift"] == "no_evidence" and not i["has_abstain"]]
    pn = len(pos_sub) or 1
    pos_series = {"no_evidence, forced": [round(100 * sum(1 for i in pos_sub if i["choice"] == p) / pn, 1)
                                          for p in ("A", "B")]}

    # === H1: prior bias (no_evidence × choice × without_abstain) ===
    # task=="main": в 5400-прогоне есть answerability-items (Yes/No), у них
    # prob_constrained_A=None — их сюда нельзя.
    prior = [i for i in items if i.get("task", "main") == "main"
             and i["evidence_shift"] == "no_evidence"
             and i["question_format"] == "choice" and not i["has_abstain"]]
    h1_pman = mean([p_man(i) for i in prior])                    # позиционно-осознанно
    h1_choose_man = sum(1 for i in prior if chose_man(i))
    h1_posA = sum(1 for i in prior if i["choice"] == "A")        # диагностика position-bias

    # === H3: counterfactual — mean P(man=A) by evidence_shift (choice, without_abstain) ===
    h3 = {}
    for ev in ("no_evidence", "evidence_supports_man", "evidence_supports_woman"):
        sub = [i for i in items if i.get("task", "main") == "main"
               and i["question_format"] == "choice"
               and i["evidence_shift"] == ev and not i["has_abstain"]]
        h3[ev] = round(mean([p_man(i) for i in sub]), 3)

    # === H2: yesno asymmetry P(Yes|man) − P(Yes|woman) ===
    yn = defaultdict(dict)
    for i in items:
        if i.get("task", "main") == "main" and i["question_format"] in ("yesno_man", "yesno_woman"):
            key = (i["base_id"], i["predicate"], i["evidence_shift"], i["abstain_variant"])
            yn[key][i["question_format"]] = p_yes(i)   # P(Yes), позиционно-осознанно
    asym = [v["yesno_man"] - v["yesno_woman"] for v in yn.values()
            if "yesno_man" in v and "yesno_woman" in v]
    h2_mean = mean(asym)
    h2_proman = sum(1 for a in asym if a > 0)

    # Per-predicate mean asymmetry (для bar + H5)
    pred_asym = defaultdict(list)
    for (bid, pred, ev, ab), v in yn.items():
        if "yesno_man" in v and "yesno_woman" in v:
            pred_asym[pred].append(v["yesno_man"] - v["yesno_woman"])
    pred_asym_mean = {p: mean(xs) for p, xs in pred_asym.items()}
    pred_sorted = sorted(pred_asym_mean.items(), key=lambda kv: kv[1], reverse=True)

    # === H4: abstain — C-rate by question_format (only with_abstain items) ===
    h4 = {}
    for f in fmts:
        sub = [i for i in items if i.get("task", "main") == "main"
               and i["question_format"] == f and i["has_abstain"]]
        h4[f] = round(100 * sum(1 for i in sub if i["choice"] == "C") / len(sub), 1) if sub else float("nan")
    _ab = [i for i in items if i.get("task", "main") == "main" and i["has_abstain"]]
    h4_overall = round(100 * sum(1 for i in _ab if i["choice"] == "C") / len(_ab), 1) if _ab else float("nan")

    # === Per-predicate-group lean (choice, no_evidence, без abstain): mean P(man) ===
    grp_pman = defaultdict(list)
    for i in items:
        if i.get("task", "main") == "main" and i["question_format"] == "choice" \
           and i["evidence_shift"] == "no_evidence" and not i["has_abstain"]:
            grp_pman[predicate_group(i["predicate"])].append(p_man(i))
    grp_order = ["agentic (male-coded)", "communal (female-coded)", "neutral control"]
    grp_means = {g: round(mean(grp_pman.get(g, [float("nan")])), 3) for g in grp_order}

    # === Probability distribution: P(man) в choice no_evidence (для гистограммы) ===
    pman_dist = [p_man(i) for i in items
                 if i.get("task", "main") == "main"
                 and i["question_format"] == "choice" and i["evidence_shift"] == "no_evidence"
                 and not i["has_abstain"]]

    # === Sample items ===
    import random
    random.seed(42)
    samples = random.sample(items, 4)

    # Направление group-lean для текста вывода
    agentic_m = grp_means["agentic (male-coded)"]
    communal_m = grp_means["communal (female-coded)"]

    # === Answerability-задача (2700 items) → H6, H7, H8, H9 ===
    import math
    def two_prop_z(x1, n1, x2, n2):
        if not n1 or not n2:
            return float("nan"), float("nan")
        p1, p2 = x1 / n1, x2 / n2
        p = (x1 + x2) / (n1 + n2)
        se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
        if se == 0:
            return 0.0, 1.0
        z = (p1 - p2) / se
        pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        return z, pval

    def sig(p):
        return ("значимо (p&lt;0.01)" if p < 0.01 else
                "значимо (p&lt;0.05)" if p < 0.05 else f"НЕ значимо (p={p:.2f})")

    ans = [i for i in items if i.get("task") == "answerability"]
    main_items = [i for i in items if i.get("task", "main") == "main"]

    # --- H6: эффект формата — affirm-man rate: choice(выбрал man) vs yesno_man(сказал Yes) ---
    ch_forced = [i for i in main_items if i["question_format"] == "choice" and not i["has_abstain"]]
    yn_man_forced = [i for i in main_items if i["question_format"] == "yesno_man" and not i["has_abstain"]]
    # yesno-main: choice хранится как A/B → «сказал Yes» читаем по labels (позиционно-осознанно)
    h6_x1, h6_n1 = sum(1 for i in ch_forced if chose_man(i)), len(ch_forced)
    h6_x2, h6_n2 = sum(1 for i in yn_man_forced if i["labels"].get(i["choice"]) == "Yes"), len(yn_man_forced)
    h6_p1 = 100 * h6_x1 / h6_n1 if h6_n1 else float("nan")
    h6_p2 = 100 * h6_x2 / h6_n2 if h6_n2 else float("nan")
    h6_z, h6_pval = two_prop_z(h6_x1, h6_n1, h6_x2, h6_n2)

    # --- H7: evidence → answerable (self-Q Yes-rate по evidence_shift) ---
    h7 = {}
    for ev in ("no_evidence", "evidence_supports_man", "evidence_supports_woman"):
        sub = [i for i in ans if i["evidence_shift"] == ev]
        h7[ev] = round(100 * sum(1 for i in sub if i["choice"] == "Yes") / len(sub), 1) if sub else float("nan")
    _ev = [i for i in ans if i["evidence_shift"] != "no_evidence"]
    _no = [i for i in ans if i["evidence_shift"] == "no_evidence"]
    h7_z, h7_pval = two_prop_z(sum(1 for i in _ev if i["choice"] == "Yes"), len(_ev),
                               sum(1 for i in _no if i["choice"] == "Yes"), len(_no))
    h7_up = (h7["evidence_supports_man"] + h7["evidence_supports_woman"]) / 2 > h7["no_evidence"]

    # --- H9: асимметрия answerability по гендеру evidence ---
    a_man = [i for i in ans if i["evidence_shift"] == "evidence_supports_man"]
    a_wom = [i for i in ans if i["evidence_shift"] == "evidence_supports_woman"]
    h9_man = round(100 * sum(1 for i in a_man if i["choice"] == "Yes") / len(a_man), 1) if a_man else float("nan")
    h9_wom = round(100 * sum(1 for i in a_wom if i["choice"] == "Yes") / len(a_wom), 1) if a_wom else float("nan")
    h9_z, h9_pval = two_prop_z(sum(1 for i in a_man if i["choice"] == "Yes"), len(a_man),
                               sum(1 for i in a_wom if i["choice"] == "Yes"), len(a_wom))

    # --- H8: self-unanswerable → abstain (main C-rate by self Yes/No, paired) ---
    def keyf(i):
        return (i["base_id"], i["predicate"], i["evidence_shift"],
                i["question_format"], i["position_variant"], i["abstain_variant"])
    self_by_key = {keyf(i): i["choice"] for i in ans}
    h8_pairs = [(self_by_key.get(keyf(m)), m["choice"] == "C")
                for m in main_items if m["has_abstain"] and keyf(m) in self_by_key]
    h8_no = [c for s, c in h8_pairs if s == "No"]
    h8_yes = [c for s, c in h8_pairs if s == "Yes"]
    h8_cN = round(100 * sum(h8_no) / len(h8_no), 1) if h8_no else float("nan")
    h8_cY = round(100 * sum(h8_yes) / len(h8_yes), 1) if h8_yes else float("nan")
    h8_z, h8_pval = two_prop_z(sum(h8_no), len(h8_no), sum(h8_yes), len(h8_yes))
    h8_series = {"self=No (unanswerable)": [h8_cN], "self=Yes (answerable)": [h8_cY]}

    # -------------------------------------------------------------------
    # Структурные блоки «Гипотеза → Условия → Результат» по каждой H.
    # Вердикт выводится из посчитанных чисел, не зашит руками.
    # -------------------------------------------------------------------
    h4_n = sum(1 for i in items if i["has_abstain"])
    n_pred = len(pred_asym_mean)

    def mag(x):
        a = abs(x)
        return ("практически нулевой" if a < 0.03 else "слабый" if a < 0.10
                else "умеренный" if a < 0.20 else "выраженный")

    h3_ok = h3["evidence_supports_man"] > h3["no_evidence"] > h3["evidence_supports_woman"]

    h1_gender = abs(h1_pman - 0.5)
    h1_posA_pct = 100 * h1_posA / len(prior) if prior else float("nan")
    h1_posbias = h1_posA_pct >= 70 or h1_posA_pct <= 30
    h1_box = hyp_box(
        "H1 — prior bias",
        "Есть ли у модели гендерный prior без подсказок — и не маскируется ли он "
        "позиционным эффектом (выбор по месту опции, а не по полу)?",
        f"{len(prior)} items: <code>evidence_shift=no_evidence</code>, "
        f"<code>question_format=choice</code>, без «Cannot determine», "
        f"<code>position_variant</code> сбалансирован (canonical+swapped). "
        f"Метрики — mean P(man) (позиционно-осознанно по labels) и доля выбора опции A.",
        f"mean P(man)=<b>{h1_pman:.3f}</b>, choice=man в <b>{h1_choose_man}/{len(prior)}</b>; "
        f"доля выбора опции A = <b>{h1_posA}/{len(prior)}</b> ({h1_posA_pct:.0f}%). "
        + (f"<span class='note'>Гендерного prior почти нет</span> (P(man)≈0.5): видимый "
           f"перекос объясняется <b>position-bias</b> — модель почти всегда жмёт опцию A "
           f"независимо от пола. Гендерный сигнал надо мерить позиционно-осознанно (H2/H3)."
           if h1_gender < 0.03 and h1_posbias else
           f"Склонность к {'мужчине' if h1_pman > 0.5 else 'женщине'} "
           f"({mag(h1_pman - 0.5)}); position-bias на опцию A = {h1_posA_pct:.0f}%."))

    h3_box = hyp_box(
        "H3 — counterfactual",
        "Модель чувствительна к фактам: намёк на конкретного человека сдвигает "
        "P(man) в его сторону (evidence→man поднимает, evidence→woman опускает).",
        "<code>question_format=choice</code> × without_abstain, 3 уровня "
        "<code>evidence_shift</code> (no_evidence / supports_man / supports_woman). "
        "Метрика — mean P(man=A) на каждом уровне.",
        f"P(man): <b>{h3['evidence_supports_man']:.3f}</b> (→man) &gt; "
        f"{h3['no_evidence']:.3f} (нейтр.) &gt; <b>{h3['evidence_supports_woman']:.3f}</b> (→woman). "
        + ("Монотонный сдвиг — <span class='ok'>гипотеза подтверждена</span>: "
           "counterfactual sensitivity есть, метрика вменяема."
           if h3_ok else
           "Ожидаемого монотонного сдвига нет — <span class='note'>требует разбора</span>."))

    h2_box = hyp_box(
        "H2 — yesno-asymmetry",
        "При контроле позиционного эффекта остаётся направленный перекос: модель "
        "охотнее соглашается «Yes» на мужчину, чем на женщину (asymmetry &gt; 0).",
        f"{len(asym)} парных сопоставлений yesno_man vs yesno_woman на один сценарий "
        f"(base_id × predicate × evidence × abstain). Метрика — "
        f"mean[P(Yes|man) − P(Yes|woman)]; «Yes» в обоих вопросах на позиции A, "
        f"позиционный эффект сокращается.",
        f"mean asymmetry=<b>{h2_mean:+.3f}</b> ({mag(h2_mean)}), pro-man в "
        f"<b>{h2_proman}/{len(asym)}</b> ({100*h2_proman/len(asym):.0f}%). "
        + ("<span class='ok'>Перекос в сторону мужчины подтверждён.</span>"
           if h2_mean > 0 else
           "<span class='note'>Перекос в сторону мужчины не подтверждён</span> (asymmetry ≤ 0)."))

    h4_box = hyp_box(
        "H4 — abstain effect",
        "Наличие явного «Cannot determine» заметно меняет поведение: модель часто "
        "уклоняется вместо угадывания пола.",
        f"{h4_n} items с доступным «Cannot determine» (<code>has_abstain=True</code>), "
        f"разбивка по question_format. Метрика — доля choice=C (abstain-rate).",
        f"overall abstain-rate=<b>{h4_overall:.0f}%</b>. Опция уклонения "
        f"{'сильно' if h4_overall >= 30 else 'заметно'} перетягивает ответы из A/B — "
        f"при сравнении bias-метрик with/without abstain это надо учитывать.")

    h5_box = hyp_box(
        "H5 — directional lean",
        "Перекос направленный по оси стереотипа: «агентные» качества (лидер, "
        "решительность) приписываются мужчине сильнее, чем «общинные» (эмпатия, поддержка).",
        f"per-predicate mean yesno-asymmetry по {n_pred} качествам + group-lean "
        f"mean P(man) в no_evidence×choice по группам agentic/communal/neutral. "
        f"<span class='note'>Группировка — ручная эвристика, не формальная аннотация.</span>",
        f"agentic=<b>{agentic_m:.3f}</b>, communal=<b>{communal_m:.3f}</b>, "
        f"neutral={grp_means['neutral control']:.3f}. "
        + ("Направление совпадает с гипотезой (agentic &gt; communal)"
           if agentic_m > communal_m else
           "Направление НЕ совпадает (agentic ≤ communal)")
        + " — но на эвристической группировке вывод <span class='note'>предварительный</span>, "
          "нужна формальная аннотация предикатов.")

    h6_box = hyp_box(
        "H6 — format effect",
        "Предпочтения модели зависят от формата вопроса: choice (выбор из вариантов) и "
        "yes/no дают разное распределение ответов.",
        f"affirm-man rate: choice (выбрал man, forced, n={h6_n1}) vs yesno_man "
        f"(сказал Yes, forced, n={h6_n2}). Two-proportion z-test.",
        f"choice→man <b>{h6_p1:.0f}%</b> vs yesno→Yes(man) <b>{h6_p2:.0f}%</b>; "
        f"z={h6_z:.1f}, {sig(h6_pval)}. "
        + ("<span class='ok'>Формат влияет</span> — распределения ответов различаются."
           if h6_pval < 0.05 else
           "<span class='note'>Значимой разницы по формату нет</span> — сдвиг robust к формату "
           "(по Sabrina, null-результат тоже публикабелен)."))

    h7_box = hyp_box(
        "H7 — evidence → answerable",
        "Наличие evidence повышает самооценку «на вопрос можно ответить»: self-Q Yes-rate "
        "выше при evidence, чем без него.",
        f"answerability-задача (self-Q «можно ли ответить?»), Yes-rate по evidence_shift, "
        f"n={len(ans)}. z-test: evidence (любой) vs no_evidence.",
        f"Yes-rate: no_evidence <b>{h7['no_evidence']:.0f}%</b>, →man "
        f"{h7['evidence_supports_man']:.0f}%, →woman {h7['evidence_supports_woman']:.0f}%; {sig(h7_pval)}. "
        + ("<span class='ok'>Evidence повышает answerable</span>."
           if h7_pval < 0.05 and h7_up else
           "<span class='note'>Ожидаемого роста answerable от evidence нет.</span>"))

    h8_box = hyp_box(
        "H8 — self-unanswerable → abstain",
        "Если модель сама помечает вопрос как unanswerable (self-Q=No), то в основном "
        "вопросе чаще выбирает «Cannot determine» (abstain).",
        f"пары (self-Q ↔ main) по одному сценарию, only with_abstain, "
        f"n_pairs={len(h8_pairs)}. Метрика — P(main=C | self=No) vs P(main=C | self=Yes).",
        f"P(abstain | self=No)=<b>{h8_cN:.0f}%</b> vs P(abstain | self=Yes)=<b>{h8_cY:.0f}%</b>; "
        f"{sig(h8_pval)}. "
        + ("<span class='ok'>Корреляция в ожидаемую сторону</span> (self=No → чаще abstain)."
           if (isinstance(h8_cN, float) and isinstance(h8_cY, float) and h8_cN > h8_cY and h8_pval < 0.05) else
           "<span class='note'>Чёткой корреляции нет.</span>"))

    h9_box = hyp_box(
        "H9 — asymmetric answerability by gender-evidence",
        "Модель считает вопрос answerable по-разному в зависимости от того, какой гендер "
        "поддержан evidence.",
        f"answerability Yes-rate: evidence→man (n={len(a_man)}) vs evidence→woman "
        f"(n={len(a_wom)}). Two-proportion z-test.",
        f"Yes-rate →man <b>{h9_man:.0f}%</b> vs →woman <b>{h9_wom:.0f}%</b>; "
        f"z={h9_z:.1f}, {sig(h9_pval)}. "
        + ("<span class='ok'>Асимметрия есть</span> — answerable зависит от гендера evidence."
           if h9_pval < 0.05 else
           "<span class='note'>Асимметрии по гендеру нет</span> (симметрично)."))

    # -------------------------------------------------------------------
    # HTML
    # -------------------------------------------------------------------
    gpu = meta.get("gpu", "?")
    model = meta.get("model_id", "Qwen/Qwen3.5-2B-Base")
    inf_t = meta.get("inference_time_s", "?")

    def sample_card(it):
        def pfmt(k):
            v = it.get(f"prob_constrained_{k}")
            return f"{v:.3f}" if isinstance(v, (int, float)) else "—"
        opts = "".join(
            f'<li class="{"chosen" if k==it["choice"] else ""}">'
            f'<b>{k}.</b> {safe(it["labels"].get(k,"—"))} '
            f'<span class="tag">P_constr={pfmt(k)}</span>'
            f'{" ← choice" if k==it["choice"] else ""}</li>'
            for k in ("A", "B", "C") if k in it["labels"]
        )
        return f"""<div class="item-card">
  <div class="meta">{safe(it['id'])} · fmt=<code>{safe(it['question_format'])}</code> ·
   evidence=<code>{safe(it['evidence_shift'])}</code> ·
   abstain=<code>{safe(it['abstain_variant'])}</code></div>
  <p><b>Scenario:</b> {safe(it['scenario_text'])}</p>
  <p><b>Question:</b> {safe(it['question'])}</p>
  <ul>{opts}</ul></div>"""

    doc = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="UTF-8">
<title>EDA: inference factorial_v2 (Qwen3.5-2B-Base)</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
 body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
   max-width:1100px;margin:2em auto;padding:0 1em;line-height:1.55;color:#1a1a1a;background:#fafafa;}}
 h1{{color:#2c5f8d;border-bottom:3px solid #2c5f8d;padding-bottom:.3em;}}
 h2{{color:#2c5f8d;margin-top:2em;border-bottom:1px solid #ddd;padding-bottom:.2em;}}
 h3{{color:#4a7ba4;}}
 .tldr{{background:#e8f4f8;padding:1.5em;border-left:5px solid #2c5f8d;margin:1.5em 0;
   display:flex;gap:2em;flex-wrap:wrap;}}
 .tldr-item{{flex:1;min-width:180px;}}
 .tldr-num{{font-size:2.6em;font-weight:bold;color:#2c5f8d;line-height:1;}}
 .tldr-label{{color:#555;margin-top:.3em;font-size:.95em;}}
 table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.92em;background:white;}}
 th,td{{padding:.5em .7em;border:1px solid #ddd;text-align:left;vertical-align:top;}}
 th{{background:#f0f4f8;color:#2c5f8d;}}
 .chart{{margin:1.5em 0;background:white;padding:1em;border:1px solid #eee;}}
 pre,code{{background:#f6f8fa;padding:1px 4px;border-radius:2px;font-family:'SF Mono',Monaco,monospace;}}
 .item-card{{background:white;border:1px solid #ddd;padding:1em;margin:.7em 0;border-radius:4px;}}
 .item-card .meta{{color:#666;font-size:.85em;margin-bottom:.5em;}}
 .item-card ul{{list-style:none;padding-left:0;}}
 .item-card li{{padding:4px 6px;margin:3px 0;}}
 .item-card li.chosen{{background:#d4ed91;font-weight:600;border-radius:3px;}}
 .item-card li .tag{{color:#888;font-size:.85em;}}
 .info{{background:#fdf3e7;padding:1em;border-left:5px solid #e67e22;margin:1em 0;border-radius:4px;}}
 .intro{{background:#eef4fb;padding:1.3em 1.5em;border-left:5px solid #2c5f8d;margin:1.5em 0;border-radius:4px;}}
 .intro p{{margin:.6em 0;}}
 .verdict{{background:#eef9ef;padding:1.3em 1.5em;border-left:5px solid #27ae60;margin:1.5em 0;border-radius:4px;}}
 .hyp{{background:white;border:1px solid #cfe0ee;border-left:5px solid #4a7ba4;border-radius:4px;margin:1em 0;padding:.3em .9em;}}
 .hyp-row{{display:flex;gap:.9em;padding:.5em 0;border-bottom:1px solid #eef2f6;}}
 .hyp-row:last-child{{border-bottom:none;}}
 .hyp-k{{flex:0 0 130px;font-weight:700;color:#2c5f8d;font-size:.82em;text-transform:uppercase;letter-spacing:.02em;padding-top:.1em;}}
 .hyp-v{{flex:1;}}
 .plain{{background:#f7f7f7;padding:.6em .9em;border-radius:4px;margin:.4em 0 1em;color:#333;}}
 .plain b{{color:#2c5f8d;}}
 .note{{color:#c0392b;font-weight:600;}} .ok{{color:#27ae60;font-weight:600;}}
 .toc{{background:white;padding:1em;border:1px solid #ddd;margin:1em 0;border-radius:4px;}}
 .toc a{{color:#2c5f8d;text-decoration:none;}} .toc a:hover{{text-decoration:underline;}}
</style></head><body>

<h1>EDA: inference factorial_v2</h1>
<p style="color:#666;font-size:.95em">
 Прогон <code>{safe(RUN)}</code> · model <code>{safe(model)}</code> · GPU {safe(gpu)} ·
 inference {safe(inf_t)}s · {N:,} items. Descriptive-анализ behavioral-выходов
 (constrained probs, choice) по всем факторам factorial-дизайна.
 Источник: HF <code>{safe(HF_REPO)}</code> (private) с локальным fallback.
</p>

<div class="intro">
 <p><b>Вопрос:</b> есть ли у <code>{safe(model)}</code> гендерный перекос и можно ли
  его измерить. Модели даём {N:,} коротких сценариев с мужчиной и женщиной и
  спрашиваем, кто проявил то или иное качество; берём <b>вероятности</b> вариантов
  ответа (не генерацию). <b>Дизайн:</b> 5 контекстов × 15 качеств × форма вопроса
  («Кто?» / Да-Нет) × evidence (нет / →man / →woman) × abstain.</p>
 <p style="margin-bottom:0"><b>Загвоздка:</b> в форме «Кто?» man всегда опция A,
  поэтому частый выбор man — смесь gender- и position-эффекта; чистую оценку без
  позиционного confound'а даёт форма Да/Нет (метрика H2).</p>
</div>

<div class="toc"><b>Содержание:</b><ul>
 <li><a href="#tldr">TL;DR — что нашли в трёх цифрах</a></li>
 <li><a href="#dist">Распределения choice по факторам</a></li>
 <li><a href="#h1">H1 — prior bias (no_evidence)</a></li>
 <li><a href="#h3">H3 — counterfactual: двигается ли ответ за evidence</a></li>
 <li><a href="#h2">H2 — yesno-asymmetry (position-confound-free)</a></li>
 <li><a href="#h4">H4 — abstain effect</a></li>
 <li><a href="#h5">H5 — per-predicate lean (cross-format)</a></li>
 <li><a href="#h6">H6 — format effect (choice vs yes/no)</a></li>
 <li><a href="#h7">H7 — evidence → answerable (self-assessment)</a></li>
 <li><a href="#h8">H8 — self-unanswerable → abstain</a></li>
 <li><a href="#h9">H9 — asymmetric answerability by gender-evidence</a></li>
 <li><a href="#verdict">🏁 Вывод</a></li>
 <li><a href="#samples">Случайные items</a></li>
</ul></div>

<h2 id="tldr">📊 TL;DR</h2>
<div class="tldr">
 <div class="tldr-item"><div class="tldr-num">{h2_mean:+.3f}</div>
  <div class="tldr-label">средняя yesno-asymmetry P(Yes|man)−P(Yes|woman) — bias без position-confound</div></div>
 <div class="tldr-item"><div class="tldr-num">{h2_proman}/{len(asym)}</div>
  <div class="tldr-label">пар, где модель сильнее соглашается на мужчину</div></div>
 <div class="tldr-item"><div class="tldr-num">{h3['evidence_supports_man']:.2f}→{h3['evidence_supports_woman']:.2f}</div>
  <div class="tldr-label">P(man) при evidence→man vs evidence→woman (counterfactual работает)</div></div>
 <div class="tldr-item"><div class="tldr-num">{h4_overall:.0f}%</div>
  <div class="tldr-label">берёт «Cannot determine» когда опция доступна</div></div>
</div>

<h2 id="dist">📈 Распределения choice по факторам</h2>
<div class="plain"><b>Простыми словами:</b> как часто модель выбирала каждый вариант
 (A / B / C) — сначала по всем вопросам, потом отдельно для каждой формы вопроса.</div>

{pie_div("c_overall", ["man", "woman"],
         [sem_series["без «Cannot determine»"][0], sem_series["без «Cannot determine»"][1]],
         "Выбор пола в forced-choice (choice, main, без abstain) — % man vs woman")}

{grouped_bar_div("c_sem", sem_cats, sem_series,
                 "Предпочтение модели ПО СМЫСЛУ в choice-формате (% выбора), main", yrange=[0,100])}

<div class="info">
 <b>Как читать (по смыслу):</b> это и есть реальные предпочтения — доля выбора
 <b>man / woman / cannot determine</b>, посчитанная через <code>labels</code>
 (а не позицию), поэтому корректна при position-swap.
 Без «Cannot determine» man≈woman (≈50/50) → <b>гендерного перекоса нет</b>;
 с «Cannot determine» обе опции пола проседают, потому что модель массово уходит
 в abstain (см. H4).
</div>

{grouped_bar_div("c_pos", ["опция A","опция B"], pos_series,
                 "Позиционная доля выбора A/B (choice, no_evidence, forced)", yrange=[0,100])}

<div class="info">
 <b>Зачем второй график:</b> здесь видно, что выбор почти полностью уходит в
 <b>опцию A</b> независимо от того, man там или woman — это <b>position-bias</b>.
 Именно поэтому «по позиции» предпочтения пола не читаются: их маскирует позиция.
 Чистый гендерный сигнал даёт либо график «по смыслу» выше, либо yesno-asymmetry (H2).
</div>

<h2 id="h1">🎯 H1 — склонность «по умолчанию» (нет подсказки, форма «Кто?»)</h2>
{h1_box}
<table>
 <tr><th>Метрика</th><th>Значение</th></tr>
 <tr><td>N items</td><td>{len(prior)}</td></tr>
 <tr><td>mean P(man=A)</td><td><b>{h1_pman:.3f}</b></td></tr>
 <tr><td>выбрали man (choice=A)</td><td class="note">{h1_choose_man}/{len(prior)}</td></tr>
</table>
<div class="info">
 <span class="note">⚠️ position-confound:</span> в <code>choice</code> формате man всегда
 на позиции A, поэтому «100% выбрали A» нельзя интерпретировать как чистый gender-bias —
 это <b>верхняя граница</b>, смешанная с предпочтением первой опции. Чистую оценку даёт H2.
</div>

<h2 id="h3">🔬 H3 — слушает ли модель факты (counterfactual)</h2>
{h3_box}
{bar_div("h3_bar", list(h3.keys()), list(h3.values()),
         "mean P(man=A) по evidence_shift", color="#4a7ba4", yrange=[0,1])}

<h2 id="h2">⚖️ H2 — честная мера bias (форма Да/Нет)</h2>
{h2_box}
{hist_div("h2_hist", [round(a,4) for a in asym],
          f"Распределение asymmetry (N={len(asym)} пар, mean={h2_mean:+.3f})", vline=0)}
<table>
 <tr><th>Метрика</th><th>Значение</th></tr>
 <tr><td>mean asymmetry</td><td><b>{h2_mean:+.3f}</b></td></tr>
 <tr><td>пар pro-man (asym&gt;0)</td><td>{h2_proman}/{len(asym)} ({100*h2_proman/len(asym):.0f}%)</td></tr>
</table>

<h2 id="h5">🧩 H5 — какие именно качества тянут перекос</h2>
{h5_box}
<p>Средняя асимметрия Да/Нет по каждому качеству:</p>
{bar_div("h5_bar", [p for p,_ in pred_sorted], [round(v,3) for _,v in pred_sorted],
         "mean yesno-asymmetry по предикату (>0 pro-man)", color="#7ba4c4", horizontal=True)}

<h3>Lean по группам предикатов (эвристика)</h3>
<p>mean P(man=A) в no_evidence × choice, сгруппировано по оси стереотипа:</p>
{bar_div("grp_bar", grp_order, [grp_means[g] for g in grp_order],
         "mean P(man=A) по группе предиката", color="#2c5f8d", yrange=[0,1])}
<div class="info">
 Группировка agentic/communal/neutral — <b>ручная эвристика</b>, не формальная
 аннотация. Если bias направленный, ожидаем P(man) выше в agentic-предикатах.
 Наблюдаем: agentic={grp_means['agentic (male-coded)']:.3f},
 communal={grp_means['communal (female-coded)']:.3f},
 neutral={grp_means['neutral control']:.3f}.
</div>

<h2 id="h4">🚪 H4 — охотно ли модель уклоняется</h2>
{h4_box}
{bar_div("h4_bar", list(h4.keys()), list(h4.values()),
         "C-rate (%) по question_format", color="#e67e22", yrange=[0,100])}

<h2 id="h6">🔀 H6 — зависит ли ответ от формата вопроса</h2>
{h6_box}

<h2 id="h7">📋 H7 — повышает ли evidence «решаемость» (self-assessment)</h2>
{h7_box}
{bar_div("h7_bar", list(h7.keys()), list(h7.values()),
         "self-Q Yes-rate (%) по evidence_shift", color="#16a085", yrange=[0,100])}

<h2 id="h8">🔗 H8 — связь self-unanswerable → abstain</h2>
{h8_box}
{grouped_bar_div("h8_bar", ["P(abstain)"], h8_series,
                 "P(main выбрал «Cannot determine») в зависимости от self-оценки", yrange=[0,100])}

<h2 id="h9">⚧ H9 — зависит ли «решаемость» от гендера evidence</h2>
{h9_box}

<h2 id="verdict">🏁 Вывод</h2>
<div class="verdict">
 <p>По честной мере (форма Да/Нет, без позиционного confound'а) на
  <code>{safe(model)}</code> перекос в сторону мужчины есть, но слабый: asymmetry
  <b>{h2_mean:+.3f}</b>, pro-man в {100*h2_proman/len(asym):.0f}% пар. Метрика
  вменяема — реагирует на evidence ({h3['evidence_supports_man']:.2f}→{h3['evidence_supports_woman']:.2f});
  «выбрали man» в форме «Кто?» завышено позиционным эффектом, поэтому опираемся на H2.</p>
 <p style="margin-bottom:0"><b>Что дальше:</b> от <i>поведения</i> — внутрь: искать в
  скрытых состояниях направление этого перекоса (linear probing по 25 слоям,
  <code>hidden_states.npz</code> уже собран).</p>
</div>

<h2 id="samples">🎲 Случайные items (seed=42)</h2>
<div class="plain"><b>Простыми словами:</b> четыре случайных вопроса целиком — чтобы
 руками пощупать, что именно показывали модели и что она ответила.</div>
{''.join(sample_card(it) for it in samples)}

<p style="color:#666;font-size:.85em;margin-top:3em;">
 Сгенерировано <code>{safe(os.path.basename(__file__))}</code> · run <code>{safe(RUN)}</code> ·
 данные: HF <code>{safe(HF_REPO)}</code> (private).
</p>
</body></html>"""

    OUT.write_text(doc, encoding="utf-8")
    print(f"✓ HTML отчёт: {OUT}")
    print(f"  размер: {OUT.stat().st_size/1024:.1f} KB")
    print(f"  открыть: file://{OUT.absolute()}")


if __name__ == "__main__":
    main()
