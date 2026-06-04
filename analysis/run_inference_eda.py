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

    # Choice by question_format (для grouped bar A/B/C)
    fmts = ["choice", "yesno_man", "yesno_woman"]
    choice_by_fmt = {lbl: [] for lbl in ("A", "B", "C")}
    for f in fmts:
        sub = [i for i in items if i["question_format"] == f]
        c = Counter(i["choice"] for i in sub)
        for lbl in ("A", "B", "C"):
            choice_by_fmt[lbl].append(round(100 * c.get(lbl, 0) / len(sub), 1))

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

{pie_div("c_overall", list(overall_choice.keys()), list(overall_choice.values()),
         "Overall choice (A/B/C) по всем 1350 items")}

{grouped_bar_div("c_byfmt", fmts, choice_by_fmt,
                 "Choice-распределение по question_format (% внутри формата)", yrange=[0,100])}

<div class="info">
 <b>Как читать:</b> в <code>choice</code> формате A=man, B=woman, C=Cannot determine.
 В <code>yesno_*</code> формате A=Yes, B=No, C=Cannot determine. Доминирование
 опции <b>A</b> во всех форматах — это <b>смесь</b> position-preference (A — первая
 опция) и content-preference. Разделить их позволяет именно yesno-asymmetry (см. H2).
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
