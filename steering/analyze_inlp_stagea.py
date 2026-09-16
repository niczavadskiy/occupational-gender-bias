"""
Шаг 1 (CPU): разбор Phase B — R_gender vs R_slot, насыщение k=8 vs 16, vs random.

Вход — каталог прогона run_inlp_stagea (нужен ranking.csv):
  results/steering/inlp_stage_a/<tag>/ranking.csv
  опционально <config_id>/per_family.jsonl для SOC-гетерогенности

Запуск:
  python -m steering.analyze_inlp_stagea \\
      --run-dir results/steering/inlp_stage_a/inlp_v1 \\
      --layers 15 --focus-ranks 8,16

  # если полного артефакта ещё нет — из лога Vast:
  python -m steering.analyze_inlp_stagea \\
      --ranking steering/analysis_fixtures/inlp_v1_ranking_from_vast_log.csv \\
      --layers 15
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent
DEFAULT_FIXTURE = STEERING_DIR / "analysis_fixtures" / "inlp_v1_ranking_from_vast_log.csv"


def _f(x) -> float | None:
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def load_ranking(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append(
            {
                "config_id": r.get("config_id", ""),
                "role": r.get("role", ""),
                "basis": r.get("basis", ""),
                "layer": int(float(r["layer"])),
                "rank": int(float(r["rank"])),
                "alpha": _f(r.get("alpha")) or 1.0,
                "auc_after_removal": _f(r.get("auc_after_removal")),
                "mean_R_gender": _f(r.get("mean_R_gender")),
                "ci_lo": _f(r.get("ci_lo")),
                "ci_hi": _f(r.get("ci_hi")),
                "mean_R_slot": _f(r.get("mean_R_slot")),
                "mean_theta_dev_reduction": _f(r.get("mean_theta_dev_reduction")),
                "flip_rate": _f(r.get("flip_rate")),
                "mean_relative_perturbation": _f(r.get("mean_relative_perturbation")),
                "fraction_R_positive": _f(r.get("fraction_R_positive")),
            }
        )
    return out


def pick(rows: list[dict], *, layer: int, rank: int, role: str, alpha: float = 1.0) -> dict | None:
    hits = [
        r
        for r in rows
        if r["layer"] == layer
        and r["rank"] == rank
        and r["role"] == role
        and abs(r["alpha"] - alpha) < 1e-9
    ]
    return hits[0] if hits else None


def fmt_pm(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:+.4f}"


def ratio(g: float | None, s: float | None) -> str:
    if g is None or s is None or abs(s) < 1e-12:
        return "—"
    return f"{g / s:.2f}"


def ci_excludes_zero(r: dict | None) -> bool | None:
    if r is None or r["ci_lo"] is None or r["ci_hi"] is None:
        return None
    return r["ci_lo"] > 0 or r["ci_hi"] < 0


def analyze_layer(rows: list[dict], layer: int, focus_ranks: list[int]) -> dict:
    cand = sorted(
        [r for r in rows if r["layer"] == layer and r["role"] == "candidate"],
        key=lambda r: (r["rank"], r["alpha"]),
    )
    ctrl = sorted(
        [r for r in rows if r["layer"] == layer and r["role"] == "control"],
        key=lambda r: (r["rank"], r["alpha"]),
    )
    table = []
    for r in cand:
        rnd = pick(rows, layer=layer, rank=r["rank"], role="control", alpha=r["alpha"])
        table.append(
            {
                "rank": r["rank"],
                "auc": r["auc_after_removal"],
                "R_gender": r["mean_R_gender"],
                "ci_lo": r["ci_lo"],
                "ci_hi": r["ci_hi"],
                "R_slot": r["mean_R_slot"],
                "R_gender_over_R_slot": ratio(r["mean_R_gender"], r["mean_R_slot"]),
                "theta_dev_red": r["mean_theta_dev_reduction"],
                "flip": r["flip_rate"],
                "rel_delta": r["mean_relative_perturbation"],
                "R_gender_random": rnd["mean_R_gender"] if rnd else None,
                "R_slot_random": rnd["mean_R_slot"] if rnd else None,
                "delta_vs_random": (
                    None
                    if r["mean_R_gender"] is None or rnd is None or rnd["mean_R_gender"] is None
                    else r["mean_R_gender"] - rnd["mean_R_gender"]
                ),
                "ci_excludes_0": ci_excludes_zero(r),
            }
        )

    # saturation: focus ranks
    sat = []
    for i, k in enumerate(focus_ranks):
        a = pick(rows, layer=layer, rank=k, role="candidate")
        if a is None:
            continue
        entry = {
            "rank": k,
            "R_gender": a["mean_R_gender"],
            "R_slot": a["mean_R_slot"],
            "ratio": ratio(a["mean_R_gender"], a["mean_R_slot"]),
            "vs_random": None,
        }
        rnd = pick(rows, layer=layer, rank=k, role="control")
        if a["mean_R_gender"] is not None and rnd and rnd["mean_R_gender"] is not None:
            entry["vs_random"] = a["mean_R_gender"] - rnd["mean_R_gender"]
        if i > 0:
            prev = pick(rows, layer=layer, rank=focus_ranks[i - 1], role="candidate")
            if prev and a["mean_R_gender"] is not None and prev["mean_R_gender"] is not None:
                entry["delta_R_gender_from_prev"] = a["mean_R_gender"] - prev["mean_R_gender"]
                denom = abs(prev["mean_R_gender"])
                entry["rel_gain_from_prev"] = (
                    entry["delta_R_gender_from_prev"] / denom if denom > 1e-12 else None
                )
        sat.append(entry)

    best = max(
        (t for t in table if t["R_gender"] is not None),
        key=lambda t: t["R_gender"],
        default=None,
    )
    return {
        "layer": layer,
        "n_candidate": len(cand),
        "n_control": len(ctrl),
        "curve": table,
        "saturation": sat,
        "best_rank": best["rank"] if best else None,
        "best_R_gender": best["R_gender"] if best else None,
    }


def soc_summary(run_dir: Path, config_id: str) -> dict | None:
    path = run_dir / config_id / "per_family.jsonl"
    if not path.exists():
        return None
    by_soc: dict[str, list[float]] = defaultdict(list)
    rg, rs = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            by_soc[str(row.get("soc_major_title") or "UNKNOWN")].append(float(row["R_gender"]))
            rg.append(float(row["R_gender"]))
            rs.append(float(row["R_slot"]))
    soc_rows = []
    for soc, vals in sorted(by_soc.items(), key=lambda kv: -np.mean(kv[1])):
        arr = np.asarray(vals, dtype=np.float64)
        soc_rows.append(
            {
                "soc_major_title": soc,
                "n": int(arr.size),
                "mean_R_gender": float(arr.mean()),
                "frac_positive": float(np.mean(arr > 0)),
            }
        )
    return {
        "config_id": config_id,
        "n_families": len(rg),
        "corr_R_gender_R_slot": float(np.corrcoef(rg, rs)[0, 1]) if len(rg) > 1 else float("nan"),
        "mean_R_gender": float(np.mean(rg)),
        "mean_R_slot": float(np.mean(rs)),
        "per_soc": soc_rows,
    }


def markdown_report(
    analyses: list[dict],
    *,
    source: str,
    focus_ranks: list[int],
    soc: list[dict | None],
) -> str:
    lines = [
        "# INLP Stage A — post-hoc (CPU)",
        "",
        f"- Source: `{source}`",
        f"- Focus ranks: {focus_ranks}",
        "",
        "## Verdict checklist",
        "",
        "1. На фокус-слое INLP `R_gender` >> random того же k?",
        "2. Насыщение: прирост k=8→16 мал относительно 1→8?",
        "3. `R_gender / R_slot`: близок к 1 → эффект общий; ≫1 → gender-специфичнее.",
        "",
    ]
    for a in analyses:
        L = a["layer"]
        lines += [
            f"## L{L}",
            "",
            "| k | AUC | R_gender | CI∌0 | R_slot | R_g/R_s | R_gender rand | Δ vs rand |",
            "|---:|---:|---:|:---:|---:|---:|---:|---:|",
        ]
        for t in a["curve"]:
            excl = t["ci_excludes_0"]
            excl_s = "yes" if excl else ("no" if excl is False else "—")
            auc = f"{t['auc']:.3f}" if t["auc"] is not None else "—"
            lines.append(
                f"| {t['rank']} | {auc} | {fmt_pm(t['R_gender'])} | {excl_s} | "
                f"{fmt_pm(t['R_slot'])} | {t['R_gender_over_R_slot']} | "
                f"{fmt_pm(t['R_gender_random'])} | {fmt_pm(t['delta_vs_random'])} |"
            )
        lines += ["", "### Saturation", ""]
        for s in a["saturation"]:
            extra = ""
            if "delta_R_gender_from_prev" in s:
                extra = (
                    f", ΔR from prev={fmt_pm(s['delta_R_gender_from_prev'])}"
                    f" (rel {fmt_pm(s.get('rel_gain_from_prev'))})"
                )
            lines.append(
                f"- k={s['rank']}: R_gender={fmt_pm(s['R_gender'])}, "
                f"R_slot={fmt_pm(s['R_slot'])}, ratio={s['ratio']}, "
                f"vs_random={fmt_pm(s['vs_random'])}{extra}"
            )
        if a["best_rank"] is not None:
            lines.append(
                f"\nBest INLP R_gender on L{L}: **k={a['best_rank']}** "
                f"({fmt_pm(a['best_R_gender'])})."
            )
        lines.append("")

    for s in soc:
        if not s:
            continue
        lines += [
            f"## SOC heterogeneity — `{s['config_id']}`",
            "",
            f"- corr(R_gender, R_slot) families = {s['corr_R_gender_R_slot']:.3f}",
            f"- mean R_gender={s['mean_R_gender']:+.4f}, mean R_slot={s['mean_R_slot']:+.4f}",
            "",
            "| soc | n | mean R_gender | frac(R>0) |",
            "| :--- | ---: | ---: | ---: |",
        ]
        for row in s["per_soc"]:
            lines.append(
                f"| {row['soc_major_title']} | {row['n']} | "
                f"{row['mean_R_gender']:+.4f} | {row['frac_positive']:.2f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=None, help="…/inlp_stage_a/<tag>")
    ap.add_argument("--ranking", type=Path, default=None, help="прямой путь к ranking.csv")
    ap.add_argument("--layers", default="15", help="слои через запятую")
    ap.add_argument("--focus-ranks", default="8,16")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--with-soc", action="store_true", help="читать per_family.jsonl фокус-конфигов")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.ranking is not None:
        ranking_path = args.ranking
        run_dir = args.run_dir
    elif args.run_dir is not None:
        run_dir = args.run_dir
        ranking_path = run_dir / "ranking.csv"
    else:
        # дефолт: fixture из лога Vast, если полного прогона нет локально
        default_run = REPO_ROOT / "results" / "steering" / "inlp_stage_a" / "inlp_v1"
        if (default_run / "ranking.csv").exists():
            run_dir = default_run
            ranking_path = default_run / "ranking.csv"
        else:
            run_dir = None
            ranking_path = DEFAULT_FIXTURE
            print(f"нет {default_run / 'ranking.csv'} — беру fixture {ranking_path}")

    if not ranking_path.exists():
        raise SystemExit(f"нет ranking: {ranking_path}")

    rows = [r for r in load_ranking(ranking_path) if abs(r["alpha"] - args.alpha) < 1e-9]
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    focus = [int(x) for x in args.focus_ranks.split(",") if x.strip()]
    analyses = [analyze_layer(rows, L, focus) for L in layers]

    soc_reports: list[dict | None] = []
    if args.with_soc and run_dir is not None:
        for L in layers:
            for k in focus:
                cid = f"inlp__L{L}__k{k}__a{args.alpha:g}".replace(".", "p")
                soc_reports.append(soc_summary(run_dir, cid))
    else:
        soc_reports = []

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = (run_dir / "posthoc") if run_dir is not None else (
            REPO_ROOT / "results" / "steering" / "inlp_stage_a" / "posthoc_from_fixture"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    md = markdown_report(analyses, source=str(ranking_path), focus_ranks=focus, soc=soc_reports)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    payload = {
        "source_ranking": str(ranking_path),
        "layers": layers,
        "focus_ranks": focus,
        "alpha": args.alpha,
        "analyses": analyses,
        "soc": [s for s in soc_reports if s],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # компактный CSV по фокус-слою
    flat = []
    for a in analyses:
        for t in a["curve"]:
            flat.append({"layer": a["layer"], **t})
    if flat:
        keys = list(flat[0].keys())
        with (out_dir / "layer_curves.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(flat)

    print(md)
    print(f"→ {out_dir / 'summary.md'}")
    print(f"→ {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
