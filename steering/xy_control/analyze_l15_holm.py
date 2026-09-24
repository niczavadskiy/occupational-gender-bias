"""Compute Stage B MMLU table + Stage C preference/capability Holm for a pack.

    python -m steering.xy_control.analyze_l15_holm \
      --stage-c results/steering/xy_control/stage_c/xy_2b_L15_a6_c \
      --stage-b results/steering/xy_control/stage_b/xy_2b_L15_a6_b

Or from an extracted pack root that contains stage_b/ and stage_c/ under
results/steering/xy_control/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

LABEL = {
    "architecture_and_engineering_occupations": "Architecture",
    "computer_and_mathematical_occupations": "Computer & Math",
    "construction_and_extraction_occupations": "Construction",
    "educational_instruction_and_library_occupations": "Education",
    "food_preparation_and_serving_related_occupations": "Food Preparation",
    "healthcare_practitioners_and_technical_occupatio": "Healthcare Pract.",
    "healthcare_support_occupations": "Healthcare Support",
    "installation_maintenance_and_repair_occupations": "Installation",
    "life_physical_and_social_science_occupations": "Life / Phys. / Social",
    "office_and_administrative_support_occupations": "Office & Admin",
    "personal_care_and_service_occupations": "Personal Care",
    "production_occupations": "Production",
    "transportation_and_material_moving_occupations": "Transportation",
}


def holm(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    out = [1.0] * n
    running = 0.0
    for rank, i in enumerate(order):
        adj = (n - rank) * pvals[i]
        running = max(running, adj)
        out[i] = min(1.0, running)
    return out


def mcnemar_exact(lost: int, gain: int) -> float:
    d = lost + gain
    if d == 0:
        return 1.0
    lo = min(lost, gain)
    return min(1.0, 2.0 * sum(math.comb(d, i) for i in range(lo + 1)) / (2**d))


def bootstrap_p_mean(xs: list[float], *, n_boot: int = 5000, seed: int = 20260820) -> tuple[float, float, float, float]:
    """Two-sided bootstrap p for H0: mean=0; also return mean and 95% CI."""
    if not xs:
        return 0.0, 0.0, 0.0, 1.0
    rng = random.Random(seed)
    n = len(xs)
    mean = sum(xs) / n
    boots = []
    for _ in range(n_boot):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot) - 1]
    # p ≈ 2 * min( fraction(boot<=0), fraction(boot>=0) )
    leq = sum(1 for b in boots if b <= 0) / n_boot
    geq = sum(1 for b in boots if b >= 0) / n_boot
    p = min(1.0, 2.0 * min(leq, geq))
    return mean, lo, hi, p


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stage_b_rows(stage_b: Path) -> list[dict]:
    keep = json.loads((stage_b / "stagec_keep.json").read_text(encoding="utf-8"))
    rows = []
    for d in keep["domains"]:
        slug = d["slug"]
        ranking = stage_b / slug / "ranking.csv"
        base = steered = loss = None
        winner = d.get("config_id", "baseline")
        if ranking.is_file() and d["decision"] == "steer":
            for r in csv.DictReader(ranking.open(encoding="utf-8")):
                if r["config_id"] == winner:
                    base = float(r["mmlu_accuracy_baseline"])
                    steered = float(r["mmlu_accuracy"])
                    loss = float(r["cap_loss"])
                    break
        elif ranking.is_file():
            # identity: show best failed candidate or baseline-only row if present
            cands = list(csv.DictReader(ranking.open(encoding="utf-8")))
            if cands:
                # pick min cap_loss among candidates for display of why identity
                best = min(cands, key=lambda r: float(r["cap_loss"]))
                base = float(best["mmlu_accuracy_baseline"])
                steered = float(best["mmlu_accuracy"])
                loss = float(best["cap_loss"])
                winner = "—"
        rows.append(
            {
                "label": LABEL.get(slug, slug),
                "slug": slug,
                "winner": winner if d["decision"] == "steer" else "—",
                "decision": d["decision"],
                "mmlu_baseline": base,
                "mmlu_steered": steered,
                "cap_loss": loss,
                "layer": d.get("layer"),
                "alpha": d.get("alpha"),
            }
        )
    return rows


def stage_c_preference(stage_c: Path) -> list[dict]:
    summary = list(csv.DictReader((stage_c / "summary.csv").open(encoding="utf-8")))
    out = []
    raw_p: list[float] = []
    steer_idx: list[int] = []
    for r in summary:
        slug = r["slug"]
        decision = r["decision"]
        n = int(float(r["n_test_families"]))
        gap_b = float(r["baseline_gender_gap"])
        gap_a = float(r["test_gender_gap"])
        agg_b = abs(1 / (1 + math.exp(-gap_b)) - 0.5) * 100
        agg_a = abs(1 / (1 + math.exp(-gap_a)) - 0.5) * 100
        before = after = red = lo = hi = p = None
        if decision == "steer":
            mpath = stage_c / slug / "metrics.json"
            m = json.loads(mpath.read_text(encoding="utf-8"))
            t = m["paired"]["theta_prob_axis"]
            before = float(t["mean_abs_before"]) * 100
            after = float(t["mean_abs_after"]) * 100
            # per-family reductions from R_gender on |theta-0.5| path: use paired per_family
            fams = m["paired"]["per_family"]
            # R for absolute theta bias: |θ_before-0.5| - |θ_after-0.5|
            xs = [
                abs(float(f["theta_prob_before"]) - 0.5) - abs(float(f["theta_prob_after"]) - 0.5)
                for f in fams
            ]
            # metrics store reduction already scaled? values are probabilities 0-1
            xs = [x * 100 for x in xs]
            red, lo, hi, p = bootstrap_p_mean(xs)
            steer_idx.append(len(out))
            raw_p.append(p)
        else:
            # identity: bias from baseline only
            base_jsonl = stage_c / slug / "baseline_preference_per_item.jsonl"
            if base_jsonl.is_file():
                # fall back to aggregate-only display
                before = after = abs(1 / (1 + math.exp(-gap_b)) - 0.5) * 100  # placeholder
            before = after = None
            # better: from summary test_mean_abs_delta is AFTER; need baseline from metrics absence
            before = after = float("nan")
        out.append(
            {
                "label": LABEL.get(slug, slug),
                "slug": slug,
                "n": n,
                "decision": decision,
                "aggregateBefore": round(agg_b, 3),
                "aggregateAfter": round(agg_a, 3),
                "aggregateReduction": round(agg_b - agg_a, 3),
                "gap_before": gap_b,
                "gap_after": gap_a,
                "before": None if before is None or (isinstance(before, float) and math.isnan(before)) else round(before, 3),
                "after": None if after is None or (isinstance(after, float) and math.isnan(after)) else round(after, 3),
                "reduction": None if red is None else round(red, 3),
                "ci": None if lo is None else [round(lo, 3), round(hi, 3)],
                "p": p,
                "holm": None,
                "sigHolm": False,
            }
        )
    if raw_p:
        adj = holm(raw_p)
        for i, h in zip(steer_idx, adj):
            out[i]["holm"] = round(h, 4)
            out[i]["sigHolm"] = h < 0.05 and out[i]["reduction"] is not None
    # fill identity family abs from JSONL if possible
    from collections import defaultdict

    for row in out:
        if row["decision"] != "identity":
            continue
        slug = row["slug"]
        path = stage_c / slug / "baseline_preference_per_item.jsonl"
        if not path.is_file():
            continue
        by_fam: dict[int, list[float]] = defaultdict(list)
        for item in load_jsonl(path):
            # theta from constrained probs if present
            pm = item.get("prob_constrained_man") or item.get("prob_man")
            pw = item.get("prob_constrained_woman") or item.get("prob_woman")
            if pm is None or pw is None:
                continue
            theta = float(pm) / (float(pm) + float(pw)) if (float(pm) + float(pw)) > 0 else 0.5
            by_fam[int(item["scenario_family_id"])].append(theta)
        if not by_fam:
            continue
        vals = [abs(sum(v) / len(v) - 0.5) * 100 for v in by_fam.values()]
        bias = sum(vals) / len(vals)
        row["before"] = row["after"] = round(bias, 3)
        row["reduction"] = 0.0
        row["ci"] = None
        row["n"] = len(vals)
    return out


def stage_c_capability(stage_c: Path) -> list[dict]:
    summary = list(csv.DictReader((stage_c / "summary.csv").open(encoding="utf-8")))
    out = []
    raw_p: list[float] = []
    steer_idx: list[int] = []
    for r in summary:
        slug = r["slug"]
        decision = r["decision"]
        base_acc = float(r["mmlu_test_accuracy_baseline"]) * 100
        after_acc = float(r["mmlu_test_accuracy"]) * 100
        lost = gain = 0
        p = 1.0
        if decision == "steer":
            a = {
                int(x.get("question_id", i)): bool(x["correct"])
                for i, x in enumerate(load_jsonl(stage_c / slug / "baseline_mmlu_per_item.jsonl"))
            }
            b_path = stage_c / slug / "mmlu_per_item.jsonl"
            b = {
                int(x.get("question_id", i)): bool(x["correct"])
                for i, x in enumerate(load_jsonl(b_path))
            }
            keys = sorted(set(a) & set(b))
            lost = sum(1 for k in keys if a[k] and not b[k])
            gain = sum(1 for k in keys if (not a[k]) and b[k])
            p = mcnemar_exact(lost, gain)
            steer_idx.append(len(out))
            raw_p.append(p)
        out.append(
            {
                "label": LABEL.get(slug, slug),
                "slug": slug,
                "decision": decision,
                "baseline": round(base_acc, 1),
                "after": round(after_acc, 1),
                "lost": lost,
                "gain": gain,
                "p": round(p, 4),
                "holm": None,
                "sigHolm": False,
            }
        )
    if raw_p:
        adj = holm(raw_p)
        for i, h in zip(steer_idx, adj):
            out[i]["holm"] = round(h, 4)
            out[i]["sigHolm"] = h < 0.05
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-c", type=Path, required=True)
    ap.add_argument("--stage-b", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    payload = {
        "stage_b": stage_b_rows(args.stage_b),
        "preference": stage_c_preference(args.stage_c),
        "capability": stage_c_capability(args.stage_c),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
