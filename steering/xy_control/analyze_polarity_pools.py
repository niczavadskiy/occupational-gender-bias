"""Pool Stage C preference / capability by H1 polarity (pro-male vs pro-female).

Builds two frozen sets from the FDR catalog + Stage C keep, then tests each
pool with family-level bootstrap (preference) and exact McNemar (capability).
Multiple testing: Bonferroni m=2 (one test per polarity).

Each family keeps its own SOC winner (v, α). Domains are represented in the
set by their Stage C test families; train/val stay out of the confirmatory
pool (already fixed inside each SOC at Stage A).

Equal-SOC summary: mean of per-SOC means (each domain weighs 1).
Family-pooled summary: every test family weighs 1 (large SOCs dominate).

    python -m steering.xy_control.analyze_polarity_pools \
      --stage-c results/steering/soc/xy_control_per_soc_stage_bc_2b_L15_a6/results/steering/xy_control/stage_c/xy_2b_L15_a6_c \
      --scale 2b \
      --out results/steering/soc/polarity_pools_2b_L15_a6

    python -m steering.xy_control.analyze_polarity_pools --help
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from steering.xy_control.analyze_l15_holm import (
    LABEL,
    bootstrap_p_mean,
    load_jsonl,
    mcnemar_exact,
)
from steering.xy_control.domains import CATALOG_JSON, load_catalog

HERE = Path(__file__).resolve().parent


def bonferroni(pvals: list[float]) -> list[float]:
    m = len(pvals)
    return [min(1.0, m * p) for p in pvals]


def stratified_bootstrap_p(
    by_soc: dict[str, list[float]],
    *,
    n_boot: int = 5000,
    seed: int = 20260926,
) -> tuple[float, float, float, float]:
    """Equal-weight mean across SOCs; bootstrap resamples within each SOC."""
    socs = sorted(k for k, xs in by_soc.items() if xs)
    if not socs:
        return 0.0, 0.0, 0.0, 1.0
    means = [sum(by_soc[s]) / len(by_soc[s]) for s in socs]
    mean = sum(means) / len(means)
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        soc_means = []
        for s in socs:
            xs = by_soc[s]
            sample = [xs[rng.randrange(len(xs))] for _ in range(len(xs))]
            soc_means.append(sum(sample) / len(sample))
        boots.append(sum(soc_means) / len(soc_means))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot) - 1]
    leq = sum(1 for b in boots if b <= 0) / n_boot
    geq = sum(1 for b in boots if b >= 0) / n_boot
    p = min(1.0, 2.0 * min(leq, geq))
    return mean, lo, hi, p


def load_keep(stage_c: Path) -> dict[str, Any]:
    keep_path = stage_c / "stagec_keep.json"
    if not keep_path.is_file():
        # Stage B keep next to sibling stage_b is uncommon; require stage_c copy
        raise SystemExit(f"missing {keep_path}")
    return json.loads(keep_path.read_text(encoding="utf-8"))


def catalog_polarity_map(scale: str, catalog_path: Path | None) -> dict[str, dict[str, Any]]:
    cat = load_catalog(catalog_path)
    out: dict[str, dict[str, Any]] = {}
    for d in cat["scales"][scale]["steer"]:
        out[d["slug"]] = {
            "soc_major_title": d["soc_major_title"],
            "slug": d["slug"],
            "polarity": d["polarity"],
            "hypothesized_alpha_sign": d["hypothesized_alpha_sign"],
            "gate": d["gate"],
            "effect": d["effect"],
        }
    return out


def preference_family_rows(stage_c: Path, slug: str, decision: str) -> list[dict[str, Any]]:
    """One row per test family with R = (|θ_b−0.5| − |θ_a−0.5|)×100."""
    if decision == "identity":
        return []
    mpath = stage_c / slug / "metrics.json"
    if not mpath.is_file():
        raise SystemExit(f"missing metrics for steered SOC: {mpath}")
    m = json.loads(mpath.read_text(encoding="utf-8"))
    rows = []
    for f in m["paired"]["per_family"]:
        tb = float(f["theta_prob_before"])
        ta = float(f["theta_prob_after"])
        r = (abs(tb - 0.5) - abs(ta - 0.5)) * 100
        rows.append(
            {
                "scenario_family_id": int(f["scenario_family_id"]),
                "slug": slug,
                "theta_prob_before": tb,
                "theta_prob_after": ta,
                "R_abs_pp": r,
                "male_pct_before": 100.0 / (1.0 + math.exp(-float(f["D_before"])))
                if "D_before" in f
                else None,
                "D_before": float(f.get("D_before", float("nan"))),
                "D_after": float(f.get("D_after", float("nan"))),
            }
        )
    return rows


def capability_discordants(stage_c: Path, slug: str, decision: str) -> tuple[int, int, int]:
    """Return (n_questions, lost, gain). Identity → (0,0,0) contribution skipped."""
    if decision == "identity":
        return 0, 0, 0
    a = {
        int(x.get("question_id", i)): bool(x["correct"])
        for i, x in enumerate(load_jsonl(stage_c / slug / "baseline_mmlu_per_item.jsonl"))
    }
    b = {
        int(x.get("question_id", i)): bool(x["correct"])
        for i, x in enumerate(load_jsonl(stage_c / slug / "mmlu_per_item.jsonl"))
    }
    keys = sorted(set(a) & set(b))
    lost = sum(1 for k in keys if a[k] and not b[k])
    gain = sum(1 for k in keys if (not a[k]) and b[k])
    return len(keys), lost, gain


def build_pool(
    *,
    polarity: str,
    keep: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    stage_c: Path,
    include_identity: bool,
) -> dict[str, Any]:
    domains = []
    family_rows: list[dict[str, Any]] = []
    for d in keep["domains"]:
        slug = d["slug"]
        meta = catalog.get(slug)
        if meta is None or meta["polarity"] != polarity:
            continue
        if d["decision"] == "identity" and not include_identity:
            domains.append(
                {
                    **{k: d[k] for k in (
                        "slug",
                        "soc_major_title",
                        "polarity",
                        "decision",
                        "config_id",
                        "layer",
                        "alpha",
                    ) if k in d},
                    "label": LABEL.get(slug, slug),
                    "n_test_families": 0,
                    "included_in_effect": False,
                    "note": "identity — listed in set, excluded from effect pool",
                }
            )
            continue
        pref = preference_family_rows(stage_c, slug, d["decision"])
        n_q, lost, gain = capability_discordants(stage_c, slug, d["decision"])
        for row in pref:
            row["polarity"] = polarity
            row["decision"] = d["decision"]
            row["config_id"] = d.get("config_id")
            row["alpha"] = d.get("alpha")
            row["layer"] = d.get("layer")
            family_rows.append(row)
        domains.append(
            {
                "slug": slug,
                "soc_major_title": d["soc_major_title"],
                "label": LABEL.get(slug, slug),
                "polarity": polarity,
                "decision": d["decision"],
                "config_id": d.get("config_id"),
                "layer": d.get("layer"),
                "alpha": d.get("alpha"),
                "alpha_matches_prior": d.get("alpha_matches_prior"),
                "n_test_families": len(pref),
                "test_family_ids": [r["scenario_family_id"] for r in pref],
                "n_mmlu": n_q,
                "mmlu_lost": lost,
                "mmlu_gain": gain,
                "included_in_effect": d["decision"] == "steer",
                "mean_R_abs_pp": (
                    sum(r["R_abs_pp"] for r in pref) / len(pref) if pref else None
                ),
            }
        )
    return {
        "polarity": polarity,
        "label": f"pro-{polarity}",
        "n_domains_listed": len(domains),
        "n_domains_effect": sum(1 for x in domains if x.get("included_in_effect")),
        "n_test_families": len(family_rows),
        "domains": domains,
        "families": family_rows,
    }


def analyze_pool(pool: dict[str, Any], *, n_boot: int, seed: int) -> dict[str, Any]:
    fams = pool["families"]
    by_soc: dict[str, list[float]] = defaultdict(list)
    xs = []
    for r in fams:
        by_soc[r["slug"]].append(r["R_abs_pp"])
        xs.append(r["R_abs_pp"])

    fam_mean, fam_lo, fam_hi, fam_p = bootstrap_p_mean(xs, n_boot=n_boot, seed=seed)
    eq_mean, eq_lo, eq_hi, eq_p = stratified_bootstrap_p(
        by_soc, n_boot=n_boot, seed=seed + 1
    )

    lost = sum(int(d.get("mmlu_lost") or 0) for d in pool["domains"] if d.get("included_in_effect"))
    gain = sum(int(d.get("mmlu_gain") or 0) for d in pool["domains"] if d.get("included_in_effect"))
    n_mmlu = sum(int(d.get("n_mmlu") or 0) for d in pool["domains"] if d.get("included_in_effect"))
    cap_p = mcnemar_exact(lost, gain)

    return {
        "polarity": pool["polarity"],
        "n_domains_effect": pool["n_domains_effect"],
        "n_test_families": pool["n_test_families"],
        "preference_family_pooled": {
            "mean_R_abs_pp": round(fam_mean, 4),
            "ci95": [round(fam_lo, 4), round(fam_hi, 4)],
            "p_raw": round(fam_p, 4),
            "note": "each test family weight 1",
        },
        "preference_equal_soc": {
            "mean_R_abs_pp": round(eq_mean, 4),
            "ci95": [round(eq_lo, 4), round(eq_hi, 4)],
            "p_raw": round(eq_p, 4),
            "note": "mean of per-SOC means; bootstrap within SOC then average",
        },
        "capability": {
            "n_questions": n_mmlu,
            "lost": lost,
            "gain": gain,
            "delta_pp": round(100.0 * (gain - lost) / n_mmlu, 3) if n_mmlu else 0.0,
            "p_raw": round(cap_p, 4),
        },
        "per_soc_mean_R": {
            d["slug"]: d["mean_R_abs_pp"]
            for d in pool["domains"]
            if d.get("included_in_effect")
        },
    }


def write_set(path: Path, pool: dict[str, Any], meta: dict[str, Any]) -> None:
    slim_families = [
        {
            "scenario_family_id": r["scenario_family_id"],
            "slug": r["slug"],
            "R_abs_pp": round(r["R_abs_pp"], 6),
            "theta_prob_before": r["theta_prob_before"],
            "theta_prob_after": r["theta_prob_after"],
            "config_id": r.get("config_id"),
            "alpha": r.get("alpha"),
            "layer": r.get("layer"),
        }
        for r in pool["families"]
    ]
    payload = {
        "schema": "steering.xy_control_polarity_pool/v1",
        **meta,
        "polarity": pool["polarity"],
        "label": pool["label"],
        "n_domains_listed": pool["n_domains_listed"],
        "n_domains_effect": pool["n_domains_effect"],
        "n_test_families": pool["n_test_families"],
        "domains": pool["domains"],
        "families": slim_families,
        "split_note": (
            "Families are Stage C test (15%) within each SOC. "
            "Train (70%) fit v_raw; val (15%) chose α. "
            "Pooling does not re-split — each SOC already contributes its own test slice."
        ),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage-c", type=Path, required=True)
    ap.add_argument("--scale", choices=["2b", "4b"], default="2b")
    ap.add_argument("--catalog", type=Path, default=CATALOG_JSON)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument(
        "--include-identity",
        action="store_true",
        help="No-op flag (identity SOCs are always listed in set files, never in effect rows)",
    )
    ap.add_argument(
        "--primary",
        choices=["equal_soc", "family"],
        default="equal_soc",
        help="Which preference estimator enters Bonferroni (default: equal_soc)",
    )
    args = ap.parse_args(argv)

    stage_c = args.stage_c
    if not stage_c.is_dir():
        raise SystemExit(f"stage-c not found: {stage_c}")

    keep = load_keep(stage_c)
    catalog = catalog_polarity_map(args.scale, args.catalog)

    male = build_pool(
        polarity="male",
        keep=keep,
        catalog=catalog,
        stage_c=stage_c,
        include_identity=True,
    )
    female = build_pool(
        polarity="female",
        keep=keep,
        catalog=catalog,
        stage_c=stage_c,
        include_identity=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "scale": args.scale,
        "stage_c": str(stage_c),
        "catalog": str(args.catalog),
        "keep_scale": keep.get("scale"),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "primary_preference": args.primary,
        "correction": "bonferroni",
        "m_tests": 2,
    }
    write_set(args.out / "pro_male_set.json", male, meta)
    write_set(args.out / "pro_female_set.json", female, meta)

    # CSV of all pooled families
    with (args.out / "families.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "polarity",
                "slug",
                "scenario_family_id",
                "R_abs_pp",
                "theta_prob_before",
                "theta_prob_after",
                "alpha",
                "layer",
                "config_id",
            ],
        )
        w.writeheader()
        for pool in (male, female):
            for r in pool["families"]:
                w.writerow(
                    {
                        "polarity": pool["polarity"],
                        "slug": r["slug"],
                        "scenario_family_id": r["scenario_family_id"],
                        "R_abs_pp": f"{r['R_abs_pp']:.6f}",
                        "theta_prob_before": f"{r['theta_prob_before']:.6f}",
                        "theta_prob_after": f"{r['theta_prob_after']:.6f}",
                        "alpha": r.get("alpha"),
                        "layer": r.get("layer"),
                        "config_id": r.get("config_id"),
                    }
                )

    a_male = analyze_pool(male, n_boot=args.n_boot, seed=args.seed)
    a_female = analyze_pool(female, n_boot=args.n_boot, seed=args.seed + 17)

    key = "preference_equal_soc" if args.primary == "equal_soc" else "preference_family_pooled"
    pref_raw = [a_male[key]["p_raw"], a_female[key]["p_raw"]]
    pref_bonf = bonferroni(pref_raw)
    cap_raw = [a_male["capability"]["p_raw"], a_female["capability"]["p_raw"]]
    cap_bonf = bonferroni(cap_raw)

    def attach(a: dict[str, Any], pref_b: float, cap_b: float) -> dict[str, Any]:
        primary = a[key]
        return {
            **a,
            "preference_primary": {
                **primary,
                "estimator": args.primary,
                "p_bonf": round(pref_b, 4),
                "sig_bonf": pref_b < 0.05,
                "direction": (
                    "bias_down"
                    if primary["mean_R_abs_pp"] > 0
                    else "bias_up"
                    if primary["mean_R_abs_pp"] < 0
                    else "flat"
                ),
            },
            "capability_bonf": {
                "p_bonf": round(cap_b, 4),
                "sig_bonf": cap_b < 0.05,
            },
        }

    summary = {
        "schema": "steering.xy_control_polarity_pool_summary/v1",
        **meta,
        "pools": {
            "male": attach(a_male, pref_bonf[0], cap_bonf[0]),
            "female": attach(a_female, pref_bonf[1], cap_bonf[1]),
        },
        "bonferroni": {
            "preference": {
                "m": 2,
                "p_raw": pref_raw,
                "p_bonf": [round(x, 4) for x in pref_bonf],
                "estimator": args.primary,
            },
            "capability": {
                "m": 2,
                "p_raw": cap_raw,
                "p_bonf": [round(x, 4) for x in cap_bonf],
            },
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Human-readable table
    lines = [
        "polarity,n_soc,n_fam,pref_mean_R_pp,pref_ci_lo,pref_ci_hi,p_raw,p_bonf,sig,cap_lost,cap_gain,cap_p_raw,cap_p_bonf",
    ]
    for name, a, pb, cb in (
        ("male", a_male, pref_bonf[0], cap_bonf[0]),
        ("female", a_female, pref_bonf[1], cap_bonf[1]),
    ):
        prim = a[key]
        lines.append(
            ",".join(
                [
                    name,
                    str(a["n_domains_effect"]),
                    str(a["n_test_families"]),
                    f"{prim['mean_R_abs_pp']:.4f}",
                    f"{prim['ci95'][0]:.4f}",
                    f"{prim['ci95'][1]:.4f}",
                    f"{prim['p_raw']:.4f}",
                    f"{pb:.4f}",
                    "yes" if pb < 0.05 else "no",
                    str(a["capability"]["lost"]),
                    str(a["capability"]["gain"]),
                    f"{a['capability']['p_raw']:.4f}",
                    f"{cb:.4f}",
                ]
            )
        )
    (args.out / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {args.out / 'pro_male_set.json'}")
    print(f"wrote {args.out / 'pro_female_set.json'}")
    print(f"wrote {args.out / 'summary.json'}")
    print()
    print(f"primary preference estimator: {args.primary}  |  Bonferroni m=2")
    for name, a, pb in (("male", a_male, pref_bonf[0]), ("female", a_female, pref_bonf[1])):
        prim = a[key]
        print(
            f"  pro-{name}: n_soc={a['n_domains_effect']} n_fam={a['n_test_families']}  "
            f"mean R={prim['mean_R_abs_pp']:+.3f} pp  CI={prim['ci95']}  "
            f"p={prim['p_raw']:.4f}  p_Bonf={pb:.4f}  "
            f"{'SIG' if pb < 0.05 else 'n.s.'}"
        )
        c = a["capability"]
        print(
            f"           capability n={c['n_questions']}  L/G={c['lost']}/{c['gain']}  "
            f"p={c['p_raw']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
