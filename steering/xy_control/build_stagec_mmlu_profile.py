"""Build XY-control Stage C MMLU-Pro profile with 250 held-out questions per SOC.

Full 22-domain equal-n=250 is impossible under disjoint category pools
(business/health). This builder covers the FDR-steered SOC union from the
XY-control catalog (13 domains), reserves Stage B domain_val_v1 IDs for those
same domains, and samples 250 fresh test questions per SOC.

Example:
    python -m steering.xy_control.build_stagec_mmlu_profile
    python -m steering.xy_control.build_stagec_mmlu_profile --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from steering.build_mmlu_profiles import (
    DEFAULT_PARQUET,
    PARQUET_URL,
    TIER_ORDER,
    largest_remainder,
    load_benchmark,
    load_config,
    sha256_file,
    stable_rng,
)
from steering.xy_control.domains import CATALOG_JSON, load_catalog

HERE = Path(__file__).resolve().parent
STEERING_DIR = HERE.parent
DEFAULT_MAP = STEERING_DIR / "configs" / "soc_mmlu_pro_map_v1.yaml"
DEFAULT_VAL = STEERING_DIR / "profiles" / "mmlu_pro_domain_val_v1.json"
DEFAULT_OUT = STEERING_DIR / "profiles" / "mmlu_pro_domain_test_xy_250_v2.json"
N_TEST = 250
MIN_N = 250
MASTER_SEED = 20260920


def fdr_titles(catalog_path: Path = CATALOG_JSON) -> set[str]:
    catalog = load_catalog(catalog_path)
    titles: set[str] = set()
    for scale in catalog["scales"].values():
        for domain in scale["steer"]:
            titles.add(domain["soc_major_title"])
    return titles


def reserve_val_ids(val_profile: Path, titles: set[str]) -> set[int]:
    doc = json.loads(val_profile.read_text(encoding="utf-8"))
    reserved: set[int] = set()
    for domain in doc["domains"]:
        if domain["soc_major_title"] in titles:
            reserved.update(int(q) for q in domain["question_ids"])
    return reserved


def build_test_domains(
    cfg: dict,
    df: pd.DataFrame,
    titles: set[str],
    reserved: set[int],
    *,
    n_test: int,
    min_n: int,
    master_seed: int,
) -> dict[str, dict]:
    pools: dict[str, list[int]] = {
        cat: sorted(int(q) for q in sub["question_id"])
        for cat, sub in df.groupby("category", sort=True)
    }
    meta_by_id = {
        int(r.question_id): {"category": r.category, "src": r.src}
        for r in df.itertuples(index=False)
    }
    generic_weights = {cat: float(len(ids)) for cat, ids in pools.items()}
    used = set(reserved)

    def domain_weights(d: dict) -> dict[str, float]:
        if d["capability_proxy"] == "generic_mmlu_pro":
            return generic_weights
        return dict(d["category_weights"])

    selected = [d for d in cfg["domains"] if d["soc_major_title"] in titles]
    missing = sorted(titles - {d["soc_major_title"] for d in selected})
    if missing:
        raise SystemExit(f"FDR titles missing from map config: {missing}")

    subject = [d for d in selected if d["capability_proxy"] == "subject_mapped"]
    generic = [d for d in selected if d["capability_proxy"] == "generic_mmlu_pro"]
    ordered_subject = sorted(
        subject,
        key=lambda d: (
            sum(len(pools[c]) for c in domain_weights(d)),
            d["soc_major_title"],
        ),
    )
    ordered_generic = sorted(generic, key=lambda d: d["soc_major_title"])

    def pick_domain(d: dict, weights: dict[str, float]) -> dict:
        title = d["soc_major_title"]
        alloc = largest_remainder(weights, n_test)
        picked: list[int] = []
        for cat in sorted(alloc):
            need = alloc[cat]
            if need == 0:
                continue
            avail = [q for q in pools[cat] if q not in used]
            if len(avail) < need:
                raise ValueError(
                    f"{title}: category {cat!r} exhausted "
                    f"({len(avail)} available, need {need}) after reserving Stage B val"
                )
            rng = stable_rng(master_seed, "xy_stagec_250", title, cat)
            drawn = [int(q) for q in rng.choice(np.asarray(avail), size=need, replace=False)]
            used.update(drawn)
            picked.extend(drawn)

        if len(picked) < min_n:
            raise ValueError(f"{title}: n={len(picked)} < min_n={min_n}")

        picked = sorted(picked)
        by_cat: dict[str, int] = {}
        for q in picked:
            cat = meta_by_id[q]["category"]
            by_cat[cat] = by_cat.get(cat, 0) + 1
        return {
            "soc_major_title": title,
            "g_base_items": d["g_base_items"],
            "preference_stratum": d["preference_stratum"],
            "capability_proxy": d["capability_proxy"],
            "mapping_confidence": d["mapping_confidence"],
            "score_tier": d["score_tier"],
            "n": len(picked),
            "n_by_category": dict(sorted(by_cat.items())),
            "question_ids": picked,
            "items": [
                {
                    "question_id": q,
                    "category": meta_by_id[q]["category"],
                    "src": meta_by_id[q]["src"],
                }
                for q in picked
            ],
        }

    test_domains: dict[str, dict] = {}
    for d in ordered_subject:
        test_domains[d["soc_major_title"]] = pick_domain(d, domain_weights(d))

    # Generic domains draw last, reweighted by remaining category mass so
    # depleted subject-mapped pools (e.g. health) do not hard-fail equal-n.
    for d in ordered_generic:
        remaining = {
            cat: float(sum(1 for q in ids if q not in used)) for cat, ids in pools.items()
        }
        if sum(remaining.values()) < n_test:
            raise ValueError(
                f"{d['soc_major_title']}: remaining pool "
                f"{int(sum(remaining.values()))} < n_test={n_test}"
            )
        test_domains[d["soc_major_title"]] = pick_domain(d, remaining)
    return test_domains


def profile_document(
    *,
    cfg: dict,
    cfg_path: Path,
    parquet: Path,
    df: pd.DataFrame,
    domains: dict[str, dict],
    val_profile: Path,
    reserved_n: int,
    n_test: int,
    master_seed: int,
) -> dict:
    tiers: dict[str, int] = {}
    for d in domains.values():
        tiers[d["score_tier"]] = tiers.get(d["score_tier"], 0) + 1
    return {
        "schema": "steering.mmlu_pro_domain_profile/v1",
        "profile": "test",
        "role": "XY-control Stage C — 250 held-out questions per FDR SOC",
        "map_version": f"{cfg['version']}+xy250",
        "map_config": cfg_path.name,
        "map_sha256": sha256_file(cfg_path),
        "benchmark": {
            **cfg["benchmark"],
            "n_rows": int(len(df)),
            "parquet_sha256": sha256_file(parquet),
        },
        "sampling": {
            "n_per_soc": n_test,
            "master_seed": master_seed,
            "disjoint_across_domains": True,
            "disjoint_across_profiles": True,
            "scope": "xy_control_fdr_union",
            "reserved_val_profile": val_profile.name,
            "n_reserved_val_ids": reserved_n,
        },
        "scoring": cfg["scoring"],
        "n_domains": len(domains),
        "n_questions": sum(d["n"] for d in domains.values()),
        "n_domains_by_tier": {
            k: tiers[k] for k in sorted(tiers, key=lambda t: TIER_ORDER[t])
        },
        "domains": [domains[k] for k in sorted(domains)],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_MAP)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--catalog", type=Path, default=CATALOG_JSON)
    ap.add_argument("--val-profile", type=Path, default=DEFAULT_VAL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--seed", type=int, default=MASTER_SEED)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    df = load_benchmark(args.parquet, cfg)
    titles = fdr_titles(args.catalog)
    reserved = reserve_val_ids(args.val_profile, titles)
    domains = build_test_domains(
        cfg,
        df,
        titles,
        reserved,
        n_test=args.n_test,
        min_n=args.min_n,
        master_seed=args.seed,
    )
    doc = profile_document(
        cfg=cfg,
        cfg_path=args.config,
        parquet=args.parquet,
        df=df,
        domains=domains,
        val_profile=args.val_profile,
        reserved_n=len(reserved),
        n_test=args.n_test,
        master_seed=args.seed,
    )
    text = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"

    if args.verify:
        if not args.out.is_file():
            raise SystemExit(f"missing profile for --verify: {args.out}")
        on_disk = args.out.read_text(encoding="utf-8")
        if on_disk != text:
            raise SystemExit(f"profile drift: {args.out}")
        print(f"OK {args.out.name}: {doc['n_domains']} domains × {args.n_test}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(
        f"wrote {args.out} · {doc['n_domains']} domains · "
        f"{doc['n_questions']} questions · reserved_val={len(reserved)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        if isinstance(exc, SystemExit):
            raise
        if "parquet" in str(exc).lower() or "не найден" in str(exc):
            print(f"download: curl -L -o \"{DEFAULT_PARQUET}\" {PARQUET_URL}", file=sys.stderr)
        raise SystemExit(1) from exc
