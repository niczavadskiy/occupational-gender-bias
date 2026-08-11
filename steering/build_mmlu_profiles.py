"""
Сборка замороженных MMLU-Pro профилей для доменного capability-контроля steering.

Вход:
  - steering/configs/soc_mmlu_pro_map_v1.yaml — маппинг soc_major_title ↔ MMLU-Pro
  - steering/.cache/mmlu_pro_test.parquet     — split=test датасета TIGER-Lab/MMLU-Pro

Выход (steering/profiles/):
  - mmlu_pro_domain_val_<ver>.json    — Stage B, по n_per_soc вопросов на домен
  - mmlu_pro_domain_test_<ver>.json   — Stage C, те же домены, непересекающиеся вопросы
  - mmlu_pro_overall_smoke_<ver>.json — общий smoke, равное n на категорию
  - coverage_<ver>.md                 — таблица покрытия по всем 22 доменам

Гарантии:
  - детерминизм: seed берётся из конфига, выбор зависит только от (master_seed, домен,
    категория) и от содержимого parquet — повторный запуск даёт побайтово те же файлы;
  - дизъюнктность: question_id не встречается дважды ни между доменами, ни между
    профилями (val ∩ test = ∅), ни в overall smoke.

Запуск из корня репозитория:
    python -m steering.build_mmlu_profiles
    python -m steering.build_mmlu_profiles --verify   # профили на диске == пересборке
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

STEERING_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = STEERING_DIR / "configs" / "soc_mmlu_pro_map_v1.yaml"
DEFAULT_PARQUET = STEERING_DIR / ".cache" / "mmlu_pro_test.parquet"
DEFAULT_OUT_DIR = STEERING_DIR / "profiles"

PARQUET_URL = (
    "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/"
    "data/test-00000-of-00001.parquet"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_rng(*parts: Any) -> np.random.Generator:
    """RNG, зависящий только от переданных ключей (не от порядка вызовов)."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return np.random.default_rng(seed)


def largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    """Разложить total по категориям пропорционально весам, без потери единиц."""
    if total <= 0 or not weights:
        return {k: 0 for k in weights}
    norm = sum(weights.values())
    if norm <= 0:
        raise ValueError(f"category_weights sum to {norm}")
    exact = {k: total * w / norm for k, w in weights.items()}
    base = {k: int(v) for k, v in exact.items()}
    left = total - sum(base.values())
    order = sorted(exact, key=lambda k: (-(exact[k] - base[k]), k))
    for k in order[:left]:
        base[k] += 1
    return base


def expected_tier(proxy: str, confidence: str) -> str:
    if proxy == "generic_mmlu_pro":
        return "generic"
    return "secondary" if confidence == "low" else "primary"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    titles = [d["soc_major_title"] for d in cfg["domains"]]
    if len(titles) != len(set(titles)):
        raise ValueError("duplicate soc_major_title in config")

    known = set(cfg["benchmark"]["categories"])
    for d in cfg["domains"]:
        title = d["soc_major_title"]
        cats = list(d.get("mmlu_pro_categories") or [])
        weights = dict(d.get("category_weights") or {})
        proxy = d["capability_proxy"]
        confidence = d["mapping_confidence"]

        unknown = sorted(set(cats) - known)
        if unknown:
            raise ValueError(f"{title}: unknown categories {unknown}")
        if sorted(weights) != sorted(cats):
            raise ValueError(f"{title}: category_weights keys != mmlu_pro_categories")
        if cats and abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError(f"{title}: category_weights sum != 1.0")
        if proxy not in ("subject_mapped", "generic_mmlu_pro"):
            raise ValueError(f"{title}: unknown capability_proxy {proxy!r}")
        if (proxy == "subject_mapped") != bool(cats):
            raise ValueError(f"{title}: capability_proxy {proxy!r} inconsistent with categories")
        if (confidence == "none") != (proxy == "generic_mmlu_pro"):
            raise ValueError(f"{title}: mapping_confidence {confidence!r} inconsistent with proxy")
        want = expected_tier(proxy, confidence)
        if d["score_tier"] != want:
            raise ValueError(f"{title}: score_tier {d['score_tier']!r} should be {want!r}")
        if d["preference_stratum"] not in ("reliable", "small"):
            raise ValueError(f"{title}: unknown preference_stratum")
        if not d.get("proxy_note"):
            raise ValueError(f"{title}: missing proxy_note")
    return cfg


def load_benchmark(parquet: Path, cfg: dict) -> pd.DataFrame:
    if not parquet.exists():
        raise SystemExit(
            f"MMLU-Pro parquet не найден: {parquet}\n"
            f"Скачать: curl -L -o \"{parquet}\" {PARQUET_URL}"
        )
    df = pd.read_parquet(parquet, columns=["question_id", "category", "src"])
    if df["question_id"].duplicated().any():
        raise ValueError("duplicate question_id in benchmark parquet")
    missing = set(cfg["benchmark"]["categories"]) - set(df["category"].unique())
    if missing:
        raise ValueError(f"categories missing from parquet: {sorted(missing)}")
    return df


TIER_ORDER = {"primary": 0, "secondary": 1, "generic": 2}


def build_domain_profiles(cfg: dict, df: pd.DataFrame) -> tuple[dict, dict, list[dict]]:
    """Вернуть (val_domains, test_domains, coverage_rows) с глобальной дизъюнктностью."""
    sampling = cfg["sampling"]
    n_per_soc = int(sampling["n_per_soc"])
    min_n = int(sampling["min_n_per_soc"])
    master_seed = sampling["master_seed"]

    pools: dict[str, list[int]] = {
        cat: sorted(int(q) for q in sub["question_id"])
        for cat, sub in df.groupby("category", sort=True)
    }
    meta_by_id = {
        int(r.question_id): {"category": r.category, "src": r.src}
        for r in df.itertuples(index=False)
    }
    # Для generic-доменов состав повторяет весь бенчмарк: доля категории = её размер.
    generic_weights = {cat: float(len(ids)) for cat, ids in pools.items()}
    used: set[int] = set()

    def domain_weights(d: dict) -> dict[str, float]:
        if d["capability_proxy"] == "generic_mmlu_pro":
            return generic_weights
        return dict(d["category_weights"])

    # Дефицитные домены выбирают первыми (generic тянут из всего пула и идут последними);
    # порядок детерминирован и не зависит от порядка записей в конфиге.
    ordered = sorted(
        cfg["domains"],
        key=lambda d: (
            sum(len(pools[c]) for c in domain_weights(d)),
            d["soc_major_title"],
        ),
    )

    val_domains: dict[str, dict] = {}
    test_domains: dict[str, dict] = {}

    for d in ordered:
        title = d["soc_major_title"]
        alloc = largest_remainder(domain_weights(d), n_per_soc)
        picked_val: list[int] = []
        picked_test: list[int] = []

        for cat in sorted(alloc):
            need = 2 * alloc[cat]
            if need == 0:
                continue
            avail = [q for q in pools[cat] if q not in used]
            if len(avail) < need:
                raise ValueError(
                    f"{title}: пул категории {cat!r} исчерпан "
                    f"({len(avail)} доступно, нужно {need}) — уменьшите n_per_soc "
                    f"или пересмотрите маппинг"
                )
            rng = stable_rng(master_seed, title, cat)
            drawn = [int(q) for q in rng.choice(np.asarray(avail), size=need, replace=False)]
            used.update(drawn)
            half = need // 2
            picked_val.extend(drawn[:half])
            picked_test.extend(drawn[half:])

        if min(len(picked_val), len(picked_test)) < min_n:
            raise ValueError(
                f"{title}: n={min(len(picked_val), len(picked_test))} < min_n_per_soc={min_n}"
            )

        for bucket, picked in ((val_domains, picked_val), (test_domains, picked_test)):
            picked = sorted(picked)
            by_cat: dict[str, int] = {}
            for q in picked:
                cat = meta_by_id[q]["category"]
                by_cat[cat] = by_cat.get(cat, 0) + 1
            bucket[title] = {
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
                    {"question_id": q, "category": meta_by_id[q]["category"], "src": meta_by_id[q]["src"]}
                    for q in picked
                ],
            }

    coverage_rows = [
        {
            "soc_major_title": d["soc_major_title"],
            "g_base_items": d["g_base_items"],
            "preference_stratum": d["preference_stratum"],
            "mmlu_pro_categories": list(d.get("mmlu_pro_categories") or []),
            "capability_proxy": d["capability_proxy"],
            "mapping_confidence": d["mapping_confidence"],
            "score_tier": d["score_tier"],
            "n_val": val_domains[d["soc_major_title"]]["n"],
            "n_test": test_domains[d["soc_major_title"]]["n"],
            "proxy_note": d["proxy_note"],
        }
        for d in cfg["domains"]
    ]
    coverage_rows.sort(key=lambda r: (TIER_ORDER[r["score_tier"]], r["soc_major_title"]))
    return val_domains, test_domains, coverage_rows


def build_overall_smoke(cfg: dict, df: pd.DataFrame, used: set[int]) -> dict:
    sampling = cfg["sampling"]
    spec = sampling["overall_smoke"]
    n_per_cat = int(spec["n_per_category"])
    seed = int(sampling["master_seed"]) + int(spec["seed_offset"])

    items: list[dict] = []
    for cat in sorted(cfg["benchmark"]["categories"]):
        avail = sorted(
            int(q)
            for q in df.loc[df["category"] == cat, "question_id"]
            if int(q) not in used
        )
        take = min(n_per_cat, len(avail))
        rng = stable_rng(seed, cat)
        drawn = sorted(int(q) for q in rng.choice(np.asarray(avail), size=take, replace=False))
        items.extend({"question_id": q, "category": cat} for q in drawn)
    return {"n_per_category": n_per_cat, "items": items}


def profile_document(
    *,
    profile: str,
    cfg: dict,
    cfg_path: Path,
    parquet: Path,
    df: pd.DataFrame,
    domains: dict[str, dict],
) -> dict:
    tiers: dict[str, int] = {}
    for d in domains.values():
        tiers[d["score_tier"]] = tiers.get(d["score_tier"], 0) + 1
    return {
        "schema": "steering.mmlu_pro_domain_profile/v1",
        "profile": profile,
        "role": cfg["sampling"]["profiles"][profile]["role"],
        "map_version": cfg["version"],
        "map_config": cfg_path.name,
        "map_sha256": sha256_file(cfg_path),
        "benchmark": {
            **cfg["benchmark"],
            "n_rows": int(len(df)),
            "parquet_sha256": sha256_file(parquet),
        },
        "sampling": {
            "n_per_soc": cfg["sampling"]["n_per_soc"],
            "master_seed": cfg["sampling"]["master_seed"],
            "disjoint_across_domains": cfg["sampling"]["disjoint_across_domains"],
            "disjoint_across_profiles": cfg["sampling"]["disjoint_across_profiles"],
        },
        "scoring": cfg["scoring"],
        "n_domains": len(domains),
        "n_questions": sum(d["n"] for d in domains.values()),
        "n_domains_by_tier": {k: tiers[k] for k in sorted(tiers, key=lambda t: TIER_ORDER[t])},
        "domains": [domains[k] for k in sorted(domains)],
    }


TIER_CAPTION = {
    "primary": "primary — профильные категории, confidence high/medium",
    "secondary": "secondary — профильные категории, confidence low (прокси натянут)",
    "generic": "generic — прокси нет, срез всего MMLU-Pro",
}


def coverage_markdown(cfg: dict, rows: list[dict], val_doc: dict, test_doc: dict) -> str:
    by_tier: dict[str, int] = {}
    for r in rows:
        by_tier[r["score_tier"]] = by_tier.get(r["score_tier"], 0) + 1
    small_g = [r["soc_major_title"] for r in rows if r["preference_stratum"] == "small"]

    lines = [
        f"# MMLU-Pro coverage по доменам SOC — map `{cfg['version']}`",
        "",
        f"- Бенчмарк: `{cfg['benchmark']['dataset']}`, split=`{cfg['benchmark']['split']}`",
        f"- n на домен: **{cfg['sampling']['n_per_soc']}**, master_seed=`{cfg['sampling']['master_seed']}`",
        f"- Доменов покрыто: **{len(rows)} из {len(rows)}** — ни один не исключён",
        f"- Вопросов: val **{val_doc['n_questions']}**, test **{test_doc['n_questions']}** (не пересекаются)",
        "",
        "Тиры:",
        "",
    ]
    for tier in sorted(by_tier, key=lambda t: TIER_ORDER[t]):
        lines.append(f"- **{by_tier[tier]}** × {TIER_CAPTION[tier]}")
    lines += [
        "",
        "| soc_major_title | G | pref | MMLU-Pro категории | confidence | тир | n_val | n_test | прокси |",
        "| :--- | ---: | :--- | :--- | :--- | :--- | ---: | ---: | :--- |",
    ]
    for r in rows:
        cats = ", ".join(f"`{c}`" for c in r["mmlu_pro_categories"]) or "весь MMLU-Pro"
        pref = "ok" if r["preference_stratum"] == "reliable" else "G<15"
        note = " ".join(str(r["proxy_note"]).split())
        lines.append(
            f"| {r['soc_major_title']} | {r['g_base_items']} | {pref} | {cats} | "
            f"{r['mapping_confidence']} | {r['score_tier']} | {r['n_val']} | {r['n_test']} | {note} |"
        )
    lines += [
        "",
        f"Headline capability-score — macro-average по всем {len(rows)} доменам; "
        "разбивка по тирам показывает, где просадка доменная, а где общая. "
        + " ".join(str(cfg["scoring"]["caveat"]).split()),
        "",
        "Колонка **pref** отмечает домены с G < 15 ("
        + ", ".join(small_g)
        + "): capability для них измеряется полноценно, но связывать её с preference "
        "по такой страте нельзя.",
        "",
    ]
    return "\n".join(lines)


def dump_json(path: Path, doc: dict) -> str:
    text = json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--verify",
        action="store_true",
        help="не перезаписывать: сравнить файлы на диске с пересборкой",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    df = load_benchmark(args.parquet, cfg)
    ver = cfg["version"]

    val_domains, test_domains, coverage_rows = build_domain_profiles(cfg, df)

    common = dict(cfg=cfg, cfg_path=args.config, parquet=args.parquet, df=df)
    val_doc = profile_document(profile="val", domains=val_domains, **common)
    test_doc = profile_document(profile="test", domains=test_domains, **common)

    used = {q for d in val_domains.values() for q in d["question_ids"]}
    used |= {q for d in test_domains.values() for q in d["question_ids"]}
    smoke = build_overall_smoke(cfg, df, used)
    smoke_doc = {
        "schema": "steering.mmlu_pro_overall_smoke/v1",
        "profile": "overall_smoke",
        "role": "Stage B — быстрый общий capability, дизъюнктно с доменными профилями",
        "map_version": ver,
        "benchmark": {**cfg["benchmark"], "n_rows": int(len(df))},
        "n_per_category": smoke["n_per_category"],
        "n_questions": len(smoke["items"]),
        "items": smoke["items"],
    }

    outputs = {
        args.out_dir / f"mmlu_pro_domain_val_{ver}.json": val_doc,
        args.out_dir / f"mmlu_pro_domain_test_{ver}.json": test_doc,
        args.out_dir / f"mmlu_pro_overall_smoke_{ver}.json": smoke_doc,
    }
    coverage_path = args.out_dir / f"coverage_{ver}.md"
    coverage_text = coverage_markdown(cfg, coverage_rows, val_doc, test_doc)

    if args.verify:
        bad = []
        for path, doc in outputs.items():
            expected = json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
            if not path.exists() or path.read_text(encoding="utf-8") != expected:
                bad.append(path.name)
        if not coverage_path.exists() or coverage_path.read_text(encoding="utf-8") != coverage_text:
            bad.append(coverage_path.name)
        if bad:
            print("MISMATCH: " + ", ".join(bad))
            return 1
        print("OK — профили совпадают с пересборкой")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path, doc in outputs.items():
        dump_json(path, doc)
    coverage_path.write_text(coverage_text, encoding="utf-8")

    tiers = ", ".join(f"{k} {v}" for k, v in val_doc["n_domains_by_tier"].items())
    print(f"map {ver}: {val_doc['n_domains']} доменов ({tiers})")
    for path, doc in outputs.items():
        print(f"  {doc['profile']:<14} {doc['n_questions']:>4} вопросов -> {path.name}")
    print(f"  {'coverage':<14} {'':>4}          -> {coverage_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
