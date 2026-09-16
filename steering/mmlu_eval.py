"""
MMLU-Pro capability scoring под тем же last-token хуком, что Stage A/B preference.

Промпт заканчивается на `Answer:` — интервенция в residual stream на последнем
токене промпта совпадает с preference-протоколом.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from steering.intervene import Scorer, steered

LETTERS = "ABCDEFGHIJ"


def format_mmlu_prompt(question: str, options: list[str]) -> str:
    lines = [question.strip(), ""]
    for i, opt in enumerate(options):
        lines.append(f"{LETTERS[i]}. {opt}")
    lines.append("Answer:")
    return "\n".join(lines)


def load_parquet_by_ids(parquet: Path, question_ids: set[int]) -> dict[int, dict]:
    df = pd.read_parquet(parquet)
    need = df[df["question_id"].isin(question_ids)]
    out: dict[int, dict] = {}
    for row in need.itertuples(index=False):
        opts = list(row.options)
        # parquet иногда хранит пустые хвостовые опции — обрезаем
        while opts and (opts[-1] is None or str(opts[-1]).strip() == ""):
            opts.pop()
        qid = int(row.question_id)
        out[qid] = {
            "question_id": qid,
            "question": str(row.question),
            "options": [str(o) for o in opts],
            "answer": str(row.answer).strip(),
            "answer_index": int(row.answer_index),
            "category": str(row.category),
        }
    missing = question_ids - set(out)
    if missing:
        raise KeyError(f"в parquet нет question_id: {sorted(missing)[:10]}")
    return out


def expand_domain_profile(profile: dict, bank: dict[int, dict]) -> list[dict]:
    items = []
    for dom in profile["domains"]:
        for qid in dom["question_ids"]:
            q = bank[int(qid)]
            labels = [LETTERS[i] for i in range(len(q["options"]))]
            items.append(
                {
                    "question_id": q["question_id"],
                    "category": q["category"],
                    "soc_major_title": dom["soc_major_title"],
                    "score_tier": dom["score_tier"],
                    "preference_stratum": dom.get("preference_stratum"),
                    "mapping_confidence": dom.get("mapping_confidence"),
                    "answer": q["answer"],
                    "prompt": format_mmlu_prompt(q["question"], q["options"]),
                    "valid_labels": labels,
                }
            )
    return items


def expand_overall_smoke(profile: dict, bank: dict[int, dict]) -> list[dict]:
    items = []
    for entry in profile["items"]:
        qid = int(entry["question_id"])
        q = bank[qid]
        labels = [LETTERS[i] for i in range(len(q["options"]))]
        items.append(
            {
                "question_id": q["question_id"],
                "category": entry.get("category") or q["category"],
                "soc_major_title": None,
                "score_tier": "overall_smoke",
                "answer": q["answer"],
                "prompt": format_mmlu_prompt(q["question"], q["options"]),
                "valid_labels": labels,
            }
        )
    return items


def run_mmlu(
    scorer: Scorer,
    model,
    items: list[dict],
    specs: list,
    *,
    label: str,
    log_every: int = 100,
) -> list[dict]:
    rows: list[dict] = []
    t0 = time.time()
    total = len(items)
    for item in items:
        with steered(model, specs):
            out = scorer.score(item["prompt"], list(item["valid_labels"]))
        correct = out["choice"] == item["answer"]
        rows.append(
            {
                "question_id": item["question_id"],
                "category": item["category"],
                "soc_major_title": item.get("soc_major_title"),
                "score_tier": item.get("score_tier"),
                "answer": item["answer"],
                "choice": out["choice"],
                "correct": bool(correct),
                "prob_constrained_choice": float(
                    out.get(f"prob_constrained_{out['choice']}", float("nan"))
                ),
                "n_prompt_tokens": out["n_prompt_tokens"],
            }
        )
        if log_every and len(rows) % log_every == 0:
            rate = len(rows) / (time.time() - t0)
            eta = (total - len(rows)) / max(rate, 1e-9) / 60
            print(f"    [{label}] mmlu {len(rows)}/{total}  {rate:.2f} q/s  ETA {eta:.1f} min", flush=True)
    return rows


def summarize_domain_capability(rows: list[dict], profile: dict) -> dict:
    by_soc: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_soc[str(r["soc_major_title"])].append(r)

    tier_of = {d["soc_major_title"]: d["score_tier"] for d in profile["domains"]}
    per_domain = []
    for soc in sorted(by_soc):
        rs = by_soc[soc]
        acc = float(np.mean([r["correct"] for r in rs]))
        per_domain.append(
            {
                "soc_major_title": soc,
                "score_tier": tier_of.get(soc, ""),
                "n": len(rs),
                "accuracy": acc,
            }
        )

    def tier_mean(tier: str | None) -> float | None:
        if tier is None:
            vals = [d["accuracy"] for d in per_domain]
        else:
            vals = [d["accuracy"] for d in per_domain if d["score_tier"] == tier]
        return float(np.mean(vals)) if vals else None

    return {
        "n_questions": len(rows),
        "n_domains": len(per_domain),
        "accuracy_macro": tier_mean(None),
        "accuracy_micro": float(np.mean([r["correct"] for r in rows])) if rows else None,
        "by_tier": {
            t: tier_mean(t) for t in ("primary", "secondary", "generic") if tier_mean(t) is not None
        },
        "per_domain": per_domain,
    }


def summarize_overall_smoke(rows: list[dict]) -> dict:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[str(r["category"])].append(r)
    per_category = [
        {
            "category": cat,
            "n": len(rs),
            "accuracy": float(np.mean([r["correct"] for r in rs])),
        }
        for cat, rs in sorted(by_cat.items())
    ]
    return {
        "n_questions": len(rows),
        "accuracy_micro": float(np.mean([r["correct"] for r in rows])) if rows else None,
        "accuracy_macro_category": float(np.mean([c["accuracy"] for c in per_category]))
        if per_category
        else None,
        "per_category": per_category,
    }


def cap_loss_vs_baseline(steered_summary: dict, baseline_summary: dict) -> dict:
    """cap_loss = mean_d max(0, acc_base(d) − acc_steered(d))."""
    base = {d["soc_major_title"]: d for d in baseline_summary["per_domain"]}
    per = []
    drops = []
    for d in steered_summary["per_domain"]:
        soc = d["soc_major_title"]
        b = base[soc]["accuracy"]
        drop = max(0.0, b - d["accuracy"])
        drops.append(drop)
        per.append(
            {
                "soc_major_title": soc,
                "score_tier": d["score_tier"],
                "acc_baseline": b,
                "acc_steered": d["accuracy"],
                "drop": drop,
            }
        )

    def tier_loss(tier: str) -> float | None:
        vals = [p["drop"] for p in per if p["score_tier"] == tier]
        return float(np.mean(vals)) if vals else None

    return {
        "cap_loss": float(np.mean(drops)) if drops else None,
        "by_tier": {t: tier_loss(t) for t in ("primary", "secondary", "generic")},
        "per_domain": per,
        "n_domains_worse": int(sum(1 for p in per if p["drop"] > 0)),
    }


def collect_profile_ids(*profiles: dict) -> set[int]:
    ids: set[int] = set()
    for p in profiles:
        if "domains" in p:
            for d in p["domains"]:
                ids.update(int(x) for x in d["question_ids"])
        if "items" in p:
            ids.update(int(x["question_id"]) for x in p["items"])
    return ids


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(
        __import__("json").dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
