"""
Adapter: v2 CSV (с уже подставленными man/woman) → JSONL для inference.py.

Substitution X→man, Y→woman сделана на стороне датасета. Position-bias control'я
через swap НЕ нужен — он заложен иначе: для каждого scenario есть парные
yesno_man + yesno_woman queries, где "Yes" всегда на позиции A. Asymmetry
P(Yes | yesno_man) − P(Yes | yesno_woman) — это bias БЕЗ position-confound'а.

Что делает скрипт:
- Читает v2 CSV (1350 rows)
- Строит полный prompt с "Answer:" в конце:
    scenario_text
    question
    A. <option_A>
    B. <option_B>
    [C. <option_C>]      ← только если abstain_variant=with_abstain
    Answer:
- Сохраняет JSONL: 1350 items, по одному на строку

Формат опций (все 3 question_format используют A/B/(C) labels):
  - choice:      A=man,   B=woman, (C=Cannot determine)
  - yesno_man:   A=Yes,   B=No,    (C=Cannot determine)  ← question: "Was man ...?"
  - yesno_woman: A=Yes,   B=No,    (C=Cannot determine)  ← question: "Was woman ...?"

Запуск:
    python3 src/prepare_factorial.py \\
        data/factorial_v2_with_gender.csv \\
        --out data/factorial_v2_prepared.jsonl

Default output: <input>.prepared.jsonl
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def build_prompt(scenario_text: str, question: str,
                 option_A: str, option_B: str, option_C,
                 abstain_variant: str) -> str:
    lines = [scenario_text, question, f"A. {option_A}", f"B. {option_B}"]
    if abstain_variant == "with_abstain" and pd.notna(option_C):
        lines.append(f"C. {option_C}")
    lines.append("Answer:")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="v2 CSV (man/woman уже подставлены)")
    p.add_argument("--out", default=None,
                   help="Выход JSONL (default: <csv>.prepared.jsonl)")
    args = p.parse_args()

    csv_path = Path(args.csv).resolve()
    out_path = Path(args.out) if args.out else csv_path.with_suffix(".prepared.jsonl")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")

    items = []
    for _, row in df.iterrows():
        has_abstain = (row["abstain_variant"] == "with_abstain"
                       and pd.notna(row["option_C"]))
        # Labels (что подставлено в A/B/(C)) — нужны downstream чтобы интерпретировать выбор
        labels = {"A": row["option_A"], "B": row["option_B"]}
        if has_abstain:
            labels["C"] = row["option_C"]

        prompt = build_prompt(
            row["scenario_text"], row["question"],
            row["option_A"], row["option_B"], row["option_C"],
            row["abstain_variant"],
        )

        item = {
            "id": f"ex{int(row['example_id'])}",
            "example_id": int(row["example_id"]),
            "base_id": int(row["base_id"]),
            "base_context": row["base_context"],   # ambiguous baseline (без evidence)
            "scenario_text": row["scenario_text"], # полный passage (как в CSV)
            "predicate": row["predicate"],
            "evidence_shift": row["evidence_shift"],
            "question_format": row["question_format"],
            "abstain_variant": row["abstain_variant"],
            "question": row["question"],
            "labels": labels,   # {"A":"man","B":"woman",...} или {"A":"Yes","B":"No",...}
            "has_abstain": has_abstain,
            "prompt": prompt,
        }
        items.append(item)

    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(items)} items → {out_path}")
    print(f"  size: {out_path.stat().st_size / 1024:.1f} KB")

    # Демо
    print(f"\n=== пример choice ===")
    print(next(i for i in items if i["question_format"] == "choice")["prompt"])
    print(f"\n=== пример yesno_man ===")
    print(next(i for i in items if i["question_format"] == "yesno_man")["prompt"])
    print(f"\n=== пример yesno_woman ===")
    print(next(i for i in items if i["question_format"] == "yesno_woman")["prompt"])

    # Распределение
    print(f"\n=== распределение ({len(items)} items) ===")
    for fmt in ("choice", "yesno_man", "yesno_woman"):
        n = sum(1 for i in items if i["question_format"] == fmt)
        print(f"  {fmt:12s}: {n}")
    n_abst = sum(1 for i in items if i["has_abstain"])
    print(f"  with_abstain:    {n_abst}")
    print(f"  without_abstain: {len(items) - n_abst}")
    for ev in ("no_evidence", "evidence_supports_man", "evidence_supports_woman"):
        n = sum(1 for i in items if i["evidence_shift"] == ev)
        print(f"  {ev:30s}: {n}")


if __name__ == "__main__":
    main()
