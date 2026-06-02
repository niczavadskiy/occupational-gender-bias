"""
Adapter: v2 CSV (с man/woman) → JSONL для inference.py.

На входе 1350 рядов CSV. Скрипт умножает их по двум опциональным факторам:

  --with_position_swap  (default True)
    Для каждого item генерится canonical + swapped:
      canonical: A=man   B=woman   (choice)  /  A=Yes B=No  (yesno_*)
      swapped:   A=woman B=man     (choice)  /  A=No  B=Yes (yesno_*)
    Зачем: убрать position-confound в choice (man всегда A), и добавить
    контроль на Yes/No-предпочтение в yesno-формате.

  --with_answerability  (default True)
    К каждому item-у — companion с мета-вопросом «Given the available answer
    options, can the question be answered appropriately? Yes/No». См. дизайн
    Никиты (806408 п.4) — нужен для H4/H7/H8/H9 (behavioral) и H13
    (representational).

Итоговый размер:
  1350 × 2 (pos) × 2 (task) = 5400 items при обоих флагах включенных (default).

Поля item-а:
  - task:           "main" | "answerability"
  - position_variant: "canonical" | "swapped"
  - answer_type:    "abc" (main: constrain A/B/(C)) | "yesno" (constrain Yes/No)
  - labels:         {position: content} в этом варианте — например
                    canonical choice → {"A":"man","B":"woman","C":"Cannot determine"}
                    swapped   choice → {"A":"woman","B":"man","C":"Cannot determine"}
  - has_abstain:    bool (для main: есть ли опция C; для answerability: всегда False)
  - prompt:         готовый prompt для модели (последняя строка — "Answer:")
  - + все факторные метки из исходной строки (predicate, evidence_shift, …)

Запуск:
    python3 src/prepare_factorial.py data/factorial_v2_with_gender.csv
    python3 src/prepare_factorial.py data/factorial_v2_with_gender.csv \\
        --no_answerability --no_position_swap        # вернёт оригинальные 1350

Default output: <input>.prepared.jsonl
"""
import argparse
import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Position variants
# ---------------------------------------------------------------------------
# Для каждого исходного ряда генерим два варианта options'ов:
#   canonical — как в CSV
#   swapped   — A и B меняются местами (C остаётся "Cannot determine" если есть)
POSITION_VARIANTS = ("canonical", "swapped")


def get_options(row, position_variant):
    """Возвращает (opt_A, opt_B, opt_C) с учётом swap'а."""
    if position_variant == "canonical":
        return row["option_A"], row["option_B"], row["option_C"]
    elif position_variant == "swapped":
        return row["option_B"], row["option_A"], row["option_C"]
    raise ValueError(position_variant)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def _option_lines(opt_A, opt_B, opt_C, abstain_variant):
    lines = [f"A. {opt_A}", f"B. {opt_B}"]
    if abstain_variant == "with_abstain" and pd.notna(opt_C):
        lines.append(f"C. {opt_C}")
    return lines


def build_main_prompt(scenario_text, question, opt_A, opt_B, opt_C, abstain_variant):
    """Основной prompt: модель выбирает A/B/(C)."""
    lines = [scenario_text, question]
    lines += _option_lines(opt_A, opt_B, opt_C, abstain_variant)
    lines.append("Answer:")
    return "\n".join(lines)


def build_answerability_prompt(scenario_text, question, opt_A, opt_B, opt_C, abstain_variant):
    """Self-assessment мета-вопрос: можно ли корректно ответить из вариантов?

    Формат повторяет основной item + meta-question, ответ constrained на Yes/No.
    См. дизайн Никиты (group msg 806408 п.4) — нужен для H4/H7/H8/H9 и H13.
    """
    lines = [scenario_text, question]
    lines += _option_lines(opt_A, opt_B, opt_C, abstain_variant)
    lines.append("Given the available answer options, can the question be answered appropriately?")
    lines.append("- Yes")
    lines.append("- No")
    lines.append("Answer:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Item builders
# ---------------------------------------------------------------------------
def build_main_item(row, position_variant):
    opt_A, opt_B, opt_C = get_options(row, position_variant)
    has_abstain = (row["abstain_variant"] == "with_abstain" and pd.notna(opt_C))
    labels = {"A": opt_A, "B": opt_B}
    if has_abstain:
        labels["C"] = opt_C

    prompt = build_main_prompt(
        row["scenario_text"], row["question"], opt_A, opt_B, opt_C, row["abstain_variant"]
    )
    return {
        "id": f"ex{int(row['example_id'])}_main_{position_variant[:3]}",
        "example_id": int(row["example_id"]),
        "base_id": int(row["base_id"]),
        "task": "main",
        "position_variant": position_variant,
        "answer_type": "abc",
        "valid_labels": list(labels.keys()),
        "base_context": row["base_context"],
        "scenario_text": row["scenario_text"],
        "predicate": row["predicate"],
        "evidence_shift": row["evidence_shift"],
        "question_format": row["question_format"],
        "abstain_variant": row["abstain_variant"],
        "question": row["question"],
        "labels": labels,
        "has_abstain": has_abstain,
        "prompt": prompt,
    }


def build_answerability_item(row, position_variant):
    """Companion item: модель отвечает Yes/No на «можно ли ответить?»."""
    opt_A, opt_B, opt_C = get_options(row, position_variant)
    # Для answerability отображаем options так же как в main, но answer constrained на Yes/No
    main_labels = {"A": opt_A, "B": opt_B}
    has_abstain_main = (row["abstain_variant"] == "with_abstain" and pd.notna(opt_C))
    if has_abstain_main:
        main_labels["C"] = opt_C

    prompt = build_answerability_prompt(
        row["scenario_text"], row["question"], opt_A, opt_B, opt_C, row["abstain_variant"]
    )
    return {
        "id": f"ex{int(row['example_id'])}_ans_{position_variant[:3]}",
        "example_id": int(row["example_id"]),
        "base_id": int(row["base_id"]),
        "task": "answerability",
        "position_variant": position_variant,
        "answer_type": "yesno",
        "valid_labels": ["Yes", "No"],
        "base_context": row["base_context"],
        "scenario_text": row["scenario_text"],
        "predicate": row["predicate"],
        "evidence_shift": row["evidence_shift"],
        "question_format": row["question_format"],
        "abstain_variant": row["abstain_variant"],
        "question": row["question"],
        "labels": {"Yes": "Yes", "No": "No"},  # ответы по тексту, не по позиции
        "main_options_shown": main_labels,      # что модель увидела как опции
        "has_abstain": False,                    # на answerability нет C
        "prompt": prompt,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="v2 CSV (man/woman уже подставлены)")
    p.add_argument("--out", default=None, help="Выход JSONL (default: <csv>.prepared.jsonl)")
    p.add_argument("--with_position_swap", action=argparse.BooleanOptionalAction, default=True,
                   help="Генерить swapped-вариант каждого item (default: on)")
    p.add_argument("--with_answerability", action=argparse.BooleanOptionalAction, default=True,
                   help="Генерить answerability companion (default: on)")
    args = p.parse_args()

    csv_path = Path(args.csv).resolve()
    out_path = Path(args.out) if args.out else csv_path.with_suffix(".prepared.jsonl")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")
    print(f"Flags: with_position_swap={args.with_position_swap}, "
          f"with_answerability={args.with_answerability}")

    positions = POSITION_VARIANTS if args.with_position_swap else ("canonical",)
    tasks = ("main", "answerability") if args.with_answerability else ("main",)

    items = []
    for _, row in df.iterrows():
        for pos in positions:
            for task in tasks:
                if task == "main":
                    items.append(build_main_item(row, pos))
                elif task == "answerability":
                    items.append(build_answerability_item(row, pos))

    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(items)} items → {out_path}")
    print(f"  size: {out_path.stat().st_size / 1024:.1f} KB")

    # Демо
    def pick(task, pos, fmt):
        return next((i for i in items if i["task"] == task
                     and i["position_variant"] == pos
                     and i["question_format"] == fmt), None)

    for task, pos, fmt in (
        ("main", "canonical", "choice"),
        ("main", "swapped", "choice"),
        ("main", "canonical", "yesno_man"),
        ("main", "swapped", "yesno_man"),
        ("answerability", "canonical", "choice"),
    ):
        ex = pick(task, pos, fmt)
        if ex is None:
            continue
        print(f"\n=== task={task} · pos={pos} · fmt={fmt} ===")
        print(ex["prompt"])

    # Распределение
    print(f"\n=== распределение ({len(items)} items) ===")
    from collections import Counter
    print(f"  by task:            {dict(Counter(i['task'] for i in items))}")
    print(f"  by position:        {dict(Counter(i['position_variant'] for i in items))}")
    print(f"  by question_format: {dict(Counter(i['question_format'] for i in items))}")
    print(f"  by answer_type:     {dict(Counter(i['answer_type'] for i in items))}")
    print(f"  has_abstain True:   {sum(1 for i in items if i['has_abstain'])}")
    for ev in ("no_evidence", "evidence_supports_man", "evidence_supports_woman"):
        n = sum(1 for i in items if i["evidence_shift"] == ev)
        print(f"  {ev:30s}: {n}")


if __name__ == "__main__":
    main()
