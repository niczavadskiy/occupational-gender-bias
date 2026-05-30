# H11: gender-preference probing

Линейные пробы (Ridge) по hidden states: target **`log_odds = log(p_man / p_woman)`** на choice-промптах.

Обзор пакета: [probes/README.md](../../../../probes/README.md) (от `repo/`).

---

## Быстрый старт

Из **корня репозитория** (`repo/`):

```powershell
cd repo
python -m probes.h11_run --abstain-variant without_abstain --evidence-mode no_evidence
python -m probes.h11_run --abstain-variant with_abstain --evidence-mode no_evidence
```

Один прогон: **layer_scan → control_task → silhouette → extract_direction → compare_directions**.

Полезные флаги: `--evidence-mode`, `--target log_odds`, `--force-split`, `--steps layer_scan` (только часть этапов).  
`python -m probes.h11_run -h`

---

## Сплит

| | |
|--|--|
| **Группа** | `scenario_family_id` = `base_id` = context × predicate |
| **Правило** | Все evidence-варианты одной family — в одном train / val / test |
| **Файл** | `../_shared/group_split_scenario_family.json` |

75 families → 45 / 15 / 15 (train / val / test).

---

## Структура результатов (этот run)

```text
probes/h11/
  README.md                 # этот файл
  without_abstain/          # промпт A/B
    meta.json               # сводка пайплайна (stages, artifacts)
    layer_scan/results.json + *.csv
    control_task/ ...
    silhouette/ ...
    extract_direction/      # *.npz локально
    compare_directions/ ...
  with_abstain/             # промпт A/B/C
    ...
```

Один **`meta.json`** на прогон; обновляется после каждого этапа. В подпапках этапов — только `results.json` и CSV.

---

## Артефакты и git

| Где | Что |
|-----|-----|
| **Git** | `*.json`, `*.md` под `results/*/probes/` |
| **Локально** | CSV (кривые по слоям, control-таблицы) |
| **HF / локально** | `hidden_states.npz`, веса `w` в `*.npz` (глобально в `.gitignore`) |


---

## Evidence и target

| `--evidence-mode` | Строк | / family |
|-------------------|-------|----------|
| `no_evidence` | 75 | 1 |
| `evidence_supports_man` / `woman` | 75 | 1 |
| `all` | 225 | 3 |

По умолчанию target: **`log_odds`**. Также: `prob_margin`, `choice` (на `no_evidence` choice почти всегда man).

---

## Abstain: 2 vs 3 исхода

| `--abstain-variant` | Промпт |
|---------------------|--------|
| `without_abstain` | A=man, B=woman |
| `with_abstain` | A=man, B=woman, C=Cannot determine |

`log_odds = log(p_A/p_B)` при 3-way softmax (масса может уходить в C).  
Для сравнения 2 vs 3 исходов держите **один** `--evidence-mode` (напр. `no_evidence`).

---

## Control tasks

| Режим | Смысл |
|-------|--------|
| **main** | Истинные метки |
| **within_scenario_family** | Shuffle `y` внутри family (осмысленно при `all`) |
| **global** | Shuffle всех `y` |
| **permuted_alignment** | Shuffle `y`, HS фиксированы |

**main ≫ global** → связь HS ↔ target не артеfact shuffle.

---

## Выбор слоя и метрики

- **best_layer** — по **`val_r2`**, не по test (test только для отчёта).
- **R²** — основная метрика регрессии; **Pearson r** — дополнение.
- **compare_directions:** mean cos(w) — грубая сводка; детали — heatmap CSV / `projection_correlation` в results.

---

## Проверка данных: `inspect_h11`

Перед probing — **без загрузки hidden states**:

```powershell
python -m probes.inspect_h11 --evidence-mode all
```

Скрипт:

- создаёт/читает split в `../_shared/group_split_scenario_family.json`;
- выводит число families и строк в train/val/test;
- сохраняет **`manifest_<evidence>.csv`**: id, split, context, predicate, choice, `log_odds`, probs;
- в meta: баланс hard choice (A vs B).

Полезно проверить, что на `no_evidence` все choice=man, но `log_odds` разный.  
Сейчас manifest только для **`without_abstain`** строк (фильтр в `h11.py`).

Отладка одного этапа: `python -m probes.h11_run --steps control_task ...`
