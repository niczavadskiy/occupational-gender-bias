# probes/

Линейный probing по hidden states inference-run (`results/<run>/`).  
Пакет рассчитан на гипотезы **H10–H13**; сейчас реализованы **H10–H12**.

Запуск из **корня репозитория** (`repo/`):

```powershell
cd repo
python -m probes.<модуль> ...
```

---

## Гипотезы (статус)

| ID | Тема | Статус | Документация |
|----|------|--------|--------------|
| **H10** | Stereotype-consistency signal (`h10_st_logit`) decodable from HS | **done** | Выводы: `results/<run>/probes/README.md` · детали: `.../h10/README.md` |
| **H11** | Gender-preference (`log_odds`) decodable from HS | **done** | Выводы: `results/<run>/probes/README.md` · детали: `.../h11/README.md` |
| **H12** | Abstain preference (`abstain_logit`) decodable from HS | **done** | Выводы: `results/<run>/probes/README.md` · детали: `.../h12/README.md` |
| **H13** | — | planned | — |

---

## Быстрый старт

Каждый прогон: **layer_scan → control_task → silhouette → extract_direction → compare_directions**.

Общие флаги (все `*_run`): `--run`, `--seed`, `--force-split`, `--steps layer_scan` (подмножество этапов), `--ridge-alpha`.  
Справка: `python -m probes.h10_run -h`, `h11_run -h`, `h12_run -h`.

### H10 — stereotype signal

Join inference ↔ аннотация **только по `example_id`**.  
Аннотация по умолчанию: `data/annotation/runs/2026-06-02_13-10-31_anthropic_claude-opus-4.8/annotations.jsonl` (`--annotation-run`).

```powershell
# полный факториал: all evidence × все форматы (по умолчанию)
python -m probes.h10_run --abstain-variant without_abstain --evidence-mode all
python -m probes.h10_run --abstain-variant with_abstain --evidence-mode all

# чистый срез (как H11): no_evidence + choice
python -m probes.h10_run --abstain-variant without_abstain --evidence-mode no_evidence --question-formats choice
python -m probes.h10_run --abstain-variant with_abstain --evidence-mode no_evidence --question-formats choice
```

Доп. флаги: `--question-formats`, `--target h10_st_logit`, `--annotation-run`, `--evidence-modes`.

Каталог вывода: `results/<run>/probes/h10/{abstain}_{evidence}_{formats...}/`  
(напр. `without_abstain_no_evidence_choice`, `with_abstain_all_choice_yesno_man_yesno_woman`).

### H11 — gender preference

```powershell
python -m probes.h11_run --abstain-variant without_abstain --evidence-mode no_evidence
python -m probes.h11_run --abstain-variant with_abstain --evidence-mode no_evidence
```

Доп. флаги: `--evidence-mode`, `--target log_odds`.

Каталог вывода: `results/<run>/probes/h11/{without_abstain|with_abstain}/`.

### H12 — abstain preference

Только `with_abstain` + `choice`; флага `--abstain-variant` нет.

```powershell
python -m probes.h12_run --evidence-mode no_evidence
python -m probes.h12_run --evidence-mode all
```

Доп. флаги: `--evidence-mode`, `--target abstain_logit`.

Каталог вывода: `results/<run>/probes/h12/{abstain}_{evidence}_{format}/`  
(напр. `with_abstain_no_evidence_choice`; abstain и format фиксированы, меняется `--evidence-mode`).

### Preflight (без hidden states)

```powershell
python -m probes.inspect_h11
```

Manifest, split по family → `_shared/group_split_scenario_family.json`, баланс choice.

---

## Сплит

| | |
|--|--|
| **Группа** | `scenario_family_id` = `base_id` = context × predicate |
| **Правило** | Все строки одной family (evidence × format) — в одном train / val / test |
| **Файл** | `results/<run>/probes/_shared/group_split_scenario_family.json` |

75 families → 45 / 15 / 15 (train / val / test), seed=0.

| Срез | n строк |
|------|---------|
| H10/H11/H12: `no_evidence` + `choice` | 75 (1 на family) |
| H12: `all` | 225 (3 evidence на family) |
| H10: `all` + 3 формата | 675 (9 на family) |

---

## Targets

### H10

| Target | Формула / смысл |
|--------|-----------------|
| **`h10_st_logit`** (default) | `log(p_st / (1 − p_st))`, масса на stereotype-опциях |
| `h10_log_prob_st` | `log(p_st)` |
| `h10_choice_stereotype` | бинарный: выбор = `stereotype_consistent` |

`p_st` — сумма `prob_constrained_*` по опциям с меткой `stereotype_consistent` из **annotation.option_labels**.

### H11

| Target | Формула |
|--------|---------|
| **`log_odds`** (default) | `log(p_A / p_B)` |
| `log_prob_man` / `log_prob_woman` | log-вероятности исходов |
| `choice` | бинарный A vs B |

### H12

| Target | Формула |
|--------|---------|
| **`abstain_logit`** (default) | `log(p_C / (1 − p_C))` |
| `log_prob_abstain` | `log(p_C)` |
| `choice_abstain` | бинарный: C vs A/B |

---

## Evidence, форматы, abstain

### `--evidence-mode`

| Значение | Уровни `evidence_shift` |
|----------|-------------------------|
| `no_evidence` | только `no_evidence` |
| `evidence_supports_man` | man-support |
| `evidence_supports_woman` | woman-support |
| `all` | все три |

### `--question-formats` (только H10)

| Формат | Описание |
|--------|----------|
| `choice` | A/B или A/B/C |
| `yesno_man` | yes/no про man |
| `yesno_woman` | yes/no про woman |

По умолчанию — все три; порядок в `run_slug`: `choice_yesno_man_yesno_woman`.

### `--abstain-variant` (H10, H11)

| Значение | Промпт |
|----------|--------|
| `without_abstain` | A=man, B=woman |
| `with_abstain` | A=man, B=woman, C=Cannot determine |

H12 всегда `with_abstain`. При 3-way `log_odds` (H11) масса может уходить в C.

---

## Control tasks

| Режим | Смысл |
|-------|--------|
| **main** | Истинные метки |
| **within_scenario_family** | Shuffle `y` внутри family (осмысленно при `rows_per_family > 1`) |
| **global** | Shuffle всех `y` |
| **permuted_alignment** | Shuffle `y`, HS фиксированы |

**main ≫ global** → связь HS ↔ target не артеfact shuffle.

---

## Выбор слоя и метрики

- **best_layer** — по **`val_r2`** (регрессия) или **`val_balanced_accuracy`** (классификация), не по test.
- **R²** — основная метрика регрессии; **Pearson r** — дополнение.
- **silhouette** — бинаризация target (`median_train`); диагностика, не главная метрика.
- **extract_direction:** projection ↔ target (r на best layer).
- **compare_directions:** mean cos(w); детали — heatmap CSV в results.

---

## Модули (библиотека)

| Файл | Роль |
|------|------|
| `h10_run.py` | Оркестратор пайплайна H10 |
| `h11_run.py` | Оркестратор пайплайна H11 |
| `h12_run.py` | Оркестратор пайплайна H12 |
| `h10.py` | Batch + target H10 из аннотации (`example_id` join) |
| `h11.py` | Batch, evidence modes, `log_odds`, фильтр abstain |
| `h12.py` | Batch, evidence modes, `abstain_logit` / `log_prob_abstain` |
| `inspect_h11.py` | Manifest + split check |
| `core.py` | Ridge / logistic, метрики, загрузка bundle |
| `splits.py` | Group split по `scenario_family_id` |
| `load_run.py`, `paths.py` | Run dir, `per_item.jsonl`, hidden states |
| `run_io.py` | `meta.json`, запись results, pipeline-meta |
| `layer_scan.py` | R², Pearson r по слоям |
| `control_task.py` | Hewitt–Liang controls |
| `silhouette.py` | Silhouette vs бинарный target |
| `extract_direction.py` | Вектор `w`, проекции |
| `compare_directions.py` | cos(w) между слоями |
| `__init__.py` | Package init |

---

## Данные и артефакты

```text
results/<run>/probes/
  _shared/, inspect_h11/              # локально (.gitignore)
  h10/
    README.md                         # результаты run
    <run_slug>/meta.json, */results.json
  h11/
    README.md
    without_abstain/ | with_abstain/
  h12/
    README.md
    with_abstain_no_evidence_choice/   # <abstain>_<evidence>_choice
      meta.json, */results.json
```

| Где | Что |
|-----|-----|
| **Git** | `*.json`, `*.md` под `results/*/probes/` |
| **Локально** | CSV, `*.npz` (hidden states, веса `w`) |

Один **`meta.json`** на прогон пайплайна; обновляется после каждого этапа.
