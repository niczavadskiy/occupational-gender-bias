# Probing: Qwen3.5-2B-Base (v3 full, pos shuffle)

Линейные пробы (Ridge) по hidden states inference-run.  
Модель: **Qwen/Qwen3.5-2B-Base**. Run: `2026-06-05_10-38-02` · tag `v3_full_pos_shuffle` (merge `man_first` + `woman_first`, 21600 items).

Методология и запуск скриптов: [probes/README.md](../../../probes/README.md) (от `repo/`).

Behavioral metrics: [metrics/2026-06-05_23-02-12/summary.md](../metrics/2026-06-05_23-02-12/summary.md).

**Probing re-run 2026-06-06:** targets H10–H12 переведены на **position-aware semantic lookup** (`probes/row_probs.py`) — `p_man`/`p_woman`/`p_abstain` берутся по `row.labels`, а не слепо из слотов A/B/C. Ниже — метрики после этого исправления (pipeline_run_id в `meta.json`).

---

## Общие условия (основной срез probing)

| Параметр | Значение |
|----------|----------|
| Layout | v3 (`task=main` / `answerability`) |
| Evidence | `all` (no_evidence + ev_man + ev_woman) |
| Форматы | `choice`, `yesno_man`, `yesno_woman` |
| Split | 10800 probe groups → train 6480 / val 2160 / test 2160 (seed 0) |
| Группа | `example_id × context_order × position_variant` |
| Слой | best по **val R²** |

Пайплайн на прогон: layer_scan → control_task → silhouette → extract_direction → compare_directions.

**H10** в этом run **не запускался** — ждём новую аннотацию (`annotation.option_labels`).

---

## Targets (логиты probing)

| ID | Target | Формула | Строки | Источник p |
|----|--------|---------|--------|------------|
| H11 | `log_odds` | `log(p_man / p_woman)` | `task=main` | semantic gender axis через `row.labels` |
| H12 | `abstain_logit` | `log(p_abstain / (1 − p_abstain))` | `task=main`, `with_abstain` | слот с `labels[key] == "Cannot determine"` |
| H13 | `h13_yes_logit` | `log(p_Yes / p_No)` | `task=answerability` | `prob_constrained_Yes/No` на self-Q |

**H11 — gender preference.** `p_man` / `p_woman` — сумма constrained prob по слотам промпта с соответствующим semantic text (position-aware: на p1 man/woman/abstain могут быть в A/B/C в разном порядке). Hard label для silhouette / `choice`:

| `question_format` | prefers man (1) | prefers woman (0) |
|-------------------|-----------------|-------------------|
| `choice` | choice → `man` | choice → `woman` |
| `yesno_man` | `Yes` | `No` |
| `yesno_woman` | `No` | `Yes` |

При `with_abstain` choice → `Cannot determine` → abstain (−1).

**H12 — abstain.** `p_abstain` — mass на «Cannot determine» в logit-форме; **не** фиксированный слот C (на p1 abstain часто в B).

**H13 — self-answerability.** Companion-prompt: модель отвечает Yes/No на «можно ли ответить при показанных опциях?». Target — **модельный** P(Yes). Прогон **2+3**: pool строк с 2 и 3 main-опциями (slug `2+3`, n=10800). Target не менялся при fix 2026-06-06.

**H10 (отложено):** `h10_st_logit = log(p_st / (1 − p_st))`, где `p_st` — сумма `prob_constrained_*` на stereotype-опциях по **annotation** (semantic join через `annotation.labels`).

---

## Сводка выводов (re-run 2026-06-06)

| ID | Ось в representations | Декодируемость (val R²) | Best layer | Контроли (main vs global @ best) |
|----|-------------------------|-------------------------|------------|----------------------------------|
| **H11** `without_abstain` | Gender `log_odds` (n=2700) | **Сильная** (~0,91) | L20 | Валидный (−1,08) |
| **H11** `with_abstain` | Gender `log_odds` (n=8100) | **Сильная** (~0,88) | L22 | Валидный (−0,46) |
| **H12** | Abstain `abstain_logit` (n=8100) | **Очень сильная** (~0,97) | L16 | Валидный (−0,44) |
| **H13** | Self-Q `h13_yes_logit` (2+3, n=10800) | **Очень сильная** (~0,99) | L24 | Валидный (−0,36) |

Кластеризация (silhouette, `median_train`) **низкая** (0,01–0,10): линейная ось декодируется с высоким R², но чёткого разделения кластеров в HS нет.

**Эффект semantic fix:** R² по H11/H12 **ниже**, чем в первом (ошибочном) прогоне со слотами A/B/C (~0,98–0,99) — inflated targets на `position_variant=p1` давали завышенную декодируемость. После fix сигнал остаётся **сильным и валидным** (main ≫ global shuffle).

Best layer сместился: H11 → L20–22, H12 → **L16** (раньше ошибочно L24).

---

## H11 — предпочтения по gender axis

**Гипотеза:** предпочтение man vs woman **линейно декодируется** из hidden states.

**Target:** `log_odds = log(p_man / p_woman)` (choice + yesno_man + yesno_woman); hard label по semantic `labels`.

Pipeline: `2026-06-06_12-52-13` (without) · `2026-06-06_12-52-10` (with).

### `without_abstain` — 2 опции (n=2700)

Каталог: [h11/without_abstain_all_choice_yesno_man_yesno_woman_main/](h11/without_abstain_all_choice_yesno_man_yesno_woman_main/)

| | |
|--|--|
| **Best layer** | **L20** |
| **val R² (main)** | **0,912** |
| **test R²** | **0,901** |
| **global shuffle @ best** | **−1,083** |
| **val silhouette (max)** | **~0,007** (L3) |

### `with_abstain` — 3 опции (n=8100)

Каталог: [h11/with_abstain_all_choice_yesno_man_yesno_woman_main/](h11/with_abstain_all_choice_yesno_man_yesno_woman_main/)

| | |
|--|--|
| **Best layer** | **L22** |
| **val R² (main)** | **0,882** |
| **test R²** | **0,891** |
| **global shuffle @ best** | **−0,463** |
| **val silhouette (max)** | **~0,011** (L21) |

### Выводы

1. Gender-preference **сильно декодируется** на полном v3 срезе; val≈test — без выраженного overfit.
2. **Контроли Hewitt–Liang:** main **≫** global shuffle → сигнал **валидный**.
3. При `with_abstain` R² ниже, чем без abstain: часть variance уходит в abstain-ось (H12); global shuffle менее отрицателен.
4. Silhouette очень низкий — как на factorial v2.

**Итог:** H11 **подтверждается** на v3 с корректными targets; декодируемость **выше**, чем на n=75 factorial v2, но **ниже** первого ошибочного v3-прогона.

---

## H12 — предпочтение «Воздержаться»

**Гипотеза:** mass на «Cannot determine» (`abstain_logit`) **линейно декодируется** из hidden states.

**Target:** `abstain_logit = log(p_abstain / (1 − p_abstain))`, только `with_abstain`; `p_abstain` — semantic, position-aware.

Pipeline (3 формата): `2026-06-06_13-02-07`.

### Основной прогон — 3 формата (n=8100)

Каталог: [h12/with_abstain_all_choice_yesno_man_yesno_woman_main/](h12/with_abstain_all_choice_yesno_man_yesno_woman_main/)

| | |
|--|--|
| **Best layer** | **L16** |
| **val R² (main)** | **0,969** |
| **test R²** | **0,968** |
| **global shuffle @ best** | **−0,442** |
| **val silhouette (max)** | **~0,026** (L20) |

### Доп. прогон — только `choice` (n=2700) — **устарел**

Каталог: [h12/with_abstain_all_choice_main/](h12/with_abstain_all_choice_main/) · pipeline `2026-06-06_11-21-57` (**до** semantic fix; val R² **0,992**, L24). Не перезапускался; для сравнения использовать основной 3-format прогон.

### Выводы

1. Abstain-preference **очень сильно декодируется**; после fix best layer **L16** (не L24).
2. **Контроли:** main **(0,969)** ≫ global **(−0,44)** → **валидный** сигнал.
3. Silhouette низкий; best silhouette (L20) не совпадает с best R² (L16).

**Итог:** H12 **подтверждается**; на v3 abstain-ось декодируется **не слабее** gender-preference (H11).

---

## H13 — self-answerability

**Гипотеза:** модельный self-assessment «можно ли ответить?» (`P(Yes)` на self-Q) **линейно декодируется** из hidden states.

**Target:** `h13_yes_logit = log(p_Yes / p_No)` на `task=answerability`.

Pipeline: `2026-06-06_14-00-43`.

### Прогон 2+3 — pool without + with abstain (n=10800)

Каталог: [h13/2+3_all_choice_yesno_man_yesno_woman_answerability/](h13/2+3_all_choice_yesno_man_yesno_woman_answerability/)

| | |
|--|--|
| **Best layer** | **L24** |
| **val R² (main)** | **0,992** |
| **test R²** | **0,992** |
| **global shuffle @ best** | **−0,361** |
| **val silhouette (max)** | **~0,098** (L24) |

### Выводы

1. Self-Q Yes/No **сильно декодируется** из activations на том же промпте (HS snapshot перед ответом на answerability).
2. **Контроли:** main **≫** global → **валидный** сигнал.
3. Silhouette **чуть выше**, чем у H11/H12 (~0,10), но всё ещё умеренный.
4. Метрики **стабильны** относительно первого прогона — target H13 не зависел от gender slot mapping.

**Итог:** H13 **подтверждается**.

---

## Сравнение осей (re-run 2026-06-06, `all` evidence, 3 формата)

| Гипотеза | Target | n | Best layer | val R² | test R² | global @ best | pipeline_run_id |
|----------|--------|---|------------|--------|---------|---------------|-----------------|
| H11 `without_abstain` | `log_odds` | 2700 | L20 | 0,912 | 0,901 | −1,083 | `2026-06-06_12-52-13` |
| H11 `with_abstain` | `log_odds` | 8100 | L22 | 0,882 | 0,891 | −0,463 | `2026-06-06_12-52-10` |
| **H12** | **`abstain_logit`** | 8100 | L16 | **0,969** | **0,968** | **−0,442** | `2026-06-06_13-02-07` |
| **H13** (2+3) | **`h13_yes_logit`** | 10800 | L24 | **0,992** | **0,992** | **−0,361** | `2026-06-06_14-00-43` |

---

## Структура каталогов

```text
probes/
  README.md              # этот файл — выводы по run
  _shared/
    group_split_h10_v3_main.json
    group_split_h13_v3_answerability.json
  h11/
    without_abstain_all_choice_yesno_man_yesno_woman_main/   # re-run 2026-06-06
    with_abstain_all_choice_yesno_man_yesno_woman_main/     # re-run 2026-06-06
  h12/
    with_abstain_all_choice_yesno_man_yesno_woman_main/     # re-run 2026-06-06 (основной)
    with_abstain_all_choice_main/                           # pre-fix, choice-only
  h13/
    2+3_all_choice_yesno_man_yesno_woman_answerability/     # 2026-06-06
```

В каждом прогоне: `meta.json` (stages, summary) и подпапки этапов с `results.json` (+ CSV/NPZ локально, `.gitignore`).

---

## Связь с behavioral

| Probing | Behavioral аналог |
|---------|-------------------|
| H11 `log_odds` | H1/H3 — gender preference по output |
| H12 `abstain_logit` | H3/H8 — mass на abstain |
| H13 `h13_yes_logit` | H4/H7 — P(Yes) на self-Q answerability |

---

## Примечание: первый vs исправленный прогон

| | Первый прогон (slot A/B/C) | Re-run 2026-06-06 (semantic) |
|--|---------------------------|------------------------------|
| H11 without val R² | 0,984 @ L24 | **0,912 @ L20** |
| H11 with val R² | 0,983 @ L24 | **0,882 @ L22** |
| H12 3-fmt val R² | 0,992 @ L24 | **0,969 @ L16** |
| H13 val R² | 0,992 @ L24 | 0,992 @ L24 (без изменений) |

Причина расхождения: на `position_variant=p1` слоты A/B/C **не совпадают** с man/woman/abstain; старый код смешивал оси (напр. `prob_C` = woman вместо abstain).
