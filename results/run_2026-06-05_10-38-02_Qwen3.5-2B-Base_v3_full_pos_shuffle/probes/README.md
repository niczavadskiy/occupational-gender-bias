# Probing: Qwen3.5-2B-Base (v3 full, pos shuffle)

Линейные пробы (Ridge) по hidden states inference-run.  
Модель: **Qwen/Qwen3.5-2B-Base**. Run: `2026-06-05_10-38-02` · tag `v3_full_pos_shuffle` (merge `man_first` + `woman_first`, 21600 items).

Методология и запуск скриптов: [probes/README.md](../../../probes/README.md) (от `repo/`).

Behavioral metrics: [metrics/2026-06-05_23-02-12/summary.md](../metrics/2026-06-05_23-02-12/summary.md).

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

Пайплайн на прогон: layer_scan → control_task → silhouette → extract_direction → compare_directions (где завершён).

**H10** в этом run **не запускался** — ждём новую аннотацию (`annotation.option_labels`).

---

## Targets (логиты probing)

| ID | Target | Формула | Строки | Источник p |
|----|--------|---------|--------|------------|
| H11 | `log_odds` | `log(p_A / p_B)` | `task=main` | `prob_constrained_A/B` на gender axis |
| H12 | `abstain_logit` | `log(p_C / (1 − p_C))` | `task=main`, `with_abstain` | `prob_constrained_C` |
| H13 | `h13_yes_logit` | `log(p_Yes / p_No)` | `task=answerability` | `prob_constrained_Yes/No` на self-Q |

**H11 — gender preference.** На всех форматах (`choice`, `yesno_man`, `yesno_woman`) target — log-отношение constrained-вероятностей **первой и второй опции** промпта. Семантика man/woman задаётся `labels` и форматом; hard label для silhouette / `choice`:

- `choice`: A=man, B=woman (C=abstain при 3-way)
- `yesno_man`: Yes → man, No → woman
- `yesno_woman`: No → man, Yes → woman (cross-axis)

**H12 — abstain.** Mass на опции «Cannot determine» (C) в logit-форме. Gender mass (A/B) при 3-way частично переходит в C → см. более слабый global control у H11 `with_abstain`.

**H13 — self-answerability.** Companion-prompt после main: модель отвечает Yes/No на «можно ли ответить при показанных опциях?». Target — **модельный** P(Yes), не gold-аннотация `answerability`. Прогон **2+3**: pool строк с 2 и 3 main-опциями (slug `2+3`).

**H10 (отложено):** `h10_st_logit = log(p_st / (1 − p_st))`, где `p_st` — сумма `prob_constrained_*` на stereotype-опциях по **annotation.option_labels**.

---

## Сводка выводов

| ID | Ось в representations | Декодируемость (val R²) | Контроли (main vs global @ best) |
|----|-------------------------|-------------------------|----------------------------------|
| **H11** `without_abstain` | Gender preference `log_odds` (2 опции, n=2700) | **Очень сильная** (~0,98) | Валидный (−0,95) |
| **H11** `with_abstain` | Gender preference `log_odds` (3 опции, n=8100) | **Очень сильная** (~0,98) | Валидный (−0,44) |
| **H12** | Abstain preference `abstain_logit` (n=8100) | **Очень сильная** (~0,99) | Валидный (−0,48) |
| **H13** | Self-answerability `h13_yes_logit` (2+3, n=10800) | **Очень сильная** (~0,99) | Валидный (−0,36) |

Кластеризация (silhouette по бинаризованному target, `median_train`) **низкая** (0,03–0,10): линейная ось декодируется с высоким R², но чёткого разделения кластеров в HS нет.

На все основные оси локализуются в **поздних слоях (L24)** — в отличие от factorial v2, где H11 `with_abstain` пикал раньше (L8).

---

## H11 — предпочтения по gender axis

**Гипотеза:** предпочтение man vs woman **линейно декодируется** из hidden states.

**Target:** `log_odds = log(p_A / p_B)` на gender axis (choice + yesno_man + yesno_woman); hard label по semantic `labels`.

### `without_abstain` — 2 опции A/B (n=2700)

Каталог: [h11/without_abstain_all_choice_yesno_man_yesno_woman_main/](h11/without_abstain_all_choice_yesno_man_yesno_woman_main/)

| | |
|--|--|
| **Best layer** | **L24** |
| **val R² (main)** | **0,984** |
| **test R²** | **0,986** |
| **global shuffle @ L24** | **−0,946** |
| **val silhouette (max)** | **~0,053** (L16) |

### `with_abstain` — 3 опции A/B/C (n=8100)

Каталог: [h11/with_abstain_all_choice_yesno_man_yesno_woman_main/](h11/with_abstain_all_choice_yesno_man_yesno_woman_main/)

| | |
|--|--|
| **Best layer** | **L24** |
| **val R² (main)** | **0,983** |
| **test R²** | **0,983** |
| **global shuffle @ L24** | **−0,440** |
| **val silhouette (max)** | **~0,028** (L16) |

### Выводы

1. Gender-preference **сильно декодируется** на полном v3 срезе (all evidence × 3 формата); val≈test — без выраженного overfit.
2. **Контроли Hewitt–Liang:** main **≫** global shuffle на best layer → сигнал **валидный**.
3. При `with_abstain` global shuffle менее отрицателен, чем при `without_abstain` (часть variance уходит в abstain-ось → H12).
4. Silhouette низкий — как на factorial v2.

**Итог:** H11 **подтверждается** на v3; декодируемость **выше**, чем на n=75 factorial v2, за счёт большего n и полного факториала.

---

## H12 — предпочтение «Воздержаться»

**Гипотеза:** mass на опции C (`abstain_logit`) **линейно декодируется** из hidden states.

**Target:** `abstain_logit = log(p_C / (1 − p_C))`, только `with_abstain`.

### Основной прогон — 3 формата (n=8100)

Каталог: [h12/with_abstain_all_choice_yesno_man_yesno_woman_main/](h12/with_abstain_all_choice_yesno_man_yesno_woman_main/)

| | |
|--|--|
| **Best layer** | **L24** |
| **val R² (main)** | **0,992** |
| **test R²** | **0,993** |
| **global shuffle @ L24** | **−0,483** |
| **val silhouette (max)** | **~0,059** (L1) |

### Доп. прогон — только `choice` (n=2700)

Каталог: [h12/with_abstain_all_choice_main/](h12/with_abstain_all_choice_main/) · val R² **0,992**, L24, test R² **0,993**.

### Выводы

1. Abstain-preference **очень сильно декодируется**; метрики стабильны между срезом 3 формата и choice-only.
2. **Контроли:** main **(0,992)** ≫ global **(−0,48)** → **валидный** сигнал.
3. Silhouette низкий; best silhouette на раннем слое (L1) не совпадает с best R² (L24).

**Итог:** H12 **подтверждается**; на v3 abstain-ось декодируется так же сильно, как gender-preference (H11).

---

## H13 — self-answerability

**Гипотеза:** модельный self-assessment «можно ли ответить?» (`P(Yes)` на self-Q) **линейно декодируется** из hidden states.

**Target:** `h13_yes_logit = log(p_Yes / p_No)` на `task=answerability`.

### Прогон 2+3 — pool without + with abstain (n=10800)

Каталог: [h13/2+3_all_choice_yesno_man_yesno_woman_answerability/](h13/2+3_all_choice_yesno_man_yesno_woman_answerability/)

| | |
|--|--|
| **Best layer** | **L24** |
| **val R² (main)** | **0,992** |
| **test R²** | **0,992** |
| **global shuffle @ L24** | **−0,361** |
| **val silhouette (max)** | **~0,098** (L24) |

### Выводы

1. Self-Q Yes/No **сильно декодируется** из activations на том же промпте (HS snapshot перед ответом на answerability).
2. **Контроли:** main **≫** global → **валидный** сигнал (не артеfact shuffle).
3. Silhouette **чуть выше**, чем у H11/H12 (~0,10), но всё ещё умеренный.
4. Согласуется с behavioral H4/H7: P(Yes) на answerability систематически связана с design (см. metrics summary).

**Итог:** H13 **подтверждается** — internal state несёт информацию о модельной оценке answerability.

---

## Сравнение осей (основные прогоны, `all` evidence, 3 формата)

| Гипотеза | Target | n | Best layer | val R² | test R² | global @ best |
|----------|--------|---|------------|--------|---------|---------------|
| H11 `without_abstain` | `log_odds` | 2700 | L24 | 0,984 | 0,986 | −0,946 |
| H11 `with_abstain` | `log_odds` | 8100 | L24 | 0,983 | 0,983 | −0,440 |
| **H12** | **`abstain_logit`** | 8100 | L24 | **0,992** | **0,993** | **−0,483** |
| **H13** (2+3) | **`h13_yes_logit`** | 10800 | L24 | **0,992** | **0,992** | **−0,361** |

---



## Структура каталогов

```text
probes/
  README.md              # этот файл — выводы по run
  _shared/
    group_split_h10_v3_main.json
    group_split_h13_v3_answerability.json
  h11/
    without_abstain_all_choice_yesno_man_yesno_woman_main/
    with_abstain_all_choice_yesno_man_yesno_woman_main/
  h12/
    with_abstain_all_choice_yesno_man_yesno_woman_main/   # основной
    with_abstain_all_choice_main/                         # choice-only, n=2700
  h13/
    2+3_all_choice_yesno_man_yesno_woman_answerability/
```

В каждом прогоне: `meta.json` (stages, summary) и подпапки этапов с `results.json` (+ CSV/NPZ локально, `.gitignore`).

---

## Связь с behavioral 

| Probing | Behavioral аналог |
|---------|-------------------|
| H11 `log_odds` | H1/H3 — gender preference по output |
| H12 `abstain_logit` | H3/H8 — mass на C |
| H13 `h13_yes_logit` | H4/H7 — P(Yes) на self-Q answerability |


