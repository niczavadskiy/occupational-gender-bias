# H12: abstain-preference probing

Линейные пробы (Ridge) по hidden states: target **`abstain_logit = log(p_C / (1 − p_C))`** на choice-промптах с опцией abstain (A/B/C).

**Выводы по run:** [../README.md](../README.md) · методология: [probes/README.md](../../../../probes/README.md).

---

## Условия (этот run)

**`no_evidence`**, **`with_abstain`**, **`choice`**, n=75. Модель: **Qwen3.5-2B-Base**. Target: **`abstain_logit`**.

---

## Структура результатов

Каталог прогона: `{abstain}_{evidence}_{format}/` (как у H10).

```text
probes/h12/
  README.md
  with_abstain_no_evidence_choice/    # этот run
    meta.json                         # run_slug в meta
    layer_scan/ control_task/ silhouette/ extract_direction/ compare_directions/
```

Раньше артефакты лежали прямо в `probes/h12/`; перенесены в `with_abstain_no_evidence_choice/`.

---

## Результаты

Слой — по max **val_r2**.

| Метрика | train | val | test |
|---------|-------|-----|------|
| **R²** | 1.00 | **0.95** | **0.98** |
| **Pearson r** | 1.00 | 0.98 | 0.99 |
| **Δ val − test (R²)** | — | — | **−0.03** |

| Дополнительно | Значение |
|---------------|----------|
| Best layer | L16 |
| val silhouette (best) | 0.065 |
| projection ↔ abstain_logit r | 0.99 |
| control global @ L16 | −1.88 |
| control permuted @ L16 | −1.19 |

**Val vs test:** **согласованы** (0.95 vs 0.98, test чуть выше val — нормально при n=15 и выборе слоя по val). Train R² ≈ 1, но held-out splits стабильны → сильный сигнал без признаков val-overfitting.

**Вывод:** abstain-preference **сильно декодируется** (на par с H11 `without_abstain`). Локализация — L16 (средние/поздние слои).

### Сравнение с H11 на том же срезе

| Гипотеза | Target | val R² | test R² | Best layer |
|----------|--------|--------|---------|------------|
| H11 `without_abstain` | `log_odds` | 0.95 | 0.60 | L19 |
| H11 `with_abstain` | `log_odds` | 0.56 | 0.55 | L8 |
| **H12** | **`abstain_logit`** | **0.95** | **0.98** | **L16** |

Подробности — в `with_abstain_no_evidence_choice/meta.json` и `layer_scan/results.json`.

---
