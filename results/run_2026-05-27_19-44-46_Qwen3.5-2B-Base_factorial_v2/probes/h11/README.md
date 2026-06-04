# H11: gender-preference probing

Линейные пробы (Ridge) по hidden states: target **`log_odds = log(p_man / p_woman)`** на choice-промптах.

**Выводы по run:** [../README.md](../README.md) · методология: [probes/README.md](../../../../probes/README.md).

---

## Условия (этот run)

**`no_evidence`**, **`choice`**, n=75. Модель: **Qwen3.5-2B-Base**. Target: **`log_odds`**.

Два прогона: `without_abstain/` (A/B), `with_abstain/` (A/B/C).

---

## Структура результатов

```text
probes/h11/
  README.md
  without_abstain/
    meta.json
    layer_scan/ control_task/ silhouette/ extract_direction/ compare_directions/
  with_abstain/
    ...
```

---

## Результаты

Слой — по max **val_r2** на best layer из `extract_direction`.

### `without_abstain` (A/B)

| Метрика | train | val | test |
|---------|-------|-----|------|
| **R²** | 1.00 | **0.95** | **0.60** |
| **Pearson r** | 1.00 | 0.98 | 0.83 |
| **Δ val − test (R²)** | — | — | **+0.35** |

| Дополнительно | Значение |
|---------------|----------|
| Best layer | L19 |
| val silhouette (best) | 0.10 |
| projection ↔ log_odds r | 0.98 |
| control global @ L19 | −1.21 |
| control permuted @ L19 | −0.36 |

**Val vs test:** val высокий, test заметно ниже (0.95 vs 0.60). **Generalization gap val→test**, не «test ≫ val». Train R² ≈ 1 → переобучение ridge на train; val сильный → сигнал **есть**.

**Вывод:** H11 **подтверждается** для 2-way. Согласованность val–test **умеренная**.

---

### `with_abstain` (A/B/C)

| Метрика | train | val | test |
|---------|-------|-----|------|
| **R²** | 1.00 | **0.56** | **0.55** |
| **Pearson r** | 1.00 | 0.82 | 0.81 |
| **Δ val − test (R²)** | — | — | **+0.01** |

| Дополнительно | Значение |
|---------------|----------|
| Best layer | L8 |
| val silhouette (best) | 0.04 |
| projection ↔ log_odds r | 0.92 |
| control global @ L8 | −1.24 |
| control permuted @ L8 | −1.21 |

**Val vs test:** **согласованы** (Δ ≈ 0.01). Нет val-specific overfitting; сигнал слабее, но стабильнее по generalization.

**Вывод:** H11 в 3-way **частично подтверждается** (val/test R² ≈ 0.55). Часть оси уходит в abstain → см. H12.

---

### Сводное сравнение

| Прогон | Best layer | val R² | test R² | val−test | Интерпретация |
|--------|------------|--------|---------|----------|---------------|
| `without_abstain` | L19 | 0.95 | 0.60 | +0.35 | Сильный val; test ниже |
| `with_abstain` | L8 | 0.56 | 0.55 | +0.01 | Умеренный, val≈test |

Подробности — в `without_abstain/meta.json`, `with_abstain/meta.json`.

---
