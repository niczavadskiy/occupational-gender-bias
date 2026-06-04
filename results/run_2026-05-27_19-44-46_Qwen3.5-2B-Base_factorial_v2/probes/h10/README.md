# H10: stereotype-consistency probing

Линейные пробы (Ridge) по hidden states: target **`h10_st_logit = log(p_st / (1 − p_st))`**, где `p_st` — сумма `prob_constrained_*` по опциям с меткой `stereotype_consistent` из внешней аннотации.

Join inference ↔ аннотация **только по `example_id`**.

**Выводы по run:** [../README.md](../README.md) · методология: [probes/README.md](../../../../probes/README.md).

---

## Условия (этот run)

| Срез | Каталог | n |
|------|---------|---|
| `no_evidence` + `choice` | `without_abstain_no_evidence_choice`, `with_abstain_no_evidence_choice` | 75 |
| `all` × 3 формата | `without_abstain_all_choice_yesno_man_yesno_woman`, `with_abstain_all_choice_yesno_man_yesno_woman` | 675 |

Модель: **Qwen3.5-2B-Base**. Target: **`h10_st_logit`**. Аннотация: Claude Opus 4.8 (`2026-06-02_13-10-31_...`).

---

## Структура результатов

```text
probes/h10/
  README.md
  without_abstain_no_evidence_choice/
    meta.json
    layer_scan/ control_task/ silhouette/ extract_direction/ compare_directions/
  with_abstain_no_evidence_choice/
  without_abstain_all_choice_yesno_man_yesno_woman/
  with_abstain_all_choice_yesno_man_yesno_woman/
```

---

## Результаты

Слой — по max **val_r2**. Для сравнения: H11 на `no_evidence` + `choice` даёт val R² ≈ **0.95**.

### Чистый срез: `no_evidence` + `choice`, n=75

| Прогон | Best layer | val R² | test R² | control global @ best |
|--------|------------|--------|---------|------------------------|
| `without_abstain_no_evidence_choice` | L4 | 0.09 | 0.99 | −2.14 |
| `with_abstain_no_evidence_choice` | L13 | 0.17 | 0.99 | −2.86 |

Сигнал **неслучайный** (controls), но **слабый** на val; большой train–val gap (train R² ≈ 1). Test ≫ val — нестабильный test на n=15, не надёжная generalization-метрика.

### Полный факториал: `all` × 3 формата, n=675

| Прогон | Best layer | val R² | test R² | control global @ best |
|--------|------------|--------|---------|------------------------|
| `without_abstain_all_choice_yesno_man_yesno_woman` | L18 | 0.34 | 0.83 | −3.55 |
| `with_abstain_all_choice_yesno_man_yesno_woman` | L9 | 0.28 | 0.80 | −2.04 |

Больше строк на family → **выше val** и **меньше разрыв val–test**, но val всё ещё далеко от H11/H12.

**Краткий вывод:** stereotype-ось в representations присутствует слабо; H10 в сильной форме не поддерживается, в мягкой («умеренный линейный сигнал при pooled data») — частично.

Подробности — в `*/meta.json` и `layer_scan/results.json`.

---
