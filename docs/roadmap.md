# Roadmap (W2–W4)

Состояние на 2026-06-02. Цель: разблокировать H-гипотезы Никиты (H1–H13) и
понять, нужно ли ONET-расширение во вторую половину проекта.

---

## Что уже сделано (W1 + начало W2)

- ✅ Inference pipeline для Qwen3.5-2B-Base + HS extraction всех 25 слоёв
- ✅ Прогон factorial v2 (1350 items): logits + constrained probs + HS, залит
  в HF Dataset `bias-subspaces-group/qwen-bias-experiments` (private)
- ✅ EDA-отчёт по прогону (`analysis/inference_eda.html`) с явным флагом
  position-confound в choice format
- ✅ Code-расширение pipeline'а (готов к запуску, ждёт GPU):
  - `prepare_factorial.py` — флаги `--with_position_swap` и
    `--with_answerability` (default on); 5400 items на выходе
  - `inference.py` — ветвление по `answer_type` (abc/yesno), `--cache_from`
    для prompt-hash кэширования (~25% экономии на следующем прогоне)

## Маппинг ONET-расширения → H-гипотезы

ONET не разблокирует *новых* гипотез — он улучшает качество данных для
конкретных. Сравним с тем что **уже** разблокируется через прогон 5400:

| Гипотеза | Что нужно | Готово после прогона? | ONET помогает? |
|---|---|---|---|
| H1 gender preference | logits + position_swap | да | нет |
| H2 stereotype choices | разметка annotator'а | нет (Claude от Никиты) | косвенно — у профессий direction чище |
| H3 evidence-following | релевантный evidence per (ctx × pred) | нет — v2 evidence привязан только к ctx | косвенно — у профессий evidence пишется естественно |
| H4 abstain → answerability | answerability + main | да | нет |
| H5 abstain × evidence | answerability + main | да | нет |
| H6 format effect | choice vs yesno на одних items | да | нет |
| H7 evidence → self-answerable | answerability + evidence | да | нет |
| H8 self-unanswerable → abstain | answerability + main paired | да | нет |
| H9 asymmetric answerability | answerability с position_swap | да | нет |
| **H10 stereo в internals** | **clean stereotype labels** для probe | пока нет (нужна разметка) | **да — главный win** |
| **H11 gender в internals** | clean gender labels | да (gender известен) | да (cleaner labels для probe training) |
| H12 abstain в internals | answerability + main HS | да | нет |
| H13 answerability в internals | answerability HS | да | нет |

## Приоритеты на W2–W4

Порядок не по красоте, а по vol-to-effort: что разблокирует больше H за меньшие
усилия — раньше.

| # | Шаг | Кто | Сколько | Что разблокирует |
|---|---|---|---|---|
| 1 | GPU прогон 5400 (main + answerability + position_swap, `--cache_from` старого run'а) | Оля, эта неделя | 1 запуск (~25 мин на 3090) | H4, H5, H6, H7, H8, H9, H13 |
| 2 | Claude-аннотация stereotype direction по 1350 base items | Никита, параллельно | по его плану | H2, H10 |
| 3 | Фикс evidence per (context × predicate) | автор v2 | 30–60 мин или Claude-генерация | H3 без mention-priming-конфаунда |
| 4 | Probing на текущих данных | Никита | W3 | baseline для H10–H13 |
| 5 | По результатам probing: если signal слабый → ONET (см. ниже); если ок → BBQ replication subset для external validity | по решению | W3–W4 | усиление H10/H11 или paper-level validity |

## ONET — минимальный дизайн (conditional, на W4)

Запускаем только если probes на текущих данных показывают слабый/шумный сигнал
по H10/H11. Цель — дать Никите датасет с чище размеченными stereotype labels.

**Дизайн:**

- **40 WinoBias профессий** (peer-reviewed bias-список) как primary axis.
  Не идём в стратифицированный ONET-50 — для probing достаточно 40 × вариантов.
- **Profession-as-predicate**: `"X and Y attended a hiring panel. Who is the {profession}?"` — самый чистый формат для probing'а, не смешивается с предикатами v2.
- **Stereotype labels через BLS** (Current Population Survey, Table 11):
  - male share > 60% → stereotype_consistent для man, anti для woman
  - male share < 40% → наоборот
  - 40–60% → neutral
  Это **deterministic ground-truth**, не нужен LLM-judge.
- **Те же factor-axes:** 3 evidence × 3 format × 2 abstain × 2 position × 2 task (main + answerability).
- **Размер:** 40 × 3 × 3 × 2 × 2 × 2 = **2880 items**, ~7 мин на RTX 3090.
- **Probing claim:** натренировать linear probe на этих 2880, протестировать на
  исходных 5400 — измеряет **cross-domain transfer of stereotype subspace**
  (= main claim проекта).

**Где брать данные:**
- WinoBias профессии: <https://github.com/uclanlp/corefBias> (40 шт, gender-labeled)
- BLS Table 11: <https://www.bls.gov/cps/cpsaat11.htm> (gender breakdown per occupation, public)

**Что НЕ делаем в этой итерации:**
- Полный ONET-873 (избыточно для probing)
- Профессия-в-контексте + наши 15 предикатов (это другое исследование)
- Claude-аннотация для ONET-items (BLS gender share — deterministic, дешевле)

## Открытые вопросы

1. **Evidence-фикс**: переписать вручную 75 evidence-фраз или сгенерить через Claude? Объём маленький, ручное надёжнее.
2. **Claude-аннотация** stereotype direction: какой формат вывода ожидает Никита для probe-тренировки? (out-of-scope для W2).
3. **Probing baseline:** через какие слои гонять linear probe сначала? (Никитино решение — H10–H13 его трек).
4. **BBQ-replication**: на каком subset BBQ? Gender_identity полностью или sample? (~5672 items full).
