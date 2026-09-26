# Dataset (shared)

Occupational factorial **не копируется** сюда.

| Что | Где |
|---|---|
| Input JSONL (45648) | junction `data/` → repo [`data/`](../../data/) |
| Rebuild | `python -m src.extract_v1_items_from_per_item` (из корня репо) |
| Source 2B outputs | `results/highlight-h3-full/per_item.jsonl` (корень) |

Счётчики: without_abstain=11412, with_abstain=34236, families=951, total=45648.

## Steering / INLP

Конфиги и чеклист после peak: [`steering/PIPELINE.md`](steering/PIPELINE.md).
