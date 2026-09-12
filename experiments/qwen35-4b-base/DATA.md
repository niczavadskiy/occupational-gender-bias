# Dataset (shared)

Occupational factorial **не копируется** сюда.

| Что | Где |
|---|---|
| Input JSONL (45648) | [`../../../repo/data/v1/`](../../../repo/data/v1/) |
| Rebuild | `cd repo && python -m src.extract_v1_items_from_per_item` |
| Source 2B outputs | `repo/results/highlight-h3-full/per_item.jsonl` |
| Local junction | `data/` → `repo/data` |

Счётчики: without_abstain=11412, with_abstain=34236, families=951, total=45648.
