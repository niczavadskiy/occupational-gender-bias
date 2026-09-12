# data/v1 — occupational inference items (input only)

Extracted from `results/highlight-h3-full/per_item.jsonl` (2B outputs stripped).

## Factorial

- scenarios (soc × profession × onet_action): **951**
- evidence_shift: no_evidence / man / woman (**3**)
- context_order: man_first / woman_first (**2**)
- without_abstain positions: **2** → 951×3×2×2 = **11412**
- with_abstain positions: **6** → 951×3×2×6 = **34236**
- **total 45648**

## Files

| File | Rows |
|---|---:|
| `inference_items_h3_full.jsonl` | 45648 |
| `inference_items_hl_man_man_first.jsonl` | 7608 |
| `inference_items_hl_man_woman_first.jsonl` | 7608 |
| `inference_items_hl_woman_man_first.jsonl` | 7608 |
| `inference_items_hl_woman_woman_first.jsonl` | 7608 |
| `inference_items_v1_man_first.jsonl` | 7608 |
| `inference_items_v1_woman_first.jsonl` | 7608 |

Rebuild:

```bash
python -m src.extract_v1_items_from_per_item
```

```json
{
  "source_per_item": "E:/~edu/AI Safety/Bias subspaces в LLM/repo/results/highlight-h3-full/per_item.jsonl",
  "n_items": 45648,
  "n_families": 951,
  "factorial": {
    "scenarios": 951,
    "evidence": 3,
    "context_order": 2,
    "without_abstain_positions": 2,
    "with_abstain_positions": 6,
    "without_abstain": 11412,
    "with_abstain": 34236,
    "total": 45648
  },
  "files": {
    "full": "inference_items_h3_full.jsonl",
    "shards": [
      "inference_items_hl_man_man_first.jsonl",
      "inference_items_hl_man_woman_first.jsonl",
      "inference_items_hl_woman_man_first.jsonl",
      "inference_items_hl_woman_woman_first.jsonl",
      "inference_items_v1_man_first.jsonl",
      "inference_items_v1_woman_first.jsonl"
    ]
  },
  "validation": {
    "n": 45648,
    "n_families": 951,
    "abstain": {
      "without_abstain": 11412,
      "with_abstain": 34236
    },
    "evidence": {
      "no_evidence": 15216,
      "man": 15216,
      "woman": 15216
    },
    "context": {
      "man_first": 22824,
      "woman_first": 22824
    },
    "position": {
      "p0": 11412,
      "p1": 11412,
      "p2": 5706,
      "p3": 5706,
      "p4": 5706,
      "p5": 5706
    },
    "missing_keys_in_sample": [],
    "ok": true,
    "errors": []
  }
}
```
