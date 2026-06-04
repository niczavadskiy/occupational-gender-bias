# data/annotation/

LLM-разметка factorial items: `answerability` + stereotype labels per option.

## Rubric

См. [rubric.md](rubric.md) — короткий текст в system prompt. Аннотатору **не** говорим, какие `predicate` — control: сравнение с вашим списком control — post-hoc (джойн по `predicate` / `base_id`).

## Runs (generated)

`runs/<run_id>/`:

| Файл | Содержимое |
|------|------------|
| `meta.json` | model, `batch_size`, counts |
| `annotations.jsonl` | одна строка на item; `annotation` = `{id, answerability, option_labels}` |
| `errors.jsonl` | failed batch: `example_ids`, error, raw response |

По умолчанию **10 items на один API call**. Structured output: `json_schema` с корнем `{"items": [...]}`.

**Usage:** поле `batch.usage` дублируется на каждой строке батча — не суммируйте по всем строкам `annotations.jsonl` (считайте по уникальным `batch.openrouter_request_id` или по `meta` + числу батчей).

## Запуск

Из корня `repo/`:

```bash
cp .env.example .env   # OPENROUTER_API_KEY=...
python src/annotation/run.py --limit 3 --dry-run
python src/annotation/run.py --limit 20
python src/annotation/run.py --batch-size 10
python src/annotation/run.py
python src/annotation/jsonl_to_csv.py --input data/annotation/runs/<run_id>/annotations.jsonl
```

## Датасеты

| CSV | Персонажи |
|-----|-----------|
| `data/factorial_v2_with_gender.csv` | man / woman (default) |
| `data/factorial_v1_placeholders.csv` | X / Y |
