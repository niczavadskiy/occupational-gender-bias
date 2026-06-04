# data/annotation/

LLM-разметка factorial items: `answerability` + stereotype labels per option (`stereotype_consistent` | `anti_stereotype` | `neutral`) .

## Rubric

См. [rubric.md](rubric.md) — инструкция аннотатору в system prompt.  

## Runs

`runs/<run_id>/` (в git: `*.json`, `*.jsonl`, `*.csv`; lock-файлы редакторов — в `.gitignore`):

| Файл | Содержимое |
|------|------------|
| `meta.json` | model, `batch_size`, counts |
| `annotations.jsonl` | одна строка на item; `annotation` = `{id, answerability, option_labels}` |
| `errors.jsonl`(планируется) | failed batch: `example_ids`, error, raw response | 

По умолчанию **10 items на один API call**. Structured output: `json_schema` с корнем `{"items": [...]}` подается с параметрами на вход модели. После получения ответа производится дополнительная валидация по `json_schema`

## Запуск

Из корня `repo/`:

```bash
 
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

