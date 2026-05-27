# Bias subspaces in LLM

Исследование локализации gender bias в скрытых состояниях `Qwen3.5-2B-Base`.
4-недельный AI safety project, mentor — Sabrina Sadiekh.

**Команда:** Olga Masaeva (этот репо — inference + HS extraction + probing pipeline),
Никита (factorial dataset + probing), Руслан (методология + статистика).

**Текущий статус (W1, 2026-05-27):** factorial v2 dataset (1350 items) прогнан на
Qwen3.5-2B-Base, hidden states собраны на всех 25 слоях, результаты в HF
`bias-subspaces-group/qwen-bias-experiments` (private).

---

## Структура

```
.
├── data/                            # Никитин factorial dataset
│   ├── factorial_v1_from_nikita.csv     # X/Y placeholders (legacy)
│   └── factorial_v2_with_gender.csv     # man/woman подставлены — каноничный input
├── src/
│   ├── prepare_factorial.py         # v2 CSV → JSONL (prompt + has_abstain)
│   ├── inference.py                 # constrained log-prob A/B/C + HS на 25 слоях
│   ├── smoke_qwen.py                # smoke-test модели
│   └── upload_to_hf.py              # заливка results в bias-subspaces-group/...
├── results/                         # gitignored — артефакты заливаются в HF
├── requirements.txt
├── .env.example                     # шаблон для HF_TOKEN
└── .gitignore
```

Большие артефакты (`*.npz`, `results/`, `*.prepared.jsonl`) **не коммитим** — они
живут в [HF Dataset](https://huggingface.co/datasets/bias-subspaces-group/qwen-bias-experiments)
(private repo внутри org `bias-subspaces-group`).

---

## Reproduction recipe

### 0. Pre-reqs

- Linux + NVIDIA GPU ≥ 12 GB VRAM (RTX 3090 / 4090 / A100 fine). Можно Vast.ai.
- Python 3.10+
- HuggingFace account с access token (write scope для upload). Получить:
  https://huggingface.co/settings/tokens
- Доступ в org `bias-subspaces-group` (приватный) — попросить инвайт у Olga.

### 1. Clone + установка зависимостей

```bash
git clone https://github.com/olyamasaeva/Bias--subspaces-in-LLM.git
cd Bias--subspaces-in-LLM
git checkout qwen_2b_experiments

# torch ставим отдельно — версия CUDA wheel должна совпадать с драйвером хоста
# Для CUDA 12.8 (типичный Vast):
pip install torch --index-url https://download.pytorch.org/whl/cu128
# Для CPU-only:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

### 2. Setup `.env`

```bash
cp .env.example .env
$EDITOR .env   # вписать HF_TOKEN=hf_...
```

Скрипты подхватят `.env` автоматически (через `python-dotenv`).

### 3. Smoke test (опционально, ~3-5 мин первый раз — качает модель ~4 GB)

```bash
python3 src/smoke_qwen.py
```

Проверяет: модель грузится, `trust_remote_code=True` работает, токены `" A"/" B"/" C"`
single-token, forward + HS extraction отдают [25, 2048].

### 4. Подготовка датасета

```bash
python3 src/prepare_factorial.py data/factorial_v2_with_gender.csv
# → data/factorial_v2_with_gender.prepared.jsonl (1350 items, ~930 KB)
```

Скрипт читает Никитин CSV (man/woman уже подставлены), собирает поле `prompt` со
структурой BBQ:

```
<scenario_text>           # base_context + optional evidence
<question>                # "Who was X?" или "Was man/woman X?"
A. <option_A>             # man / Yes
B. <option_B>             # woman / No
[C. <option_C>]           # Cannot determine — только если abstain_variant=with_abstain
Answer:
```

Position-bias control встроен через парные `yesno_man`/`yesno_woman` queries —
swap'ов не нужно. См. [src/prepare_factorial.py](src/prepare_factorial.py) docstring.

### 5. Inference

```bash
python3 src/inference.py \
    --items_file data/factorial_v2_with_gender.prepared.jsonl \
    --run_tag factorial_v2
```

Время: ~7-8 минут на RTX 3090, ~3-4 мин на 4090.

На выход в `$RESULTS_DIR/run_<timestamp>_Qwen3.5-2B-Base_factorial_v2/`:
- `hidden_states.npz` — `[1350, 25, 2048]` float16, ~100 MB
- `per_item.jsonl` — 1350 строк × 24 поля (id, prompt, raw logits, log-probs над full vocab, constrained probs, choice)
- `meta.json` — version info, runtime, hyperparams

`$RESULTS_DIR` дефолт `/workspace/results`, переопределить через `.env` (RESULTS_DIR=…).

### 6. Upload в HF Dataset

```bash
python3 src/upload_to_hf.py results/run_*/
```

Создаст приватный repo `bias-subspaces-group/qwen-bias-experiments` (если ещё нет)
и зальёт run как sub-path.

---

## Что в `per_item.jsonl`

| Поле | Значение |
|---|---|
| `id`, `example_id`, `base_id` | идентификаторы |
| `base_context`, `scenario_text` | ambiguous baseline / полный passage |
| `predicate`, `evidence_shift`, `question_format`, `abstain_variant` | factorial labels |
| `question`, `labels`, `has_abstain`, `prompt` | input в модель |
| `logit_A`, `logit_B`, `logit_C` | raw logits на token positions " A"/" B"/" C" |
| `logprob_vocab_A/B/C` | `log_softmax` над full vocab — true log P(token \| prompt) |
| `prob_constrained_A/B/C` | softmax только над valid options (для without_abstain — над `[A,B]`; иначе над `[A,B,C]`) |
| `valid_labels`, `choice` | какие опции валидны для item и argmax |


---

## License / citation

Внутренний research project (AI safety mentorship). Датасет — Никитин,
inference + reproducibility pipeline — Olga. Использование/repost — спросить
сначала.
