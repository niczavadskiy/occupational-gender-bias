# Experiments

Корень репозитория — прогон **Qwen3.5-2B-Base** (`src/`, `steering/`, `results/`, `docs/`).

Новые модели не получают отдельный GitHub-репозиторий: каталог здесь.

| Каталог | Модель |
|---|---|
| *(корень)* | `Qwen/Qwen3.5-2B-Base` — отчёт [Pages](https://niczavadskiy.github.io/occupational-gender-bias/) |
| [`qwen35-4b-base/`](qwen35-4b-base/) | `Qwen/Qwen3.5-4B-Base` — тот же H1·H3·H5·H11 (см. README внутри) |
| [`gemma3-1b-pt/`](gemma3-1b-pt/) | `google/gemma-3-1b-pt` — beh → probe → INLP → pool XY |
| [`gemma3-4b-pt/`](gemma3-4b-pt/) | `google/gemma-3-4b-pt` — beh → probe → INLP → pool XY |
| [`ministral3-3b-base/`](ministral3-3b-base/) | `mistralai/Ministral-3-3B-Base-2512` — beh → probe → INLP → pool XY |
| [`ministral3-8b-base/`](ministral3-8b-base/) | `mistralai/Ministral-3-8B-Base-2512` — beh → probe → INLP → pool XY |

Код метрик и steering-runners общие (корень). В каталоге модели: configs, results, vectors. Датасет общий: junction `data/` → repo `data/`.
