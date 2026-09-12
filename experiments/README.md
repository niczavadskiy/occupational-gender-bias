# Experiments

Корень репозитория — прогон **Qwen3.5-2B-Base** (`src/`, `steering/`, `results/`, `docs/`).

Новые модели не получают отдельный GitHub-репозиторий: каталог здесь.

| Каталог | Модель |
|---|---|
| *(корень)* | `Qwen/Qwen3.5-2B-Base` — отчёт [Pages](https://niczavadskiy.github.io/occupational-gender-bias/) |
| [`qwen35-4b-base/`](qwen35-4b-base/) | `Qwen/Qwen3.5-4B-Base` — тот же H1·H3·H5·H11 (см. README внутри) |

Код метрик и steering-runners общие (корень). В каталоге модели: configs, results, vectors. Датасет общий: `../../repo/data` (локальный junction `data/`).
