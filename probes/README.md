# probes/

Линейный probing по hidden states inference-run (`results/<run>/`).  
Пакет рассчитан на гипотезы **H10–H13**; сейчас реализован **H11**.

Запуск из **корня репозитория** (`repo/`):

```powershell
cd repo
python -m probes.<модуль> ...
```

---

## Гипотезы (статус)

| ID | Тема | Статус | Документация |
|----|------|--------|--------------|
| **H10** | — | planned | — |
| **H11** | Gender-preference (`log_odds`) decodable from HS | **done** | `results/<run>/probes/h11/README.md` |
| **H12** | — | planned | — |
| **H13** | — | planned | — |

Детали H11 — в README рядом с артефактами run: `results/<run>/probes/h11/README.md`.

---

## Точки входа (CLI)

| Команда | Назначение |
|---------|------------|
| `python -m probes.h11_run --abstain-variant without_abstain` | **H11:** полный пайплайн probing |
| `python -m probes.h11_run --abstain-variant with_abstain` | H11 с опцией C (abstain) |
| `python -m probes.inspect_h11` | Без hidden states: фильтр choice/`without_abstain`, split по family → `_shared/group_split_scenario_family.json`, manifest CSV и баланс choice — preflight перед `h11_run` |

---

## Модули (библиотека)

| Файл | Роль |
|------|------|
| `h11_run.py` | Оркестратор пайплайна H11 |
| `h11.py` | Batch, evidence modes, `log_odds`, фильтр abstain |
| `inspect_h11.py` | Manifest + split check |
| `core.py` | Ridge / logistic, метрики, загрузка bundle |
| `splits.py` | Group split по `scenario_family_id` |
| `load_run.py`, `paths.py` | Run dir, `per_item.jsonl`, hidden states |
| `run_io.py` | `meta.json`, запись results, pipeline-meta |
| `layer_scan.py` | R², Pearson r по слоям |
| `control_task.py` | Hewitt–Liang controls |
| `silhouette.py` | Silhouette vs бинарный `log_odds` |
| `extract_direction.py` | Вектор `w`, проекции |
| `compare_directions.py` | cos(w) между слоями |
| `__init__.py` | Package init |

---

## Данные и артефакты

```text
  probes/
    _shared/, inspect_h11/              # локально (.gitignore)
    h11/
      without_abstain/ | with_abstain/
      README.md
      meta.json, */results.json   # git
      *.csv, *.npz                # локально
```


