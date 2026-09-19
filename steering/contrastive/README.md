# Contrastive steering

Направление — **разность средних** last-prompt-token HS по живому выбору модели:

\[
v = \mathrm{mean}(h \mid y=M) - \mathrm{mean}(h \mid y=F)
\]

Классы взвешены поровну (сначала среднее группы, потом вычитание).  
\(y\) = argmax constrained \(P(\mathrm{man}\mid x)\) vs \(P(\mathrm{woman}\mid x)\) на live forward.

Интервенция — та же, что H1 **center**:

\[
h' = h - \alpha(s - c)\hat v, \qquad s = h\cdot\hat v
\]

\(c\) — mid-point средних проекций классов на **train-семьях** captured sample.

Глобальный сдвиг \(h-\alpha v\) здесь специально не основной: он не схлопывает
\(\mu_M-\mu_F\) и уводит женские профессии ещё дальше.

## Каталог

```
steering/contrastive/
├── configs/
│   ├── contrastive_gender_2b_v1.yaml      # пояс L14–20, якорь L16
│   └── contrastive_gender_4b_v1.yaml      # пояс L20–26, якорь L23
├── candidates/
│   ├── contrastive_gender_2b_candidates_v1.json
│   └── contrastive_gender_4b_candidates_v1.json
├── vectors/                               # v̂ npz после build_vectors
├── captures/                              # last-token HS dump (не в git)
├── rank/                                  # спектр D + вердикт (json/md)
├── build_candidates.py
├── build_vectors.py                       # live HS → v̂, c (+ capture)
├── analyze_rank.py                        # per-SOC PCA, shuffle-null, V_k
├── run_stagea.py                          # обёртка run_h1_stagea
├── run_pca_stagea.py                      # Stage A bake-off k=1..4 vs random
├── CONTRASTIVE_VAST.md
└── scripts/
    ├── vast_contrastive_stagea_and_pack.sh
    ├── vast_contrastive_rank_and_pack.sh
    └── vast_contrastive_full_instance.sh
```

Сетка (и 2B, и 4B): **46 кандидатов** + baseline.

| family | n | role |
|---|---:|---|
| `main_center_core` | 24 | candidate |
| `main_center_edge` | 6 | candidate |
| `main_project_out` | 7 | candidate |
| `causality_shift` | 4 | control |
| `anti_steering` | 2 | control |
| `control_random` | 3 | control |

Primary metric Stage A — `mean_i |θ_i − 0.5|` на тех же 95 val-семьях, что H1.

## Запуск

Из корня `occupational-gender-bias`:

```powershell
# 1. Заморозить каталог кандидатов (CPU, без модели)
python -m steering.contrastive.build_candidates
python -m steering.contrastive.build_candidates --scale 4b
python -m steering.contrastive.build_candidates --verify
python -m steering.contrastive.build_candidates --scale 4b --verify

# 2. Снять HS и посчитать v̂, c (GPU)
python -m steering.contrastive.build_vectors --device cuda
python -m steering.contrastive.build_vectors --scale 4b --device cuda

# 3. Stage A
python -m steering.contrastive.run_stagea --device cuda --tag contrastive_2b_a_v1
python -m steering.contrastive.run_stagea --scale 4b --device cuda --tag contrastive_4b_a_v1
```

Smoke:

```powershell
python -m steering.contrastive.build_vectors --n-items 3 --device cuda --tag smoke
python -m steering.contrastive.run_stagea --limit-items 3 --tag smoke `
  --candidates main_center_core__vmd__L16__center__a1,anti_steering__vmd__L16__center__am1
```

На 4B smoke-якорь — L23: `main_center_core__vmd__L23__center__a1`.

Выход Stage A: `results/steering/contrastive/stage_a/<tag>/` (тот же `ranking.csv`, что у H1).

## Vast (2B)

Подробно: [`CONTRASTIVE_VAST.md`](CONTRASTIVE_VAST.md). Ноутбук без SSH: [`notebooks/vast_contrastive_2b.ipynb`](../../notebooks/vast_contrastive_2b.ipynb).

```bash
export HF_TOKEN=hf_xxx
cd /workspace/occupational-gender-bias && git pull
bash steering/contrastive/scripts/vast_contrastive_full_instance.sh
```

## На что смотреть в мета векторов

В `*_vectors_v1.json` на якоре слоя:

- `auc_train` / `auc_test` — разделяет ли \(s=h\cdot\hat v\) live-choice
- `cos_with_w_gender` — если рядом лежит H1 npz; ≈1 значит mean-diff ≈ проба
- `separation` — зазор средних проекций на train

## Rank PCA — одно ли направление?

\(v_G\) — первый момент. Чтобы не предполагать одномерность, собираем матрицу
**per-SOC** контрастов на train-семьях (не per-row):

\[
d_s=\mathrm{mean}(h\mid s,y=M)-\mathrm{mean}(h\mid s,y=F),\qquad
D=\begin{bmatrix}d_{s_1}\\ \vdots\\ d_{s_n}\end{bmatrix}
\]

Дальше: uncentered SVD (PC1 должен совпасть с \(v_G\)); **unit-row centered**
PCA + shuffle \(y\) внутри страты (null). Если угловой спектр не выше null —
гипотеза одного направления жива. Если 1–3 компоненты над null — пишем
\(V_k=[v_G,\mathrm{PC}_1^\perp,\ldots]\) и гоняем Stage A center
\(h'=h-\alpha V(V^\top h-c)\). k=1 в этом базисе **есть** contrastive \(v_G\).

```powershell
python -m steering.contrastive.test_rank
python -m steering.contrastive.analyze_rank --device cuda
python -m steering.contrastive.run_pca_stagea --device cuda --tag pca_2b_a_v1
```

Если capture уже снят `build_vectors`, `analyze_rank` подхватит его без GPU.
Переснять: `--recapture`. Vast только диагностика (без сетки 46):

```bash
bash steering/contrastive/scripts/vast_contrastive_rank_and_pack.sh
RUN_PCA_STAGEA=1 bash steering/contrastive/scripts/vast_contrastive_rank_and_pack.sh
```

Выход: `steering/contrastive/rank/*.md`, `steering/subspaces/contrastive_pca_{scale}_{tag}.npz`.
Смотреть `verdict`, `cos(PC1, v_G)`, таблицу SOC, и в Stage A `mean_R_gender`
для k=1 vs k=2..4 vs random того же ранга.
