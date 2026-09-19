# CONTRASTIVE steering candidates — `v1`

- Конфиг: `contrastive_gender_4b_v1.yaml`
- Кандидатов: **46** (candidate 37, control 9) + baseline
- Forward estimate Stage A: **17860** (380 × n+1)
- Hook: `residual_stream_hidden_state`, `last_prompt_token`

## По семействам

| family | n | role |
| :--- | ---: | :--- |
| `anti_steering` | 2 | control |
| `causality_shift` | 4 | control |
| `control_random` | 3 | control |
| `main_center_core` | 24 | candidate |
| `main_center_edge` | 6 | candidate |
| `main_project_out` | 7 | candidate |

## Primary metric (Stage A)

- `mean_abs_theta_dev`: mean_i |θ_i − 0.5|, θ_i = M_i/(M_i+W_i) по base item
- keep_top: **10**

Полный список id — в JSON (`candidates[].id`).
