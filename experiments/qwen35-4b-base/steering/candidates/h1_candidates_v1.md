# H1 steering candidates — `v1`

- Конфиг: `h1_steering_candidates_v1.yaml`
- Кандидатов: **105** (candidate 94, control 11) + baseline
- Forward estimate Stage A: **40280** (380 × n+1)
- Hook: `residual_stream_hidden_state`, `last_prompt_token`

## По семействам

| family | n | role |
| :--- | ---: | :--- |
| `anti_steering` | 2 | control |
| `causality_shift` | 4 | control |
| `control_other_axes` | 2 | control |
| `control_random` | 3 | control |
| `main_center_core` | 56 | candidate |
| `main_center_edge` | 24 | candidate |
| `main_project_out` | 14 | candidate |

## Primary metric (Stage A)

- `mean_abs_theta_dev`: mean_i |θ_i − 0.5|, θ_i = M_i/(M_i+W_i) по base item
- keep_top: **15**

Полный список id — в JSON (`candidates[].id`).
