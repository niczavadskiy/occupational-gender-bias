# SLOT steering candidates — `v1`

- Конфиг: `slot_steering_candidates_v1.yaml`
- Кандидатов: **87** (candidate 75, control 12) + baseline
- Forward estimate Stage A: **33440** (380 × n+1)
- Hook: `residual_stream_hidden_state`, `last_prompt_token`

## По семействам

| family | n | role |
| :--- | ---: | :--- |
| `anti_steering` | 2 | control |
| `causality_shift` | 6 | control |
| `control_gender` | 1 | control |
| `control_random` | 3 | control |
| `main_center_core` | 42 | candidate |
| `main_center_edge` | 18 | candidate |
| `main_project_out` | 12 | candidate |
| `mid_project_out` | 3 | candidate |

## Primary metric (Stage A)

- `mean_abs_phi_dev`: mean_i |φ_i − 0.5|, φ_i = A_i/(A_i+B_i) по base item
- keep_top: **15**

Полный список id — в JSON (`candidates[].id`).
