# XY_CONTROL steering candidates — `v1`

- Конфиг: `xy_control_4b_v1.yaml`
- Кандидатов: **47** (candidate 44, control 3) + baseline
- Forward estimate Stage A: **27264** (568 × n+1)
- Hook: `residual_stream_hidden_state`, `last_prompt_token`

## По семействам

| family | n | role |
| :--- | ---: | :--- |
| `control_random` | 3 | control |
| `main_add_core` | 32 | candidate |
| `main_add_edge` | 12 | candidate |

## Primary metric (Stage A)

- `gender_gap`: mean_i Δ_i, Δ_i = mean_layouts (log P(man) − log P(woman))
- keep_top: **10**

Полный список id — в JSON (`candidates[].id`).
