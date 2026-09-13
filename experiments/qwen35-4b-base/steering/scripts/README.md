# 4B steering scripts

| Script | Purpose |
|---|---|
| `build_artifacts.sh` / `.ps1` | candidates + vectors + INLP subspaces → this tree |
| `vast_h1_stagea_and_pack.sh` | GPU Stage A gender (probe directions) |
| `vast_slot_stagea_and_pack.sh` | GPU Stage A slot |
| `vast_inlp_gender_stagea_and_pack.sh` | GPU INLP gender Stage A |
| `vast_inlp_slot_stagea_and_pack.sh` | GPU INLP slot Stage A |

On Vast set `REPO` to the inference repo root and keep this `steering/` tree available.
Model default: `Qwen/Qwen3.5-4B-Base`.
