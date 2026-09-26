# Polarity-pooled INLP

Matched intervention arm to **XY-control polarity pool**: one subspace per
pro-male / pro-female FDR set, same family split, same candidate **layer belt**
(probe peak + two layers below).

| | XY polarity pool | INLP polarity pool (this) |
|---|---|---|
| Unit | one \(v_{\mathrm{raw}}\) per polarity | one \(W\) (INLP) per polarity |
| Fit | train families, paired \(h_g-h_{xy}\) | train families, gendered HS → `gender_prob` |
| Select | layer × α on pooled val | layer × rank (α=1) on pooled val |
| Layers | L16 / L15 / L14 | **same** L16 / L15 / L14 |
| Confirm | Stage B MMLU → Stage C test | same A→B→C, MMLU = union of member SOCs |

Frozen artifacts:

- Config: [`configs/inlp_polarity_pool_2b_peak_prepeak.yaml`](configs/inlp_polarity_pool_2b_peak_prepeak.yaml)
- Sets: [`domains/inlp_polarity_sets_2b_v1.json`](domains/inlp_polarity_sets_2b_v1.json)
- SOC catalog (shared): [`xy_control/domains/h1_soc_fdr_v1.json`](xy_control/domains/h1_soc_fdr_v1.json)
- Family universe: [`xy_control/data/xy_pairs_full_v1.json`](xy_control/data/xy_pairs_full_v1.json)

---

## 1. Candidate layers (matched to XY)

From gender_prob probe peak (`inlp_gender_prob_v1.yaml` → **L16**):

| role | layer |
|------|------:|
| peak | **16** |
| peak−1 | 15 |
| peak−2 | 14 |

Do **not** add layers above the peak in the primary grid (same rule as XY
`xy_control_2b_peak_prepeak_a6`).

Stage A grid (preferred):

\[
\{16,15,14\} \times \{k=1,4,8,16\} \times \{\alpha=1\}
\]

Stage A **clamps** ranks to subspace `k_found`. Polarity `gender_prob` often
stops at `k_found=1` (chance corr) — then the grid is just L16/L15/L14 × k=1.

---

## 2. Polarity sets

Same 2B FDR SOCs as XY `polarity_sets_2b_v1.json`:

| set_id | polarity | n SOC |
|--------|----------|------:|
| `promale` | male | 8 |
| `profemale` | female | 2 |

One subspace / one winner **per set**. No per-SOC INLP in this protocol
(per-SOC stays exploratory / appendix).

---

## 3. Split (must match XY)

Inside each member `soc_major_title`:

| split | frac | role |
|-------|-----:|------|
| train | 0.70 | fit \(W\) (and centers \(c\)) |
| val | 0.15 | Stage A: pick (L, k) |
| test | 0.15 | Stage C lock |

Then **union** family ids across member SOCs → pooled train/val/test.

- Seed **0**, `family_split3_by_soc` (same helper as XY).
- Consequence: for a given SOC, family membership matches XY polarity pool →
  Stage C can compare INLP vs XY on the **same** test families.

During INLP fit, chance-stopping / held-out AUC uses families **not** in
train (val∪test of the polarity split). Stage A still scores only **val**.

---

## 4. Pipeline

```
[0] build sample (gendered rows from xy_pairs, filtered to set)
[1] fit subspace on train HS @ L16,15,14
[2] Stage A  — center-steer on val; shortlist
[3] Stage B  — MMLU domain_val (union member SOCs); freeze keep
[4] Stage C  — preference on test + MMLU domain_test
```

### Commands (2B)

```powershell
# sample + split manifest (CPU)
python -m steering.build_polarity_pool_inlp_sample --set-id promale
python -m steering.build_polarity_pool_inlp_sample --set-id profemale

# Phase A: live INLP fit (GPU)
python -m steering.build_polarity_pool_inlp_subspace `
  --set-id promale --device cuda --dtype float32

# Stage A
python -m steering.run_polarity_pool_inlp_stagea `
  --set-id promale --device cuda --dtype float32

# Stage B / C (reuse classical runners + polarity sample/subspace)
python -m steering.run_polarity_pool_inlp_stageb --set-id promale --device cuda
python -m steering.run_polarity_pool_inlp_stagec --set-id promale --device cuda
```

Vast (both sets A→B→C):

```bash
export HF_TOKEN=hf_xxx
BRANCH=qwen_2b_experiments bash steering/scripts/vast_inlp_polarity_pool_full_instance.sh
```

Copy-paste / env knobs: [`INLP_POLARITY_POOL_VAST.md`](INLP_POLARITY_POOL_VAST.md).

Smoke: `SMOKE=1` (anchor layer only, rank 4, few families).

---

## 5. Metrics (report both)

Primary selection (Stage A): family-level \(R\) on gender axis
(\(|D|_{\mathrm{before}} - |D|_{\mathrm{after}}\)), bootstrap CI, beat random.

**Always report on Stage C:**

1. **Signed** P(man)% = \(100\cdot\sigma(\mathrm{GenderGap})\) — mean direction.
2. **Absolute** |P(man)−50| pp (family mean) — strength of preference.

XY pooled male showed signed→neutral with |bias|↑; INLP must be judged on
the same pair of numbers. Bonferroni **m=2** across the two polarity sets
(pre-registered).

Capability: `cap_loss ≤ 0.03` on MMLU-Pro domain profile (union of member
SOC titles), same gate spirit as XY Stage B.

---

## 6. Outputs

```
steering/samples/inlp_polarity_pool_<set>_v1.json
steering/samples/inlp_polarity_pool_<set>_{train,val,test}_v1.json
steering/subspaces/inlp_gender_prob_polarity_pool_<set>_v1.{npz,json}

results/steering/inlp_polarity_pool/inlp_stage_a/<tag_a>/
results/steering/inlp_polarity_pool/inlp_stage_b/<tag_b>/
results/steering/inlp_polarity_pool/inlp_stage_c/<tag_c>/
```

Pack names (Vast): `inlp_polarity_pool_{promale,profemale}_stage_abc_2b.tar.gz`

---

## 7. Relation to global INLP

Global `inlp_gender_prob` (all SOCs) remains available as appendix /
diagnostic. **Primary multi-model arm** is this polarity-pooled protocol,
parallel to XY polarity pool:

```
beh + HS-probe  →  INLP polarity pool  →  XY polarity pool
```

(order of interventions is conceptual; runs are independent on the same
splits.)

---

## 8. 4B / other scales

Copy the pattern: set `probe_peak` from that scale’s gender_prob config,
layers = `{peak, peak−1, peak−2}`, reuse FDR polarity membership for that
scale’s catalog, keep split seed/rule fixed.

### 4B frozen artifacts

| | path |
|---|---|
| Config | [`configs/inlp_polarity_pool_4b_peak_prepeak.yaml`](configs/inlp_polarity_pool_4b_peak_prepeak.yaml) |
| Sets | [`domains/inlp_polarity_sets_4b_v1.json`](domains/inlp_polarity_sets_4b_v1.json) |
| XY twin | [`xy_control/domains/polarity_sets_4b_v1.json`](xy_control/domains/polarity_sets_4b_v1.json) |
| Layers | **L24 / L23 / L22** (gender_prob peak L24) |
| SOCs | promale **4**, profemale **6** (4B FDR) |

Samples / subspaces use `*_4b_*` stems so 2B artifacts are not overwritten.

Vast:

```bash
export HF_TOKEN=hf_xxx
BRANCH=qwen_2b_experiments \
  bash steering/scripts/vast_inlp_polarity_pool_4b_full_instance.sh
```

Packs: `/workspace/inlp_polarity_pool_{promale,profemale}_stage_abc_4b.tar.gz`
