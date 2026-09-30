# Ministral 3 8B Base — steering results layout

Canonical outputs after Vast packs land here.

```
results/steering/
  inlp_stage_{a,b,c}/          # classical gender + slot
  inlp_polarity_pool/          # pool INLP
  xy_control/                  # pool XY A→B→C
steering/subspaces/            # W npz+json (often mirrored to repo steering/)
steering/samples/              # polarity-pool family splits
```

| Protocol | Status |
|----------|--------|
| Classical INLP | pending (peaks TBD) |
| Pool INLP | pending (FDR stub empty) |
| Pool XY | pending (FDR stub empty) |

Dtype: **bfloat16**.
