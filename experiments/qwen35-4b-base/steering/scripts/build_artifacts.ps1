# Rebuild 4B steering artifacts (Windows)
$ErrorActionPreference = "Stop"
$STEER = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EXP = (Resolve-Path (Join-Path $STEER "..")).Path
$PACK = if ($env:PACK) { $env:PACK } else { Join-Path $EXP "results\qwen35_4b_h1h3_pack" }
$REPO = if ($env:REPO) { $env:REPO } else {
  (Resolve-Path (Join-Path $EXP "..\..\..\repo")).Path
}

$env:PYTHONPATH = $REPO
Set-Location $REPO
New-Item -ItemType Directory -Force -Path "$STEER\candidates","$STEER\vectors","$STEER\subspaces","$STEER\samples" | Out-Null

Write-Host "REPO=$REPO"
Write-Host "STEER=$STEER"
Write-Host "PACK=$PACK"

python -m steering.build_h1_candidates --config "$STEER\configs\h1_steering_candidates_v1.yaml" --out-dir "$STEER\candidates" --out-prefix h1_candidates
python -m steering.build_h1_candidates --config "$STEER\configs\slot_steering_candidates_v1.yaml" --out-dir "$STEER\candidates" --out-prefix slot_candidates
python -m steering.build_h1_vectors --config "$STEER\configs\h1_steering_candidates_v1.yaml" --results-root $PACK --out-dir "$STEER\vectors" --out-prefix h1_vectors
python -m steering.build_h1_vectors --config "$STEER\configs\slot_steering_candidates_v1.yaml" --results-root $PACK --out-dir "$STEER\vectors" --out-prefix slot_vectors

if ($env:SKIP_INLP -ne "1") {
  python -m steering.build_inlp_subspace --config "$STEER\configs\inlp_gender_v1.yaml" --results-root $PACK --out-dir "$STEER\subspaces"
  python -m steering.build_inlp_subspace --config "$STEER\configs\inlp_slot_v1.yaml" --results-root $PACK --out-dir "$STEER\subspaces"
}

Write-Host "DONE artifacts under $STEER"
