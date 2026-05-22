#!/usr/bin/env bash

# Usage:
# nohup nice -n 10 ./cache_generator.sh > ../caches/cache_generation_all_datasets.log 2>&1 &

set -euo pipefail

PHASE1="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$PHASE1/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"

cd "$PHASE1"

for dataset in \
  abdomenct1k \
  totalsegmentermri \
  kits23 \
  cirrmri600 \
  duke_liver \
  hecktor25 \
  imagecas \
  abdomenatlas \
  jhu_stroke \
  pansegdata \
  totalsegmenter_ct
do
  echo "=== Precomputing perturbation cache for $dataset ==="
  "$PYTHON_BIN" build_perturbation_cache.py \
    -m "../data_manifests/phase1_anisotropy_robustness/$dataset/original_bins/manifest_sampled.json" \
    -a "$dataset" \
    --crop-size 96 112
done