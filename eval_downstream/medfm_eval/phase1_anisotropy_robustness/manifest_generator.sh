#!/usr/bin/env bash
set -euo pipefail

PHASE1="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"

cd "$PHASE1"

build_manifest() {
  local dataset=$1
  local variant=${2:-original}

  echo "=== Building manifest for ${dataset} (${variant}) ==="
  "$PYTHON_BIN" build_phase1_manifest.py \
    --dataset "$dataset" \
    --output-suffix "$dataset" \
    --binning-scheme "$variant"
}

# Original-bin manifests used by the active Phase 1 surface.
for dataset in \
  abdomenatlas \
  abdomenct1k \
  cirrmri600 \
  duke_liver \
  hecktor25 \
  imagecas \
  jhu_stroke \
  kits23 \
  pansegdata \
  totalsegmenter_ct \
  totalsegmentermri
do
  build_manifest "$dataset" original
done

# # Coarse-bin variants that still have launcher presets.
# for dataset in \
#   abdomenatlas \
#   abdomenct1k
# do
#   build_manifest "$dataset" coarse_bins
# done

echo "=== Phase 1 manifest regeneration complete ==="