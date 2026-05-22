#!/bin/bash
# Run the full Phase 2 checkpoint slate for the MMWHS validation surface.
#
# Usage:
#   ./scripts/run_phase2_full_checkpoint_validation_mmwhs.sh
#   GPU_IDS="0 1 2 3" ./scripts/run_phase2_full_checkpoint_validation_mmwhs.sh
#
# Examples:
#   nohup ./scripts/run_phase2_full_checkpoint_validation_mmwhs.sh \
#     > outputs_phase2/mmwhs_ct_mr/phase2/core/logs/mmwhs_full_checkpoint_validation.log 2>&1 &

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export PHASE2_PRESET=mmwhs_ct_mr_core
unset PHASE2_CHECKPOINTS
PHASE2_FEATURE_TYPE=${PHASE2_FEATURE_TYPE:-cls}

echo "[Phase2] Validation surface : mmwhs_ct_mr"
echo "[Phase2] Checkpoint slate   : full canonical slate"
echo "[Phase2] Feature family     : ${PHASE2_FEATURE_TYPE}"
echo "[Phase2] GPU IDs            : ${GPU_IDS:-0 1 2 3}"

bash "$ROOT_DIR/scripts/run_phase2_parallel_cls.sh" "$@"


# nohup ./scripts/run_phase2_full_checkpoint_validation_mmwhs.sh > outputs_phase2/mmwhs_ct_mr/phase2/core/logs/mmwhs_full_checkpoint_validation.log 2>&1 &
# GPU_IDS="3" PHASE2_BATCH_SIZE=4 nohup ./scripts/run_phase2_full_checkpoint_validation_mmwhs.sh > outputs_phase2/mmwhs_ct_mr/phase2/core/logs/mmwhs_full_checkpoint_validation.log 2>&1 &



