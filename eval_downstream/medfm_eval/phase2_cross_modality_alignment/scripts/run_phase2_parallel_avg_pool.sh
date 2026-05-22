#!/bin/bash
# Phase 2 Parallel Evaluation - AVG_POOL feature family.
#
# Thin wrapper over run_phase2_parallel_cls.sh that keeps the existing
# checkpoint scheduling and cache reuse behavior while selecting avg_pool.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PHASE2_FEATURE_TYPE=avg_pool

bash "$ROOT_DIR/scripts/run_phase2_parallel_cls.sh" "$@"

# GPU_IDS="0 1 2" PHASE2_PRESET=mmwhs_ct_mr_core nohup ./scripts/run_phase2_parallel_avg_pool.sh > outputs_phase2/mmwhs_ct_mr/phase2/core/nohup_parallel_avg_pool.log 2>&1 &
# GPU_IDS="0 2" PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core nohup ./scripts/run_phase2_parallel_avg_pool.sh > outputs_phase2/totalsegmentermri/phase2/core/nohup_parallel_avg_pool.log 2>&1 &


