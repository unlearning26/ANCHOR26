#!/bin/bash
# Phase 1 Parallel Evaluation - multilayer features (Setting A ONLY)
# Distributes checkpoints across 4 A100 GPUs
#
# This runs Setting A (Track A/B analysis, cross-bin transfer) but SKIPS Setting B
# For Setting B (controlled perturbation), run: ./run_perturbation_parallel_multilayer.sh
# Usage:
#   PHASE1_PRESET=abdomenatlas_core ./run_phase1_parallel_multilayer.sh
#   PHASE1_PRESET=totalsegmentermri_core ./run_phase1_parallel_multilayer.sh
#   ANALYSIS_NAME=my_dataset MANIFEST=../../data_manifests/phase1_anisotropy_robustness/abdomenatlas/original_bins/manifest_sampled.json ./run_phase1_parallel_multilayer.sh

print_usage() {
    sed -n '2,8p' "$0"
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    print_usage
    exit 0
fi

set -e
cd "$(dirname "$0")"
ROOT_DIR="$(cd .. && pwd)"
PYTHON_BIN="${PYTHON:-python}"
source ./phase1_dataset_presets.sh

# ========== CONFIGURATION ==========
if [ -n "${ANALYSIS_NAME:-}" ] || [ -n "${MANIFEST:-}" ]; then
    if [ -z "${ANALYSIS_NAME:-}" ] || [ -z "${MANIFEST:-}" ]; then
        echo "ERROR: Set both ANALYSIS_NAME and MANIFEST, or neither." >&2
        exit 1
    fi
    PRESET_LABEL="manual"
else
    PHASE1_PRESET="${PHASE1_PRESET:-abdomenatlas_core}"
    resolve_phase1_preset "$PHASE1_PRESET"
    if ! phase1_preset_supports "full_phase1"; then
        echo "ERROR: Preset '$PHASE1_PRESET' is not valid for Setting A / full Phase 1 runs." >&2
        echo "Use a perturbation launcher for controlled-only presets." >&2
        exit 1
    fi
    ANALYSIS_NAME="$PHASE1_PRESET_ANALYSIS_NAME"
    MANIFEST="$PHASE1_PRESET_MANIFEST"
    PRESET_LABEL="$PHASE1_PRESET"
fi
FEATURE_TYPE="multilayer"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MANIFEST_VARIANT="$(phase1_manifest_variant_from_manifest "$MANIFEST")"
OUTPUT_ROOT="$ROOT_DIR/outputs_phase1/${ANALYSIS_NAME}/phase1/${MANIFEST_VARIANT}"
# ===================================

LOGDIR="$OUTPUT_ROOT/logs/${FEATURE_TYPE}"
mkdir -p "$LOGDIR"

run_gpu() {
    local gpu=$1
    shift
    for ckpt in "$@"; do
        echo "[GPU $gpu] Starting $ckpt ($FEATURE_TYPE)"
        CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" "$ROOT_DIR/phase1_evaluation_pipeline.py" --full --no-setting-b \
            -m "$MANIFEST" -a "$ANALYSIS_NAME" \
            -c "$ckpt" -f "$FEATURE_TYPE" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
            > "$LOGDIR/${ckpt}.log" 2>&1
        echo "[GPU $gpu] Finished $ckpt"
    done
}

echo "=== Phase 1 Parallel Evaluation ($FEATURE_TYPE) ==="
echo "Preset: $PRESET_LABEL"
echo "Analysis: $ANALYSIS_NAME"
echo "Manifest: $MANIFEST"
echo "Logs: $LOGDIR"
echo "Starting at $(date)"

echo "Pre-computing semantic label cache..."
"$PYTHON_BIN" "$ROOT_DIR/phase1_evaluation_pipeline.py" \
    --precompute-labels \
    -m "$MANIFEST" \
    -a "$ANALYSIS_NAME"
echo "Semantic label cache ready."

# Launch 4 GPU streams in parallel
pids=()
run_gpu 0 Med3DINO_REL_c96 Med3DINO_REL_c112 Med3DINO_ISO_c96 &
pids+=("$!")
run_gpu 1 Med3DINO_SA_c96 Med3DINO_SA_c112 Med3DINO_ISO_c112 &
pids+=("$!")
run_gpu 2 Med3DINO_Base_c96 Med3DINO_Base_c112 &
pids+=("$!")
run_gpu 3 3dinov2 &
pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

if [ "$failed" -ne 0 ]; then
    echo "=== One or more checkpoints failed. See logs in $LOGDIR ===" >&2
    exit 1
fi

echo "=== All checkpoints completed at $(date) ==="
