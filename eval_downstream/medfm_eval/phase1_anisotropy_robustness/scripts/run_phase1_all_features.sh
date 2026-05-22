#!/bin/bash
# Phase 1 Unified Evaluation - All Feature Types (Setting A)
#
# Runs cls → avg_pool → multilayer SEQUENTIALLY, each with 4 parallel GPU streams.
# Max 4 concurrent processes (not 12), which:
#   - Eliminates semantic label cache thundering herd
#   - Prevents DataLoader worker OOM kills from memory pressure
#
# Pre-computes the semantic label cache once before any GPU work.
#
# Usage:
#   PHASE1_PRESET=abdomenatlas_core ./run_phase1_all_features.sh
#   PHASE1_PRESET=abdomenatlas_coarse ./run_phase1_all_features.sh
#   PHASE1_PRESET=totalsegmentermri_core ./run_phase1_all_features.sh
#   PHASE1_PRESET=abdomenct1k_coarse nohup ./run_phase1_all_features.sh > ../outputs_phase1/run_all_features_abdomenct1k_coarse.log 2>&1 &

print_usage() {
    sed -n '2,15p' "$0"
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

BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MANIFEST_VARIANT="$(phase1_manifest_variant_from_manifest "$MANIFEST")"
OUTPUT_ROOT="$ROOT_DIR/outputs_phase1/${ANALYSIS_NAME}/phase1/${MANIFEST_VARIANT}"
FEATURE_TYPES="${FEATURE_TYPES:-cls avg_pool multilayer}"
# ===================================

echo "================================================================="
echo "Phase 1 Unified Evaluation (Setting A) - All Feature Types"
echo "================================================================="
echo "Preset:       $PRESET_LABEL"
echo "Analysis:     $ANALYSIS_NAME"
echo "Manifest:     $MANIFEST"
echo "Feature types: $FEATURE_TYPES"
echo "Batch size:   $BATCH_SIZE"
echo "Num workers:  $NUM_WORKERS"
echo "Output root:  $OUTPUT_ROOT"
echo "Started at:   $(date)"
echo "================================================================="

# ------------------------------------------------------------------
# Step 0: Pre-compute semantic label cache (single process, no GPU)
# ------------------------------------------------------------------
echo ""
echo "[$(date '+%H:%M:%S')] Pre-computing semantic label cache..."
"$PYTHON_BIN" "$ROOT_DIR/phase1_evaluation_pipeline.py" \
    --precompute-labels \
    -m "$MANIFEST" \
    -a "$ANALYSIS_NAME"
echo "[$(date '+%H:%M:%S')] Semantic label cache ready."

# ------------------------------------------------------------------
# Helper: run one feature type across 4 GPUs
# ------------------------------------------------------------------
run_feature_type() {
    local ft="$1"
    local logdir="$OUTPUT_ROOT/logs/${ft}"
    mkdir -p "$logdir"

    echo ""
    echo "================================================================="
    echo "[$(date '+%H:%M:%S')] Starting feature type: $ft"
    echo "================================================================="

    run_gpu() {
        local gpu=$1
        shift
        for ckpt in "$@"; do
            echo "[$(date '+%H:%M:%S')] GPU $gpu: $ckpt ($ft)"
            CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" "$ROOT_DIR/phase1_evaluation_pipeline.py" \
                --full --no-setting-b \
                -m "$MANIFEST" -a "$ANALYSIS_NAME" \
                -c "$ckpt" -f "$ft" \
                --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
                > "$logdir/${ckpt}.log" 2>&1
            echo "[$(date '+%H:%M:%S')] GPU $gpu: $ckpt ($ft) done"
        done
    }

    local pids=()

    # Distribute checkpoints across 4 GPUs
    run_gpu 0 Med3DINO_REL_c96 Med3DINO_REL_c112 Med3DINO_ISO_c96 &
    pids+=("$!")
    run_gpu 1 Med3DINO_SA_c96 Med3DINO_SA_c112 Med3DINO_ISO_c112 &
    pids+=("$!")
    run_gpu 2 Med3DINO_Base_c96 Med3DINO_Base_c112 &
    pids+=("$!")
    run_gpu 3 3dinov2 &
    pids+=("$!")

    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done

    if [ "$failed" -ne 0 ]; then
        echo "[$(date '+%H:%M:%S')] Feature type $ft failed. Check logs in $logdir" >&2
        exit 1
    fi

    echo "[$(date '+%H:%M:%S')] Feature type $ft completed."
}

# ------------------------------------------------------------------
# Step 1-3: Run each feature type sequentially (4 GPUs each)
# ------------------------------------------------------------------
for ft in $FEATURE_TYPES; do
    run_feature_type "$ft"
done

echo ""
echo "================================================================="
echo "All feature types completed at $(date)"
echo "================================================================="