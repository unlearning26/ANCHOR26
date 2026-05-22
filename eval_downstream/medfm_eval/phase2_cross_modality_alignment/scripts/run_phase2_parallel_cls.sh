#!/bin/bash
# Phase 2 Parallel Evaluation - default CLS surface
# Distributes the full checkpoint slate across a configurable GPU set.
#
# This launcher mirrors the Phase 1 parallel pattern: each GPU processes an
# assigned subset of checkpoints serially, while GPU groups run in parallel.
#
# Usage:
#   PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core ./run_phase2_parallel_cls.sh
#   PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core GPU_IDS="0 1 2 3" ./run_phase2_parallel_cls.sh
#
# Examples:
#   PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core ./run_phase2_parallel_cls.sh
#
#   PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core \
#   GPU_IDS="0 1 2 3" \
#   PHASE2_CHECKPOINTS="Med3DINO_REL_c96,Med3DINO_SA_c96,3dinov2" \
#   ./run_phase2_parallel_cls.sh

print_usage() {
    sed -n '2,7p' "$0"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    print_usage
    exit 0
fi

set -euo pipefail

cd "$(dirname "$0")"
ROOT_DIR="$(cd .. && pwd)"
PYTHON_BIN="${PYTHON:-python}"
source "$ROOT_DIR/scripts/phase2_dataset_presets.sh"
phase2_init_namespace
PHASE2_FEATURE_TYPE="$(phase2_normalize_feature_type "${PHASE2_FEATURE_TYPE:-cls}")"
PHASE2_SKIP_EVALUATION="${PHASE2_SKIP_EVALUATION:-${PHASE2_SKIP_EVALUATION_DEFAULT:-0}}"

if [[ -n "${GPU_DIS:-}" && -z "${GPU_IDS:-}" ]]; then
    echo "WARNING: GPU_DIS is deprecated; using it as an alias for GPU_IDS." >&2
    GPU_IDS="$GPU_DIS"
elif [[ -n "${GPU_DIS:-}" && -n "${GPU_IDS:-}" ]]; then
    echo "WARNING: Both GPU_DIS and GPU_IDS are set; using GPU_IDS and ignoring GPU_DIS." >&2
fi

GPU_IDS_RAW="${GPU_IDS:-0 1 2 3}"
GPU_IDS_RAW="${GPU_IDS_RAW//,/ }"
read -r -a GPU_ID_LIST <<< "$GPU_IDS_RAW"
if [[ "${#GPU_ID_LIST[@]}" -eq 0 ]]; then
    echo "ERROR: GPU_IDS must contain at least one GPU id." >&2
    exit 1
fi

MANIFEST_VARIANT="$(phase2_manifest_variant_from_manifest "$PHASE2_MANIFEST")"
OUTPUT_ROOT="$(phase2_resolve_output_root "$ROOT_DIR" "$PHASE2_ANALYSIS_NAME" "$MANIFEST_VARIANT")"
PHASE2_SKIP_EXTRACTION="${PHASE2_SKIP_EXTRACTION:-0}"
if [[ -n "${PHASE2_SKIP_AGGREGATION:-}" ]]; then
    PHASE2_SKIP_AGGREGATION="${PHASE2_SKIP_AGGREGATION}"
elif [[ "$PHASE2_SKIP_EVALUATION" == "1" ]]; then
    PHASE2_SKIP_AGGREGATION=1
else
    PHASE2_SKIP_AGGREGATION=0
fi
LOG_SUFFIX="$PHASE2_FEATURE_TYPE"
if [[ "$PHASE2_SKIP_EXTRACTION" == "1" ]]; then
    if [[ "$PHASE2_FEATURE_TYPE" == "cls" ]]; then
        LOGDIR="$OUTPUT_ROOT/logs/parallel_cls_eval_only"
    else
        LOGDIR="$OUTPUT_ROOT/logs/parallel_${LOG_SUFFIX}_eval_only"
    fi
    RUN_MODE_LABEL="eval_only"
else
    if [[ "$PHASE2_FEATURE_TYPE" == "cls" ]]; then
        LOGDIR="$OUTPUT_ROOT/logs/parallel_cls"
    else
        LOGDIR="$OUTPUT_ROOT/logs/parallel_${LOG_SUFFIX}"
    fi
    RUN_MODE_LABEL="extract_and_eval"
fi
if [[ "$PHASE2_SKIP_EVALUATION" == "1" && "$PHASE2_SKIP_EXTRACTION" != "1" ]]; then
    if [[ "$PHASE2_FEATURE_TYPE" == "cls" ]]; then
        LOGDIR="$OUTPUT_ROOT/logs/parallel_cls_extract_only"
    else
        LOGDIR="$OUTPUT_ROOT/logs/parallel_${LOG_SUFFIX}_extract_only"
    fi
    RUN_MODE_LABEL="extract_only"
fi
mkdir -p "$LOGDIR"

mapfile -t CHECKPOINTS < <(phase2_all_checkpoints)
if [[ -n "${PHASE2_CHECKPOINTS:-}" ]]; then
    mapfile -t CHECKPOINTS < <(tr ', ' '\n\n' <<< "$PHASE2_CHECKPOINTS" | sed '/^$/d')
fi

if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
    echo "ERROR: No Phase 2 checkpoints selected." >&2
    exit 1
fi

declare -A seen_crop_sizes=()
declare -a crop_sizes=()
for ckpt in "${CHECKPOINTS[@]}"; do
    crop_size="$(phase2_checkpoint_crop_size "$ckpt")"
    if [[ -z "${seen_crop_sizes[$crop_size]:-}" ]]; then
        seen_crop_sizes[$crop_size]=1
        crop_sizes+=("$crop_size")
    fi
done

declare -a gpu_assignments
for i in "${!GPU_ID_LIST[@]}"; do
    gpu_assignments[$i]=""
done

for i in "${!CHECKPOINTS[@]}"; do
    slot=$(( i % ${#GPU_ID_LIST[@]} ))
    gpu_assignments[$slot]="${gpu_assignments[$slot]} ${CHECKPOINTS[$i]}"
done

run_gpu_group() {
    local gpu_id=$1
    shift
    local checkpoints=("$@")

    for ckpt in "${checkpoints[@]}"; do
        [[ -z "$ckpt" ]] && continue
        echo "[$(date '+%H:%M:%S')] GPU $gpu_id: Starting $ckpt"
        CUDA_VISIBLE_DEVICES="$gpu_id" \
        PHASE2_CHECKPOINT="$ckpt" \
        PHASE2_FEATURE_TYPE="$PHASE2_FEATURE_TYPE" \
        PHASE2_SKIP_EVALUATION="$PHASE2_SKIP_EVALUATION" \
        PHASE2_DEVICE=cuda \
        bash "$ROOT_DIR/scripts/run_phase2_extract_and_eval.sh" \
            > "$LOGDIR/${ckpt}.log" 2>&1
        echo "[$(date '+%H:%M:%S')] GPU $gpu_id: Finished $ckpt"
    done
}

echo "=== Phase 2 Parallel ${PHASE2_FEATURE_TYPE} Evaluation ==="
echo "Analysis: $PHASE2_ANALYSIS_NAME"
echo "Manifest: $PHASE2_MANIFEST"
echo "Variant : $MANIFEST_VARIANT"
echo "Feature : $PHASE2_FEATURE_TYPE"
echo "Run mode: $RUN_MODE_LABEL"
echo "Skip eval: $PHASE2_SKIP_EVALUATION"
echo "GPU IDs : ${GPU_ID_LIST[*]}"
echo "Checkpoints: ${CHECKPOINTS[*]}"
echo "Crop sizes : ${crop_sizes[*]}"
echo "Logs: $LOGDIR"
echo "Starting at $(date)"
echo ""

if [[ "$PHASE2_SKIP_EXTRACTION" == "1" ]]; then
    echo "Skipping crop-cache preparation because PHASE2_SKIP_EXTRACTION=1"
else
    for crop_size in "${crop_sizes[@]}"; do
        echo "Preparing crop cache for crop${crop_size}"
        PHASE2_CROP_SIZE="$crop_size" \
        PHASE2_CROP_CACHE_WORKERS="${PHASE2_CROP_CACHE_WORKERS:-4}" \
        bash "$ROOT_DIR/scripts/run_phase2_build_crop_cache.sh"
    done
fi

echo ""

pids=()
for i in "${!GPU_ID_LIST[@]}"; do
    assignment="${gpu_assignments[$i]}"
    if [[ -z "${assignment// }" ]]; then
        continue
    fi
    read -r -a ckpt_group <<< "$assignment"
    echo "GPU ${GPU_ID_LIST[$i]} checkpoints: ${ckpt_group[*]}"
    run_gpu_group "${GPU_ID_LIST[$i]}" "${ckpt_group[@]}" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

if [[ "$failed" -ne 0 ]]; then
    echo "=== One or more Phase 2 checkpoints failed. See logs in $LOGDIR ===" >&2
    exit 1
fi

if [[ "$PHASE2_SKIP_AGGREGATION" == "1" ]]; then
    echo "Skipping aggregated comparison refresh because PHASE2_SKIP_AGGREGATION=1"
else
    echo "Refreshing aggregated checkpoint comparison artifacts"
    PHASE2_FEATURE_TYPE="$PHASE2_FEATURE_TYPE" bash "$ROOT_DIR/scripts/run_phase2_aggregate_results.sh"
fi

echo ""
echo "=== All Phase 2 ${PHASE2_FEATURE_TYPE} checkpoints completed at $(date) ==="
