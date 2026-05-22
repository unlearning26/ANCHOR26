#!/usr/bin/env bash
set -euo pipefail

# End-to-end launcher for one checkpoint and one feature family.
#
# Required variables:
#   CHECKPOINT_ROOT=/path/to/checkpoints
#   PHASE1_MANIFEST=/path/to/phase1/manifest_sampled.json
#   PHASE2_MANIFEST=/path/to/phase2/manifest_sampled.json
#
# Minimal example:
#   CHECKPOINT_ROOT=/path/to/checkpoints \
#   PHASE1_MANIFEST=/path/to/phase1/manifest_sampled.json \
#   PHASE2_MANIFEST=/path/to/phase2/manifest_sampled.json \
#   bash eval_downstream/medfm_eval/cmd.sh
# 
# Notes:
# - CHECKPOINT_ROOT must be the directory that contains chkpts/, not chkpts/ itself.
# - Defaults: GPU=0, CHECKPOINT_NAME=Med3DINO_SA_c96, FEATURE_TYPE=cls,
#   PHASE1_ANALYSIS=totalsegmentermri, PHASE2_ANALYSIS=amos_ct_mr, PHASE2_VARIANT=core.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${CMD_ENV_FILE:-$SCRIPT_DIR/cmd.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

GPU="${GPU:-0}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-Med3DINO_SA_c96}" # Med3DINO_REL_c96
FEATURE_TYPE="${FEATURE_TYPE:-cls}"

require_env() {
  local name="$1"
  local hint="$2"

  if [[ -n "${!name:-}" ]]; then
    return 0
  fi

  cat >&2 <<EOF
Missing required variable: $name
$hint

Example:
  CHECKPOINT_ROOT=/path/to/checkpoints \\
  PHASE1_MANIFEST=/path/to/phase1/manifest_sampled.json \\
  PHASE2_MANIFEST=/path/to/phase2/manifest_sampled.json \\
  bash eval_downstream/medfm_eval/cmd.sh
EOF
  exit 1
}

require_env "CHECKPOINT_ROOT" "Set CHECKPOINT_ROOT to the directory that contains chkpts/."
require_env "PHASE1_MANIFEST" "Set PHASE1_MANIFEST to the Phase 1 manifest path."
require_env "PHASE2_MANIFEST" "Set PHASE2_MANIFEST to the Phase 2 manifest path."

PHASE1_ANALYSIS="${PHASE1_ANALYSIS:-totalsegmentermri}"

PHASE2_ANALYSIS="${PHASE2_ANALYSIS:-amos_ct_mr}"
PHASE2_VARIANT="${PHASE2_VARIANT:-core}"
PHASE2_FEATURE_NPZ="${PHASE2_FEATURE_NPZ:-$ROOT/eval_downstream/medfm_eval/phase2_cross_modality_alignment/outputs_phase2/$PHASE2_ANALYSIS/phase2/$PHASE2_VARIANT/features/$CHECKPOINT_NAME/$FEATURE_TYPE/phase2_organ_cls_embeddings.npz}"

MODULE3_MANIFEST_DIR="${MODULE3_MANIFEST_DIR:-$ROOT/eval_downstream/medfm_eval/data_manifests/module3_anatomical_generalization/$PHASE2_ANALYSIS/$PHASE2_VARIANT}"
MODULE3_MANIFEST="${MODULE3_MANIFEST:-$MODULE3_MANIFEST_DIR/manifest_sampled.json}"

cd "$ROOT"

echo "[1/5] Phase 1 full evaluation"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" eval_downstream/medfm_eval/phase1_anisotropy_robustness/phase1_evaluation_pipeline.py \
  --full \
  -m "$PHASE1_MANIFEST" \
  -a "$PHASE1_ANALYSIS" \
  -c "$CHECKPOINT_NAME" \
  -f "$FEATURE_TYPE" \
  --batch-size 4 \
  --num-workers 4 \
  --checkpoint-root "$CHECKPOINT_ROOT"

echo "[2/5] Phase 2 organ feature extraction"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" eval_downstream/medfm_eval/phase2_cross_modality_alignment/phase2_organ_feature_extractor.py \
  -m "$PHASE2_MANIFEST" \
  -c "$CHECKPOINT_NAME" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --feature-type "$FEATURE_TYPE" \
  --batch-size 4 \
  --crop-cache-workers 4 \
  --device cuda \
  -o "$PHASE2_FEATURE_NPZ"

echo "[3/5] Phase 2 primary metrics"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" eval_downstream/medfm_eval/phase2_cross_modality_alignment/phase2_evaluation_pipeline.py \
  -m "$PHASE2_MANIFEST" \
  -a "$PHASE2_ANALYSIS" \
  --manifest-variant "$PHASE2_VARIANT" \
  --embeddings-npz "$PHASE2_FEATURE_NPZ" \
  --checkpoint-name "$CHECKPOINT_NAME" \
  --feature-type "$FEATURE_TYPE" \
  --required-modalities ct mr \
  --min-samples-per-modality 5

echo "[4/5] Module 3 manifest build"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" eval_downstream/medfm_eval/module3_anatomical_generalization/build_module3_manifest.py \
  -m "$PHASE2_MANIFEST" \
  -a "$PHASE2_ANALYSIS" \
  --manifest-variant "$PHASE2_VARIANT" \
  --required-modalities ct mr \
  --min-holdout-organs 2 \
  --min-samples-per-modality 5 \
  --output-dir "$MODULE3_MANIFEST_DIR"

echo "[5/5] Module 3 evaluation"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" eval_downstream/medfm_eval/module3_anatomical_generalization/module3_evaluation_pipeline.py \
  -m "$MODULE3_MANIFEST" \
  -a "$PHASE2_ANALYSIS" \
  --manifest-variant "$PHASE2_VARIANT" \
  --embeddings-npz "$PHASE2_FEATURE_NPZ" \
  --checkpoint-name "$CHECKPOINT_NAME" \
  --feature-type "$FEATURE_TYPE" \
  --required-modalities ct mr \
  --min-holdout-organs 2 \
  --min-samples-per-modality 5


# Example:
# cd /path/to/ANCHOR26 && nohup env \
#   CMD_ENV_FILE=/dev/null \
#   CHECKPOINT_ROOT=path/to/eval_downstream/checkpoints \
#   PHASE1_MANIFEST=path/to/data_manifests/phase1_anisotropy_robustness/totalsegmentermri/original_bins/manifest_sampled.json \
#   PHASE2_MANIFEST=path/to/data_manifests/phase2_cross_modality_alignment/amos_ct_mr/core/manifest_sampled.json \
#   PHASE1_ANALYSIS=totalsegmentermri \
#   PHASE2_ANALYSIS=amos_ct_mr \
#   PHASE2_VARIANT=core \
#   CHECKPOINT_NAME=Med3DINO_SA_c96 \
#   FEATURE_TYPE=cls \
#   bash eval_downstream/medfm_eval/cmd.sh \
#   > eval_downstream/medfm_eval/cmd_totalsegmentermri_amos_ct_mr_sa_c96_cls.nohup.log 2>&1 < /dev/null &


