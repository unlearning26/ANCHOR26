#!/usr/bin/env python

"""Run Module 3 anatomical generalization over finished upstream organ feature bundles.

This script is the canonical Module 3 evaluator. It supports:

    - single-bundle mode: evaluate one explicit Phase 2 NPZ bundle
    - batch mode: discover all available bundles for a dataset surface and write
      aggregate summaries

Required parameters:
    Single-bundle mode:
        --manifest or --analysis-name
        --checkpoint-name
        --embeddings-npz

    Batch mode:
        --manifest or --analysis-name

Examples:
    python module3_evaluation_pipeline.py \
        --manifest ../data_manifests/module3_anatomical_generalization/totalsegmenter_ct_mr_anchor/core/manifest_sampled.json \
        --analysis-name totalsegmenter_ct_mr_anchor \
        --checkpoint-name Med3DINO_REL_c96 \
        --feature-type cls \
        --embeddings-npz ../phase2_cross_modality_alignment/outputs_phase2/totalsegmenter_ct_mr_anchor/phase2/core/features/Med3DINO_REL_c96/cls/phase2_organ_cls_embeddings.npz

    python module3_evaluation_pipeline.py \
        --analysis-name totalsegmenter_ct_mr_anchor \
        --manifest-variant core \
        --feature-types cls avg_pool
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from config import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_CHECKPOINT_POLICY,
    DEFAULT_FEW_SHOT_QUERY_PER_MODALITY,
    DEFAULT_FEW_SHOT_SEEDS,
    DEFAULT_FEW_SHOT_SUPPORT_PER_MODALITY,
    DEFAULT_MIN_HOLDOUT_ORGANS,
    DEFAULT_MIN_SAMPLES_PER_MODALITY,
    DEFAULT_REQUIRED_MODALITIES,
    FEATURE_TYPES,
    SUMMARY_CSV_NAME,
    SUMMARY_JSON_NAME,
    ensure_output_directories,
    get_checkpoint_metrics_path,
    get_dataset_name_from_manifest_path,
    get_manifest_variant_from_manifest_path,
    get_output_paths,
    get_phase2_feature_npz_path,
    get_module3_manifest_path,
    get_summary_csv_path,
    get_summary_json_path,
    normalize_checkpoint_name,
    normalize_feature_type,
    normalize_manifest_variant,
)
from few_shot_transfer_analysis import evaluate_leave_one_organ_out_few_shot_transfer
from holdout_generalization_analysis import evaluate_leave_one_organ_out_transfer
from module3_data_loader import (
    get_eligible_holdout_organs,
    load_module3_manifest,
    summarize_module3_manifest,
    validate_module3_manifest,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MANIFEST_SUMMARY_NAME = "module3_manifest_summary.json"


def _build_surface_a_metric_aliases(analysis: Dict[str, Any]) -> Dict[str, Any]:
    macro = analysis.get("macro_mean", {})
    return {
        "top1_matching_accuracy": macro.get("top@1"),
    }


def _build_surface_b_metric_aliases(analysis: Dict[str, Any]) -> Dict[str, Any]:
    macro = analysis.get("macro_mean", {})
    return {
        "balanced_accuracy": macro.get("balanced_accuracy"),
        "nearest_centroid_baseline_balanced_accuracy": macro.get("nearest_centroid_balanced_accuracy"),
        "adaptation_gain_over_nearest_centroid": macro.get("adaptation_gain_balanced_accuracy"),
        "relative_recovery_over_nearest_centroid": macro.get("relative_recovery_balanced_accuracy"),
    }


def _build_surface_a_diagnostics(analysis: Dict[str, Any]) -> Dict[str, Any]:
    macro = analysis.get("macro_mean", {})
    return {
        "retrieval_map": macro.get("map"),
        "top5_matching_accuracy": macro.get("top@5"),
        "heldout_centroid_distance": macro.get("heldout_centroid_distance"),
        "heldout_silhouette": macro.get("heldout_silhouette"),
        "nearest_neighbor_purity": macro.get("nearest_neighbor_purity"),
        "generalization_gap_vs_seen_organ_reference": {
            organ: payload.get("generalization_gap")
            for organ, payload in sorted((analysis.get("per_organ") or {}).items())
        },
    }


def _build_surface_b_diagnostics(analysis: Dict[str, Any]) -> Dict[str, Any]:
    macro = analysis.get("macro_mean", {})
    macro_std = analysis.get("macro_mean_std", {})
    return {
        "held_out_organ_accuracy": macro.get("held_out_accuracy"),
        "in_distribution_accuracy": macro.get("in_distribution_accuracy"),
        "generalization_gap": macro.get("generalization_gap"),
        "transfer_efficiency": macro.get("transfer_efficiency"),
        "legacy_fewshot_probe_score": macro.get("fewshot_probe_score"),
        "nearest_centroid_baseline_balanced_accuracy": macro.get("nearest_centroid_balanced_accuracy"),
        "nearest_centroid_baseline_held_out_accuracy": macro.get("nearest_centroid_held_out_accuracy"),
        "adaptation_gain_balanced_accuracy": macro.get("adaptation_gain_balanced_accuracy"),
        "adaptation_gain_held_out_accuracy": macro.get("adaptation_gain_held_out_accuracy"),
        "relative_recovery_balanced_accuracy": macro.get("relative_recovery_balanced_accuracy"),
        "seed_std_balanced_accuracy": macro_std.get("balanced_accuracy"),
        "seed_std_nearest_centroid_baseline_balanced_accuracy": macro_std.get(
            "nearest_centroid_balanced_accuracy"
        ),
        "seed_std_adaptation_gain_balanced_accuracy": macro_std.get("adaptation_gain_balanced_accuracy"),
        "seed_std_relative_recovery_balanced_accuracy": macro_std.get("relative_recovery_balanced_accuracy"),
    }


def _build_headline_metrics(
    surface_a_analysis: Dict[str, Any],
    surface_b_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    surface_b_macro = surface_b_analysis.get("macro_mean", {})
    return {
        "ba_centroid_recoverability": {
            "metric_name": "nearest_centroid_balanced_accuracy",
            "metric_value": surface_b_macro.get("nearest_centroid_balanced_accuracy"),
            "n_supported_organs": surface_b_analysis.get("n_supported_organs"),
            "n_evaluated_organs": surface_b_analysis.get("n_evaluated_organs"),
            "support_per_modality": surface_b_analysis.get("support_per_modality"),
        },
        "ba_probe_recoverability": {
            "metric_name": "balanced_accuracy",
            "metric_value": surface_b_macro.get("balanced_accuracy"),
            "n_supported_organs": surface_b_analysis.get("n_supported_organs"),
            "n_evaluated_organs": surface_b_analysis.get("n_evaluated_organs"),
            "support_per_modality": surface_b_analysis.get("support_per_modality"),
        },
        "adaptation_gain_over_centroid": {
            "metric_name": "adaptation_gain_balanced_accuracy_over_nearest_centroid",
            "metric_value": surface_b_macro.get("adaptation_gain_balanced_accuracy"),
            "baseline_metric_name": "nearest_centroid_balanced_accuracy",
            "baseline_metric_value": surface_b_macro.get("nearest_centroid_balanced_accuracy"),
            "adapted_metric_name": "balanced_accuracy",
            "adapted_metric_value": surface_b_macro.get("balanced_accuracy"),
            "relative_recovery": surface_b_macro.get("relative_recovery_balanced_accuracy"),
            "n_supported_organs": surface_b_analysis.get("n_supported_organs"),
            "n_evaluated_organs": surface_b_analysis.get("n_evaluated_organs"),
            "support_per_modality": surface_b_analysis.get("support_per_modality"),
        },
    }


def _build_module3_protocol(
    summary: Dict[str, Any],
    required_modalities: Sequence[str],
    supported_organs: Sequence[str],
) -> Dict[str, Any]:
    required_modalities = [str(modality).lower() for modality in required_modalities]
    surface_scope = "within_modality" if len(required_modalities) == 1 else "cross_modality"
    exposures = set((summary.get("pretraining_exposure") or {}).keys())
    if surface_scope == "within_modality":
        surface_stratum = f"{required_modalities[0]}_within_dataset_holdout_surface"
    elif exposures == {"pretraining_unseen_source"}:
        surface_stratum = "pretraining_unseen_ct_mr_validation"
    elif "pretraining_seen_source" in exposures and "pretraining_unseen_source" in exposures:
        surface_stratum = "mixed_exposure_ct_mr_anchor"
    elif exposures == {"pretraining_seen_source"}:
        surface_stratum = "pretraining_seen_ct_mr_surface"
    else:
        surface_stratum = "mixed_or_unknown_exposure_surface"
    return {
        "held_out_definition": "held_out_during_adaptation",
        "held_out_definition_note": (
            "Held-out organ is defined by the Module 3 leave-one-organ-out adaptation fold, not solely by whether "
            "the source dataset was seen during encoder pretraining."
        ),
        "fold_policy": "leave_one_organ_out",
        "case_split_policy": "global_case_disjoint_partitions",
        "within_modality_policy": "within_dataset_seen_during_adaptation_vs_held_out_during_adaptation",
        "surface_scope": surface_scope,
        "cross_modality_surface_stratum": surface_stratum if surface_scope == "cross_modality" else None,
        "within_modality_surface_stratum": surface_stratum if surface_scope == "within_modality" else None,
        "checkpoint_policy": DEFAULT_CHECKPOINT_POLICY,
        "required_modalities": list(required_modalities),
        "supported_organs": list(sorted(supported_organs)),
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write a flat summary table for batch Module 3 results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_batch_summary(
    summary_output: Path,
    csv_output: Path,
    analysis_name: str,
    manifest_path: Path,
    manifest_variant: str,
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
    rows: Sequence[Dict[str, Any]],
    missing_entries: Sequence[Dict[str, Any]],
) -> None:
    """Write the current batch summary payload and CSV snapshot."""
    summary_payload = {
        "analysis_name": analysis_name,
        "manifest_path": str(manifest_path),
        "manifest_variant": manifest_variant,
        "requested_checkpoints": list(checkpoints),
        "requested_feature_types": list(feature_types),
        "completed_rows": list(rows),
        "missing_entries": list(missing_entries),
    }
    _write_json(summary_output, summary_payload)
    _write_csv(csv_output, rows)


def _load_embedding_bundle(npz_path: Path, manifest_samples: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Load one Phase 2 embedding bundle and map feature rows to sample IDs.

    Required parameters:
        npz_path: path to the Phase 2 embedding NPZ.
        manifest_samples: normalized Module 3 samples used to recover sample IDs
            when the NPZ omitted them.
    """
    payload = np.load(npz_path, allow_pickle=True)
    if "features" not in payload:
        raise ValueError(f"Embedding bundle {npz_path} must contain a 'features' array")

    features = np.asarray(payload["features"], dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected features to have shape [n_samples, dim], got {features.shape}")

    if "sample_ids" in payload:
        sample_ids = [str(item) for item in payload["sample_ids"].tolist()]
    else:
        if features.shape[0] != len(manifest_samples):
            raise ValueError(
                "Embedding bundle omitted sample_ids, so manifest order must match features length exactly"
            )
        sample_ids = [sample["sample_id"] for sample in manifest_samples]

    if len(sample_ids) != features.shape[0]:
        raise ValueError("sample_ids length does not match features length")

    return {sample_id: features[index] for index, sample_id in enumerate(sample_ids)}


def _checkpoint_sort_key(checkpoint_name: str) -> Tuple[int, str]:
    """Sort known checkpoints according to the canonical Module 3 order."""
    checkpoint_name = normalize_checkpoint_name(checkpoint_name)
    known_order = {name: idx for idx, name in enumerate(DEFAULT_CHECKPOINTS)}
    return (known_order.get(checkpoint_name, len(known_order)), checkpoint_name)


def _ordered_checkpoint_names(checkpoint_names: Iterable[str]) -> List[str]:
    """Deduplicate and order checkpoint names for batch evaluation."""
    unique_names = list(dict.fromkeys(normalize_checkpoint_name(name) for name in checkpoint_names))
    return sorted(unique_names, key=_checkpoint_sort_key)


def _metric_value(payload: Dict[str, Any], *keys: str) -> Any:
    """Safely read a nested metric from a Module 3 result payload."""
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _write_manifest_summary(
    output_root: Path,
    analysis_name: str,
    manifest_path: Path,
    validation: Dict[str, Any],
    summary: Dict[str, Any],
) -> None:
    """Persist manifest validation and summary metadata for one run root."""
    _write_json(
        output_root / MANIFEST_SUMMARY_NAME,
        {
            "analysis_name": analysis_name,
            "manifest_path": str(manifest_path),
            "validation": validation,
            "summary": summary,
        },
    )


def _evaluate_single_bundle(
    manifest_path: Path,
    analysis_name: str,
    manifest_variant: str,
    checkpoint_name: str,
    feature_type: str,
    embeddings_npz: Path,
    required_modalities: Sequence[str],
    min_holdout_organs: int,
    min_samples_per_modality: int,
    top_ks: Sequence[int],
    max_queries_per_organ: int | None,
    max_targets_per_organ: int | None,
    few_shot_support_per_modality: int,
    few_shot_query_per_modality: int | None,
    few_shot_seeds: Sequence[int],
    seed: int,
    output_path: Optional[Path],
) -> Dict[str, Any]:
    """Evaluate one checkpoint/feature bundle against one Module 3 manifest.

    Required parameters:
        manifest_path: Module 3 manifest JSON path.
        analysis_name: dataset namespace for outputs.
        manifest_variant: manifest variant label, usually ``core``.
        checkpoint_name: checkpoint label.
        feature_type: one of ``cls``, ``avg_pool``, or ``multilayer``.
        embeddings_npz: explicit Phase 2 embedding bundle.
    """
    ensure_output_directories(analysis_name, manifest_variant)
    output_paths = get_output_paths(analysis_name, manifest_variant)

    samples = load_module3_manifest(manifest_path)
    validation = validate_module3_manifest(
        samples,
        required_modalities=required_modalities,
        min_holdout_organs=min_holdout_organs,
        min_samples_per_modality=min_samples_per_modality,
    )
    if not validation["is_valid"]:
        raise ValueError(f"Module 3 manifest validation failed for {manifest_path}: {validation}")
    supported_organs = get_eligible_holdout_organs(
        samples,
        required_modalities=required_modalities,
        min_samples_per_modality=min_samples_per_modality,
    )
    features_by_id = _load_embedding_bundle(embeddings_npz, samples)
    analysis = evaluate_leave_one_organ_out_transfer(
        features_by_id,
        samples,
        supported_organs=supported_organs,
        required_modalities=required_modalities,
        top_ks=top_ks,
        max_queries_per_organ=max_queries_per_organ,
        max_targets_per_organ=max_targets_per_organ,
        seed=seed,
    )
    few_shot_analysis = evaluate_leave_one_organ_out_few_shot_transfer(
        features_by_id,
        samples,
        supported_organs=supported_organs,
        required_modalities=required_modalities,
        support_per_modality=few_shot_support_per_modality,
        query_per_modality=few_shot_query_per_modality,
        seeds=few_shot_seeds,
    )
    manifest_summary = summarize_module3_manifest(samples)
    module3_protocol = _build_module3_protocol(manifest_summary, required_modalities, supported_organs)
    surface_a_payload = {
        "status": analysis.get("status"),
        "surface_name": "surface_a_feature_space_evidence",
        "paper_role": "diagnostic_feature_space_analysis_not_headline",
        "surface_scope": module3_protocol.get("surface_scope"),
        "module3_protocol": module3_protocol,
        "metric_aliases": _build_surface_a_metric_aliases(analysis),
        "diagnostics": _build_surface_a_diagnostics(analysis),
        "leave_one_organ_out_analysis": analysis,
    }
    surface_b_payload = {
        "status": few_shot_analysis.get("status"),
        "surface_name": "surface_b_few_shot_transfer_evidence",
        "paper_role": "headline_recoverability_surface",
        "surface_scope": module3_protocol.get("surface_scope"),
        "module3_protocol": module3_protocol,
        "metric_aliases": _build_surface_b_metric_aliases(few_shot_analysis),
        "diagnostics": _build_surface_b_diagnostics(few_shot_analysis),
        "few_shot_transfer_analysis": few_shot_analysis,
    }
    headline_metrics = _build_headline_metrics(analysis, few_shot_analysis)

    payload = {
        "analysis_name": analysis_name,
        "checkpoint_name": checkpoint_name,
        "feature_type": feature_type,
        "manifest_path": str(manifest_path),
        "embeddings_npz": str(embeddings_npz),
        "required_modalities": list(required_modalities),
        "module3_protocol": module3_protocol,
        "headline_metrics": headline_metrics,
        "metric_contract": {
            "headline_metrics": [
                "headline_metrics.ba_centroid_recoverability.metric_value",
                "headline_metrics.ba_probe_recoverability.metric_value",
                "headline_metrics.adaptation_gain_over_centroid.metric_value",
                "headline_metrics.adaptation_gain_over_centroid.baseline_metric_value",
                "headline_metrics.adaptation_gain_over_centroid.adapted_metric_value",
            ],
            "diagnostic_metrics": [
                "surface_a_feature_space_evidence.diagnostics",
                "surface_b_few_shot_transfer_evidence.diagnostics",
                "surface_a_feature_space_evidence.leave_one_organ_out_analysis",
                "surface_b_few_shot_transfer_evidence.few_shot_transfer_analysis",
            ],
        },
        "manifest_validation": validation,
        "manifest_summary": manifest_summary,
        "surface_a_feature_space_evidence": surface_a_payload,
        "surface_b_few_shot_transfer_evidence": surface_b_payload,
        "leave_one_organ_out_transfer": analysis,
    }

    resolved_output_path = output_path or get_checkpoint_metrics_path(output_paths["results"], checkpoint_name, feature_type)
    _write_manifest_summary(output_paths["results"], analysis_name, manifest_path, validation, payload["manifest_summary"])
    _write_json(resolved_output_path, payload)
    return payload


def _discover_bundle_map(
    analysis_name: str,
    manifest_variant: str,
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
) -> Dict[Tuple[str, str], Path]:
    """Discover which requested upstream feature bundles already exist on disk.

    Batch discovery currently reuses the canonical organ-embedding layout owned
    by the Phase 2 package under outputs_phase2. That is a filesystem contract,
    not a scientific requirement that the source dataset be valid for Phase 2
    paired CT-MR alignment.
    """
    bundle_map: Dict[Tuple[str, str], Path] = {}
    for checkpoint_name in checkpoints:
        for feature_type in feature_types:
            candidate = get_phase2_feature_npz_path(analysis_name, manifest_variant, checkpoint_name, feature_type)
            if candidate.exists():
                bundle_map[(checkpoint_name, feature_type)] = candidate
    return bundle_map


def _build_summary_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one Module 3 result payload into one batch-summary row."""
    headline_metrics = payload.get("headline_metrics", {})
    ba_centroid = headline_metrics.get("ba_centroid_recoverability", {})
    ba_probe = headline_metrics.get("ba_probe_recoverability", {})
    adaptation_gain = headline_metrics.get("adaptation_gain_over_centroid", {})
    surface_a = payload.get("surface_a_feature_space_evidence", {})
    return {
        "checkpoint_name": payload.get("checkpoint_name"),
        "feature_type": payload.get("feature_type"),
        "surface_a_status": surface_a.get("status"),
        "surface_b_status": (payload.get("surface_b_few_shot_transfer_evidence") or {}).get("status"),
        "n_supported_organs": adaptation_gain.get("n_supported_organs") or ba_probe.get("n_supported_organs") or ba_centroid.get("n_supported_organs"),
        "n_evaluated_organs": adaptation_gain.get("n_evaluated_organs") or ba_probe.get("n_evaluated_organs") or ba_centroid.get("n_evaluated_organs"),
        "ba_centroid_recoverability": ba_centroid.get("metric_value"),
        "ba_probe_recoverability": ba_probe.get("metric_value"),
        "adaptation_gain_balanced_accuracy": adaptation_gain.get("metric_value"),
        "adaptation_gain_relative_recovery": adaptation_gain.get("relative_recovery"),
        "support_per_modality": adaptation_gain.get("support_per_modality") or ba_probe.get("support_per_modality") or ba_centroid.get("support_per_modality"),
        "diagnostic_top1_matching_accuracy": _metric_value(surface_a, "metric_aliases", "top1_matching_accuracy"),
    }


def main() -> None:
    """CLI entrypoint for Module 3 evaluation.

    The command resolves one Module 3 manifest, validates it, and then runs
    either single-bundle or batch evaluation depending on whether
    ``--embeddings-npz`` was provided.
    """
    parser = argparse.ArgumentParser(
        description="Module 3 anatomical generalization over frozen organ embedding bundles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python module3_evaluation_pipeline.py -m ../data_manifests/module3_anatomical_generalization/totalsegmenter_ct_mr_anchor/core/manifest_sampled.json --analysis-name totalsegmenter_ct_mr_anchor --checkpoint-name Med3DINO_REL_c96 --feature-type cls --embeddings-npz ../phase2_cross_modality_alignment/outputs_phase2/totalsegmenter_ct_mr_anchor/phase2/core/features/Med3DINO_REL_c96/cls/phase2_organ_cls_embeddings.npz\n"
            "  python module3_evaluation_pipeline.py --analysis-name totalsegmenter_ct_mr_anchor --manifest-variant core --feature-types cls avg_pool"
        ),
    )
    parser.add_argument("-m", "--manifest", default=None, help="Path to the Module 3 manifest JSON")
    parser.add_argument("-a", "--analysis-name", default=None, help="Dataset namespace for outputs")
    parser.add_argument("--manifest-variant", default="core", help="Manifest variant label")
    parser.add_argument("--embeddings-npz", default=None, help="Optional NPZ bundle with Phase 2 features")
    parser.add_argument("--checkpoint-name", default=None, help="Checkpoint label for single-bundle mode")
    parser.add_argument("--feature-type", default="cls", help="Feature family label for single-bundle mode")
    parser.add_argument(
        "--feature-types",
        nargs="+",
        default=None,
        help="Feature families for batch mode. Defaults to all registered Phase 2 feature families.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="Optional checkpoint subset for batch mode",
    )
    parser.add_argument(
        "--required-modalities",
        nargs="+",
        default=list(DEFAULT_REQUIRED_MODALITIES),
        help="Required modalities for cross-modal hold-out support",
    )
    parser.add_argument(
        "--min-holdout-organs",
        type=int,
        default=DEFAULT_MIN_HOLDOUT_ORGANS,
        help="Minimum number of cross-modal organs required by the manifest",
    )
    parser.add_argument(
        "--min-samples-per-modality",
        type=int,
        default=DEFAULT_MIN_SAMPLES_PER_MODALITY,
        help="Minimum support per organ and modality for hold-out eligibility",
    )
    parser.add_argument("--top-ks", nargs="+", type=int, default=[1, 5], help="Retrieval cutoffs for Module 3")
    parser.add_argument("--max-queries-per-organ", type=int, default=None)
    parser.add_argument("--max-targets-per-organ", type=int, default=None)
    parser.add_argument(
        "--few-shot-support-per-modality",
        type=int,
        default=DEFAULT_FEW_SHOT_SUPPORT_PER_MODALITY,
        help="Balanced Surface B support budget per organ and modality",
    )
    parser.add_argument(
        "--few-shot-query-per-modality",
        type=int,
        default=DEFAULT_FEW_SHOT_QUERY_PER_MODALITY,
        help="Optional cap for the balanced Surface B query budget per organ and modality",
    )
    parser.add_argument(
        "--few-shot-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_FEW_SHOT_SEEDS),
        help="Random seeds used for Surface B balanced support/query sampling",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", default=None, help="Explicit output path for single-bundle mode")
    parser.add_argument("--summary-output", default=None, help=f"Optional batch summary JSON path ({SUMMARY_JSON_NAME})")
    parser.add_argument("--csv-output", default=None, help=f"Optional batch summary CSV path ({SUMMARY_CSV_NAME})")
    parser.add_argument("--require-complete", action="store_true", help="Fail batch mode if any requested bundle is missing")
    args = parser.parse_args()

    manifest_variant = normalize_manifest_variant(args.manifest_variant)
    manifest_path = Path(args.manifest).resolve() if args.manifest else None
    if manifest_path is None and args.analysis_name is None:
        raise ValueError("Either --manifest or --analysis-name is required")

    analysis_name = args.analysis_name or get_dataset_name_from_manifest_path(manifest_path)
    resolved_manifest_path = manifest_path or get_module3_manifest_path(analysis_name, "sampled", manifest_variant)
    if not resolved_manifest_path.exists():
        raise FileNotFoundError(f"Module 3 manifest not found: {resolved_manifest_path}")

    ensure_output_directories(analysis_name, manifest_variant)
    output_paths = get_output_paths(analysis_name, manifest_variant)
    required_modalities = [str(modality).strip().lower() for modality in args.required_modalities]

    if args.embeddings_npz:
        if not args.checkpoint_name:
            raise ValueError("--checkpoint-name is required in single-bundle mode")
        feature_type = normalize_feature_type(args.feature_type)
        logger.info(
            "Starting Module 3 single-bundle evaluation: analysis=%s checkpoint=%s feature_type=%s manifest=%s embeddings=%s",
            analysis_name,
            args.checkpoint_name,
            feature_type,
            resolved_manifest_path,
            Path(args.embeddings_npz).resolve(),
        )
        start_time = perf_counter()
        payload = _evaluate_single_bundle(
            manifest_path=resolved_manifest_path,
            analysis_name=analysis_name,
            manifest_variant=manifest_variant,
            checkpoint_name=args.checkpoint_name,
            feature_type=feature_type,
            embeddings_npz=Path(args.embeddings_npz).resolve(),
            required_modalities=required_modalities,
            min_holdout_organs=args.min_holdout_organs,
            min_samples_per_modality=args.min_samples_per_modality,
            top_ks=args.top_ks,
            max_queries_per_organ=args.max_queries_per_organ,
            max_targets_per_organ=args.max_targets_per_organ,
            few_shot_support_per_modality=args.few_shot_support_per_modality,
            few_shot_query_per_modality=args.few_shot_query_per_modality,
            few_shot_seeds=args.few_shot_seeds,
            seed=args.seed,
            output_path=Path(args.output_path).resolve() if args.output_path else None,
        )
        logger.info(
            "Completed Module 3 single-bundle evaluation: checkpoint=%s feature_type=%s status=%s elapsed_sec=%.1f",
            args.checkpoint_name,
            feature_type,
            payload["surface_a_feature_space_evidence"].get("status"),
            perf_counter() - start_time,
        )
        if payload["surface_a_feature_space_evidence"].get("status") != "ok":
            raise ValueError(f"Module 3 evaluation did not produce organ scores for {args.embeddings_npz}")
        return

    checkpoints = _ordered_checkpoint_names(args.checkpoints or DEFAULT_CHECKPOINTS)
    feature_types = [normalize_feature_type(feature_type) for feature_type in (args.feature_types or FEATURE_TYPES)]
    bundle_map = _discover_bundle_map(analysis_name, manifest_variant, checkpoints, feature_types)
    rows: List[Dict[str, Any]] = []
    missing_entries: List[Dict[str, Any]] = []
    summary_output = Path(args.summary_output).resolve() if args.summary_output else get_summary_json_path(output_paths["results"])
    csv_output = Path(args.csv_output).resolve() if args.csv_output else get_summary_csv_path(output_paths["results"])
    total_requested = len(checkpoints) * len(feature_types)
    logger.info(
        "Starting Module 3 batch evaluation: analysis=%s variant=%s requested_bundles=%d available_bundles=%d checkpoints=%s feature_types=%s",
        analysis_name,
        manifest_variant,
        total_requested,
        len(bundle_map),
        checkpoints,
        feature_types,
    )
    processed_bundle_count = 0

    for checkpoint_index, checkpoint_name in enumerate(checkpoints, start=1):
        logger.info(
            "Starting checkpoint %s (%d/%d)",
            checkpoint_name,
            checkpoint_index,
            len(checkpoints),
        )
        for feature_index, feature_type in enumerate(feature_types, start=1):
            processed_bundle_count += 1
            embeddings_npz = bundle_map.get((checkpoint_name, feature_type))
            if embeddings_npz is None:
                missing_entries.append({
                    "checkpoint_name": checkpoint_name,
                    "feature_type": feature_type,
                    "reason": "missing_phase2_feature_bundle",
                })
                logger.warning(
                    "Skipping missing Phase 2 bundle %d/%d: checkpoint=%s feature_type=%s",
                    processed_bundle_count,
                    total_requested,
                    checkpoint_name,
                    feature_type,
                )
                _write_batch_summary(
                    summary_output=summary_output,
                    csv_output=csv_output,
                    analysis_name=analysis_name,
                    manifest_path=resolved_manifest_path,
                    manifest_variant=manifest_variant,
                    checkpoints=checkpoints,
                    feature_types=feature_types,
                    rows=rows,
                    missing_entries=missing_entries,
                )
                continue
            logger.info(
                "Evaluating bundle %d/%d: checkpoint=%s (%d/%d) feature_type=%s (%d/%d)",
                processed_bundle_count,
                total_requested,
                checkpoint_name,
                checkpoint_index,
                len(checkpoints),
                feature_type,
                feature_index,
                len(feature_types),
            )
            bundle_start_time = perf_counter()
            payload = _evaluate_single_bundle(
                manifest_path=resolved_manifest_path,
                analysis_name=analysis_name,
                manifest_variant=manifest_variant,
                checkpoint_name=checkpoint_name,
                feature_type=feature_type,
                embeddings_npz=embeddings_npz,
                required_modalities=required_modalities,
                min_holdout_organs=args.min_holdout_organs,
                min_samples_per_modality=args.min_samples_per_modality,
                top_ks=args.top_ks,
                max_queries_per_organ=args.max_queries_per_organ,
                max_targets_per_organ=args.max_targets_per_organ,
                few_shot_support_per_modality=args.few_shot_support_per_modality,
                few_shot_query_per_modality=args.few_shot_query_per_modality,
                few_shot_seeds=args.few_shot_seeds,
                seed=args.seed,
                output_path=None,
            )
            rows.append(_build_summary_row(payload))
            _write_batch_summary(
                summary_output=summary_output,
                csv_output=csv_output,
                analysis_name=analysis_name,
                manifest_path=resolved_manifest_path,
                manifest_variant=manifest_variant,
                checkpoints=checkpoints,
                feature_types=feature_types,
                rows=rows,
                missing_entries=missing_entries,
            )
            logger.info(
                "Completed bundle %d/%d: checkpoint=%s feature_type=%s status=%s elapsed_sec=%.1f completed_rows=%d missing_entries=%d",
                processed_bundle_count,
                total_requested,
                checkpoint_name,
                feature_type,
                payload.get("surface_a_feature_space_evidence", {}).get("status"),
                perf_counter() - bundle_start_time,
                len(rows),
                len(missing_entries),
            )

        logger.info(
            "Finished checkpoint %s (%d/%d): completed_rows=%d missing_entries=%d",
            checkpoint_name,
            checkpoint_index,
            len(checkpoints),
            len(rows),
            len(missing_entries),
        )

    _write_batch_summary(
        summary_output=summary_output,
        csv_output=csv_output,
        analysis_name=analysis_name,
        manifest_path=resolved_manifest_path,
        manifest_variant=manifest_variant,
        checkpoints=checkpoints,
        feature_types=feature_types,
        rows=rows,
        missing_entries=missing_entries,
    )

    if args.require_complete and missing_entries:
        raise ValueError(f"Module 3 batch run is incomplete: {missing_entries}")

    logger.info("Wrote Module 3 batch summary to %s", summary_output)


if __name__ == "__main__":
    main()