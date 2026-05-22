#!/usr/bin/env python
# phase1_evaluation_pipeline.py
# Phase 1: Spacing/Anisotropy Robustness - Full Evaluation Pipeline
#
# Orchestrates representation geometry, spacing readout, semantic readout,
# cross-bin semantic transfer, and controlled spacing perturbation across checkpoints.
#
# Supports dataset-specific evaluation with parameterized output paths:
#   --analysis-name abdomenatlas  → outputs/abdomenatlas/{results,figures}/
#   --analysis-name abdomenct1k   → outputs/abdomenct1k/{results,figures}/

"""
cd eval_downstream/medfm_eval/phase1_anisotropy_robustness

CUDA_VISIBLE_DEVICES=0 python phase1_evaluation_pipeline.py --full \
    -m ../data_manifests/phase1_anisotropy_robustness/abdomenatlas/original_bins/manifest_sampled.json \
    -a abdomenatlas \
    --batch-size 16 --num-workers 4

CUDA_VISIBLE_DEVICES=0 python phase1_evaluation_pipeline.py --setting-b-only \
    -m ../data_manifests/phase1_anisotropy_robustness/totalsegmenter_ct/original_bins/manifest_sampled.json \
    -a totalsegmenter_ct \
    -c Med3DINO_REL_c96 -f cls
"""


import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import Counter
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CANONICAL_CHECKPOINT_NAMES,
    CHECKPOINTS,
    PHASE1_MANIFESTS,
    CROSS_BIN_TRANSFER_CONFIG,
    compute_anisotropy_ratio,
    get_available_checkpoint_names,
    get_phase1_manifest_path,
    get_checkpoint_feature_dir,
    get_dataset_name_from_manifest_path,
    get_manifest_variant_from_manifest_path,
    get_output_paths,
    normalize_checkpoint_name,
    ensure_output_directories,
    get_binning_scheme_from_manifest,
    get_bin_name_map,
)
from checkpoint_feature_extractor import FeatureExtractor
from phase1_data_loader import create_phase1_dataloader, warmup_observational_cache
from representation_geometry_analysis import run_track_a_analysis, compare_checkpoints_track_a
from anisotropy_semantic_analysis import (
    run_track_b_analysis,
    compare_checkpoints_track_b,
    run_full_track_b_analysis,
    _select_balanced_bin_indices,
    serialize_spacing_regression_results,
    spacing_regression_probe,
    run_semantic_probing as run_semantic_probing_fn,
    run_multilabel_cross_bin_transfer,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

OBSERVATIONAL_ANALYSIS_KEY = "observational_bin_analysis"
CONTROLLED_PERTURBATION_KEY = "controlled_spacing_perturbation"
REPRESENTATION_GEOMETRY_KEY = "representation_geometry"
SPACING_READOUT_KEY = "anisotropy_regression"
SEMANTIC_READOUT_KEY = "semantic_readout"
CROSS_BIN_SEMANTIC_TRANSFER_KEY = "cross_bin_semantic_transfer"

FEATURE_TYPES = ["cls", "avg_pool", "multilayer"]
RESULT_FILE_DESCRIPTIONS = {
    "track_a": {
        "task_name": REPRESENTATION_GEOMETRY_KEY,
        "description": "Per-checkpoint representation geometry metrics from Track A.",
    },
    "track_b": {
        "task_name": "legacy_track_b_domain_readout",
        "description": "Legacy Track B domain probing output from the older entrypoints.",
    },
    "track_b_multiseed": {
        "task_name": "legacy_track_b_domain_readout_multiseed",
        "description": "Legacy Track B domain probing aggregated across evaluation seeds.",
    },
    "spacing_regression": {
        "task_name": SPACING_READOUT_KEY,
        "description": "Per-checkpoint spacing regression readout used by the current Setting A pipeline.",
    },
    "semantic_probing": {
        "task_name": SEMANTIC_READOUT_KEY,
        "description": "Per-checkpoint semantic probing results on informative labels.",
    },
    "semantic_transfer": {
        "task_name": CROSS_BIN_SEMANTIC_TRANSFER_KEY,
        "description": "Per-checkpoint cross-bin transfer using semantic labels.",
    },
    "perturbation_robustness": {
        "task_name": CONTROLLED_PERTURBATION_KEY,
        "description": "Per-checkpoint controlled spacing perturbation results for Setting B.",
    },
}


def _resolve_phase1_namespace(manifest_path: Path, analysis_name: Optional[str]) -> tuple[str, str]:
    dataset_name = analysis_name or get_dataset_name_from_manifest_path(manifest_path)
    manifest_variant = get_manifest_variant_from_manifest_path(manifest_path)
    return dataset_name, manifest_variant


def _ordered_checkpoint_names(checkpoint_names: List[str]) -> List[str]:
    normalized_names = [normalize_checkpoint_name(name) for name in checkpoint_names]
    known_order = {name: idx for idx, name in enumerate(CANONICAL_CHECKPOINT_NAMES)}
    return sorted(
        normalized_names,
        key=lambda name: (known_order.get(name, len(known_order)), name),
    )


def _checkpoint_crop_size(checkpoint_name: str) -> int:
    checkpoint_name = normalize_checkpoint_name(checkpoint_name)
    if checkpoint_name in CHECKPOINTS:
        return CHECKPOINTS[checkpoint_name].crop_size
    return 112 if "c112" in checkpoint_name else 96


def _feature_cache_path(
    features_dir: Path,
    checkpoint_name: str,
    feature_type: str,
    manifest_path: Path,
) -> Path:
    crop_size = _checkpoint_crop_size(checkpoint_name)
    manifest_hash = hashlib.md5(manifest_path.read_bytes()).hexdigest()[:8]
    feature_cache_dir = get_checkpoint_feature_dir(features_dir, checkpoint_name, feature_type)
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    return feature_cache_dir / f"features_c{crop_size}_{manifest_hash}.npz"


def _maybe_build_crop_dataloader(
    dataloaders: Dict[int, Any],
    manifest_path: Path,
    crop_size: int,
    batch_size: int,
    num_workers: int,
):
    dataloader = dataloaders.get(crop_size)
    if dataloader is not None:
        return dataloader

    dataloader = create_phase1_dataloader(
        manifest_path,
        crop_size=crop_size,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    dataloaders[crop_size] = dataloader
    return dataloader


def _warmup_required_observational_caches(
    manifest_path: Path,
    checkpoints: Optional[List[str]],
    batch_size: int,
    num_workers: int,
) -> None:
    target_checkpoints = checkpoints if checkpoints else get_available_checkpoint_names()
    crop_sizes = sorted({_checkpoint_crop_size(name) for name in target_checkpoints if name in CHECKPOINTS})

    for crop_size in crop_sizes:
        warmup_observational_cache(
            manifest_path=manifest_path,
            crop_size=crop_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )


def _build_checkpoint_scope(
    requested_checkpoints: Optional[List[str]],
    evaluated_checkpoints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    requested = _ordered_checkpoint_names(list(requested_checkpoints) if requested_checkpoints is not None else get_available_checkpoint_names())
    evaluated = _ordered_checkpoint_names(list(evaluated_checkpoints or requested))
    full_checkpoint_set = set(CANONICAL_CHECKPOINT_NAMES)
    is_full_checkpoint_sweep = set(evaluated) == full_checkpoint_set and len(evaluated) == len(full_checkpoint_set)

    if not evaluated:
        scope_kind = "empty"
        scope_tag = "empty"
    elif is_full_checkpoint_sweep:
        scope_kind = "all_checkpoints"
        scope_tag = "full"
    elif len(evaluated) == 1:
        scope_kind = "single_checkpoint"
        scope_tag = evaluated[0]
    else:
        scope_kind = "checkpoint_subset"
        digest = hashlib.md5(",".join(evaluated).encode("utf-8")).hexdigest()[:8]
        scope_tag = f"subset_{len(evaluated)}of{len(CANONICAL_CHECKPOINT_NAMES)}_{digest}"

    return {
        "requested_checkpoints": requested,
        "evaluated_checkpoints": evaluated,
        "n_requested": len(requested),
        "n_evaluated": len(evaluated),
        "scope_kind": scope_kind,
        "scope_tag": scope_tag,
        "is_full_checkpoint_sweep": is_full_checkpoint_sweep,
    }


def _get_aggregate_results_filename(feature_type: str, checkpoint_scope: Dict[str, Any]) -> str:
    if checkpoint_scope["is_full_checkpoint_sweep"]:
        return f"phase1_full_results_{feature_type}.json"
    return f"phase1_results_{feature_type}__{checkpoint_scope['scope_tag']}.json"


def _classify_results_file(file_name: str) -> Dict[str, Any]:
    if file_name.startswith("phase1_full_results_") and file_name.endswith(".json"):
        feature_type = file_name[len("phase1_full_results_"):-len(".json")]
        return {
            "category": "aggregate_results",
            "feature_type": feature_type,
            "task_name": "phase1_aggregate_results",
            "description": "Aggregate Phase 1 results for a checkpoint sweep. Full sweeps use this legacy filename.",
        }

    if file_name.startswith("phase1_results_") and file_name.endswith(".json"):
        stem = file_name[:-len(".json")]
        parts = stem.split("__", 1)
        prefix = parts[0]
        feature_type = prefix[len("phase1_results_"):]
        scope_tag = parts[1] if len(parts) == 2 else "unknown"
        return {
            "category": "aggregate_results",
            "feature_type": feature_type,
            "scope_tag": scope_tag,
            "task_name": "phase1_aggregate_results",
            "description": "Aggregate Phase 1 results for a single-checkpoint or subset run.",
        }

    if file_name.startswith("phase1_summary_") and file_name.endswith(".json"):
        feature_type = file_name[len("phase1_summary_"):-len(".json")]
        return {
            "category": "summary",
            "feature_type": feature_type,
            "task_name": "phase1_summary",
            "description": "Legacy multi-checkpoint summary from the older Track A/Track B entrypoint.",
        }

    summary_files = {
        "track_a_summary.json": {
            "category": "summary",
            "task_name": "track_a_summary",
            "description": "Cross-checkpoint Track A ranking summary.",
        },
        "track_a_checkpoint_comparison.json": {
            "category": "comparison",
            "task_name": "track_a_checkpoint_comparison",
            "description": "Cross-checkpoint comparison of Track A geometry metrics.",
        },
        "track_b_checkpoint_comparison.json": {
            "category": "comparison",
            "task_name": "track_b_checkpoint_comparison",
            "description": "Cross-checkpoint comparison of legacy Track B domain readout metrics.",
        },
    }
    if file_name in summary_files:
        return summary_files[file_name]

    for checkpoint_name in sorted(CHECKPOINTS.keys(), key=len, reverse=True):
        prefix = f"{checkpoint_name}_"
        if not file_name.startswith(prefix):
            continue

        remainder = file_name[len(prefix):]
        for feature_type in sorted(FEATURE_TYPES, key=len, reverse=True):
            feature_prefix = f"{feature_type}_"
            if not remainder.startswith(feature_prefix) or not remainder.endswith(".json"):
                continue

            artifact_name = remainder[len(feature_prefix):-len(".json")]
            description = RESULT_FILE_DESCRIPTIONS.get(
                artifact_name,
                {
                    "task_name": artifact_name,
                    "description": "Per-checkpoint result artifact.",
                },
            )
            return {
                "category": "per_checkpoint",
                "checkpoint": checkpoint_name,
                "feature_type": feature_type,
                "artifact_name": artifact_name,
                **description,
            }

    return {
        "category": "unclassified",
        "task_name": "unknown",
        "description": "Unclassified JSON artifact in the results directory.",
    }


def _classify_results_path(results_dir: Path, json_path: Path) -> Dict[str, Any]:
    relative_parts = json_path.relative_to(results_dir).parts

    if len(relative_parts) == 1:
        return _classify_results_file(relative_parts[0])

    if len(relative_parts) >= 3 and relative_parts[0] == "summaries":
        return {
            "category": "summary",
            "feature_type": relative_parts[1],
            "task_name": Path(relative_parts[-1]).stem,
            "description": "Nested summary artifact.",
        }

    if len(relative_parts) >= 3:
        checkpoint_name, feature_type = relative_parts[0], relative_parts[1]
        artifact_name = Path(relative_parts[-1]).stem
        description = RESULT_FILE_DESCRIPTIONS.get(
            artifact_name,
            {
                "task_name": artifact_name,
                "description": "Per-checkpoint nested result artifact.",
            },
        )
        return {
            "category": "per_checkpoint",
            "checkpoint": checkpoint_name,
            "feature_type": feature_type,
            "artifact_name": artifact_name,
            **description,
        }

    return _classify_results_file(json_path.name)


def write_results_catalog(
    results_dir: Path,
    analysis_name: str,
    manifest_path: Path,
) -> Path:
    catalog = {
        "analysis_name": analysis_name,
        "manifest": str(manifest_path),
        "results_dir": str(results_dir),
        "aggregate_files": [],
        "summary_files": [],
        "comparison_files": [],
        "per_checkpoint_files": {},
        "unclassified_files": [],
    }

    for json_path in sorted(results_dir.rglob("*.json")):
        if json_path.name == "results_catalog.json":
            continue

        classification = _classify_results_path(results_dir, json_path)
        entry = {
            "file_name": json_path.name,
            "relative_path": str(json_path.relative_to(results_dir)),
            "path": str(json_path),
            **classification,
        }

        category = classification["category"]
        if category == "aggregate_results":
            catalog["aggregate_files"].append(entry)
            continue
        if category == "summary":
            catalog["summary_files"].append(entry)
            continue
        if category == "comparison":
            catalog["comparison_files"].append(entry)
            continue
        if category == "per_checkpoint":
            checkpoint_name = classification["checkpoint"]
            feature_type = classification["feature_type"]
            catalog["per_checkpoint_files"].setdefault(checkpoint_name, {}).setdefault(feature_type, []).append(entry)
            continue
        catalog["unclassified_files"].append(entry)

    catalog_path = results_dir / "results_catalog.json"
    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2)

    logger.info(f"Updated results catalog at {catalog_path}")
    return catalog_path


def run_single_checkpoint(
    checkpoint_name: str,
    manifest_path: Path,
    feature_type: str = "cls",
    batch_size: int = 8,
    num_workers: int = 4,
    run_track_a: bool = True,
    run_track_b: bool = True,
    cache_features: bool = True,
    analysis_name: str = "default",
):
    """Run evaluation for a single checkpoint.
    
    Args:
        cache_features: If True (default), cache extracted features to disk
                       and reload on subsequent runs. Dramatically speeds up
                       repeated evaluations.
        analysis_name: Name for output directory (e.g., "abdomenatlas", "abdomenct1k").
    """
    dataset_name, manifest_variant = _resolve_phase1_namespace(manifest_path, analysis_name)
    output_paths = get_output_paths(dataset_name, manifest_variant)
    ensure_output_directories(dataset_name, manifest_variant)
    features_dir = output_paths["features"]
    results_dir = output_paths["results"]
    figures_dir = output_paths["figures"]
    
    logger.info(f"\n{'='*70}")
    logger.info(f"EVALUATING: {checkpoint_name}")
    logger.info(f"{'='*70}")
    
    # Load manifest
    with open(manifest_path) as f:
        manifest_data = json.load(f)
    volumes = manifest_data["volumes"]
    binning_scheme = get_binning_scheme_from_manifest(manifest_data)
    bin_names = get_bin_name_map(binning_scheme)
    
    # Labels
    bin_labels = np.array([v["anisotropy_bin"] for v in volumes])
    anisotropy_ratios = np.array([v.get("anisotropy_ratio", 1.0) for v in volumes])
    unique_datasets = sorted(set(v["dataset"] for v in volumes))
    dataset_to_id = {d: i for i, d in enumerate(unique_datasets)}
    task_labels = np.array([dataset_to_id[v["dataset"]] for v in volumes])
    
    logger.info(f"Loaded {len(volumes)} volumes")
    logger.info(f"Bin distribution: {dict(zip(*np.unique(bin_labels, return_counts=True)))}")
    logger.info(f"Number of datasets: {len(unique_datasets)}")
    logger.info(f"Binning scheme: {binning_scheme}")
    
    # Get crop size from checkpoint
    crop_size = 112 if "c112" in checkpoint_name else 96
    
    feature_cache_path = _feature_cache_path(features_dir, checkpoint_name, feature_type, manifest_path)
    
    # Try loading from cache
    features = None
    if cache_features and feature_cache_path.exists():
        logger.info(f"Loading cached features from {feature_cache_path}")
        try:
            cached = np.load(feature_cache_path)
            if len(cached["features"]) == len(volumes):
                features = cached["features"]
                logger.info(f"Loaded features from cache: {features.shape}")
            else:
                logger.warning(f"Cache size mismatch ({len(cached['features'])} vs {len(volumes)}), re-extracting")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, re-extracting")
    
    # Extract features if not loaded from cache
    if features is None:
        dataloader = create_phase1_dataloader(
            manifest_path,
            crop_size=crop_size,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        
        # Extract features
        logger.info(f"\nExtracting {feature_type} features (crop_size={crop_size})...")
        extractor = FeatureExtractor(checkpoint_name)
        features_dict = extractor.extract_from_dataloader(dataloader, show_progress=True)
        features = features_dict[feature_type]
        
        logger.info(f"Features shape: {features.shape}")
        
        # Save to cache
        if cache_features:
            feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                feature_cache_path,
                features=features,
                bin_labels=bin_labels,
                task_labels=task_labels,
                volumes=[v.get("image_path") or v.get("file_path") for v in volumes],
            )
            logger.info(f"Cached features to {feature_cache_path}")
    
    results = {"checkpoint": checkpoint_name, "feature_type": feature_type}
    
    # Representation geometry
    if run_track_a:
        logger.info("\n--- Representation Geometry ---")
        track_a_results = run_track_a_analysis(
            features=features,
            bin_labels=bin_labels,
            checkpoint_name=checkpoint_name,
            feature_type=feature_type,
            compute_tsne=True,
            output_dir=results_dir,
            figures_dir=figures_dir,
            bin_names=bin_names,
            binning_scheme=binning_scheme,
            dataset_name=dataset_name,
        )
        representation_geometry = {
            "mean_cross_cka": track_a_results.cka_stats["mean_cross_cka"],
            "mean_mmd_rbf": track_a_results.observational_distances["summary"]["mean_mmd_rbf"],
            "mean_sliced_wasserstein": track_a_results.observational_distances["summary"]["mean_sliced_wasserstein"],
            "silhouette": track_a_results.geometry_metrics["silhouette_score"],
            "separability": track_a_results.geometry_metrics["separability_ratio"],
        }
        if track_a_results.balanced_bin_sensitivity is not None:
            representation_geometry["balanced_bin_sensitivity"] = {
                "sample_count_per_bin": track_a_results.balanced_bin_sensitivity["sample_count_per_bin"],
                "mean_cross_cka": track_a_results.balanced_bin_sensitivity["cka_stats"]["mean_cross_cka"],
                "silhouette": track_a_results.balanced_bin_sensitivity["geometry"]["silhouette_score"],
                "separability": track_a_results.balanced_bin_sensitivity["geometry"]["separability_ratio"],
                "mean_mmd_rbf": track_a_results.balanced_bin_sensitivity["observational_distances"]["summary"]["mean_mmd_rbf"],
                "mean_sliced_wasserstein": track_a_results.balanced_bin_sensitivity["observational_distances"]["summary"]["mean_sliced_wasserstein"],
            }
        results[REPRESENTATION_GEOMETRY_KEY] = representation_geometry
        results["track_a"] = representation_geometry
    
    # Spacing readout
    if run_track_b:
        logger.info("\n--- Spacing Readout ---")
        track_b_results, spacing_reg_results = run_track_b_analysis(
            features=features,
            bin_labels=bin_labels,
            task_labels=task_labels,
            checkpoint_name=checkpoint_name,
            feature_type=feature_type,
            anisotropy_ratios=anisotropy_ratios,
            output_dir=results_dir,
            bin_names=bin_names,
            binning_scheme=binning_scheme,
            analysis_name=dataset_name,
        )
        spacing_readout = {
            "overall_accuracy": track_b_results.overall_accuracy,
            "bin_accuracies": track_b_results.bin_accuracies,
            "max_bin_gap": track_b_results.max_bin_gap,
            "accuracy_std": track_b_results.accuracy_std_across_bins,
        }
        if track_b_results.balanced_bin_sensitivity is not None:
            spacing_readout["balanced_bin_sensitivity"] = {
                "sample_count_per_bin": track_b_results.balanced_bin_sensitivity["sample_count_per_bin"],
                "overall_accuracy": track_b_results.balanced_bin_sensitivity["overall_accuracy"],
                "overall_balanced_accuracy": track_b_results.balanced_bin_sensitivity["overall_balanced_accuracy"],
                "bin_accuracies": track_b_results.balanced_bin_sensitivity["bin_accuracies"],
            }
        results[SPACING_READOUT_KEY] = spacing_readout
        results["track_b"] = spacing_readout
        # Add spacing regression results if available
        if spacing_reg_results is not None:
            spacing_readout["spacing_regression"] = serialize_spacing_regression_results(spacing_reg_results)
            if track_b_results.balanced_bin_sensitivity is not None and "spacing_regression" in track_b_results.balanced_bin_sensitivity:
                spacing_readout["balanced_bin_sensitivity"]["spacing_regression"] = track_b_results.balanced_bin_sensitivity["spacing_regression"]

    write_results_catalog(results_dir, dataset_name, manifest_path)
    
    return results


def run_all_checkpoints(
    manifest_path: Path,
    feature_type: str = "cls",
    batch_size: int = 8,
    num_workers: int = 4,
    checkpoints: list = None,
    cache_features: bool = True,
    analysis_name: str = "default",
):
    """Run evaluation on all (or specified) checkpoints."""
    if checkpoints is None:
        checkpoints = get_available_checkpoint_names()
    
    dataset_name, manifest_variant = _resolve_phase1_namespace(manifest_path, analysis_name)
    output_paths = get_output_paths(dataset_name, manifest_variant)
    ensure_output_directories(dataset_name, manifest_variant)
    features_dir = output_paths["features"]
    results_dir = output_paths["results"]
    
    all_results = []
    track_a_results = []
    track_b_results = []
    
    # Load manifest once for comparison
    with open(manifest_path) as f:
        manifest_data = json.load(f)
    volumes = manifest_data["volumes"]
    binning_scheme = get_binning_scheme_from_manifest(manifest_data)
    bin_names = get_bin_name_map(binning_scheme)
    bin_labels = np.array([v["anisotropy_bin"] for v in volumes])
    anisotropy_ratios = np.array([v.get("anisotropy_ratio", 1.0) for v in volumes])
    unique_datasets = sorted(set(v["dataset"] for v in volumes))
    dataset_to_id = {d: i for i, d in enumerate(unique_datasets)}
    task_labels = np.array([dataset_to_id[v["dataset"]] for v in volumes])
    
    dataloaders_by_crop: Dict[int, Any] = {}
    
    for ckpt_name in checkpoints:
        if ckpt_name not in CHECKPOINTS:
            logger.warning(f"Checkpoint {ckpt_name} not found, skipping")
            continue
        
        crop_size = _checkpoint_crop_size(ckpt_name)
        feature_cache_path = _feature_cache_path(features_dir, ckpt_name, feature_type, manifest_path)
        
        # Try loading from cache
        features = None
        if cache_features and feature_cache_path.exists():
            logger.info(f"Loading cached features from {feature_cache_path.name}")
            try:
                cached = np.load(feature_cache_path)
                if len(cached["features"]) == len(volumes):
                    features = cached["features"]
                    logger.info(f"Loaded features from cache: {features.shape}")
                else:
                    logger.warning(f"Cache size mismatch, re-extracting")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, re-extracting")
        
        # Extract features if not loaded from cache
        if features is None:
            dataloader = _maybe_build_crop_dataloader(
                dataloaders_by_crop,
                manifest_path,
                crop_size,
                batch_size,
                num_workers,
            )
            
            extractor = FeatureExtractor(ckpt_name)
            features_dict = extractor.extract_from_dataloader(dataloader, show_progress=True)
            features = features_dict[feature_type]
            
            # Save to cache
            if cache_features:
                feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    feature_cache_path,
                    features=features,
                    bin_labels=bin_labels,
                    task_labels=task_labels,
                )
                logger.info(f"Cached features to {feature_cache_path.name}")
        
        # Track A
        ta_results = run_track_a_analysis(
            features=features,
            bin_labels=bin_labels,
            checkpoint_name=ckpt_name,
            feature_type=feature_type,
            compute_tsne=True,
            output_dir=results_dir,
            figures_dir=results_dir.parent / "figures",
            bin_names=bin_names,
            binning_scheme=binning_scheme,
            dataset_name=dataset_name,
        )
        track_a_results.append(ta_results)
        
        # Track B
        tb_results, spacing_reg = run_track_b_analysis(
            features=features,
            bin_labels=bin_labels,
            task_labels=task_labels,
            checkpoint_name=ckpt_name,
            feature_type=feature_type,
            anisotropy_ratios=anisotropy_ratios,
            output_dir=results_dir,
            bin_names=bin_names,
            binning_scheme=binning_scheme,
            analysis_name=dataset_name,
        )
        track_b_results.append(tb_results)
        
        track_b_dict = {
            "overall_accuracy": tb_results.overall_accuracy,
            "max_bin_gap": tb_results.max_bin_gap,
            "accuracy_std": tb_results.accuracy_std_across_bins,
        }
        if spacing_reg is not None:
            track_b_dict["spacing_regression_r2"] = spacing_reg.r2_score
        
        checkpoint_summary = {
            "checkpoint": ckpt_name,
            REPRESENTATION_GEOMETRY_KEY: {
                "mean_cross_cka": ta_results.cka_stats["mean_cross_cka"],
                    "mean_mmd_rbf": ta_results.observational_distances["summary"]["mean_mmd_rbf"],
                    "mean_sliced_wasserstein": ta_results.observational_distances["summary"]["mean_sliced_wasserstein"],
                "silhouette": ta_results.geometry_metrics["silhouette_score"],
                "separability": ta_results.geometry_metrics["separability_ratio"],
            },
            SPACING_READOUT_KEY: track_b_dict,
        }
        if ta_results.balanced_bin_sensitivity is not None:
            checkpoint_summary[REPRESENTATION_GEOMETRY_KEY]["balanced_bin_sensitivity"] = {
                "sample_count_per_bin": ta_results.balanced_bin_sensitivity["sample_count_per_bin"],
                "mean_cross_cka": ta_results.balanced_bin_sensitivity["cka_stats"]["mean_cross_cka"],
            }
        if tb_results.balanced_bin_sensitivity is not None:
            checkpoint_summary[SPACING_READOUT_KEY]["balanced_bin_sensitivity"] = {
                "sample_count_per_bin": tb_results.balanced_bin_sensitivity["sample_count_per_bin"],
                "overall_accuracy": tb_results.balanced_bin_sensitivity["overall_accuracy"],
            }
        checkpoint_summary["track_a"] = checkpoint_summary[REPRESENTATION_GEOMETRY_KEY]
        checkpoint_summary["track_b"] = checkpoint_summary[SPACING_READOUT_KEY]
        all_results.append(checkpoint_summary)
    
    # Cross-checkpoint comparisons
    if len(track_a_results) > 1:
        logger.info("\n" + "="*70)
        logger.info("CROSS-CHECKPOINT COMPARISON - REPRESENTATION GEOMETRY")
        logger.info("="*70)
        compare_checkpoints_track_a(track_a_results, output_dir=results_dir)
    
    if len(track_b_results) > 1:
        logger.info("\n" + "="*70)
        logger.info("CROSS-CHECKPOINT COMPARISON - SPACING READOUT")
        logger.info("="*70)
        compare_checkpoints_track_b(track_b_results, output_dir=results_dir)
    
    # Save summary
    summary = {
        "manifest": str(manifest_path),
        "feature_type": feature_type,
        "n_volumes": len(volumes),
        "n_datasets": len(unique_datasets),
        "checkpoints": all_results,
    }
    
    # Use consistent filename (overwrites on each run)
    summary_path = results_dir / f"phase1_summary_{feature_type}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved summary to {summary_path}")
    write_results_catalog(results_dir, dataset_name, manifest_path)
    
    # Print final summary table
    logger.info("\n" + "="*70)
    logger.info("FINAL SUMMARY")
    logger.info("="*70)
    logger.info(f"{'Checkpoint':<20} {'CKA':>8} {'Silh':>8} {'Sep':>8} {'Acc':>8} {'Gap':>8}")
    logger.info("-"*70)
    for r in all_results:
        logger.info(
            f"{r['checkpoint']:<20} "
            f"{r[REPRESENTATION_GEOMETRY_KEY]['mean_cross_cka']:>8.4f} "
            f"{r[REPRESENTATION_GEOMETRY_KEY]['silhouette']:>8.4f} "
            f"{r[REPRESENTATION_GEOMETRY_KEY]['separability']:>8.4f} "
            f"{r[SPACING_READOUT_KEY]['overall_accuracy']:>8.4f} "
            f"{r[SPACING_READOUT_KEY]['max_bin_gap']:>8.4f}"
        )
    
    return all_results


# =============================================================================
# FULL PHASE 1 EVALUATION (observational + interventional analyses)
# =============================================================================

def run_full_phase1_evaluation(
    manifest_path: Path,
    checkpoints: list = None,
    feature_type: str = "cls",
    batch_size: int = 8,
    num_workers: int = 4,
    run_setting_a: bool = True,
    run_setting_b: bool = True,
    do_cross_bin_transfer: bool = True,
    run_semantic_probing: bool = True,
    transfer_dataset: Optional[str] = None,
    cache_features: bool = True,
    analysis_name: str = "default",
):
    """
    Run complete Phase 1 evaluation with all components.
    
    This function orchestrates:
    - Setting A: Observational robustness (representation geometry + anisotropy regression)
    - Setting B: Controlled perturbation (resampling experiments)
    - Semantic probing within bins
    - Cross-bin semantic transfer
    
    Args:
        manifest_path: Path to manifest JSON
        checkpoints: List of checkpoint names (default: all)
        feature_type: Feature type to extract
        batch_size: Batch size for extraction
        num_workers: DataLoader workers
        run_setting_a: Whether to run Setting A
        run_setting_b: Whether to run Setting B
        do_cross_bin_transfer: Whether to run cross-bin transfer
        run_semantic_probing: Whether to run semantic probing
        transfer_dataset: Dataset for transfer experiments (auto-detect if None)
        cache_features: Whether to cache extracted features (default True)
        analysis_name: Name for output directory (e.g., "abdomenatlas", "abdomenct1k")
        
    Returns:
        Dict with all results
    """
    dataset_name, manifest_variant = _resolve_phase1_namespace(manifest_path, analysis_name)
    output_paths = get_output_paths(dataset_name, manifest_variant)
    ensure_output_directories(dataset_name, manifest_variant)
    
    results_dir = output_paths["results"]
    figures_dir = output_paths["figures"]
    features_dir = output_paths["features"]
    
    logger.info(f"\n{'='*70}")
    logger.info(f"DATASET: {dataset_name}")
    logger.info(f"MANIFEST VARIANT: {manifest_variant}")
    logger.info(f"Output paths:")
    logger.info(f"  Results: {results_dir}")
    logger.info(f"  Figures: {figures_dir}")
    logger.info(f"{'='*70}")
    
    if checkpoints is None:
        checkpoints = get_available_checkpoint_names()
    
    # Load manifest
    with open(manifest_path) as f:
        manifest_data = json.load(f)
    volumes = manifest_data["volumes"]
    binning_scheme = get_binning_scheme_from_manifest(manifest_data)
    bin_names = get_bin_name_map(binning_scheme)
    
    # Recompute bin labels with new formula (if not already done)
    # This ensures consistency even if manifest hasn't been regenerated
    bin_labels = np.array([v["anisotropy_bin"] for v in volumes])
    anisotropy_ratios = np.array([v.get("anisotropy_ratio", 1.0) for v in volumes])
    file_paths = [v["file_path"] for v in volumes]
    
    unique_datasets = sorted(set(v["dataset"] for v in volumes))
    
    logger.info(f"Loaded {len(volumes)} volumes")
    logger.info(f"Bin distribution: {dict(zip(*np.unique(bin_labels, return_counts=True)))}")
    logger.info(f"Datasets: {unique_datasets}")
    logger.info(f"Binning scheme: {binning_scheme}")
    
    # Auto-detect transfer dataset if not specified
    # Selects dataset with most samples (typically the one with labels)
    if transfer_dataset is None:
        dataset_counts = Counter(v["dataset"] for v in volumes)
        transfer_dataset = dataset_counts.most_common(1)[0][0]
        logger.info(f"Auto-detected transfer_dataset: '{transfer_dataset}' ({dataset_counts[transfer_dataset]} samples)")
    
    # Semantic labels (extract if needed)
    semantic_labels = None
    semantic_metadata = None
    label_names = None
    
    if run_semantic_probing or do_cross_bin_transfer:
        try:
            from semantic_label_builder import compute_semantic_task_labels
            
            # Filter to volumes with labels from transfer_dataset
            logger.info(f"\nExtracting semantic labels for {transfer_dataset}...")
            semantic_labels_full, sem_paths, sem_meta = compute_semantic_task_labels(
                manifest_path,
                mode="auto",
                filter_dataset=transfer_dataset,
                cache_dir=features_dir,
            )
            
            label_names = sem_meta.get("organ_names", [])
            semantic_metadata = sem_meta
            
            # Map back to full volume list
            sem_path_set = set(sem_paths)
            sem_path_to_index = {fp: idx for idx, fp in enumerate(sem_paths)}
            semantic_labels = np.zeros((len(volumes), semantic_labels_full.shape[1]), dtype=np.int32)
            
            for i, (fp, vol) in enumerate(zip(file_paths, volumes)):
                if fp in sem_path_set:
                    idx = sem_path_to_index[fp]
                    semantic_labels[i] = semantic_labels_full[idx]
            
            logger.info(f"Extracted semantic labels: {semantic_labels.shape}")
            logger.info(
                f"Informative semantic labels kept: {len(label_names)} / "
                f"{len(sem_meta.get('all_organ_names', label_names))}"
            )
            logger.info(f"Label names: {label_names[:5]}...")
            
        except Exception as e:
            logger.warning(f"Failed to extract semantic labels: {e}")
            semantic_labels = None
            semantic_metadata = None
    
    all_results = {
        OBSERVATIONAL_ANALYSIS_KEY: {},
        CONTROLLED_PERTURBATION_KEY: {},
        "metadata": {
            "dataset_name": dataset_name,
            "manifest_variant": manifest_variant,
            "feature_type": feature_type,
            "manifest": str(manifest_path),
            "n_volumes": len(volumes),
            "datasets": unique_datasets,
            "task_names": {
                "observational": OBSERVATIONAL_ANALYSIS_KEY,
                "interventional": CONTROLLED_PERTURBATION_KEY,
                "geometry": REPRESENTATION_GEOMETRY_KEY,
                "anisotropy_regression": SPACING_READOUT_KEY,
                "semantic_readout": SEMANTIC_READOUT_KEY,
                "cross_bin_transfer": CROSS_BIN_SEMANTIC_TRANSFER_KEY,
            },
            "legacy_aliases": {
                "setting_a": OBSERVATIONAL_ANALYSIS_KEY,
                "setting_b": CONTROLLED_PERTURBATION_KEY,
                "track_a": REPRESENTATION_GEOMETRY_KEY,
                "track_b": SPACING_READOUT_KEY,
                "track_b_semantic": SEMANTIC_READOUT_KEY,
                "cross_bin_transfer_semantic": CROSS_BIN_SEMANTIC_TRANSFER_KEY,
            },
        }
    }
    all_results["setting_a"] = all_results[OBSERVATIONAL_ANALYSIS_KEY]
    all_results["setting_b"] = all_results[CONTROLLED_PERTURBATION_KEY]
    
    # ==========================================================================
    # Observational bin analysis
    # ==========================================================================
    if run_setting_a:
        logger.info(f"\n{'='*70}")
        logger.info("OBSERVATIONAL BIN ANALYSIS")
        logger.info(f"{'='*70}")
        dataloaders_by_crop: Dict[int, Any] = {}
        
        for ckpt_name in checkpoints:
            if ckpt_name not in CHECKPOINTS:
                logger.warning(f"Checkpoint {ckpt_name} not found, skipping")
                continue
            
            logger.info(f"\n--- Checkpoint: {ckpt_name} ---")
            
            crop_size = _checkpoint_crop_size(ckpt_name)
            feature_cache_path = _feature_cache_path(features_dir, ckpt_name, feature_type, manifest_path)
            
            # Try loading from cache
            features = None
            if cache_features and feature_cache_path.exists():
                logger.info(f"  Loading cached features from {feature_cache_path.name}")
                try:
                    cached = np.load(feature_cache_path)
                    if len(cached["features"]) == len(volumes):
                        features = cached["features"]
                        logger.info(f"  Loaded features from cache: {features.shape}")
                    else:
                        logger.warning(f"  Cache size mismatch, re-extracting")
                except Exception as e:
                    logger.warning(f"  Failed to load cache: {e}, re-extracting")
            
            # Extract features if not loaded from cache
            if features is None:
                dataloader = _maybe_build_crop_dataloader(
                    dataloaders_by_crop,
                    manifest_path,
                    crop_size,
                    batch_size,
                    num_workers,
                )
                
                # Extract features
                logger.info(f"  Extracting {feature_type} features...")
                extractor = FeatureExtractor(ckpt_name)
                features_dict = extractor.extract_from_dataloader(dataloader, show_progress=True)
                features = features_dict[feature_type]
                
                logger.info(f"  Features shape: {features.shape}")
                
                # Save to cache
                if cache_features:
                    feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        feature_cache_path,
                        features=features,
                        bin_labels=bin_labels,
                        file_paths=file_paths,
                    )
                    logger.info(f"  Cached features to {feature_cache_path.name}")
            
            ckpt_results = {
                REPRESENTATION_GEOMETRY_KEY: None,
                SPACING_READOUT_KEY: None,
                SEMANTIC_READOUT_KEY: None,
                CROSS_BIN_SEMANTIC_TRANSFER_KEY: None,
            }
            
            # Representation geometry
            logger.info("\n  Representation Geometry")
            track_a = run_track_a_analysis(
                features=features,
                bin_labels=bin_labels,
                checkpoint_name=ckpt_name,
                feature_type=feature_type,
                compute_tsne=True,
                output_dir=results_dir,
                figures_dir=figures_dir,
                bin_names=bin_names,
                binning_scheme=binning_scheme,
                dataset_name=dataset_name,
            )
            representation_geometry = {
                "mean_cross_cka": track_a.cka_stats["mean_cross_cka"],
                "mean_mmd_rbf": track_a.observational_distances["summary"]["mean_mmd_rbf"],
                "mean_sliced_wasserstein": track_a.observational_distances["summary"]["mean_sliced_wasserstein"],
                "silhouette": track_a.geometry_metrics["silhouette_score"],
                "separability": track_a.geometry_metrics["separability_ratio"],
            }
            if track_a.balanced_bin_sensitivity is not None:
                representation_geometry["balanced_bin_sensitivity"] = {
                    "sample_count_per_bin": track_a.balanced_bin_sensitivity["sample_count_per_bin"],
                    "mean_cross_cka": track_a.balanced_bin_sensitivity["cka_stats"]["mean_cross_cka"],
                    "silhouette": track_a.balanced_bin_sensitivity["geometry"]["silhouette_score"],
                    "separability": track_a.balanced_bin_sensitivity["geometry"]["separability_ratio"],
                    "mean_mmd_rbf": track_a.balanced_bin_sensitivity["observational_distances"]["summary"]["mean_mmd_rbf"],
                    "mean_sliced_wasserstein": track_a.balanced_bin_sensitivity["observational_distances"]["summary"]["mean_sliced_wasserstein"],
                }
            ckpt_results[REPRESENTATION_GEOMETRY_KEY] = representation_geometry
            ckpt_results["track_a"] = representation_geometry
            
            logger.info("\n  Spacing Readout")
            spacing_reg = spacing_regression_probe(
                features=features,
                anisotropy_ratios=anisotropy_ratios,
                bin_labels=bin_labels,
            )
            spacing_reg.checkpoint_name = ckpt_name
            spacing_reg.feature_type = feature_type
            spacing_readout = serialize_spacing_regression_results(spacing_reg)
            balanced_idx, _, _ = _select_balanced_bin_indices(bin_labels, random_state=42)
            if balanced_idx is not None:
                balanced_spacing_reg = spacing_regression_probe(
                    features=features[balanced_idx],
                    anisotropy_ratios=anisotropy_ratios[balanced_idx],
                    bin_labels=bin_labels[balanced_idx],
                )
                spacing_readout["balanced_bin_sensitivity"] = {
                    "sample_count_per_bin": int(min(np.sum(bin_labels == b) for b in np.unique(bin_labels))),
                    **serialize_spacing_regression_results(balanced_spacing_reg),
                }
            ckpt_results[SPACING_READOUT_KEY] = spacing_readout
            ckpt_results["spacing_regression"] = spacing_readout
            ckpt_results["track_b"] = spacing_readout
            spacing_results_dir = get_checkpoint_feature_dir(results_dir, ckpt_name, feature_type)
            spacing_results_dir.mkdir(parents=True, exist_ok=True)
            spacing_results_path = spacing_results_dir / "anisotropy_regression.json"
            with open(spacing_results_path, "w") as f:
                json.dump(
                    {
                        "checkpoint": ckpt_name,
                        "feature_type": feature_type,
                        "task_name": SPACING_READOUT_KEY,
                        "legacy_alias": "spacing_regression",
                        "binning_scheme": binning_scheme,
                        SPACING_READOUT_KEY: spacing_readout,
                        "spacing_regression": spacing_readout,
                    },
                    f,
                    indent=2,
                )

            if run_semantic_probing and semantic_labels is not None and semantic_labels.shape[1] > 0:
                transfer_mask = np.array([v["dataset"] == transfer_dataset for v in volumes])
                if transfer_mask.sum() > 50:
                    transfer_features = features[transfer_mask]
                    transfer_bins = bin_labels[transfer_mask]
                    transfer_semantic = semantic_labels[transfer_mask]

                    logger.info("\n  Semantic Readout")
                    try:
                        semantic_probe = run_semantic_probing_fn(
                            features=transfer_features,
                            bin_labels=transfer_bins,
                            semantic_labels=transfer_semantic,
                            checkpoint_name=ckpt_name,
                            feature_type=feature_type,
                            label_names=label_names,
                            anisotropy_ratios=anisotropy_ratios[transfer_mask],
                            output_dir=results_dir,
                            analysis_name=dataset_name,
                            manifest_variant=manifest_variant,
                        )
                        semantic_readout = {
                            "metric_name": "balanced_accuracy",
                            "mean_balanced_accuracy": semantic_probe.mean_balanced_accuracy,
                            "bin_balanced_accuracies": semantic_probe.bin_balanced_accuracies,
                            "n_labels": len(label_names),
                        }
                        if semantic_probe.balanced_bin_sensitivity is not None:
                            semantic_readout["balanced_bin_sensitivity"] = semantic_probe.balanced_bin_sensitivity
                        ckpt_results[SEMANTIC_READOUT_KEY] = semantic_readout
                        ckpt_results["track_b_semantic"] = semantic_readout
                        if semantic_metadata is not None:
                            semantic_readout["label_prevalence_filter"] = semantic_metadata.get("prevalence_filter")
                            semantic_readout["excluded_labels"] = semantic_metadata.get("excluded_labels", [])
                    except Exception as e:
                        logger.warning(f"Semantic probing failed: {e}")
            
            # Semantic probing + transfer
            if run_semantic_probing and semantic_labels is not None and semantic_labels.shape[1] > 0:
                # Filter to transfer_dataset
                transfer_mask = np.array([v["dataset"] == transfer_dataset for v in volumes])
                if transfer_mask.sum() > 50:
                    transfer_features = features[transfer_mask]
                    transfer_bins = bin_labels[transfer_mask]
                    transfer_semantic = semantic_labels[transfer_mask]

                    logger.info(f"\n  Cross-Bin Semantic Transfer on {transfer_dataset}")
                    
                    try:
                        semantic_transfer = run_multilabel_cross_bin_transfer(
                            features=transfer_features,
                            bin_labels=transfer_bins,
                            semantic_labels=transfer_semantic,
                            checkpoint_name=ckpt_name,
                            feature_type=feature_type,
                            dataset=transfer_dataset,
                            output_dir=results_dir,
                            analysis_name=dataset_name,
                            manifest_variant=manifest_variant,
                        )
                        cross_bin_transfer = {
                            "metric_name": semantic_transfer.metric_name,
                            "transfer_gap": semantic_transfer.transfer_gap,
                            "cross_bin_accuracy": semantic_transfer.cross_bin_accuracy,
                        }
                        ckpt_results[CROSS_BIN_SEMANTIC_TRANSFER_KEY] = cross_bin_transfer
                        ckpt_results["cross_bin_transfer_semantic"] = cross_bin_transfer
                    except Exception as e:
                        logger.warning(f"Semantic transfer failed: {e}")
            
            all_results[OBSERVATIONAL_ANALYSIS_KEY][ckpt_name] = ckpt_results
    
    # ==========================================================================
    # Controlled spacing perturbation
    # ==========================================================================
    if run_setting_b:
        logger.info(f"\n{'='*70}")
        logger.info("CONTROLLED SPACING PERTURBATION")
        logger.info(f"{'='*70}")
        
        try:
            from perturbation_robustness_analysis import run_controlled_perturbation
            
            setting_b_cache = output_paths["cache_root"]
            
            for ckpt_name in checkpoints:
                if ckpt_name not in CHECKPOINTS:
                    continue
                
                logger.info(f"\n--- Checkpoint: {ckpt_name} ---")
                
                setting_b_results = run_controlled_perturbation(
                    manifest_path=manifest_path,
                    checkpoint_name=ckpt_name,
                    feature_type=feature_type,
                    output_dir=results_dir,
                    figures_dir=figures_dir,
                    cache_dir=setting_b_cache,
                    analysis_name=dataset_name,
                )
                
                if setting_b_results:
                    all_results[CONTROLLED_PERTURBATION_KEY][ckpt_name] = {
                        "representation_drift": setting_b_results.representation_drift,
                        "representation_drift_std": setting_b_results.representation_drift_std,
                        "cka_matrix": setting_b_results.cka_matrix,
                        "matched_semantic_probing": setting_b_results.matched_semantic_probing,
                        "matched_semantic_transfer": setting_b_results.matched_semantic_transfer,
                        "semantic_metadata": setting_b_results.semantic_metadata,
                        "n_volumes": setting_b_results.n_source_volumes,
                    }
        except Exception as e:
            logger.error(f"Controlled spacing perturbation failed: {e}")
            import traceback
            traceback.print_exc()
    
    evaluated_checkpoints = _ordered_checkpoint_names(
        list(
            set(all_results[OBSERVATIONAL_ANALYSIS_KEY].keys())
            | set(all_results[CONTROLLED_PERTURBATION_KEY].keys())
        )
    )
    checkpoint_scope = _build_checkpoint_scope(checkpoints, evaluated_checkpoints)
    all_results["metadata"].update(checkpoint_scope)

    # Save comprehensive results with scope-aware naming.
    results_file_name = _get_aggregate_results_filename(feature_type, checkpoint_scope)
    results_path = results_dir / results_file_name
    with open(results_path, 'w') as f:
        # Convert any remaining numpy types
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            return obj
        
        json.dump(convert_for_json(all_results), f, indent=2)
    
    logger.info(f"\nSaved comprehensive results to {results_path}")
    write_results_catalog(results_dir, dataset_name, manifest_path)
    
    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 1 EVALUATION SUMMARY")
    logger.info(f"{'='*70}")
    
    if run_setting_a:
        logger.info("\nObservational Bin Analysis Results:")
        logger.info(
            f"{'Checkpoint':<15} {'CKA':>8} {'MMD':>8} {'SWD':>8} {'Silh':>8} {'R2':>8} {'SemAcc':>8} {'SemGap':>8}"
        )
        logger.info("-" * 96)
        for ckpt, res in all_results[OBSERVATIONAL_ANALYSIS_KEY].items():
            ta = res.get(REPRESENTATION_GEOMETRY_KEY, {})
            sr = res.get(SPACING_READOUT_KEY) or {}
            sp = res.get(SEMANTIC_READOUT_KEY) or {}
            tr = res.get(CROSS_BIN_SEMANTIC_TRANSFER_KEY) or {}
            logger.info(
                f"{ckpt:<15} "
                f"{ta.get('mean_cross_cka', 0):.4f}  "
                f"{ta.get('mean_mmd_rbf', 0):.4f}  "
                f"{ta.get('mean_sliced_wasserstein', 0):.4f}  "
                f"{ta.get('silhouette', 0):.4f}  "
                f"{sr.get('r2_score', 0):.4f}  "
                f"{sp.get('mean_balanced_accuracy', 0):.4f}  "
                f"{tr.get('transfer_gap', 0):.4f}"
            )
            ta_bal = ta.get("balanced_bin_sensitivity") or {}
            sr_bal = sr.get("balanced_bin_sensitivity") or {}
            sp_bal = sp.get("balanced_bin_sensitivity") or {}
            if ta_bal or sr_bal or sp_bal:
                logger.info(
                    "  Balanced sensitivity: n/bin=%s, CKA=%.4f, MMD=%.4f, SWD=%.4f, Silh=%.4f, R2=%.4f, SemAcc=%.4f",
                    ta_bal.get("sample_count_per_bin", sr_bal.get("sample_count_per_bin", sp_bal.get("sample_count_per_bin", 0))),
                    ta_bal.get("mean_cross_cka", 0.0),
                    ta_bal.get("mean_mmd_rbf", 0.0),
                    ta_bal.get("mean_sliced_wasserstein", 0.0),
                    ta_bal.get("silhouette", 0.0),
                    sr_bal.get("r2_score", 0.0),
                    sp_bal.get("mean_balanced_accuracy", 0.0),
                )
            if sr.get("split_strategy"):
                logger.info("  Spacing regression split policy: %s", sr["split_strategy"])
    
    if run_setting_b:
        logger.info("\nControlled Spacing Perturbation Results:")
        for ckpt, res in all_results[CONTROLLED_PERTURBATION_KEY].items():
            logger.info(f"  {ckpt}:")
            for spacing, drift in res.get("representation_drift", {}).items():
                drift_std = res.get("representation_drift_std", {}).get(spacing, 0)
                logger.info(f"    Drift to {spacing}: {drift:.4f} ± {drift_std:.4f}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Spacing/Anisotropy Robustness Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python phase1_evaluation_pipeline.py --full -m ../data_manifests/phase1_anisotropy_robustness/abdomenatlas/original_bins/manifest_sampled.json -a abdomenatlas -c Med3DINO_REL_c96 -f cls\n"
            "  python phase1_evaluation_pipeline.py --full --no-setting-b -m ../data_manifests/phase1_anisotropy_robustness/totalsegmentermri/original_bins/manifest_sampled.json -a totalsegmentermri -c Med3DINO_REL_c96 -f avg_pool\n"
            "  python phase1_evaluation_pipeline.py --setting-b-only -m ../data_manifests/phase1_anisotropy_robustness/totalsegmenter_ct/original_bins/manifest_sampled.json -a totalsegmenter_ct -c Med3DINO_REL_c96 -f cls"
        ),
    )
    parser.add_argument(
        "-m", "--manifest",
        type=Path,
        default=None,
        help="Path to manifest JSON (for example: abdomenatlas/original_bins/manifest_sampled.json under the phase1 manifest directory)",
    )
    parser.add_argument(
        "-c", "--checkpoints",
        nargs="*",
        type=str,
        default=None,
        help="Checkpoints to evaluate (default: all available)",
    )
    parser.add_argument(
        "-f", "--feature-type",
        type=str,
        choices=["cls", "avg_pool", "multilayer"],
        default="cls",
        help="Feature type to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--no-cache-features",
        action="store_true",
        help="Disable feature caching (re-extract every run)",
    )
    parser.add_argument(
        "--track-a-only",
        action="store_true",
        help="Only run representation geometry (legacy Track A)",
    )
    parser.add_argument(
        "--track-b-only",
        action="store_true",
        help="Only run spacing readout (legacy Track B)",
    )
    
    # New comprehensive evaluation options
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full Phase 1 evaluation (observational analysis, controlled perturbation, semantic readout, cross-bin transfer)",
    )
    parser.add_argument(
        "--no-setting-b",
        action="store_true",
        help="Skip controlled spacing perturbation in full mode",
    )
    parser.add_argument(
        "--no-cross-bin-transfer",
        action="store_true",
        help="Skip cross-bin transfer experiments in full mode",
    )
    parser.add_argument(
        "--no-semantic-probing",
        action="store_true",
        help="Skip semantic label probing in full mode",
    )
    parser.add_argument(
        "--transfer-dataset",
        type=str,
        default=None,
        help="Dataset for cross-bin transfer experiments (auto-detect largest dataset if not specified)",
    )
    parser.add_argument(
        "--setting-b-only",
        action="store_true",
        help="Run ONLY controlled spacing perturbation - use after observational analysis completes",
    )
    parser.add_argument(
        "--precompute-labels",
        action="store_true",
        help="Pre-compute semantic label cache and exit. Run once before parallel GPU evaluation.",
    )
    parser.add_argument(
        "--semantic-label-workers",
        type=int,
        default=4,
        help="Worker count for semantic label precomputation.",
    )
    parser.add_argument(
        "--semantic-cache-flush-every",
        type=int,
        default=128,
        help="Flush partial semantic label cache every N completed volumes.",
    )
    parser.add_argument(
        "--warmup-observational-cache",
        action="store_true",
        help="Materialize deterministic observational preprocessing caches for the required crop sizes and exit.",
    )
    parser.add_argument(
        "-a", "--analysis-name",
        type=str,
        default="default",
        help="Dataset name override for output routing (e.g., 'abdomenatlas', 'abdomenct1k').",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Optional external checkpoint root that contains 20k/, 42k/, 62k/, and 3dinov2/",
    )
    
    args = parser.parse_args()

    if args.checkpoint_root is not None:
        os.environ["MED3DINO_CHECKPOINT_ROOT"] = str(args.checkpoint_root.resolve())
    
    # Set manifest path
    if args.manifest is None:
        manifest_path = get_phase1_manifest_path("abdomenatlas", "sampled", "original_bins")
    else:
        manifest_path = args.manifest
    
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)
    
    dataset_name, manifest_variant = _resolve_phase1_namespace(manifest_path, args.analysis_name)
    output_paths = get_output_paths(dataset_name, manifest_variant)
    ensure_output_directories(dataset_name, manifest_variant)
    
    # Pre-compute semantic label cache and exit
    if args.precompute_labels:
        from semantic_label_builder import compute_semantic_task_labels
        from collections import Counter
        
        features_dir = output_paths["features"]
        
        with open(manifest_path) as f:
            manifest_data = json.load(f)
        volumes = manifest_data["volumes"]
        dataset_counts = Counter(v["dataset"] for v in volumes)
        transfer_dataset = args.transfer_dataset or dataset_counts.most_common(1)[0][0]
        
        logger.info(f"Pre-computing semantic labels for '{transfer_dataset}' (cache_dir: {features_dir})")
        semantic_labels_full, sem_paths, sem_meta = compute_semantic_task_labels(
            manifest_path,
            mode="multi_binary",
            filter_dataset=transfer_dataset,
            cache_dir=features_dir,
            num_workers=args.semantic_label_workers,
            cache_flush_every=args.semantic_cache_flush_every,
        )
        logger.info(
            f"Done: {semantic_labels_full.shape[0]} volumes, "
            f"{len(sem_meta.get('organ_names', []))} informative labels cached"
        )
        return

    if args.warmup_observational_cache:
        checkpoints = args.checkpoints if args.checkpoints else get_available_checkpoint_names()
        logger.info(
            "Warming observational preprocessing cache for %s",
            ", ".join(_ordered_checkpoint_names(checkpoints)),
        )
        _warmup_required_observational_caches(
            manifest_path=manifest_path,
            checkpoints=checkpoints,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        return
    
    # Controlled perturbation only mode (after observational analysis completes)
    if args.setting_b_only:
        logger.info(f"{'='*70}")
        logger.info(f"CONTROLLED SPACING PERTURBATION ONLY (dataset: {dataset_name}, variant: {manifest_variant})")
        logger.info(f"{'='*70}")
        
        from perturbation_robustness_analysis import run_controlled_perturbation
        
        checkpoints = args.checkpoints if args.checkpoints else get_available_checkpoint_names()
        
        for ckpt_name in checkpoints:
            if ckpt_name not in CHECKPOINTS:
                logger.warning(f"Unknown checkpoint: {ckpt_name}")
                continue
            
            logger.info(f"\n--- Checkpoint: {ckpt_name} ---")
            
            try:
                setting_b_results = run_controlled_perturbation(
                    manifest_path=manifest_path,
                    checkpoint_name=ckpt_name,
                    feature_type=args.feature_type,
                    output_dir=None,
                    figures_dir=None,
                    cache_dir=output_paths["cache_root"],
                    analysis_name=dataset_name,
                )
                
                if setting_b_results:
                    logger.info(f"Controlled spacing perturbation completed for {ckpt_name}")
                    for spacing, drift in setting_b_results.representation_drift.items():
                        drift_std = setting_b_results.representation_drift_std.get(spacing, 0)
                        logger.info(f"  Drift to {spacing}: {drift:.4f} ± {drift_std:.4f}")
            except Exception as e:
                logger.error(f"Controlled spacing perturbation failed for {ckpt_name}: {e}")
                import traceback
                traceback.print_exc()

        write_results_catalog(output_paths["results"], dataset_name, manifest_path)
        
        return
    
    # Full evaluation mode (new comprehensive approach)
    if args.full:
        run_full_phase1_evaluation(
            manifest_path=manifest_path,
            checkpoints=args.checkpoints,
            feature_type=args.feature_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            run_setting_a=True,
            run_setting_b=not args.no_setting_b,
            do_cross_bin_transfer=not args.no_cross_bin_transfer,
            run_semantic_probing=not args.no_semantic_probing,
            transfer_dataset=args.transfer_dataset,
            cache_features=not args.no_cache_features,
            analysis_name=dataset_name,
        )
        return
    
    # Legacy mode (original Track A / Track B entrypoints)
    # Determine which tracks to run
    run_track_a = not args.track_b_only
    run_track_b = not args.track_a_only
    
    if args.checkpoints and len(args.checkpoints) == 1:
        # Single checkpoint mode
        run_single_checkpoint(
            checkpoint_name=args.checkpoints[0],
            manifest_path=manifest_path,
            feature_type=args.feature_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            run_track_a=run_track_a,
            run_track_b=run_track_b,
            cache_features=not args.no_cache_features,
            analysis_name=dataset_name,
        )
    else:
        # Multi-checkpoint mode
        run_all_checkpoints(
            manifest_path=manifest_path,
            feature_type=args.feature_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            checkpoints=args.checkpoints,
            cache_features=not args.no_cache_features,
            analysis_name=dataset_name,
        )


if __name__ == "__main__":
    main()

