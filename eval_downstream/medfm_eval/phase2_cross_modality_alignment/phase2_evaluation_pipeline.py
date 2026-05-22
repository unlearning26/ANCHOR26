#!/usr/bin/env python

"""Run Phase 2 manifest auditing and primary-metric evaluation.

This CLI validates the organ-level Phase 2 manifest, builds cross-modal organ
cohorts, optionally loads a precomputed embeddings bundle, and writes the
canonical Phase 2 result artifacts under outputs_phase2.

Usage examples:
    python phase2_evaluation_pipeline.py \
        -m ../data_manifests/phase2_cross_modality_alignment/totalsegmenter_ct_mr_anchor/core/manifest_sampled.json

    python phase2_evaluation_pipeline.py \
        -m ../data_manifests/phase2_cross_modality_alignment/totalsegmenter_ct_mr_anchor/core/manifest_sampled.json \
        --checkpoint-name Med3DINO_REL_c96 \
        --embeddings-npz outputs_phase2/totalsegmenter_ct_mr_anchor/phase2/core/features/Med3DINO_REL_c96/cls/phase2_organ_cls_embeddings.npz
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from phase2_config import (  # noqa: E402
    DEFAULT_MANIFEST_VARIANT,
    DEFAULT_REQUIRED_MODALITIES,
    ensure_output_directories,
    get_checkpoint_metrics_path,
    get_dataset_name_from_manifest_path,
    get_phase2_manifest_path,
    get_manifest_variant_from_manifest_path,
    get_output_paths,
    normalize_feature_type,
    normalize_manifest_variant,
)
from correspondence_builder import CorrespondenceConfig, build_population_cohorts  # noqa: E402
from cross_modal_alignment_analysis import compute_anatomy_over_modality_margin, compute_balanced_lisi  # noqa: E402
from phase2_data_loader import (  # noqa: E402
    load_phase2_manifest,
    summarize_phase2_manifest,
    validate_phase2_manifest,
)
from retrieval_analysis import compute_bidirectional_cross_modal_retrieval  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MANIFEST_KIND_BY_FILE_NAME = {
    "manifest_full.json": "full",
    "manifest_sampled.json": "sampled",
    "manifest_meta.json": "meta",
}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_optional_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_existing_path(path_arg: str | Path) -> Path | None:
    candidate = Path(path_arg).expanduser()
    if candidate.exists():
        return candidate.resolve()

    if not candidate.is_absolute():
        project_candidate = (PROJECT_ROOT / candidate).resolve()
        if project_candidate.exists():
            return project_candidate

    return None


def _embedding_summary_path(npz_path: Path) -> Path:
    return npz_path.with_name(npz_path.stem + "_summary.json")


def _manifest_kind_from_path(manifest_path: Path) -> str | None:
    return MANIFEST_KIND_BY_FILE_NAME.get(manifest_path.name)


def _infer_surface_from_embeddings_path(embeddings_path: Path) -> Tuple[str | None, str | None]:
    parts = embeddings_path.resolve().parts
    try:
        outputs_index = parts.index("outputs_phase2")
    except ValueError:
        return (None, None)

    if len(parts) <= outputs_index + 3:
        return (None, None)

    analysis_name = parts[outputs_index + 1]
    manifest_variant = parts[outputs_index + 3] if parts[outputs_index + 2] == "phase2" else None
    return (analysis_name, manifest_variant)


def _resolve_manifest_path(
    manifest_arg: str,
    analysis_name: str | None,
    manifest_variant: str | None,
    embeddings_npz: str | None,
) -> Path:
    manifest_path = Path(manifest_arg).expanduser()
    resolved_manifest_path = _resolve_existing_path(manifest_path)
    if resolved_manifest_path is not None:
        return resolved_manifest_path

    is_shorthand_manifest = not manifest_path.is_absolute() and len(manifest_path.parts) == 1
    embeddings_path = _resolve_existing_path(embeddings_npz) if embeddings_npz else None

    if is_shorthand_manifest and embeddings_path is not None:
        summary_path = _embedding_summary_path(embeddings_path)
        summary_payload = _load_optional_json(summary_path)
        summary_manifest = summary_payload.get("manifest_path") if summary_payload else None
        if summary_manifest:
            resolved_manifest = Path(str(summary_manifest)).expanduser().resolve()
            if resolved_manifest.exists():
                logger.info(
                    "Resolved shorthand manifest %s using embedding summary %s",
                    manifest_path,
                    summary_path,
                )
                return resolved_manifest

    manifest_kind = _manifest_kind_from_path(manifest_path)
    inferred_analysis_name = analysis_name
    inferred_manifest_variant = normalize_manifest_variant(manifest_variant or DEFAULT_MANIFEST_VARIANT)
    if embeddings_path is not None:
        embedding_analysis_name, embedding_manifest_variant = _infer_surface_from_embeddings_path(embeddings_path)
        if inferred_analysis_name is None:
            inferred_analysis_name = embedding_analysis_name
        if manifest_variant is None and embedding_manifest_variant:
            inferred_manifest_variant = normalize_manifest_variant(embedding_manifest_variant)

    if is_shorthand_manifest and inferred_analysis_name and manifest_kind:
        canonical_manifest = get_phase2_manifest_path(
            dataset_name=inferred_analysis_name,
            manifest_kind=manifest_kind,
            manifest_variant=inferred_manifest_variant,
        ).resolve()
        if canonical_manifest.exists():
            logger.info(
                "Resolved shorthand manifest %s to canonical Phase 2 manifest %s",
                manifest_path,
                canonical_manifest,
            )
            return canonical_manifest

    hint = ""
    if is_shorthand_manifest and inferred_analysis_name and manifest_kind:
        hinted_manifest = get_phase2_manifest_path(
            dataset_name=inferred_analysis_name,
            manifest_kind=manifest_kind,
            manifest_variant=inferred_manifest_variant,
        ).resolve()
        hint = f" Tried canonical Phase 2 manifest {hinted_manifest}."
    elif embeddings_path is not None:
        hint = f" Checked embedding summary {_embedding_summary_path(embeddings_path)}."

    raise FileNotFoundError(f"Manifest not found: {manifest_path.resolve()}.{hint}")


def _load_embedding_bundle(npz_path: Path, manifest_samples: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    payload = np.load(npz_path, allow_pickle=True)
    if "features" not in payload:
        raise ValueError(f"Embedding bundle {npz_path} must contain a 'features' array")

    features = np.asarray(payload["features"])
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


def _resolve_embeddings_path(embeddings_arg: str) -> Path:
    resolved_embeddings_path = _resolve_existing_path(embeddings_arg)
    if resolved_embeddings_path is not None:
        return resolved_embeddings_path
    raise FileNotFoundError(f"Embeddings bundle not found: {Path(embeddings_arg).expanduser().resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 cross-modality alignment scaffold",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python phase2_evaluation_pipeline.py -m ../data_manifests/phase2_cross_modality_alignment/totalsegmenter_ct_mr_anchor/core/manifest_sampled.json\n"
            "  python phase2_evaluation_pipeline.py -m ../data_manifests/phase2_cross_modality_alignment/totalsegmenter_ct_mr_anchor/core/manifest_sampled.json --embeddings-npz outputs_phase2/totalsegmenter_ct_mr_anchor/phase2/core/features/Med3DINO_REL_c96/cls/phase2_organ_cls_embeddings.npz"
        ),
    )
    parser.add_argument("-m", "--manifest", required=True, help="Path to the Phase 2 manifest JSON")
    parser.add_argument("-a", "--analysis-name", default=None, help="Dataset namespace for outputs")
    parser.add_argument(
        "--manifest-variant",
        default=None,
        help="Optional Phase 2 manifest variant label for shorthand manifest resolution",
    )
    parser.add_argument(
        "--required-modalities",
        nargs="+",
        default=list(DEFAULT_REQUIRED_MODALITIES),
        help="Required modalities for cross-modal cohort support",
    )
    parser.add_argument(
        "--min-samples-per-modality",
        type=int,
        default=5,
        help="Minimum support per organ and modality for cohort retention",
    )
    parser.add_argument(
        "--embeddings-npz",
        default=None,
        help="Optional NPZ bundle with 'features' and optional 'sample_ids'",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=None,
        help="Optional checkpoint label used to write per-checkpoint primary metrics",
    )
    parser.add_argument(
        "--feature-type",
        default="cls",
        help="Feature family label for per-checkpoint metrics output",
    )
    parser.add_argument(
        "--max-samples-per-pool",
        type=int,
        default=None,
        help="Optional cap for per-query positive and negative pools in the anatomy-over-modality metric",
    )
    parser.add_argument(
        "--max-queries-per-organ",
        type=int,
        default=None,
        help="Optional cap for balanced retrieval queries per organ",
    )
    parser.add_argument(
        "--max-targets-per-organ",
        type=int,
        default=None,
        help="Optional cap for balanced retrieval targets per organ",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=1000,
        help="Bootstrap resamples for the anatomy-over-modality margin confidence interval",
    )
    parser.add_argument(
        "--max-lisi-samples-per-group",
        type=int,
        default=None,
        help="Optional cap for balanced LISI sampling per organ and modality group",
    )
    parser.add_argument(
        "--lisi-perplexity",
        type=float,
        default=30.0,
        help="Target LISI perplexity for local-neighborhood weighting",
    )
    parser.add_argument(
        "--lisi-k-neighbors",
        type=int,
        default=None,
        help="Optional explicit neighbor count for LISI; defaults to ceil(3 * perplexity)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for balanced sampling in Phase 2 primary metrics",
    )
    args = parser.parse_args()

    requested_manifest_variant = normalize_manifest_variant(args.manifest_variant) if args.manifest_variant else None
    manifest_path = _resolve_manifest_path(
        args.manifest,
        args.analysis_name,
        requested_manifest_variant,
        args.embeddings_npz,
    )
    analysis_name = args.analysis_name or get_dataset_name_from_manifest_path(manifest_path)
    manifest_variant = get_manifest_variant_from_manifest_path(
        manifest_path,
        fallback=requested_manifest_variant or DEFAULT_MANIFEST_VARIANT,
    )

    ensure_output_directories(analysis_name, manifest_variant)
    output_paths = get_output_paths(analysis_name, manifest_variant)

    samples = load_phase2_manifest(manifest_path)
    validation = validate_phase2_manifest(samples, args.required_modalities)
    manifest_summary = summarize_phase2_manifest(samples)
    cohorts = build_population_cohorts(
        samples,
        CorrespondenceConfig(
            required_modalities=tuple(args.required_modalities),
            min_samples_per_modality=args.min_samples_per_modality,
        ),
    )

    _write_json(
        output_paths["results"] / "phase2_manifest_summary.json",
        {
            "analysis_name": analysis_name,
            "manifest_path": str(manifest_path),
            "manifest_variant": manifest_variant,
            "validation": validation,
            "summary": manifest_summary,
        },
    )
    _write_json(output_paths["results"] / "phase2_cohort_summary.json", cohorts)

    logger.info("Wrote manifest summary to %s", output_paths["results"])

    if not args.embeddings_npz:
        logger.info("No embeddings bundle provided; Phase 2 scaffold run completed after manifest and cohort audit")
        return

    embeddings_path = _resolve_embeddings_path(args.embeddings_npz)
    checkpoint_name = args.checkpoint_name
    feature_type = normalize_feature_type(args.feature_type)
    if not checkpoint_name:
        raise ValueError(
            "--checkpoint-name is required whenever --embeddings-npz is provided. "
            "Canonical Phase 2 metric files live under results/<checkpoint>/phase2_primary_metrics.json; "
            "root-level results are reserved for manifest/cohort summaries and aggregated checkpoint comparisons."
        )
    features_by_id = _load_embedding_bundle(embeddings_path, samples)
    supported_organs = sorted(cohorts.get("supported_organs", {}).keys())

    anatomy_margin = compute_anatomy_over_modality_margin(
        features_by_id,
        samples,
        supported_organs=supported_organs,
        max_samples_per_pool=args.max_samples_per_pool,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )

    retrieval = compute_bidirectional_cross_modal_retrieval(
        features_by_id,
        samples,
        supported_organs=supported_organs,
        max_queries_per_organ=args.max_queries_per_organ,
        max_targets_per_organ=args.max_targets_per_organ,
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )

    balanced_lisi = compute_balanced_lisi(
        features_by_id,
        samples,
        supported_organs=supported_organs,
        required_modalities=args.required_modalities,
        max_samples_per_group=args.max_lisi_samples_per_group,
        perplexity=args.lisi_perplexity,
        k_neighbors=args.lisi_k_neighbors,
        seed=args.seed,
    )

    primary_metrics = {
        "analysis_name": analysis_name,
        "checkpoint_name": checkpoint_name,
        "manifest_path": str(manifest_path),
        "embeddings_npz": str(embeddings_path),
        "feature_contract": "cls_only_phase2_v2" if feature_type == "cls" else f"phase2_{feature_type}_v1",
        "metric_contract": {
            "primary_metrics": [
                "balanced_cross_modal_retrieval.bidirectional_mean.top@1",
                "balanced_cross_modal_retrieval.bidirectional_mean.top@5",
                "balanced_cross_modal_retrieval.bidirectional_mean.map",
                "balanced_lisi.overall.modality_ilisi.mean",
                "balanced_lisi.overall.organ_clisi.mean",
            ],
            "diagnostic_metrics": [
                "anatomy_over_modality_margin",
            ],
            "exploratory_metrics_not_run_by_default": [
                "cross_modal_cka",
                "silhouette",
                "2d_embedding_plots",
            ],
        },
        "required_modalities": list(args.required_modalities),
        "cohorts": {
            "n_supported_organs": cohorts["n_supported_organs"],
            "n_dropped_organs": cohorts["n_dropped_organs"],
            "supported_organs": supported_organs,
        },
        "anatomy_over_modality_margin": anatomy_margin,
        "balanced_cross_modal_retrieval": retrieval,
        "balanced_lisi": balanced_lisi,
    }

    metrics_output_path = get_checkpoint_metrics_path(output_paths["results"], checkpoint_name, feature_type)
    _write_json(metrics_output_path, primary_metrics)
    logger.info("Wrote Phase 2 primary metrics to %s", metrics_output_path.parent)


if __name__ == "__main__":
    main()