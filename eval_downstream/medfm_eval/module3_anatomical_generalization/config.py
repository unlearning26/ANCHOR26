"""Canonical constants and path helpers for Module 3.

This module centralizes the filesystem contract between Module 3 and its
upstream organ-level manifest and feature-bundle sources. At present the
canonical bundle layout lives under the Phase 2 package, but that is a storage
contract rather than a claim that every Module 3 source surface is scientifically
valid for Phase 2 paired CT-MR alignment.

It should be the first place to update when Module 3 manifests, feature bundle
locations, or output naming conventions change.

Required identifiers for most helpers:
    - dataset_name: Module 3 dataset namespace, for example
            ``totalsegmenter_ct_mr_anchor`` or ``mmwhs_ct_mr``.
        - manifest_variant: usually ``core``.
    - checkpoint_name: a registered checkpoint label such as ``Med3DINO_REL_c96``.
        - feature_type: one of ``cls``, ``avg_pool``, or ``multilayer``.

Examples:
    >>> get_module3_manifest_path("totalsegmenter_ct_mr_anchor", "sampled", "core")
    PosixPath('.../data_manifests/module3_anatomical_generalization/totalsegmenter_ct_mr_anchor/core/manifest_sampled.json')

        >>> get_phase2_feature_npz_path("totalsegmenter_ct_mr_anchor", "core", "Med3DINO_REL_c96", "cls")
        PosixPath('.../phase2_cross_modality_alignment/outputs_phase2/totalsegmenter_ct_mr_anchor/phase2/core/features/Med3DINO_REL_c96/cls/phase2_organ_cls_embeddings.npz')
"""

from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).parents[3]
DOWNSTREAM_ROOT = PROJECT_ROOT / "eval_downstream"
MEDFM_EVAL_ROOT = DOWNSTREAM_ROOT / "medfm_eval"

PHASE2_ROOT = MEDFM_EVAL_ROOT / "phase2_cross_modality_alignment"
PHASE2_OUTPUTS = PHASE2_ROOT / "outputs_phase2"
PHASE2_MANIFESTS = MEDFM_EVAL_ROOT / "data_manifests" / "phase2_cross_modality_alignment"

MODULE3_ROOT = MEDFM_EVAL_ROOT / "module3_anatomical_generalization"
MODULE3_OUTPUTS = MODULE3_ROOT / "outputs_module3"
MODULE3_MANIFESTS = MEDFM_EVAL_ROOT / "data_manifests" / "module3_anatomical_generalization"

MODULE3_NAME = "module3"
DEFAULT_MODULE3_DATASET = "totalsegmenter_ct_mr_anchor"
DEFAULT_MANIFEST_VARIANT = "core"
DEFAULT_REQUIRED_MODALITIES = ("ct", "mr")
DEFAULT_FEATURE_TYPE = "cls"
DEFAULT_MIN_HOLDOUT_ORGANS = 2
DEFAULT_MIN_SAMPLES_PER_MODALITY = 5
DEFAULT_FEW_SHOT_SUPPORT_PER_MODALITY = 2
DEFAULT_FEW_SHOT_QUERY_PER_MODALITY = None
DEFAULT_FEW_SHOT_SEEDS = (42, 123, 456)

FEATURE_TYPES = ("cls", "avg_pool", "multilayer")
FEATURE_TYPE_ALIASES = {
    "cls": "cls",
    "avg_pool": "avg_pool",
    "avgpool": "avg_pool",
    "avg-pool": "avg_pool",
    "multilayer": "multilayer",
    "multi_layer": "multilayer",
    "multi-layer": "multilayer",
}
FEATURE_EMBEDDING_FILE_NAMES = {
    "cls": "phase2_organ_cls_embeddings.npz",
    "avg_pool": "phase2_organ_avg_pool_embeddings.npz",
    "multilayer": "phase2_organ_multilayer_embeddings.npz",
}
MANIFEST_VARIANT_ALIASES = {
    "core": "core",
    "default": "core",
    "shared_organs": "core",
    "paired": "paired",
    "paired_core": "paired",
}
MANIFEST_FILE_NAMES = {
    "full": "manifest_full.json",
    "sampled": "manifest_sampled.json",
    "meta": "manifest_meta.json",
}

LEGACY_CHECKPOINT_NAME_ALIASES = {
    "c96_rel": "Med3DINO_REL_c96",
    "c112_rel": "Med3DINO_REL_c112",
    "c96_sa": "Med3DINO_SA_c96",
    "c112_sa": "Med3DINO_SA_c112",
    "c96_iso": "Med3DINO_ISO_c96",
    "c112_iso": "Med3DINO_ISO_c112",
    "c96_base": "Med3DINO_Base_c96",
    "c112_base": "Med3DINO_Base_c112",
}

RESULT_JSON_NAME = "module3_anatomical_generalization.json"
SUMMARY_JSON_NAME = "module3_anatomical_generalization_summary.json"
SUMMARY_CSV_NAME = "module3_anatomical_generalization_summary.csv"
DEFAULT_CHECKPOINT_POLICY = "full_canonical_checkpoint_slate"

DEFAULT_CHECKPOINTS = [
    "Med3DINO_REL_c96",
    "Med3DINO_REL_c112",
    "Med3DINO_ISO_c96",
    "Med3DINO_SA_c96",
    "Med3DINO_SA_c112",
    "Med3DINO_ISO_c112",
    "Med3DINO_Base_c96",
    "Med3DINO_Base_c112",
    "3dinov2",
]


def normalize_checkpoint_name(checkpoint_name: str) -> str:
    return LEGACY_CHECKPOINT_NAME_ALIASES.get(str(checkpoint_name).strip(), checkpoint_name)


def normalize_feature_type(feature_type: str = DEFAULT_FEATURE_TYPE) -> str:
    """Normalize a Module 3 feature-family label.

    Required parameter:
        feature_type: one of ``cls``, ``avg_pool``, or ``multilayer``.

    Example:
        >>> normalize_feature_type("avg-pool")
        'avg_pool'
    """
    normalized = FEATURE_TYPE_ALIASES.get(str(feature_type).strip().lower())
    if normalized is None:
        raise ValueError(f"Unknown feature type: {feature_type}")
    return normalized


def normalize_manifest_variant(manifest_variant: str = DEFAULT_MANIFEST_VARIANT) -> str:
    """Normalize a manifest variant label used by Phase 2 and Module 3.

    Required parameter:
        manifest_variant: typically ``core`` or an alias such as ``default``.

    Example:
        >>> normalize_manifest_variant("shared_organs")
        'core'
    """
    return MANIFEST_VARIANT_ALIASES.get(str(manifest_variant).strip().lower(), manifest_variant)


def get_module3_manifest_dir(
    dataset_name: str = DEFAULT_MODULE3_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    """Return the canonical Module 3 manifest directory.

    Required parameters:
        dataset_name: Module 3 dataset namespace.
        manifest_variant: manifest variant label, usually ``core``.
    """
    return MODULE3_MANIFESTS / dataset_name / normalize_manifest_variant(manifest_variant)


def get_module3_manifest_path(
    dataset_name: str = DEFAULT_MODULE3_DATASET,
    manifest_kind: str = "sampled",
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    """Return one canonical Module 3 manifest path.

    Required parameters:
        dataset_name: Module 3 dataset namespace.
        manifest_kind: one of ``full``, ``sampled``, or ``meta``.
        manifest_variant: manifest variant label, usually ``core``.

    Example:
        >>> get_module3_manifest_path("mmwhs_ct_mr", "meta", "core")
    """
    if manifest_kind not in MANIFEST_FILE_NAMES:
        raise ValueError(f"Unknown manifest kind: {manifest_kind}")
    return get_module3_manifest_dir(dataset_name, manifest_variant) / MANIFEST_FILE_NAMES[manifest_kind]


def get_phase2_manifest_path(
    dataset_name: str = DEFAULT_MODULE3_DATASET,
    manifest_kind: str = "sampled",
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    """Return the canonical upstream organ-manifest path used by Module 3.

    Required parameters are the same as ``get_module3_manifest_path``.
    """
    if manifest_kind not in MANIFEST_FILE_NAMES:
        raise ValueError(f"Unknown manifest kind: {manifest_kind}")
    return PHASE2_MANIFESTS / dataset_name / normalize_manifest_variant(manifest_variant) / MANIFEST_FILE_NAMES[manifest_kind]


def get_output_paths(
    dataset_name: str = DEFAULT_MODULE3_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Dict[str, Path]:
    """Return the canonical Module 3 output directories for one dataset surface.

    Required parameters:
        dataset_name: Module 3 dataset namespace.
        manifest_variant: manifest variant label, usually ``core``.
    """
    manifest_variant = normalize_manifest_variant(manifest_variant)
    root = MODULE3_OUTPUTS / dataset_name / MODULE3_NAME / manifest_variant
    return {
        "root": root,
        "results": root / "results",
        "figures": root / "figures",
        "logs": root / "logs",
        "manifests": get_module3_manifest_dir(dataset_name, manifest_variant),
        "source_phase2_root": PHASE2_OUTPUTS / dataset_name / "phase2" / manifest_variant,
    }


def get_phase2_feature_npz_path(
    dataset_name: str,
    manifest_variant: str,
    checkpoint_name: str,
    feature_type: str = DEFAULT_FEATURE_TYPE,
) -> Path:
    """Return the expected upstream embedding bundle path for Module 3 evaluation.

    Required parameters:
        dataset_name: Module 3 dataset namespace.
        manifest_variant: manifest variant label, usually ``core``.
        checkpoint_name: checkpoint label such as ``Med3DINO_REL_c96``.
        feature_type: one of ``cls``, ``avg_pool``, or ``multilayer``.

        The current canonical implementation resolves this path under the Phase 2
        package because that package owns the standard organ-level embedding
        namespace.
    """
    feature_type = normalize_feature_type(feature_type)
    return (
        PHASE2_OUTPUTS
        / dataset_name
        / "phase2"
        / normalize_manifest_variant(manifest_variant)
        / "features"
        / normalize_checkpoint_name(checkpoint_name)
        / feature_type
        / FEATURE_EMBEDDING_FILE_NAMES[feature_type]
    )


def get_checkpoint_metrics_dir(base_dir: Path, checkpoint_name: str, feature_type: str = DEFAULT_FEATURE_TYPE) -> Path:
    """Return the result directory for one checkpoint and feature family."""
    feature_type = normalize_feature_type(feature_type)
    checkpoint_dir = Path(base_dir) / normalize_checkpoint_name(checkpoint_name)
    return checkpoint_dir / feature_type


def get_checkpoint_metrics_path(base_dir: Path, checkpoint_name: str, feature_type: str = DEFAULT_FEATURE_TYPE) -> Path:
    """Return the canonical Module 3 per-checkpoint result JSON path."""
    return get_checkpoint_metrics_dir(base_dir, checkpoint_name, feature_type) / RESULT_JSON_NAME


def get_summary_json_path(base_dir: Path) -> Path:
    """Return the canonical batch summary JSON path."""
    return Path(base_dir) / SUMMARY_JSON_NAME


def get_summary_csv_path(base_dir: Path) -> Path:
    """Return the canonical batch summary CSV path."""
    return Path(base_dir) / SUMMARY_CSV_NAME


def get_dataset_name_from_manifest_path(
    manifest_path: Path,
    fallback: str = DEFAULT_MODULE3_DATASET,
) -> str:
    """Infer the dataset namespace from a Phase 2 or Module 3 manifest path.

    Required parameter:
        manifest_path: absolute or relative path under the canonical manifest
        roots.
    """
    manifest_path = Path(manifest_path)
    candidate_roots = (MODULE3_MANIFESTS.resolve(), PHASE2_MANIFESTS.resolve())
    for root in candidate_roots:
        try:
            relative_path = manifest_path.resolve().relative_to(root)
        except ValueError:
            continue
        if len(relative_path.parts) >= 1:
            return relative_path.parts[0]
    return fallback


def get_manifest_variant_from_manifest_path(
    manifest_path: Path,
    fallback: str = DEFAULT_MANIFEST_VARIANT,
) -> str:
    """Infer the manifest variant from a Phase 2 or Module 3 manifest path."""
    manifest_path = Path(manifest_path)
    candidate_roots = (MODULE3_MANIFESTS.resolve(), PHASE2_MANIFESTS.resolve())
    for root in candidate_roots:
        try:
            relative_path = manifest_path.resolve().relative_to(root)
        except ValueError:
            continue
        if len(relative_path.parts) >= 2:
            return normalize_manifest_variant(relative_path.parts[1])
    return normalize_manifest_variant(fallback)


def ensure_output_directories(
    dataset_name: str = DEFAULT_MODULE3_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> None:
    """Create the canonical Module 3 output directories if they do not exist.

    Required parameters:
        dataset_name: Module 3 dataset namespace.
        manifest_variant: manifest variant label, usually ``core``.

    Example:
        >>> ensure_output_directories("totalsegmenter_ct_mr_anchor", "core")
    """
    paths = get_output_paths(dataset_name, manifest_variant)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
