from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).parents[3]
DOWNSTREAM_ROOT = PROJECT_ROOT / "eval_downstream"
MEDFM_EVAL_ROOT = DOWNSTREAM_ROOT / "medfm_eval"

PHASE2_ROOT = MEDFM_EVAL_ROOT / "phase2_cross_modality_alignment"
PHASE2_OUTPUTS = PHASE2_ROOT / "outputs_phase2"
PHASE2_MANIFESTS = MEDFM_EVAL_ROOT / "data_manifests" / "phase2_cross_modality_alignment"
PHASE_SHARED_CACHES = MEDFM_EVAL_ROOT / "caches"

PHASE2_NAME = "phase2"
DEFAULT_PHASE2_DATASET = "ct_mr_alignment"
DEFAULT_MANIFEST_VARIANT = "core"
DEFAULT_REQUIRED_MODALITIES = ("ct", "mr")
DEFAULT_FEATURE_TYPE = "cls"
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
PRIMARY_METRICS_FILE_NAME = "phase2_primary_metrics.json"
CHECKPOINT_COMPARISON_JSON_BY_FEATURE = {
    "cls": "phase2_checkpoint_comparison.json",
    "avg_pool": "phase2_checkpoint_comparison_avg_pool.json",
    "multilayer": "phase2_checkpoint_comparison_multilayer.json",
}
CHECKPOINT_COMPARISON_CSV_BY_FEATURE = {
    "cls": "phase2_checkpoint_comparison.csv",
    "avg_pool": "phase2_checkpoint_comparison_avg_pool.csv",
    "multilayer": "phase2_checkpoint_comparison_multilayer.csv",
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

CANONICAL_CHECKPOINT_NAMES = (
    "Med3DINO_REL_c96",
    "Med3DINO_REL_c112",
    "Med3DINO_ISO_c96",
    "Med3DINO_SA_c96",
    "Med3DINO_SA_c112",
    "Med3DINO_ISO_c112",
    "Med3DINO_Base_c96",
    "Med3DINO_Base_c112",
    "3dinov2",
)


def normalize_feature_type(feature_type: str = DEFAULT_FEATURE_TYPE) -> str:
    normalized = FEATURE_TYPE_ALIASES.get(str(feature_type).strip().lower())
    if normalized is None:
        raise ValueError(f"Unknown feature type: {feature_type}")
    return normalized


def normalize_manifest_variant(manifest_variant: str = DEFAULT_MANIFEST_VARIANT) -> str:
    return MANIFEST_VARIANT_ALIASES.get(manifest_variant, manifest_variant)


def normalize_checkpoint_name(checkpoint_name: str) -> str:
    return LEGACY_CHECKPOINT_NAME_ALIASES.get(str(checkpoint_name).strip(), checkpoint_name)


def get_available_checkpoint_names() -> list[str]:
    return list(CANONICAL_CHECKPOINT_NAMES)


def get_phase2_manifest_dir(
    dataset_name: str = DEFAULT_PHASE2_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    return PHASE2_MANIFESTS / dataset_name / normalize_manifest_variant(manifest_variant)


def get_phase2_manifest_path(
    dataset_name: str = DEFAULT_PHASE2_DATASET,
    manifest_kind: str = "sampled",
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    if manifest_kind not in MANIFEST_FILE_NAMES:
        raise ValueError(f"Unknown manifest kind: {manifest_kind}")
    return get_phase2_manifest_dir(dataset_name, manifest_variant) / MANIFEST_FILE_NAMES[manifest_kind]


def get_cache_root(
    dataset_name: str = DEFAULT_PHASE2_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    return PHASE_SHARED_CACHES / dataset_name / PHASE2_NAME / normalize_manifest_variant(manifest_variant)


def get_crop_cache_dir(
    dataset_name: str = DEFAULT_PHASE2_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
    crop_size: int = 96,
) -> Path:
    return get_cache_root(dataset_name, manifest_variant) / f"crop{int(crop_size)}" / "organ_crop_cache"


def get_output_paths(
    dataset_name: str = DEFAULT_PHASE2_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Dict[str, Path]:
    manifest_variant = normalize_manifest_variant(manifest_variant)
    root = PHASE2_OUTPUTS / dataset_name / PHASE2_NAME / manifest_variant
    return {
        "root": root,
        "results": root / "results",
        "figures": root / "figures",
        "features": root / "features",
        "logs": root / "logs",
        "cache_root": get_cache_root(dataset_name, manifest_variant),
        "crop_cache_root": get_cache_root(dataset_name, manifest_variant),
        "manifests": get_phase2_manifest_dir(dataset_name, manifest_variant),
    }


def get_checkpoint_feature_dir(base_dir: Path, checkpoint_name: str, feature_type: str) -> Path:
    return Path(base_dir) / normalize_checkpoint_name(checkpoint_name) / normalize_feature_type(feature_type)


def get_feature_embedding_file_name(feature_type: str = DEFAULT_FEATURE_TYPE) -> str:
    feature_type = normalize_feature_type(feature_type)
    return FEATURE_EMBEDDING_FILE_NAMES[feature_type]


def get_feature_summary_file_name(feature_type: str = DEFAULT_FEATURE_TYPE) -> str:
    embedding_name = get_feature_embedding_file_name(feature_type)
    stem, suffix = embedding_name.rsplit(".", 1)
    return f"{stem}_summary.{suffix.replace('npz', 'json')}"


def get_checkpoint_feature_npz_path(base_dir: Path, checkpoint_name: str, feature_type: str) -> Path:
    feature_dir = get_checkpoint_feature_dir(base_dir, checkpoint_name, feature_type)
    return feature_dir / get_feature_embedding_file_name(feature_type)


def get_checkpoint_metrics_dir(base_dir: Path, checkpoint_name: str, feature_type: str = DEFAULT_FEATURE_TYPE) -> Path:
    feature_type = normalize_feature_type(feature_type)
    checkpoint_dir = Path(base_dir) / normalize_checkpoint_name(checkpoint_name)
    if feature_type == "cls":
        return checkpoint_dir
    return checkpoint_dir / feature_type


def get_checkpoint_metrics_path(base_dir: Path, checkpoint_name: str, feature_type: str = DEFAULT_FEATURE_TYPE) -> Path:
    return get_checkpoint_metrics_dir(base_dir, checkpoint_name, feature_type) / PRIMARY_METRICS_FILE_NAME


def get_checkpoint_comparison_json_path(base_dir: Path, feature_type: str = DEFAULT_FEATURE_TYPE) -> Path:
    feature_type = normalize_feature_type(feature_type)
    return Path(base_dir) / CHECKPOINT_COMPARISON_JSON_BY_FEATURE[feature_type]


def get_checkpoint_comparison_csv_path(base_dir: Path, feature_type: str = DEFAULT_FEATURE_TYPE) -> Path:
    feature_type = normalize_feature_type(feature_type)
    return Path(base_dir) / CHECKPOINT_COMPARISON_CSV_BY_FEATURE[feature_type]


def get_dataset_name_from_manifest_path(
    manifest_path: Path,
    fallback: str = DEFAULT_PHASE2_DATASET,
) -> str:
    manifest_path = Path(manifest_path)
    try:
        relative_path = manifest_path.resolve().relative_to(PHASE2_MANIFESTS.resolve())
    except ValueError:
        return fallback
    if len(relative_path.parts) >= 1:
        return relative_path.parts[0]
    return fallback


def get_manifest_variant_from_manifest_path(
    manifest_path: Path,
    fallback: str = DEFAULT_MANIFEST_VARIANT,
) -> str:
    manifest_path = Path(manifest_path)
    try:
        relative_path = manifest_path.resolve().relative_to(PHASE2_MANIFESTS.resolve())
    except ValueError:
        return normalize_manifest_variant(fallback)
    if len(relative_path.parts) >= 2:
        return normalize_manifest_variant(relative_path.parts[1])
    return normalize_manifest_variant(fallback)


def ensure_output_directories(
    dataset_name: str = DEFAULT_PHASE2_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> None:
    paths = get_output_paths(dataset_name, manifest_variant)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)