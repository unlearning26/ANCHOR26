# config.py
# Phase 1: Spacing/Anisotropy Robustness - Configuration
#
# This file centralizes paths, parameters, and settings for Phase 1 evaluation.
# Dataset roots default to repo-local placeholders and can be overridden via
# environment variables.

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


PROJECT_ROOT = _repo_root()
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_ROOT = _env_path("MED3DINO_RAW_DATA_ROOT", DATA_ROOT / "raw")
KITS23_RAW_ROOT = _env_path("MED3DINO_KITS23_RAW_ROOT", RAW_DATA_ROOT / "kits23" / "dataset")

# =============================================================================
# BASE PATHS
# =============================================================================

# Downstream evaluation data (preprocessed)
DOWNSTREAM_ROOT = PROJECT_ROOT / "eval_downstream"
DOWNSTREAM_DATASETS = DOWNSTREAM_ROOT / "datasets"

# Phase 1 outputs
PHASE1_ROOT = DOWNSTREAM_ROOT / "medfm_eval" / "phase1_anisotropy_robustness"
PHASE1_OUTPUTS = PHASE1_ROOT / "outputs_phase1"
PHASE1_MANIFESTS = DOWNSTREAM_ROOT / "medfm_eval" / "data_manifests" / "phase1_anisotropy_robustness"
PHASE1_CACHES = DOWNSTREAM_ROOT / "medfm_eval" / "caches"
PHASE1_NAME = "phase1"
DEFAULT_PHASE1_DATASET = "abdomenatlas"
DEFAULT_BINNING_SCHEME = "original"
DEFAULT_MANIFEST_VARIANT = "original_bins"

MANIFEST_VARIANT_ALIASES = {
    "original": "original_bins",
    "original_bin": "original_bins",
    "original_bins": "original_bins",
    "coarse": "coarse_bins",
    "coarse_bin": "coarse_bins",
    "coarse_bins": "coarse_bins",
    "coarse_ratio_thickness": "coarse_bins",
    "coarse_ratio_thickness_bins": "coarse_bins",
}

MANIFEST_FILE_NAMES = {
    "full": "manifest_full.json",
    "sampled": "manifest_sampled.json",
    "meta": "manifest_meta.json",
}


def normalize_manifest_variant(manifest_variant: str = DEFAULT_MANIFEST_VARIANT) -> str:
    """Normalize manifest variant names to the canonical directory names."""
    return MANIFEST_VARIANT_ALIASES.get(manifest_variant, manifest_variant)


def get_phase1_manifest_dir(
    dataset_name: str = DEFAULT_PHASE1_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    """Return the dataset and manifest-variant directory for Phase 1 manifests."""
    return PHASE1_MANIFESTS / dataset_name / normalize_manifest_variant(manifest_variant)


def get_phase1_manifest_path(
    dataset_name: str = DEFAULT_PHASE1_DATASET,
    manifest_kind: str = "sampled",
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    """Return the canonical Phase 1 manifest path for the requested dataset and variant."""
    if manifest_kind not in MANIFEST_FILE_NAMES:
        raise ValueError(f"Unknown manifest kind: {manifest_kind}")
    return get_phase1_manifest_dir(dataset_name, manifest_variant) / MANIFEST_FILE_NAMES[manifest_kind]


def get_cache_root(
    dataset_name: str = DEFAULT_PHASE1_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    """Return the canonical cache root for perturbation caches."""
    return PHASE1_CACHES / dataset_name / PHASE1_NAME / normalize_manifest_variant(manifest_variant)

# =============================================================================
# RAW DATASET REGISTRY (Track A: Representation Analysis)
# =============================================================================
# These datasets are used for CKA and embedding geometry analysis.
# Status: VERIFIED paths from local exploration.

@dataclass
class RawDatasetConfig:
    """Configuration for a raw (unprocessed) dataset."""
    name: str
    modality: str
    path: Path
    has_labels: bool
    label_path: Optional[Path] = None
    file_pattern: str = "*.nii.gz"
    notes: str = ""
    is_pretraining_data: bool = True  # False = unseen/held-out for generalization test
    is_validated: bool = False  # True = corrupted files already removed, skip validation
    is_enabled: bool = True  # False = skip this dataset in manifest building


RAW_DATASETS: Dict[str, RawDatasetConfig] = {
    # MRI Datasets
    "jhu_stroke": RawDatasetConfig(
        name="jhu_stroke",
        modality="mr",
            path=RAW_DATA_ROOT / "mri" / "JHUS_Stroke_preprocessed" / "images",
            has_labels=True,
            label_path=RAW_DATA_ROOT / "mri" / "JHUS_Stroke_preprocessed" / "masks",
            file_pattern="*_dwi.nii.gz",
            notes="Preprocessed JHUS DWI volumes with paired stroke masks, full anisotropy spectrum (0.92-13.47).",
        is_enabled=False,  # Disabled: needs validation, enable when ready
    ),
    
    # SA-Med3D CT (including private)
    "samed3d_ct": RawDatasetConfig(
        name="samed3d_ct",
        modality="ct",
        path=RAW_DATA_ROOT / "samed3d_multimodal" / "imagesTr_ct",
        has_labels=True,
        label_path=RAW_DATA_ROOT / "samed3d_multimodal" / "labelsTr",
        file_pattern="*.nii.gz",
        notes="~7,848 volumes, mostly isotropic (1.0)",
        is_validated=True  # Cleaned 2024-03: 33 corrupted files removed
    ),
    "samed3d_ct_private": RawDatasetConfig(
        name="samed3d_ct_private",
        modality="ct",
        path=RAW_DATA_ROOT / "samed3d_multimodal" / "imagesTr_private",
        has_labels=True,
        label_path=RAW_DATA_ROOT / "samed3d_multimodal" / "labelsTr",
        file_pattern="*.nii.gz",
        notes="Private CT subset from SA-Med3D",
        is_validated=True  # Cleaned 2024-03: 12 corrupted files removed
    ),
    
    # SA-Med3D MR
    "samed3d_mr": RawDatasetConfig(
        name="samed3d_mr",
        modality="mr",
        path=RAW_DATA_ROOT / "samed3d_multimodal" / "imagesTr_mr",
        has_labels=True,
        label_path=RAW_DATA_ROOT / "samed3d_multimodal" / "labelsTr",
        file_pattern="*.nii.gz",
        notes="~5,753 volumes",
        is_validated=True  # Cleaned 2024-03: 7 corrupted files removed
    ),
    
    # PET/CT Datasets
    "autopet": RawDatasetConfig(
        name="autopet",
        modality="pet_ct",
        path=RAW_DATA_ROOT / "ct_pet" / "autoPET_v1.1" / "imagesTr",
        has_labels=True,
        label_path=RAW_DATA_ROOT / "ct_pet" / "autoPET_v1.1" / "lbl",
        file_pattern="*.nii.gz",
        notes="~2,597 volumes, near-isotropic (0.49-1.47)",
        is_validated=True  # Cleaned 2024-03: 1 corrupted file removed
    ),
    "hecktor25": RawDatasetConfig(
        name="hecktor25",
        modality="pet_ct",
        path=RAW_DATA_ROOT / "ct_pet" / "hecktor25",
        has_labels=True,
        file_pattern="*.nii.gz",
        notes="PET/CT for head-neck cancer",
        is_validated=True  # Cleaned 2024-03: 2 corrupted files removed
    ),
    
    # CTA Dataset
    "imagecas": RawDatasetConfig(
        name="imagecas",
        modality="cta",
        path=RAW_DATA_ROOT / "cta" / "imagecas",
        has_labels=True,  # Coronary segmentation labels (co-located .label.nii.gz)
        label_path=None,  # Labels found via filename replacement (.img -> .label)
        file_pattern="*.img.nii.gz",  # Only match image files, not labels
        notes="~1,000 volumes, mild anisotropy (1.08-1.73)",
        is_validated=True  # Cleaned 2024-03: no corrupted files found
    ),
    
    # CBCT Datasets
    "toothfairy3": RawDatasetConfig(
        name="toothfairy3",
        modality="cbct",
        path=RAW_DATA_ROOT / "cbct" / "toothFairy3" / "imagesTr",
        has_labels=True,
        label_path=RAW_DATA_ROOT / "cbct" / "toothFairy3" / "labelsTr",
        file_pattern="*.nii.gz",
        notes="~532 volumes (dental CBCT)",
        is_validated=True  # Cleaned 2024-03: no corrupted files found
    ),
    "sts_tooth3d": RawDatasetConfig(
        name="sts_tooth3d",
        modality="cbct",
        path=RAW_DATA_ROOT / "cbct" / "sts-tooth3d" / "Integrity" / "Labeled" / "Image",
        has_labels=True,
        label_path=None,  # Labels found via path replacement (Image/ -> Mask/)
        file_pattern="*.nii.gz",
        notes="Additional dental CBCT",
        is_validated=True  # Cleaned 2024-03: no corrupted files found
    ),
    
    # Abdominal CT
    "abdomenatlas": RawDatasetConfig(
        name="abdomenatlas",
        modality="ct",
        path=RAW_DATA_ROOT / "ct" / "abdomenatlas1" / "AbdomenAtlas1.1Mini" / "data",
        has_labels=True,
        label_path=None,  # Labels are co-located with images (combined_labels.nii.gz)
        file_pattern="ct.nii.gz",  # Each subject has ct.nii.gz
        notes="AbdomenAtlas 1.1 Mini: ~9262 volumes, comprehensive abdominal CT with multi-organ segmentation",
        is_validated=True  # Cleaned 2024-03: no corrupted files found
    ),
    "abdomenct1k": RawDatasetConfig(
        name="abdomenct1k",
        modality="ct",
        path=RAW_DATA_ROOT / "ct" / "AbdomenCT1K" / "images",
        has_labels=True,
        label_path=RAW_DATA_ROOT / "ct" / "AbdomenCT1K" / "masks",
        file_pattern="*.nii.gz",
        notes="Cleaned AbdomenCT-1K paired subset with strong natural three-bin coverage; Core Phase 1 CT dataset.",
        is_pretraining_data=False,
        is_validated=True,
    ),
    "kits23": RawDatasetConfig(
        name="kits23",
        modality="ct",
        path=KITS23_RAW_ROOT,
        has_labels=True,
        label_path=None,
        file_pattern="imaging.nii.gz",
        notes="KiTS23 kidney CT with case-level imaging.nii.gz and segmentation.nii.gz; strong pathology-semantic Phase 1 candidate.",
        is_pretraining_data=False,
        is_validated=False,
    ),
    "totalsegmenter_ct": RawDatasetConfig(
        name="totalsegmenter_ct",
        modality="ct",
        path=RAW_DATA_ROOT / "ct" / "totalsegmenter_dataset",
        has_labels=True,
        label_path=None,  # Per-case segmentations directory resolved from image path.
        file_pattern="ct.nii.gz",
        notes="TotalSegmentator CT with per-structure segmentation directory; controlled-only Phase 1 candidate.",
        is_pretraining_data=False,
        is_validated=False,
        is_enabled=False,
    ),
    "totalsegmentermri": RawDatasetConfig(
        name="totalsegmentermri",
        modality="mr",
        path=RAW_DATA_ROOT / "mri" / "TotalSegmenterMRI",
        has_labels=True,
        label_path=None,  # Per-case segmentations directory resolved from image path.
        file_pattern="mri.nii.gz",
        notes="TotalSegmenterMRI with per-structure segmentation directory; MRI anchor-module candidate.",
        is_pretraining_data=False,
        is_validated=False,
        is_enabled=False,
    ),
    "duke_liver": RawDatasetConfig(
        name="duke_liver",
        modality="mr",
        path=RAW_DATA_ROOT / "mri" / "Duke_Liver" / "images",
        has_labels=True,
        label_path=RAW_DATA_ROOT / "mri" / "Duke_Liver" / "masks",
        file_pattern="*.nii.gz",
        notes="Cleaned Duke Liver MRI set with paired image/mask files under images/ and masks/.",
        is_pretraining_data=False,
        is_validated=True,
    ),
    "cirrmri600": RawDatasetConfig(
        name="cirrmri600",
        modality="mr",
        path=RAW_DATA_ROOT / "mri" / "CirrMRI600_cleaned" / "images",
        has_labels=True,
        label_path=RAW_DATA_ROOT / "mri" / "CirrMRI600_cleaned" / "masks",
        file_pattern="*.nii.gz",
        notes="Cleaned CirrMRI600 paired MRI set with T1/T2 images and masks.",
        is_pretraining_data=False,
        is_validated=True,
    ),
    "pansegdata": RawDatasetConfig(
        name="pansegdata",
        modality="mr",
        path=RAW_DATA_ROOT / "mri" / "PanSegData_cleaned" / "images",
        has_labels=True,
        label_path=RAW_DATA_ROOT / "mri" / "PanSegData_cleaned" / "masks",
        file_pattern="*.nii.gz",
        notes="Cleaned PanSegData paired MRI set with T1/T2 images and masks.",
        is_pretraining_data=False,
        is_validated=True,
    ),
}


# =============================================================================
# DOWNSTREAM DATASET REGISTRY (Track B: Spacing-Stratified Probing)
# =============================================================================
# These datasets have labels and are used for downstream task evaluation.
# Status: VERIFIED paths from local exploration.

@dataclass
class DownstreamDatasetConfig:
    """Configuration for a preprocessed downstream dataset."""
    name: str
    task: str  # "segmentation" or "classification"
    modality: str
    num_classes: int
    datalist_path: Path
    data_root: Path
    metrics_class: str  # Class name from dinov2/eval/segmentation_3d/metrics.py
    spacing_stratifiable: bool = True  # Whether spacing stratification makes sense
    notes: str = ""


DOWNSTREAM_DATASETS_CONFIG: Dict[str, DownstreamDatasetConfig] = {
    "btcv": DownstreamDatasetConfig(
        name="btcv",
        task="segmentation",
        modality="ct",
        num_classes=14,
        datalist_path=DOWNSTREAM_DATASETS / "segmentation" / "BTCV_100_datalist.json",
        data_root=DOWNSTREAM_DATASETS / "segmentation" / "BTCV",
        metrics_class="BTCVMetrics",
        spacing_stratifiable=True,  # UNKNOWN: Need to verify spacing distribution
        notes="14-class abdominal CT segmentation"
    ),
    "brats2023": DownstreamDatasetConfig(
        name="brats2023",
        task="segmentation",
        modality="mr",
        num_classes=3,  # TC, WT, ET (multi-label)
        datalist_path=DOWNSTREAM_DATASETS / "segmentation" / "BraTS_100_datalist_3dinov2.json",
        data_root=DOWNSTREAM_DATASETS / "segmentation" / "BraTS2023",
        metrics_class="BraTSMetrics",
        spacing_stratifiable=True,  # UNKNOWN: Need to verify spacing distribution
        notes="Multi-label brain tumor segmentation (4-channel input)"
    ),
    "amos22": DownstreamDatasetConfig(
        name="amos22",
        task="segmentation",
        modality="ct_mr",  # Both CT and MR available
        num_classes=16,
        datalist_path=DOWNSTREAM_DATASETS / "segmentation" / "AMOS22_100_datalist.json",
        data_root=_env_path("MED3DINO_AMOS_ROOT", DOWNSTREAM_DATASETS / "segmentation" / "amos22"),
        metrics_class="AMOS22Metrics",
        spacing_stratifiable=True,  # CONFIRMED: Full anisotropy spectrum
        notes="16-class abdominal segmentation, CT and MR"
    ),
    "isles22": DownstreamDatasetConfig(
        name="isles22",
        task="segmentation",
        modality="mr",
        num_classes=2,  # Background + lesion (binary)
        datalist_path=DOWNSTREAM_DATASETS / "segmentation" / "ISLES22_100_datalist.json",
        data_root=DOWNSTREAM_DATASETS / "segmentation" / "ISLES22",
        metrics_class="ISLES22Metrics",
        spacing_stratifiable=True,  # UNKNOWN: Need to verify spacing distribution
        notes="Binary stroke lesion segmentation (DWI/ADC)"
    ),
    "la_seg": DownstreamDatasetConfig(
        name="la_seg",
        task="segmentation",
        modality="mr",
        num_classes=2,
        datalist_path=DOWNSTREAM_DATASETS / "segmentation" / "LA-SEG_100_datalist.json",
        data_root=DOWNSTREAM_DATASETS / "segmentation" / "Left_Atrium",
        metrics_class="LASEGMetrics",
        spacing_stratifiable=True,  # UNKNOWN: Need to verify
        notes="Binary left atrium segmentation (LGE cardiac MRI)"
    ),
    "tdsc_abus": DownstreamDatasetConfig(
        name="tdsc_abus",
        task="segmentation",
        modality="us",  # Ultrasound
        num_classes=2,
        datalist_path=DOWNSTREAM_DATASETS / "segmentation" / "TDSC-ABUS_100_datalist.json",
        data_root=DOWNSTREAM_DATASETS / "segmentation" / "TDSC_ABUS2023",
        metrics_class="LASEGMetrics",  # Binary segmentation, same as LA-SEG
        spacing_stratifiable=False,  # Ultrasound has different physics
        notes="Binary breast tumor segmentation (automated breast ultrasound)"
    ),
}


# =============================================================================
# CHECKPOINT REGISTRY
# =============================================================================
# All available pretrained checkpoints for evaluation.

@dataclass
class CheckpointConfig:
    """Configuration for a pretrained checkpoint."""
    name: str
    path: Path
    crop_size: int
    spacing_mode: str  # "relative" or "spacing_aware"
    training_data: str = ""
    arch: str = "vit_large_3d"  # "vit_base_3d" or "vit_large_3d"
    notes: str = ""


CHECKPOINT_ROOT = DOWNSTREAM_ROOT / "checkpoints"

LEGACY_CHECKPOINT_NAME_ALIASES: Dict[str, str] = {
    "c96_rel": "Med3DINO_REL_c96",
    "c112_rel": "Med3DINO_REL_c112",
    "c96_sa": "Med3DINO_SA_c96",
    "c112_sa": "Med3DINO_SA_c112",
    "c96_iso": "Med3DINO_ISO_c96",
    "c112_iso": "Med3DINO_ISO_c112",
    # "c96_base": "Med3DINO_Base_c96",
    # "c112_base": "Med3DINO_Base_c112",
}

CANONICAL_CHECKPOINT_NAMES: List[str] = [
    "Med3DINO_REL_c96",
    "Med3DINO_REL_c112",
    "Med3DINO_ISO_c96",
    "Med3DINO_SA_c96",
    "Med3DINO_SA_c112",
    "Med3DINO_ISO_c112",
    # "Med3DINO_Base_c96",
    # "Med3DINO_Base_c112",
    "3dinov2",
]


def normalize_checkpoint_name(checkpoint_name: str) -> str:
    """Resolve legacy compact checkpoint labels to the canonical Med3DINO names."""
    return LEGACY_CHECKPOINT_NAME_ALIASES.get(checkpoint_name, checkpoint_name)


def get_available_checkpoint_names() -> List[str]:
    """Return the canonical checkpoint names in presentation order."""
    return list(CANONICAL_CHECKPOINT_NAMES)


_CANONICAL_CHECKPOINTS: Dict[str, CheckpointConfig] = {
    # Relative spacing regime
    "Med3DINO_REL_c96": CheckpointConfig(
        name="Med3DINO_REL_c96",
        path=CHECKPOINT_ROOT / "chkpts" / "med3dino_rel" / "c96_rel_teacher_checkpoint.pth",
        crop_size=96,
        spacing_mode="relative",
    ),
    "Med3DINO_REL_c112": CheckpointConfig(
        name="Med3DINO_REL_c112",
        path=CHECKPOINT_ROOT / "chkpts" / "med3dino_rel" / "c112_rel_teacher_checkpoint.pth",
        crop_size=112,
        spacing_mode="relative",
        notes="REL checkpoint at crop 112"
    ),
    
    # Spacing-aware regime
    "Med3DINO_SA_c96": CheckpointConfig(
        name="Med3DINO_SA_c96",
        path=CHECKPOINT_ROOT / "chkpts" / "med3dino_sa" / "c96_sa_teacher_checkpoint.pth",
        crop_size=96,
        spacing_mode="spacing_aware",
        notes="Spacing-aware checkpoint"
    ),
    "Med3DINO_SA_c112": CheckpointConfig(
        name="Med3DINO_SA_c112",
        path=CHECKPOINT_ROOT / "chkpts" / "med3dino_sa" / "c112_sa_teacher_checkpoint.pth",
        crop_size=112,
        spacing_mode="spacing_aware",
        notes="Spacing-aware checkpoint at crop 112"
    ),
    
    # # Base regime (iso regime)
    # "Med3DINO_Base_c96": CheckpointConfig(
    #     name="Med3DINO_Base_c96",
    #     path=CHECKPOINT_ROOT / "chkpts" / "med3dino_base" / "c96_base_teacher_checkpoint.pth",
    #     crop_size=96,
    #     spacing_mode="relative",
    #     arch="vit_base_3d",  # NOTE: This is ViT-Base, not ViT-Large!
    #     notes="Base checkpoint (ViT-Base)"
    # ),
    # "Med3DINO_Base_c112": CheckpointConfig(
    #     name="Med3DINO_Base_c112",
    #     path=CHECKPOINT_ROOT / "chkpts" / "med3dino_base" / "c112_base_teacher_checkpoint.pth",
    #     crop_size=112,
    #     spacing_mode="relative",
    #     arch="vit_base_3d",  # NOTE: This is ViT-Base, not ViT-Large!
    #     notes="Base checkpoint at crop 112 (ViT-Base)"
    # ),
    
    # ISO regime
    "Med3DINO_ISO_c96": CheckpointConfig(
        name="Med3DINO_ISO_c96",
        path=CHECKPOINT_ROOT / "chkpts" / "med3dino_iso" / "c96_iso_teacher_checkpoint.pth",
        crop_size=96,
        spacing_mode="relative",
        notes="ISO checkpoint"
    ),
    "Med3DINO_ISO_c112": CheckpointConfig(
        name="Med3DINO_ISO_c112",
        path=CHECKPOINT_ROOT / "chkpts" / "med3dino_iso" / "c112_iso_teacher_checkpoint.pth",
        crop_size=112,
        spacing_mode="relative",
        notes="ISO checkpoint at crop 112"
    ),
    
    # Baseline (3DINO original)
    "3dinov2": CheckpointConfig(
        name="3dinov2",
        path=CHECKPOINT_ROOT / "chkpts" / "3dinov2" / "3dinov2_teacher_checkpoint.pth",
        crop_size=112,  # Original 3DINO trained at 112³ (7³=343 tokens + 1 cls = 344)
        spacing_mode="relative",
        notes="Original 3DINO baseline"
    ),
}


def build_checkpoint_registry(checkpoint_root: Optional[Path] = None) -> Dict[str, CheckpointConfig]:
    """Build the checkpoint registry, optionally rebased to one external root."""
    if checkpoint_root is None:
        checkpoint_root = _env_path("MED3DINO_CHECKPOINT_ROOT", CHECKPOINT_ROOT)

    resolved_root = Path(checkpoint_root).expanduser().resolve()
    if resolved_root == CHECKPOINT_ROOT.resolve():
        canonical = dict(_CANONICAL_CHECKPOINTS)
    else:
        canonical = {
            name: replace(
                config,
                path=resolved_root / config.path.relative_to(CHECKPOINT_ROOT),
            )
            for name, config in _CANONICAL_CHECKPOINTS.items()
        }

    registry: Dict[str, CheckpointConfig] = dict(canonical)
    for legacy_name, canonical_name in LEGACY_CHECKPOINT_NAME_ALIASES.items():
        registry[legacy_name] = canonical[canonical_name]
    return registry


CHECKPOINTS: Dict[str, CheckpointConfig] = build_checkpoint_registry()


# =============================================================================
# ANISOTROPY BIN CONFIGURATION
# =============================================================================

@dataclass
class AnisotropyBinConfig:
    """Configuration for anisotropy ratio stratification."""
    bin_id: int
    name: str
    ratio_min: float
    ratio_max: float
    description: str


COARSE_ANISOTROPY_BINS: List[AnisotropyBinConfig] = [
    AnisotropyBinConfig(
        bin_id=0,
        name="coarse_low_ratio_thin_slice",
        ratio_min=1.0,
        ratio_max=2.0,
        description="Low anisotropy and thin-slice: ratio < 2.0 and max spacing < 2.0 mm",
    ),
    AnisotropyBinConfig(
        bin_id=1,
        name="coarse_intermediate_ratio_thickness",
        ratio_min=2.0,
        ratio_max=4.0,
        description="Intermediate ratio/thickness: not bin 0, with ratio < 4.0 and max spacing < 4.0 mm",
    ),
    AnisotropyBinConfig(
        bin_id=2,
        name="coarse_high_ratio_or_thick_slice",
        ratio_min=4.0,
        ratio_max=float('inf'),
        description="High anisotropy or thick-slice: ratio >= 4.0 or max spacing >= 4.0 mm",
    ),
]


# Anisotropy ratio formula: max(sx, sy, sz) / min(sx, sy, sz)
# This is orientation-agnostic and captures anisotropy across all axes.
#
# 3-bin scheme for spacing robustness evaluation:
# - Near-Isotropic: 1.0 ≤ ratio < 1.5 (nearly cubic voxels)
# - Moderately Anisotropic: 1.5 ≤ ratio < 3.0 (common clinical CT/MRI)
# - Highly Anisotropic: ratio ≥ 3.0 (thick-slice acquisitions)
#
# These thresholds are designed to be:
# - Clinically meaningful (1.5x and 3x are natural boundaries)
# - Dataset-agnostic (work with any modality/anatomy)
# - Sensitive to common anisotropy patterns in medical imaging
ANISOTROPY_BINS: List[AnisotropyBinConfig] = [
    AnisotropyBinConfig(
        bin_id=0,
        name="near_isotropic",
        ratio_min=1.0,  # min possible ratio is 1.0
        ratio_max=1.5,
        description="Near-isotropic voxels (1.0 ≤ ratio < 1.5)"
    ),
    AnisotropyBinConfig(
        bin_id=1,
        name="moderately_anisotropic",
        ratio_min=1.5,
        ratio_max=3.0,
        description="Moderate anisotropy (1.5 ≤ ratio < 3.0)"
    ),
    AnisotropyBinConfig(
        bin_id=2,
        name="highly_anisotropic",
        ratio_min=3.0,
        ratio_max=float('inf'),
        description="Highly anisotropic (ratio ≥ 3.0)"
    ),
]


def compute_anisotropy_ratio(spacing: Tuple[float, float, float]) -> float:
    """
    Compute global anisotropy ratio from voxel spacing.
    
    Formula: max(sx, sy, sz) / min(sx, sy, sz)
    
    This is orientation-agnostic and robust to:
    - Different axis ordering conventions
    - Non-standard slice orientations
    - Volumes reoriented during preprocessing
    
    Args:
        spacing: Tuple of (sx, sy, sz) voxel spacings in mm
        
    Returns:
        Anisotropy ratio (≥ 1.0, where 1.0 = perfectly isotropic)
    """
    sx, sy, sz = spacing
    return max(sx, sy, sz) / min(sx, sy, sz)


def compute_slice_thickness_proxy(spacing: Tuple[float, float, float]) -> float:
    """
    Compute an orientation-agnostic proxy for slice thickness.

    We use the largest voxel spacing as a conservative surrogate for the thick axis.
    This keeps the rule dataset-agnostic without assuming a fixed slice dimension.
    """
    return max(float(s) for s in spacing)


def get_bin_configs(scheme: str = "original") -> List[AnisotropyBinConfig]:
    """Return bin definitions for the requested scheme."""
    if not scheme:
        scheme = DEFAULT_BINNING_SCHEME
    if scheme == "original":
        return ANISOTROPY_BINS
    if scheme in {"coarse", "coarse_ratio_thickness"}:
        return COARSE_ANISOTROPY_BINS
    raise ValueError(f"Unknown binning scheme: {scheme}")


def get_binning_scheme_from_manifest(manifest_data: Optional[Dict[str, Any]]) -> str:
    """Resolve binning scheme from manifest metadata with backward compatibility."""
    if not manifest_data:
        return DEFAULT_BINNING_SCHEME
    return manifest_data.get("binning_scheme", DEFAULT_BINNING_SCHEME)


def get_bin_name_map(scheme: str = DEFAULT_BINNING_SCHEME) -> Dict[int, str]:
    """Return a stable bin-id to name mapping for the requested scheme."""
    return {bin_config.bin_id: bin_config.name for bin_config in get_bin_configs(scheme)}


def get_bin_description_map(scheme: str = DEFAULT_BINNING_SCHEME) -> Dict[int, str]:
    """Return a stable bin-id to description mapping for the requested scheme."""
    return {bin_config.bin_id: bin_config.description for bin_config in get_bin_configs(scheme)}


def get_anisotropy_bin(
    ratio: float,
    spacing: Optional[Tuple[float, float, float]] = None,
    scheme: str = "original",
) -> int:
    """Assign a volume to an anisotropy bin using the requested binning scheme."""
    if scheme in {"coarse", "coarse_ratio_thickness"}:
        if spacing is None:
            raise ValueError("spacing is required for coarse binning")
        thickness = compute_slice_thickness_proxy(spacing)
        if ratio < 2.0 and thickness < 2.0:
            return 0
        if ratio < 4.0 and thickness < 4.0:
            return 1
        return 2

    for bin_config in ANISOTROPY_BINS:
        if bin_config.ratio_min <= ratio < bin_config.ratio_max:
            return bin_config.bin_id
    return len(ANISOTROPY_BINS) - 1  # Default to highest bin


# =============================================================================
# SAMPLING CONFIGURATION
# =============================================================================

@dataclass
class SamplingConfig:
    """Configuration for Track A sample selection."""
    min_samples_per_bin: int = 500
    max_samples_per_bin: int = 1000
    stratify_by_modality: bool = True
    random_seed: int = 42
    
    # Modality handling for low-sample cases
    # If a modality has < min_samples_per_bin in a bin, either:
    # - "include_all": Use all available samples
    # - "exclude": Exclude the modality from that bin
    # - "merge_bins": Merge adjacent bins for the modality
    low_sample_strategy: str = "include_all"


SAMPLING_CONFIG = SamplingConfig()


# =============================================================================
# ANALYSIS CONFIGURATION - CT-Only vs Multi-Modality
# =============================================================================
# Addresses modality-spacing confounding issue:
# - MR, CBCT are entirely isotropic → cannot assess spacing robustness
# - CT has full spacing spectrum → primary analysis track
# - Multi-modality at iso-only → secondary cross-modality analysis

@dataclass
class AnalysisConfig:
    """Configuration for a specific analysis track."""
    name: str
    description: str
    
    # Modality filtering
    allowed_modalities: List[str]  # Empty = all modalities
    
    # Bin filtering  
    allowed_bins: List[int]  # Empty = all bins
    
    # Sampling parameters
    min_per_bin: int = 500
    max_per_bin: int = 1000
    
    # Output manifest name
    manifest_suffix: str = ""
    
    # Priority for reporting (lower = higher priority)
    priority: int = 0


# Primary Analysis: CT-only with full spacing spectrum
# - Scientifically valid for spacing robustness claims
# - No modality-spacing confounding
CT_ONLY_ANALYSIS = AnalysisConfig(
    name="ct_only",
    description="CT-only analysis with full anisotropy spectrum (primary)",
    allowed_modalities=["ct", "cta"],  # CT and CT-Angiography
    allowed_bins=[0, 1, 2],  # All bins - CT has full spectrum
    min_per_bin=500,
    max_per_bin=1000,
    manifest_suffix="_ct_only",
    priority=1
)

# Secondary Analysis: Multi-modality at isotropic spacing only
# - Valid for cross-modality generalization claims
# - Limited to isotropic bin only (fair comparison)
MULTIMODALITY_ISO_ANALYSIS = AnalysisConfig(
    name="multimodality_iso",
    description="Multi-modality analysis at isotropic spacing only (secondary)",
    allowed_modalities=[],  # All modalities
    allowed_bins=[0],  # Isotropic only - fair comparison
    min_per_bin=500,
    max_per_bin=2000,  # More samples available
    manifest_suffix="_multimod_iso",
    priority=2
)

# PET-CT Analysis: Has some anisotropic data but limited (16 highly aniso)
# - Optional track for PET-CT specific claims
PETCT_ANALYSIS = AnalysisConfig(
    name="petct",
    description="PET-CT analysis with partial spacing spectrum",
    allowed_modalities=["pet_ct"],
    allowed_bins=[0, 1],  # Skip bin 2 (only 16 samples)
    min_per_bin=100,  # Lower threshold for PET-CT
    max_per_bin=500,
    manifest_suffix="_petct",
    priority=3
)

ANALYSIS_CONFIGS: Dict[str, AnalysisConfig] = {
    "ct_only": CT_ONLY_ANALYSIS,
    "multimodality_iso": MULTIMODALITY_ISO_ANALYSIS,
    "petct": PETCT_ANALYSIS,
}

# Default analysis for Phase 1
DEFAULT_ANALYSIS = "ct_only"


# =============================================================================
# SEMANTIC LABEL CONFIGURATION
# =============================================================================
# Configuration for semantic probing labels derived from segmentation masks.

@dataclass
class SemanticLabelConfig:
    """Configuration for semantic label extraction from segmentation masks."""
    # Label extraction mode
    # - "multi_binary": Multi-label binary (organ presence/absence)
    # - "dominant_organ": Single-label (organ with largest volume)
    mode: str = "multi_binary"
    
    # Minimum voxel count for organ presence (for multi_binary mode)
    min_organ_voxels: int = 100

    # Keep only labels with enough variation to support a meaningful probe.
    # Labels that are nearly always absent or nearly always present inflate
    # plain accuracy and are not informative for single-dataset Phase 1.
    min_label_prevalence: float = 0.05
    max_label_prevalence: float = 0.95
    
    # AbdomenAtlas organ mapping (label_id -> organ_name)
    # These are the 25 anatomical structures in AbdomenAtlas 1.1
    abdomenatlas_organs: Dict[int, str] = None
    
    def __post_init__(self):
        if self.abdomenatlas_organs is None:
            self.abdomenatlas_organs = {
                1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
                5: "liver", 6: "stomach", 7: "pancreas", 8: "adrenal_right",
                9: "adrenal_left", 10: "lung_upper_left", 11: "lung_lower_left",
                12: "lung_upper_right", 13: "lung_middle_right", 14: "lung_lower_right",
                15: "esophagus", 16: "trachea", 17: "thyroid", 18: "small_bowel",
                19: "duodenum", 20: "colon", 21: "urinary_bladder", 22: "prostate",
                23: "kidney_cyst_left", 24: "kidney_cyst_right", 25: "aorta"
            }


SEMANTIC_LABEL_CONFIG = SemanticLabelConfig()


@dataclass(frozen=True)
class SemanticTargetSpec:
    """Dataset-aware semantic target definition for Phase 1 probing and transfer.

    The extraction mode determines how semantic targets are derived from the
    label source associated with each case. Modes currently supported are:

    1. `label_id_multilabel`: one multi-label vector from an integer-valued mask.
    2. `segmentation_dir_presence`: one multi-label vector from a directory of
       binary structure masks.
    3. `foreground_volume_tertiles`: a one-hot target derived from foreground
         volume quantiles for single-structure binary masks.
    """

    name: str
    extraction_mode: str
    description: str
    integer_label_mapping: Optional[Dict[int, str]] = None
    segmentation_names: Optional[List[str]] = None
    semantic_label_names: Optional[List[str]] = None
    quantile_bins: int = 3
    min_voxels: int = 100
    min_label_prevalence: float = 0.05
    max_label_prevalence: float = 0.95


TOTALSEGMENTER_SHARED_STRUCTURES: List[str] = [
    "spleen",
    "kidney_left",
    "kidney_right",
    "gallbladder",
    "liver",
    "stomach",
    "pancreas",
    "small_bowel",
    "colon",
    "urinary_bladder",
    "aorta",
    "inferior_vena_cava",
]


FOREGROUND_VOLUME_TERTILES_SPEC = SemanticTargetSpec(
    name="foreground_volume_tertiles",
    extraction_mode="foreground_volume_tertiles",
    description="One-hot semantic target derived from foreground-volume tertiles for binary masks.",
    quantile_bins=3,
)


ABDOMENCT1K_CORE_ORGANS: Dict[int, str] = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
}


HECKTOR25_TARGETS: Dict[int, str] = {
    1: "primary_tumor",
    2: "metastatic_lymph_nodes",
}


KITS23_TARGETS: Dict[int, str] = {
    1: "kidney",
    2: "tumor",
    3: "cyst",
}


IMAGECAS_CORONARY_BURDEN_SPEC = SemanticTargetSpec(
    name="imagecas_coronary_burden_tertiles",
    extraction_mode="foreground_volume_tertiles",
    description="Dataset-specific coronary-tree burden strata derived from coronary foreground volume tertiles.",
    semantic_label_names=[
        "coronary_tree_burden_low",
        "coronary_tree_burden_mid",
        "coronary_tree_burden_high",
    ],
    quantile_bins=3,
)


DATASET_SEMANTIC_TARGET_SPECS: Dict[str, SemanticTargetSpec] = {
    "abdomenatlas": SemanticTargetSpec(
        name="abdomenatlas_multi_organ_presence",
        extraction_mode="label_id_multilabel",
        description="Multi-organ presence targets from the AbdomenAtlas combined-label mask.",
        integer_label_mapping=SEMANTIC_LABEL_CONFIG.abdomenatlas_organs,
        min_voxels=SEMANTIC_LABEL_CONFIG.min_organ_voxels,
        min_label_prevalence=SEMANTIC_LABEL_CONFIG.min_label_prevalence,
        max_label_prevalence=SEMANTIC_LABEL_CONFIG.max_label_prevalence,
    ),
    "totalsegmenter_ct": SemanticTargetSpec(
        name="totalsegmenter_shared_structure_presence",
        extraction_mode="segmentation_dir_presence",
        description="Shared-structure presence targets derived from the TotalSegmentator per-structure directory.",
        segmentation_names=TOTALSEGMENTER_SHARED_STRUCTURES,
        min_voxels=100,
    ),
    "totalsegmentermri": SemanticTargetSpec(
        name="totalsegmentermri_shared_structure_presence",
        extraction_mode="segmentation_dir_presence",
        description="Shared-structure presence targets derived from the TotalSegmenterMRI per-structure directory.",
        segmentation_names=TOTALSEGMENTER_SHARED_STRUCTURES,
        min_voxels=100,
    ),
    "abdomenct1k": SemanticTargetSpec(
        name="abdomenct1k_core_organ_presence",
        extraction_mode="label_id_multilabel",
        description="Core abdominal organ presence targets from the AbdomenCT1K integer-valued mask.",
        integer_label_mapping=ABDOMENCT1K_CORE_ORGANS,
        min_voxels=SEMANTIC_LABEL_CONFIG.min_organ_voxels,
        min_label_prevalence=SEMANTIC_LABEL_CONFIG.min_label_prevalence,
        max_label_prevalence=SEMANTIC_LABEL_CONFIG.max_label_prevalence,
    ),
    "cirrmri600": FOREGROUND_VOLUME_TERTILES_SPEC,
    "duke_liver": FOREGROUND_VOLUME_TERTILES_SPEC,
    "hecktor25": SemanticTargetSpec(
        name="hecktor25_lesion_presence",
        extraction_mode="label_id_multilabel",
        description="Primary-tumor and nodal-lesion presence targets from the HECKTOR Task 1 mask.",
        integer_label_mapping=HECKTOR25_TARGETS,
        min_voxels=100,
        min_label_prevalence=SEMANTIC_LABEL_CONFIG.min_label_prevalence,
        max_label_prevalence=SEMANTIC_LABEL_CONFIG.max_label_prevalence,
    ),
    "jhu_stroke": FOREGROUND_VOLUME_TERTILES_SPEC,
    "kits23": SemanticTargetSpec(
        name="kits23_kidney_tumor_cyst_presence",
        extraction_mode="label_id_multilabel",
        description="Kidney, tumor, and cyst presence targets from the KiTS23 case segmentation.",
        integer_label_mapping=KITS23_TARGETS,
        min_voxels=100,
        min_label_prevalence=SEMANTIC_LABEL_CONFIG.min_label_prevalence,
        max_label_prevalence=SEMANTIC_LABEL_CONFIG.max_label_prevalence,
    ),
    "pansegdata": FOREGROUND_VOLUME_TERTILES_SPEC,
    "imagecas": IMAGECAS_CORONARY_BURDEN_SPEC,
    "toothfairy3": FOREGROUND_VOLUME_TERTILES_SPEC,
    "sts_tooth3d": FOREGROUND_VOLUME_TERTILES_SPEC,
}


MODALITY_SEMANTIC_TARGET_SPECS: Dict[str, SemanticTargetSpec] = {
    "ct": FOREGROUND_VOLUME_TERTILES_SPEC,
    "mr": FOREGROUND_VOLUME_TERTILES_SPEC,
    "cta": FOREGROUND_VOLUME_TERTILES_SPEC,
    "pet_ct": FOREGROUND_VOLUME_TERTILES_SPEC,
    "cbct": FOREGROUND_VOLUME_TERTILES_SPEC,
}


def resolve_semantic_target_spec(
    dataset_name: Optional[str] = None,
    modality: Optional[str] = None,
) -> SemanticTargetSpec:
    """Resolve the semantic target definition for a dataset or modality.

    Dataset-specific specifications take precedence over modality defaults.
    The fallback target for datasets without explicit structure semantics is a
    foreground-volume tertile target, which keeps the semantic claim narrow and
    target-specific for binary-mask datasets.
    """

    if dataset_name and dataset_name in DATASET_SEMANTIC_TARGET_SPECS:
        return DATASET_SEMANTIC_TARGET_SPECS[dataset_name]
    if modality and modality in MODALITY_SEMANTIC_TARGET_SPECS:
        return MODALITY_SEMANTIC_TARGET_SPECS[modality]
    return FOREGROUND_VOLUME_TERTILES_SPEC


# =============================================================================
# CONTROLLED PERTURBATION CONFIGURATION (Setting B)
# =============================================================================
# Configuration for synthetic spacing perturbation experiments.
# These isolate spacing as the causal factor by resampling the same volume
# to different target spacings.

@dataclass
class ControlledPerturbationConfig:
    """Configuration for Setting B: Controlled spacing perturbation."""
    protocol_name: str = "fixed_absolute"

    # Target spacings for resampling (mm)
    # Each volume will be resampled to these target spacings
    target_spacings: List[Tuple[float, float, float]] = None

    # MRI extension protocol: keep native in-plane spacing and vary only z-spacing.
    target_z_spacings: Optional[List[float]] = None
    
    # Source: Use isotropic volumes from Setting A Bin 0
    source_bin: int = 0
    
    # Number of source volumes to use (None = all available)
    n_source_volumes: Optional[int] = None
    
    # Resampling interpolation mode
    interpolation_mode: str = "trilinear"  # or "nearest" for labels
    
    # Whether to anti-alias before downsampling
    anti_alias: bool = True
    
    def __post_init__(self):
        if self.protocol_name == "fixed_absolute" and self.target_spacings is None:
            # Default target spacings representing the 3 anisotropy regimes
            self.target_spacings = [
                (1.0, 1.0, 1.0),  # Isotropic (ratio = 1.0)
                (1.0, 1.0, 3.0),  # Moderate anisotropy (ratio = 3.0)
                (1.0, 1.0, 5.0),  # High anisotropy (ratio = 5.0)
            ]
        if self.protocol_name == "native_inplane_z" and self.target_z_spacings is None:
            self.target_z_spacings = [2.0, 4.0, 6.0]

    def resolve_variant_specs(
        self,
        source_spacing: Optional[Tuple[float, float, float]] = None,
    ) -> List[Tuple[str, Tuple[float, float, float]]]:
        """Resolve named target spacing variants for a source volume."""
        if self.protocol_name == "fixed_absolute":
            return [
                (f"{sx:.1f}x{sy:.1f}x{sz:.1f}", (sx, sy, sz))
                for sx, sy, sz in self.target_spacings
            ]

        if self.protocol_name == "native_inplane_z":
            if source_spacing is None:
                raise ValueError("source_spacing is required for native_inplane_z perturbation")
            sx, sy, _ = [float(value) for value in source_spacing]
            return [
                (f"native_xy_z{z:.1f}", (sx, sy, float(z)))
                for z in self.target_z_spacings
            ]

        raise ValueError(f"Unknown perturbation protocol: {self.protocol_name}")

    def reference_variant_name(self, source_spacing: Optional[Tuple[float, float, float]] = None) -> str:
        """Return the baseline variant name used for representation drift."""
        return self.resolve_variant_specs(source_spacing)[0][0]

    def cache_signature(self) -> str:
        """Return a stable cache signature for the perturbation protocol."""
        interpolation_tag = str(self.interpolation_mode).lower().replace("_", "-")
        anti_alias_tag = "aa1" if self.anti_alias else "aa0"
        if self.protocol_name == "fixed_absolute":
            parts = [f"{sx:.1f}x{sy:.1f}x{sz:.1f}" for sx, sy, sz in self.target_spacings]
            return f"fixed_v2_bin{self.source_bin}_{anti_alias_tag}_{interpolation_tag}_" + "_".join(parts)
        if self.protocol_name == "native_inplane_z":
            parts = [f"z{z:.1f}" for z in self.target_z_spacings]
            return f"native_xy_v2_bin{self.source_bin}_{anti_alias_tag}_{interpolation_tag}_" + "_".join(parts)
        return f"protocol_v2_{self.protocol_name}_bin{self.source_bin}_{anti_alias_tag}_{interpolation_tag}"

    def describe_targets(self) -> List[str]:
        """Return human-readable target descriptions for logging/serialization."""
        if self.protocol_name == "fixed_absolute":
            return [f"{sx:.1f}x{sy:.1f}x{sz:.1f}" for sx, sy, sz in self.target_spacings]
        if self.protocol_name == "native_inplane_z":
            return [f"native_xy_z{z:.1f}" for z in self.target_z_spacings]
        return [self.protocol_name]


CONTROLLED_PERTURBATION_CONFIG = ControlledPerturbationConfig()
MRI_CONTROLLED_PERTURBATION_CONFIG = ControlledPerturbationConfig(
    protocol_name="native_inplane_z",
    target_spacings=None,
    target_z_spacings=[2.0, 4.0, 6.0],
    source_bin=1,
)


def get_controlled_perturbation_config(modality: str = "ct") -> ControlledPerturbationConfig:
    """Return the canonical Setting B protocol for a modality."""
    if str(modality or "ct").lower() == "mr":
        return MRI_CONTROLLED_PERTURBATION_CONFIG
    return CONTROLLED_PERTURBATION_CONFIG


def get_controlled_perturbation_config_from_manifest(
    manifest_data: Dict[str, Any],
) -> ControlledPerturbationConfig:
    """Return the canonical Setting B protocol for a manifest payload."""
    return get_controlled_perturbation_config(str(manifest_data.get("modality", "ct")))


# =============================================================================
# CROSS-BIN TRANSFER CONFIGURATION
# =============================================================================
# Configuration for cross-bin transfer experiments in Track B.

@dataclass
class CrossBinTransferConfig:
    """Configuration for cross-bin transfer probing experiments."""
    # Datasets allowed for cross-bin transfer (must span multiple bins)
    # Only datasets with sufficient samples in multiple bins are valid
    allowed_datasets: List[str] = None
    
    # Minimum samples required per bin for a dataset to be included
    min_samples_per_bin: int = 50
    
    # Probing tasks to evaluate
    # - "domain": Dataset/subdomain classification
    # - "semantic": Organ presence (multi-label binary)
    probing_tasks: List[str] = None
    
    def __post_init__(self):
        if self.allowed_datasets is None:
            # Only AbdomenAtlas spans all bins with sufficient samples
            self.allowed_datasets = ["abdomenatlas"]
        if self.probing_tasks is None:
            self.probing_tasks = ["semantic"]


CROSS_BIN_TRANSFER_CONFIG = CrossBinTransferConfig()


# =============================================================================
# PREPROCESSING CONFIGURATION
# =============================================================================
# Follows pretraining preprocessing exactly.

@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    # Intensity normalization (percentile-based)
    # Match the reference training pipeline in train3d.py.
    intensity_percentile_lower: float = 0.05
    intensity_percentile_upper: float = 99.95
    intensity_output_min: float = -1.0
    intensity_output_max: float = 1.0
    
    # Cropping
    crop_foreground: bool = True
    foreground_threshold: float = -1.0  # Training uses select_fn=lambda x: x > -1

    # Evaluation should fail loudly if a sample cannot be loaded.
    fail_on_load_error: bool = True
    
    # Target crop size (matches checkpoint)
    crop_size: int = 96
    
    # Data format
    output_dtype: str = "float32"


PREPROCESSING_CONFIG = PreprocessingConfig()


# =============================================================================
# FEATURE EXTRACTION CONFIGURATION
# =============================================================================

@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction from ViT encoder."""
    # Global representation methods
    use_cls_token: bool = True
    use_avg_pooled_patches: bool = True  # Aggregate patch tokens via spatial average pooling
    
    # Multi-scale analysis
    use_multilayer: bool = True
    layer_indices: Tuple[int, ...] = (-4, -3, -2, -1)  # Last 4 blocks
    
    # Patch tokens (for CKA on local representations)
    extract_patch_tokens: bool = True
    
    # Batch processing
    batch_size: int = 4
    num_workers: int = 4
    
    # Output format
    save_format: str = "pt"  # "pt" (PyTorch) or "npy" (NumPy)


FEATURE_EXTRACTION_CONFIG = FeatureExtractionConfig()


# =============================================================================
# METRICS CONFIGURATION
# =============================================================================
# Metrics for Track B evaluation.

@dataclass
class MetricsConfig:
    """Configuration for evaluation metrics."""
    # Segmentation metrics
    compute_dice: bool = True
    compute_hd95: bool = True
    compute_asd: bool = True
    compute_nsd: bool = True
    nsd_tolerance_mm: float = 2.0
    
    # Classification metrics (if applicable)
    compute_accuracy: bool = True
    compute_auroc: bool = True
    compute_f1: bool = True


METRICS_CONFIG = MetricsConfig()


# =============================================================================
# EVALUATION SEEDS (for reproducibility)
# =============================================================================
# Run all probing experiments with multiple seeds and report mean ± std
EVALUATION_SEEDS: List[int] = [42, 123, 456]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_directories():
    """Create shared Phase 1 directories."""
    directories = [
        PHASE1_OUTPUTS,
        PHASE1_MANIFESTS,
        PHASE1_CACHES,
    ]
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)


def validate_checkpoint_paths() -> Dict[str, bool]:
    """Validate that all checkpoint files exist."""
    results = {}
    for name, config in _CANONICAL_CHECKPOINTS.items():
        results[name] = config.path.exists()
    return results


def validate_raw_dataset_paths() -> Dict[str, bool]:
    """Validate that all raw dataset directories exist."""
    results = {}
    for name, config in RAW_DATASETS.items():
        results[name] = config.path.exists()
    return results


def get_modality_abbreviation(modality: str) -> str:
    """Get standardized modality abbreviation."""
    mapping = {
        "ct": "CT",
        "mr": "MR",
        "pet": "PET",
        "pet_ct": "PET-CT",
        "cta": "CTA",
        "cbct": "CBCT",
        "us": "US",
        "oct": "OCT",
    }
    return mapping.get(modality.lower(), modality.upper())


# =============================================================================
# DATASET-SPECIFIC OUTPUT PATH FUNCTIONS
# =============================================================================
# Functions to generate output paths for dataset-specific experiments.
# Example: outputs_phase1/abdomenatlas/phase1/original_bins/results/, etc.

def get_output_paths(
    dataset_name: str = DEFAULT_PHASE1_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Dict[str, Path]:
    """
    Get output path dictionary for a specific dataset and manifest variant.
    
    Args:
        dataset_name: Dataset name (e.g., "abdomenatlas", "abdomenct1k")
        manifest_variant: Manifest variant directory (e.g., "original_bins", "coarse_bins")
        
    Returns:
        Dictionary with keys: "root", "results", "figures", "features", "logs", "cache_root", "manifests"
    """
    manifest_variant = normalize_manifest_variant(manifest_variant)
    root = PHASE1_OUTPUTS / dataset_name / PHASE1_NAME / manifest_variant
    return {
        "root": root,
        "results": root / "results",
        "figures": root / "figures",
        "features": root / "features",
        "logs": root / "logs",
        "cache_root": get_cache_root(dataset_name, manifest_variant),
        "manifests": get_phase1_manifest_dir(dataset_name, manifest_variant),
    }


def get_checkpoint_feature_dir(base_dir: Path, checkpoint_name: str, feature_type: str) -> Path:
    """Return the nested directory for a checkpoint and feature family."""
    return Path(base_dir) / normalize_checkpoint_name(checkpoint_name) / feature_type


def get_summary_feature_dir(base_dir: Path, feature_type: str) -> Path:
    """Return the nested summary directory for a feature family."""
    return Path(base_dir) / "summaries" / feature_type


def get_dataset_name_from_manifest_path(
    manifest_path: Path,
    fallback: str = DEFAULT_PHASE1_DATASET,
) -> str:
    """Infer the dataset name from a manifest path under the shared manifest root."""
    manifest_path = Path(manifest_path)

    try:
        relative_path = manifest_path.resolve().relative_to(PHASE1_MANIFESTS.resolve())
    except ValueError:
        return fallback

    if len(relative_path.parts) >= 1:
        return relative_path.parts[0]
    return fallback


def get_manifest_variant_from_manifest_path(
    manifest_path: Path,
    fallback: str = DEFAULT_MANIFEST_VARIANT,
) -> str:
    """Infer the manifest variant from a manifest path under the shared manifest root."""
    manifest_path = Path(manifest_path)

    try:
        relative_path = manifest_path.resolve().relative_to(PHASE1_MANIFESTS.resolve())
    except ValueError:
        return normalize_manifest_variant(fallback)

    if len(relative_path.parts) >= 2:
        return normalize_manifest_variant(relative_path.parts[1])
    return normalize_manifest_variant(fallback)


def ensure_output_directories(
    dataset_name: str = DEFAULT_PHASE1_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
):
    """Create all necessary output directories for a dataset and manifest variant."""
    paths = get_output_paths(dataset_name, manifest_variant)
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Usage: python config.py
    # Prints a quick validation summary for checkpoints, raw datasets, and bin definitions.
    # Quick validation
    print("=" * 60)
    print("Phase 1 Configuration Validation")
    print("=" * 60)
    
    print("\n--- Checkpoint Paths ---")
    for name, exists in validate_checkpoint_paths().items():
        status = "✓" if exists else "✗"
        print(f"  [{status}] {name}: {CHECKPOINTS[name].path}")
    
    print("\n--- Raw Dataset Paths ---")
    for name, exists in validate_raw_dataset_paths().items():
        status = "✓" if exists else "✗"
        print(f"  [{status}] {name}: {RAW_DATASETS[name].path}")
    
    print("\n--- Anisotropy Bins ---")
    for bin_config in ANISOTROPY_BINS:
        print(f"  Bin {bin_config.bin_id} ({bin_config.name}): {bin_config.description}")

    print("\n--- Coarse Ratio+Thickness Bins ---")
    for bin_config in COARSE_ANISOTROPY_BINS:
        print(f"  Bin {bin_config.bin_id} ({bin_config.name}): {bin_config.description}")
