#!/usr/bin/env python3
"""Shared helpers for Phase 1 dataset preparation scripts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk


MIN_VARIANCE = 1e-6
MAX_CONSTANT_RATIO = 0.99


@dataclass
class PreparedVolumeRecord:
    """Metadata for a cleaned image-mask pair saved to disk."""

    case_id: str
    image_path: str
    mask_path: str
    source_image_path: str
    source_mask_path: str
    shape: Tuple[int, int, int]
    spacing: Tuple[float, float, float]
    anisotropy_ratio: float
    physical_fov_mm: Tuple[float, float, float]
    label_values: list[int]
    mask_voxels: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def ensure_3d(array: np.ndarray, source_name: str) -> np.ndarray:
    """Squeeze singleton trailing channels and enforce a 3D volume."""
    if array.ndim == 4 and array.shape[3] == 1:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"Expected 3D volume for {source_name}, got shape {array.shape}")
    return array


def compute_anisotropy_ratio(spacing: Sequence[float]) -> float:
    spacing = tuple(float(value) for value in spacing[:3])
    return max(spacing) / min(spacing)


def compute_physical_fov(shape: Sequence[int], spacing: Sequence[float]) -> Tuple[float, float, float]:
    return tuple(round(float(dim) * float(step), 2) for dim, step in zip(shape[:3], spacing[:3]))


def get_nifti_spacing(image: nib.Nifti1Image) -> Tuple[float, float, float]:
    return tuple(float(value) for value in image.header.get_zooms()[:3])


def load_nifti_image_and_array(file_path: Path) -> Tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(file_path))
    data = np.asanyarray(image.dataobj)
    return image, ensure_3d(data, str(file_path))


def validate_nifti_volume_integrity(
    file_path: Path,
    min_variance: float = MIN_VARIANCE,
    max_constant_ratio: float = MAX_CONSTANT_RATIO,
) -> Tuple[bool, str]:
    """Validate a NIfTI volume for corruption and degenerate intensity content."""
    try:
        if not file_path.exists():
            return False, "File does not exist"

        try:
            image = nib.load(str(file_path))
        except Exception as exc:
            return False, f"Failed to load NIfTI: {exc}"

        shape = image.shape
        if len(shape) == 4 and shape[3] == 1:
            shape = shape[:3]
        if len(shape) < 3:
            return False, f"Invalid shape: {shape}"
        if any(dim <= 0 for dim in shape[:3]):
            return False, f"Invalid dimensions: {shape}"
        if any(dim > 2048 for dim in shape[:3]):
            return False, f"Unreasonably large dimensions: {shape}"

        try:
            data = image.get_fdata()
        except Exception as exc:
            return False, f"Failed to read data: {exc}"

        data = ensure_3d(data, str(file_path))
        if np.any(np.isnan(data)):
            return False, "Contains NaN values"
        if np.any(np.isinf(data)):
            return False, "Contains Inf values"

        variance = float(np.var(data))
        if variance < min_variance:
            return False, f"Near-constant intensity (variance={variance:.2e})"

        unique_values, counts = np.unique(data, return_counts=True)
        if unique_values.size == 0:
            return False, "Empty volume"
        max_ratio = float(counts.max()) / float(data.size)
        if max_ratio > max_constant_ratio:
            return False, f"Single value dominates ({max_ratio:.1%} of voxels)"

        return True, "OK"
    except Exception as exc:
        return False, f"Validation error: {exc}"


def validate_label_array(
    mask_data: np.ndarray,
    expected_shape: Sequence[int],
    *,
    binary: bool,
) -> Tuple[bool, str, list[int]]:
    """Validate either a binary or integer-valued segmentation mask array."""
    try:
        mask_data = ensure_3d(mask_data, "mask")
        expected_shape = tuple(int(dim) for dim in expected_shape[:3])
        if tuple(mask_data.shape) != expected_shape:
            return False, f"Shape mismatch: expected {expected_shape}, got {mask_data.shape}", []
        if np.any(np.isnan(mask_data)):
            return False, "Mask contains NaN values", []
        if np.any(np.isinf(mask_data)):
            return False, "Mask contains Inf values", []

        rounded = np.rint(mask_data)
        if not np.allclose(mask_data, rounded, atol=1e-3):
            return False, "Mask contains non-integer label values", []
        int_mask = rounded.astype(np.int64)
        if int_mask.min() < 0:
            return False, "Mask contains negative labels", []

        label_values = sorted(int(value) for value in np.unique(int_mask))
        if binary and any(value not in (0, 1) for value in label_values):
            return False, f"Binary mask contains values outside {{0,1}}: {label_values[:10]}", []
        if np.count_nonzero(int_mask) == 0:
            return False, "Mask has no foreground voxels", []
        return True, "OK", label_values
    except Exception as exc:
        return False, f"Mask validation error: {exc}", []


def write_nifti(array: np.ndarray, affine: np.ndarray, output_path: Path, dtype: np.dtype) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(array.astype(dtype), affine)
    nib.save(image, str(output_path))


def read_dicom_series_sitk(dicom_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not file_names:
        raise ValueError(f"No DICOM series found in {dicom_dir}")
    reader.SetFileNames(file_names)
    return reader.Execute()


def validate_sitk_image_and_mask(image: sitk.Image, mask: sitk.Image) -> Tuple[bool, str]:
    if image.GetSize() != mask.GetSize():
        return False, f"Size mismatch: image {image.GetSize()} vs mask {mask.GetSize()}"
    if not np.allclose(image.GetSpacing(), mask.GetSpacing(), atol=1e-5):
        return False, f"Spacing mismatch: image {image.GetSpacing()} vs mask {mask.GetSpacing()}"
    if not np.allclose(image.GetOrigin(), mask.GetOrigin(), atol=1e-5):
        return False, f"Origin mismatch: image {image.GetOrigin()} vs mask {mask.GetOrigin()}"
    if not np.allclose(image.GetDirection(), mask.GetDirection(), atol=1e-5):
        return False, "Direction mismatch between image and mask"
    return True, "OK"


def sitk_shape_xyz(image: sitk.Image) -> Tuple[int, int, int]:
    return tuple(int(value) for value in image.GetSize())


def sitk_spacing_xyz(image: sitk.Image) -> Tuple[float, float, float]:
    return tuple(float(value) for value in image.GetSpacing())


def sitk_to_binary_mask(mask: sitk.Image) -> Tuple[sitk.Image, list[int], int]:
    mask_array = sitk.GetArrayFromImage(mask)
    is_valid, reason, label_values = validate_label_array(
        mask_array,
        expected_shape=tuple(reversed(mask.GetSize())),
        binary=False,
    )
    if not is_valid:
        raise ValueError(reason)
    binary_array = (mask_array > 0).astype(np.uint8)
    binary_image = sitk.GetImageFromArray(binary_array)
    binary_image.CopyInformation(mask)
    return binary_image, label_values, int(binary_array.sum())


def make_manifest_payload(
    *,
    dataset_name: str,
    modality: str,
    description: str,
    source_root: Path,
    output_root: Path,
    records: Iterable[PreparedVolumeRecord],
    filtering_config: Dict[str, Any],
    extra_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    records = list(records)
    ratios = [record.anisotropy_ratio for record in records]
    bin_counts = {"bin_0_near_isotropic": 0, "bin_1_moderate": 0, "bin_2_highly_anisotropic": 0}
    for ratio in ratios:
        if ratio < 1.5:
            bin_counts["bin_0_near_isotropic"] += 1
        elif ratio < 3.0:
            bin_counts["bin_1_moderate"] += 1
        else:
            bin_counts["bin_2_highly_anisotropic"] += 1

    payload: Dict[str, Any] = {
        "version": "1.0",
        "dataset": dataset_name,
        "modality": modality,
        "description": description,
        "source": str(source_root),
        "output_root": str(output_root),
        "total_volumes": len(records),
        "statistics": {
            "anisotropy_ratio": {
                "min": round(min(ratios), 4) if ratios else None,
                "max": round(max(ratios), 4) if ratios else None,
                "mean": round(float(np.mean(ratios)), 4) if ratios else None,
                "median": round(float(np.median(ratios)), 4) if ratios else None,
            },
            "anisotropy_bins": bin_counts,
        },
        "filtering_config": filtering_config,
        "volumes": [record.to_dict() for record in records],
    }
    if extra_metadata:
        payload.update(extra_metadata)
    return payload


def save_json(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
