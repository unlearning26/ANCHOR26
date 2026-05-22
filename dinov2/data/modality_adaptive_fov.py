"""
Modality-Adaptive FOV Normalization for SSL Training

This module implements FOV (Field of View) bounding transforms that normalize
the through-plane (Z-axis) extent of 3D medical images based on modality-specific
empirical distributions. This ensures consistent scale diversity for self-supervised
learning while respecting the natural FOV characteristics of different imaging modalities.

Empirical Bounds (from 68,868 scan analysis):
- CT:  [295, 507]mm (p25-p75 from 44,695 scans, natural distribution)
- CTA: [358, 876]mm (p25-p75 from 200 scans, bimodal chest/aorta)
- MR:  [140, 180]mm (widened from [156, 160]mm to add scale diversity)
- PET: [800, 1000]mm (widened from [852, 978]mm for standardized protocols)
"""

import logging
from typing import Dict, Hashable, Mapping, Optional, Tuple

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable
from monai.transforms.croppad.array import CenterSpatialCrop, SpatialPad
from monai.utils import fall_back_tuple

logger = logging.getLogger("dinov2")


class ModalityAdaptiveFOVCropd(MapTransform, Randomizable):
    """
    Crop or pad the through-plane (Z-axis) dimension to a modality-specific FOV range.
    
    This transform ensures that all images within a modality have comparable physical
    extent along the through-plane axis, which is critical for multi-scale self-supervised
    learning (DINO/iBOT). Without this normalization, a CT scan (775mm FOV) and MR scan
    (160mm FOV) would produce vastly different crop sizes during random augmentation,
    leading to inconsistent teacher-student agreement.
    
    Operation:
    1. Read modality from metadata (e.g., 'ct', 'mr', 'pet', 'cta')
    2. Compute current FOV: shape[2] × spacing[2] (in mm)
    3. Sample target FOV uniformly from modality-specific [min, max] range
    4. If current < target: pad symmetrically along Z-axis
    5. If current > target: center crop along Z-axis
    6. Update metadata with new FOV and shape
    
    Args:
        keys: Keys of the corresponding items to be transformed
        modality_fov_bounds: Dict mapping modality names to [min_fov_mm, max_fov_mm]
            Example: {'ct': [295.0, 507.0], 'mr': [140.0, 180.0]}
        modality_key: Metadata key containing modality string (default: 'modality')
        allow_missing_keys: If True, missing keys won't raise errors
        
    Example:
        >>> bounds = {'ct': [295.0, 507.0], 'mr': [140.0, 180.0]}
        >>> transform = ModalityAdaptiveFOVCropd(
        ...     keys=['image'], 
        ...     modality_fov_bounds=bounds
        ... )
        >>> # Apply to a CT scan with 775mm FOV → will crop to ~400mm (random in [295, 507])
        >>> data = {'image': ct_volume, 'modality': 'ct', 'spacing': [0.97, 0.97, 2.5]}
        >>> result = transform(data)
    """
    
    def __init__(
        self,
        keys: KeysCollection,
        modality_fov_bounds: Dict[str, Tuple[float, float]],
        modality_key: str = "modality",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.modality_fov_bounds = modality_fov_bounds
        self.modality_key = modality_key
        
        # Validate bounds
        for modality, (min_fov, max_fov) in modality_fov_bounds.items():
            if min_fov >= max_fov:
                raise ValueError(
                    f"Invalid FOV bounds for {modality}: min={min_fov} >= max={max_fov}"
                )
        
        logger.info(f"[ModalityAdaptiveFOVCropd] Initialized with bounds: {modality_fov_bounds}")
    
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        """
        Apply modality-adaptive FOV bounding to the data dictionary.
        
        Args:
            data: Dictionary containing image, metadata, spacing info
            
        Returns:
            Modified dictionary with bounded FOV
        """
        d = dict(data)
        
        # Get modality from metadata
        modality = d.get(self.modality_key, None)
        if modality is None:
            logger.warning(
                f"[ModalityAdaptiveFOVCropd] Missing '{self.modality_key}' in metadata. "
                f"Skipping FOV bounding."
            )
            return d
        
        # Normalize modality string (lowercase, strip whitespace)
        modality = str(modality).lower().strip()
        
        # Check if bounds exist for this modality
        if modality not in self.modality_fov_bounds:
            logger.warning(
                f"[ModalityAdaptiveFOVCropd] No FOV bounds defined for modality '{modality}'. "
                f"Available modalities: {list(self.modality_fov_bounds.keys())}. "
                f"Skipping FOV bounding."
            )
            return d
        
        min_fov_mm, max_fov_mm = self.modality_fov_bounds[modality]
        
        # Sample target FOV uniformly (adds scale diversity for SSL)
        target_fov_mm = self.R.uniform(min_fov_mm, max_fov_mm)
        
        # Process each key
        for key in self.key_iterator(d):
            # Get image and spacing
            image = d[key]
            
            # Try to get spacing from multiple possible locations
            spacing = None
            if f"{key}_meta_dict" in d and "spacing" in d[f"{key}_meta_dict"]:
                spacing = d[f"{key}_meta_dict"]["spacing"]
            elif "spacing" in d:
                spacing = d["spacing"]
            
            if spacing is None:
                logger.warning(
                    f"[ModalityAdaptiveFOVCropd] Missing spacing for key '{key}'. "
                    f"Cannot compute FOV. Skipping."
                )
                continue
            
            # Ensure spacing is array-like
            if isinstance(spacing, (list, tuple)):
                spacing = np.array(spacing)
            elif torch.is_tensor(spacing):
                spacing = spacing.cpu().numpy()
            
            # Validate spacing (must be positive for FOV calculation)
            if spacing[2] <= 0:
                logger.warning(
                    f"[ModalityAdaptiveFOVCropd] Invalid spacing[2]={spacing[2]} for key '{key}'. "
                    f"Spacing must be positive. Skipping FOV normalization."
                )
                continue
            
            # Get current shape and FOV
            # MONAI convention: image shape is (C, D, H, W) where D=depth (through-plane/Z-axis)
            # After shape[-3:], we get (D, H, W) where index 0 is the through-plane dimension
            current_shape = image.shape[-3:]  # (D, H, W)
            current_fov_mm = current_shape[0] * spacing[2]  # D-axis FOV (spacing[2] is Z-spacing)
            
            # Compute target number of slices
            target_slices = int(np.round(target_fov_mm / spacing[2]))
            target_slices = max(target_slices, 1)  # At least 1 slice
            
            # Determine operation
            if target_slices == current_shape[0]:
                # No change needed
                logger.debug(
                    f"[ModalityAdaptiveFOVCropd] {modality.upper()} - No change needed: "
                    f"current={current_fov_mm:.1f}mm, target={target_fov_mm:.1f}mm"
                )
                continue
            
            elif target_slices > current_shape[0]:
                # PAD: current FOV < target FOV
                pad_total = target_slices - current_shape[0]
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                
                # MONAI expects padding as [(low, high), ...] for each spatial dimension
                # We only pad Z-axis (dim 0), keep H and W unchanged
                spatial_pad = SpatialPad(
                    spatial_size=(target_slices, -1, -1),  # -1 means keep original size
                    mode="edge",  # Replicate edge values
                )
                
                d[key] = spatial_pad(image)
                
                logger.debug(
                    f"[ModalityAdaptiveFOVCropd] {modality.upper()} - PADDED: "
                    f"{current_fov_mm:.1f}mm → {target_fov_mm:.1f}mm "
                    f"(slices: {current_shape[0]} → {target_slices}, +{pad_total})"
                )
            
            else:
                # CROP: current FOV > target FOV
                # Use center crop to preserve anatomical centering
                crop_transform = CenterSpatialCrop(roi_size=(target_slices, -1, -1))
                d[key] = crop_transform(image)
                
                slices_removed = current_shape[0] - target_slices
                logger.debug(
                    f"[ModalityAdaptiveFOVCropd] {modality.upper()} - CROPPED: "
                    f"{current_fov_mm:.1f}mm → {target_fov_mm:.1f}mm "
                    f"(slices: {current_shape[0]} → {target_slices}, -{slices_removed})"
                )
            
            # Update metadata with new FOV
            if f"{key}_meta_dict" in d:
                d[f"{key}_meta_dict"]["original_fov_mm"] = float(current_fov_mm)
                d[f"{key}_meta_dict"]["bounded_fov_mm"] = float(target_fov_mm)
                d[f"{key}_meta_dict"]["original_shape_z"] = int(current_shape[0])
                d[f"{key}_meta_dict"]["bounded_shape_z"] = int(target_slices)
        
        return d
    
    def randomize(self, data: Optional[np.ndarray] = None) -> None:
        """
        Randomize internal state for target FOV sampling.
        Called automatically by MONAI's Compose transform.
        """
        super().randomize(data)


# Convenience function for quick usage
def get_modality_adaptive_fov_transform(
    keys: KeysCollection = ["image"],
    modality_fov_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> ModalityAdaptiveFOVCropd:
    """
    Factory function to create ModalityAdaptiveFOVCropd with default or custom bounds.
    
    Args:
        keys: Keys to transform (default: ['image'])
        modality_fov_bounds: Custom bounds, or None for empirical defaults
        
    Returns:
        Configured ModalityAdaptiveFOVCropd transform
    """
    if modality_fov_bounds is None:
        # Default empirical bounds from 68K scan analysis
        modality_fov_bounds = {
            'ct': [295.0, 507.0],
            'cta': [358.0, 876.0],
            'mr': [140.0, 180.0],
            'pet': [800.0, 1000.0],
        }
    
    return ModalityAdaptiveFOVCropd(
        keys=keys,
        modality_fov_bounds=modality_fov_bounds,
    )
