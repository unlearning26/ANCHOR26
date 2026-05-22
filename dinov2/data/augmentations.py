# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import logging
import warnings

import numpy as np
from torchvision import transforms
from monai.transforms import (
    Crop,
    Randomizable,
    RandFlipd,
    Compose,
    RandRotate90d,
    OneOf,
    RandAdjustContrastd,
    RandGaussianSharpend,
    RandGaussianSmoothd,
    RandGaussianNoised,
    RandHistogramShiftd,
    RandGibbsNoised,
    CropForeground,
    ToTensord,
)
from monai.data.utils import get_random_patch, get_valid_patch_size
from torch.nn.functional import interpolate
import math

logger = logging.getLogger("dinov2")


class RandomResizedCrop3d(Crop, Randomizable):
    def __init__(
        self,
        size,
        in_slice_scale,
        cross_slice_scale,
        interpolation='trilinear',
        aspect_ratio=(0.9, 1/0.9)
    ):
        """
        Adapting torch RandomResizedCrop to 3D data by separating in-slice/in-plane and cross-slice dimensions.

        Args:
            size: Size of output image.
            in_slice_scale: Range of the random size of the cropped in-slice/in-plane dimensions.
            cross_slice_scale: Range of the random size of the cropped cross-slice dimensions.
            interpolation: 3D interpolation method, defaults to 'trilinear'.
            aspect_ratio: Range of aspect ratios of the cropped in-slice/in-plane dimensions.
        """
        super().__init__()
        # Ensure size is a 3-tuple
        if isinstance(size, int):
            self.size = (size, size, size)
        else:
            self.size = size
        self.in_slice_scale = in_slice_scale
        self.cross_slice_scale = cross_slice_scale
        self.interpolation = interpolation
        self.aspect_ratio = aspect_ratio
        self._slices: tuple[slice, ...] = ()

    def get_in_slice_crop(self, height, width):
        """
        Adapted from torchvision RandomResizedCrop, applied to the in-slice/in-plane dimensions
        """
        area = height * width

        log_ratio = math.log(self.aspect_ratio[0]), math.log(self.aspect_ratio[1])
        for _ in range(10):
            target_area = area * self.R.uniform(*self.in_slice_scale)
            aspect_ratio = math.exp(self.R.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                return h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.aspect_ratio):
            w = width
            h = int(round(w / min(self.aspect_ratio)))
        elif in_ratio > max(self.aspect_ratio):
            h = height
            w = int(round(h * max(self.aspect_ratio)))
        else:  # whole image
            w = width
            h = height
        return h, w

    def randomize(self, img_size):
        """
        Compute random crop slices.
        
        Args:
            img_size: (H, W, D) spatial dimensions
        """
        # first two dimensions are dicom slice dims/in-plane dims, third is number of slices
        height, width, depth = img_size

        # get in-slice crop size
        crop_h, crop_w = self.get_in_slice_crop(height, width)

        # get cross-slice crop size
        crop_d = int(round(depth * self.R.uniform(*self.cross_slice_scale)))

        crop_size = (crop_h, crop_w, crop_d)
        valid_size = get_valid_patch_size(img_size, crop_size)
        
        # Random crop (unconstrained)
        self._slices = get_random_patch(img_size, valid_size, self.R)

    def __call__(self, data, lazy=False):
        # Handle both dictionary (MONAI Compose) and tensor (direct) input
        if isinstance(data, dict):
            img = data["image"]

            is_dict_input = True
        else:
            img = data

            is_dict_input = False
            
        self.randomize(img.shape[1:])
        cropped = super().__call__(img=img, slices=self._slices)
        resized = interpolate(cropped.unsqueeze(0), size=self.size, mode=self.interpolation).squeeze(0)

        # Return in same format as input
        if is_dict_input:
            result = dict(data)
            result["image"] = resized
            return result
        else:
            return resized


class CropForegroundSwapSliceDims(CropForeground):
    """
    Same functionality as CropForeground, but permutes in-plane dimensions to first two spatial dims for
    RandomResizedCrop3d.
    """
    @staticmethod
    def get_permutation_v2(spacing):
        # === IMPROVEMENT: deterministic slice axis ===
        # Pick slice axis = axis with largest spacing (lowest resolution)
        slice_axis = int(np.argmax(spacing))
        inplane = [ax for ax in (0,1,2) if ax != slice_axis]

        # Original expected output: (C,H,W,D) => perm = [0, <inplane+1>, slice_axis+1]
        return [0] + [ax+1 for ax in inplane] + [slice_axis+1]

    def __call__(self, img_dict, mode=None, lazy=None, **pad_kwargs):
        # get image spacing and spatial dims
        img_spacing = img_dict['spacing']
        img = img_dict['image']
        spatial_dims = img.shape[1:]

        # try getting from pixel spacing first, NOTE: verified that at least two dims have similar spacing in datasets
        if img_spacing is not None:
            perm = self.get_permutation_v2(img_spacing)
        else:
            perm = self.get_permutation_v2(spatial_dims)

        # swap slice dims
        img = img.permute(*perm)

        # crop foreground and return dictionary to maintain MONAI Compose chain
        cropped_img = super().__call__(img, mode, lazy, **pad_kwargs)
        
        # Return dictionary format to maintain MONAI Compose compatibility
        result = dict(img_dict)  # Copy all keys
        result['image'] = cropped_img  # Update image
        return result


class DataAugmentationDINO3d(object):
    """
    Dual-mode 3D data augmentation for DINO self-supervised learning.
    
    Mode 1 (3DINO Baseline): Uses relative-scale cropping (RandomResizedCrop3d)
    Mode 2 (Spacing-Aware): Uses physical-space cropping (RandPhysicalCropd) with spacing-aware augmentations
    
    Args:
        # Baseline parameters (3DINO - relative scale mode)
        global_crops_in_slice_scale: Range for in-plane scale (e.g., [0.48, 1.0])
        global_crops_cross_slice_scale: Range for cross-slice scale (e.g., [0.5, 1.0])
        local_crops_in_slice_scale: Range for local in-plane scale (e.g., [0.16, 0.48])
        local_crops_cross_slice_scale: Range for local cross-slice scale (e.g., [0.2, 0.5])
        
        # Enhanced parameters (spacing-aware hybrid cropping mode)
        global_crops_physical_scale_mm: Physical size range in mm for through-plane (e.g., (96.0, 224.0))
        local_crops_physical_scale_mm: Physical size range in mm for through-plane (e.g., (32.0, 80.0))
        anisotropy_threshold: Threshold for gating cross-plane rotations (default: 2.0)
        
        Note: Enhanced mode uses hybrid strategy:
        - In-plane (H, W): Relative scale (48-100% for global, 16-48% for local)
        - Through-plane (D): Physical scale with automatic anisotropy adjustment
        This avoids empty crops while preserving anatomical content.
        
        # Common parameters
        local_crops_number: Number of local crops (default: 8)
        global_crops_size: Output voxel size for global crops (default: 96)
        local_crops_size: Output voxel size for local crops (default: 48)
        
        # Mode control
        use_spacing_aware: If True, use spacing-aware physical-space mode; if False, use 3DINO baseline (default: False)
    """

    def __init__(
        self,
        global_crops_in_slice_scale=None,
        global_crops_cross_slice_scale=None,
        local_crops_in_slice_scale=None,
        local_crops_cross_slice_scale=None,
        local_crops_number=8,
        global_crops_size=96,
        local_crops_size=48,
        # Spacing-aware parameters
        global_crops_physical_scale_mm=None,
        local_crops_physical_scale_mm=None,
        anisotropy_threshold=2.0,
        # Mode flag
        use_spacing_aware=False,
    ):
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.use_spacing_aware = use_spacing_aware
        self.anisotropy_threshold = anisotropy_threshold
        
        # Validate parameters based on mode
        if use_spacing_aware:
            if global_crops_physical_scale_mm is None or local_crops_physical_scale_mm is None:
                raise ValueError(
                    "When use_spacing_aware=True, you must provide "
                    "global_crops_physical_scale_mm and local_crops_physical_scale_mm"
                )
            self.global_crops_physical_scale_mm = global_crops_physical_scale_mm
            self.local_crops_physical_scale_mm = local_crops_physical_scale_mm
            self.global_crops_in_slice_scale = global_crops_in_slice_scale if global_crops_in_slice_scale is not None else [0.48, 1.0]
            self.local_crops_in_slice_scale = local_crops_in_slice_scale if local_crops_in_slice_scale is not None else [0.16, 0.48]
            
            logger.info("###############################################################")
            logger.info("Spacing-Aware Mode: Using physical-space augmentation")
            logger.info(f"global_crops_physical_scale_mm: {self.global_crops_physical_scale_mm}")
            logger.info(f"local_crops_physical_scale_mm: {self.local_crops_physical_scale_mm}")
            logger.info(f"anisotropy_threshold: {self.anisotropy_threshold}")
            logger.info(f"local_crops_number: {self.local_crops_number}")
            logger.info(f"global_crops_size: {self.global_crops_size}")
            logger.info(f"local_crops_size: {self.local_crops_size}")
            logger.info("###############################################################")
            
            # Import spacing-aware transforms
            from dinov2.data.spacing_aware_transforms import RandPhysicalCropd, AnisotropyGatedRotation
            
            # Spacing-aware hybrid cropping (relative in-plane + physical through-plane)
            self.geometric_augmentation_global = Compose([
                RandPhysicalCropd(
                    keys=["image"],
                    inplane_relative_scale=self.global_crops_in_slice_scale,  # Relative scale for H, W => (0.48, 1.0)
                    throughplane_physical_range_mm=self.global_crops_physical_scale_mm,
                    output_size_voxels=self.global_crops_size,
                    auto_adjust_anisotropy=True
                ),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[0]),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[1]),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[2]),
                AnisotropyGatedRotation(
                    keys=["image"],
                    inplane_prob=0.5,
                    crossplane_prob=0.3,
                    anisotropy_threshold=self.anisotropy_threshold
                )
            ])
            
            self.geometric_augmentation_local = Compose([
                RandPhysicalCropd(
                    keys=["image"],
                    inplane_relative_scale=self.local_crops_in_slice_scale,  # Relative scale for H, W
                    throughplane_physical_range_mm=self.local_crops_physical_scale_mm,
                    output_size_voxels=self.local_crops_size,
                    auto_adjust_anisotropy=True
                ),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[0]),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[1]),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[2]),
                AnisotropyGatedRotation(
                    keys=["image"],
                    inplane_prob=0.5,
                    crossplane_prob=0.3,
                    anisotropy_threshold=self.anisotropy_threshold
                )
            ])
        else:
            if None in [global_crops_in_slice_scale, global_crops_cross_slice_scale,
                       local_crops_in_slice_scale, local_crops_cross_slice_scale]:
                raise ValueError(
                    "When use_spacing_aware=False (baseline mode), you must provide "
                    "global_crops_in_slice_scale, global_crops_cross_slice_scale, "
                    "local_crops_in_slice_scale, and local_crops_cross_slice_scale"
                )
            self.global_crops_in_slice_scale = global_crops_in_slice_scale
            self.global_crops_cross_slice_scale = global_crops_cross_slice_scale
            self.local_crops_in_slice_scale = local_crops_in_slice_scale
            self.local_crops_cross_slice_scale = local_crops_cross_slice_scale
            
            logger.info("###################################")
            logger.info("3DINO Baseline Mode: Using relative-scale augmentation")
            logger.info(f"global_crops_in_slice_scale: {self.global_crops_in_slice_scale}")
            logger.info(f"global_crops_cross_slice_scale: {self.global_crops_cross_slice_scale}")
            logger.info(f"local_crops_in_slice_scale: {self.local_crops_in_slice_scale}")
            logger.info(f"local_crops_cross_slice_scale: {self.local_crops_cross_slice_scale}")
            logger.info(f"local_crops_number: {self.local_crops_number}")
            logger.info(f"global_crops_size: {self.global_crops_size}")
            logger.info(f"local_crops_size: {self.local_crops_size}")
            logger.info("###################################")
            
            # 3DINO Baseline: Relative-scale cropping with unconditional rotations
            self.geometric_augmentation_global = Compose([
                RandomResizedCrop3d(
                    self.global_crops_size,
                    in_slice_scale=self.global_crops_in_slice_scale,
                    cross_slice_scale=self.global_crops_cross_slice_scale
                ),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[0]),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[1]),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[2]),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(1, 2)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 2))
            ])
            
            self.geometric_augmentation_local = Compose([
                RandomResizedCrop3d(
                    self.local_crops_size,
                    in_slice_scale=self.local_crops_in_slice_scale,
                    cross_slice_scale=self.local_crops_cross_slice_scale
                ),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[0]),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[1]),
                RandFlipd(keys=["image"], prob=0.3, spatial_axis=[2]),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(1, 2)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 2))
            ])
        
        # noise, contrast, blurring
        gaussian_transforms = OneOf(
            [
                RandAdjustContrastd(keys=["image"], prob=0.8, gamma=(0.5, 2)),
                RandGaussianNoised(keys=["image"], prob=0.8, std=0.002),
                RandHistogramShiftd(keys=["image"], num_control_points=10, prob=0.8),
            ]
        )

        global_transfo1_extra = OneOf(
            [
                RandGaussianSmoothd(keys=["image"], prob=1.0),
                RandGaussianSharpend(keys=["image"], prob=1.0),
            ]
        )

        global_transfo2_extra = transforms.Compose(
            [
                OneOf(
                    [
                        RandGaussianSmoothd(keys=["image"], prob=0.1),
                        RandGaussianSharpend(keys=["image"], prob=0.1),
                    ]
                ),
                RandGibbsNoised(keys=["image"], prob=0.2)
            ]
        )

        local_transfo_extra = RandGaussianSmoothd(keys=["image"], prob=0.5)

        self.global_transfo1 = Compose([gaussian_transforms, global_transfo1_extra, ToTensord(keys=["image"])])
        self.global_transfo2 = Compose([gaussian_transforms, global_transfo2_extra, ToTensord(keys=["image"])])
        self.local_transfo = Compose([gaussian_transforms, local_transfo_extra, ToTensord(keys=["image"])])

    def __call__(self, data):
        output = {}

        # Now CropForegroundSwapSliceDims returns dictionary format
        # Extract image tensor for processing
        if isinstance(data, dict):
            image_dict = data
        else:
            # Fallback for direct tensor input
            image_dict = {"image": data}

        # global crops:
        im1_base = self.geometric_augmentation_global(image_dict)
        global_crop_1 = self.global_transfo1(im1_base)["image"]  # Extract tensor from dict

        im2_base = self.geometric_augmentation_global(image_dict)
        global_crop_2 = self.global_transfo2(im2_base)["image"]  # Extract tensor from dict

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image_dict))["image"]  # Extract tensor from dict
            for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output