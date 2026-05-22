# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    MapTransform,
    EnsureTyped,
    RandSpatialCropSamplesd,
    RandScaleIntensityd,
    ConcatItemsd,
    DeleteItemsd,
    SpatialPadd,
    Lambdad
)
import torch
import numpy as np

# Import ITKReader for NRRD file support (LA-SEG dataset)
try:
    from monai.data import ITKReader
    HAS_ITK = True
except ImportError:
    HAS_ITK = False


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 3 is GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            result = []
            # merge label 1 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 1, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 3 is ET
            result.append(d[key] == 3)
            d[key] = torch.cat(result, dim=0).float()
        return d


def make_transforms(dataset_name, image_size, resize_scale, min_int):

    test_transforms = None

    if dataset_name == 'BTCV':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 2.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=min_int, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int, allow_smaller=True),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 2.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=min_int, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int, allow_smaller=True),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    elif dataset_name == 'BraTS':

        train_transforms = Compose(
            [
                # load 4 Nifti images and stack them together
                LoadImaged(keys=["image1", "image2", "image3", "image4", "label"], ensure_channel_first=True),
                ConcatItemsd(keys=["image1", "image2", "image3", "image4"], name='image', dim=0),
                DeleteItemsd(keys=["image1", "image2", "image3", "image4"]),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandSpatialCropSamplesd(
                    keys=["image", "label"], num_samples=4, roi_size=(image_size, image_size, image_size), random_size=False
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image1", "image2", "image3", "image4", "label"], ensure_channel_first=True),
                ConcatItemsd(keys=["image1", "image2", "image3", "image4"], name='image', dim=0),
                DeleteItemsd(keys=["image1", "image2", "image3", "image4"]),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
    elif dataset_name == 'LA-SEG':
        # LA-SEG uses NRRD format which requires ITKReader
        reader = ITKReader() if HAS_ITK else None
        
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"], reader=reader),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Lambdad(keys=["label"], func=lambda x: (x == 255).astype(np.uint8)),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 0.5 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"], reader=reader),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Lambdad(keys=["label"], func=lambda x: (x == 255).astype(np.uint8)),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 0.5 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )

    elif dataset_name == 'TDSC-ABUS':

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )
        test_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(
                    keys=["image", "label"],
                    source_key="image",
                    select_fn=lambda x: x > min_int,
                    allow_smaller=True,
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )

    elif dataset_name == 'ISLES22':
        # ISLES22: Acute Ischemic Stroke Lesion Segmentation
        # 3-channel input: DWI (image1) + ADC (image2) + FLAIR (image3, registered)
        # Binary segmentation (ischemic stroke lesion)
        # Data preprocessed to 2.0mm isotropic
        
        train_transforms = Compose(
            [
                # Load 3 NIfTI images and stack them together
                LoadImaged(keys=["image1", "image2", "image3", "label"], ensure_channel_first=True),
                ConcatItemsd(keys=["image1", "image2", "image3"], name='image', dim=0),
                DeleteItemsd(keys=["image1", "image2", "image3"]),
                EnsureTyped(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                # Data is already 2.0mm isotropic, Spacingd maintains consistency
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(2.0 / resize_scale, 2.0 / resize_scale, 2.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                # Percentile-based normalization for MRI (channel-wise)
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandSpatialCropSamplesd(
                    keys=["image", "label"], num_samples=4, roi_size=(image_size, image_size, image_size), random_size=False
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image1", "image2", "image3", "label"], ensure_channel_first=True),
                ConcatItemsd(keys=["image1", "image2", "image3"], name='image', dim=0),
                DeleteItemsd(keys=["image1", "image2", "image3"]),
                EnsureTyped(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(2.0 / resize_scale, 2.0 / resize_scale, 2.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )

    elif dataset_name in ('AMOS22', 'AMOS22_CT', 'AMOS22_MR'):
        # AMOS22: Multi-Organ Segmentation (Task 2: CT + MRI mixed)
        # 15 foreground classes + background
        # Single-channel input, mixed modality (CT and MRI)
        # Uses percentile-based normalization for modality flexibility
        
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 1.5 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                # Percentile-based normalization works for both CT and MRI
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.5, upper=99.5, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int, allow_smaller=True),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 1.5 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.5, upper=99.5, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int, allow_smaller=True),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )

    elif dataset_name == 'KiTS23':
        # KiTS23: Kidney Tumor Segmentation Challenge
        # 4 classes: Background (0), Kidney (1), Tumor (2), Cyst (3)
        # CT scans with kidney-specific HU window
        # Using soft tissue/abdominal window for kidney visualization
        
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 1.5 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                # Kidney CT window: soft tissue window (-100 to 300 HU)
                ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=300, b_min=min_int, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int, allow_smaller=True),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 1.5 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=300, b_min=min_int, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int, allow_smaller=True),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )

    elif dataset_name == 'LiTS':
        # LiTS: Liver Tumor Segmentation
        # 3 classes: Background (0), Liver (1), Tumor (2)
        # CT scans with soft-tissue abdominal window and isotropic target spacing

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 1.5 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=min_int, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int, allow_smaller=True),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 1.5 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=min_int, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int, allow_smaller=True),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )

    elif dataset_name == 'WORD':
        # WORD: Whole abdomen ORgan segmentation Dataset
        # 17 classes: Background (0) + 16 abdominal organs
        # CT scans with standard abdominal window (same as BTCV)
        
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 2.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=min_int, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int, allow_smaller=True),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 2.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=min_int, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int, allow_smaller=True),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )

    elif dataset_name == 'TotalSegmenterCT':
        # TotalSegmenter CT: Whole-body 104-organ segmentation
        # 105 classes: Background (0) + 104 organs
        # CT scans with standard abdominal window (same as BTCV/WORD)
        
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 2.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=min_int, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int, allow_smaller=True),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5 / resize_scale, 1.5 / resize_scale, 2.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=min_int, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > min_int, allow_smaller=True),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    if test_transforms is None:
        test_transforms = val_transforms

    # Flatten transforms to allow caching of non-random transforms if needed
    train_transforms = train_transforms.flatten()
    val_transforms = val_transforms.flatten()
    test_transforms = test_transforms.flatten()

    return train_transforms, val_transforms, test_transforms
