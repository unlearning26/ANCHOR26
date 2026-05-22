# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar
import time
import os
import subprocess
from pathlib import Path
from copy import deepcopy
import random
import hashlib
import pickle

import torch
import torch.distributed as dist
from torch.utils.data import Sampler, Dataset
from monai.data import CacheNTransDataset, PersistentDataset
from monai.transforms import Compose, apply_transform
import json

from .samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler


logger = logging.getLogger("dinov2")


class ReadOnlyCacheDataset(Dataset):
    """
    A read-only cache dataset that loads from precomputed cache files and NEVER writes.
    
    This prevents Lustre filesystem race conditions when multiple workers try to
    write cache files simultaneously.
    
    Cache files are expected to follow MONAI's CacheNTransDataset format:
    - Filename: {md5_hash_of_data_item}.pt
    - Content: Preprocessed tensor after cache_n_trans transforms
    
    If a cache file is missing, the transforms are applied on-the-fly without caching.
    """
    
    def __init__(
        self,
        data: List[dict],
        transform: Optional[Callable],
        cache_dir: str,
        cache_n_trans: int = 5,
    ):
        """
        Args:
            data: List of data items (dicts with 'image', 'spacing', etc.)
            transform: Full transform pipeline (Compose)
            cache_dir: Directory containing precomputed .pt cache files
            cache_n_trans: Number of transforms cached (for applying remaining transforms)
        """
        self.data = data
        self.transform = transform
        self.cache_dir = Path(cache_dir)
        self.cache_n_trans = cache_n_trans
        
        # Split transforms into cached and remaining
        if isinstance(transform, Compose):
            self.cached_transforms = Compose(transform.transforms[:cache_n_trans])
            self.remaining_transforms = Compose(transform.transforms[cache_n_trans:])
        else:
            # Fallback: assume all transforms are cached
            self.cached_transforms = transform
            self.remaining_transforms = None
        
        # Pre-compute cache file paths (hash only depends on data item, not transforms)
        self._cache_files = []
        for item in data:
            cache_hash = hashlib.md5(pickle.dumps(item)).hexdigest()
            self._cache_files.append(self.cache_dir / f"{cache_hash}.pt")
        
        logger.info(f"[ReadOnlyCacheDataset] Initialized with {len(data)} samples, cache_dir={cache_dir}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        cache_file = self._cache_files[index]
        
        if cache_file.exists():
            # Load from precomputed cache (read-only, no writes)
            try:
                cached_data = torch.load(cache_file, map_location='cpu', weights_only=False)
                
                # Apply remaining transforms (the ones after cache_n_trans)
                if self.remaining_transforms is not None:
                    return apply_transform(self.remaining_transforms, cached_data)
                return cached_data
                
            except Exception as e:
                logger.warning(f"[ReadOnlyCacheDataset] Failed to load cache file {cache_file}: {e}. "
                             f"Falling back to on-the-fly transform (no caching).")
        
        # Cache miss or load failed - apply full transform without caching
        # This is slow but prevents any writes to disk
        data_item = self.data[index]
        if self.transform is not None:
            return apply_transform(self.transform, data_item)
        return data_item


class SamplerType(Enum):
    DISTRIBUTED = 0
    EPOCH = 1
    INFINITE = 2
    SHARDED_INFINITE = 3
    SHARDED_INFINITE_NEW = 4


def _make_bool_str(b: bool) -> str:
    return "yes" if b else "no"


def _make_sample_transform(image_transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
    def transform(sample):
        image, target = sample
        if image_transform is not None:
            image = image_transform(image)
        if target_transform is not None:
            target = target_transform(target)
        return image, target

    return transform


def make_dataset_3d(
    *,
    dataset_path: str,
    cache_path: str,
    data_min_axis_size: int,
    transform: Optional[Callable] = None,
    rank0_cache: bool = False,
    read_only_cache: bool = False,
    # 1. ADD THE REQUIRED ARGUMENT (Default for standard training runs)
    cache_n_trans: int = 5  # Cache transforms 0-4 (up to CropForegroundSwapSliceDimsV2)
):
    """
    Creates a 3d input dataset with the specified parameters.

    Args:
        dataset_path: A path to a list of sample paths for MONAI datasets.
        cache_path: A path to a directory to cache the dataset.
        data_min_axis_size: The minimum size of the smallest axis of the data.
        transform: A transform to apply to images.
        rank0_cache: If True, only rank 0 writes cache, others wait and read.
        read_only_cache: If True, verify all cache files exist before training.
            Fails fast on cache miss instead of regenerating. Use with precomputed cache.
    Returns:
        The created dataset.
    """
    logger.info(f'creating 3d dataset from datalist: {dataset_path}')

    # load datalist
    with open(dataset_path, 'r') as json_f:
        datalist = json.load(json_f)
    
    # debug
    logger.info(f"original dataset size: {len(datalist):,d}")
  
    # filter overly small data
    datalist = [x for x in datalist if min(x['shape'][:3]) > data_min_axis_size]
    
    # Verify precomputed cache when read_only_cache=True
    if read_only_cache:
        logger.info(f"[read_only_cache] Verifying cache at: {cache_path}")
        cache_dir = Path(cache_path)
        
        if not cache_dir.exists():
            raise RuntimeError(
                f"read_only_cache=True but cache directory does not exist: {cache_path}\n"
                f"Run precompute_cache_v2.py first to generate cache files."
            )
        
        # Check each sample has a corresponding cache file
        # MONAI uses: hashlib.md5(pickle.dumps(item)).hexdigest() + ".pt"
        missing_files = []
        for i, item in enumerate(datalist):
            cache_hash = hashlib.md5(pickle.dumps(item)).hexdigest()
            cache_file = cache_dir / f"{cache_hash}.pt"
            if not cache_file.exists():
                missing_files.append((i, item.get("image", "unknown"), cache_file.name))
                # Limit logging for large datasets
                if len(missing_files) <= 10:
                    logger.error(f"Missing cache file [{i}]: {item.get('image', 'unknown')}")
        
        if missing_files:
            raise RuntimeError(
                f"read_only_cache=True but {len(missing_files)} of {len(datalist)} cache files are missing.\n"
                f"First 10 missing: {[m[1] for m in missing_files[:10]]}\n"
                f"Run precompute_cache_v2.py first to generate ALL cache files."
            )
        
        logger.info(f"[read_only_cache] Cache verified: all {len(datalist)} files present in {cache_path}")
        
        # Use ReadOnlyCacheDataset to prevent any disk writes
        logger.info(f"[read_only_cache] Using ReadOnlyCacheDataset (no disk writes)")
        dataset = ReadOnlyCacheDataset(
            data=datalist,
            transform=transform,
            cache_dir=cache_path,
            cache_n_trans=cache_n_trans,
        )
        
    # Rank 0 caching strategy - prevents race conditions on Lustre (only when NOT read_only_cache)
    elif rank0_cache and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        if rank == 0:
            logger.info(f"[Rank 0] Building cache for {len(datalist)} samples...")
            dataset = CacheNTransDataset(datalist, transform=transform, cache_n_trans=cache_n_trans, cache_dir=cache_path)
            
            # Force filesystem sync to ensure all cache files are fully written to Lustre
            logger.info(f"[Rank 0] Cache complete. Forcing filesystem sync...")
            try:
                subprocess.run(['sync'], check=True, timeout=120)
                # Verify cache files are accessible
                cache_dir = Path(cache_path)
                cache_files = list(cache_dir.glob("*.pt"))
                logger.info(f"[Rank 0] Filesystem synced. {len(cache_files)} cache files ready.")
            except Exception as e:
                logger.warning(f"[Rank 0] Filesystem sync warning (non-fatal): {e}")
            
            logger.info(f"[Rank 0] Signaling other ranks...")
        
        # Barrier: rank 0 finishes caching before others proceed
        dist.barrier()
        
        # Extra delay after barrier for Lustre metadata propagation across nodes
        if rank != 0:
            # Base delay for Lustre sync + staggered delay to prevent thundering herd
            lustre_sync_delay = 5.0  # Wait for Lustre metadata to propagate
            stagger_delay = rank * 0.1  # 0.1s per rank to spread load
            total_delay = lustre_sync_delay + stagger_delay
            logger.info(f"[Rank {rank}] Waiting {total_delay:.1f}s for Lustre sync + staggered access...")
            time.sleep(total_delay)
            logger.info(f"[Rank {rank}] Loading from precomputed cache...")
            dataset = CacheNTransDataset(datalist, transform=transform, cache_n_trans=cache_n_trans, cache_dir=cache_path)
    else:
        # Standard behavior when rank0_cache=False: all ranks may write cache (race-prone)
        dataset = CacheNTransDataset(datalist, transform=transform, cache_n_trans=cache_n_trans, cache_dir=cache_path)

    # Aggregated datasets do not expose (yet) these attributes, so add them.
    if not hasattr(dataset, "transform"):
        setattr(dataset, "transform", transform)

    return dataset


def make_segmentation_dataset_3d(
    dataset_name: str,
    dataset_percent: int,
    base_directory: str,
    train_transforms: Callable,
    val_transforms: Callable,
    test_transforms: Callable | None,
    cache_path: str,
    batch_size: int,
):
    """
    Creates a 3d segmentation dataset with the specified parameters.

    Args:
        dataset_name: Name of the segmentation dataset (BTCV, BraTS, LA-SEG, TDSC-ABUS).
        dataset_percent: Percentage of the dataset to use for training.
        base_directory: Base directory where dataset json files are stored.
        train_transforms: Training transforms to apply to images.
        val_transforms: Validation transforms to apply to images.
        test_transforms: Test transforms to apply to images. If None, validation transforms are reused.
        cache_path: A path to a directory to cache the dataset, used in PersistentDataset.
        batch_size: Batch size for the dataset.
    Returns:
        Created train, val, and test datasets, number of input channels, and number of classes for the dataset.
    """

    if dataset_name == 'BTCV':
        datalist_path = os.path.join(base_directory, 'BTCV_100_datalist.json')
        class_num = 14
        input_channels = 1
    elif dataset_name == 'BraTS':
        # Use 3DINOv2 paper-aligned datalist (TCGA samples excluded)
        # This matches the paper's 758/108/218 train/val/test split
        datalist_path = os.path.join(base_directory, 'BraTS_100_datalist_3dinov2.json')
        class_num = 3
        input_channels = 4
    elif dataset_name == 'BraTS-full':
        # Alternative: Use full BraTS 2023 dataset (includes TCGA)
        datalist_path = os.path.join(base_directory, 'BraTS_GLI_100_datalist.json')
        class_num = 3
        input_channels = 4
    elif dataset_name == 'LA-SEG':
        datalist_path = os.path.join(base_directory, 'LA-SEG_100_datalist.json')
        class_num = 2
        input_channels = 1
    elif dataset_name == 'TDSC-ABUS':
        datalist_path = os.path.join(base_directory, 'TDSC-ABUS_100_datalist.json')
        class_num = 2
        input_channels = 1
    elif dataset_name == 'ISLES22':
        # ISLES 2022 Acute Ischemic Stroke Lesion Segmentation
        # 3-channel input: DWI + ADC + FLAIR (registered)
        # Binary segmentation (ischemic stroke lesion)
        datalist_path = os.path.join(base_directory, 'ISLES22_100_datalist.json')
        class_num = 1
        input_channels = 3
    elif dataset_name == 'AMOS22':
        # AMOS22 Multi-Organ Segmentation (Task 2: CT + MRI mixed)
        # 15 foreground classes + background = 16 total
        # Single-channel input (CT or MRI per sample)
        datalist_path = os.path.join(base_directory, 'AMOS22_100_datalist.json')
        class_num = 16
        input_channels = 1
    elif dataset_name == 'AMOS22_CT':
        # AMOS22 CT-only subset (IDs <= 500)
        datalist_path = os.path.join(base_directory, 'AMOS22_CT_100_datalist.json')
        class_num = 16
        input_channels = 1
    elif dataset_name == 'AMOS22_MR':
        # AMOS22 MRI-only subset (IDs > 500)
        datalist_path = os.path.join(base_directory, 'AMOS22_MR_100_datalist.json')
        class_num = 16
        input_channels = 1
    elif dataset_name == 'KiTS23':
        # KiTS23 Kidney Tumor Segmentation Challenge
        # 4 classes: Background (0), Kidney (1), Tumor (2), Cyst (3)
        # CT scans, single-channel input
        datalist_path = os.path.join(base_directory, 'KiTS23_100_datalist.json')
        class_num = 4
        input_channels = 1
    elif dataset_name == 'LiTS':
        # LiTS Liver Tumor Segmentation
        # 3 classes: Background (0), Liver (1), Tumor (2)
        # CT scans, single-channel input
        datalist_path = os.path.join(base_directory, 'LiTS_100_datalist.json')
        class_num = 3
        input_channels = 1
    elif dataset_name == 'WORD':
        # WORD: Whole abdomen ORgan segmentation Dataset
        # 17 classes: Background (0) + 16 organs
        # CT scans, single-channel input
        datalist_path = os.path.join(base_directory, 'WORD_100_datalist.json')
        class_num = 17
        input_channels = 1
    elif dataset_name == 'TotalSegmenterCT':
        # TotalSegmenter CT: Whole-body 104-organ segmentation
        # 105 classes: Background (0) + 104 organs
        # CT scans, single-channel input
        datalist_path = os.path.join(base_directory, 'TotalSegmenterCT_100_datalist.json')
        class_num = 105
        input_channels = 1
    else:
        raise ValueError(f'Unsupported dataset "{dataset_name}"')

    with open(datalist_path, 'r') as json_f:
        datalist = json.load(json_f)

    train_data_ind = int(round(len(datalist['training']) * (dataset_percent / 100)))

    train_datalist = datalist['training'][:train_data_ind]
    val_datalist = datalist['validation']
    test_datalist = datalist['test']
    logger.info(f"# of train samples: {len(train_datalist):,d}")
    logger.info(f"# of val samples: {len(val_datalist):,d}")
    logger.info(f"# of test samples: {len(test_datalist):,d}")

    if len(train_datalist) < batch_size:
        logger.info(f"copying train samples to match batch size: {batch_size:,d}")
        copied_datalist = []
        for i in range(batch_size // len(train_datalist)):
            copied_datalist.extend(deepcopy(train_datalist))
        assert len(copied_datalist) == batch_size
        train_datalist = copied_datalist

    if test_transforms is None:
        test_transforms = val_transforms

    test_cache_path = cache_path
    if test_transforms is not val_transforms:
        test_cache_path = os.path.join(cache_path, "test")

    train_dataset = PersistentDataset(train_datalist, transform=train_transforms, cache_dir=cache_path)
    val_dataset = PersistentDataset(val_datalist, transform=val_transforms, cache_dir=cache_path)
    test_dataset = PersistentDataset(test_datalist, transform=test_transforms, cache_dir=test_cache_path)

    return train_dataset, val_dataset, test_dataset, input_channels, class_num


def make_classification_dataset_3d(
    dataset_name: str,
    dataset_percent: int,
    base_directory: str,
    train_transforms: Callable,
    val_transforms: Callable,
    cache_path: str,
    dataset_seed: int,
):
    """
    Creates a 3d classification dataset with the specified parameters.

    Args:
        dataset_name: Name of the classification dataset (ICBM, COVID-CT-MD).
        dataset_percent: Percentage of the dataset to use for training.
        base_directory: Base directory where dataset json files are stored.
        train_transforms: Training transforms to apply to images.
        val_transforms: Validation transforms to apply to images.
        cache_path: A path to a directory to cache the dataset, used in PersistentDataset.
        dataset_seed: Seed for random shuffling of the dataset.
    Returns:
        Created train, val, and test datasets, and number of classes for the dataset.
    """

    if dataset_name == 'ICBM':
        datalist_path = os.path.join(base_directory, 'ICBM_cls_datalist.json')
        class_num = 4
    elif dataset_name == 'COVID-CT-MD':
        datalist_path = os.path.join(base_directory, 'COVID-CT-MD_cls_datalist.json')
        class_num = 3
    else:
        raise ValueError(f'Unsupported dataset "{dataset_name}"')

    with open(datalist_path, 'r') as json_f:
        datalist = json.load(json_f)

    # filter ages for icbm
    if dataset_name == 'ICBM':

        # NOTE: Commented out - our preprocessed ICBM datalist already has correct paths (*_brain.nii.gz)
        # for k in datalist:
        #     for item in datalist[k]:
        #         item['image'] = item['image'].replace('.nii.gz', '_mask.nii.gz')

        datalist['training'] = [x for x in datalist['training'] if 20 <= x['label'] <= 60]
        datalist['validation'] = [x for x in datalist['validation'] if 20 <= x['label'] <= 60]
        # ICBM uses 'testing' key, not 'test'
        test_key = 'testing' if 'testing' in datalist else 'test'
        datalist[test_key] = [x for x in datalist[test_key] if 20 <= x['label'] <= 60]

    # ensure reproducible shuffling
    random.Random(dataset_seed).shuffle(datalist['training'])
    print(f'Shuffled with seed: {dataset_seed}')

    train_data_ind = int(round(len(datalist['training']) * (dataset_percent / 100)))
    train_datalist = datalist['training'][:train_data_ind]
    val_datalist = datalist['validation']
    # Handle both 'test' and 'testing' keys
    test_datalist = datalist.get('test', datalist.get('testing', []))

    logger.info(f"# of train samples: {len(train_datalist):,d}")
    logger.info(f"# of val samples: {len(val_datalist):,d}")
    logger.info(f"# of test samples: {len(test_datalist):,d}")

    train_dataset = PersistentDataset(train_datalist, transform=train_transforms, cache_dir=cache_path)
    val_dataset = PersistentDataset(val_datalist, transform=val_transforms, cache_dir=cache_path)
    test_dataset = PersistentDataset(test_datalist, transform=val_transforms, cache_dir=cache_path)

    return train_dataset, val_dataset, test_dataset, class_num


def _make_sampler(
    *,
    dataset,
    type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
    advance: int = 0,
) -> Optional[Sampler]:
    sample_count = len(dataset)

    if type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
        )
    elif type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        logger.info("sampler: sharded infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        # TODO: Remove support for old shuffling
        use_new_shuffle_tensor_slice = type == SamplerType.SHARDED_INFINITE_NEW
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=use_new_shuffle_tensor_slice,
        )
    elif type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        if advance > 0:
            raise NotImplementedError("sampler advance > 0 is not supported")
        size = size if size > 0 else sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )
    elif type == SamplerType.DISTRIBUTED:
        logger.info("sampler: distributed")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        if advance > 0:
            raise ValueError("sampler advance > 0 is invalid")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    logger.info("sampler: none")
    return None


T = TypeVar("T")


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    sampler_size: int = -1,
    sampler_advance: int = 0,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[List[T]], Any]] = None,
):
    """
    Creates a data loader with the specified parameters.

    Args:
        dataset: A dataset (third party, LaViDa or WebDataset).
        batch_size: The size of batches to generate.
        num_workers: The number of workers to use.
        shuffle: Whether to shuffle samples.
        seed: The random seed to use.
        sampler_type: Which sampler to use: EPOCH, INFINITE, SHARDED_INFINITE, SHARDED_INFINITE_NEW, DISTRIBUTED or None.
        sampler_size: The number of images per epoch (when applicable) or -1 for the entire dataset.
        sampler_advance: How many samples to skip (when applicable).
        drop_last: Whether the last non-full batch of data should be dropped.
        persistent_workers: maintain the workers Dataset instances alive after a dataset has been consumed once.
        collate_fn: Function that performs batch collation
    """

    sampler = _make_sampler(
        dataset=dataset,
        type=sampler_type,
        shuffle=shuffle,
        seed=seed,
        size=sampler_size,
        advance=sampler_advance,
    )

    logger.info("using PyTorch data loader")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:  # data loader has no length
        logger.info("infinite data loader")
    return data_loader
