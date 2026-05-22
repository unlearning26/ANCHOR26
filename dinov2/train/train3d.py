# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from pathlib import Path
import signal
import sys
import time
from functools import partial
from monai.transforms import Compose, LoadImaged, ScaleIntensityRangePercentilesd, Lambdad
import random

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch

from dinov2.data import SamplerType, make_data_loader, make_dataset_3d
from dinov2.data import collate_data_and_cast, DataAugmentationDINO3d, MaskingGenerator3d
from dinov2.data import CropForegroundSwapSliceDimsV2
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup_3d
from dinov2.utils.utils import CosineScheduler

from dinov2.train.ssl_meta_arch import SSLMetaArch

torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def generate_experiment_name(cfg, dataset_path=None, num_nodes=None, timestamp=False, high_res=None):
    """
    Generate descriptive experiment directory name from config parameters.
    
    Format: med3dino_{dataset}_{model}_{resolution}_{crop}_{batch}_{epochs}_{nodes}
    Example: med3dino_62k_vitL16_c96_sa_bs384_100ep_6n
    
    Args:
        cfg: OmegaConf configuration object
        dataset_path: Path to dataset manifest (to extract size)
        num_nodes: Number of training nodes (if None, extracted from distributed)
        timestamp: Whether to append timestamp (default: False for restart safety)
        high_res: Global crop size override (if None, uses cfg.crops.global_crops_size)
    
    Returns:
        str: Descriptive experiment name
    """
    from datetime import datetime
    import json
    
    components = ["med3dino"]
    
    # Debug logging for dataset path
    logger.info(f"[AutoNaming] generate_experiment_name called with:")
    logger.info(f"  dataset_path: {dataset_path}")
    logger.info(f"  dataset_path type: {type(dataset_path)}")
    if dataset_path:
        logger.info(f"  dataset_path exists: {os.path.exists(dataset_path)}")
    
    # 1. Dataset size (e.g., "62k", "140k")
    if dataset_path and os.path.exists(dataset_path):
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
                num_samples = len(data) if isinstance(data, list) else data.get('num_samples', 0)
                if num_samples >= 1000:
                    components.append(f"{num_samples // 1000}k")
                else:
                    components.append(f"{num_samples}")
                logger.info(f"Dataset size extracted: {num_samples} samples from {dataset_path}")
        except Exception as e:
            logger.warning(f"Failed to extract dataset size from {dataset_path}: {e}")
            components.append("unkn")
    else:
        if dataset_path:
            logger.warning(f"Dataset path provided but does not exist: {dataset_path}")
        else:
            logger.warning("No dataset path provided for auto-naming")
        components.append("unkn")
    
    # 2. Model architecture (e.g., "vitL16" for ViT-Large patch_size=16)
    model_map = {
        "vit_large": "vitL",
        "vit_large_3d": "vitL",
        "vit_base": "vitB",
        "vit_base_3d": "vitB",
        "vit_small": "vitS",
        "vit_small_3d": "vitS",
        "vit_giant": "vitG"
    }
    model_name = model_map.get(cfg.student.arch, cfg.student.arch.replace('_3d', '').replace('vit_', 'vit')[:5])
    patch_size = cfg.student.patch_size
    components.append(f"{model_name}{patch_size}")
    
    # 3. Resolution/crop size (e.g., "c96", "c128", "c192")
    crop_size = high_res if high_res is not None else cfg.crops.global_crops_size
    components.append(f"c{crop_size}")
    
    # 4. Crop mode ("sa" for spacing-aware, "rel" for relative)
    crop_mode = "sa" if cfg.crops.use_spacing_aware else "rel"
    components.append(crop_mode)
    
    # 5. Total batch size (e.g., "bs384" for 6 nodes × 4 GPUs × 16 per GPU)
    if num_nodes is None:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                world_size = dist.get_world_size()
            else:
                world_size = 1
        except:
            world_size = 1
    else:
        gpus_per_node = 4  # Leonardo cluster config
        world_size = num_nodes * gpus_per_node
    
    batch_size_per_gpu = cfg.train.batch_size_per_gpu
    total_batch_size = world_size * batch_size_per_gpu
    components.append(f"bs{total_batch_size}")
    
    # 6. Number of epochs (e.g., "100ep")
    epochs = cfg.optim.epochs
    components.append(f"{epochs}ep")
    
    # 7. Number of nodes (e.g., "6n")
    if num_nodes:
        components.append(f"{num_nodes}n")
    
    # 8. Optional timestamp (YYYYMMDD_HHMMSS)
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        components.append(ts)
    
    return "_".join(components)


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("3DINO training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    parser.add_argument("--local-rank", default=0, type=int, help="Variable for distributed computing.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        type=str,
        help="path to cache directory for monai persistent dataset"
    )
    parser.add_argument(
        "--auto-name-dir",
        action="store_true",
        help="Automatically generate descriptive experiment name from config parameters"
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        type=str,
        help="Path to dataset manifest (used for auto-naming to extract dataset size)"
    )
    parser.add_argument(
        "--num-nodes",
        default=None,
        type=int,
        help="Number of training nodes (used for auto-naming if distributed not initialized)"
    )
    # New arguments for quick testing and job chaining
    parser.add_argument(
        "--checkpoint-period",
        default=None,
        type=int,
        help="Checkpoint every N iterations (overrides default epoch-based checkpointing)"
    )
    parser.add_argument(
        "--max-iter",
        default=None,
        type=int,
        help="Maximum training iterations (overrides epochs * OFFICIAL_EPOCH_LENGTH)"
    )
    parser.add_argument(
        "--max-runtime-hours",
        default=23.5,
        type=float,
        help="Maximum runtime in hours before saving checkpoint and exiting (default: 23.5h for 24h jobs)"
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def do_test(cfg, model, iteration):
    new_state_dict = model.teacher.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)


def do_train(cfg, model, args, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training
    
    # Global variables for signal handling
    should_stop = False
    current_checkpointer = None
    current_iteration = 0
    
    def signal_handler(signum, frame):
        nonlocal should_stop, current_checkpointer, current_iteration
        signal_names = {signal.SIGTERM: "SIGTERM", signal.SIGINT: "SIGINT", signal.SIGUSR1: "SIGUSR1"}
        sig_name = signal_names.get(signum, str(signum))
        logger.info(f"Signal {sig_name} ({signum}) received - saving emergency checkpoint")
        if current_checkpointer is not None and hasattr(current_checkpointer, 'dirpath') and current_checkpointer.dirpath:
            try:
                checkpoint_name = f"model_{current_iteration:07d}"
                current_checkpointer.save(checkpoint_name, iteration=current_iteration)
                logger.info(f"Emergency checkpoint saved: {checkpoint_name}")
                # Verify checkpoint was saved
                expected_path = Path(current_checkpointer.dirpath) / f"{checkpoint_name}.rank_0.pth"
                if expected_path.exists():
                    logger.info(f"Checkpoint verified: {expected_path}")
                else:
                    logger.warning(f"Checkpoint file not found after save: {expected_path}")
                # Update last_checkpoint file
                import torch.distributed as dist
                if not dist.is_initialized() or dist.get_rank() == 0:
                    last_checkpoint_file = Path(current_checkpointer.dirpath) / "last_checkpoint"
                    with open(last_checkpoint_file, 'w') as f:
                        f.write(f"{checkpoint_name}.rank_0.pth")
                # Exit immediately after successful checkpoint save
                # This ensures we don't wait for the training loop to check should_stop
                logger.info(f"Signal handler exiting gracefully with code 0 for job chaining")
                logging.shutdown()
                sys.exit(0)
            except Exception as e:
                logger.error(f"Emergency checkpoint failed: {e}")
                # Still try to exit cleanly even if checkpoint failed
                logging.shutdown()
                sys.exit(1)
        else:
            logger.warning("Cannot save emergency checkpoint: checkpointer not initialized")
            # Exit with error if we can't save checkpoint
            logging.shutdown()
            sys.exit(1)
    
    # Register signal handlers for SLURM termination signals
    signal.signal(signal.SIGTERM, signal_handler)  # SLURM sends this before job timeout
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGUSR1, signal_handler)  # Custom signal

    # setup optimizer
    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    # checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
    current_checkpointer = checkpointer  # For signal handler
    
    # Enhanced checkpoint loading with better error handling
    checkpoint_data = {}
    if resume:
        try:
            checkpoint_data = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=True)
            logger.info(f"Successfully resumed from checkpoint. Iteration: {checkpoint_data.get('iteration', -1) + 1}")
        except Exception as e:
            logger.warning(f"Failed to resume from checkpoint: {e}. Starting from scratch.")
            checkpoint_data = {}
    else:
        if cfg.MODEL.WEIGHTS:
            try:
                checkpoint_data = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
                logger.info(f"Loaded pretrained weights from: {cfg.MODEL.WEIGHTS}")
            except Exception as e:
                logger.warning(f"Failed to load pretrained weights: {e}")
                checkpoint_data = {}
    
    start_iter = checkpoint_data.get("iteration", -1) + 1
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
    
    # Allow overriding max_iter via command line (for testing job chaining)
    if args.max_iter is not None and args.max_iter > 0:
        max_iter = args.max_iter
        logger.info(f"Using custom max_iter from command line: {max_iter}")
    
    # Allow configurable checkpoint period (default: every epoch)
    checkpoint_period = OFFICIAL_EPOCH_LENGTH  # Default: every epoch
    if args.checkpoint_period is not None and args.checkpoint_period > 0:
        checkpoint_period = args.checkpoint_period
        logger.info(f"Using custom checkpoint_period from command line: {checkpoint_period}")

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=checkpoint_period,
        max_iter=max_iter,
        max_to_keep=5,  # Keep more checkpoints for safety (default is 3)
    )

    # setup data preprocessing
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 3
    mask_generator = MaskingGenerator3d(
        input_size=(img_size // patch_size, img_size // patch_size, img_size // patch_size)
    )

    def random_select_time(x):  # this function is operational only when mulitple channels/time axis exists.
        # if time axis exists, select random time slice
        if x.shape[0] > 1:
            t = random.randint(0, x.shape[0] - 1)
            x = x[t:t + 1]
        return x



    # This is the primary transform pipeline for SSL training.
    # Architecture:
    #   1. Load & normalize: LoadImaged → ScaleIntensityRangePercentilesd
    #   2. Foreground crop: CropForegroundSwapSliceDimsV2 (anisotropy-aware axis swap)
    
    base_transforms = [
        LoadImaged(keys=["image"], ensure_channel_first=True, meta_keys=["image_meta_dict"]),
        Lambdad(keys=["image"], func=random_select_time),
        Lambdad(
            keys=["image"], func=lambda x: torch.nan_to_num(x, torch.nanmean(x).item())
        ),  # replace NaNs with mean
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True),
        # Anisotropy-aware foreground crop + axis permutation
        CropForegroundSwapSliceDimsV2(select_fn=lambda x: x > -1),
        
    ]
    
    # Add augmentation
    base_transforms.append(
        DataAugmentationDINO3d(
            # Baseline (3DINO) parameters
            global_crops_in_slice_scale=cfg.crops.global_crops_in_slice_scale,
            global_crops_cross_slice_scale=cfg.crops.global_crops_cross_slice_scale,
            local_crops_in_slice_scale=cfg.crops.local_crops_in_slice_scale,
            local_crops_cross_slice_scale=cfg.crops.local_crops_cross_slice_scale,
            # Spacing-aware parameters
            global_crops_physical_scale_mm=cfg.crops.global_crops_physical_scale_mm,
            local_crops_physical_scale_mm=cfg.crops.local_crops_physical_scale_mm,
            anisotropy_threshold=cfg.crops.anisotropy_threshold,
            # Common parameters
            local_crops_number=cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            # Mode control
            use_spacing_aware=cfg.crops.use_spacing_aware,
        )
    )
    
    data_transform = Compose(base_transforms)

    # data collate
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # setup data loader
    cache_base = args.cache_dir if args.cache_dir is not None else getattr(cfg.train, 'cache_dir', None)
    rank0_cache = getattr(cfg.train, 'rank0_cache', False)
    
    # Determine cache path
    # Priority:
    # 1. read_only_cache=True → use cache_base DIRECTLY (precomputed cache, no nesting)
    # 2. cache_base contains "cache_med3dino_" → already has experiment name, use directly
    # 3. Otherwise → auto-generate nested subdirectory with experiment name
    read_only_cache = getattr(cfg.train, 'read_only_cache', False)
    
    if cache_base:
        if read_only_cache:
            # Precomputed cache mode: use path directly, no nesting!
            # Precomputed caches are named like "cache_60k_precomputed" with *.pt files at root
            cache_path = cache_base
            logger.info(f"Using precomputed cache directory (read_only): {cache_path}")
        elif "cache_med3dino_" in os.path.basename(cache_base):
            # Already includes experiment name from setup_3d(), use directly
            cache_path = cache_base
            logger.info(f"Using pre-configured cache directory: {cache_path}")
        else:
            # Raw base path - generate full path with experiment name
            cache_name = generate_experiment_name(cfg, dataset_path=cfg.train.dataset_path, num_nodes=args.num_nodes)
            cache_name = "cache_" + cache_name  # Prefix with "cache_"
            cache_path = os.path.join(cache_base, cache_name)
            logger.info(f"Auto-generated cache directory: {cache_path}")
    else:
        cache_path = None
        logger.warning("No cache directory specified - caching disabled")
    
    # Log caching strategy (read_only_cache already set above)
    if read_only_cache:
        logger.info("Using read_only_cache mode (precomputed cache, fail on miss)")
    elif rank0_cache:
        logger.info("Using rank0_cache mode (rank 0 generates, others wait)")
    else:
        logger.info("Using standard caching (all ranks may write - not recommended for multi-node)")
    
    dataset = make_dataset_3d(
        dataset_path=cfg.train.dataset_path,
        cache_path=cache_path,
        data_min_axis_size=cfg.train.data_min_axis_size,
        transform=data_transform,
        rank0_cache=rank0_cache,
        read_only_cache=read_only_cache,
    )
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        # persistent_workers=True,
        shuffle=True,
        seed=start_iter,
        sampler_type=sampler_type,
        sampler_advance=0,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # training loop
    iteration = start_iter
    
    # Time-based checkpointing for HPC walltime limits
    training_start_time = time.time()
    max_runtime_seconds = args.max_runtime_hours * 3600 if hasattr(args, 'max_runtime_hours') else 23.5 * 3600

    logger.info("Starting training from iteration {}".format(start_iter))
    logger.info(f"Max runtime: {max_runtime_seconds/3600:.1f} hours")
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    st = time.time()

    for data in metric_logger.log_every(
        data_loader,
        5,
        header,
        max_iter,
        start_iter,
    ):
        print(f'batch time: {time.time() - st}')
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        current_iteration = iteration  # Update for signal handler
        
        # Check for termination conditions
        if iteration > max_iter or should_stop:
            if should_stop:
                logger.info("Training stopped due to signal. Checkpoint already saved by signal handler.")
            logger.info(f"Training complete. Final iteration: {iteration}")
            return 0  # Explicit success exit code
        
        # Time-based checkpoint: save and exit before walltime
        elapsed_time = time.time() - training_start_time
        if elapsed_time > max_runtime_seconds:
            logger.info(f"Approaching walltime limit ({elapsed_time/3600:.2f}h elapsed)")
            logger.info(f"Saving checkpoint at iteration {iteration} and exiting gracefully")
            # Synchronize all ranks before checkpoint
            if distributed.get_global_size() > 1:
                torch.distributed.barrier()
            checkpoint_name = f"model_{iteration:07d}"
            periodic_checkpointer.checkpointer.save(checkpoint_name, iteration=iteration)
            # Synchronize after checkpoint to ensure all ranks complete
            if distributed.get_global_size() > 1:
                torch.distributed.barrier()
            # Update last_checkpoint file
            if distributed.is_main_process():
                last_checkpoint_file = Path(cfg.train.output_dir) / "last_checkpoint"
                with open(last_checkpoint_file, 'w') as f:
                    f.write(f"{checkpoint_name}.rank_0.pth")
            logger.info(f"Time-based checkpoint saved. Exiting for job chain to continue.")
            # Flush logs before exit
            logging.shutdown()
            return 0  # Success exit for job chaining
        

        # apply schedules
        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        # clip gradients
        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # perform teacher EMA update
        model.update_teacher(mom)

        #########################################################
        # 0. PREVENT NaN propagation BEFORE any all_reduce
        #    CRITICAL: Must synchronize NaN detection across ALL ranks
        #    to prevent collective mismatch (some ranks skip, others don't)
        #########################################################

        def _finite_or_log(name, tensor):
            if not torch.isfinite(tensor).all():
                logger.warning(f"[NaNGuard] Non-finite loss detected in {name}.")
                return False
            return True

        # Check for NaN/Inf locally on this rank
        has_nan_local = not all(_finite_or_log(k, v) for k, v in loss_dict.items())

        # Synchronize: if ANY rank has NaN, ALL ranks must skip together
        if distributed.get_global_size() > 1:
            has_nan_tensor = torch.tensor([1.0 if has_nan_local else 0.0], device='cuda')
            torch.distributed.all_reduce(has_nan_tensor, op=torch.distributed.ReduceOp.MAX)
            has_nan_any_rank = has_nan_tensor.item() > 0
        else:
            has_nan_any_rank = has_nan_local

        if has_nan_any_rank:
            if has_nan_local:
                logger.warning("[NaNGuard] Skipping batch due to NaN/Inf in loss dictionary.")
            else:
                logger.warning("[NaNGuard] Skipping batch because another rank detected NaN/Inf.")
            continue
        #########################################################

        # logging DDP/FSDP synchronization
        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        # if math.isnan(sum(loss_dict_reduced.values())):
        #     logger.info("NaN detected")
        #     # raise AssertionError

        #########################################################
        # 3. FINAL NaN CHECK (post-reduce)
        #########################################################
        if math.isnan(sum(loss_dict_reduced.values())):
            logger.warning("[NaNGuard] NaN detected AFTER reduction. Skipping batch.")
            continue

        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        # checkpointing and testing
        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        iteration = iteration + 1

        st = time.time()
    
    # Save final checkpoint
    logger.info("Training completed. Saving final checkpoint...")
    checkpointer.save("model_final", iteration=iteration)
    
    metric_logger.synchronize_between_processes()
    logger.info(f"Training finished at iteration {iteration}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup_3d(args)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    result = do_train(cfg, model, args, resume=not args.no_resume)
    # Ensure we exit with proper code for SLURM job chaining:
    # - If do_train returns an int, treat it as an explicit exit code.
    # - Otherwise (e.g. dict of metrics), treat that as success (=0).
    if isinstance(result, int):
        sys.exit(result)
    else:
        sys.exit(0)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)


"""
export PYTHONPATH=$(pwd)

torchrun --nproc_per_node=4 --master_port=29501 \
  dinov2/train/train3d.py \
  --config-file 'dinov2/configs/ssl3d_default_config.yaml' \
    --output-dir './debug_out' \
    --cache-dir './debug_cache'

  
If using legacy launcher:

python -m torch.distributed.launch \
  --nproc_per_node=4 --master_port=29501 \
  dinov2/train/train3d.py \
  --config-file 'dinov2/configs/ssl3d_default_config.yaml' \
    --output-dir './debug_out' \
    --cache-dir './debug_cache'


PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node 4 --master_port 29501 dinov2/train/train3d.py \
  --config-file 'dinov2/configs/ssl3d_default_config.yaml' \
    --output-dir './debug_out' \
    --cache-dir './debug_cache' 
"""