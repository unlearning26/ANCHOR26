# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

from dinov2.data.loaders import make_segmentation_dataset_3d
from dinov2.data import SamplerType, make_data_loader
from dinov2.eval.segmentation_3d.segmentation_heads import UNETRHead, LinearDecoderHead, ViTAdapterUNETRHead
from dinov2.eval.setup import get_args_parser, setup_and_build_model_3d
from dinov2.eval.segmentation_3d.augmentations import make_transforms
from dinov2.eval.segmentation_3d.metrics import get_metric, get_hd95_metric, get_asd_metric, get_nsd_metric

import os
import torch
import json
import warnings
from functools import partial
from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference
from monai.data.utils import list_data_collate
from monai.optimizers import WarmupCosineSchedule

# Suppress MONAI warnings about empty classes in distance metrics
# These are expected when certain organs are not present in a scan
warnings.filterwarnings("ignore", message=".*ground truth of class.*all 0.*")
warnings.filterwarnings("ignore", message=".*prediction of class.*all 0.*")

# PyTorch 2.6+ compatibility: Restore weights_only=False default for MONAI cache
# PyTorch 2.6 changed torch.load() default from weights_only=False to True,
# which breaks MONAI's CacheDataset that stores numpy arrays and MetaTensors.
# This monkey-patch restores backward compatibility.
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    """Wrapper that defaults weights_only=False for PyTorch 2.6+ compatibility."""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load


def add_seg_args(parser):
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Name of finetuning dataset",
    )
    parser.add_argument(
        "--dataset-percent",
        type=int,
        help="Percent of finetuning dataset to use",
        default=100
    )
    parser.add_argument(
        "--base-data-dir",
        type=str,
        help="Base data directory for finetuning dataset",
    )
    parser.add_argument(
        "--segmentation-head",
        type=str,
        help="Segmentation head",
    )
    parser.add_argument(
        "--train-feature-model",
        action="store_true",
        help="Freeze feature model or not",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Total epochs",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        help="Iterations to perform per epoch",
    )
    parser.add_argument(
        "--eval-iters",
        type=int,
        help="Iterations to perform per evaluation",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        help="Warmup iterations",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        help="Image side length",
    )
    parser.add_argument(
        "--resize-scale",
        type=float,
        help="Scale factor for resizing images",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="path to cache directory for monai persistent dataset"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint.pth if it exists"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="Stop training after N eval periods without val dice improvement (default: 15, 0 to disable)"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Skip training and run test evaluation using existing best_model.pth"
    )
    parser.add_argument(
        "--test-overlap",
        type=float,
        default=0.75,
        help="Overlap for sliding window inference during test (default: 0.75). Lower values are faster but may reduce accuracy. Use 0.5 for large volumes."
    )

    return parser


def train_iter(model, batch, optimizer, scheduler, loss_function, scaler):
    x, y = (batch["image"].cuda(), batch["label"].cuda())
    logits = model(x)
    loss = loss_function(logits, y)
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return loss.item()


def val_iter(model, batch, metric, image_size, batch_size, overlap=0.5):
    x, y = (batch["image"].cuda(), batch["label"].cuda())
    logits = sliding_window_inference(x, image_size, batch_size, model, overlap=overlap)

    iter_metric = metric(logits, y)
    return iter_metric


def do_finetune(feature_model, autocast_dtype, args):
    # In --test-only mode, apply safe defaults for params only needed during training
    if args.test_only:
        best_model_path = os.path.join(args.output_dir, "best_model.pth")
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"--test-only requires best_model.pth but not found: {best_model_path}")
        print(f"\n[Test-Only Mode] Will load best model from: {best_model_path}")
        if args.resize_scale is None:
            args.resize_scale = 1.0
        if args.batch_size is None:
            args.batch_size = 2
        if args.num_workers is None:
            args.num_workers = 8

    # get transforms, dataset, dataloaders
    train_transforms, val_transforms, test_transforms = make_transforms(
        args.dataset_name,
        args.image_size,
        args.resize_scale,
        min_int=-1.0
    )
    train_ds, val_ds, test_ds, input_channels, num_classes = make_segmentation_dataset_3d(
        args.dataset_name,
        args.dataset_percent,
        args.base_data_dir,
        train_transforms,
        val_transforms,
        test_transforms,
        args.cache_dir,
        args.batch_size
    )
    if args.test_only:
        train_loader = None
        val_loader = None
    else:
        train_loader = make_data_loader(
            dataset=train_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            seed=0,
            sampler_type=SamplerType.SHARDED_INFINITE,
            drop_last=False,
            persistent_workers=args.num_workers > 0,
            collate_fn=list_data_collate
        )
        val_loader = make_data_loader(
            dataset=val_ds,
            batch_size=1,
            num_workers=args.num_workers,
            shuffle=False,
            seed=0,
            sampler_type=SamplerType.DISTRIBUTED,
            drop_last=False,
            persistent_workers=False,
            collate_fn=list_data_collate
        )
    test_loader = make_data_loader(
        dataset=test_ds,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        seed=0,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        persistent_workers=False,
        collate_fn=list_data_collate
    )

    # get model
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    scaler = torch.cuda.amp.GradScaler()
    if args.segmentation_head == 'UNETR':
        seg_model = UNETRHead(feature_model, input_channels, args.image_size, num_classes, autocast_ctx)
    elif args.segmentation_head == 'Linear':
        seg_model = LinearDecoderHead(feature_model, input_channels, args.image_size, num_classes, autocast_ctx)
    elif args.segmentation_head == 'ViTAdapterUNETR':
        seg_model = ViTAdapterUNETRHead(feature_model, input_channels, args.image_size, num_classes, autocast_ctx)
    else:
        raise ValueError(f"Unknown segmentation head: {args.segmentation_head}")

    if args.train_feature_model:
        if args.segmentation_head == 'ViTAdapterUNETR':
            seg_model.feature_model.vit_model.train()
        else:
            seg_model.feature_model.train()

    else:
        if args.segmentation_head == 'ViTAdapterUNETR':
            seg_model.feature_model.vit_model.eval()
            for param in seg_model.feature_model.vit_model.parameters():
                param.requires_grad = False
        else:
            seg_model.feature_model.eval()
            for param in seg_model.feature_model.parameters():
                param.requires_grad = False

    trainable_params = [name for name, param in seg_model.named_parameters() if param.requires_grad]
    print(f"Trainable parameters: {trainable_params}")

    seg_model.cuda()

    # Dice metric is needed for both training validation and test evaluation
    dice_metric = get_metric(args.dataset_name)

    # Training history (populated during training, empty for test-only)
    iters_list = []
    train_loss_list = []
    val_dice_list = []
    val_per_cls_dice_list = []

    # Skip optimizer, scheduler, loss, and training loop if --test-only mode
    if not args.test_only:
        # get optimizer, scheduler, loss function, metric
        optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, seg_model.parameters()), lr=args.learning_rate)
        max_iter = args.epochs * args.epoch_length
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=args.warmup_iters,
            t_total=max_iter
        )

        if args.dataset_name == 'BTCV' or args.dataset_name == 'LA-SEG' or args.dataset_name == 'TDSC-ABUS':
            loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        elif args.dataset_name in ('BraTS', 'BraTS-full'):
            loss_fn = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
        elif args.dataset_name == 'ISLES22':
            # Binary segmentation with sigmoid activation (ischemic stroke lesion)
            loss_fn = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
        elif args.dataset_name in ('AMOS22', 'AMOS22_CT', 'AMOS22_MR'):
            # Multi-class segmentation (16 classes: 15 foreground + background)
            loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        elif args.dataset_name == 'KiTS23':
            # Multi-class segmentation (4 classes: Background, Kidney, Tumor, Cyst)
            loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        elif args.dataset_name == 'LiTS':
            # Multi-class segmentation (3 classes: Background, Liver, Tumor)
            loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        elif args.dataset_name == 'WORD':
            # Multi-class segmentation (17 classes: Background + 16 abdominal organs)
            loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        elif args.dataset_name == 'TotalSegmenterCT':
            # Multi-class segmentation (105 classes: Background + 104 organs)
            loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        else:
            raise ValueError(f"Unknown dataset name: {args.dataset_name}")

        loss_fn.cuda()

        best_val_dice = -1
        train_loss_sum = 0
        start_iter = 0
        patience_counter = 0  # For early stopping

        # Resume from checkpoint if requested
        checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
        if args.resume and os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cuda')
            seg_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_iter = checkpoint['iteration'] + 1
            best_val_dice = checkpoint['best_val_dice']
            iters_list = checkpoint.get('iters_list', [])
            train_loss_list = checkpoint.get('train_loss_list', [])
            val_dice_list = checkpoint.get('val_dice_list', [])
            val_per_cls_dice_list = checkpoint.get('val_per_cls_dice_list', [])
            patience_counter = checkpoint.get('patience_counter', 0)
            print(f"Resumed from iteration {start_iter}, best_val_dice: {best_val_dice:.4f}, patience: {patience_counter}")
        # enumerate starts at start_iter for resume (no need to iterate through
        # all previous batches - SHARDED_INFINITE sampler yields random samples)
        for it, train_data in enumerate(train_loader, start=start_iter):
            # train for one iteration
            train_loss = train_iter(
                model=seg_model,
                batch=train_data,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_function=loss_fn,
                scaler=scaler
            )
            train_loss_sum += train_loss

            if it % 100 == 0:
                print(f"[Iter {it}], Train loss: {train_loss}", flush=True)

            if it % args.eval_iters == 0:
                # valdation
                total_val_dice = 0
                total_per_cls_val_dice = [0 for _ in range(num_classes)]
                val_steps = 0
                seg_model.eval()
                with torch.no_grad():
                    for val_data in val_loader:
                        val_dice, val_per_cls_dice = val_iter(
                            model=seg_model,
                            batch=val_data,
                            image_size=(args.image_size,) * 3,
                            batch_size=args.batch_size,
                            metric=dice_metric,
                            overlap=0.
                        )

                        total_val_dice += val_dice
                        for i in range(num_classes):
                            total_per_cls_val_dice[i] += val_per_cls_dice[i]
                        val_steps += 1

                avg_val_dice = total_val_dice / val_steps
                avg_per_cls_val_dice = [total_per_cls_val_dice[i] / val_steps for i in range(num_classes)]
                avg_train_loss = train_loss_sum / args.eval_iters

                train_loss_list.append(avg_train_loss)
                val_dice_list.append(avg_val_dice)
                val_per_cls_dice_list.append(avg_per_cls_val_dice)
                iters_list.append(it)
                train_loss_sum = 0

                print(f"[Iter {it}], Train loss: {avg_train_loss}, Val dice: {avg_val_dice}")
                print(f"Val per class dice: {avg_per_cls_val_dice}")

                # save best model and update early stopping counter
                if avg_val_dice > best_val_dice:
                    best_val_dice = avg_val_dice
                    patience_counter = 0  # Reset patience on improvement
                    print(f"Saving best model with val dice: {best_val_dice} on iter: {it}")
                    torch.save(seg_model.state_dict(), args.output_dir + "/best_model.pth")
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{args.early_stopping_patience}")

                # save checkpoint for resume capability
                checkpoint = {
                    'iteration': it,
                    'model_state_dict': seg_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_dice': best_val_dice,
                    'iters_list': iters_list,
                    'train_loss_list': train_loss_list,
                    'val_dice_list': val_dice_list,
                    'val_per_cls_dice_list': val_per_cls_dice_list,
                    'patience_counter': patience_counter,
                }
                torch.save(checkpoint, args.output_dir + "/checkpoint.pth")
                print(f"Checkpoint saved at iteration {it}")

                # Early stopping check
                if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                    print(f"\n{'='*50}")
                    print(f"Early stopping triggered at iteration {it}")
                    print(f"No improvement for {patience_counter} eval periods")
                    print(f"Best val dice: {best_val_dice:.4f}")
                    print(f"{'='*50}\n")
                    break

                # set back to train mode
                seg_model.train()
                if not args.train_feature_model:
                    if args.segmentation_head == 'ViTAdapterUNETR':
                        seg_model.feature_model.vit_model.eval()
                    else:
                        seg_model.feature_model.eval()

            if it >= max_iter:
                break

    # test with Dice, HD95, and ASD metrics
    seg_model.load_state_dict(torch.load(args.output_dir + "/best_model.pth"))
    seg_model.eval()

    # In test-only mode, try to load training history from existing results.json
    if args.test_only:
        existing_results_path = os.path.join(args.output_dir, "results.json")
        if os.path.exists(existing_results_path):
            with open(existing_results_path) as f:
                existing_results = json.load(f)
            iters_list = existing_results.get('iters_list', [])
            train_loss_list = existing_results.get('train_loss_list', [])
            val_dice_list = existing_results.get('val_dice_list', [])
            val_per_cls_dice_list = existing_results.get('val_per_cls_dice_list', [])
            print(f"[Test-Only] Loaded training history from existing results.json")

    # Initialize all metrics
    hd95_metric = get_hd95_metric(args.dataset_name)
    asd_metric = get_asd_metric(args.dataset_name)
    nsd_metric = get_nsd_metric(args.dataset_name)

    total_test_dice = 0
    total_per_cls_test_dice = [0 for _ in range(num_classes)]
    total_test_hd95 = 0
    total_per_cls_test_hd95 = [0 for _ in range(num_classes)]
    total_test_asd = 0
    total_per_cls_test_asd = [0 for _ in range(num_classes)]
    total_test_nsd = 0
    total_per_cls_test_nsd = [0 for _ in range(num_classes)]
    test_steps = 0

    # Track valid samples for HD95/ASD/NSD (some may have NaN due to empty predictions)
    hd95_valid_steps = 0
    hd95_per_cls_valid = [0 for _ in range(num_classes)]
    asd_valid_steps = 0
    asd_per_cls_valid = [0 for _ in range(num_classes)]
    nsd_valid_steps = 0
    nsd_per_cls_valid = [0 for _ in range(num_classes)]

    # Per-scan metrics for statistical analysis (bootstrap CI, Wilcoxon tests)
    scan_metrics = {
        'dice': [],           # Per-scan mean Dice
        'dice_per_cls': [],   # Per-scan per-class Dice (list of lists)
        'hd95': [],           # Per-scan mean HD95
        'hd95_per_cls': [],   # Per-scan per-class HD95
        'asd': [],            # Per-scan mean ASD
        'asd_per_cls': [],    # Per-scan per-class ASD
        'nsd': [],            # Per-scan mean NSD
        'nsd_per_cls': [],    # Per-scan per-class NSD
    }

    print("Running test evaluation with Dice, HD95, ASD, and NSD metrics...", flush=True)
    with torch.no_grad():
        for test_idx, test_data in enumerate(test_loader):
            x, y = (test_data["image"].cuda(), test_data["label"].cuda())
            logits = sliding_window_inference(
                x, (args.image_size,) * 3, args.batch_size, seg_model, overlap=args.test_overlap
            )

            # Dice metric
            test_dice, test_per_cls_dice = dice_metric(logits, y)
            total_test_dice += test_dice
            for i in range(num_classes):
                total_per_cls_test_dice[i] += test_per_cls_dice[i]
            
            # Store per-scan Dice
            scan_metrics['dice'].append(float(test_dice))
            scan_metrics['dice_per_cls'].append([float(d) for d in test_per_cls_dice])

            # Move to CPU for surface distance metrics (HD95, ASD, NSD)
            # MONAI's cucim/cupy GPU acceleration has a JIT compilation bug
            logits_cpu = logits.cpu()
            y_cpu = y.cpu()

            # HD95 metric
            scan_hd95 = float('nan')
            scan_hd95_per_cls = [float('nan')] * num_classes
            try:
                test_hd95, test_per_cls_hd95 = hd95_metric(logits_cpu, y_cpu)
                scan_hd95 = float(test_hd95)
                # Check for both NaN and inf (empty classes produce inf)
                if not (test_hd95 != test_hd95) and test_hd95 != float('inf'):
                    total_test_hd95 += test_hd95
                    hd95_valid_steps += 1
                for i in range(num_classes):
                    val = test_per_cls_hd95[i] if isinstance(test_per_cls_hd95, (list, tuple)) else test_per_cls_hd95
                    scan_hd95_per_cls[i] = float(val)
                    if val != float('inf') and not (val != val):  # Not inf and not NaN
                        total_per_cls_test_hd95[i] += val
                        hd95_per_cls_valid[i] += 1
            except Exception as e:
                print(f"HD95 computation failed for sample {test_idx}: {e}")
            
            # Store per-scan HD95
            scan_metrics['hd95'].append(scan_hd95)
            scan_metrics['hd95_per_cls'].append(scan_hd95_per_cls)

            # ASD metric
            scan_asd = float('nan')
            scan_asd_per_cls = [float('nan')] * num_classes
            try:
                test_asd, test_per_cls_asd = asd_metric(logits_cpu, y_cpu)
                scan_asd = float(test_asd)
                # Check for both NaN and inf (empty classes produce inf)
                if not (test_asd != test_asd) and test_asd != float('inf'):
                    total_test_asd += test_asd
                    asd_valid_steps += 1
                for i in range(num_classes):
                    val = test_per_cls_asd[i] if isinstance(test_per_cls_asd, (list, tuple)) else test_per_cls_asd
                    scan_asd_per_cls[i] = float(val)
                    if val != float('inf') and not (val != val):  # Not inf and not NaN
                        total_per_cls_test_asd[i] += val
                        asd_per_cls_valid[i] += 1
            except Exception as e:
                print(f"ASD computation failed for sample {test_idx}: {e}")
            
            # Store per-scan ASD
            scan_metrics['asd'].append(scan_asd)
            scan_metrics['asd_per_cls'].append(scan_asd_per_cls)

            # NSD metric (Normalized Surface Dice)
            scan_nsd = float('nan')
            scan_nsd_per_cls = [float('nan')] * num_classes
            try:
                test_nsd, test_per_cls_nsd = nsd_metric(logits_cpu, y_cpu)
                scan_nsd = float(test_nsd)
                # Check for both NaN and 0.0 (empty predictions may give 0)
                if not (test_nsd != test_nsd):  # Not NaN
                    total_test_nsd += test_nsd
                    nsd_valid_steps += 1
                for i in range(num_classes):
                    val = test_per_cls_nsd[i] if isinstance(test_per_cls_nsd, (list, tuple)) else test_per_cls_nsd
                    scan_nsd_per_cls[i] = float(val)
                    if not (val != val):  # Not NaN
                        total_per_cls_test_nsd[i] += val
                        nsd_per_cls_valid[i] += 1
            except Exception as e:
                print(f"NSD computation failed for sample {test_idx}: {e}")
            
            # Store per-scan NSD
            scan_metrics['nsd'].append(scan_nsd)
            scan_metrics['nsd_per_cls'].append(scan_nsd_per_cls)

            test_steps += 1
            if (test_idx + 1) % 10 == 0 or test_idx == 0:
                print(f"Test progress: {test_idx + 1}/{len(test_loader)}", flush=True)

    # Compute averages
    avg_test_dice = total_test_dice / test_steps
    avg_per_cls_test_dice = [total_per_cls_test_dice[i] / test_steps for i in range(num_classes)]

    avg_test_hd95 = total_test_hd95 / hd95_valid_steps if hd95_valid_steps > 0 else float('inf')
    avg_per_cls_test_hd95 = [
        total_per_cls_test_hd95[i] / hd95_per_cls_valid[i] if hd95_per_cls_valid[i] > 0 else float('inf')
        for i in range(num_classes)
    ]

    avg_test_asd = total_test_asd / asd_valid_steps if asd_valid_steps > 0 else float('inf')
    avg_per_cls_test_asd = [
        total_per_cls_test_asd[i] / asd_per_cls_valid[i] if asd_per_cls_valid[i] > 0 else float('inf')
        for i in range(num_classes)
    ]

    avg_test_nsd = total_test_nsd / nsd_valid_steps if nsd_valid_steps > 0 else 0.0
    avg_per_cls_test_nsd = [
        total_per_cls_test_nsd[i] / nsd_per_cls_valid[i] if nsd_per_cls_valid[i] > 0 else 0.0
        for i in range(num_classes)
    ]

    print(f"\n{'='*50}")
    print(f"Test Results:")
    print(f"{'='*50}")
    print(f"Test Dice: {avg_test_dice:.4f}")
    print(f"Test per class Dice: {[f'{d:.4f}' for d in avg_per_cls_test_dice]}")
    print(f"Test HD95: {avg_test_hd95:.4f} (valid samples: {hd95_valid_steps}/{test_steps})")
    print(f"Test per class HD95: {[f'{d:.4f}' for d in avg_per_cls_test_hd95]}")
    print(f"Test ASD: {avg_test_asd:.4f} (valid samples: {asd_valid_steps}/{test_steps})")
    print(f"Test per class ASD: {[f'{d:.4f}' for d in avg_per_cls_test_asd]}")
    print(f"Test NSD: {avg_test_nsd:.4f} (valid samples: {nsd_valid_steps}/{test_steps})")
    print(f"Test per class NSD: {[f'{d:.4f}' for d in avg_per_cls_test_nsd]}")
    print(f"{'='*50}\n")

    with open(f'{args.output_dir}/results.json', 'w') as fp:
        json.dump({
            'iters_list': iters_list,
            'train_loss_list': train_loss_list,
            'val_dice_list': val_dice_list,
            'val_per_cls_dice_list': val_per_cls_dice_list,
            # Aggregate test metrics
            'test_dice': avg_test_dice,
            'test_per_cls_dice': avg_per_cls_test_dice,
            'test_hd95': avg_test_hd95,
            'test_per_cls_hd95': avg_per_cls_test_hd95,
            'test_asd': avg_test_asd,
            'test_per_cls_asd': avg_per_cls_test_asd,
            'test_nsd': avg_test_nsd,
            'test_per_cls_nsd': avg_per_cls_test_nsd,
            # Per-scan metrics for statistical analysis (bootstrap CI, Wilcoxon tests)
            'scan_metrics': scan_metrics,
            'n_test_scans': test_steps,
        }, fp, indent=2)


def main(args):
    feature_model, autocast_dtype = setup_and_build_model_3d(args)
    do_finetune(feature_model, autocast_dtype, args)


if __name__ == "__main__":
    args = add_seg_args(get_args_parser(add_help=True)).parse_args()
    main(args)
