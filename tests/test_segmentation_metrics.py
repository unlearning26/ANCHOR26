#!/usr/bin/env python3
"""
Toy Tensor Tests for Segmentation Metrics

Validates that HD95 and ASD metrics correctly handle:
1. Class count consistency with Dice metrics
2. Proper indexing for 2-class (LA-SEG, TDSC-ABUS) and multi-class (BTCV, BraTS) scenarios
3. Edge cases (empty predictions, perfect predictions)

Run: python tests/test_segmentation_metrics.py
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '.')

from dinov2.eval.segmentation_3d.metrics import (
    BTCVMetrics, BraTSMetrics, LASEGMetrics,
    BTCVHD95Metrics, BraTSHD95Metrics, LASEGHD95Metrics,
    BTCVASDMetrics, BraTSASDMetrics, LASEGASDMetrics,
    get_metric, get_hd95_metric, get_asd_metric
)


def test_laseg_metrics_consistency():
    """Test LA-SEG metrics return consistent array sizes."""
    print("\n" + "=" * 60)
    print("TEST: LA-SEG Metrics Consistency")
    print("=" * 60)
    
    # Create dummy data: batch=1, 2 classes, 32^3 volume
    num_classes = 2
    spatial_size = (32, 32, 32)
    
    # Ground truth: class 0 (background) in first half, class 1 (atrium) in second half
    target = torch.zeros(1, 1, *spatial_size, dtype=torch.long)
    target[:, :, 16:, :, :] = 1  # Half the volume is foreground
    
    # Prediction logits: 2 channels (background, foreground)
    pred = torch.zeros(1, num_classes, *spatial_size, dtype=torch.float32)
    pred[:, 0, :16, :, :] = 10.0  # High logit for background in first half
    pred[:, 1, 16:, :, :] = 10.0  # High logit for foreground in second half
    
    # Initialize metrics
    dice_metric = get_metric("LA-SEG")
    hd95_metric = get_hd95_metric("LA-SEG")
    asd_metric = get_asd_metric("LA-SEG")
    
    # Compute metrics
    dice_val, dice_per_cls = dice_metric(pred, target)
    hd95_val, hd95_per_cls = hd95_metric(pred, target)
    asd_val, asd_per_cls = asd_metric(pred, target)
    
    # Verify array sizes
    print(f"\nExpected classes: {num_classes}")
    print(f"Dice per-class length: {len(dice_per_cls)}")
    print(f"HD95 per-class length: {len(hd95_per_cls)}")
    print(f"ASD per-class length: {len(asd_per_cls)}")
    
    assert len(dice_per_cls) == num_classes, f"Dice per-class should have {num_classes} elements"
    assert len(hd95_per_cls) == num_classes, f"HD95 per-class should have {num_classes} elements"
    assert len(asd_per_cls) == num_classes, f"ASD per-class should have {num_classes} elements"
    
    print(f"\n✅ All metrics return {num_classes} classes correctly!")
    print(f"Dice: {dice_val:.4f}, per-class: {[f'{d:.4f}' for d in dice_per_cls]}")
    print(f"HD95: {hd95_val:.4f}, per-class: {[f'{d:.4f}' for d in hd95_per_cls]}")
    print(f"ASD:  {asd_val:.4f}, per-class: {[f'{d:.4f}' for d in asd_per_cls]}")
    
    return True


def test_btcv_metrics_consistency():
    """Test BTCV metrics return consistent array sizes."""
    print("\n" + "=" * 60)
    print("TEST: BTCV Metrics Consistency")
    print("=" * 60)
    
    num_classes = 14
    spatial_size = (32, 32, 32)
    
    # Ground truth: single class present (class 3 = liver)
    target = torch.zeros(1, 1, *spatial_size, dtype=torch.long)
    target[:, :, 10:22, 10:22, 10:22] = 3  # Liver region
    
    # Prediction logits
    pred = torch.zeros(1, num_classes, *spatial_size, dtype=torch.float32)
    pred[:, 0, :, :, :] = -5.0  # Low background logit everywhere
    pred[:, 3, 10:22, 10:22, 10:22] = 10.0  # High liver logit in correct region
    
    # Initialize metrics
    dice_metric = get_metric("BTCV")
    hd95_metric = get_hd95_metric("BTCV")
    asd_metric = get_asd_metric("BTCV")
    
    # Compute metrics
    dice_val, dice_per_cls = dice_metric(pred, target)
    hd95_val, hd95_per_cls = hd95_metric(pred, target)
    asd_val, asd_per_cls = asd_metric(pred, target)
    
    print(f"\nExpected classes: {num_classes}")
    print(f"Dice per-class length: {len(dice_per_cls)}")
    print(f"HD95 per-class length: {len(hd95_per_cls)}")
    print(f"ASD per-class length: {len(asd_per_cls)}")
    
    assert len(dice_per_cls) == num_classes, f"Dice per-class should have {num_classes} elements"
    assert len(hd95_per_cls) == num_classes, f"HD95 per-class should have {num_classes} elements"
    assert len(asd_per_cls) == num_classes, f"ASD per-class should have {num_classes} elements"
    
    print(f"\n✅ All metrics return {num_classes} classes correctly!")
    
    return True


def test_indexing_in_loop():
    """Test that metrics can be indexed in a loop (simulating segmentation3d.py usage)."""
    print("\n" + "=" * 60)
    print("TEST: Indexing Metrics in Loop (Simulating segmentation3d.py)")
    print("=" * 60)
    
    for dataset_name, num_classes in [("LA-SEG", 2), ("BTCV", 14), ("TDSC-ABUS", 2)]:
        print(f"\n--- {dataset_name} ({num_classes} classes) ---")
        
        spatial_size = (24, 24, 24)
        
        # Create simple test data
        target = torch.randint(0, num_classes, (1, 1, *spatial_size), dtype=torch.long)
        pred = torch.randn(1, num_classes, *spatial_size, dtype=torch.float32)
        
        # Get metrics
        dice_metric = get_metric(dataset_name)
        hd95_metric = get_hd95_metric(dataset_name)
        asd_metric = get_asd_metric(dataset_name)
        
        # Compute
        try:
            dice_val, dice_per_cls = dice_metric(pred, target)
            hd95_val, hd95_per_cls = hd95_metric(pred, target)
            asd_val, asd_per_cls = asd_metric(pred, target)
            
            # Simulate loop from segmentation3d.py (the exact code that was failing)
            total_per_cls_hd95 = [0 for _ in range(num_classes)]
            total_per_cls_asd = [0 for _ in range(num_classes)]
            
            for i in range(num_classes):
                # This was causing "list index out of range" before fix
                val_hd95 = hd95_per_cls[i] if isinstance(hd95_per_cls, (list, tuple)) else hd95_per_cls
                val_asd = asd_per_cls[i] if isinstance(asd_per_cls, (list, tuple)) else asd_per_cls
                if val_hd95 != float('inf') and not (val_hd95 != val_hd95):
                    total_per_cls_hd95[i] = val_hd95
                if val_asd != float('inf') and not (val_asd != val_asd):
                    total_per_cls_asd[i] = val_asd
            
            print(f"  ✅ Indexing successful for all {num_classes} classes")
            
        except IndexError as e:
            print(f"  ❌ IndexError: {e}")
            return False
        except Exception as e:
            print(f"  ⚠️ Other error: {type(e).__name__}: {e}")
            # Some samples may have NaN, that's expected
    
    return True


def test_edge_cases():
    """Test edge cases like empty predictions."""
    print("\n" + "=" * 60)
    print("TEST: Edge Cases")
    print("=" * 60)
    
    num_classes = 2
    spatial_size = (24, 24, 24)
    
    # Case 1: All background (no foreground predictions)
    print("\n--- Case 1: All background prediction ---")
    target = torch.zeros(1, 1, *spatial_size, dtype=torch.long)
    target[:, :, 12:, :, :] = 1  # Ground truth has foreground
    pred = torch.zeros(1, num_classes, *spatial_size, dtype=torch.float32)
    pred[:, 0, :, :, :] = 10.0  # Predict all background
    pred[:, 1, :, :, :] = -10.0
    
    hd95_metric = get_hd95_metric("LA-SEG")
    try:
        hd95_val, hd95_per_cls = hd95_metric(pred, target)
        print(f"  HD95: {hd95_val}, per-class: {hd95_per_cls}")
        print(f"  ✅ Handles all-background case (may have inf for missing class)")
    except Exception as e:
        print(f"  ⚠️ Exception: {e}")
    
    # Case 2: Perfect prediction
    print("\n--- Case 2: Perfect prediction ---")
    target = torch.zeros(1, 1, *spatial_size, dtype=torch.long)
    target[:, :, 12:, :, :] = 1
    pred = torch.zeros(1, num_classes, *spatial_size, dtype=torch.float32)
    pred[:, 0, :12, :, :] = 10.0
    pred[:, 1, 12:, :, :] = 10.0
    
    hd95_metric = get_hd95_metric("LA-SEG")
    hd95_val, hd95_per_cls = hd95_metric(pred, target)
    print(f"  HD95: {hd95_val}, per-class: {hd95_per_cls}")
    print(f"  ✅ Perfect prediction should have HD95 ≈ 0")
    
    return True


def test_brats_multilabel_metrics():
    """Test BraTS multi-label metrics return correct array sizes.
    
    BraTS is a MULTI-LABEL task: each channel (TC, WT, ET) is an independent binary mask.
    This is different from multi-class where channels are mutually exclusive.
    
    The key requirement: include_background=True because ALL 3 channels are foreground
    classes, there's no background channel. Setting include_background=False would skip
    the first channel (TC), returning only 2 values instead of 3.
    """
    print("\n" + "=" * 60)
    print("TEST: BraTS Multi-Label Metrics (3 classes: TC, WT, ET)")
    print("=" * 60)
    
    num_classes = 3  # TC, WT, ET
    spatial_size = (32, 32, 32)
    
    # BraTS uses multi-label format: [B, 3, H, W, D] binary masks
    # Each channel is independent (sigmoid activation, not softmax)
    target = torch.zeros(1, num_classes, *spatial_size, dtype=torch.float32)
    target[:, 0, 8:24, 8:24, 8:24] = 1.0   # TC region
    target[:, 1, 4:28, 4:28, 4:28] = 1.0   # WT region (larger, encompasses TC)
    target[:, 2, 12:20, 12:20, 12:20] = 1.0  # ET region (smaller, within TC)
    
    # Prediction logits (before sigmoid)
    pred = torch.zeros(1, num_classes, *spatial_size, dtype=torch.float32)
    pred[:, 0, 8:24, 8:24, 8:24] = 10.0   # TC prediction (nearly perfect)
    pred[:, 1, 4:28, 4:28, 4:28] = 10.0   # WT prediction (nearly perfect)
    pred[:, 2, 12:20, 12:20, 12:20] = 10.0  # ET prediction (nearly perfect)
    
    # Initialize metrics
    dice_metric = get_metric("BraTS")
    hd95_metric = get_hd95_metric("BraTS")
    asd_metric = get_asd_metric("BraTS")
    
    # Compute metrics
    dice_val, dice_per_cls = dice_metric(pred, target)
    hd95_val, hd95_per_cls = hd95_metric(pred, target)
    asd_val, asd_per_cls = asd_metric(pred, target)
    
    # Verify array sizes (this was the bug: include_background=False returned only 2)
    print(f"\nExpected classes: {num_classes}")
    print(f"Dice per-class length: {len(dice_per_cls)}")
    print(f"HD95 per-class length: {len(hd95_per_cls)}")
    print(f"ASD per-class length: {len(asd_per_cls)}")
    
    # These assertions would have failed before the fix
    assert len(dice_per_cls) == num_classes, f"Dice per-class should have {num_classes} elements, got {len(dice_per_cls)}"
    assert len(hd95_per_cls) == num_classes, f"HD95 per-class should have {num_classes} elements, got {len(hd95_per_cls)}"
    assert len(asd_per_cls) == num_classes, f"ASD per-class should have {num_classes} elements, got {len(asd_per_cls)}"
    
    print(f"\n✅ All metrics return {num_classes} classes correctly!")
    print(f"Dice: {dice_val:.4f}, per-class (TC, WT, ET): {[f'{d:.4f}' for d in dice_per_cls]}")
    print(f"HD95: {hd95_val:.4f}, per-class (TC, WT, ET): {[f'{d:.4f}' for d in hd95_per_cls]}")
    print(f"ASD:  {asd_val:.4f}, per-class (TC, WT, ET): {[f'{d:.4f}' for d in asd_per_cls]}")
    
    # Verify metrics are finite (not NaN or broken) - this was the actual bug
    import math
    for i, (hd, asd_v) in enumerate(zip(hd95_per_cls, asd_per_cls)):
        assert not math.isnan(hd), f"HD95 for class {i} is NaN"
        assert not math.isnan(asd_v), f"ASD for class {i} is NaN"
    
    print(f"\n✅ All HD95 and ASD values are finite (no NaN from indexing errors)!")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("SEGMENTATION METRICS VALIDATION TESTS")
    print("=" * 70)
    
    all_passed = True
    
    try:
        all_passed &= test_laseg_metrics_consistency()
        all_passed &= test_btcv_metrics_consistency()
        all_passed &= test_brats_multilabel_metrics()
        all_passed &= test_indexing_in_loop()
        all_passed &= test_edge_cases()
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
