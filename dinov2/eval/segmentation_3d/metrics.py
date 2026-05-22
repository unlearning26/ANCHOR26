# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric, SurfaceDiceMetric
from monai.transforms import AsDiscrete, Compose, Activations
from monai.data import decollate_batch


# Default NSD tolerance in mm (surface points within this distance are considered matching)
NSD_TOLERANCE_MM = 2.0


class BTCVMetrics:

    def __init__(self):
        self.post_label = AsDiscrete(to_onehot=14)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=14)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)

    def __call__(self, pred, target):
        target_list = decollate_batch(target)
        target_list = [self.post_label(t) for t in target_list]
        pred_list = decollate_batch(pred)
        pred_list = [self.post_pred(p) for p in pred_list]

        self.dice_metric(y_pred=pred_list, y=target_list)
        self.dice_metric_batch(y_pred=pred_list, y=target_list)

        avg_dice = self.dice_metric.aggregate().item()
        class_dice = self.dice_metric_batch.aggregate()
        class_dice = [d.item() for d in class_dice]

        self.dice_metric.reset()
        self.dice_metric_batch.reset()

        return avg_dice, class_dice


class BraTSMetrics:

    def __init__(self):
        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    def __call__(self, pred, target):
        pred_list = decollate_batch(pred)
        pred_list = [self.post_pred(p) for p in pred_list]

        self.dice_metric(y_pred=pred_list, y=target)
        self.dice_metric_batch(y_pred=pred_list, y=target)

        avg_dice = self.dice_metric.aggregate().item()
        class_dice = self.dice_metric_batch.aggregate()
        metric_tc = class_dice[0].item()
        metric_wt = class_dice[1].item()
        metric_et = class_dice[2].item()

        self.dice_metric.reset()
        self.dice_metric_batch.reset()

        return avg_dice, (metric_tc, metric_wt, metric_et)


class LASEGMetrics(BTCVMetrics):

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=2)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=2)


class ISLES22Metrics:
    """Dice metric for ISLES22 dataset (binary stroke lesion segmentation).
    
    Uses sigmoid activation and threshold at 0.5, similar to BraTS but for single class.
    """

    def __init__(self):
        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")

    def __call__(self, pred, target):
        pred_list = decollate_batch(pred)
        pred_list = [self.post_pred(p) for p in pred_list]

        self.dice_metric(y_pred=pred_list, y=target)
        self.dice_metric_batch(y_pred=pred_list, y=target)

        avg_dice = self.dice_metric.aggregate().item()
        class_dice = self.dice_metric_batch.aggregate()
        # Single class (stroke lesion)
        metric_lesion = class_dice[0].item() if len(class_dice) > 0 else avg_dice

        self.dice_metric.reset()
        self.dice_metric_batch.reset()

        return avg_dice, (metric_lesion,)


class AMOS22Metrics(BTCVMetrics):
    """Dice metric for AMOS22 dataset (16 classes: 15 organs + background).
    
    Similar to BTCV but with 16 classes instead of 14.
    """

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=16)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=16)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)


class KiTS23Metrics(BTCVMetrics):
    """Dice metric for KiTS23 dataset (4 classes: Background, Kidney, Tumor, Cyst).
    
    Similar to BTCV but with 4 classes instead of 14.
    """

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=4)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=4)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)


class LiTSMetrics(BTCVMetrics):
    """Dice metric for LiTS dataset (3 classes: Background, Liver, Tumor)."""

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=3)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=3)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)


class WORDMetrics(BTCVMetrics):
    """Dice metric for WORD dataset (17 classes: Background + 16 abdominal organs).
    
    Similar to BTCV but with 17 classes instead of 14.
    """

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=17)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=17)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)


class TotalSegmenterCTMetrics(BTCVMetrics):
    """Dice metric for TotalSegmenter CT dataset (105 classes: Background + 104 organs).
    
    Similar to BTCV but with 105 classes instead of 14.
    """

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=105)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=105)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)


# ============================================================================
# HD95 Metrics (Hausdorff Distance 95th percentile)
# ============================================================================

class BTCVHD95Metrics:
    """HD95 metric for BTCV dataset (14 classes including background).
    
    Note: Uses include_background=True to ensure class array size matches num_classes.
    Background HD95 is typically very high (entire volume) and may be ignored in analysis.
    """

    def __init__(self):
        self.post_label = AsDiscrete(to_onehot=14)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=14)
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=True, percentile=95, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.hd95_metric_batch = HausdorffDistanceMetric(
            include_background=True, percentile=95, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )

    def __call__(self, pred, target):
        target_list = decollate_batch(target)
        target_list = [self.post_label(t) for t in target_list]
        pred_list = decollate_batch(pred)
        pred_list = [self.post_pred(p) for p in pred_list]

        self.hd95_metric(y_pred=pred_list, y=target_list)
        self.hd95_metric_batch(y_pred=pred_list, y=target_list)

        avg_hd95 = self.hd95_metric.aggregate().item()
        class_hd95 = self.hd95_metric_batch.aggregate()
        class_hd95 = [d.item() if not d.isnan() else float('inf') for d in class_hd95]

        self.hd95_metric.reset()
        self.hd95_metric_batch.reset()

        return avg_hd95, class_hd95


class BraTSHD95Metrics:
    """HD95 metric for BraTS dataset (3 classes: TC, WT, ET).
    
    Note: Uses include_background=True because BraTS is a multi-label task where all 3
    channels (TC, WT, ET) are independent foreground classes, not a multi-class task
    with a background. Setting include_background=False would skip the first channel.
    """

    def __init__(self):
        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=True, percentile=95, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.hd95_metric_batch = HausdorffDistanceMetric(
            include_background=True, percentile=95, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )

    def __call__(self, pred, target):
        pred_list = decollate_batch(pred)
        pred_list = [self.post_pred(p) for p in pred_list]

        self.hd95_metric(y_pred=pred_list, y=target)
        self.hd95_metric_batch(y_pred=pred_list, y=target)

        avg_hd95 = self.hd95_metric.aggregate().item()
        class_hd95 = self.hd95_metric_batch.aggregate()
        metric_tc = class_hd95[0].item() if not class_hd95[0].isnan() else float('inf')
        metric_wt = class_hd95[1].item() if not class_hd95[1].isnan() else float('inf')
        metric_et = class_hd95[2].item() if not class_hd95[2].isnan() else float('inf')

        self.hd95_metric.reset()
        self.hd95_metric_batch.reset()

        return avg_hd95, (metric_tc, metric_wt, metric_et)


class LASEGHD95Metrics(BTCVHD95Metrics):
    """HD95 metric for LA-SEG dataset (2 classes).
    
    Note: Uses include_background=True to ensure class array size matches num_classes=2.
    This enables consistent indexing with Dice metrics.
    """

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=2)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=2)
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=True, percentile=95, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.hd95_metric_batch = HausdorffDistanceMetric(
            include_background=True, percentile=95, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )


class ISLES22HD95Metrics:
    """HD95 metric for ISLES22 dataset (binary stroke lesion segmentation).
    
    Uses sigmoid activation and threshold at 0.5.
    """

    def __init__(self):
        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=False, percentile=95, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.hd95_metric_batch = HausdorffDistanceMetric(
            include_background=False, percentile=95, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )

    def __call__(self, pred, target):
        pred_list = decollate_batch(pred)
        pred_list = [self.post_pred(p) for p in pred_list]

        self.hd95_metric(y_pred=pred_list, y=target)
        self.hd95_metric_batch(y_pred=pred_list, y=target)

        avg_hd95 = self.hd95_metric.aggregate().item()
        class_hd95 = self.hd95_metric_batch.aggregate()
        metric_lesion = class_hd95[0].item() if (len(class_hd95) > 0 and not class_hd95[0].isnan()) else float('inf')

        self.hd95_metric.reset()
        self.hd95_metric_batch.reset()

        return avg_hd95, (metric_lesion,)


class AMOS22HD95Metrics(BTCVHD95Metrics):
    """HD95 metric for AMOS22 dataset (16 classes: 15 organs + background).
    
    Similar to BTCV but with 16 classes instead of 14.
    """

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=16)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=16)
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=True, percentile=95, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.hd95_metric_batch = HausdorffDistanceMetric(
            include_background=True, percentile=95, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )


class KiTS23HD95Metrics(BTCVHD95Metrics):
    """HD95 metric for KiTS23 dataset (4 classes: Background, Kidney, Tumor, Cyst).
    
    Similar to BTCV but with 4 classes instead of 14.
    """

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=4)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=4)
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=True, percentile=95, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.hd95_metric_batch = HausdorffDistanceMetric(
            include_background=True, percentile=95, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )


class LiTSHD95Metrics(BTCVHD95Metrics):
    """HD95 metric for LiTS dataset (3 classes: Background, Liver, Tumor)."""

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=3)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=3)
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=True, percentile=95, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.hd95_metric_batch = HausdorffDistanceMetric(
            include_background=True, percentile=95, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )


class WORDHD95Metrics(BTCVHD95Metrics):
    """HD95 metric for WORD dataset (17 classes: Background + 16 abdominal organs).
    
    Similar to BTCV but with 17 classes instead of 14.
    """

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=17)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=17)
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=True, percentile=95, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.hd95_metric_batch = HausdorffDistanceMetric(
            include_background=True, percentile=95, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )


class TotalSegmenterCTHD95Metrics(BTCVHD95Metrics):
    """HD95 metric for TotalSegmenter CT dataset (105 classes: Background + 104 organs).
    
    Similar to BTCV but with 105 classes instead of 14.
    """

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=105)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=105)
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=True, percentile=95, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.hd95_metric_batch = HausdorffDistanceMetric(
            include_background=True, percentile=95, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )


# ============================================================================
# ASD Metrics (Average Surface Distance)
# ============================================================================

class BTCVASDMetrics:
    """ASD metric for BTCV dataset (14 classes including background).
    
    Note: Uses include_background=True to ensure class array size matches num_classes.
    """

    def __init__(self):
        self.post_label = AsDiscrete(to_onehot=14)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=14)
        self.asd_metric = SurfaceDistanceMetric(
            include_background=True, symmetric=True, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.asd_metric_batch = SurfaceDistanceMetric(
            include_background=True, symmetric=True, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )

    def __call__(self, pred, target):
        target_list = decollate_batch(target)
        target_list = [self.post_label(t) for t in target_list]
        pred_list = decollate_batch(pred)
        pred_list = [self.post_pred(p) for p in pred_list]

        self.asd_metric(y_pred=pred_list, y=target_list)
        self.asd_metric_batch(y_pred=pred_list, y=target_list)

        avg_asd = self.asd_metric.aggregate().item()
        class_asd = self.asd_metric_batch.aggregate()
        class_asd = [d.item() if not d.isnan() else float('inf') for d in class_asd]

        self.asd_metric.reset()
        self.asd_metric_batch.reset()

        return avg_asd, class_asd


class BraTSASDMetrics:
    """ASD metric for BraTS dataset (3 classes: TC, WT, ET).
    
    Note: Uses include_background=True because BraTS is a multi-label task where all 3
    channels (TC, WT, ET) are independent foreground classes, not a multi-class task
    with a background. Setting include_background=False would skip the first channel.
    """

    def __init__(self):
        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.asd_metric = SurfaceDistanceMetric(
            include_background=True, symmetric=True, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.asd_metric_batch = SurfaceDistanceMetric(
            include_background=True, symmetric=True, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )

    def __call__(self, pred, target):
        pred_list = decollate_batch(pred)
        pred_list = [self.post_pred(p) for p in pred_list]

        self.asd_metric(y_pred=pred_list, y=target)
        self.asd_metric_batch(y_pred=pred_list, y=target)

        avg_asd = self.asd_metric.aggregate().item()
        class_asd = self.asd_metric_batch.aggregate()
        metric_tc = class_asd[0].item() if not class_asd[0].isnan() else float('inf')
        metric_wt = class_asd[1].item() if not class_asd[1].isnan() else float('inf')
        metric_et = class_asd[2].item() if not class_asd[2].isnan() else float('inf')

        self.asd_metric.reset()
        self.asd_metric_batch.reset()

        return avg_asd, (metric_tc, metric_wt, metric_et)


class LASEGASDMetrics(BTCVASDMetrics):
    """ASD metric for LA-SEG dataset (2 classes).
    
    Note: Uses include_background=True to ensure class array size matches num_classes=2.
    """

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=2)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=2)
        self.asd_metric = SurfaceDistanceMetric(
            include_background=True, symmetric=True, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.asd_metric_batch = SurfaceDistanceMetric(
            include_background=True, symmetric=True, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )


class ISLES22ASDMetrics:
    """ASD metric for ISLES22 dataset (binary stroke lesion segmentation).
    
    Uses sigmoid activation and threshold at 0.5.
    """

    def __init__(self):
        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.asd_metric = SurfaceDistanceMetric(
            include_background=False, symmetric=True, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.asd_metric_batch = SurfaceDistanceMetric(
            include_background=False, symmetric=True, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )

    def __call__(self, pred, target):
        pred_list = decollate_batch(pred)
        pred_list = [self.post_pred(p) for p in pred_list]

        self.asd_metric(y_pred=pred_list, y=target)
        self.asd_metric_batch(y_pred=pred_list, y=target)

        avg_asd = self.asd_metric.aggregate().item()
        class_asd = self.asd_metric_batch.aggregate()
        metric_lesion = class_asd[0].item() if (len(class_asd) > 0 and not class_asd[0].isnan()) else float('inf')

        self.asd_metric.reset()
        self.asd_metric_batch.reset()

        return avg_asd, (metric_lesion,)


class AMOS22ASDMetrics(BTCVASDMetrics):
    """ASD metric for AMOS22 dataset (16 classes: 15 organs + background).
    
    Similar to BTCV but with 16 classes instead of 14.
    """

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=16)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=16)
        self.asd_metric = SurfaceDistanceMetric(
            include_background=True, symmetric=True, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.asd_metric_batch = SurfaceDistanceMetric(
            include_background=True, symmetric=True, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )


class KiTS23ASDMetrics(BTCVASDMetrics):
    """ASD metric for KiTS23 dataset (4 classes: Background, Kidney, Tumor, Cyst).
    
    Similar to BTCV but with 4 classes instead of 14.
    """

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=4)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=4)
        self.asd_metric = SurfaceDistanceMetric(
            include_background=True, symmetric=True, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.asd_metric_batch = SurfaceDistanceMetric(
            include_background=True, symmetric=True, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )


class LiTSASDMetrics(BTCVASDMetrics):
    """ASD metric for LiTS dataset (3 classes: Background, Liver, Tumor)."""

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=3)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=3)
        self.asd_metric = SurfaceDistanceMetric(
            include_background=True, symmetric=True, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.asd_metric_batch = SurfaceDistanceMetric(
            include_background=True, symmetric=True, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )


class WORDASDMetrics(BTCVASDMetrics):
    """ASD metric for WORD dataset (17 classes: Background + 16 abdominal organs).
    
    Similar to BTCV but with 17 classes instead of 14.
    """

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=17)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=17)
        self.asd_metric = SurfaceDistanceMetric(
            include_background=True, symmetric=True, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.asd_metric_batch = SurfaceDistanceMetric(
            include_background=True, symmetric=True, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )


class TotalSegmenterCTASDMetrics(BTCVASDMetrics):
    """ASD metric for TotalSegmenter CT dataset (105 classes: Background + 104 organs).
    
    Similar to BTCV but with 105 classes instead of 14.
    """

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=105)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=105)
        self.asd_metric = SurfaceDistanceMetric(
            include_background=True, symmetric=True, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.asd_metric_batch = SurfaceDistanceMetric(
            include_background=True, symmetric=True, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )


# ============================================================================
# NSD Metrics (Normalized Surface Dice)
# ============================================================================

class BTCVNSDMetrics:
    """NSD metric for BTCV dataset (14 classes including background).
    
    Normalized Surface Dice measures the fraction of surface points within
    a tolerance distance between prediction and ground truth.
    
    Note: Uses include_background=True to ensure class array size matches num_classes.
    """

    def __init__(self, tolerance_mm=NSD_TOLERANCE_MM):
        self.num_classes = 14
        self.post_label = AsDiscrete(to_onehot=self.num_classes)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=self.num_classes)
        # class_thresholds is a list of tolerances for each class (in mm)
        self.nsd_metric = SurfaceDiceMetric(
            class_thresholds=[tolerance_mm] * self.num_classes,
            include_background=True, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.nsd_metric_batch = SurfaceDiceMetric(
            class_thresholds=[tolerance_mm] * self.num_classes,
            include_background=True, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )

    def __call__(self, pred, target):
        target_list = decollate_batch(target)
        target_list = [self.post_label(t) for t in target_list]
        pred_list = decollate_batch(pred)
        pred_list = [self.post_pred(p) for p in pred_list]

        self.nsd_metric(y_pred=pred_list, y=target_list)
        self.nsd_metric_batch(y_pred=pred_list, y=target_list)

        avg_nsd = self.nsd_metric.aggregate().item()
        class_nsd = self.nsd_metric_batch.aggregate()
        class_nsd = [d.item() if not d.isnan() else 0.0 for d in class_nsd]

        self.nsd_metric.reset()
        self.nsd_metric_batch.reset()

        return avg_nsd, class_nsd


class BraTSNSDMetrics:
    """NSD metric for BraTS dataset (3 classes: TC, WT, ET).
    
    Note: Uses include_background=True because BraTS is a multi-label task where all 3
    channels (TC, WT, ET) are independent foreground classes, not a multi-class task
    with a background. Setting include_background=False would skip the first channel.
    """

    def __init__(self, tolerance_mm=NSD_TOLERANCE_MM):
        self.num_classes = 3
        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.nsd_metric = SurfaceDiceMetric(
            class_thresholds=[tolerance_mm] * self.num_classes,
            include_background=True, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.nsd_metric_batch = SurfaceDiceMetric(
            class_thresholds=[tolerance_mm] * self.num_classes,
            include_background=True, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )

    def __call__(self, pred, target):
        pred_list = decollate_batch(pred)
        pred_list = [self.post_pred(p) for p in pred_list]

        self.nsd_metric(y_pred=pred_list, y=target)
        self.nsd_metric_batch(y_pred=pred_list, y=target)

        avg_nsd = self.nsd_metric.aggregate().item()
        class_nsd = self.nsd_metric_batch.aggregate()
        metric_tc = class_nsd[0].item() if not class_nsd[0].isnan() else 0.0
        metric_wt = class_nsd[1].item() if not class_nsd[1].isnan() else 0.0
        metric_et = class_nsd[2].item() if not class_nsd[2].isnan() else 0.0

        self.nsd_metric.reset()
        self.nsd_metric_batch.reset()

        return avg_nsd, (metric_tc, metric_wt, metric_et)


class LASEGNSDMetrics(BTCVNSDMetrics):
    """NSD metric for LA-SEG dataset (2 classes).
    
    Note: Uses include_background=True to ensure class array size matches num_classes=2.
    """

    def __init__(self, tolerance_mm=NSD_TOLERANCE_MM):
        super().__init__(tolerance_mm)
        self.num_classes = 2
        self.post_label = AsDiscrete(to_onehot=self.num_classes)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=self.num_classes)
        self.nsd_metric = SurfaceDiceMetric(
            class_thresholds=[tolerance_mm] * self.num_classes,
            include_background=True, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.nsd_metric_batch = SurfaceDiceMetric(
            class_thresholds=[tolerance_mm] * self.num_classes,
            include_background=True, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )


class ISLES22NSDMetrics:
    """NSD metric for ISLES22 dataset (binary stroke lesion segmentation).
    
    Uses sigmoid activation and threshold at 0.5.
    """

    def __init__(self, tolerance_mm=NSD_TOLERANCE_MM):
        self.num_classes = 1
        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.nsd_metric = SurfaceDiceMetric(
            class_thresholds=[tolerance_mm] * self.num_classes,
            include_background=False, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.nsd_metric_batch = SurfaceDiceMetric(
            class_thresholds=[tolerance_mm] * self.num_classes,
            include_background=False, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )

    def __call__(self, pred, target):
        pred_list = decollate_batch(pred)
        pred_list = [self.post_pred(p) for p in pred_list]

        self.nsd_metric(y_pred=pred_list, y=target)
        self.nsd_metric_batch(y_pred=pred_list, y=target)

        avg_nsd = self.nsd_metric.aggregate().item()
        class_nsd = self.nsd_metric_batch.aggregate()
        metric_lesion = class_nsd[0].item() if (len(class_nsd) > 0 and not class_nsd[0].isnan()) else 0.0

        self.nsd_metric.reset()
        self.nsd_metric_batch.reset()

        return avg_nsd, (metric_lesion,)


class AMOS22NSDMetrics(BTCVNSDMetrics):
    """NSD metric for AMOS22 dataset (16 classes: 15 organs + background).
    
    Similar to BTCV but with 16 classes instead of 14.
    """

    def __init__(self, tolerance_mm=NSD_TOLERANCE_MM):
        super().__init__(tolerance_mm)
        self.num_classes = 16
        self.post_label = AsDiscrete(to_onehot=self.num_classes)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=self.num_classes)
        self.nsd_metric = SurfaceDiceMetric(
            class_thresholds=[tolerance_mm] * self.num_classes,
            include_background=True, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.nsd_metric_batch = SurfaceDiceMetric(
            class_thresholds=[tolerance_mm] * self.num_classes,
            include_background=True, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )


class KiTS23NSDMetrics(BTCVNSDMetrics):
    """NSD metric for KiTS23 dataset (4 classes: Background, Kidney, Tumor, Cyst).
    
    Similar to BTCV but with 4 classes instead of 14.
    """

    def __init__(self, tolerance_mm=NSD_TOLERANCE_MM):
        super().__init__(tolerance_mm)
        self.num_classes = 4
        self.post_label = AsDiscrete(to_onehot=self.num_classes)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=self.num_classes)
        self.nsd_metric = SurfaceDiceMetric(
            class_thresholds=[tolerance_mm] * self.num_classes,
            include_background=True, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.nsd_metric_batch = SurfaceDiceMetric(
            class_thresholds=[tolerance_mm] * self.num_classes,
            include_background=True, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )


class LiTSNSDMetrics(BTCVNSDMetrics):
    """NSD metric for LiTS dataset (3 classes: Background, Liver, Tumor)."""

    def __init__(self, tolerance_mm=NSD_TOLERANCE_MM):
        super().__init__(tolerance_mm)
        self.num_classes = 3
        self.post_label = AsDiscrete(to_onehot=self.num_classes)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=self.num_classes)
        self.nsd_metric = SurfaceDiceMetric(
            class_thresholds=[tolerance_mm] * self.num_classes,
            include_background=True, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.nsd_metric_batch = SurfaceDiceMetric(
            class_thresholds=[tolerance_mm] * self.num_classes,
            include_background=True, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )


class WORDNSDMetrics(BTCVNSDMetrics):
    """NSD metric for WORD dataset (17 classes: Background + 16 abdominal organs).
    
    Similar to BTCV but with 17 classes instead of 14.
    """

    def __init__(self, tolerance_mm=NSD_TOLERANCE_MM):
        super().__init__(tolerance_mm)
        self.num_classes = 17
        self.post_label = AsDiscrete(to_onehot=self.num_classes)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=self.num_classes)
        self.nsd_metric = SurfaceDiceMetric(
            class_thresholds=[tolerance_mm] * self.num_classes,
            include_background=True, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.nsd_metric_batch = SurfaceDiceMetric(
            class_thresholds=[tolerance_mm] * self.num_classes,
            include_background=True, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )


class TotalSegmenterCTNSDMetrics(BTCVNSDMetrics):
    """NSD metric for TotalSegmenter CT dataset (105 classes: Background + 104 organs).
    
    Similar to BTCV but with 105 classes instead of 14.
    """

    def __init__(self, tolerance_mm=NSD_TOLERANCE_MM):
        super().__init__(tolerance_mm)
        self.num_classes = 105
        self.post_label = AsDiscrete(to_onehot=self.num_classes)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=self.num_classes)
        self.nsd_metric = SurfaceDiceMetric(
            class_thresholds=[tolerance_mm] * self.num_classes,
            include_background=True, reduction="mean", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )
        self.nsd_metric_batch = SurfaceDiceMetric(
            class_thresholds=[tolerance_mm] * self.num_classes,
            include_background=True, reduction="mean_batch", get_not_nans=False,
            distance_metric="euclidean"  # Force CPU-based scipy computation
        )


def get_metric(dataset_name):
    if dataset_name == "BTCV":
        return BTCVMetrics()
    elif dataset_name in ("BraTS", "BraTS-full"):
        return BraTSMetrics()
    elif dataset_name == "LA-SEG":
        return LASEGMetrics()
    elif dataset_name == "TDSC-ABUS":
        return LASEGMetrics()  # same as LA-SEG
    elif dataset_name == "ISLES22":
        return ISLES22Metrics()
    elif dataset_name in ("AMOS22", "AMOS22_CT", "AMOS22_MR"):
        return AMOS22Metrics()
    elif dataset_name == "KiTS23":
        return KiTS23Metrics()
    elif dataset_name == "LiTS":
        return LiTSMetrics()
    elif dataset_name == "WORD":
        return WORDMetrics()
    elif dataset_name == "TotalSegmenterCT":
        return TotalSegmenterCTMetrics()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def get_hd95_metric(dataset_name):
    """Get HD95 (Hausdorff Distance 95th percentile) metric for dataset."""
    if dataset_name == "BTCV":
        return BTCVHD95Metrics()
    elif dataset_name in ("BraTS", "BraTS-full"):
        return BraTSHD95Metrics()
    elif dataset_name == "LA-SEG":
        return LASEGHD95Metrics()
    elif dataset_name == "TDSC-ABUS":
        return LASEGHD95Metrics()  # same as LA-SEG
    elif dataset_name == "ISLES22":
        return ISLES22HD95Metrics()
    elif dataset_name in ("AMOS22", "AMOS22_CT", "AMOS22_MR"):
        return AMOS22HD95Metrics()
    elif dataset_name == "KiTS23":
        return KiTS23HD95Metrics()
    elif dataset_name == "LiTS":
        return LiTSHD95Metrics()
    elif dataset_name == "WORD":
        return WORDHD95Metrics()
    elif dataset_name == "TotalSegmenterCT":
        return TotalSegmenterCTHD95Metrics()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def get_asd_metric(dataset_name):
    """Get ASD (Average Surface Distance) metric for dataset."""
    if dataset_name == "BTCV":
        return BTCVASDMetrics()
    elif dataset_name in ("BraTS", "BraTS-full"):
        return BraTSASDMetrics()
    elif dataset_name == "LA-SEG":
        return LASEGASDMetrics()
    elif dataset_name == "TDSC-ABUS":
        return LASEGASDMetrics()  # same as LA-SEG
    elif dataset_name == "ISLES22":
        return ISLES22ASDMetrics()
    elif dataset_name in ("AMOS22", "AMOS22_CT", "AMOS22_MR"):
        return AMOS22ASDMetrics()
    elif dataset_name == "KiTS23":
        return KiTS23ASDMetrics()
    elif dataset_name == "LiTS":
        return LiTSASDMetrics()
    elif dataset_name == "WORD":
        return WORDASDMetrics()
    elif dataset_name == "TotalSegmenterCT":
        return TotalSegmenterCTASDMetrics()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def get_nsd_metric(dataset_name, tolerance_mm=NSD_TOLERANCE_MM):
    """Get NSD (Normalized Surface Dice) metric for dataset.
    
    Args:
        dataset_name: Name of the dataset
        tolerance_mm: Surface distance tolerance in mm (default: 2.0mm)
    
    Returns:
        NSD metric instance for the specified dataset
    """
    if dataset_name == "BTCV":
        return BTCVNSDMetrics(tolerance_mm)
    elif dataset_name in ("BraTS", "BraTS-full"):
        return BraTSNSDMetrics(tolerance_mm)
    elif dataset_name == "LA-SEG":
        return LASEGNSDMetrics(tolerance_mm)
    elif dataset_name == "TDSC-ABUS":
        return LASEGNSDMetrics(tolerance_mm)  # same as LA-SEG
    elif dataset_name == "ISLES22":
        return ISLES22NSDMetrics(tolerance_mm)
    elif dataset_name in ("AMOS22", "AMOS22_CT", "AMOS22_MR", "AMOS22_MRI"):
        return AMOS22NSDMetrics(tolerance_mm)
    elif dataset_name == "KiTS23":
        return KiTS23NSDMetrics(tolerance_mm)
    elif dataset_name == "LiTS":
        return LiTSNSDMetrics(tolerance_mm)
    elif dataset_name == "WORD":
        return WORDNSDMetrics(tolerance_mm)
    elif dataset_name == "TotalSegmenterCT":
        return TotalSegmenterCTNSDMetrics(tolerance_mm)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
