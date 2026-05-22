# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from enum import Enum
import logging
from typing import Any, Dict, Optional

from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC


logger = logging.getLogger("dinov2")


class MetricType(Enum):
    MEAN_ACCURACY = "mean_accuracy"
    MEAN_PER_CLASS_ACCURACY = "mean_per_class_accuracy"
    PER_CLASS_ACCURACY = "per_class_accuracy"

    @property
    def accuracy_averaging(self):
        return getattr(AccuracyAveraging, self.name, None)

    def __str__(self):
        return self.value


class AccuracyAveraging(Enum):
    MEAN_ACCURACY = "micro"
    MEAN_PER_CLASS_ACCURACY = "macro"
    PER_CLASS_ACCURACY = "none"

    def __str__(self):
        return self.value


def build_metric(metric_type: MetricType, *, num_classes: int, ks: Optional[tuple] = None):
    if metric_type.accuracy_averaging is not None:
        return build_topk_accuracy_metric(
            average_type=metric_type.accuracy_averaging,
            num_classes=num_classes,
            ks=(1, 5) if ks is None else ks,
        )

    raise ValueError(f"Unknown metric type {metric_type}")


def build_topk_accuracy_metric(average_type: AccuracyAveraging, num_classes: int, ks: tuple = (1, 5)):
    metrics: Dict[str, Metric] = {
        f"top-{k}": MulticlassAccuracy(top_k=k, num_classes=int(num_classes), average=average_type.value) for k in ks
    }
    return MetricCollection(metrics)


def build_classification_metrics(num_classes: int, include_auc: bool = True, include_f1: bool = True):
    """
    Build a comprehensive set of classification metrics including accuracy, F1, and AUC.
    
    Args:
        num_classes: Number of classes in the classification task.
        include_auc: Whether to include AUROC metric (requires probabilities/logits).
        include_f1: Whether to include F1 score metric.
    
    Returns:
        MetricCollection with accuracy, optional F1, and optional AUC metrics.
    """
    metrics: Dict[str, Metric] = {
        "top-1": MulticlassAccuracy(top_k=1, num_classes=int(num_classes), average="micro"),
    }
    
    if include_f1:
        # Macro F1 - average F1 across all classes (good for imbalanced datasets)
        metrics["f1_macro"] = MulticlassF1Score(num_classes=int(num_classes), average="macro")
        # Weighted F1 - weighted by class support
        metrics["f1_weighted"] = MulticlassF1Score(num_classes=int(num_classes), average="weighted")
    
    if include_auc:
        # AUROC - macro average across classes
        # Note: MulticlassAUROC expects logits or probabilities, not hard predictions
        metrics["auroc_macro"] = MulticlassAUROC(num_classes=int(num_classes), average="macro")
        # Weighted AUROC
        metrics["auroc_weighted"] = MulticlassAUROC(num_classes=int(num_classes), average="weighted")
    
    return MetricCollection(metrics)