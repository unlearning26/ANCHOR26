# Phase 2: Cross-Modality Anatomical Alignment Evaluation

"""
Phase 2 evaluates whether frozen representations align the same anatomy across
modalities, with CT-to-MR as the current primary benchmark surface.

The initial scaffold intentionally reuses the Phase 1 artifact grammar:
- manifest-driven cohort definition
- phase-scoped caches and outputs
- per-checkpoint and per-feature-family reporting

The current v0 implementation focuses on:
- manifest validation
- cohort construction
- organ-aware CLS extraction from mask-defined ROIs
- optional NPZ-based alignment analysis over precomputed features
"""

__version__ = "0.1.0"