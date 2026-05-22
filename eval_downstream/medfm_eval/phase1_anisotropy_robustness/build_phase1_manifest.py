#!/usr/bin/env python
# build_phase1_manifest.py
# Phase 1: Create manifest for single-dataset evaluation (e.g., CT-only AbdomenAtlas)
#
# This script generates manifests for modality-controlled experiments by
# filtering to a single dataset, eliminating modality/dataset confounding.
#
# Usage:
# cd eval_downstream/medfm_eval/phase1_anisotropy_robustness

# python build_phase1_manifest.py --dataset abdomenatlas --output-suffix abdomenatlas --binning-scheme original
# python build_phase1_manifest.py --dataset abdomenatlas --output-suffix abdomenatlas --binning-scheme coarse_bins
# python build_phase1_manifest.py --dataset totalsegmentermri --output-suffix totalsegmentermri --binning-scheme original


import argparse
from pathlib import Path

from config import (
    RAW_DATASETS,
    DEFAULT_BINNING_SCHEME,
)
from manifest_generation import (
    build_single_dataset_manifest,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build single-dataset manifest for Phase 1 evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python build_phase1_manifest.py --dataset abdomenatlas --output-suffix abdomenatlas --binning-scheme original\n"
            "  python build_phase1_manifest.py --dataset abdomenct1k --output-suffix abdomenct1k --binning-scheme coarse_bins\n"
            "  python build_phase1_manifest.py --dataset totalsegmentermri --output-suffix totalsegmentermri --binning-scheme original"
        ),
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        required=True,
        help=f"Dataset name. Available: {list(RAW_DATASETS.keys())}",
    )
    parser.add_argument(
        "-s", "--output-suffix",
        type=str,
        required=True,
        help="Output suffix (e.g., 'abdomenatlas', 'abdomenct1k')",
    )
    parser.add_argument(
        "--binning-scheme",
        type=str,
        default=DEFAULT_BINNING_SCHEME,
        choices=["original", "coarse_bins"],
        help="Binning scheme / manifest variant to generate (default: original)",
    )
    parser.add_argument(
        "--min-per-bin",
        type=int,
        default=500,
        help="Minimum samples per anisotropy bin (default: 500)",
    )
    parser.add_argument(
        "--max-per-bin",
        type=int,
        default=1000,
        help="Maximum samples per anisotropy bin (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate volume integrity (skip if pre-validated)",
    )
    parser.add_argument(
        "--force-validate",
        action="store_true",
        help="Force validation even for pre-validated datasets",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: PHASE1_MANIFESTS)",
    )
    
    args = parser.parse_args()
    
    build_single_dataset_manifest(
        dataset_name=args.dataset,
        output_suffix=args.output_suffix,
        binning_scheme=args.binning_scheme,
        validate_integrity=args.validate,
        force_validate=args.force_validate,
        min_per_bin=args.min_per_bin,
        max_per_bin=args.max_per_bin,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
