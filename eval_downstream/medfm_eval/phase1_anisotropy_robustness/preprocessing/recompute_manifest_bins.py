#!/usr/bin/env python
"""
Recompute anisotropy bins in existing manifests using updated bin thresholds.

This script updates the anisotropy_bin field without regenerating the entire manifest,
which is useful when only the bin boundaries have changed.

Usage:
    python scripts/recompute_manifest_bins.py --input manifest.json --output manifest_new_bins.json
    
Or update in place:
    python scripts/recompute_manifest_bins.py --input manifest.json --inplace
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # phase1_anisotropy_robustness/
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    get_anisotropy_bin,
    compute_anisotropy_ratio,
    get_bin_configs,
    get_bin_description_map,
)


def recompute_bins(
    input_path: Path,
    output_path: Path = None,
    inplace: bool = False,
    scheme: str = "original",
):
    """
    Recompute anisotropy bins in a manifest file.
    
    Args:
        input_path: Path to input manifest JSON
        output_path: Path to output manifest JSON (optional)
        inplace: If True, update input file directly
    """
    print(f"Loading manifest: {input_path}")
    with open(input_path) as f:
        data = json.load(f)
    
    volumes = data.get("volumes", [])
    print(f"Found {len(volumes)} volumes")
    
    bin_configs = get_bin_configs(scheme)
    bin_descriptions = get_bin_description_map(scheme)

    # Show current bin boundaries
    print(f"\nNew bin boundaries (scheme={scheme}):")
    for b in bin_configs:
        print(f"  Bin {b.bin_id} ({b.name}): {bin_descriptions[b.bin_id]}")
    
    # Count old bins
    old_bins = Counter(v.get("anisotropy_bin", -1) for v in volumes)
    print(f"\nOld bin distribution: {dict(sorted(old_bins.items()))}")
    
    # Recompute bins
    changes = 0
    new_bins = []
    
    for v in volumes:
        ratio = v.get("anisotropy_ratio")
        
        # If ratio not stored, compute from spacing
        if ratio is None:
            spacing = v.get("spacing", [1.0, 1.0, 1.0])
            ratio = compute_anisotropy_ratio(spacing)
            v["anisotropy_ratio"] = round(ratio, 4)
        spacing = tuple(v.get("spacing", [1.0, 1.0, 1.0]))
        
        old_bin = v.get("anisotropy_bin", -1)
        if scheme != "original" and "anisotropy_bin_original" not in v:
            v["anisotropy_bin_original"] = old_bin
        new_bin = get_anisotropy_bin(ratio, spacing=spacing, scheme=scheme)
        
        if old_bin != new_bin:
            changes += 1
            v["anisotropy_bin"] = new_bin
        
        new_bins.append(new_bin)
    
    # Count new bins
    new_bin_counts = Counter(new_bins)
    print(f"New bin distribution: {dict(sorted(new_bin_counts.items()))}")
    print(f"\nChanged {changes} / {len(volumes)} bin assignments ({100*changes/len(volumes):.1f}%)")
    
    # Determine output path
    if inplace:
        output_path = input_path
    elif output_path is None:
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_rebinned.json"
    
    # Save
    data["binning_scheme"] = scheme
    print(f"\nSaving to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("Done!")
    return data


def main():
    parser = argparse.ArgumentParser(description="Recompute anisotropy bins in manifest")
    parser.add_argument("--input", type=Path, required=True, help="Input manifest JSON")
    parser.add_argument("--output", type=Path, help="Output manifest JSON (default: input_rebinned.json)")
    parser.add_argument("--inplace", action="store_true", help="Update input file in place")
    parser.add_argument(
        "--scheme",
        type=str,
        default="original",
        choices=["original", "coarse_ratio_thickness"],
        help="Binning scheme to apply",
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if args.inplace and args.output:
        print("Error: Cannot use both --inplace and --output")
        sys.exit(1)
    
    recompute_bins(args.input, args.output, args.inplace, args.scheme)


if __name__ == "__main__":
    main()
