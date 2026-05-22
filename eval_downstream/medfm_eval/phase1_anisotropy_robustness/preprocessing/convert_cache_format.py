#!/usr/bin/env python
"""
Convert Setting B cache from MetaTensor format to plain tensor format.

This speeds up cache loading from ~0.4s to ~0.05s per file.
Run once after the cache is generated.

Usage:
    python scripts/convert_cache_format.py
"""

import torch
from pathlib import Path
from tqdm import tqdm
import time

CACHE_DIR = Path(__file__).parent.parent / "outputs/setting_b_cache/crop96"


def convert_file(cache_path: Path) -> bool:
    """Convert a single cache file to plain tensor format."""
    try:
        # Try loading with weights_only=True (already converted)
        try:
            data = torch.load(cache_path, map_location="cpu", weights_only=True)
            return False  # Already in new format
        except Exception:
            pass
        
        # Load with weights_only=False (old MetaTensor format)
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        
        # Check if any value is a MetaTensor
        needs_conversion = any(hasattr(v, 'as_tensor') for v in data.values())
        
        if not needs_conversion:
            return False  # Already plain tensors
        
        # Convert MetaTensors to plain tensors
        plain_data = {}
        for k, v in data.items():
            if hasattr(v, 'as_tensor'):
                plain_data[k] = v.as_tensor().clone()
            else:
                plain_data[k] = v.clone() if isinstance(v, torch.Tensor) else v
        
        # Save back
        torch.save(plain_data, cache_path)
        return True
        
    except Exception as e:
        print(f"Error converting {cache_path.name}: {e}")
        return False


def main():
    if not CACHE_DIR.exists():
        print(f"Cache directory not found: {CACHE_DIR}")
        return
    
    cache_files = list(CACHE_DIR.glob("*.pt"))
    print(f"Found {len(cache_files)} cache files")
    
    # Test loading speed before conversion
    print("\nTesting loading speed before conversion...")
    test_files = cache_files[:10]
    start = time.time()
    for f in test_files:
        try:
            data = torch.load(f, map_location="cpu", weights_only=True)
        except:
            data = torch.load(f, map_location="cpu", weights_only=False)
        del data
    before_time = (time.time() - start) / len(test_files)
    print(f"  Average load time: {before_time:.3f}s per file")
    
    # Convert
    print("\nConverting cache files...")
    converted = 0
    for f in tqdm(cache_files, desc="Converting"):
        if convert_file(f):
            converted += 1
    
    print(f"\nConverted {converted}/{len(cache_files)} files")
    
    # Test loading speed after conversion
    print("\nTesting loading speed after conversion...")
    start = time.time()
    for f in test_files:
        data = torch.load(f, map_location="cpu", weights_only=True)
        del data
    after_time = (time.time() - start) / len(test_files)
    print(f"  Average load time: {after_time:.3f}s per file")
    print(f"  Speedup: {before_time/after_time:.1f}x")
    print(f"  Estimated total load time: {len(cache_files) * after_time:.1f}s ({len(cache_files) * after_time / 60:.1f} min)")


if __name__ == "__main__":
    main()
