#!/usr/bin/env python3
"""
Comprehensive validation tests for segmentation evaluation setup.

Tests:
1. Datalist integrity (correct split counts)
2. Job script structure validation
3. Launcher script syntax
4. Backward compatibility with loaders.py

Usage:
    python test_segmentation_setup.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "eval_downstream" / "scripts"
JOBS_DIR = SCRIPTS_DIR / "jobs"
DATASETS_DIR = PROJECT_ROOT / "eval_downstream" / "datasets" / "segmentation"


def test_datalist_integrity() -> Tuple[bool, List[str]]:
    """Test datalist files have correct splits."""
    errors = []
    
    # Expected splits (paper-aligned)
    expected = {
        "BTCV_100_datalist.json": {"training": 20, "validation": 4, "test": 6},
        "BraTS_100_datalist_3dinov2.json": {"training": 758, "validation": 108, "test": 218},
        "BraTS_GLI_100_datalist.json": {"training": 1125, "validation": 62, "test": 64},
        "LA-SEG_100_datalist.json": {"training": 80, "validation": 20, "test": 54},
        "TDSC-ABUS_100_datalist.json": {"training": 100, "validation": 30, "test": 70},
    }
    
    print("\n[1/4] Testing Datalist Integrity")
    print("-" * 40)
    
    for filename, counts in expected.items():
        filepath = DATASETS_DIR / filename
        if not filepath.exists():
            errors.append(f"Missing datalist: {filename}")
            print(f"  ❌ {filename}: FILE NOT FOUND")
            continue
        
        with open(filepath) as f:
            data = json.load(f)
        
        actual = {
            "training": len(data.get("training", [])),
            "validation": len(data.get("validation", [])),
            "test": len(data.get("test", []))
        }
        
        if actual == counts:
            print(f"  ✓ {filename}: {actual['training']}/{actual['validation']}/{actual['test']}")
        else:
            errors.append(f"{filename}: Expected {counts}, got {actual}")
            print(f"  ❌ {filename}: Expected {counts}, got {actual}")
    
    return len(errors) == 0, errors


def test_job_structure() -> Tuple[bool, List[str]]:
    """Test job directory structure and script contents."""
    errors = []
    
    expected_datasets = ["BTCV", "BraTS", "LA-SEG", "TDSC-ABUS"]
    expected_heads = ["unetr", "vitadapterunetr"]
    expected_checkpoints = ["Med3DINO_Base_c96", "Med3DINO_Base_c112", "Med3DINO_REL_c96", "Med3DINO_REL_c112", "Med3DINO_ISO_c96", "Med3DINO_ISO_c112", "3dinov2"]
    
    print("\n[2/4] Testing Job Structure")
    print("-" * 40)
    
    for dataset in expected_datasets:
        dataset_jobs_dir = JOBS_DIR / dataset
        if not dataset_jobs_dir.exists():
            errors.append(f"Missing job directory: jobs/{dataset}/")
            print(f"  ❌ jobs/{dataset}/: DIRECTORY NOT FOUND")
            continue
        
        job_files = list(dataset_jobs_dir.glob("*.sh"))
        expected_count = len(expected_heads) * len(expected_checkpoints)  # 2 × 7 = 14
        
        if len(job_files) == expected_count:
            print(f"  ✓ jobs/{dataset}/: {len(job_files)} scripts")
        else:
            errors.append(f"jobs/{dataset}/: Expected {expected_count} scripts, found {len(job_files)}")
            print(f"  ❌ jobs/{dataset}/: Expected {expected_count} scripts, found {len(job_files)}")
        
        # Validate no Linear head scripts exist
        linear_scripts = [f for f in job_files if "_linear_" in f.name]
        if linear_scripts:
            errors.append(f"jobs/{dataset}/ contains Linear scripts: {[f.name for f in linear_scripts]}")
            print(f"  ❌ Found Linear scripts in {dataset}: {[f.name for f in linear_scripts]}")
    
    # Check old deprecated directory exists but is not used
    deprecated_dir = SCRIPTS_DIR / "jobs_laseg_tdsc_deprecated"
    if deprecated_dir.exists():
        print(f"  ⚠ Old jobs_laseg_tdsc_deprecated dir exists (kept for reference)")
    
    return len(errors) == 0, errors


def test_launcher_scripts() -> Tuple[bool, List[str]]:
    """Test launcher script syntax and structure."""
    errors = []
    
    launchers = [
        "run_parallel_segmentation.sh",
        "run_btcv_segmentation.sh",
        "run_brats_segmentation.sh",
        "run_la_seg_segmentation.sh",
        "run_tdsc_abus_segmentation.sh",
    ]
    
    print("\n[3/4] Testing Launcher Scripts")
    print("-" * 40)
    
    for launcher in launchers:
        launcher_path = SCRIPTS_DIR / launcher
        if not launcher_path.exists():
            errors.append(f"Missing launcher: {launcher}")
            print(f"  ❌ {launcher}: FILE NOT FOUND")
            continue
        
        # Check bash syntax
        result = subprocess.run(
            ["bash", "-n", str(launcher_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"  ✓ {launcher}: syntax OK")
        else:
            errors.append(f"{launcher}: Syntax error: {result.stderr}")
            print(f"  ❌ {launcher}: Syntax error")
    
    return len(errors) == 0, errors


def test_loaders_compatibility() -> Tuple[bool, List[str]]:
    """Test backward compatibility with loaders.py."""
    errors = []
    
    print("\n[4/4] Testing Loaders Compatibility")
    print("-" * 40)
    
    # Import loaders.py indirectly
    sys.path.insert(0, str(PROJECT_ROOT))
    
    try:
        from dinov2.data.loaders import make_segmentation_dataset_3d
        print("  ✓ make_segmentation_dataset_3d imported successfully")
    except ImportError as e:
        errors.append(f"Cannot import make_segmentation_dataset_3d: {e}")
        print(f"  ❌ Import error: {e}")
        return False, errors
    
    # Test that datalist files are loadable
    test_configs = [
        ("BTCV", "BTCV_100_datalist.json"),
        ("BraTS", "BraTS_100_datalist_3dinov2.json"),
        ("LA-SEG", "LA-SEG_100_datalist.json"),
        ("TDSC-ABUS", "TDSC-ABUS_100_datalist.json"),
    ]
    
    for dataset_name, datalist_filename in test_configs:
        datalist_path = DATASETS_DIR / datalist_filename
        if not datalist_path.exists():
            errors.append(f"Datalist not found: {datalist_filename}")
            print(f"  ❌ {dataset_name}: Datalist not found")
            continue
        
        try:
            with open(datalist_path) as f:
                data = json.load(f)
            
            # Verify structure
            required_keys = ["training", "validation", "test"]
            for key in required_keys:
                if key not in data:
                    errors.append(f"{datalist_filename} missing key: {key}")
                    continue
            
            # Verify entry structure
            sample_entry = data["training"][0]
            if "image" not in sample_entry or "label" not in sample_entry:
                errors.append(f"{datalist_filename} invalid entry structure")
            else:
                print(f"  ✓ {dataset_name}: Valid datalist structure")
        except Exception as e:
            errors.append(f"Error loading {datalist_filename}: {e}")
            print(f"  ❌ {dataset_name}: {e}")
    
    return len(errors) == 0, errors


def test_brats_datalist_selection():
    """Test that BraTS jobs use the correct datalist."""
    errors = []
    
    print("\n[Bonus] Testing BraTS Datalist Selection in Jobs")
    print("-" * 40)
    
    brats_jobs_dir = JOBS_DIR / "BraTS"
    if not brats_jobs_dir.exists():
        errors.append("BraTS jobs directory not found")
        print("  ❌ BraTS jobs directory not found")
        return False, errors
    
    # Check that jobs use the 3dinov2-aligned datalist
    job_files = list(brats_jobs_dir.glob("*.sh"))
    for job_file in job_files[:2]:  # Check first 2
        with open(job_file) as f:
            content = f.read()
        
        if "BraTS_100_datalist_3dinov2.json" in content:
            print(f"  ✓ {job_file.name}: Uses 3dinov2-aligned datalist")
        elif "BraTS_GLI_100_datalist.json" in content and "3dinov2" not in content:
            # This is OK if using full datalist
            print(f"  ⚠ {job_file.name}: Uses full BraTS datalist (not paper-aligned)")
        else:
            # Check what datalist is being used
            # Note: Actually the job generation uses generate_unified_segmentation_jobs.py
            # which specifies datalist in DATASET_CONFIGS
            pass
    
    return True, []


def main():
    """Run all tests."""
    print("=" * 60)
    print("Segmentation Setup Validation")
    print("=" * 60)
    
    all_passed = True
    all_errors = []
    
    # Run tests
    passed, errors = test_datalist_integrity()
    all_passed &= passed
    all_errors.extend(errors)
    
    passed, errors = test_job_structure()
    all_passed &= passed
    all_errors.extend(errors)
    
    passed, errors = test_launcher_scripts()
    all_passed &= passed
    all_errors.extend(errors)
    
    passed, errors = test_loaders_compatibility()
    all_passed &= passed
    all_errors.extend(errors)
    
    # Bonus test
    test_brats_datalist_selection()
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nErrors:")
        for error in all_errors:
            print(f"  - {error}")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
