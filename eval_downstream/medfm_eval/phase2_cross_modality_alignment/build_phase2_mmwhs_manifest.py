#!/usr/bin/env python

"""Build the Phase 2 MMWHS CT/MR validation manifest.

This builder converts the paired MMWHS CT and MR training splits into the
canonical Phase 2 manifest trio:

1. ``manifest_full.json``
2. ``manifest_sampled.json``
3. ``manifest_meta.json``

Each retained sample is one organ extracted from a multiclass cardiac label
volume. Progress bars are shown while scanning cases so support accumulation is
visible during long manifest builds.

Usage examples:
    python build_phase2_mmwhs_manifest.py
    python build_phase2_mmwhs_manifest.py --min-voxels 500
    python build_phase2_mmwhs_manifest.py --min-samples-per-modality 10 --min-voxels 250
    python build_phase2_mmwhs_manifest.py --dataset-root /path/to/mmwhs
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import SimpleITK as sitk
from tqdm import tqdm

from phase2_config import get_phase2_manifest_dir


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DATASET_NAME = "mmwhs_ct_mr"
MANIFEST_VARIANT = "core"
EVIDENCE_TYPE = "paired_same_case_ct_mr"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = Path(os.environ.get("MED3DINO_MMWHS_ROOT", PROJECT_ROOT / "data" / "mmwhs"))
ORGAN_LABELS: Dict[int, str] = {
    205: "myocardium",
    420: "left_atrium",
    500: "left_ventricle",
    550: "right_atrium",
    600: "right_ventricle",
    820: "ascending_aorta",
    850: "pulmonary_artery",
}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_image_path(image_entry: Path) -> Path:
    """Resolve the actual image file for an MMWHS image entry.

    Some surfaces expose the image directly as a file, while others wrap the
    image in a one-file directory.
    """
    if image_entry.is_file():
        return image_entry
    if image_entry.is_dir():
        files = sorted(path for path in image_entry.iterdir() if path.is_file())
        if len(files) != 1:
            raise ValueError(f"Expected exactly one image file under {image_entry}, found {len(files)}")
        return files[0]
    raise FileNotFoundError(f"Missing image entry: {image_entry}")


def _load_image_metadata(image_path: Path) -> Tuple[List[int], List[float]]:
    """Read shape and spacing for an MMWHS image volume."""
    image = sitk.ReadImage(str(image_path))
    return list(reversed(image.GetSize()[:3])), [float(x) for x in image.GetSpacing()[:3]]


def _load_label_counts(label_path: Path, min_voxels: int) -> Dict[int, int]:
    """Count retained voxels for each supported cardiac label value."""
    label = sitk.ReadImage(str(label_path))
    array = sitk.GetArrayFromImage(label)
    counts: Dict[int, int] = {}
    for label_value in ORGAN_LABELS:
        voxel_count = int((array == label_value).sum())
        counts[label_value] = voxel_count if voxel_count >= min_voxels else 0
    return counts


def _build_case_records(dataset_root: Path, modality: str, min_voxels: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Build case-level records for one modality and track per-organ support."""
    subset_dir = dataset_root / f"{modality}_train"
    support_counts = {organ: 0 for organ in ORGAN_LABELS.values()}
    case_records: List[Dict[str, Any]] = []

    label_paths = sorted(subset_dir.glob(f"{modality}_train_*_label.nii"))
    iterator = tqdm(label_paths, desc=f"Scanning MMWHS {modality.upper()} cases", unit="case")
    for label_path in iterator:
        case_id = label_path.stem.replace("_label", "")
        image_entry = subset_dir / f"{case_id}_image.nii"
        image_path = _resolve_image_path(image_entry)
        shape, spacing = _load_image_metadata(image_path)
        organ_counts = _load_label_counts(label_path, min_voxels=min_voxels)
        present_organs = [organ_name for label_value, organ_name in ORGAN_LABELS.items() if organ_counts[label_value] > 0]

        for organ_name in present_organs:
            support_counts[organ_name] += 1

        patient_id = case_id.split("_")[-1]
        case_records.append(
            {
                "dataset": DATASET_NAME,
                "modality": modality.replace("ct", "ct").replace("mr", "mr"),
                "case_id": case_id,
                "patient_id": patient_id,
                "file_path": str(image_path),
                "label_path": str(label_path),
                "shape": shape,
                "spacing": spacing,
                "organ_counts": {ORGAN_LABELS[label_value]: count for label_value, count in organ_counts.items()},
            }
        )
        iterator.set_postfix(retained_cases=len(case_records), supported_organs=sum(count > 0 for count in support_counts.values()))

    logger.info("Prepared %d %s cases", len(case_records), modality)
    return case_records, support_counts


def _build_samples(
    ct_cases: List[Dict[str, Any]],
    mr_cases: List[Dict[str, Any]],
    retained_organs: List[str],
) -> List[Dict[str, Any]]:
    """Explode paired case-level records into organ-level Phase 2 samples."""
    label_value_by_organ = {organ_name: label_value for label_value, organ_name in ORGAN_LABELS.items()}
    samples: List[Dict[str, Any]] = []
    for case in [*ct_cases, *mr_cases]:
        for organ in retained_organs:
            voxel_count = int(case["organ_counts"].get(organ, 0))
            if voxel_count <= 0:
                continue
            samples.append(
                {
                    "sample_id": f"{case['modality']}:{case['case_id']}:{organ}",
                    "file_path": case["file_path"],
                    "mask_path": case["label_path"],
                    "mask_label_value": int(label_value_by_organ[organ]),
                    "dataset": DATASET_NAME,
                    "modality": case["modality"],
                    "primary_organ": organ,
                    "organs": [organ],
                    "patient_id": case["patient_id"],
                    "source_case_id": case["case_id"],
                    "evidence_type": EVIDENCE_TYPE,
                    "shape": case["shape"],
                    "spacing": case["spacing"],
                    "label_voxels": voxel_count,
                }
            )
    return sorted(samples, key=lambda sample: (sample["primary_organ"], sample["modality"], sample["source_case_id"]))


def build_manifest(dataset_root: Path, min_voxels: int, min_samples_per_modality: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Build the full, sampled, and metadata manifest artifacts for MMWHS."""
    ct_cases, ct_support = _build_case_records(dataset_root, "ct", min_voxels=min_voxels)
    mr_cases, mr_support = _build_case_records(dataset_root, "mr", min_voxels=min_voxels)

    retained_organs = [
        organ
        for organ in ORGAN_LABELS.values()
        if ct_support.get(organ, 0) >= min_samples_per_modality and mr_support.get(organ, 0) >= min_samples_per_modality
    ]
    excluded_organs = {
        organ: {
            "ct_cases": int(ct_support.get(organ, 0)),
            "mr_cases": int(mr_support.get(organ, 0)),
            "reason": "insufficient_cross_modal_case_support",
        }
        for organ in ORGAN_LABELS.values()
        if organ not in retained_organs
    }

    samples = _build_samples(ct_cases, mr_cases, retained_organs)
    per_modality_sample_counts = defaultdict(int)
    for sample in samples:
        per_modality_sample_counts[sample["modality"]] += 1

    manifest_full = {
        "version": "1.0",
        "description": "Phase 2 full manifest: MMWHS CT/MR cardiac validation surface",
        "phase": "phase2",
        "dataset": DATASET_NAME,
        "manifest_variant": MANIFEST_VARIANT,
        "evidence_type": EVIDENCE_TYPE,
        "retained_organs": retained_organs,
        "samples": samples,
    }
    manifest_sampled = {
        **manifest_full,
        "description": "Phase 2 sampled manifest: MMWHS CT/MR cardiac validation surface",
        "sampling": {"applied": False, "strategy": "none", "reason": "manifest_sampled_equals_manifest_full"},
    }
    manifest_meta = {
        "dataset": DATASET_NAME,
        "phase": "phase2",
        "manifest_variant": MANIFEST_VARIANT,
        "source_root": str(DATASET_ROOT),
        "source_case_counts": {"ct": len(ct_cases), "mr": len(mr_cases)},
        "retained_organs": retained_organs,
        "excluded_organs": excluded_organs,
        "organ_label_map": {str(label_value): organ_name for label_value, organ_name in ORGAN_LABELS.items()},
        "support_by_organ": {
            organ: {"ct_cases": int(ct_support.get(organ, 0)), "mr_cases": int(mr_support.get(organ, 0))}
            for organ in retained_organs
        },
        "total_samples_full": len(samples),
        "total_samples_sampled": len(samples),
        "sample_counts_by_modality": dict(sorted(per_modality_sample_counts.items())),
        "evidence_type": EVIDENCE_TYPE,
        "min_voxels": int(min_voxels),
        "min_samples_per_modality": int(min_samples_per_modality),
        "notes": [
            "Each MMWHS sample is one organ label extracted from a multiclass CT or MR label volume.",
            "CT and MR case ids are paired by dataset naming convention and retained as patient identifiers.",
        ],
    }
    return manifest_full, manifest_sampled, manifest_meta


def main() -> None:
    """Parse CLI arguments, build the MMWHS manifest trio, and write it to disk."""
    parser = argparse.ArgumentParser(
        description="Build the Phase 2 MMWHS CT/MR manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python build_phase2_mmwhs_manifest.py\n"
            "  python build_phase2_mmwhs_manifest.py --min-voxels 500\n"
            "  python build_phase2_mmwhs_manifest.py --min-samples-per-modality 10 --min-voxels 250\n"
            "  python build_phase2_mmwhs_manifest.py --dataset-root /path/to/mmwhs"
        ),
    )
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT), help="Path to the MMWHS dataset root")
    parser.add_argument("--min-voxels", type=int, default=100, help="Minimum voxels required to retain an organ in a case")
    parser.add_argument(
        "--min-samples-per-modality",
        type=int,
        default=5,
        help="Minimum case support per organ and modality for retention",
    )
    args = parser.parse_args()

    manifest_dir = get_phase2_manifest_dir(DATASET_NAME, MANIFEST_VARIANT)
    dataset_root = Path(args.dataset_root).resolve()
    logger.info("Starting MMWHS manifest build")
    logger.info("Source root: %s", dataset_root)
    logger.info("Output dir: %s", manifest_dir)
    logger.info("Min voxels per organ: %d", args.min_voxels)
    logger.info("Min cases per modality: %d", args.min_samples_per_modality)
    manifest_full, manifest_sampled, manifest_meta = build_manifest(
        dataset_root=dataset_root,
        min_voxels=args.min_voxels,
        min_samples_per_modality=args.min_samples_per_modality,
    )
    _write_json(manifest_dir / "manifest_full.json", manifest_full)
    _write_json(manifest_dir / "manifest_sampled.json", manifest_sampled)
    _write_json(manifest_dir / "manifest_meta.json", manifest_meta)
    logger.info("Wrote Phase 2 MMWHS manifests to %s", manifest_dir)
    logger.info("Retained organs: %s", ", ".join(manifest_full["retained_organs"]) or "none")
    logger.info("Total organ-level samples: %d", len(manifest_full["samples"]))


if __name__ == "__main__":
    main()