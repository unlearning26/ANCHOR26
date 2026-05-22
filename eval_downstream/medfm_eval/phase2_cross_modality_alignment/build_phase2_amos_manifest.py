#!/usr/bin/env python

"""Build the Phase 2 AMOS-2022 CT/MR abdominal-organ manifest.

This builder converts the labeled AMOS-2022 train and validation splits into
the canonical Phase 2 manifest trio:

1. ``manifest_full.json``
2. ``manifest_sampled.json``
3. ``manifest_meta.json``

Each retained sample is one organ label extracted from a multiclass AMOS label
volume. AMOS stores modality implicitly in the case id: ids below 500 are CT,
while ids 500 and above are MRI. The local ``dataset.json`` declares only CT,
so this builder intentionally uses the id convention documented by AMOS.

Usage examples:
    python build_phase2_amos_manifest.py
    python build_phase2_amos_manifest.py --min-voxels 500
    python build_phase2_amos_manifest.py --min-samples-per-modality 20 --splits training validation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import nibabel as nib
import numpy as np
from tqdm import tqdm

from phase2_config import get_phase2_manifest_dir


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DATASET_NAME = "amos_ct_mr"
MANIFEST_VARIANT = "core"
EVIDENCE_TYPE = "cohort_level_same_organ_ct_mr"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = Path(os.environ.get("MED3DINO_AMOS_ROOT", PROJECT_ROOT / "data" / "amos22"))
DEFAULT_SPLITS = ("training", "validation")

ORGAN_LABELS: Dict[int, str] = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "esophagus",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "inferior_vena_cava",
    10: "pancreas",
    11: "adrenal_gland_right",
    12: "adrenal_gland_left",
    13: "duodenum",
    14: "urinary_bladder",
    15: "prostate_uterus",
}
EXCLUDED_ORGAN_NAMES = {"urinary_bladder", "prostate_uterus"}
RAW_LABEL_NAME_OVERRIDES: Dict[int, str] = {
    8: "arota",
    9: "postcava",
    15: "prostate/uterus",
}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_dataset_json(dataset_root: Path) -> Dict[str, Any]:
    """Load the AMOS dataset metadata file."""
    metadata_path = dataset_root / "dataset.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing AMOS dataset.json: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _case_id_from_image_entry(image_entry: str) -> str:
    """Return a case id such as ``amos_0001`` from a metadata image path."""
    return Path(image_entry).name.removesuffix(".nii.gz")


def _case_number(case_id: str) -> int:
    """Return the numeric AMOS id from a case id such as ``amos_0001``."""
    try:
        return int(case_id.split("_")[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unable to parse AMOS case id: {case_id}") from exc


def _infer_modality(case_id: str) -> str:
    """Infer AMOS modality using the documented id convention."""
    return "ct" if _case_number(case_id) < 500 else "mr"


def _resolve_path(dataset_root: Path, relative_path: str) -> Path:
    """Resolve a path from AMOS dataset.json."""
    return (dataset_root / relative_path).resolve()


def _load_image_metadata(image_path: Path) -> Tuple[List[int], List[float]]:
    """Read shape and affine-column-norm spacing for one NIfTI volume."""
    image = nib.load(str(image_path))
    shape = [int(value) for value in image.shape[:3]]
    spacing = [float(value) for value in np.linalg.norm(image.affine[:3, :3], axis=0)]
    return shape, spacing


def _load_label_counts(label_path: Path, min_voxels: int) -> Dict[int, int]:
    """Count retained voxels for each supported AMOS label value."""
    label_array = np.asanyarray(nib.load(str(label_path)).dataobj)
    if label_array.ndim != 3:
        raise ValueError(f"Expected 3D AMOS label volume, got shape {label_array.shape} for {label_path}")
    counts = np.bincount(label_array.astype(np.int16, copy=False).ravel(), minlength=max(ORGAN_LABELS) + 1)
    return {
        label_value: int(counts[label_value]) if int(counts[label_value]) >= min_voxels else 0
        for label_value in ORGAN_LABELS
    }


def _iter_labeled_entries(metadata: Dict[str, Any], splits: Iterable[str]) -> List[Tuple[str, Dict[str, str]]]:
    """Return labeled metadata entries from selected AMOS splits."""
    entries: List[Tuple[str, Dict[str, str]]] = []
    for split in splits:
        for entry in metadata.get(split, []):
            if not isinstance(entry, dict) or "image" not in entry or "label" not in entry:
                continue
            entries.append((split, entry))
    return entries


def _build_case_records(
    metadata: Dict[str, Any],
    dataset_root: Path,
    splits: Iterable[str],
    min_voxels: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """Build AMOS case-level records and track per-organ support by modality."""
    entries = _iter_labeled_entries(metadata, splits)
    support_counts = {organ: {"ct": 0, "mr": 0} for organ in ORGAN_LABELS.values()}
    case_records: List[Dict[str, Any]] = []

    iterator = tqdm(entries, desc="Scanning AMOS labeled cases", unit="case")
    for split, entry in iterator:
        case_id = _case_id_from_image_entry(entry["image"])
        modality = _infer_modality(case_id)
        image_path = _resolve_path(dataset_root, entry["image"])
        label_path = _resolve_path(dataset_root, entry["label"])
        if not image_path.exists():
            raise FileNotFoundError(f"Missing AMOS image: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing AMOS label: {label_path}")

        shape, spacing = _load_image_metadata(image_path)
        organ_counts_by_label = _load_label_counts(label_path, min_voxels=min_voxels)
        organ_counts = {
            ORGAN_LABELS[label_value]: int(count)
            for label_value, count in organ_counts_by_label.items()
        }
        present_organs = [organ for organ, count in organ_counts.items() if count > 0]
        for organ in present_organs:
            support_counts[organ][modality] += 1

        case_records.append(
            {
                "dataset": DATASET_NAME,
                "source_dataset": "amos22",
                "modality": modality,
                "case_id": case_id,
                "patient_id": case_id,
                "split": split,
                "file_path": str(image_path),
                "label_path": str(label_path),
                "shape": shape,
                "spacing": spacing,
                "organ_counts": organ_counts,
            }
        )
        iterator.set_postfix(
            retained_cases=len(case_records),
            ct=sum(1 for record in case_records if record["modality"] == "ct"),
            mr=sum(1 for record in case_records if record["modality"] == "mr"),
        )

    return case_records, support_counts


def _retained_organs(support_counts: Dict[str, Dict[str, int]], min_samples_per_modality: int) -> List[str]:
    """Return AMOS organs retained for the CT/MR surface."""
    return [
        organ
        for organ in ORGAN_LABELS.values()
        if organ not in EXCLUDED_ORGAN_NAMES
        and support_counts.get(organ, {}).get("ct", 0) >= min_samples_per_modality
        and support_counts.get(organ, {}).get("mr", 0) >= min_samples_per_modality
    ]


def _build_samples(case_records: List[Dict[str, Any]], retained_organs: List[str]) -> List[Dict[str, Any]]:
    """Explode AMOS case-level records into organ-level Phase 2 samples."""
    label_value_by_organ = {organ: label_value for label_value, organ in ORGAN_LABELS.items()}
    retained_set = set(retained_organs)
    samples: List[Dict[str, Any]] = []
    for case in case_records:
        for organ in retained_organs:
            voxel_count = int(case["organ_counts"].get(organ, 0))
            if voxel_count <= 0 or organ not in retained_set:
                continue
            samples.append(
                {
                    "sample_id": f"{case['modality']}:{case['case_id']}:{organ}",
                    "file_path": case["file_path"],
                    "mask_path": case["label_path"],
                    "label_path": case["label_path"],
                    "mask_label_value": int(label_value_by_organ[organ]),
                    "dataset": DATASET_NAME,
                    "source_dataset": "amos22",
                    "modality": case["modality"],
                    "primary_organ": organ,
                    "organs": [organ],
                    "patient_id": case["patient_id"],
                    "source_case_id": case["case_id"],
                    "split": case["split"],
                    "evidence_type": EVIDENCE_TYPE,
                    "pretraining_exposure": "pretraining_unseen_source",
                    "shape": case["shape"],
                    "spacing": case["spacing"],
                    "label_voxels": voxel_count,
                }
            )
    return sorted(samples, key=lambda sample: (sample["primary_organ"], sample["modality"], sample["source_case_id"]))


def _support_by_organ(support_counts: Dict[str, Dict[str, int]], organs: Iterable[str]) -> Dict[str, Dict[str, int]]:
    """Serialize per-organ modality support."""
    return {
        organ: {
            "ct_cases": int(support_counts.get(organ, {}).get("ct", 0)),
            "mr_cases": int(support_counts.get(organ, {}).get("mr", 0)),
        }
        for organ in organs
    }


def build_manifest(
    dataset_root: Path,
    splits: Iterable[str],
    min_voxels: int,
    min_samples_per_modality: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Build the full, sampled, and metadata manifest artifacts for AMOS."""
    metadata = _load_dataset_json(dataset_root)
    selected_splits = list(splits)
    case_records, support_counts = _build_case_records(
        metadata=metadata,
        dataset_root=dataset_root,
        splits=selected_splits,
        min_voxels=min_voxels,
    )
    retained_organs = _retained_organs(support_counts, min_samples_per_modality=min_samples_per_modality)
    samples = _build_samples(case_records, retained_organs)

    excluded_organs = {
        organ: {
            **_support_by_organ(support_counts, [organ])[organ],
            "reason": (
                "excluded_by_protocol_low_mr_support"
                if organ in EXCLUDED_ORGAN_NAMES
                else "insufficient_cross_modal_case_support"
            ),
        }
        for organ in ORGAN_LABELS.values()
        if organ not in retained_organs
    }
    source_case_counts = defaultdict(int)
    per_modality_sample_counts = defaultdict(int)
    per_split_case_counts = defaultdict(int)
    for record in case_records:
        source_case_counts[record["modality"]] += 1
        per_split_case_counts[f"{record['split']}_{record['modality']}"] += 1
    for sample in samples:
        per_modality_sample_counts[sample["modality"]] += 1

    manifest_full = {
        "version": "1.0",
        "description": "Phase 2 full manifest: AMOS-2022 cohort-level CT/MR abdominal-organ validation surface",
        "phase": "phase2",
        "dataset": DATASET_NAME,
        "manifest_variant": MANIFEST_VARIANT,
        "evidence_type": EVIDENCE_TYPE,
        "retained_organs": retained_organs,
        "samples": samples,
    }
    manifest_sampled = {
        **manifest_full,
        "description": "Phase 2 sampled manifest: AMOS-2022 cohort-level CT/MR abdominal-organ validation surface",
        "sampling": {"applied": False, "strategy": "none", "reason": "manifest_sampled_equals_manifest_full"},
    }
    manifest_meta = {
        "dataset": DATASET_NAME,
        "phase": "phase2",
        "manifest_variant": MANIFEST_VARIANT,
        "source_root": str(dataset_root),
        "source_case_counts": dict(sorted(source_case_counts.items())),
        "source_case_counts_by_split": dict(sorted(per_split_case_counts.items())),
        "retained_organs": retained_organs,
        "excluded_organs": excluded_organs,
        "organ_label_map": {str(label_value): organ_name for label_value, organ_name in ORGAN_LABELS.items()},
        "raw_organ_label_map": {
            str(label_value): RAW_LABEL_NAME_OVERRIDES.get(label_value, organ_name)
            for label_value, organ_name in ORGAN_LABELS.items()
        },
        "support_by_organ": _support_by_organ(support_counts, retained_organs),
        "total_samples_full": len(samples),
        "total_samples_sampled": len(samples),
        "sample_counts_by_modality": dict(sorted(per_modality_sample_counts.items())),
        "evidence_type": EVIDENCE_TYPE,
        "min_voxels": int(min_voxels),
        "min_samples_per_modality": int(min_samples_per_modality),
        "selected_splits": selected_splits,
        "modality_rule": "AMOS case ids below 500 are CT; ids 500 and above are MRI.",
        "claim_boundary": (
            "AMOS supports cohort-level same-organ CT/MR alignment and cross-modal anatomical hold-out analysis; "
            "it does not provide paired same-patient or registration-level CT/MR evidence."
        ),
        "notes": [
            "Each AMOS sample is one organ label extracted from a multiclass CT or MR label volume.",
            "The local AMOS dataset.json declares only CT, so modality is derived from the documented case-id convention.",
            "Bladder and prostate/uterus are excluded from the CT/MR surface because the labeled MR split contains too few positive cases.",
        ],
    }
    return manifest_full, manifest_sampled, manifest_meta


def main() -> None:
    """Parse CLI arguments, build the AMOS manifest trio, and write it to disk."""
    parser = argparse.ArgumentParser(
        description="Build the Phase 2 AMOS-2022 CT/MR organ manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python build_phase2_amos_manifest.py\n"
            "  python build_phase2_amos_manifest.py --min-voxels 500\n"
            "  python build_phase2_amos_manifest.py --min-samples-per-modality 20 --splits training validation"
        ),
    )
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT), help="Path to the AMOS-2022 dataset root")
    parser.add_argument("--min-voxels", type=int, default=100, help="Minimum voxels required to retain an organ in a case")
    parser.add_argument(
        "--min-samples-per-modality",
        type=int,
        default=5,
        help="Minimum case support per organ and modality for retention",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=("training", "validation"),
        help="Labeled AMOS splits to include",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    manifest_dir = get_phase2_manifest_dir(DATASET_NAME, MANIFEST_VARIANT)
    logger.info("Starting AMOS manifest build")
    logger.info("Source root: %s", dataset_root)
    logger.info("Output dir: %s", manifest_dir)
    logger.info("Selected splits: %s", ", ".join(args.splits))
    logger.info("Min voxels per organ: %d", args.min_voxels)
    logger.info("Min cases per modality: %d", args.min_samples_per_modality)

    manifest_full, manifest_sampled, manifest_meta = build_manifest(
        dataset_root=dataset_root,
        splits=args.splits,
        min_voxels=args.min_voxels,
        min_samples_per_modality=args.min_samples_per_modality,
    )
    _write_json(manifest_dir / "manifest_full.json", manifest_full)
    _write_json(manifest_dir / "manifest_sampled.json", manifest_sampled)
    _write_json(manifest_dir / "manifest_meta.json", manifest_meta)
    logger.info("Wrote Phase 2 AMOS manifests to %s", manifest_dir)
    logger.info("Retained organs: %s", ", ".join(manifest_full["retained_organs"]) or "none")
    logger.info("Total organ-level samples: %d", len(manifest_full["samples"]))


if __name__ == "__main__":
    main()