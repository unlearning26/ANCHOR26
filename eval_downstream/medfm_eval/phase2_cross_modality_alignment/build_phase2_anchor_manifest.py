#!/usr/bin/env python

"""Build the Phase 2 TotalSegmenter CT/MR anchor manifest.

This script materializes the first executable Phase 2 surface from the existing
Phase 1 TotalSegmenter CT and TotalSegmenterMRI manifests.

What it does:
1. loads the Phase 1 full manifests for the CT and MRI anchor datasets,
2. resolves shared-organ presence for each case,
3. keeps only organs with enough support on both modalities,
4. explodes each case into organ-level Phase 2 samples,
5. writes `manifest_full.json`, `manifest_sampled.json`, and `manifest_meta.json`.

How to use it:
        cd eval_downstream/medfm_eval/phase2_cross_modality_alignment
        /path/to/python build_phase2_anchor_manifest.py

Example with stricter support filtering:
        /path/to/python build_phase2_anchor_manifest.py \
                --min-voxels 500 \
                --min-samples-per-modality 20

Notes:
- The Phase 2 anchor is cohort-level same-organ CT/MR evidence, not paired
    same-case or same-patient evidence.
- `manifest_sampled.json` is intentionally identical to `manifest_full.json`
    unless a later workflow introduces explicit subsampling.
- TotalSegmenterMRI can reuse a cached semantic-label file from Phase 1. The CT
    side currently scans segmentation directories directly, which is why progress
    reporting matters during the first pass.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import nibabel as nib
import numpy as np
from tqdm import tqdm

from phase2_config import get_phase2_manifest_dir


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


PHASE1_MANIFESTS = Path(__file__).resolve().parents[1] / "data_manifests" / "phase1_anisotropy_robustness"
ANCHOR_DATASET_NAME = "totalsegmenter_ct_mr_anchor"
ANCHOR_MANIFEST_VARIANT = "core"
EVIDENCE_TYPE = "cohort_level_same_organ_ct_mr"
SHARED_ORGAN_LABEL_SET = [
    "spleen",
    "kidney_left",
    "kidney_right",
    "gallbladder",
    "liver",
    "stomach",
    "pancreas",
    "small_bowel",
    "colon",
    "urinary_bladder",
    "aorta",
    "inferior_vena_cava",
]


def _load_json(path: Path) -> Dict[str, Any] | List[Dict[str, Any]]:
    """Load a JSON file and return its decoded Python payload."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON payload with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _phase1_manifest_path(dataset_name: str, manifest_kind: str = "full") -> Path:
    """Return the Phase 1 manifest path used as a source surface for Phase 2."""
    return PHASE1_MANIFESTS / dataset_name / "original_bins" / f"manifest_{manifest_kind}.json"


def _phase1_semantic_cache_path(dataset_name: str) -> Path | None:
    """Return the Phase 1 semantic-label cache path when one exists.

    Phase 2 uses the Phase 1 cache only as a fast path for structure-presence
    counts. If no cache exists, the builder falls back to scanning the
    segmentation directory directly.
    """
    outputs_root = Path(__file__).resolve().parents[0] / ".." / "phase1_anisotropy_robustness" / "outputs_phase1"
    outputs_root = outputs_root.resolve()
    if dataset_name == "totalsegmentermri":
        cache_path = outputs_root / dataset_name / "phase1" / "original_bins" / "features" / (
            "semantic_labels_totalsegmentermri_shared_structure_presence_totalsegmentermri.json"
        )
        return cache_path if cache_path.exists() else None
    if dataset_name == "totalsegmenter_ct":
        return None
    return None


def _count_mask_voxels(mask_path: Path) -> int:
    """Count foreground voxels in a binary organ mask."""
    img = nib.load(str(mask_path))
    data = np.asarray(img.dataobj)
    return int(np.count_nonzero(data > 0))


def _load_presence_from_cache(cache_path: Path) -> Dict[str, Dict[str, int]]:
    """Load per-case organ voxel counts from a cached Phase 1 semantic-label file."""
    payload = _load_json(cache_path)
    return {
        str(file_path): {
            str(organ): int(count)
            for organ, count in (entry.get("organ_volumes") or {}).items()
        }
        for file_path, entry in payload.items()
    }


def _extract_structure_counts(label_dir: Path, min_voxels: int) -> Dict[str, int]:
    """Scan a TotalSegmenter segmentation directory for shared-organ presence.

    Any organ with fewer than `min_voxels` foreground voxels is treated as
    absent for Phase 2.
    """
    organ_counts: Dict[str, int] = {}
    for organ in SHARED_ORGAN_LABEL_SET:
        mask_path = label_dir / f"{organ}.nii.gz"
        if not mask_path.exists():
            organ_counts[organ] = 0
            continue
        try:
            voxel_count = _count_mask_voxels(mask_path)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", mask_path, exc)
            voxel_count = 0
        organ_counts[organ] = voxel_count if voxel_count >= min_voxels else 0
    return organ_counts


def _load_source_volumes(dataset_name: str) -> List[Dict[str, Any]]:
    """Load the Phase 1 full manifest entries used as Phase 2 source cases."""
    manifest = _load_json(_phase1_manifest_path(dataset_name, "full"))
    if isinstance(manifest, dict):
        return list(manifest.get("volumes") or manifest.get("samples") or [])
    return list(manifest)


def _build_case_records(
    dataset_name: str,
    modality: str,
    min_voxels: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Build case-level records with shared-organ presence counts.

    The output remains case-level so the builder can first decide which organs
    are valid for the anchor surface before exploding each case into organ-level
    samples.
    """
    volumes = _load_source_volumes(dataset_name)
    cached_counts = {}
    cache_path = _phase1_semantic_cache_path(dataset_name)
    used_cache = False
    if cache_path is not None:
        logger.info("Loading cached structure counts from %s", cache_path)
        cached_counts = _load_presence_from_cache(cache_path)
        used_cache = True

    case_records: List[Dict[str, Any]] = []
    support_counts = {organ: 0 for organ in SHARED_ORGAN_LABEL_SET}
    cache_hits = 0
    scanned_cases = 0
    iterator = tqdm(
        volumes,
        total=len(volumes),
        desc=f"Building {dataset_name} case records",
        unit="case",
    )
    for volume in iterator:
        image_path = Path(volume["file_path"])
        label_dir = Path(volume["label_path"])
        case_id = image_path.parent.name

        organ_counts = cached_counts.get(str(image_path))
        if organ_counts is None:
            organ_counts = _extract_structure_counts(label_dir, min_voxels)
            scanned_cases += 1
        else:
            cache_hits += 1

        present_organs = [organ for organ in SHARED_ORGAN_LABEL_SET if int(organ_counts.get(organ, 0)) > 0]
        for organ in present_organs:
            support_counts[organ] += 1

        case_records.append(
            {
                "dataset": dataset_name,
                "modality": modality,
                "case_id": case_id,
                "patient_id": case_id,
                "file_path": str(image_path),
                "label_path": str(label_dir),
                "shape": volume.get("shape"),
                "spacing": volume.get("spacing"),
                "present_organs": present_organs,
                "organ_counts": {organ: int(organ_counts.get(organ, 0)) for organ in SHARED_ORGAN_LABEL_SET},
            }
        )
        iterator.set_postfix(cached=cache_hits, scanned=scanned_cases, present=len(present_organs))

    logger.info(
        "Prepared %d %s cases (%d cache hits, %d scanned, cache_available=%s)",
        len(case_records),
        dataset_name,
        cache_hits,
        scanned_cases,
        used_cache,
    )

    return case_records, support_counts


def _build_phase2_samples(
    ct_cases: List[Dict[str, Any]],
    mr_cases: List[Dict[str, Any]],
    retained_organs: Iterable[str],
) -> List[Dict[str, Any]]:
    """Explode case-level records into organ-level Phase 2 samples."""
    retained_set = set(retained_organs)
    samples: List[Dict[str, Any]] = []
    for case in [*ct_cases, *mr_cases]:
        for organ in case["present_organs"]:
            if organ not in retained_set:
                continue
            samples.append(
                {
                    "sample_id": f"{case['modality']}:{case['case_id']}:{organ}",
                    "file_path": case["file_path"],
                    "mask_path": str(Path(case["label_path"]) / f"{organ}.nii.gz"),
                    "label_path": case["label_path"],
                    "dataset": case["dataset"],
                    "modality": case["modality"],
                    "primary_organ": organ,
                    "organs": [organ],
                    "patient_id": case["patient_id"],
                    "source_case_id": case["case_id"],
                    "evidence_type": EVIDENCE_TYPE,
                    "shape": case.get("shape"),
                    "spacing": case.get("spacing"),
                    "label_voxels": int(case["organ_counts"].get(organ, 0)),
                }
            )
    return sorted(samples, key=lambda sample: (sample["primary_organ"], sample["modality"], sample["source_case_id"]))


def build_anchor_manifest(min_voxels: int, min_samples_per_modality: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Build the three canonical Phase 2 anchor manifest artifacts.

    Returns:
        `(manifest_full, manifest_sampled, manifest_meta)`
    """
    ct_cases, ct_support = _build_case_records("totalsegmenter_ct", "ct", min_voxels)
    mr_cases, mr_support = _build_case_records("totalsegmentermri", "mr", min_voxels)

    retained_organs = [
        organ
        for organ in SHARED_ORGAN_LABEL_SET
        if ct_support.get(organ, 0) >= min_samples_per_modality and mr_support.get(organ, 0) >= min_samples_per_modality
    ]
    excluded_organs = {
        organ: {
            "ct_cases": int(ct_support.get(organ, 0)),
            "mr_cases": int(mr_support.get(organ, 0)),
            "reason": "insufficient_cross_modal_case_support",
        }
        for organ in SHARED_ORGAN_LABEL_SET
        if organ not in retained_organs
    }

    samples = _build_phase2_samples(ct_cases, mr_cases, retained_organs)
    per_modality_sample_counts = defaultdict(int)
    per_organ_support = {}
    for sample in samples:
        per_modality_sample_counts[sample["modality"]] += 1
    for organ in retained_organs:
        per_organ_support[organ] = {
            "ct_cases": int(ct_support.get(organ, 0)),
            "mr_cases": int(mr_support.get(organ, 0)),
        }

    logger.info("Retained %d shared organs after support filtering", len(retained_organs))
    for organ in retained_organs:
        logger.info(
            "  organ=%s ct_cases=%d mr_cases=%d",
            organ,
            per_organ_support[organ]["ct_cases"],
            per_organ_support[organ]["mr_cases"],
        )

    full_manifest = {
        "version": "1.0",
        "description": "Phase 2 full manifest: TotalSegmenter CT + TotalSegmenterMRI anchor surface",
        "phase": "phase2",
        "dataset": ANCHOR_DATASET_NAME,
        "manifest_variant": ANCHOR_MANIFEST_VARIANT,
        "evidence_type": EVIDENCE_TYPE,
        "retained_organs": retained_organs,
        "samples": samples,
    }
    sampled_manifest = {
        **full_manifest,
        "description": "Phase 2 sampled manifest: TotalSegmenter CT + TotalSegmenterMRI anchor surface",
        "sampling": {
            "applied": False,
            "strategy": "none",
            "reason": "manifest_sampled_equals_manifest_full",
        },
    }
    manifest_meta = {
        "dataset": ANCHOR_DATASET_NAME,
        "phase": "phase2",
        "manifest_variant": ANCHOR_MANIFEST_VARIANT,
        "source_manifests": {
            "ct": str(_phase1_manifest_path("totalsegmenter_ct", "full")),
            "mr": str(_phase1_manifest_path("totalsegmentermri", "full")),
        },
        "source_case_counts": {
            "ct": len(ct_cases),
            "mr": len(mr_cases),
        },
        "retained_organs": retained_organs,
        "excluded_organs": excluded_organs,
        "shared_organ_label_set": SHARED_ORGAN_LABEL_SET,
        "support_by_organ": per_organ_support,
        "total_samples_full": len(samples),
        "total_samples_sampled": len(samples),
        "sample_counts_by_modality": dict(sorted(per_modality_sample_counts.items())),
        "evidence_type": EVIDENCE_TYPE,
        "min_voxels": int(min_voxels),
        "min_samples_per_modality": int(min_samples_per_modality),
        "sampling": {
            "applied": False,
            "strategy": "none",
            "reason": "manifest_sampled_equals_manifest_full",
        },
        "notes": [
            "This is an organ-level manifest built from the Phase 1 TotalSegmenter CT and TotalSegmenterMRI full manifests.",
            "Each sample represents one organ present in one case.",
            "The anchor surface is cohort-level same-organ CT-MR evidence, not paired same-case or same-patient evidence.",
        ],
    }
    return full_manifest, sampled_manifest, manifest_meta


def main() -> None:
    """Parse CLI arguments, build the anchor manifests, and write them to disk."""
    parser = argparse.ArgumentParser(
        description="Build the Phase 2 TotalSegmenter CT plus TotalSegmenterMRI anchor manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "How it works:\n"
            "  1. Read the Phase 1 full manifests for TotalSegmenter CT and MRI.\n"
            "  2. Resolve shared-organ presence for each case.\n"
            "  3. Keep only organs with enough support on both modalities.\n"
            "  4. Write organ-level Phase 2 manifest artifacts under:\n"
            f"     {get_phase2_manifest_dir(ANCHOR_DATASET_NAME, ANCHOR_MANIFEST_VARIANT)}\n\n"
            "Examples:\n"
            "  python build_phase2_anchor_manifest.py\n"
            "  python build_phase2_anchor_manifest.py --min-voxels 500 --min-samples-per-modality 20"
        ),
    )
    parser.add_argument("--min-voxels", type=int, default=100, help="Minimum foreground voxels for organ presence")
    parser.add_argument(
        "--min-samples-per-modality",
        type=int,
        default=5,
        help="Minimum case support per organ and modality for retention",
    )
    args = parser.parse_args()

    manifest_dir = get_phase2_manifest_dir(ANCHOR_DATASET_NAME, ANCHOR_MANIFEST_VARIANT)
    logger.info("Starting Phase 2 anchor manifest build")
    logger.info("Output directory: %s", manifest_dir)
    logger.info("Min voxels per organ: %d", args.min_voxels)
    logger.info("Min cases per modality: %d", args.min_samples_per_modality)
    full_manifest, sampled_manifest, manifest_meta = build_anchor_manifest(
        min_voxels=args.min_voxels,
        min_samples_per_modality=args.min_samples_per_modality,
    )

    _write_json(manifest_dir / "manifest_full.json", full_manifest)
    _write_json(manifest_dir / "manifest_sampled.json", sampled_manifest)
    _write_json(manifest_dir / "manifest_meta.json", manifest_meta)
    logger.info("Wrote Phase 2 anchor manifests to %s", manifest_dir)
    logger.info("Retained organs: %s", ", ".join(full_manifest["retained_organs"]))
    logger.info("Total organ-level samples: %d", len(full_manifest["samples"]))


if __name__ == "__main__":
    main()