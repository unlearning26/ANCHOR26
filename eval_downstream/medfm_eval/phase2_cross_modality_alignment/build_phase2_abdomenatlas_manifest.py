#!/usr/bin/env python

"""Build the Phase 2 AbdomenAtlas organ-level manifest.

This materializes a within-modality upstream surface for Module 3 from the
existing Phase 1 AbdomenAtlas sampled manifest plus the cached multi-organ
presence labels already computed during Phase 1.

What it does:
1. loads the Phase 1 sampled AbdomenAtlas CT manifest,
2. loads the cached integer-label organ volumes for the same sampled cases,
3. explodes each case into organ-level Phase 2 samples,
4. writes a Phase 2-style manifest namespace under
   data_manifests/phase2_cross_modality_alignment/abdomenatlas/core/.

Notes:
- This is intentionally a CT-only, within-modality upstream surface.
- Because the available semantic cache is currently sampled-surface only,
  manifest_full.json and manifest_sampled.json are identical by design and the
  metadata records that sampled-only materialization explicitly.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from phase2_config import get_phase2_manifest_dir


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE1_MANIFEST = (
    PROJECT_ROOT
    / "eval_downstream"
    / "medfm_eval"
    / "data_manifests"
    / "phase1_anisotropy_robustness"
    / "abdomenatlas"
    / "original_bins"
    / "manifest_sampled.json"
)
PHASE1_SEMANTIC_CACHE = (
    PROJECT_ROOT
    / "eval_downstream"
    / "medfm_eval"
    / "phase1_anisotropy_robustness"
    / "outputs_phase1"
    / "abdomenatlas"
    / "phase1"
    / "original_bins"
    / "features"
    / "semantic_labels_abdomenatlas_multi_organ_presence_abdomenatlas.json"
)

ABDOMENATLAS_ORGANS: Dict[int, str] = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_right",
    9: "adrenal_left",
    10: "lung_upper_left",
    11: "lung_lower_left",
    12: "lung_upper_right",
    13: "lung_middle_right",
    14: "lung_lower_right",
    15: "esophagus",
    16: "trachea",
    17: "thyroid",
    18: "small_bowel",
    19: "duodenum",
    20: "colon",
    21: "urinary_bladder",
    22: "prostate",
    23: "kidney_cyst_left",
    24: "kidney_cyst_right",
    25: "aorta",
}

EVIDENCE_TYPE = "cohort_level_same_organ_ct"


def _load_json(path: Path) -> Dict[str, Any] | List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_phase1_volumes(path: Path) -> List[Dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, dict):
        return list(payload.get("volumes") or payload.get("samples") or [])
    return list(payload)


def _resolve_case_id(volume: Dict[str, Any]) -> str:
    explicit = volume.get("source_case_id") or volume.get("patient_id") or volume.get("case_id")
    if explicit:
        return str(explicit)
    image_path = volume.get("file_path") or volume.get("image")
    if image_path:
        return Path(str(image_path)).resolve().parent.name
    raise ValueError(f"Unable to resolve case id for volume: {volume}")


def _select_case_subset(
    volumes: List[Dict[str, Any]],
    subset_case_count: int | None,
    subset_seed: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if subset_case_count is None:
        return volumes, {
            "applied": False,
            "strategy": "none",
            "subset_seed": subset_seed,
            "requested_case_count": None,
            "selected_case_count": len(volumes),
        }

    if subset_case_count <= 0:
        raise ValueError(f"subset_case_count must be positive when provided, got {subset_case_count}")

    indexed_volumes = sorted(
        [(_resolve_case_id(volume), volume) for volume in volumes],
        key=lambda item: item[0],
    )
    if subset_case_count >= len(indexed_volumes):
        return [volume for _, volume in indexed_volumes], {
            "applied": False,
            "strategy": "requested_at_least_full_surface",
            "subset_seed": subset_seed,
            "requested_case_count": subset_case_count,
            "selected_case_count": len(indexed_volumes),
        }

    rng = random.Random(subset_seed)
    selected_case_ids = set(rng.sample([case_id for case_id, _ in indexed_volumes], subset_case_count))
    selected_volumes = [volume for case_id, volume in indexed_volumes if case_id in selected_case_ids]
    selected_case_ids_sorted = sorted(selected_case_ids)
    return selected_volumes, {
        "applied": True,
        "strategy": "deterministic_case_level_subsample",
        "subset_seed": subset_seed,
        "requested_case_count": subset_case_count,
        "selected_case_count": len(selected_volumes),
        "selected_case_ids": selected_case_ids_sorted,
    }


def _load_semantic_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected semantic cache dict payload, got {type(payload)}")
    return payload


def _build_samples(
    volumes: List[Dict[str, Any]],
    semantic_cache: Dict[str, Dict[str, Any]],
    min_voxels: int,
    min_samples_per_organ: int,
) -> tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, Any]]:
    preliminary_samples: List[Dict[str, Any]] = []
    support_counter: Counter[str] = Counter()
    missing_cache_entries: List[str] = []

    for volume in volumes:
        image_path = str(Path(volume["file_path"]).resolve())
        label_path = str(Path(volume["label_path"]).resolve())
        semantic_entry = semantic_cache.get(image_path)
        if semantic_entry is None:
            missing_cache_entries.append(image_path)
            continue

        case_id = Path(image_path).parent.name
        organ_volumes = semantic_entry.get("organ_volumes") or {}
        for label_id, organ_name in ABDOMENATLAS_ORGANS.items():
            voxel_count = int(organ_volumes.get(str(label_id), 0) or 0)
            if voxel_count < min_voxels:
                continue
            support_counter[organ_name] += 1
            preliminary_samples.append(
                {
                    "sample_id": f"ct:{case_id}:{organ_name}",
                    "file_path": image_path,
                    "mask_path": label_path,
                    "label_path": label_path,
                    "dataset": "abdomenatlas",
                    "modality": "ct",
                    "primary_organ": organ_name,
                    "organs": [organ_name],
                    "patient_id": case_id,
                    "source_case_id": case_id,
                    "evidence_type": EVIDENCE_TYPE,
                    "shape": volume.get("shape"),
                    "spacing": volume.get("spacing"),
                    "label_voxels": voxel_count,
                    "mask_label_value": label_id,
                    "mask_loader": "integer_label_mask",
                }
            )

    retained_organs = {
        organ_name for organ_name, count in support_counter.items() if count >= min_samples_per_organ
    }
    filtered_samples = [
        sample for sample in preliminary_samples if sample["primary_organ"] in retained_organs
    ]
    organ_support = {
        organ_name: support_counter[organ_name] for organ_name in sorted(retained_organs)
    }
    diagnostics = {
        "missing_cache_entries": missing_cache_entries,
        "dropped_organs": {
            organ_name: count
            for organ_name, count in sorted(support_counter.items())
            if organ_name not in retained_organs
        },
    }
    return (
        sorted(filtered_samples, key=lambda sample: (sample["primary_organ"], sample["source_case_id"])),
        organ_support,
        diagnostics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the AbdomenAtlas Phase 2 organ-level manifest")
    parser.add_argument(
        "--phase1-manifest",
        default=str(PHASE1_MANIFEST),
        help="Path to the Phase 1 sampled AbdomenAtlas manifest",
    )
    parser.add_argument(
        "--semantic-cache",
        default=str(PHASE1_SEMANTIC_CACHE),
        help="Path to the cached AbdomenAtlas semantic-label presence JSON",
    )
    parser.add_argument(
        "--analysis-name",
        default="abdomenatlas",
        help="Phase 2 dataset namespace to materialize",
    )
    parser.add_argument(
        "--manifest-variant",
        default="core",
        help="Phase 2 manifest variant to materialize",
    )
    parser.add_argument(
        "--min-voxels",
        type=int,
        default=100,
        help="Minimum organ voxels required to keep an organ instance",
    )
    parser.add_argument(
        "--min-samples-per-organ",
        type=int,
        default=20,
        help="Minimum number of CT cases required to retain an organ on the surface",
    )
    parser.add_argument(
        "--subset-case-count",
        type=int,
        default=None,
        help="Optional deterministic case-level subset size before organ explosion",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=42,
        help="Random seed used for deterministic case-level subsetting",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output manifest directory override",
    )
    args = parser.parse_args()

    phase1_manifest_path = Path(args.phase1_manifest).resolve()
    semantic_cache_path = Path(args.semantic_cache).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else get_phase2_manifest_dir(args.analysis_name, args.manifest_variant)
    )

    logger.info("Loading Phase 1 manifest from %s", phase1_manifest_path)
    volumes = _load_phase1_volumes(phase1_manifest_path)
    logger.info("Loading semantic cache from %s", semantic_cache_path)
    semantic_cache = _load_semantic_cache(semantic_cache_path)
    selected_volumes, selection = _select_case_subset(
        volumes,
        subset_case_count=args.subset_case_count,
        subset_seed=args.subset_seed,
    )
    if selection["applied"]:
        logger.info(
            "Using deterministic AbdomenAtlas subset: requested=%d selected=%d seed=%d",
            selection["requested_case_count"],
            selection["selected_case_count"],
            selection["subset_seed"],
        )

    samples, organ_support, diagnostics = _build_samples(
        selected_volumes,
        semantic_cache,
        min_voxels=args.min_voxels,
        min_samples_per_organ=args.min_samples_per_organ,
    )

    materialization_scope = "sampled_subset" if selection["applied"] else "sampled_only"
    description = "Phase 2 AbdomenAtlas CT-only organ-level manifest"
    if selection["applied"]:
        description += f" (deterministic {selection['selected_case_count']}-case subset materialization)"
    else:
        description += " (sampled-source materialization)"

    notes = [
        "AbdomenAtlas is materialized as a CT-only upstream Phase 2 surface for within-modality Module 3.",
        "Phase 2 cross-modal metrics are not canonical for this namespace; the primary purpose is organ feature extraction for Module 3.",
    ]
    if selection["applied"]:
        notes.extend(
            [
                f"This manifest variant is a deterministic case-level subset of the sampled AbdomenAtlas surface using seed={selection['subset_seed']} and requested_case_count={selection['requested_case_count']}.",
                "manifest_full.json and manifest_sampled.json are identical because this variant is materialized directly as one deterministic organ-level subset surface.",
            ]
        )
    else:
        notes.append(
            "manifest_full.json and manifest_sampled.json are identical because the currently available semantic cache is sampled-surface only."
        )

    payload = {
        "version": "1.0",
        "description": description,
        "phase": "phase2",
        "dataset": args.analysis_name,
        "modality": "ct",
        "surface_scope": "within_modality",
        "evidence_type": EVIDENCE_TYPE,
        "required_modalities": ["ct"],
        "materialization_scope": materialization_scope,
        "min_voxels": args.min_voxels,
        "min_samples_per_organ": args.min_samples_per_organ,
        "selection": selection,
        "retained_organs": sorted(organ_support),
        "organ_support": organ_support,
        "total_cases": len(selected_volumes),
        "total_samples": len(samples),
        "samples": samples,
    }
    meta = {
        "dataset": args.analysis_name,
        "phase": "phase2",
        "manifest_variant": args.manifest_variant,
        "modality": "ct",
        "surface_scope": "within_modality",
        "required_modalities": ["ct"],
        "evidence_type": EVIDENCE_TYPE,
        "source_phase1_manifest": str(phase1_manifest_path),
        "source_phase1_semantic_cache": str(semantic_cache_path),
        "source_phase1_manifest_kind": "sampled",
        "materialization_scope": materialization_scope,
        "full_manifest": "manifest_full.json",
        "sampled_manifest": "manifest_sampled.json",
        "selection": selection,
        "total_cases": len(selected_volumes),
        "total_samples": len(samples),
        "min_voxels": args.min_voxels,
        "min_samples_per_organ": args.min_samples_per_organ,
        "retained_organs": sorted(organ_support),
        "organ_support": organ_support,
        "diagnostics": diagnostics,
        "notes": notes,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "manifest_full.json", payload)
    _write_json(output_dir / "manifest_sampled.json", payload)
    _write_json(output_dir / "manifest_meta.json", meta)
    logger.info("Wrote AbdomenAtlas Phase 2 manifests to %s", output_dir)


if __name__ == "__main__":
    main()