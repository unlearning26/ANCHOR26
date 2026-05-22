#!/usr/bin/env python

"""Build a Module 3 manifest from an existing upstream organ manifest.

This script narrows an upstream organ-level manifest to the subset of organs
that are eligible for Module 3 leave-one-organ-out analysis.

The upstream manifest is often stored under the Phase 2 namespace because that
package already owns the canonical organ-level embedding extraction surface.
That storage lineage is an implementation convenience, not a scientific claim
that the source dataset is itself valid for Phase 2 paired CT-MR alignment.

Required parameters:
    --source-manifest: path to the upstream organ manifest.

Strongly recommended parameters:
    --analysis-name: dataset namespace to use for the Module 3 manifest root.
    --manifest-variant: usually ``core`` when the source path is ambiguous.

Examples:
    python build_module3_manifest.py \
        --source-manifest ../data_manifests/phase2_cross_modality_alignment/totalsegmenter_ct_mr_anchor/core/manifest_sampled.json \
        --analysis-name totalsegmenter_ct_mr_anchor \
        --manifest-variant core

    python build_module3_manifest.py \
        --source-manifest ../data_manifests/phase2_cross_modality_alignment/mmwhs_ct_mr/core/manifest_sampled.json \
        --analysis-name mmwhs_ct_mr \
        --manifest-variant core \
        --min-holdout-organs 2 \
        --min-samples-per-modality 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from config import (
    DEFAULT_MIN_HOLDOUT_ORGANS,
    DEFAULT_MIN_SAMPLES_PER_MODALITY,
    DEFAULT_REQUIRED_MODALITIES,
    get_dataset_name_from_manifest_path,
    get_manifest_variant_from_manifest_path,
    get_module3_manifest_dir,
)
from module3_data_loader import (
    compute_organ_modality_support,
    get_eligible_holdout_organs,
    load_module3_manifest,
    summarize_module3_manifest,
    validate_module3_manifest,
)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write one JSON payload with canonical formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_manifest_payload(path: Path) -> Dict[str, Any]:
    """Load a source manifest while accepting either dict or list payloads."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return {"samples": payload}


def _normalize_required_modalities(values: List[str]) -> List[str]:
    """Normalize a CLI-provided list of required modalities."""
    return [str(value).strip().lower() for value in values if str(value).strip()]


def _build_holdout_eligibility_rule(
    required_modalities: List[str],
    min_samples_per_modality: int,
) -> Dict[str, Any]:
    """Serialize the exact Module 3 organ-holdout eligibility rule.

    "eligible hold-out organ" means an organ that can serve as the held-out
    target in at least one leave-one-organ-out fold under the selected modality
    constraint. It does not mean all listed organs are withheld simultaneously.
    """
    return {
        "required_modalities": list(required_modalities),
        "min_samples_per_modality": int(min_samples_per_modality),
        "definition": (
            "An organ is hold-out eligible iff it has at least min_samples_per_modality samples "
            "in every required modality. Eligibility is defined per fold: each listed organ can be "
            "withheld as the target organ in a separate leave-one-organ-out analysis."
        ),
        "within_modality_note": (
            "For single-modality surfaces such as CT-only AbdomenAtlas, eligibility reduces to sufficient "
            "support in that one modality plus case-disjoint splitting inside the evaluator."
        ),
    }


def _infer_surface_stratum(samples: List[Dict[str, Any]], required_modalities: List[str]) -> str:
    if len(required_modalities) == 1:
        return f"{required_modalities[0]}_within_dataset_holdout_surface"
    exposures = {sample.get("pretraining_exposure") for sample in samples if sample.get("pretraining_exposure")}
    if exposures == {"pretraining_unseen_source"}:
        return "pretraining_unseen_ct_mr_validation"
    if "pretraining_seen_source" in exposures and "pretraining_unseen_source" in exposures:
        return "mixed_exposure_ct_mr_anchor"
    if exposures == {"pretraining_seen_source"}:
        return "pretraining_seen_ct_mr_surface"
    return "mixed_or_unknown_exposure_surface"


def _build_serialized_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    raw_sample = sample.get("raw_sample") or {}
    return {
        "sample_id": sample.get("sample_id"),
        "patient_id": sample.get("patient_id"),
        "modality": sample.get("modality"),
        "organ_name": sample.get("organ_name") or sample.get("primary_organ"),
        "organ_family": sample.get("organ_family"),
        "organ_status": sample.get("organ_status") or "holdout_candidate",
        "source_dataset": sample.get("source_dataset") or sample.get("dataset"),
        "pretraining_exposure": sample.get("pretraining_exposure"),
        "evidence_type": sample.get("evidence_type"),
        "image_path": sample.get("image_path") or sample.get("file_path"),
        "mask_path": sample.get("mask_path"),
        "mask_metadata": sample.get("mask_metadata"),
        "source_case_id": sample.get("source_case_id"),
        "split": sample.get("split"),
        "organs": sample.get("organs"),
        "primary_organ": sample.get("primary_organ"),
        "dataset": raw_sample.get("dataset", sample.get("dataset")),
        "file_path": raw_sample.get("file_path", sample.get("file_path")),
        "label_path": raw_sample.get("label_path"),
        "label_voxels": raw_sample.get("label_voxels"),
        "mask_path_legacy": raw_sample.get("mask_path"),
        "shape": raw_sample.get("shape"),
        "spacing": raw_sample.get("spacing"),
    }


def main() -> None:
    """CLI entrypoint for Module 3 manifest construction.

    Required CLI parameter:
        --source-manifest

    The command fails when:
        -- required modalities are absent from the source manifest
                - fewer than ``--min-holdout-organs`` organs satisfy the required-modality
          support threshold
    """
    parser = argparse.ArgumentParser(description="Build the Module 3 anatomical generalization manifest")
    parser.add_argument("-m", "--source-manifest", required=True, help="Path to the upstream organ manifest JSON")
    parser.add_argument("-a", "--analysis-name", default=None, help="Dataset namespace for the output manifest")
    parser.add_argument(
        "--manifest-variant",
        default=None,
        help="Manifest variant label; defaults to the variant inferred from the source manifest",
    )
    parser.add_argument(
        "--required-modalities",
        nargs="+",
        default=list(DEFAULT_REQUIRED_MODALITIES),
        help="Required modalities for Module 3 anatomical hold-out support",
    )
    parser.add_argument(
        "--min-holdout-organs",
        type=int,
        default=DEFAULT_MIN_HOLDOUT_ORGANS,
        help="Minimum number of organs required under the selected modality-support rule",
    )
    parser.add_argument(
        "--min-samples-per-modality",
        type=int,
        default=DEFAULT_MIN_SAMPLES_PER_MODALITY,
        help="Minimum support per organ and modality for hold-out eligibility",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory. Defaults to the canonical Module 3 manifest namespace.",
    )
    args = parser.parse_args()

    source_manifest = Path(args.source_manifest).resolve()
    analysis_name = args.analysis_name or get_dataset_name_from_manifest_path(source_manifest)
    manifest_variant = args.manifest_variant or get_manifest_variant_from_manifest_path(source_manifest)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else get_module3_manifest_dir(analysis_name, manifest_variant)
    required_modalities = _normalize_required_modalities(args.required_modalities)

    source_payload = _load_manifest_payload(source_manifest)
    source_samples = load_module3_manifest(source_manifest)
    validation = validate_module3_manifest(
        source_samples,
        required_modalities=required_modalities,
        min_holdout_organs=args.min_holdout_organs,
        min_samples_per_modality=args.min_samples_per_modality,
    )
    if validation["missing_modalities"]:
        raise ValueError(
            f"Source manifest {source_manifest} is missing required modalities: {validation['missing_modalities']}"
        )

    eligible_holdout_organs = get_eligible_holdout_organs(
        source_samples,
        required_modalities=required_modalities,
        min_samples_per_modality=args.min_samples_per_modality,
    )
    if len(eligible_holdout_organs) < args.min_holdout_organs:
        raise ValueError(
            f"Module 3 requires at least {args.min_holdout_organs} eligible hold-out organs, "
            f"under required modalities {required_modalities}, "
            f"but {source_manifest} provides {len(eligible_holdout_organs)} ({eligible_holdout_organs})"
        )

    organ_support = compute_organ_modality_support(source_samples, required_modalities)
    excluded_organs = {
        organ: {
            "counts": counts,
            "reason": "insufficient_required_modality_holdout_support",
        }
        for organ, counts in organ_support.items()
        if organ not in set(eligible_holdout_organs)
    }

    filtered_samples = [
        sample
        for sample in source_samples
        if (sample.get("organ_name") or sample.get("primary_organ")) in set(eligible_holdout_organs)
    ]
    serialized_samples = [_build_serialized_sample(sample) for sample in filtered_samples]
    source_summary = summarize_module3_manifest(filtered_samples)
    surface_stratum = _infer_surface_stratum(filtered_samples, required_modalities)
    surface_scope = "within_modality" if len(required_modalities) == 1 else "cross_modality"
    holdout_eligibility_rule = _build_holdout_eligibility_rule(
        required_modalities=required_modalities,
        min_samples_per_modality=args.min_samples_per_modality,
    )
    module3_protocol = {
        "held_out_definition": "held_out_during_adaptation",
        "held_out_definition_note": (
            "Held-out organ refers to an organ class withheld during the Module 3 adaptation or probe-fitting fold, "
            "not merely an organ from a dataset unseen during encoder pretraining."
        ),
        "fold_policy": "leave_one_organ_out",
        "holdout_eligibility_rule": holdout_eligibility_rule,
        "within_modality_policy": "within_dataset_seen_during_adaptation_vs_held_out_during_adaptation",
        "surface_scope": surface_scope,
        "cross_modality_surface_stratum": surface_stratum if surface_scope == "cross_modality" else None,
        "within_modality_surface_stratum": surface_stratum if surface_scope == "within_modality" else None,
        "checkpoint_policy": "full_canonical_checkpoint_slate",
        "required_modalities": required_modalities,
        "eligible_holdout_organs": eligible_holdout_organs,
    }

    manifest_payload = {
        "dataset": analysis_name,
        "description": (
            "Module 3 canonical manifest: organ-level anatomical generalization surface derived from the existing "
            "upstream organ manifest and prepared for leave-one-organ-out held-out evaluation"
        ),
        "evidence_type": source_payload.get("evidence_type", "cohort_level_same_organ_ct_mr"),
        "manifest_variant": manifest_variant,
        "phase": "module3",
        "required_modalities": required_modalities,
        "module3_protocol": module3_protocol,
        "retained_organs": eligible_holdout_organs,
        "fold_eligible_holdout_organs": eligible_holdout_organs,
        "eligible_holdout_organs": eligible_holdout_organs,
        "eligible_holdout_organs_note": holdout_eligibility_rule["definition"],
        "samples": serialized_samples,
        "upstream_manifest": str(source_manifest),
        "upstream_manifest_phase": source_payload.get("phase"),
        "upstream_manifest_contract": "organ_level_samples",
        "source_phase2_manifest": str(source_manifest) if source_payload.get("phase") == "phase2" else None,
        "notes": [
            "Module 3 is materialized only for datasets with at least two eligible organs under the selected required-modality support rule.",
            "A Phase 2 storage path is lineage metadata only; it does not imply that the source dataset is scientifically valid for Phase 2 paired CT-MR alignment.",
            "This manifest exposes canonical Module 3 fields such as organ_name, source_dataset, image_path, and pretraining_exposure while preserving selected upstream aliases for compatibility.",
            "Held-out organ is defined by the Module 3 leave-one-organ-out adaptation protocol rather than dataset-level pretraining provenance.",
        ],
    }
    meta_payload = {
        "dataset": analysis_name,
        "phase": "module3",
        "manifest_variant": manifest_variant,
        "required_modalities": required_modalities,
        "module3_protocol": module3_protocol,
        "upstream_manifest": str(source_manifest),
        "upstream_manifest_phase": source_payload.get("phase"),
        "upstream_manifest_contract": "organ_level_samples",
        "source_phase2_manifest": str(source_manifest) if source_payload.get("phase") == "phase2" else None,
        "source_phase2_phase": source_payload.get("phase") if source_payload.get("phase") == "phase2" else None,
        "evidence_type": source_payload.get("evidence_type"),
        "fold_eligible_holdout_organs": eligible_holdout_organs,
        "eligible_holdout_organs": eligible_holdout_organs,
        "eligible_holdout_organs_note": holdout_eligibility_rule["definition"],
        "excluded_organs": excluded_organs,
        "support_by_organ": organ_support,
        "min_holdout_organs": int(args.min_holdout_organs),
        "min_samples_per_modality": int(args.min_samples_per_modality),
        "validation": validation,
        "summary": source_summary,
        "total_samples_full": len(serialized_samples),
        "total_samples_sampled": len(serialized_samples),
        "sampling": {
            "applied": False,
            "reason": "manifest_sampled_equals_manifest_full",
            "strategy": "none",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "manifest_full.json", manifest_payload)
    _write_json(output_dir / "manifest_sampled.json", manifest_payload)
    _write_json(output_dir / "manifest_meta.json", meta_payload)

    print(f"Wrote Module 3 manifest to {output_dir}")
    print(f"Eligible hold-out organs ({len(eligible_holdout_organs)}): {', '.join(eligible_holdout_organs)}")


if __name__ == "__main__":
    main()
