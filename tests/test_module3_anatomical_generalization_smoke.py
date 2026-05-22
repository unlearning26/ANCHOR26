"""Smoke validation for the Module 3 package.

This test verifies the implemented Module 3 surface rather than benchmark
scientific validity. It covers:

    1. Python CLI parsing for the builder and evaluator
    2. shell syntax of the Module 3 launchers
    3. toy manifest conversion from Phase 2 to Module 3
    4. toy leave-one-organ-out retrieval with expected high scores
    5. toy CT-only within-modality leave-one-organ-out retrieval
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE3_ROOT = REPO_ROOT / "eval_downstream" / "medfm_eval" / "module3_anatomical_generalization"
SCRIPTS_DIR = MODULE3_ROOT / "scripts"
PYTHON_BIN = Path(os.environ.get("PYTHON", sys.executable))


def _run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run one subprocess and capture stdout and stderr for assertions."""
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        capture_output=True,
        text=True,
    )


def _build_toy_source_manifest(
    path: Path,
    dataset_name: str = "toy_ct_mr",
    modalities: tuple[str, ...] = ("ct", "mr"),
    cases_per_organ: int = 3,
) -> dict:
    """Create a small synthetic Phase 2-style manifest for smoke validation."""
    organs = ["aorta", "liver", "spleen"]
    samples = []
    for organ_index, organ in enumerate(organs):
        for modality_index, modality in enumerate(modalities):
            for case_index in range(cases_per_organ):
                samples.append(
                    {
                        "dataset": dataset_name,
                        "evidence_type": "cohort_level_same_organ_ct_mr" if len(modalities) > 1 else "within_dataset_multi_organ",
                        "file_path": f"/tmp/toy/{modality}/{organ}/{case_index}.nii.gz",
                        "mask_path": f"/tmp/toy/{modality}/{organ}/{case_index}_mask.nii.gz",
                        "modality": modality,
                        "organs": [organ],
                        "patient_id": f"{modality}_{case_index:02d}",
                        "primary_organ": organ,
                        "sample_id": f"{modality}:{organ}:{case_index}",
                        "source_case_id": f"{modality}_{case_index:02d}",
                        "spacing": [1.0, 1.0, 1.0 + 0.1 * modality_index],
                        "shape": [32, 32, 32],
                        "label_voxels": 1000 + 25 * organ_index + case_index,
                    }
                )
    payload = {
        "dataset": dataset_name,
        "description": "Toy Phase 2 manifest for Module 3 smoke validation",
        "evidence_type": "cohort_level_same_organ_ct_mr" if len(modalities) > 1 else "within_dataset_multi_organ",
        "manifest_variant": "core",
        "phase": "phase2",
        "retained_organs": organs,
        "samples": samples,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _build_toy_embeddings(path: Path, sample_ids: list[str]) -> None:
    """Create separable synthetic embeddings for the toy Module 3 evaluation."""
    organ_centers = {
        "aorta": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "liver": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        "spleen": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    }
    modality_shift = {
        "ct": np.asarray([0.05, 0.00, 0.00], dtype=np.float32),
        "mr": np.asarray([0.00, 0.05, 0.00], dtype=np.float32),
        "pet": np.asarray([0.00, 0.00, 0.05], dtype=np.float32),
    }
    features = []
    for sample_id in sample_ids:
        modality, organ, case_index = sample_id.split(":")
        jitter = 0.01 * float(case_index)
        feature = organ_centers[organ] + modality_shift[modality] + np.asarray([jitter, -jitter, jitter], dtype=np.float32)
        features.append(feature)
    np.savez(path, features=np.stack(features, axis=0), sample_ids=np.asarray(sample_ids, dtype=object))


def main() -> None:
    """Run the end-to-end Module 3 smoke suite."""
    print("[1/5] Validating Module 3 Python CLIs")
    _run([str(PYTHON_BIN), str(MODULE3_ROOT / "build_module3_manifest.py"), "--help"], cwd=MODULE3_ROOT)
    _run([str(PYTHON_BIN), str(MODULE3_ROOT / "module3_evaluation_pipeline.py"), "--help"], cwd=MODULE3_ROOT)

    print("[2/5] Validating Module 3 shell launchers")
    for script_name in [
        "module3_dataset_presets.sh",
        "run_module3_build_manifest.sh",
        "run_module3_anatomical_generalization.sh",
    ]:
        _run(["bash", "-n", str(SCRIPTS_DIR / script_name)], cwd=REPO_ROOT)

    with tempfile.TemporaryDirectory(prefix="module3_smoke_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        source_manifest = tmp_dir / "phase2_manifest_sampled.json"
        module3_manifest_dir = tmp_dir / "module3_manifest"
        output_json = tmp_dir / "module3_result.json"
        embeddings_npz = tmp_dir / "phase2_organ_cls_embeddings.npz"

        print("[3/5] Building toy Module 3 manifest")
        source_payload = _build_toy_source_manifest(source_manifest)
        build_result = _run(
            [
                str(PYTHON_BIN),
                str(MODULE3_ROOT / "build_module3_manifest.py"),
                "--source-manifest",
                str(source_manifest),
                "--analysis-name",
                "toy_ct_mr",
                "--manifest-variant",
                "core",
                "--min-holdout-organs",
                "3",
                "--min-samples-per-modality",
                "2",
                "--output-dir",
                str(module3_manifest_dir),
            ],
            cwd=MODULE3_ROOT,
        )
        assert "Eligible hold-out organs (3)" in build_result.stdout

        built_manifest = json.loads((module3_manifest_dir / "manifest_sampled.json").read_text(encoding="utf-8"))
        assert built_manifest["phase"] == "module3"
        assert sorted(built_manifest["eligible_holdout_organs"]) == sorted(source_payload["retained_organs"])
        assert built_manifest["module3_protocol"]["held_out_definition"] == "held_out_during_adaptation"
        first_sample = built_manifest["samples"][0]
        assert first_sample["organ_name"] in source_payload["retained_organs"]
        assert first_sample["image_path"]
        assert first_sample["source_dataset"] == "toy_ct_mr"
        assert first_sample["pretraining_exposure"] == "pretraining_exposure_unknown"

        print("[4/5] Evaluating toy leave-one-organ-out transfer")
        sample_ids = [str(sample["sample_id"]) for sample in built_manifest["samples"]]
        _build_toy_embeddings(embeddings_npz, sample_ids)
        _run(
            [
                str(PYTHON_BIN),
                str(MODULE3_ROOT / "module3_evaluation_pipeline.py"),
                "--manifest",
                str(module3_manifest_dir / "manifest_sampled.json"),
                "--analysis-name",
                "toy_ct_mr",
                "--checkpoint-name",
                "toy_ckpt",
                "--feature-type",
                "cls",
                "--embeddings-npz",
                str(embeddings_npz),
                "--min-holdout-organs",
                "3",
                "--min-samples-per-modality",
                "2",
                "--top-ks",
                "1",
                "3",
                "--output-path",
                str(output_json),
            ],
            cwd=MODULE3_ROOT,
        )

        payload = json.loads(output_json.read_text(encoding="utf-8"))
        module3_metrics = payload["surface_a_feature_space_evidence"]["leave_one_organ_out_analysis"]
        assert module3_metrics["status"] == "ok"
        assert module3_metrics["n_evaluated_organs"] == 3
        assert payload["module3_protocol"]["held_out_definition"] == "held_out_during_adaptation"
        assert payload["surface_a_feature_space_evidence"]["metric_aliases"]["top1_matching_accuracy"] > 0.95
        assert payload["surface_a_feature_space_evidence"]["diagnostics"]["retrieval_map"] > 0.95
        assert payload["surface_a_feature_space_evidence"]["diagnostics"]["heldout_centroid_distance"] < 0.02
        assert payload["surface_a_feature_space_evidence"]["diagnostics"]["heldout_silhouette"] > 0.5
        assert payload["surface_a_feature_space_evidence"]["diagnostics"]["nearest_neighbor_purity"] > 0.9
        assert payload["surface_a_feature_space_evidence"]["paper_role"] == "diagnostic_feature_space_analysis_not_headline"
        assert payload["headline_metrics"]["ba_centroid_recoverability"]["metric_name"] == "nearest_centroid_balanced_accuracy"
        assert payload["headline_metrics"]["ba_centroid_recoverability"]["metric_value"] > 0.95
        assert payload["headline_metrics"]["ba_probe_recoverability"]["metric_name"] == "balanced_accuracy"
        assert payload["headline_metrics"]["ba_probe_recoverability"]["metric_value"] > 0.95
        assert payload["headline_metrics"]["adaptation_gain_over_centroid"]["metric_name"] == "adaptation_gain_balanced_accuracy_over_nearest_centroid"
        assert payload["headline_metrics"]["adaptation_gain_over_centroid"]["adapted_metric_value"] > 0.95
        assert payload["headline_metrics"]["adaptation_gain_over_centroid"]["baseline_metric_value"] > 0.95
        assert module3_metrics["macro_mean"]["top@1"] > 0.95
        assert module3_metrics["macro_mean"]["map"] > 0.95
        surface_b = payload["surface_b_few_shot_transfer_evidence"]
        assert surface_b["status"] == "ok"
        assert surface_b["metric_aliases"]["balanced_accuracy"] > 0.95
        assert surface_b["metric_aliases"]["nearest_centroid_baseline_balanced_accuracy"] > 0.95
        assert surface_b["metric_aliases"]["adaptation_gain_over_nearest_centroid"] is not None
        assert surface_b["diagnostics"]["transfer_efficiency"] > 0.95
        assert "seed_std_adaptation_gain_balanced_accuracy" in surface_b["diagnostics"]
        assert payload["surface_b_few_shot_transfer_evidence"]["paper_role"] == "headline_recoverability_surface"
        for organ in ["aorta", "liver", "spleen"]:
            organ_payload = module3_metrics["per_organ"][organ]
            assert organ_payload["bidirectional_mean"]["top@1"] > 0.95
            assert "generalization_gap" in organ_payload
            assert organ_payload["fold_context"]["held_out_during_adaptation"] == organ
            assert organ_payload["bidirectional_mean"]["nearest_neighbor_purity"] > 0.9
            assert organ_payload["bidirectional_mean"]["heldout_silhouette"] > 0.4

            surface_b_organ = surface_b["few_shot_transfer_analysis"]["per_organ"][organ]
            assert surface_b_organ["status"] == "ok"
            assert surface_b_organ["fewshot_probe_score"] > 0.95
            assert surface_b_organ["balanced_accuracy"] > 0.95
            assert surface_b_organ["nearest_centroid_balanced_accuracy"] > 0.95
            assert surface_b_organ["adaptation_gain_balanced_accuracy"] is not None
            assert surface_b_organ["transfer_efficiency"] > 0.95
            assert surface_b_organ["seed_results"][0]["nearest_centroid_balanced_accuracy"] > 0.95
            assert "adaptation_gain_balanced_accuracy" in surface_b_organ["seed_results"][0]

        print("[5/5] Evaluating toy CT-only within-modality transfer")
        ct_only_source_manifest = tmp_dir / "phase2_manifest_sampled_ct_only.json"
        ct_only_module3_manifest_dir = tmp_dir / "module3_manifest_ct_only"
        ct_only_output_json = tmp_dir / "module3_result_ct_only.json"
        ct_only_embeddings_npz = tmp_dir / "phase2_organ_cls_embeddings_ct_only.npz"
        
        ct_only_source_payload = _build_toy_source_manifest(
        ct_only_source_manifest,
        dataset_name="toy_ct_only",
        modalities=("ct",),
        cases_per_organ=4,
        )
        build_ct_only_result = _run(
        [
        str(PYTHON_BIN),
        str(MODULE3_ROOT / "build_module3_manifest.py"),
        "--source-manifest",
        str(ct_only_source_manifest),
        "--analysis-name",
                "toy_ct_only",
                "--manifest-variant",
                "core",
                "--required-modalities",
                "ct",
                "--min-holdout-organs",
                "3",
                "--min-samples-per-modality",
                "2",
                "--output-dir",
                str(ct_only_module3_manifest_dir),
            ],
            cwd=MODULE3_ROOT,
        )
        assert "Eligible hold-out organs (3)" in build_ct_only_result.stdout

        built_ct_only_manifest = json.loads((ct_only_module3_manifest_dir / "manifest_sampled.json").read_text(encoding="utf-8"))
        assert built_ct_only_manifest["module3_protocol"]["surface_scope"] == "within_modality"
        assert built_ct_only_manifest["module3_protocol"]["within_modality_surface_stratum"] == "ct_within_dataset_holdout_surface"
        assert built_ct_only_manifest["module3_protocol"]["cross_modality_surface_stratum"] is None
        assert sorted(built_ct_only_manifest["eligible_holdout_organs"]) == sorted(ct_only_source_payload["retained_organs"])

        ct_only_sample_ids = [str(sample["sample_id"]) for sample in built_ct_only_manifest["samples"]]
        _build_toy_embeddings(ct_only_embeddings_npz, ct_only_sample_ids)
        _run(
            [
                str(PYTHON_BIN),
                str(MODULE3_ROOT / "module3_evaluation_pipeline.py"),
                "--manifest",
                str(ct_only_module3_manifest_dir / "manifest_sampled.json"),
                "--analysis-name",
                "toy_ct_only",
                "--checkpoint-name",
                "toy_ckpt",
                "--feature-type",
                "cls",
                "--embeddings-npz",
                str(ct_only_embeddings_npz),
                "--required-modalities",
                "ct",
                "--min-holdout-organs",
                "3",
                "--min-samples-per-modality",
                "2",
                "--top-ks",
                "1",
                "3",
                "--output-path",
                str(ct_only_output_json),
            ],
            cwd=MODULE3_ROOT,
        )

        ct_only_payload = json.loads(ct_only_output_json.read_text(encoding="utf-8"))
        ct_only_metrics = ct_only_payload["surface_a_feature_space_evidence"]["leave_one_organ_out_analysis"]
        assert ct_only_payload["module3_protocol"]["surface_scope"] == "within_modality"
        assert ct_only_payload["module3_protocol"]["within_modality_surface_stratum"] == "ct_within_dataset_holdout_surface"
        assert ct_only_metrics["surface_scope"] == "within_modality"
        assert ct_only_metrics["required_modalities"] == ["ct"]
        assert ct_only_payload["surface_a_feature_space_evidence"]["metric_aliases"]["top1_matching_accuracy"] > 0.95
        assert ct_only_payload["surface_a_feature_space_evidence"]["diagnostics"]["retrieval_map"] > 0.95
        assert ct_only_payload["surface_a_feature_space_evidence"]["diagnostics"]["heldout_centroid_distance"] < 0.02
        assert ct_only_payload["surface_a_feature_space_evidence"]["diagnostics"]["heldout_silhouette"] > 0.5
        assert ct_only_payload["surface_a_feature_space_evidence"]["diagnostics"]["nearest_neighbor_purity"] >= 0.6
        ct_only_surface_b = ct_only_payload["surface_b_few_shot_transfer_evidence"]
        assert ct_only_surface_b["surface_scope"] == "within_modality"
        assert ct_only_surface_b["few_shot_transfer_analysis"]["transfer_protocol"] == "fixed_budget_few_shot_transfer_case_disjoint"
        assert ct_only_surface_b["metric_aliases"]["balanced_accuracy"] > 0.95
        assert ct_only_surface_b["metric_aliases"]["nearest_centroid_baseline_balanced_accuracy"] > 0.95
        assert ct_only_payload["headline_metrics"]["ba_centroid_recoverability"]["metric_value"] is not None
        assert ct_only_payload["headline_metrics"]["ba_probe_recoverability"]["metric_value"] is not None

    print("[5/5] All Module 3 anatomical generalization smoke tests passed.")


if __name__ == "__main__":
    main()