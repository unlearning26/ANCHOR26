#!/usr/bin/env python3
"""Synthetic smoke tests for the refactored Phase 1 anisotropy evaluation stack.

This validates the active operational surface before fresh CT reruns:
1. Python CLI entrypoints parse and import with `--help`
2. Shell launchers have valid bash syntax and preset resolution
3. Track A and Track B analyses run on toy feature tensors
4. Setting B resampling and cache helpers run on a synthetic NIfTI volume

Run:
    /path/to/python tests/test_phase1_anisotropy_smoke.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE1_DIR = PROJECT_ROOT / "eval_downstream" / "medfm_eval" / "phase1_anisotropy_robustness"
SCRIPTS_DIR = PHASE1_DIR / "scripts"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PHASE1_DIR))

from anisotropy_semantic_analysis import (  # noqa: E402
    compare_checkpoints_track_b,
    run_multilabel_cross_bin_transfer,
    run_semantic_probing,
    run_track_b_analysis,
)
from config import (  # noqa: E402
    ControlledPerturbationConfig,
    get_cache_root,
    get_controlled_perturbation_config_from_manifest,
    get_output_paths,
)
from phase1_evaluation_pipeline import (  # noqa: E402
    write_results_catalog,
)
from perturbation_robustness_analysis import (  # noqa: E402
    compute_spacing_variant_diagnostics,
    load_from_cache,
    prepare_perturbed_dataset,
    resample_volume,
    save_to_cache,
)
from representation_geometry_analysis import (  # noqa: E402
    compare_checkpoints_track_a,
    run_track_a_analysis,
)


def run_command(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise AssertionError(
            f"Command failed: {' '.join(args)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def make_toy_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    features = []
    bin_labels = []
    task_labels = []
    semantic_labels = []
    anisotropy_ratios = []

    ratios_by_bin = {0: 1.2, 1: 2.1, 2: 4.3}
    samples_per_class_per_bin = 6

    for bin_id in range(3):
        for task_class in range(2):
            for sample_idx in range(samples_per_class_per_bin):
                feature = np.zeros(12, dtype=np.float32)
                feature[bin_id] = 3.0
                feature[3 + task_class] = 2.0
                feature[5] = 1.0 if sample_idx % 2 == 0 else -1.0
                feature[6] = float(bin_id + task_class)
                feature[7:] = rng.normal(0.0, 0.05, size=5)
                features.append(feature)
                bin_labels.append(bin_id)
                task_labels.append(task_class)
                semantic_labels.append([
                    1 if task_class == 1 else 0,
                    1 if sample_idx % 2 == 0 else 0,
                ])
                anisotropy_ratios.append(ratios_by_bin[bin_id] + sample_idx * 0.01)

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(bin_labels, dtype=np.int64),
        np.asarray(task_labels, dtype=np.int64),
        np.asarray(semantic_labels, dtype=np.int64),
        np.asarray(anisotropy_ratios, dtype=np.float32),
    )


def test_python_cli_entrypoints() -> None:
    print("\n[1/5] Testing Phase 1 Python CLI entrypoints")
    print("-" * 40)

    cli_files = [
        "phase1_evaluation_pipeline.py",
        "build_perturbation_cache.py",
        "perturbation_robustness_analysis.py",
        "anisotropy_semantic_analysis.py",
        "representation_geometry_analysis.py",
        "checkpoint_feature_extractor.py",
        "build_phase1_manifest.py",
        "manifest_generation.py",
    ]

    for cli_file in cli_files:
        result = run_command([sys.executable, cli_file, "--help"], cwd=PHASE1_DIR)
        require("usage:" in result.stdout.lower(), f"Missing help output for {cli_file}")
        print(f"  OK {cli_file}")


def test_shell_launchers() -> None:
    print("\n[2/5] Testing Phase 1 shell launchers")
    print("-" * 40)

    launchers = [
        "run_phase1_parallel_cls.sh",
        "run_phase1_parallel_avg_pool.sh",
        "run_phase1_parallel_multilayer.sh",
        "run_perturbation_parallel_cls.sh",
        "run_perturbation_parallel_avg_pool.sh",
        "run_perturbation_parallel_multilayer.sh",
    ]

    for launcher in launchers:
        run_command(["bash", "-n", launcher], cwd=SCRIPTS_DIR)
        print(f"  OK {launcher}")

    preset_check = run_command(
        [
            "bash",
            "-lc",
            "source ./phase1_dataset_presets.sh && "
            "resolve_phase1_preset abdomenct1k_core && "
            "printf '%s|%s|%s' \"$PHASE1_PRESET_ANALYSIS_NAME\" \"$PHASE1_PRESET_MANIFEST\" \"$PHASE1_PRESET_SUPPORTED_TASKS\"",
        ],
        cwd=SCRIPTS_DIR,
    )
    require(
        preset_check.stdout.strip()
        == "abdomenct1k|../../data_manifests/phase1_anisotropy_robustness/abdomenct1k/original_bins/manifest_sampled.json|full_phase1 perturbation_only",
        f"Unexpected preset resolution output: {preset_check.stdout!r}",
    )

    controlled_only_check = run_command(
        [
            "bash",
            "-lc",
            "source ./phase1_dataset_presets.sh && "
            "resolve_phase1_preset totalsegmenter_ct_core && "
            "if phase1_preset_supports full_phase1; then printf 'bad'; else printf 'ok'; fi && "
            "printf '|%s' \"$PHASE1_PRESET_MANIFEST\"",
        ],
        cwd=SCRIPTS_DIR,
    )
    require(
        controlled_only_check.stdout.strip()
        == "ok|../../data_manifests/phase1_anisotropy_robustness/totalsegmenter_ct/original_bins/manifest_sampled.json",
        f"Controlled-only preset resolution is inconsistent: {controlled_only_check.stdout!r}",
    )
    print("  OK phase1_dataset_presets.sh")


def test_output_namespace_helpers() -> None:
    print("\n[3/5] Testing canonical output namespace helpers")
    print("-" * 40)

    original_paths = get_output_paths("abdomenatlas", "original_bins")
    coarse_paths = get_output_paths("abdomenct1k", "coarse")

    require(
        original_paths["root"] == PHASE1_DIR / "outputs_phase1" / "abdomenatlas" / "phase1" / "original_bins",
        f"Unexpected original root path: {original_paths['root']}",
    )
    require(
        original_paths["results"] == PHASE1_DIR / "outputs_phase1" / "abdomenatlas" / "phase1" / "original_bins" / "results",
        f"Unexpected original results path: {original_paths['results']}",
    )
    require(
        coarse_paths["root"] == PHASE1_DIR / "outputs_phase1" / "abdomenct1k" / "phase1" / "coarse_bins",
        f"Unexpected coarse root path: {coarse_paths['root']}",
    )
    require(
        coarse_paths["manifests"] == PROJECT_ROOT / "eval_downstream" / "medfm_eval" / "data_manifests" / "phase1_anisotropy_robustness" / "abdomenct1k" / "coarse_bins",
        f"Unexpected coarse manifest dir: {coarse_paths['manifests']}",
    )
    print("  OK canonical output roots and manifest directories")


def test_toy_feature_analyses() -> None:
    print("\n[4/5] Testing toy-tensor Track A/Track B analyses")
    print("-" * 40)

    features, bin_labels, task_labels, semantic_labels, anisotropy_ratios = make_toy_features()

    with tempfile.TemporaryDirectory(prefix="phase1_smoke_") as tmpdir:
        tmp_path = Path(tmpdir)
        results_root = tmp_path / "results"
        figures_root = tmp_path / "figures"

        track_a_base = run_track_a_analysis(
            features=features,
            bin_labels=bin_labels,
            checkpoint_name="toy_ckpt",
            feature_type="cls",
            compute_tsne=False,
            output_dir=results_root,
            figures_dir=figures_root,
            dataset_name="toyct",
            manifest_variant="original_bins",
        )
        track_a_sa = run_track_a_analysis(
            features=features * 1.01,
            bin_labels=bin_labels,
            checkpoint_name="toy_ckpt_sa",
            feature_type="cls",
            compute_tsne=False,
            output_dir=results_root,
            figures_dir=figures_root,
            dataset_name="toyct",
            manifest_variant="original_bins",
        )
        compare_checkpoints_track_a(
            [track_a_base, track_a_sa],
            output_dir=results_root,
            dataset_name="toyct",
            manifest_variant="original_bins",
        )

        require(
            (results_root / "toy_ckpt" / "cls" / "representation_geometry.json").exists(),
            "Track A results file missing",
        )
        track_a_payload = json.loads(
            (results_root / "toy_ckpt" / "cls" / "representation_geometry.json").read_text(encoding="utf-8")
        )
        require(
            "balanced_bin_sensitivity" in track_a_payload,
            "Track A should emit balanced-bin sensitivity output",
        )
        require(
            (results_root / "summaries" / "cls" / "representation_geometry_comparison.json").exists(),
            "Track A comparison file missing",
        )

        track_b_base, spacing_base = run_track_b_analysis(
            features=features,
            bin_labels=bin_labels,
            task_labels=task_labels,
            checkpoint_name="toy_ckpt",
            feature_type="cls",
            n_cv_splits=3,
            output_dir=tmp_path / "track_b",
            anisotropy_ratios=anisotropy_ratios,
            analysis_name="toyct",
        )
        track_b_sa, _ = run_track_b_analysis(
            features=features * 0.99,
            bin_labels=bin_labels,
            task_labels=task_labels,
            checkpoint_name="toy_ckpt_sa",
            feature_type="cls",
            n_cv_splits=3,
            output_dir=tmp_path / "track_b",
            anisotropy_ratios=anisotropy_ratios,
            analysis_name="toyct",
        )
        compare_checkpoints_track_b(
            [track_b_base, track_b_sa],
            output_dir=tmp_path / "track_b",
            analysis_name="toyct",
        )
        semantic_results = run_semantic_probing(
            features=features,
            bin_labels=bin_labels,
            semantic_labels=semantic_labels,
            checkpoint_name="toy_ckpt",
            feature_type="cls",
            label_names=["label_task", "label_even"],
            anisotropy_ratios=anisotropy_ratios,
            n_cv_splits=3,
            output_dir=tmp_path / "semantic",
            analysis_name="toyct",
            manifest_variant="original_bins",
        )
        semantic_transfer = run_multilabel_cross_bin_transfer(
            features=features,
            bin_labels=bin_labels,
            semantic_labels=semantic_labels,
            checkpoint_name="toy_ckpt",
            feature_type="cls",
            dataset="toyct",
            output_dir=tmp_path / "semantic",
            analysis_name="toyct",
            manifest_variant="original_bins",
        )

        require(track_b_base.overall_accuracy > 0.0, "Track B overall accuracy should be positive")
        require(spacing_base is not None, "Spacing regression result should be present")
        require(semantic_results.mean_balanced_accuracy > 0.0, "Semantic probing should be positive")
        require(semantic_transfer.cross_bin_accuracy >= 0.0, "Semantic transfer should be computed")
        require(
            (tmp_path / "track_b" / "toy_ckpt_cls_track_b.json").exists(),
            "Track B results file missing",
        )
        track_b_payload = json.loads(
            (tmp_path / "track_b" / "toy_ckpt_cls_track_b.json").read_text(encoding="utf-8")
        )
        require(
            track_b_payload.get("spacing_regression", {}).get("split_strategy") in {"ratio_quantile", "kfold_fallback"},
            "Spacing regression should record its split strategy",
        )
        require(
            "balanced_bin_sensitivity" in track_b_payload,
            "Track B should emit balanced-bin sensitivity output",
        )
        require(
            (tmp_path / "track_b" / "track_b_checkpoint_comparison.json").exists(),
            "Track B comparison file missing",
        )
        require(
            (tmp_path / "semantic" / "toy_ckpt" / "cls" / "semantic_probing.json").exists(),
            "Semantic probing file missing",
        )
        semantic_payload = json.loads(
            (tmp_path / "semantic" / "toy_ckpt" / "cls" / "semantic_probing.json").read_text(encoding="utf-8")
        )
        require(
            "balanced_bin_sensitivity" in semantic_payload,
            "Semantic probing should emit balanced-bin sensitivity output",
        )
        require(
            (tmp_path / "semantic" / "toy_ckpt" / "cls" / "semantic_transfer.json").exists(),
            "Semantic transfer file missing",
        )

        catalog_path = write_results_catalog(results_root, "toyct", tmp_path / "toy_manifest.json")
        require(catalog_path.exists(), "Results catalog file missing")
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        require(
            catalog["per_checkpoint_files"]["toy_ckpt"]["cls"][0]["relative_path"]
            == "toy_ckpt/cls/representation_geometry.json",
            "Canonical nested Track A result path was not cataloged correctly",
        )
        require(
            any(
                entry["relative_path"] == "summaries/cls/representation_geometry_comparison.json"
                for entry in catalog["summary_files"]
            ),
            "Summary comparison path was not cataloged correctly",
        )
        print("  OK Track A, Track B, semantic probing, semantic transfer")


def test_synthetic_perturbation_flow() -> None:
    print("\n[5/5] Testing synthetic NIfTI perturbation, cache flow, and precompute CLI")
    print("-" * 40)

    with tempfile.TemporaryDirectory(prefix="phase1_perturb_") as tmpdir:
        tmp_path = Path(tmpdir)
        image_path = tmp_path / "toy_volume.nii.gz"
        manifest_path = tmp_path / "manifest_sampled.json"
        cache_root = tmp_path / "cache"

        volume = np.zeros((20, 18, 16), dtype=np.float32)
        volume[4:16, 4:14, 3:13] = 100.0
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        nib.save(nib.Nifti1Image(volume, affine), image_path)

        manifest = {
            "dataset": "toyct",
            "modality": "ct",
            "volumes": [
                {
                    "file_path": str(image_path),
                    "spacing": [1.0, 1.0, 1.0],
                    "anisotropy_ratio": 1.0,
                    "anisotropy_bin": 0,
                    "dataset": "toyct",
                    "has_label": False,
                }
            ],
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        config = ControlledPerturbationConfig(
            target_spacings=[(1.0, 1.0, 1.0), (1.0, 1.0, 3.0), (1.0, 1.0, 5.0)],
            n_source_volumes=1,
        )
        variant_specs = config.resolve_variant_specs((1.0, 1.0, 1.0))
        variants = resample_volume(
            file_path=image_path,
            variant_specs=variant_specs,
            crop_size=16,
            source_spacing=(1.0, 1.0, 1.0),
            config=config,
        )

        require(all(tensor is not None for tensor in variants.values()), "Resampling returned empty variants")
        require(all(tuple(tensor.shape) == (1, 16, 16, 16) for tensor in variants.values()), "Unexpected variant shape")
        diagnostics = compute_spacing_variant_diagnostics(variants, reference_variant="1.0x1.0x1.0")
        require(
            diagnostics["1.0x1.0x3.0"]["mean_absolute_difference"] > 0.01,
            "Moderate-spacing perturbation is too weak",
        )
        require(
            (variants["1.0x1.0x5.0"] - variants["1.0x1.0x3.0"]).abs().mean().item() > 0.005,
            "Moderate and high-spacing perturbations should remain distinguishable from each other",
        )
        require(
            diagnostics["1.0x1.0x5.0"]["mean_absolute_difference"] > 0.01,
            "High-spacing perturbation is too weak",
        )
        require(
            diagnostics["1.0x1.0x5.0"]["normalized_cross_correlation"] < 0.999,
            "High-spacing perturbation is too correlated with the reference",
        )
        require(
            save_to_cache(
                file_path=image_path,
                variants=variants,
                crop_size=16,
                cache_dir=cache_root,
                cache_signature=config.cache_signature(),
            ),
            "Saving perturbation cache failed",
        )
        cached = load_from_cache(
            file_path=image_path,
            crop_size=16,
            cache_dir=cache_root,
            cache_signature=config.cache_signature(),
        )
        require(cached is not None, "Loading perturbation cache failed")
        require(sorted(cached.keys()) == sorted(variants.keys()), "Cached variant keys mismatch")

        perturbed = prepare_perturbed_dataset(
            manifest_path=manifest_path,
            config=config,
            crop_size=16,
            cache_dir=cache_root,
        )
        require(len(perturbed) == 1, f"Expected 1 perturbed volume, got {len(perturbed)}")
        require(
            sorted(perturbed[0].spacing_variants.keys()) == sorted(variants.keys()),
            "Prepared perturbation variants mismatch",
        )

        cache_namespace = "phase1_smoke_cache"
        canonical_cache_root = get_cache_root(cache_namespace, "original_bins")
        canonical_precompute_config = get_controlled_perturbation_config_from_manifest(manifest)
        if canonical_cache_root.exists():
            shutil.rmtree(canonical_cache_root)

        try:
            run_command(
                [
                    sys.executable,
                    "build_perturbation_cache.py",
                    "-m",
                    str(manifest_path),
                    "-a",
                    cache_namespace,
                    "--crop-size",
                    "16",
                ],
                cwd=PHASE1_DIR,
            )

            crop_cache_root = canonical_cache_root / "crop16"
            signature_cache_root = crop_cache_root / canonical_precompute_config.cache_signature()
            cached_files = sorted(signature_cache_root.rglob("*.pt"))

            require(canonical_cache_root.exists(), f"Canonical cache root was not created: {canonical_cache_root}")
            require(crop_cache_root.exists(), f"Crop cache root was not created: {crop_cache_root}")
            require(
                signature_cache_root.exists(),
                f"Cache signature directory was not created: {signature_cache_root}",
            )
            require(
                len(cached_files) == 1,
                "Expected 1 precomputed cache file under "
                f"{signature_cache_root}, found {len(cached_files)}: {cached_files}",
            )

            signature_result = run_command(
                [
                    "bash",
                    "-lc",
                    f"source ./phase1_dataset_presets.sh && phase1_resolve_cache_signature '{manifest_path}'",
                ],
                cwd=SCRIPTS_DIR,
            )
            require(
                signature_result.stdout.strip() == canonical_precompute_config.cache_signature(),
                f"Shell cache signature drifted from Python config: {signature_result.stdout!r}",
            )
        finally:
            if canonical_cache_root.exists():
                shutil.rmtree(canonical_cache_root)

        print("  OK perturbation resampling, cache flow, and precompute CLI")


def main() -> None:
    test_python_cli_entrypoints()
    test_shell_launchers()
    test_output_namespace_helpers()
    test_toy_feature_analyses()
    test_synthetic_perturbation_flow()
    print("\nAll Phase 1 anisotropy smoke tests passed.")


if __name__ == "__main__":
    main()