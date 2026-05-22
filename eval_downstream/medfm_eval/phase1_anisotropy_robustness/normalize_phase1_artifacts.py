#!/usr/bin/env python
"""Normalize legacy Phase 1 outputs into the canonical nested artifact layout.

Example:
    python normalize_phase1_artifacts.py
"""

from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import CHECKPOINTS
from phase1_evaluation_pipeline import write_results_catalog


FEATURE_TYPES = ["avg_pool", "multilayer", "cls"]
AGGREGATE_FILES = {
    "phase1_full_results_cls.json",
    "phase1_full_results_avg_pool.json",
    "phase1_full_results_multilayer.json",
    "results_catalog.json",
}


def parse_checkpoint_feature(parts: tuple[str, ...]) -> tuple[str | None, str | None, str]:
    slug = "_".join(parts)
    checkpoint = next((name for name in sorted(CHECKPOINTS.keys(), key=len, reverse=True) if slug.startswith(name)), None)
    if checkpoint is None:
        return None, None, slug

    remainder = slug[len(checkpoint):].lstrip("_")
    feature = next(
        (
            name
            for name in FEATURE_TYPES
            if remainder.startswith(name) or f"_{name}_" in f"_{remainder}_" or remainder == name
        ),
        None,
    )
    return checkpoint, feature, slug


def move_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src == dst:
        return
    if dst.exists():
        src.unlink()
        return
    src.rename(dst)


def normalize_results(results_dir: Path, legacy_dir: Path) -> None:
    for file in list(results_dir.rglob("*.json")):
        rel = file.relative_to(results_dir)
        if rel.parts[0] == "summaries":
            continue
        if file.name in AGGREGATE_FILES:
            continue
        if "track_b" in file.name or "domain_transfer" in file.name or file.name == "track_b_checkpoint_comparison.json":
            move_file(file, legacy_dir / file.name)
            continue

        checkpoint, feature, slug = parse_checkpoint_feature(rel.parts[:-1] + (file.stem,))
        if checkpoint is None or feature is None:
            continue

        target_dir = results_dir / checkpoint / feature
        if file.name == "representation_geometry.json" or "track_a" in slug:
            target = target_dir / "representation_geometry.json"
        elif file.name == "semantic_probing.json" or "semantic_probing" in slug:
            target = target_dir / "semantic_probing.json"
        elif file.name == "semantic_transfer.json" or "semantic_transfer" in slug:
            target = target_dir / "semantic_transfer.json"
        elif file.name == "perturbation_robustness.json" or "setting_b" in slug:
            target = target_dir / "perturbation_robustness.json"
        elif file.name == "anisotropy_regression.json" or "spacing_regression" in slug:
            target = target_dir / "anisotropy_regression.json"
        else:
            continue
        move_file(file, target)


def normalize_figures(figures_dir: Path) -> None:
    for file in list(figures_dir.rglob("*.png")):
        checkpoint, feature, slug = parse_checkpoint_feature(file.relative_to(figures_dir).parts[:-1] + (file.stem,))
        if checkpoint is None or feature is None:
            continue
        target_dir = figures_dir / checkpoint / feature
        if file.name == "tsne.png" or "tsne" in slug:
            target = target_dir / "tsne.png"
        elif file.name == "drift_curve.png" or "drift_curve" in slug:
            target = target_dir / "drift_curve.png"
        else:
            continue
        move_file(file, target)


def normalize_features(features_dir: Path) -> None:
    for file in list(features_dir.rglob("*.npz")):
        checkpoint, feature, _ = parse_checkpoint_feature(file.relative_to(features_dir).parts[:-1] + (file.stem,))
        if checkpoint is None or feature is None:
            continue
        move_file(file, features_dir / checkpoint / feature / file.name)

    for file in list(features_dir.glob("semantic_labels_*.json")):
        move_file(file, features_dir / "semantic_labels" / file.name)


def prune_empty_dirs(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def scrub_legacy_timestamps(results_dir: Path) -> None:
    for json_path in results_dir.glob("phase1*_results*.json"):
        data = json.loads(json_path.read_text())
        if isinstance(data, dict) and "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"].pop("timestamp", None)
        data.pop("generated_at", None) if isinstance(data, dict) else None
        json_path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    outputs_root = ROOT / "outputs_phase1"
    manifests_root = ROOT.parent / "data_manifests" / "phase1_anisotropy_robustness"

    for variant_root in outputs_root.glob("*/phase1/*"):
        if not variant_root.is_dir():
            continue

        results_dir = variant_root / "results"
        figures_dir = variant_root / "figures"
        features_dir = variant_root / "features"
        legacy_dir = variant_root / "legacy_removed" / "results"
        legacy_dir.mkdir(parents=True, exist_ok=True)

        if results_dir.exists():
            normalize_results(results_dir, legacy_dir)
        if figures_dir.exists():
            normalize_figures(figures_dir)
        if features_dir.exists():
            normalize_features(features_dir)

        prune_empty_dirs(variant_root)
        if results_dir.exists():
            scrub_legacy_timestamps(results_dir)

        dataset_name = variant_root.parts[-3]
        manifest_variant = variant_root.parts[-1]
        manifest_path = manifests_root / dataset_name / manifest_variant / "manifest_sampled.json"
        if results_dir.exists() and manifest_path.exists():
            write_results_catalog(results_dir, dataset_name, manifest_path)

    print("artifact normalization complete")


if __name__ == "__main__":
    main()