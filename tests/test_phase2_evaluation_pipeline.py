import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "eval_downstream"
    / "medfm_eval"
    / "phase2_cross_modality_alignment"
    / "phase2_evaluation_pipeline.py"
)
MODULE_DIR = MODULE_PATH.parent

if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

SPEC = importlib.util.spec_from_file_location("phase2_evaluation_pipeline", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

RETRIEVAL_SPEC = importlib.util.spec_from_file_location("retrieval_analysis", MODULE_DIR / "retrieval_analysis.py")
RETRIEVAL = importlib.util.module_from_spec(RETRIEVAL_SPEC)
sys.modules[RETRIEVAL_SPEC.name] = RETRIEVAL
assert RETRIEVAL_SPEC.loader is not None
RETRIEVAL_SPEC.loader.exec_module(RETRIEVAL)


class Phase2EvaluationPipelineTests(unittest.TestCase):
    def test_retrieval_can_return_query_rows_and_bootstrap_ci(self):
        samples = []
        features_by_id = {}
        organ_centers = {
            "aorta": [1.0, 0.0, 0.0],
            "liver": [0.0, 1.0, 0.0],
        }
        for organ in organ_centers:
            for modality in ("ct", "mr"):
                for sample_index in range(3):
                    sample_id = f"{modality}:{organ}:{sample_index}"
                    samples.append(
                        {
                            "sample_id": sample_id,
                            "modality": modality,
                            "primary_organ": organ,
                        }
                    )
                    features_by_id[sample_id] = MODULE.np.asarray(
                        organ_centers[organ],
                        dtype=MODULE.np.float32,
                    ) + 0.001 * sample_index

        retrieval = RETRIEVAL.compute_bidirectional_cross_modal_retrieval(
            features_by_id,
            samples,
            supported_organs=["aorta", "liver"],
            top_ks=(1, 5),
            seed=42,
            return_query_results=True,
            bootstrap_resamples=25,
        )

        self.assertEqual(retrieval["status"], "ok")
        self.assertIn("bidirectional_bootstrap_95ci", retrieval)
        self.assertIn("top@1", retrieval["bidirectional_bootstrap_95ci"])
        self.assertGreater(retrieval["bidirectional_mean"]["top@1"], 0.99)
        self.assertGreater(len(retrieval["ct_to_mr"]["query_results"]), 0)
        self.assertIn("average_precision", retrieval["ct_to_mr"]["query_results"][0])

    def test_single_organ_phase2_retrieval_falls_back_to_paired_case_matching(self):
        samples = []
        features_by_id = {}
        for case_index in range(4):
            case_id = f"case-{case_index}"
            for modality in ("ct", "mr"):
                sample_id = f"{modality}:{case_id}:liver"
                samples.append(
                    {
                        "sample_id": sample_id,
                        "modality": modality,
                        "primary_organ": "liver",
                        "source_case_id": case_id,
                        "patient_id": case_id,
                    }
                )
                features_by_id[sample_id] = MODULE.np.asarray(
                    [1.0, float(case_index), 0.0],
                    dtype=MODULE.np.float32,
                )

        retrieval = RETRIEVAL.compute_bidirectional_cross_modal_retrieval(
            features_by_id,
            samples,
            supported_organs=["liver"],
            top_ks=(1, 5),
            seed=42,
            return_query_results=True,
            bootstrap_resamples=25,
        )

        self.assertEqual(retrieval["status"], "ok")
        self.assertEqual(retrieval["ct_to_mr"]["evaluation"], "paired_case_single_organ_cross_modal_retrieval")
        self.assertGreater(retrieval["bidirectional_mean"]["top@1"], 0.99)
        self.assertGreater(retrieval["bidirectional_mean"]["map"], 0.99)

    def test_resolve_existing_path_uses_repo_root_for_missing_cwd_relative_path(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest_sampled.json"
            manifest_path.write_text("[]", encoding="utf-8")

            repo_relative_path = manifest_path.relative_to(REPO_ROOT)
            resolved_path = MODULE._resolve_existing_path(repo_relative_path)

            self.assertEqual(resolved_path, manifest_path.resolve())

    def test_resolve_manifest_path_uses_embedding_summary_for_shorthand_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            manifest_path = tmp_path / "manifest_sampled.json"
            manifest_path.write_text("[]", encoding="utf-8")

            embeddings_path = (
                tmp_path
                / "outputs_phase2"
                / "mmwhs_ct_mr"
                / "phase2"
                / "core"
                / "features"
                / "3dinov2"
                / "cls"
                / "phase2_organ_cls_embeddings.npz"
            )
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            embeddings_path.write_bytes(b"npz")
            summary_path = embeddings_path.with_name("phase2_organ_cls_embeddings_summary.json")
            summary_path.write_text(json.dumps({"manifest_path": str(manifest_path)}), encoding="utf-8")

            resolved_path = MODULE._resolve_manifest_path(
                "manifest_sampled.json",
                analysis_name=None,
                manifest_variant=None,
                embeddings_npz=str(embeddings_path),
            )

            self.assertEqual(resolved_path, manifest_path.resolve())

    def test_resolve_manifest_path_uses_canonical_phase2_manifest_for_shorthand_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            manifest_path = tmp_path / "manifest_sampled.json"
            manifest_path.write_text("[]", encoding="utf-8")

            with patch.object(MODULE, "get_phase2_manifest_path", return_value=manifest_path):
                resolved_path = MODULE._resolve_manifest_path(
                    "manifest_sampled.json",
                    analysis_name="mmwhs_ct_mr",
                    manifest_variant="core",
                    embeddings_npz=None,
                )

            self.assertEqual(resolved_path, manifest_path.resolve())


if __name__ == "__main__":
    unittest.main()