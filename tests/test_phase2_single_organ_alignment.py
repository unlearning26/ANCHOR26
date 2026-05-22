import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "eval_downstream"
    / "medfm_eval"
    / "phase2_cross_modality_alignment"
    / "phase2_single_organ_alignment.py"
)
MODULE_DIR = MODULE_PATH.parent

if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

SPEC = importlib.util.spec_from_file_location("phase2_single_organ_alignment", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Phase2SingleOrganAlignmentTests(unittest.TestCase):
    def test_build_paired_case_records_aggregates_multiple_sequences(self):
        samples = [
            {"sample_id": "ct:1:liver", "primary_organ": "liver", "modality": "ct", "patient_id": "1"},
            {"sample_id": "mr:1:t1:liver", "primary_organ": "liver", "modality": "mr", "patient_id": "1"},
            {"sample_id": "mr:1:t2:liver", "primary_organ": "liver", "modality": "mr", "patient_id": "1"},
            {"sample_id": "ct:2:liver", "primary_organ": "liver", "modality": "ct", "patient_id": "2"},
            {"sample_id": "mr:2:t1:liver", "primary_organ": "liver", "modality": "mr", "patient_id": "2"},
        ]
        features_by_id = {
            "ct:1:liver": np.array([1.0, 0.0], dtype=np.float32),
            "mr:1:t1:liver": np.array([1.0, 0.0], dtype=np.float32),
            "mr:1:t2:liver": np.array([1.0, 0.0], dtype=np.float32),
            "ct:2:liver": np.array([0.0, 1.0], dtype=np.float32),
            "mr:2:t1:liver": np.array([0.0, 1.0], dtype=np.float32),
        }

        paired_cases = MODULE._build_paired_case_records(
            samples=samples,
            features_by_id=features_by_id,
            organ="liver",
            required_modalities=("ct", "mr"),
            pair_id_field="patient_id",
        )

        self.assertEqual(len(paired_cases), 2)
        self.assertEqual(paired_cases[0]["feature_count_by_modality"]["mr"], 2)
        np.testing.assert_allclose(paired_cases[0]["features"]["mr"], np.array([1.0, 0.0], dtype=np.float32))

    def test_compute_bidirectional_patient_retrieval_returns_perfect_scores_for_identity_pairs(self):
        paired_cases = [
            {"pair_id": "1", "features": {"ct": np.array([1.0, 0.0], dtype=np.float32), "mr": np.array([1.0, 0.0], dtype=np.float32)}},
            {"pair_id": "2", "features": {"ct": np.array([0.0, 1.0], dtype=np.float32), "mr": np.array([0.0, 1.0], dtype=np.float32)}},
        ]

        retrieval = MODULE._compute_bidirectional_patient_retrieval(paired_cases, ("ct", "mr"))

        self.assertEqual(retrieval["status"], "ok")
        self.assertEqual(retrieval["bidirectional_mean"]["top@1"], 1.0)
        self.assertEqual(retrieval["bidirectional_mean"]["map"], 1.0)

    def test_paired_case_records_preserve_sorted_pair_ids_for_eight_liver_pairs(self):
        pair_ids = ["1", "2", "5", "8", "10", "19", "21", "22"]
        samples = []
        features_by_id = {}
        for pair_id in pair_ids:
            ct_sample_id = f"ct:{pair_id}:liver"
            mr_sample_id = f"mr:{pair_id}:liver"
            samples.extend(
                [
                    {"sample_id": ct_sample_id, "primary_organ": "liver", "modality": "ct", "patient_id": pair_id},
                    {"sample_id": mr_sample_id, "primary_organ": "liver", "modality": "mr", "patient_id": pair_id},
                ]
            )
            features_by_id[ct_sample_id] = np.asarray([1.0, float(pair_id)], dtype=np.float32)
            features_by_id[mr_sample_id] = np.asarray([1.0, float(pair_id)], dtype=np.float32)

        paired_cases = MODULE._build_paired_case_records(
            samples=samples,
            features_by_id=features_by_id,
            organ="liver",
            required_modalities=("ct", "mr"),
            pair_id_field="patient_id",
        )

        self.assertEqual(len(paired_cases), 8)
        self.assertEqual([case["pair_id"] for case in paired_cases], ["1", "2", "5", "8", "10", "19", "21", "22"])


if __name__ == "__main__":
    unittest.main()