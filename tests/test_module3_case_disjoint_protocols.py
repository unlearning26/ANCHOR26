import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE3_ROOT = REPO_ROOT / "eval_downstream" / "medfm_eval" / "module3_anatomical_generalization"

if str(MODULE3_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE3_ROOT))


def _load_module(module_name: str):
    module_path = MODULE3_ROOT / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CASE_SPLITS = _load_module("case_disjoint_partitions")
FEW_SHOT = _load_module("few_shot_transfer_analysis")
HOLDOUT = _load_module("holdout_generalization_analysis")


class Module3CaseDisjointProtocolTests(unittest.TestCase):
    def _build_records_and_features(self):
        records = []
        features_by_id = {}
        organ_centers = {
            "liver": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "spleen": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        }
        for case_index in range(6):
            case_id = f"case-{case_index}"
            for organ, center in organ_centers.items():
                for modality, modality_shift in (("ct", 0.00), ("mr", 0.02)):
                    sample_id = f"{modality}:{case_id}:{organ}"
                    records.append(
                        {
                            "sample_id": sample_id,
                            "source_case_id": case_id,
                            "patient_id": case_id,
                            "primary_organ": organ,
                            "modality": modality,
                        }
                    )
                    features_by_id[sample_id] = center + np.asarray(
                        [modality_shift, 0.0, 0.001 * case_index],
                        dtype=np.float32,
                    )
        return records, features_by_id

    def test_global_case_split_map_is_disjoint_across_organs_and_modalities(self):
        records, _ = self._build_records_and_features()
        split_map = CASE_SPLITS.build_global_case_split_map(
            records=records,
            supported_organs=["liver", "spleen"],
            required_modalities=("ct", "mr"),
            seed=7,
            min_count_by_partition={"split_a": 1, "split_b": 1},
        )

        self.assertTrue(split_map)
        split_a_case_ids = {
            CASE_SPLITS.case_id_for_record(record)
            for organ_map in split_map.values()
            for modality_map in organ_map.values()
            for record in modality_map["split_a"]
        }
        split_b_case_ids = {
            CASE_SPLITS.case_id_for_record(record)
            for organ_map in split_map.values()
            for modality_map in organ_map.values()
            for record in modality_map["split_b"]
        }
        self.assertTrue(split_a_case_ids)
        self.assertTrue(split_b_case_ids)
        self.assertTrue(split_a_case_ids.isdisjoint(split_b_case_ids))

    def test_cross_modal_few_shot_transfer_uses_case_disjoint_protocol(self):
        records, features_by_id = self._build_records_and_features()
        analysis = FEW_SHOT.evaluate_leave_one_organ_out_few_shot_transfer(
            features_by_id=features_by_id,
            samples=records,
            supported_organs=["liver", "spleen"],
            required_modalities=("ct", "mr"),
            support_per_modality=1,
            query_per_modality=1,
            seeds=(7,),
        )

        self.assertEqual(analysis["status"], "ok")
        self.assertEqual(analysis["transfer_protocol"], "fixed_budget_few_shot_transfer_case_disjoint")
        self.assertEqual(analysis["case_split_policy"], "global_case_disjoint_partitions")
        self.assertEqual(analysis["n_evaluated_organs"], 2)

    def test_cross_modal_holdout_transfer_reports_global_case_split_policy(self):
        records, features_by_id = self._build_records_and_features()
        analysis = HOLDOUT.evaluate_leave_one_organ_out_transfer(
            features_by_id=features_by_id,
            samples=records,
            supported_organs=["liver", "spleen"],
            required_modalities=("ct", "mr"),
            top_ks=(1,),
            seed=7,
        )

        self.assertEqual(analysis["status"], "ok")
        self.assertEqual(analysis["surface_scope"], "cross_modality")
        self.assertEqual(analysis["case_split_policy"], "global_case_disjoint_partitions")
        self.assertEqual(analysis["n_evaluated_organs"], 2)


if __name__ == "__main__":
    unittest.main()