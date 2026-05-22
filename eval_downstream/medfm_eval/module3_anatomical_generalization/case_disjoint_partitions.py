from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Sequence

import numpy as np


def case_id_for_record(record: Dict[str, Any]) -> str:
    return str(record.get("source_case_id") or record.get("patient_id") or record.get("sample_id"))


def _empty_split_map(
    supported_organs: Sequence[str],
    required_modalities: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, list[Dict[str, Any]]]]]:
    return {
        str(organ): {
            str(modality): {"split_a": [], "split_b": []}
            for modality in required_modalities
        }
        for organ in supported_organs
    }


def _has_partition_coverage(
    split_map: Dict[str, Dict[str, Dict[str, list[Dict[str, Any]]]]],
    supported_organs: Sequence[str],
    required_modalities: Sequence[str],
    min_count_by_partition: Dict[str, int],
) -> bool:
    for organ in supported_organs:
        for modality in required_modalities:
            partition_map = split_map.get(str(organ), {}).get(str(modality), {})
            for partition_name, min_count in min_count_by_partition.items():
                if len(partition_map.get(partition_name, [])) < int(min_count):
                    return False
    return True


def build_global_case_split_map(
    records: Sequence[Dict[str, Any]],
    supported_organs: Sequence[str],
    required_modalities: Sequence[str],
    seed: int,
    min_count_by_partition: Dict[str, int] | None = None,
    max_attempts: int = 128,
) -> Dict[str, Dict[str, Dict[str, list[Dict[str, Any]]]]]:
    supported_organs = tuple(sorted(str(organ) for organ in supported_organs))
    required_modalities = tuple(str(modality).lower() for modality in required_modalities)
    supported_organ_set = set(supported_organs)
    required_modality_set = set(required_modalities)

    filtered_records = [
        record
        for record in records
        if str(record.get("primary_organ")) in supported_organ_set
        and str(record.get("modality")).lower() in required_modality_set
    ]
    records_by_case: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for record in filtered_records:
        records_by_case[case_id_for_record(record)].append(record)

    case_ids = sorted(records_by_case)
    split_a_weight = max(1, int((min_count_by_partition or {}).get("split_a", 1)))
    split_b_weight = max(1, int((min_count_by_partition or {}).get("split_b", 1)))
    split_a_size = int(np.ceil(len(case_ids) * split_a_weight / (split_a_weight + split_b_weight)))
    if split_a_size <= 0 or split_a_size >= len(case_ids):
        return {}

    min_count_by_partition = {
        "split_a": 1,
        "split_b": 1,
        **(min_count_by_partition or {}),
    }

    base_seed = int(seed)
    for attempt in range(int(max_attempts)):
        rng = np.random.default_rng(base_seed + attempt)
        shuffled_case_ids = [case_ids[int(index)] for index in rng.permutation(len(case_ids))]
        split_a_case_ids = set(shuffled_case_ids[:split_a_size])
        split_b_case_ids = set(shuffled_case_ids[split_a_size:])

        split_map = _empty_split_map(supported_organs, required_modalities)
        for partition_name, partition_case_ids in (("split_a", split_a_case_ids), ("split_b", split_b_case_ids)):
            for case_id in partition_case_ids:
                for record in records_by_case[case_id]:
                    organ = str(record.get("primary_organ"))
                    modality = str(record.get("modality")).lower()
                    if organ in supported_organ_set and modality in required_modality_set:
                        split_map[organ][modality][partition_name].append(record)

        if _has_partition_coverage(split_map, supported_organs, required_modalities, min_count_by_partition):
            return split_map

    return {}