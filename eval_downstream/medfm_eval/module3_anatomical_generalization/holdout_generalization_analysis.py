"""Leave-one-organ-out transfer metrics over Phase 2 organ embeddings.

This module implements the core Module 3 metric. It does not read files or
discover manifests itself; instead it consumes:

    - ``features_by_id``: mapping from sample ID to embedding vector
    - ``samples``: normalized Module 3 manifest samples
    - ``supported_organs``: organs already validated for Module 3 hold-out

Evaluation strategy:
    1. restrict to evaluable records under the required modality rule
    2. build balanced target pools across organs
    3. score paired retrieval directions for each held-out organ
    4. average the two directions into one bidirectional score
    5. compute per-organ generalization gaps relative to the remaining organs

Example:
    >>> evaluate_leave_one_organ_out_transfer(
    ...     features_by_id=features_by_id,
    ...     samples=samples,
    ...     supported_organs=["aorta", "liver", "spleen"],
    ...     top_ks=(1, 5),
    ...     seed=42,
    ... )
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import comb
from typing import Any, Dict, Sequence

import numpy as np

from case_disjoint_partitions import build_global_case_split_map

try:
    from sklearn.metrics import silhouette_score
except ImportError:  # pragma: no cover - optional dependency guard
    silhouette_score = None


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    """L2-normalize a feature matrix row-wise."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return features / norms


def _l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    """L2-normalize one feature vector."""
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def _sample_records(records: Sequence[Dict[str, Any]], sample_count: int, rng: np.random.Generator) -> list[Dict[str, Any]]:
    """Sample records without replacement while preserving record structure."""
    entries = list(records)
    if sample_count >= len(entries):
        return entries
    indices = rng.choice(len(entries), size=sample_count, replace=False)
    return [entries[index] for index in indices]


def _balanced_top_chance(n_organs: int, samples_per_organ: int, k: int) -> float | None:
    """Compute analytical chance performance for balanced top-k retrieval."""
    total_targets = n_organs * samples_per_organ
    if n_organs <= 1 or samples_per_organ <= 0 or total_targets <= 0:
        return None
    k = min(k, total_targets)
    if k <= 0:
        return None
    if total_targets - samples_per_organ < k:
        return 1.0
    missed = comb(total_targets - samples_per_organ, k) / comb(total_targets, k)
    return float(1.0 - missed)


def _average_precision(relevance: np.ndarray) -> float:
    """Compute average precision for one ranked binary relevance vector."""
    positive_ranks = np.flatnonzero(relevance) + 1
    if positive_ranks.size == 0:
        return 0.0
    precision_at_hits = np.arange(1, positive_ranks.size + 1, dtype=np.float64) / positive_ranks
    return float(np.mean(precision_at_hits))


def _summary_stats(values: Sequence[float]) -> Dict[str, Any]:
    """Summarize a numeric sequence with scalar descriptive statistics."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "n_samples": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "n_samples": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _top_metric_key(k: int) -> str:
    """Convert an integer k to the serialized metric key used in outputs."""
    return f"top@{k}"


LOWER_IS_BETTER_METRICS = {
    "heldout_centroid_distance",
}


def _eligible_records(
    samples: Sequence[Dict[str, Any]],
    features_by_id: Dict[str, np.ndarray],
    supported_organs: Sequence[str],
    required_modalities: Sequence[str],
) -> list[Dict[str, Any]]:
    """Keep only samples that can participate in Module 3 evaluation."""
    supported = set(supported_organs)
    required = {str(modality).lower() for modality in required_modalities}
    return [
        sample
        for sample in samples
        if sample.get("sample_id") in features_by_id
        and sample.get("primary_organ") in supported
        and sample.get("modality") in required
    ]


def _group_records_by_organ_and_modality(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, list[Dict[str, Any]]]]:
    """Group normalized records as organ -> modality -> records."""
    grouped: Dict[str, Dict[str, list[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record["primary_organ"]][record["modality"]].append(record)
    return grouped


def _split_records_by_case(
    records: Sequence[Dict[str, Any]],
    seed: int,
) -> Dict[str, list[Dict[str, Any]]]:
    """Split records into two case-disjoint partitions for within-modality evaluation."""
    grouped_by_case: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        case_id = str(record.get("source_case_id") or record.get("patient_id") or record.get("sample_id"))
        grouped_by_case[case_id].append(record)

    case_ids = sorted(grouped_by_case)
    if len(case_ids) < 2:
        return {"split_a": [], "split_b": []}

    rng = np.random.default_rng(seed)
    shuffled_case_ids = [case_ids[int(index)] for index in rng.permutation(len(case_ids))]
    midpoint = len(shuffled_case_ids) // 2
    if midpoint <= 0 or midpoint >= len(shuffled_case_ids):
        return {"split_a": [], "split_b": []}

    split_a_case_ids = shuffled_case_ids[:midpoint]
    split_b_case_ids = shuffled_case_ids[midpoint:]
    return {
        "split_a": [record for case_id in split_a_case_ids for record in grouped_by_case[case_id]],
        "split_b": [record for case_id in split_b_case_ids for record in grouped_by_case[case_id]],
    }


def _build_within_modality_case_splits(
    grouped_records: Dict[str, Dict[str, list[Dict[str, Any]]]],
    supported_organs: Sequence[str],
    modality: str,
    seed: int,
) -> Dict[str, Dict[str, list[Dict[str, Any]]]]:
    """Build case-disjoint split-a/split-b pools for a single-modality Module 3 surface."""
    split_map: Dict[str, Dict[str, list[Dict[str, Any]]]] = {}
    for organ_index, organ in enumerate(sorted(supported_organs)):
        split_map[organ] = _split_records_by_case(
            grouped_records.get(organ, {}).get(modality, []),
            seed=seed + organ_index,
        )
    return split_map


def _compute_cross_modal_centroid_distances(
    grouped_records: Dict[str, Dict[str, list[Dict[str, Any]]]],
    features_by_id: Dict[str, np.ndarray],
    supported_organs: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    per_organ: Dict[str, Dict[str, Any]] = {}
    for organ in sorted(supported_organs):
        ct_records = list(grouped_records.get(organ, {}).get("ct", []))
        mr_records = list(grouped_records.get(organ, {}).get("mr", []))
        if not ct_records or not mr_records:
            per_organ[organ] = {
                "status": "skipped",
                "reason": "missing_cross_modal_support",
            }
            continue

        ct_stack = _l2_normalize(np.stack([features_by_id[record["sample_id"]] for record in ct_records], axis=0))
        mr_stack = _l2_normalize(np.stack([features_by_id[record["sample_id"]] for record in mr_records], axis=0))
        ct_centroid = _l2_normalize_vector(np.mean(ct_stack, axis=0))
        mr_centroid = _l2_normalize_vector(np.mean(mr_stack, axis=0))
        cosine_similarity = float(np.clip(np.dot(ct_centroid, mr_centroid), -1.0, 1.0))
        per_organ[organ] = {
            "status": "ok",
            "cross_modal_centroid_cosine_similarity": cosine_similarity,
            "cross_modal_centroid_cosine_distance": float(1.0 - cosine_similarity),
            "within_modality_dispersion": {
                "ct": float(np.mean(1.0 - np.clip(ct_stack @ ct_centroid, -1.0, 1.0))),
                "mr": float(np.mean(1.0 - np.clip(mr_stack @ mr_centroid, -1.0, 1.0))),
            },
        }
    return per_organ


def _compute_within_modality_split_centroid_distances(
    split_map: Dict[str, Dict[str, list[Dict[str, Any]]]],
    features_by_id: Dict[str, np.ndarray],
    supported_organs: Sequence[str],
    modality: str,
) -> Dict[str, Dict[str, Any]]:
    per_organ: Dict[str, Dict[str, Any]] = {}
    for organ in sorted(supported_organs):
        split_a_records = list(split_map.get(organ, {}).get("split_a", []))
        split_b_records = list(split_map.get(organ, {}).get("split_b", []))
        if not split_a_records or not split_b_records:
            per_organ[organ] = {
                "status": "skipped",
                "reason": "insufficient_case_disjoint_support",
                "modality": modality,
            }
            continue

        split_a_stack = _l2_normalize(np.stack([features_by_id[record["sample_id"]] for record in split_a_records], axis=0))
        split_b_stack = _l2_normalize(np.stack([features_by_id[record["sample_id"]] for record in split_b_records], axis=0))
        split_a_centroid = _l2_normalize_vector(np.mean(split_a_stack, axis=0))
        split_b_centroid = _l2_normalize_vector(np.mean(split_b_stack, axis=0))
        cosine_similarity = float(np.clip(np.dot(split_a_centroid, split_b_centroid), -1.0, 1.0))
        per_organ[organ] = {
            "status": "ok",
            "modality": modality,
            "within_modality_split_centroid_cosine_similarity": cosine_similarity,
            "within_modality_split_centroid_cosine_distance": float(1.0 - cosine_similarity),
            "within_split_dispersion": {
                "split_a": float(np.mean(1.0 - np.clip(split_a_stack @ split_a_centroid, -1.0, 1.0))),
                "split_b": float(np.mean(1.0 - np.clip(split_b_stack @ split_b_centroid, -1.0, 1.0))),
            },
        }
    return per_organ


def _compute_case_partition_centroid_distances(
    split_map: Dict[str, Dict[str, Dict[str, list[Dict[str, Any]]]]],
    features_by_id: Dict[str, np.ndarray],
    supported_organs: Sequence[str],
    required_modalities: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    required_modalities = tuple(str(modality).lower() for modality in required_modalities)
    per_organ: Dict[str, Dict[str, Any]] = {}
    for organ in sorted(supported_organs):
        if len(required_modalities) == 1:
            modality = required_modalities[0]
            split_a_records = list(split_map.get(organ, {}).get(modality, {}).get("split_a", []))
            split_b_records = list(split_map.get(organ, {}).get(modality, {}).get("split_b", []))
            if not split_a_records or not split_b_records:
                per_organ[organ] = {
                    "status": "skipped",
                    "reason": "insufficient_case_disjoint_support",
                    "modality": modality,
                }
                continue

            split_a_stack = _l2_normalize(np.stack([features_by_id[record["sample_id"]] for record in split_a_records], axis=0))
            split_b_stack = _l2_normalize(np.stack([features_by_id[record["sample_id"]] for record in split_b_records], axis=0))
            split_a_centroid = _l2_normalize_vector(np.mean(split_a_stack, axis=0))
            split_b_centroid = _l2_normalize_vector(np.mean(split_b_stack, axis=0))
            cosine_similarity = float(np.clip(np.dot(split_a_centroid, split_b_centroid), -1.0, 1.0))
            per_organ[organ] = {
                "status": "ok",
                "modality": modality,
                "within_modality_split_centroid_cosine_similarity": cosine_similarity,
                "within_modality_split_centroid_cosine_distance": float(1.0 - cosine_similarity),
                "case_partition_centroid_cosine_similarity": cosine_similarity,
                "case_partition_centroid_cosine_distance": float(1.0 - cosine_similarity),
            }
            continue

        ct_split_a = list(split_map.get(organ, {}).get("ct", {}).get("split_a", []))
        mr_split_b = list(split_map.get(organ, {}).get("mr", {}).get("split_b", []))
        mr_split_a = list(split_map.get(organ, {}).get("mr", {}).get("split_a", []))
        ct_split_b = list(split_map.get(organ, {}).get("ct", {}).get("split_b", []))
        if not ct_split_a or not mr_split_b or not mr_split_a or not ct_split_b:
            per_organ[organ] = {
                "status": "skipped",
                "reason": "insufficient_case_disjoint_cross_modal_support",
            }
            continue

        ct_split_a_stack = _l2_normalize(np.stack([features_by_id[record["sample_id"]] for record in ct_split_a], axis=0))
        mr_split_b_stack = _l2_normalize(np.stack([features_by_id[record["sample_id"]] for record in mr_split_b], axis=0))
        mr_split_a_stack = _l2_normalize(np.stack([features_by_id[record["sample_id"]] for record in mr_split_a], axis=0))
        ct_split_b_stack = _l2_normalize(np.stack([features_by_id[record["sample_id"]] for record in ct_split_b], axis=0))
        first_similarity = float(
            np.clip(
                np.dot(
                    _l2_normalize_vector(np.mean(ct_split_a_stack, axis=0)),
                    _l2_normalize_vector(np.mean(mr_split_b_stack, axis=0)),
                ),
                -1.0,
                1.0,
            )
        )
        second_similarity = float(
            np.clip(
                np.dot(
                    _l2_normalize_vector(np.mean(mr_split_a_stack, axis=0)),
                    _l2_normalize_vector(np.mean(ct_split_b_stack, axis=0)),
                ),
                -1.0,
                1.0,
            )
        )
        mean_similarity = float((first_similarity + second_similarity) / 2.0)
        per_organ[organ] = {
            "status": "ok",
            "cross_modal_centroid_cosine_similarity": mean_similarity,
            "cross_modal_centroid_cosine_distance": float(1.0 - mean_similarity),
            "case_partition_cross_modal_centroid_cosine_similarity": mean_similarity,
            "case_partition_cross_modal_centroid_cosine_distance": float(1.0 - mean_similarity),
        }
    return per_organ


def _compute_holdout_vs_rest_silhouette(
    records: Sequence[Dict[str, Any]],
    features_by_id: Dict[str, np.ndarray],
    supported_organs: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    if silhouette_score is None:
        return {
            organ: {"status": "skipped", "reason": "sklearn_not_available"}
            for organ in sorted(supported_organs)
        }

    feature_matrix = _l2_normalize(
        np.stack([features_by_id[record["sample_id"]] for record in records], axis=0)
    )
    organ_labels = np.asarray([str(record["primary_organ"]) for record in records], dtype=object)
    per_organ: Dict[str, Dict[str, Any]] = {}
    for organ in sorted(supported_organs):
        binary_labels = (organ_labels == organ).astype(np.int32)
        if len(np.unique(binary_labels)) < 2 or np.sum(binary_labels == 1) < 2 or np.sum(binary_labels == 0) < 2:
            per_organ[organ] = {
                "status": "skipped",
                "reason": "insufficient_binary_support",
            }
            continue
        score = float(silhouette_score(feature_matrix, binary_labels, metric="cosine"))
        per_organ[organ] = {
            "status": "ok",
            "heldout_vs_rest_silhouette": score,
        }
    return per_organ


def _compute_nearest_neighbor_purity(
    records: Sequence[Dict[str, Any]],
    features_by_id: Dict[str, np.ndarray],
    supported_organs: Sequence[str],
    neighbor_count: int = 5,
) -> Dict[str, Dict[str, Any]]:
    if len(records) <= 1:
        return {
            organ: {"status": "skipped", "reason": "insufficient_records"}
            for organ in sorted(supported_organs)
        }

    feature_matrix = _l2_normalize(
        np.stack([features_by_id[record["sample_id"]] for record in records], axis=0)
    )
    organ_labels = np.asarray([str(record["primary_organ"]) for record in records], dtype=object)
    similarity = feature_matrix @ feature_matrix.T
    np.fill_diagonal(similarity, -np.inf)
    effective_k = min(int(neighbor_count), len(records) - 1)

    per_organ: Dict[str, Dict[str, Any]] = {}
    for organ in sorted(supported_organs):
        organ_indices = np.flatnonzero(organ_labels == organ)
        if organ_indices.size == 0:
            per_organ[organ] = {"status": "skipped", "reason": "missing_organ_records"}
            continue
        purities = []
        for index in organ_indices:
            ranking = np.argsort(-similarity[index])[:effective_k]
            purities.append(float(np.mean(organ_labels[ranking] == organ)))
        per_organ[organ] = {
            "status": "ok",
            "neighbor_count": effective_k,
            "nearest_neighbor_purity": float(np.mean(purities)) if purities else None,
        }
    return per_organ


def _compute_retrieval_from_records(
    query_records: Sequence[Dict[str, Any]],
    target_records_by_organ: Dict[str, Sequence[Dict[str, Any]]],
    features_by_id: Dict[str, np.ndarray],
    organ: str,
    supported_organs: Sequence[str],
    top_ks: Sequence[int],
    max_queries_per_organ: int | None,
    max_targets_per_organ: int | None,
    seed: int,
    query_group_field: str,
    query_group_value: str,
    target_group_field: str,
    target_group_value: str,
) -> Dict[str, Any]:
    """Evaluate one retrieval direction for one held-out organ.

    Required parameters:
        query_records: query-side records for the held-out organ.
        target_records_by_organ: balanced target candidates keyed by organ.
    """
    target_organs = [
        candidate_organ
        for candidate_organ in supported_organs
        if target_records_by_organ.get(candidate_organ)
    ]
    query_records = list(query_records)
    if not query_records or len(target_organs) < 2 or organ not in target_organs:
        return {
            "status": "skipped",
            "reason": "insufficient_query_or_target_support",
            query_group_field: query_group_value,
            target_group_field: target_group_value,
            "n_queries": len(query_records),
            "n_target_organs": len(target_organs),
        }

    targets_per_organ = min(len(target_records_by_organ[candidate]) for candidate in target_organs)
    if max_targets_per_organ is not None:
        targets_per_organ = min(targets_per_organ, max_targets_per_organ)
    if targets_per_organ <= 0:
        return {
            "status": "skipped",
            "reason": "no_balanced_target_pool",
            query_group_field: query_group_value,
            target_group_field: target_group_value,
        }

    rng = np.random.default_rng(seed)
    if max_queries_per_organ is not None and len(query_records) > max_queries_per_organ:
        query_records = _sample_records(query_records, max_queries_per_organ, rng)

    balanced_targets = [
        target_record
        for candidate_organ in target_organs
        for target_record in _sample_records(target_records_by_organ[candidate_organ], targets_per_organ, rng)
    ]

    query_features = _l2_normalize(np.stack([features_by_id[record["sample_id"]] for record in query_records], axis=0))
    target_features = _l2_normalize(np.stack([features_by_id[record["sample_id"]] for record in balanced_targets], axis=0))
    similarity = query_features @ target_features.T
    target_labels = np.asarray([record["primary_organ"] for record in balanced_targets], dtype=object)

    top_ks = tuple(sorted(set(int(k) for k in top_ks if int(k) > 0)))
    top_counts = {k: 0 for k in top_ks}
    average_precision_total = 0.0
    hardest_negative_counter: Counter[str] = Counter()
    per_query_margin: list[float] = []
    per_query_positive_similarity: list[float] = []
    per_query_negative_similarity: list[float] = []

    for query_index in range(len(query_records)):
        ranking = np.argsort(-similarity[query_index])
        ranked_labels = target_labels[ranking]
        relevance = ranked_labels == organ
        average_precision_total += _average_precision(relevance)
        for k in top_ks:
            top_counts[k] += int(np.any(relevance[:k]))

        positive_mask = target_labels == organ
        negative_mask = ~positive_mask
        positive_scores = similarity[query_index][positive_mask]
        negative_scores = similarity[query_index][negative_mask]
        if positive_scores.size and negative_scores.size:
            positive_similarity = float(np.max(positive_scores))
            hardest_negative_index = int(np.argmax(negative_scores))
            hardest_negative_similarity = float(negative_scores[hardest_negative_index])
            hardest_negative_label = np.asarray(target_labels[negative_mask], dtype=object)[hardest_negative_index]
            hardest_negative_counter[str(hardest_negative_label)] += 1
            per_query_margin.append(positive_similarity - hardest_negative_similarity)
            per_query_positive_similarity.append(positive_similarity)
            per_query_negative_similarity.append(hardest_negative_similarity)

    n_queries = len(query_records)
    return {
        "status": "ok",
        query_group_field: query_group_value,
        target_group_field: target_group_value,
        "n_queries": n_queries,
        "n_target_organs": len(target_organs),
        "targets_per_organ": targets_per_organ,
        "overall": {
            **{_top_metric_key(k): float(top_counts[k] / n_queries) for k in top_ks},
            "map": float(average_precision_total / n_queries),
        },
        "balanced_chance": {
            _top_metric_key(k): _balanced_top_chance(len(target_organs), targets_per_organ, k)
            for k in top_ks
        },
        "cross_modal_margin": {
            "margin": _summary_stats(per_query_margin),
            "positive_similarity": _summary_stats(per_query_positive_similarity),
            "hardest_negative_similarity": _summary_stats(per_query_negative_similarity),
            "hardest_negative_organs": dict(sorted(hardest_negative_counter.items())),
        },
    }


def _compute_organ_direction_retrieval(
    grouped_records: Dict[str, Dict[str, list[Dict[str, Any]]]],
    features_by_id: Dict[str, np.ndarray],
    organ: str,
    supported_organs: Sequence[str],
    query_modality: str,
    target_modality: str,
    top_ks: Sequence[int],
    max_queries_per_organ: int | None,
    max_targets_per_organ: int | None,
    seed: int,
) -> Dict[str, Any]:
    return _compute_retrieval_from_records(
        query_records=grouped_records.get(organ, {}).get(query_modality, []),
        target_records_by_organ={
            candidate_organ: grouped_records.get(candidate_organ, {}).get(target_modality, [])
            for candidate_organ in supported_organs
        },
        features_by_id=features_by_id,
        organ=organ,
        supported_organs=supported_organs,
        top_ks=top_ks,
        max_queries_per_organ=max_queries_per_organ,
        max_targets_per_organ=max_targets_per_organ,
        seed=seed,
        query_group_field="query_modality",
        query_group_value=query_modality,
        target_group_field="target_modality",
        target_group_value=target_modality,
    )


def _compute_within_modality_direction_retrieval(
    split_map: Dict[str, Dict[str, Dict[str, list[Dict[str, Any]]]]],
    features_by_id: Dict[str, np.ndarray],
    organ: str,
    supported_organs: Sequence[str],
    modality: str,
    query_split: str,
    target_split: str,
    top_ks: Sequence[int],
    max_queries_per_organ: int | None,
    max_targets_per_organ: int | None,
    seed: int,
) -> Dict[str, Any]:
    result = _compute_retrieval_from_records(
        query_records=split_map.get(organ, {}).get(modality, {}).get(query_split, []),
        target_records_by_organ={
            candidate_organ: split_map.get(candidate_organ, {}).get(modality, {}).get(target_split, [])
            for candidate_organ in supported_organs
        },
        features_by_id=features_by_id,
        organ=organ,
        supported_organs=supported_organs,
        top_ks=top_ks,
        max_queries_per_organ=max_queries_per_organ,
        max_targets_per_organ=max_targets_per_organ,
        seed=seed,
        query_group_field="query_partition",
        query_group_value=query_split,
        target_group_field="target_partition",
        target_group_value=target_split,
    )
    result["modality"] = modality
    return result


def _compute_case_partition_direction_retrieval(
    split_map: Dict[str, Dict[str, Dict[str, list[Dict[str, Any]]]]],
    features_by_id: Dict[str, np.ndarray],
    organ: str,
    supported_organs: Sequence[str],
    query_modality: str,
    target_modality: str,
    query_split: str,
    target_split: str,
    top_ks: Sequence[int],
    max_queries_per_organ: int | None,
    max_targets_per_organ: int | None,
    seed: int,
) -> Dict[str, Any]:
    return _compute_retrieval_from_records(
        query_records=split_map.get(organ, {}).get(query_modality, {}).get(query_split, []),
        target_records_by_organ={
            candidate_organ: split_map.get(candidate_organ, {}).get(target_modality, {}).get(target_split, [])
            for candidate_organ in supported_organs
        },
        features_by_id=features_by_id,
        organ=organ,
        supported_organs=supported_organs,
        top_ks=top_ks,
        max_queries_per_organ=max_queries_per_organ,
        max_targets_per_organ=max_targets_per_organ,
        seed=seed,
        query_group_field="query_partition",
        query_group_value=query_split,
        target_group_field="target_partition",
        target_group_value=target_split,
    )


def _combine_bidirectional_metrics(first: Dict[str, Any], second: Dict[str, Any]) -> Dict[str, Any] | None:
    """Average CT->MR and MR->CT metrics when both directions are valid."""
    if first.get("status") != "ok" or second.get("status") != "ok":
        return None

    overall_keys = sorted(set(first["overall"].keys()) & set(second["overall"].keys()))
    margin_mean_first = first["cross_modal_margin"]["margin"].get("mean")
    margin_mean_second = second["cross_modal_margin"]["margin"].get("mean")
    return {
        **{
            key: float((first["overall"][key] + second["overall"][key]) / 2.0)
            for key in overall_keys
        },
        "cross_modal_margin": (
            None
            if margin_mean_first is None or margin_mean_second is None
            else float((margin_mean_first + margin_mean_second) / 2.0)
        ),
    }


def _mean_metric_across_organs(organ_payloads: Dict[str, Dict[str, Any]], metric_name: str, exclude: str) -> float | None:
    """Compute one macro reference metric while excluding the held-out organ."""
    values = []
    for organ, payload in organ_payloads.items():
        if organ == exclude:
            continue
        bidirectional = payload.get("bidirectional_mean") or {}
        value = bidirectional.get(metric_name)
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return float(np.mean(values))


def _generalization_gap(reference_value: float | None, held_out_value: float | None, metric_name: str) -> float | None:
    if reference_value is None or held_out_value is None:
        return None
    if metric_name in LOWER_IS_BETTER_METRICS:
        return float(held_out_value - reference_value)
    return float(reference_value - held_out_value)


def evaluate_leave_one_organ_out_transfer(
    features_by_id: Dict[str, np.ndarray],
    samples: Sequence[Dict[str, Any]],
    supported_organs: Sequence[str],
    required_modalities: Sequence[str] = ("ct", "mr"),
    top_ks: Sequence[int] = (1, 5),
    max_queries_per_organ: int | None = None,
    max_targets_per_organ: int | None = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run the full Module 3 leave-one-organ-out evaluation.

    Required parameters:
        features_by_id: mapping from ``sample_id`` to embedding vector.
        samples: normalized Module 3 samples from ``load_module3_manifest()``.
        supported_organs: organs that satisfy the Module 3 hold-out support
            rule.

    Optional controls:
        top_ks: retrieval cutoffs to report.
        max_queries_per_organ: cap the number of query samples per organ.
        max_targets_per_organ: cap the number of target samples per organ.
        seed: random seed used for balanced sampling.

    Returns:
        A JSON-serializable dictionary with macro metrics and per-organ details.
    """
    required_modalities = tuple(str(modality).lower() for modality in required_modalities)
    records = _eligible_records(samples, features_by_id, supported_organs, required_modalities)
    silhouette_metrics = _compute_holdout_vs_rest_silhouette(records, features_by_id, supported_organs)
    nn_purity_metrics = _compute_nearest_neighbor_purity(records, features_by_id, supported_organs)
    surface_scope = "within_modality" if len(required_modalities) == 1 else "cross_modality"

    if surface_scope == "cross_modality" and tuple(required_modalities) != ("ct", "mr"):
        raise ValueError(
            f"Cross-modality Module 3 currently supports required_modalities=('ct', 'mr'), got {required_modalities}"
        )

    if surface_scope == "within_modality":
        evaluation_modality = required_modalities[0]
        split_map = build_global_case_split_map(
            records,
            supported_organs,
            required_modalities,
            seed=seed,
        )
        centroid_metrics = _compute_case_partition_centroid_distances(
            split_map,
            features_by_id,
            supported_organs,
            required_modalities,
        )
    else:
        evaluation_modality = None
        split_map = build_global_case_split_map(
            records,
            supported_organs,
            required_modalities,
            seed=seed,
        )
        centroid_metrics = _compute_case_partition_centroid_distances(
            split_map,
            features_by_id,
            supported_organs,
            required_modalities,
        )

    organ_payloads: Dict[str, Dict[str, Any]] = {}
    for organ_index, organ in enumerate(sorted(supported_organs)):
        if surface_scope == "within_modality":
            first_direction = _compute_within_modality_direction_retrieval(
                split_map,
                features_by_id,
                organ,
                supported_organs,
                modality=evaluation_modality,
                query_split="split_a",
                target_split="split_b",
                top_ks=top_ks,
                max_queries_per_organ=max_queries_per_organ,
                max_targets_per_organ=max_targets_per_organ,
                seed=seed + organ_index,
            )
            second_direction = _compute_within_modality_direction_retrieval(
                split_map,
                features_by_id,
                organ,
                supported_organs,
                modality=evaluation_modality,
                query_split="split_b",
                target_split="split_a",
                top_ks=top_ks,
                max_queries_per_organ=max_queries_per_organ,
                max_targets_per_organ=max_targets_per_organ,
                seed=seed + organ_index + 997,
            )
            feature_space_geometry = {
                "case_partition_alignment": centroid_metrics.get(organ),
                "heldout_vs_rest_silhouette": silhouette_metrics.get(organ),
                "nearest_neighbor_purity": nn_purity_metrics.get(organ),
            }
            organ_payload = {
                "split_a_to_split_b": first_direction,
                "split_b_to_split_a": second_direction,
            }
        else:
            first_direction = _compute_case_partition_direction_retrieval(
                split_map,
                features_by_id,
                organ,
                supported_organs,
                query_modality="ct",
                target_modality="mr",
                query_split="split_a",
                target_split="split_b",
                top_ks=top_ks,
                max_queries_per_organ=max_queries_per_organ,
                max_targets_per_organ=max_targets_per_organ,
                seed=seed + organ_index,
            )
            second_direction = _compute_case_partition_direction_retrieval(
                split_map,
                features_by_id,
                organ,
                supported_organs,
                query_modality="mr",
                target_modality="ct",
                query_split="split_b",
                target_split="split_a",
                top_ks=top_ks,
                max_queries_per_organ=max_queries_per_organ,
                max_targets_per_organ=max_targets_per_organ,
                seed=seed + organ_index + 997,
            )
            feature_space_geometry = {
                "case_partition_cross_modal_alignment": centroid_metrics.get(organ),
                "heldout_vs_rest_silhouette": silhouette_metrics.get(organ),
                "nearest_neighbor_purity": nn_purity_metrics.get(organ),
            }
            organ_payload = {
                "ct_to_mr": first_direction,
                "mr_to_ct": second_direction,
            }

        organ_payloads[organ] = {
            "fold_context": {
                "held_out_during_adaptation": organ,
                "seen_during_adaptation": [candidate for candidate in sorted(supported_organs) if candidate != organ],
                "fold_policy": "leave_one_organ_out",
                "surface_scope": surface_scope,
                "case_split_policy": "global_case_disjoint_partitions",
            },
            **organ_payload,
            "feature_space_geometry": feature_space_geometry,
            "bidirectional_mean": _combine_bidirectional_metrics(first_direction, second_direction),
        }

        bidirectional = organ_payloads[organ].get("bidirectional_mean")
        if bidirectional is not None:
            centroid_payload = centroid_metrics.get(organ, {})
            silhouette_payload = silhouette_metrics.get(organ, {})
            purity_payload = nn_purity_metrics.get(organ, {})
            if centroid_payload.get("status") == "ok":
                centroid_distance = centroid_payload.get("cross_modal_centroid_cosine_distance")
                if centroid_distance is None:
                    centroid_distance = centroid_payload.get("within_modality_split_centroid_cosine_distance")
                bidirectional["heldout_centroid_distance"] = centroid_distance
            if silhouette_payload.get("status") == "ok":
                bidirectional["heldout_silhouette"] = silhouette_payload.get("heldout_vs_rest_silhouette")
            if purity_payload.get("status") == "ok":
                bidirectional["nearest_neighbor_purity"] = purity_payload.get("nearest_neighbor_purity")

    macro_values: Dict[str, list[float]] = defaultdict(list)
    for organ, payload in organ_payloads.items():
        bidirectional = payload.get("bidirectional_mean") or {}
        for metric_name, value in bidirectional.items():
            if value is not None:
                macro_values[metric_name].append(float(value))

    for organ, payload in organ_payloads.items():
        bidirectional = payload.get("bidirectional_mean") or {}
        reference = {
            metric_name: _mean_metric_across_organs(organ_payloads, metric_name, exclude=organ)
            for metric_name in bidirectional.keys()
        }
        payload["in_distribution_reference"] = reference
        payload["generalization_gap"] = {
            metric_name: _generalization_gap(reference.get(metric_name), bidirectional.get(metric_name), metric_name)
            for metric_name in bidirectional.keys()
        }

    evaluated_organs = [organ for organ, payload in organ_payloads.items() if payload.get("bidirectional_mean")]
    return {
        "status": "ok" if evaluated_organs else "skipped",
        "surface_scope": surface_scope,
        "case_split_policy": "global_case_disjoint_partitions",
        "required_modalities": list(required_modalities),
        "held_out_definition": "held_out_during_adaptation",
        "fold_policy": "leave_one_organ_out",
        "n_supported_organs": len(list(supported_organs)),
        "n_evaluated_organs": len(evaluated_organs),
        "supported_organs": list(sorted(supported_organs)),
        "macro_mean": {
            metric_name: float(np.mean(values))
            for metric_name, values in sorted(macro_values.items())
            if values
        },
        "per_organ": organ_payloads,
    }
