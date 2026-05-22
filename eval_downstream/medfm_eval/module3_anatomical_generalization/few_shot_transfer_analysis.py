"""Fixed-budget few-shot transfer analysis for Module 3 anatomical generalization.

This module implements the Surface B evaluation layer: for each held-out organ,
it samples a balanced few-shot support set across organs and modalities, trains a
lightweight linear probe, and measures how well the held-out organ transfers to
disjoint query examples.

The protocol is intentionally simple and deterministic:

    1. keep only samples with valid features and supported organs
    2. sample a fixed support budget per organ and modality
    3. build a balanced query pool from the remaining records
    4. train one multiclass linear probe on the support pool
    5. report held-out-organ query accuracy, in-distribution reference accuracy,
       generalization gap, and transfer efficiency

This gives Module 3 a task-facing transfer surface without introducing a new
segmentation trainer inside the package.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from case_disjoint_partitions import build_global_case_split_map


def _eligible_records(
    samples: Sequence[Dict[str, Any]],
    features_by_id: Dict[str, np.ndarray],
    supported_organs: Sequence[str],
    required_modalities: Sequence[str],
) -> list[Dict[str, Any]]:
    supported = set(supported_organs)
    required = set(required_modalities)
    return [
        sample
        for sample in samples
        if sample.get("sample_id") in features_by_id
        and sample.get("primary_organ") in supported
        and sample.get("modality") in required
    ]


def _group_by_organ_and_modality(
    records: Sequence[Dict[str, Any]],
    required_modalities: Sequence[str],
) -> Dict[str, Dict[str, list[Dict[str, Any]]]]:
    grouped: Dict[str, Dict[str, list[Dict[str, Any]]]] = defaultdict(
        lambda: {str(modality): [] for modality in required_modalities}
    )
    for record in records:
        grouped[str(record["primary_organ"])][str(record["modality"])].append(record)
    return grouped


def _sample_without_replacement(
    records: Sequence[Dict[str, Any]],
    sample_count: int,
    rng: np.random.Generator,
) -> list[Dict[str, Any]]:
    entries = list(records)
    if sample_count >= len(entries):
        return entries
    indices = rng.choice(len(entries), size=sample_count, replace=False)
    return [entries[int(index)] for index in indices]


def _build_support_and_query_from_split_map(
    split_map: Dict[str, Dict[str, Dict[str, list[Dict[str, Any]]]]],
    supported_organs: Sequence[str],
    required_modalities: Sequence[str],
    support_per_modality: int,
    query_per_modality: int | None,
    rng: np.random.Generator,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], int] | tuple[None, None, None]:
    min_query_count = None
    for organ in supported_organs:
        for modality in required_modalities:
            support_pool = list(split_map.get(organ, {}).get(modality, {}).get("split_a", []))
            query_pool = list(split_map.get(organ, {}).get(modality, {}).get("split_b", []))
            if len(support_pool) < support_per_modality or not query_pool:
                return None, None, None
            min_query_count = len(query_pool) if min_query_count is None else min(min_query_count, len(query_pool))

    if min_query_count is None or min_query_count <= 0:
        return None, None, None

    balanced_query_count = min_query_count if query_per_modality is None else min(min_query_count, int(query_per_modality))
    if balanced_query_count <= 0:
        return None, None, None

    support_records: list[Dict[str, Any]] = []
    query_records: list[Dict[str, Any]] = []
    for organ in supported_organs:
        for modality in required_modalities:
            support_pool = list(split_map[organ][modality]["split_a"])
            query_pool = list(split_map[organ][modality]["split_b"])
            support_records.extend(_sample_without_replacement(support_pool, support_per_modality, rng))
            query_records.extend(_sample_without_replacement(query_pool, balanced_query_count, rng))

    return support_records, query_records, balanced_query_count


def _build_balanced_support_and_query_sets(
    grouped_records: Dict[str, Dict[str, list[Dict[str, Any]]]],
    supported_organs: Sequence[str],
    required_modalities: Sequence[str],
    support_per_modality: int,
    query_per_modality: int | None,
    rng: np.random.Generator,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], int] | tuple[None, None, None]:
    min_remaining = None
    for organ in supported_organs:
        for modality in required_modalities:
            modality_records = list(grouped_records.get(organ, {}).get(modality, []))
            if len(modality_records) <= support_per_modality:
                return None, None, None
            remaining = len(modality_records) - support_per_modality
            min_remaining = remaining if min_remaining is None else min(min_remaining, remaining)

    if min_remaining is None or min_remaining <= 0:
        return None, None, None

    balanced_query_count = min_remaining if query_per_modality is None else min(min_remaining, int(query_per_modality))
    if balanced_query_count <= 0:
        return None, None, None

    support_records: list[Dict[str, Any]] = []
    query_records: list[Dict[str, Any]] = []
    for organ in supported_organs:
        for modality in required_modalities:
            modality_records = list(grouped_records[organ][modality])
            support_subset = _sample_without_replacement(modality_records, support_per_modality, rng)
            support_ids = {record["sample_id"] for record in support_subset}
            query_candidates = [record for record in modality_records if record["sample_id"] not in support_ids]
            query_subset = _sample_without_replacement(query_candidates, balanced_query_count, rng)
            support_records.extend(support_subset)
            query_records.extend(query_subset)

    return support_records, query_records, balanced_query_count


def _sample_case_representatives(
    records: Sequence[Dict[str, Any]],
    case_count: int,
    rng: np.random.Generator,
) -> tuple[list[Dict[str, Any]], set[str]]:
    grouped_by_case: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        case_id = str(record.get("source_case_id") or record.get("patient_id") or record.get("sample_id"))
        grouped_by_case[case_id].append(record)

    case_ids = sorted(grouped_by_case)
    if case_count >= len(case_ids):
        selected_case_ids = case_ids
    else:
        selected_case_ids = [case_ids[int(index)] for index in rng.choice(len(case_ids), size=case_count, replace=False)]

    selected_records = []
    for case_id in selected_case_ids:
        case_records = grouped_by_case[case_id]
        selected_records.append(case_records[int(rng.integers(len(case_records)))])
    return selected_records, set(selected_case_ids)


def _build_case_disjoint_support_and_query_sets(
    grouped_records: Dict[str, Dict[str, list[Dict[str, Any]]]],
    supported_organs: Sequence[str],
    required_modalities: Sequence[str],
    support_per_modality: int,
    query_per_modality: int | None,
    rng: np.random.Generator,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], int] | tuple[None, None, None]:
    min_remaining_cases = None
    for organ in supported_organs:
        for modality in required_modalities:
            modality_records = list(grouped_records.get(organ, {}).get(modality, []))
            unique_case_ids = {
                str(record.get("source_case_id") or record.get("patient_id") or record.get("sample_id"))
                for record in modality_records
            }
            if len(unique_case_ids) <= support_per_modality:
                return None, None, None
            remaining_cases = len(unique_case_ids) - support_per_modality
            min_remaining_cases = (
                remaining_cases if min_remaining_cases is None else min(min_remaining_cases, remaining_cases)
            )

    if min_remaining_cases is None or min_remaining_cases <= 0:
        return None, None, None

    balanced_query_count = (
        min_remaining_cases if query_per_modality is None else min(min_remaining_cases, int(query_per_modality))
    )
    if balanced_query_count <= 0:
        return None, None, None

    support_records: list[Dict[str, Any]] = []
    query_records: list[Dict[str, Any]] = []
    for organ in supported_organs:
        for modality in required_modalities:
            modality_records = list(grouped_records[organ][modality])
            support_subset, support_case_ids = _sample_case_representatives(modality_records, support_per_modality, rng)
            query_candidates = [
                record
                for record in modality_records
                if str(record.get("source_case_id") or record.get("patient_id") or record.get("sample_id")) not in support_case_ids
            ]
            query_subset, _ = _sample_case_representatives(query_candidates, balanced_query_count, rng)
            support_records.extend(support_subset)
            query_records.extend(query_subset)

    return support_records, query_records, balanced_query_count


def _organ_accuracy(
    records: Sequence[Dict[str, Any]],
    predictions: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    grouped_true: Dict[str, list[int]] = defaultdict(list)
    grouped_pred: Dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        organ = str(record["primary_organ"])
        grouped_true[organ].append(int(labels[index]))
        grouped_pred[organ].append(int(predictions[index]))
    return {
        organ: float(accuracy_score(grouped_true[organ], grouped_pred[organ]))
        for organ in sorted(grouped_true)
    }


def _accuracy_by_modality(
    records: Sequence[Dict[str, Any]],
    predictions: np.ndarray,
    labels: np.ndarray,
    target_organ: str,
) -> Dict[str, float]:
    grouped_true: Dict[str, list[int]] = defaultdict(list)
    grouped_pred: Dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if str(record["primary_organ"]) != target_organ:
            continue
        modality = str(record["modality"])
        grouped_true[modality].append(int(labels[index]))
        grouped_pred[modality].append(int(predictions[index]))
    return {
        modality: float(accuracy_score(grouped_true[modality], grouped_pred[modality]))
        for modality in sorted(grouped_true)
    }


def _predict_nearest_centroid(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(sorted(set(int(label) for label in support_labels)), dtype=np.int64)
    centroids = np.stack(
        [support_features[support_labels == label].mean(axis=0) for label in labels],
        axis=0,
    )
    squared_distances = np.sum((query_features[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return labels[np.argmin(squared_distances, axis=1)]


def _relative_recovery(adapted_score: float | None, baseline_score: float | None) -> float | None:
    if adapted_score is None or baseline_score is None:
        return None
    denominator = 1.0 - float(baseline_score)
    if denominator <= 1e-12:
        return None
    return float((float(adapted_score) - float(baseline_score)) / denominator)


def _summarize_predictions(
    *,
    records: Sequence[Dict[str, Any]],
    predictions: np.ndarray,
    labels: np.ndarray,
    target_organ: str,
) -> Dict[str, Any]:
    balanced_accuracy = float(balanced_accuracy_score(labels, predictions))
    organ_accuracy = _organ_accuracy(records, predictions, labels)
    held_out_accuracy = organ_accuracy.get(target_organ)
    in_distribution_scores = [score for candidate, score in organ_accuracy.items() if candidate != target_organ]
    in_distribution_accuracy = _mean(in_distribution_scores)
    generalization_gap = (
        None
        if held_out_accuracy is None or in_distribution_accuracy is None
        else float(in_distribution_accuracy - held_out_accuracy)
    )
    transfer_efficiency = (
        None
        if held_out_accuracy is None or in_distribution_accuracy in (None, 0.0)
        else float(held_out_accuracy / in_distribution_accuracy)
    )
    return {
        "balanced_accuracy": balanced_accuracy,
        "held_out_accuracy": held_out_accuracy,
        "in_distribution_accuracy": in_distribution_accuracy,
        "generalization_gap": generalization_gap,
        "transfer_efficiency": transfer_efficiency,
        "per_organ_query_accuracy": organ_accuracy,
        "held_out_accuracy_by_modality": _accuracy_by_modality(
            records,
            predictions,
            labels,
            target_organ=target_organ,
        ),
    }


def _mean(values: Iterable[float]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return float(np.mean(numeric))


def _std(values: Iterable[float]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if len(numeric) < 2:
        return None
    return float(np.std(numeric, ddof=1))


def evaluate_leave_one_organ_out_few_shot_transfer(
    features_by_id: Dict[str, np.ndarray],
    samples: Sequence[Dict[str, Any]],
    supported_organs: Sequence[str],
    required_modalities: Sequence[str] = ("ct", "mr"),
    support_per_modality: int = 2,
    query_per_modality: int | None = None,
    seeds: Sequence[int] = (42, 123, 456),
) -> Dict[str, Any]:
    """Run the Surface B fixed-budget few-shot transfer evaluation.

    The held-out organ remains the target of transfer analysis, but Surface B
    allows a fixed, balanced k-shot support budget for that organ so the probe
    can be adapted and evaluated on disjoint held-out queries.
    """
    records = _eligible_records(samples, features_by_id, supported_organs, required_modalities)
    surface_scope = "within_modality" if len(required_modalities) == 1 else "cross_modality"
    transfer_protocol = "fixed_budget_few_shot_transfer_case_disjoint"

    label_to_index = {organ: index for index, organ in enumerate(sorted(supported_organs))}
    organ_payloads: Dict[str, Dict[str, Any]] = {}
    macro_fewshot_scores: list[float] = []
    macro_balanced_accuracies: list[float] = []
    macro_seen_scores: list[float] = []
    macro_gaps: list[float] = []
    macro_efficiencies: list[float] = []
    macro_centroid_balanced_accuracies: list[float] = []
    macro_centroid_held_out_scores: list[float] = []
    macro_adaptation_gains: list[float] = []
    macro_held_out_adaptation_gains: list[float] = []
    macro_relative_recoveries: list[float] = []

    for organ in sorted(supported_organs):
        seed_results: list[Dict[str, Any]] = []
        for seed in seeds:
            rng = np.random.default_rng(int(seed))
            split_map = build_global_case_split_map(
                records=records,
                supported_organs=supported_organs,
                required_modalities=required_modalities,
                seed=int(seed),
                min_count_by_partition={
                    "split_a": int(support_per_modality),
                    "split_b": max(1, int(query_per_modality)) if query_per_modality is not None else 1,
                },
            )
            if not split_map:
                continue
            support_records, query_records, balanced_query_count = _build_support_and_query_from_split_map(
                split_map,
                supported_organs,
                required_modalities,
                support_per_modality=support_per_modality,
                query_per_modality=query_per_modality,
                rng=rng,
            )
            if support_records is None or query_records is None:
                continue

            support_features = np.stack([features_by_id[record["sample_id"]] for record in support_records], axis=0)
            query_features = np.stack([features_by_id[record["sample_id"]] for record in query_records], axis=0)
            support_labels = np.asarray([label_to_index[str(record["primary_organ"])] for record in support_records], dtype=np.int64)
            query_labels = np.asarray([label_to_index[str(record["primary_organ"])] for record in query_records], dtype=np.int64)

            scaler = StandardScaler()
            support_scaled = scaler.fit_transform(support_features)
            query_scaled = scaler.transform(query_features)

            classifier = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                random_state=int(seed),
            )
            classifier.fit(support_scaled, support_labels)
            predictions = classifier.predict(query_scaled)
            probe_summary = _summarize_predictions(
                records=query_records,
                predictions=predictions,
                labels=query_labels,
                target_organ=organ,
            )
            centroid_predictions = _predict_nearest_centroid(support_scaled, support_labels, query_scaled)
            centroid_summary = _summarize_predictions(
                records=query_records,
                predictions=centroid_predictions,
                labels=query_labels,
                target_organ=organ,
            )
            held_out_query_records = [record for record in query_records if str(record["primary_organ"]) == organ]
            adaptation_gain_balanced_accuracy = float(
                probe_summary["balanced_accuracy"] - centroid_summary["balanced_accuracy"]
            )
            adaptation_gain_held_out_accuracy = (
                None
                if probe_summary["held_out_accuracy"] is None or centroid_summary["held_out_accuracy"] is None
                else float(probe_summary["held_out_accuracy"] - centroid_summary["held_out_accuracy"])
            )

            seed_results.append(
                {
                    "seed": int(seed),
                    "support_per_modality": int(support_per_modality),
                    "query_per_modality": int(balanced_query_count),
                    "n_support_samples": int(len(support_records)),
                    "n_query_samples": int(len(query_records)),
                    "held_out_query_samples": int(len(held_out_query_records)),
                    "in_distribution_query_samples": int(len(query_records) - len(held_out_query_records)),
                    "balanced_accuracy": probe_summary["balanced_accuracy"],
                    "fewshot_probe_score": probe_summary["held_out_accuracy"],
                    "held_out_accuracy": probe_summary["held_out_accuracy"],
                    "in_distribution_accuracy": probe_summary["in_distribution_accuracy"],
                    "generalization_gap": probe_summary["generalization_gap"],
                    "transfer_efficiency": probe_summary["transfer_efficiency"],
                    "per_organ_query_accuracy": probe_summary["per_organ_query_accuracy"],
                    "held_out_accuracy_by_modality": probe_summary["held_out_accuracy_by_modality"],
                    "nearest_centroid_balanced_accuracy": centroid_summary["balanced_accuracy"],
                    "nearest_centroid_held_out_accuracy": centroid_summary["held_out_accuracy"],
                    "nearest_centroid_in_distribution_accuracy": centroid_summary["in_distribution_accuracy"],
                    "nearest_centroid_generalization_gap": centroid_summary["generalization_gap"],
                    "nearest_centroid_transfer_efficiency": centroid_summary["transfer_efficiency"],
                    "nearest_centroid_per_organ_query_accuracy": centroid_summary["per_organ_query_accuracy"],
                    "nearest_centroid_held_out_accuracy_by_modality": centroid_summary["held_out_accuracy_by_modality"],
                    "adaptation_gain_balanced_accuracy": adaptation_gain_balanced_accuracy,
                    "adaptation_gain_held_out_accuracy": adaptation_gain_held_out_accuracy,
                    "relative_recovery_balanced_accuracy": _relative_recovery(
                        probe_summary["balanced_accuracy"],
                        centroid_summary["balanced_accuracy"],
                    ),
                    "relative_recovery_held_out_accuracy": _relative_recovery(
                        probe_summary["held_out_accuracy"],
                        centroid_summary["held_out_accuracy"],
                    ),
                }
            )

        if not seed_results:
            organ_payloads[organ] = {
                "status": "skipped",
                "reason": "insufficient_support_for_few_shot_surface",
                "fold_context": {
                    "held_out_during_adaptation": organ,
                    "seen_during_adaptation": [candidate for candidate in sorted(supported_organs) if candidate != organ],
                    "fold_policy": "leave_one_organ_out",
                    "surface_scope": surface_scope,
                    "case_split_policy": "global_case_disjoint_partitions",
                },
            }
            continue

        fewshot_scores = [result["fewshot_probe_score"] for result in seed_results if result.get("fewshot_probe_score") is not None]
        balanced_scores = [result["balanced_accuracy"] for result in seed_results if result.get("balanced_accuracy") is not None]
        seen_scores = [result["in_distribution_accuracy"] for result in seed_results if result.get("in_distribution_accuracy") is not None]
        gaps = [result["generalization_gap"] for result in seed_results if result.get("generalization_gap") is not None]
        efficiencies = [result["transfer_efficiency"] for result in seed_results if result.get("transfer_efficiency") is not None]
        centroid_balanced_scores = [
            result["nearest_centroid_balanced_accuracy"]
            for result in seed_results
            if result.get("nearest_centroid_balanced_accuracy") is not None
        ]
        centroid_held_out_scores = [
            result["nearest_centroid_held_out_accuracy"]
            for result in seed_results
            if result.get("nearest_centroid_held_out_accuracy") is not None
        ]
        adaptation_gains = [
            result["adaptation_gain_balanced_accuracy"]
            for result in seed_results
            if result.get("adaptation_gain_balanced_accuracy") is not None
        ]
        held_out_adaptation_gains = [
            result["adaptation_gain_held_out_accuracy"]
            for result in seed_results
            if result.get("adaptation_gain_held_out_accuracy") is not None
        ]
        relative_recoveries = [
            result["relative_recovery_balanced_accuracy"]
            for result in seed_results
            if result.get("relative_recovery_balanced_accuracy") is not None
        ]

        organ_summary = {
            "status": "ok",
            "fold_context": {
                "held_out_during_adaptation": organ,
                "seen_during_adaptation": [candidate for candidate in sorted(supported_organs) if candidate != organ],
                "fold_policy": "leave_one_organ_out",
                "surface_scope": surface_scope,
                "surface_b_protocol": transfer_protocol,
                "case_split_policy": "global_case_disjoint_partitions",
            },
            "seed_results": seed_results,
            "balanced_accuracy": _mean(balanced_scores),
            "balanced_accuracy_std": _std(balanced_scores),
            "fewshot_probe_score": _mean(fewshot_scores),
            "fewshot_probe_score_std": _std(fewshot_scores),
            "held_out_accuracy": _mean(fewshot_scores),
            "held_out_accuracy_std": _std(fewshot_scores),
            "in_distribution_accuracy": _mean(seen_scores),
            "generalization_gap": _mean(gaps),
            "transfer_efficiency": _mean(efficiencies),
            "nearest_centroid_balanced_accuracy": _mean(centroid_balanced_scores),
            "nearest_centroid_balanced_accuracy_std": _std(centroid_balanced_scores),
            "nearest_centroid_held_out_accuracy": _mean(centroid_held_out_scores),
            "nearest_centroid_held_out_accuracy_std": _std(centroid_held_out_scores),
            "adaptation_gain_balanced_accuracy": _mean(adaptation_gains),
            "adaptation_gain_balanced_accuracy_std": _std(adaptation_gains),
            "adaptation_gain_held_out_accuracy": _mean(held_out_adaptation_gains),
            "adaptation_gain_held_out_accuracy_std": _std(held_out_adaptation_gains),
            "relative_recovery_balanced_accuracy": _mean(relative_recoveries),
            "relative_recovery_balanced_accuracy_std": _std(relative_recoveries),
        }
        organ_payloads[organ] = organ_summary

        if organ_summary["balanced_accuracy"] is not None:
            macro_balanced_accuracies.append(float(organ_summary["balanced_accuracy"]))
        if organ_summary["fewshot_probe_score"] is not None:
            macro_fewshot_scores.append(float(organ_summary["fewshot_probe_score"]))
        if organ_summary["in_distribution_accuracy"] is not None:
            macro_seen_scores.append(float(organ_summary["in_distribution_accuracy"]))
        if organ_summary["generalization_gap"] is not None:
            macro_gaps.append(float(organ_summary["generalization_gap"]))
        if organ_summary["transfer_efficiency"] is not None:
            macro_efficiencies.append(float(organ_summary["transfer_efficiency"]))
        if organ_summary["nearest_centroid_balanced_accuracy"] is not None:
            macro_centroid_balanced_accuracies.append(float(organ_summary["nearest_centroid_balanced_accuracy"]))
        if organ_summary["nearest_centroid_held_out_accuracy"] is not None:
            macro_centroid_held_out_scores.append(float(organ_summary["nearest_centroid_held_out_accuracy"]))
        if organ_summary["adaptation_gain_balanced_accuracy"] is not None:
            macro_adaptation_gains.append(float(organ_summary["adaptation_gain_balanced_accuracy"]))
        if organ_summary["adaptation_gain_held_out_accuracy"] is not None:
            macro_held_out_adaptation_gains.append(float(organ_summary["adaptation_gain_held_out_accuracy"]))
        if organ_summary["relative_recovery_balanced_accuracy"] is not None:
            macro_relative_recoveries.append(float(organ_summary["relative_recovery_balanced_accuracy"]))

    evaluated_organs = [organ for organ, payload in organ_payloads.items() if payload.get("status") == "ok"]
    macro_seed_results = []
    for seed in seeds:
        seed_payloads = [
            result
            for payload in organ_payloads.values()
            if payload.get("status") == "ok"
            for result in payload.get("seed_results", [])
            if int(result.get("seed")) == int(seed)
        ]
        if not seed_payloads:
            continue
        macro_seed_results.append(
            {
                "seed": int(seed),
                "balanced_accuracy": _mean(result.get("balanced_accuracy") for result in seed_payloads),
                "fewshot_probe_score": _mean(result.get("fewshot_probe_score") for result in seed_payloads),
                "held_out_accuracy": _mean(result.get("held_out_accuracy") for result in seed_payloads),
                "nearest_centroid_balanced_accuracy": _mean(
                    result.get("nearest_centroid_balanced_accuracy") for result in seed_payloads
                ),
                "nearest_centroid_held_out_accuracy": _mean(
                    result.get("nearest_centroid_held_out_accuracy") for result in seed_payloads
                ),
                "adaptation_gain_balanced_accuracy": _mean(
                    result.get("adaptation_gain_balanced_accuracy") for result in seed_payloads
                ),
                "adaptation_gain_held_out_accuracy": _mean(
                    result.get("adaptation_gain_held_out_accuracy") for result in seed_payloads
                ),
                "relative_recovery_balanced_accuracy": _mean(
                    result.get("relative_recovery_balanced_accuracy") for result in seed_payloads
                ),
                "n_evaluated_organs": len(seed_payloads),
            }
        )

    return {
        "status": "ok" if evaluated_organs else "skipped",
        "surface_name": "surface_b_few_shot_transfer_evidence",
        "surface_scope": surface_scope,
        "case_split_policy": "global_case_disjoint_partitions",
        "held_out_definition": "held_out_during_adaptation",
        "transfer_protocol": transfer_protocol,
        "required_modalities": list(required_modalities),
        "support_per_modality": int(support_per_modality),
        "query_per_modality": query_per_modality,
        "evaluation_seeds": [int(seed) for seed in seeds],
        "n_supported_organs": len(list(supported_organs)),
        "n_evaluated_organs": len(evaluated_organs),
        "macro_mean": {
            "balanced_accuracy": _mean(macro_balanced_accuracies),
            "fewshot_probe_score": _mean(macro_fewshot_scores),
            "held_out_accuracy": _mean(macro_fewshot_scores),
            "in_distribution_accuracy": _mean(macro_seen_scores),
            "generalization_gap": _mean(macro_gaps),
            "transfer_efficiency": _mean(macro_efficiencies),
            "nearest_centroid_balanced_accuracy": _mean(macro_centroid_balanced_accuracies),
            "nearest_centroid_held_out_accuracy": _mean(macro_centroid_held_out_scores),
            "adaptation_gain_balanced_accuracy": _mean(macro_adaptation_gains),
            "adaptation_gain_held_out_accuracy": _mean(macro_held_out_adaptation_gains),
            "relative_recovery_balanced_accuracy": _mean(macro_relative_recoveries),
        },
        "macro_mean_std": {
            "balanced_accuracy": _std(result.get("balanced_accuracy") for result in macro_seed_results),
            "fewshot_probe_score": _std(result.get("fewshot_probe_score") for result in macro_seed_results),
            "held_out_accuracy": _std(result.get("held_out_accuracy") for result in macro_seed_results),
            "nearest_centroid_balanced_accuracy": _std(
                result.get("nearest_centroid_balanced_accuracy") for result in macro_seed_results
            ),
            "nearest_centroid_held_out_accuracy": _std(
                result.get("nearest_centroid_held_out_accuracy") for result in macro_seed_results
            ),
            "adaptation_gain_balanced_accuracy": _std(
                result.get("adaptation_gain_balanced_accuracy") for result in macro_seed_results
            ),
            "adaptation_gain_held_out_accuracy": _std(
                result.get("adaptation_gain_held_out_accuracy") for result in macro_seed_results
            ),
            "relative_recovery_balanced_accuracy": _std(
                result.get("relative_recovery_balanced_accuracy") for result in macro_seed_results
            ),
        },
        "macro_seed_results": macro_seed_results,
        "per_organ": organ_payloads,
    }