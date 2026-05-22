#!/usr/bin/env python

"""Aggregate Phase 2 checkpoint results into comparison tables.

This script reads the per-checkpoint Phase 2 primary metric files produced by
the evaluation pipeline and writes a compact comparison table in both CSV and
JSON formats under the canonical results root.

Usage examples:
    python aggregate_phase2_checkpoint_results.py \
        --analysis-name totalsegmenter_ct_mr_anchor \
        --manifest-variant core

    python aggregate_phase2_checkpoint_results.py \
        --results-dir ./outputs_phase2/totalsegmenter_ct_mr_anchor/phase2/core/results
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from phase2_config import (
    get_checkpoint_comparison_csv_path,
    get_checkpoint_comparison_json_path,
    get_output_paths,
    normalize_feature_type,
)


def _get_metric(metrics: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _archive_root_primary_metrics(results_dir: Path) -> str | None:
    root_metrics_path = results_dir / "phase2_primary_metrics.json"
    if not root_metrics_path.exists():
        return None

    archived_path = results_dir / "phase2_primary_metrics.legacy_root.json"
    archived_path.write_text(root_metrics_path.read_text(encoding="utf-8"), encoding="utf-8")
    root_metrics_path.unlink()
    return str(archived_path)


def _checkpoint_family(checkpoint_name: str) -> str:
    if checkpoint_name == "3dinov2":
        return "3dinov2"
    if checkpoint_name.endswith("_sa"):
        return "spacing_aware"
    return "relative"


def _checkpoint_crop_size(checkpoint_name: str) -> int | None:
    if "c96" in checkpoint_name:
        return 96
    if "c112" in checkpoint_name or checkpoint_name == "3dinov2":
        return 112
    return None


def _collect_rows(results_dir: Path, feature_type: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if feature_type == "cls":
        metrics_paths = sorted(results_dir.glob("*/phase2_primary_metrics.json"))
    else:
        metrics_paths = sorted(results_dir.glob(f"*/{feature_type}/phase2_primary_metrics.json"))

    for metrics_path in metrics_paths:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        checkpoint_name = payload.get("checkpoint_name") or metrics_path.parent.name
        margin = payload.get("anatomy_over_modality_margin", {})
        retrieval = payload.get("balanced_cross_modal_retrieval", {})
        lisi = payload.get("balanced_lisi", {})
        margin_overall = margin.get("overall") or {}
        ct_to_mr = retrieval.get("ct_to_mr") or {}
        mr_to_ct = retrieval.get("mr_to_ct") or {}
        bidirectional_mean = retrieval.get("bidirectional_mean") or {}
        ct_to_mr_overall = ct_to_mr.get("overall") or {}
        mr_to_ct_overall = mr_to_ct.get("overall") or {}
        overall_lisi = lisi.get("overall", {})
        modality_ilisi = overall_lisi.get("modality_ilisi", {})
        organ_clisi = overall_lisi.get("organ_clisi", {})
        bootstrap_ci = margin_overall.get("bootstrap_95ci") or {}

        row = {
            "checkpoint_name": checkpoint_name,
            "checkpoint_family": _checkpoint_family(checkpoint_name),
            "crop_size": _checkpoint_crop_size(checkpoint_name),
            "margin_status": margin.get("status"),
            "margin_reason": margin.get("reason"),
            "mean_margin": margin_overall.get("mean_margin"),
            "median_margin": margin_overall.get("median_margin"),
            "fraction_positive_margin": margin_overall.get("fraction_positive_margin"),
            "margin_ci_lower": bootstrap_ci.get("lower"),
            "margin_ci_upper": bootstrap_ci.get("upper"),
            "retrieval_status": retrieval.get("status"),
            "retrieval_top@1": _get_metric(bidirectional_mean, "top@1", "top_at_1", "hit_at_1"),
            "retrieval_top@5": _get_metric(bidirectional_mean, "top@5", "top_at_5", "hit_at_5"),
            "retrieval_map": bidirectional_mean.get("map"),
            "ct_to_mr_status": ct_to_mr.get("status"),
            "ct_to_mr_reason": ct_to_mr.get("reason"),
            "ct_to_mr_top@1": _get_metric(ct_to_mr_overall, "top@1", "top_at_1", "hit_at_1"),
            "ct_to_mr_top@5": _get_metric(ct_to_mr_overall, "top@5", "top_at_5", "hit_at_5"),
            "ct_to_mr_map": ct_to_mr_overall.get("map"),
            "mr_to_ct_status": mr_to_ct.get("status"),
            "mr_to_ct_reason": mr_to_ct.get("reason"),
            "mr_to_ct_top@1": _get_metric(mr_to_ct_overall, "top@1", "top_at_1", "hit_at_1"),
            "mr_to_ct_top@5": _get_metric(mr_to_ct_overall, "top@5", "top_at_5", "hit_at_5"),
            "mr_to_ct_map": mr_to_ct_overall.get("map"),
            "lisi_status": lisi.get("status"),
            "lisi_reason": lisi.get("reason"),
            "modality_ilisi": modality_ilisi.get("mean"),
            "organ_clisi": organ_clisi.get("mean"),
            "margin_queries": margin.get("n_queries"),
            "retrieval_ct_queries": ct_to_mr.get("n_queries"),
            "retrieval_mr_queries": mr_to_ct.get("n_queries"),
            "lisi_samples": lisi.get("n_samples"),
            "metrics_path": str(metrics_path),
        }
        rows.append(row)
    return rows


def _sort_metric_value(item: Dict[str, Any], key: str, reverse: bool) -> float:
    value = item.get(key)
    if value is None:
        return float("-inf") if reverse else float("inf")
    return float(value)


def _best_row(rows: List[Dict[str, Any]], key: str, reverse: bool = True) -> Dict[str, Any] | None:
    valid_rows = [row for row in rows if row.get(key) is not None]
    if not valid_rows:
        return None
    return max(valid_rows, key=lambda item: item[key]) if reverse else min(valid_rows, key=lambda item: item[key])


def _assign_rank(rows: List[Dict[str, Any]], metric_key: str, rank_key: str, reverse: bool) -> None:
    valid_rows = [row for row in rows if row.get(metric_key) is not None]
    for row in rows:
        row[rank_key] = None
    ordered_rows = sorted(valid_rows, key=lambda item: _sort_metric_value(item, metric_key, reverse), reverse=reverse)
    for index, row in enumerate(ordered_rows, start=1):
        row[rank_key] = index


def _rank_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = [dict(row) for row in rows]

    _assign_rank(ranked, "mean_margin", "rank_mean_margin", reverse=True)
    _assign_rank(ranked, "fraction_positive_margin", "rank_fraction_positive_margin", reverse=True)
    _assign_rank(ranked, "retrieval_top@1", "rank_retrieval_top@1", reverse=True)
    _assign_rank(ranked, "retrieval_top@5", "rank_retrieval_top@5", reverse=True)
    _assign_rank(ranked, "retrieval_map", "rank_retrieval_map", reverse=True)
    _assign_rank(ranked, "modality_ilisi", "rank_modality_ilisi", reverse=True)
    _assign_rank(ranked, "organ_clisi", "rank_organ_clisi", reverse=False)

    return sorted(
        ranked,
        key=lambda item: (
            item["rank_retrieval_map"] is None,
            item["rank_retrieval_map"] if item["rank_retrieval_map"] is not None else float("inf"),
            item["checkpoint_name"],
        ),
    )


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate finished Phase 2 checkpoint metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python aggregate_phase2_checkpoint_results.py --analysis-name totalsegmenter_ct_mr_anchor --manifest-variant core\n"
            "  python aggregate_phase2_checkpoint_results.py --results-dir ./outputs_phase2/totalsegmenter_ct_mr_anchor/phase2/core/results\n"
        ),
    )
    parser.add_argument("--analysis-name", default="totalsegmenter_ct_mr_anchor", help="Phase 2 analysis namespace")
    parser.add_argument("--manifest-variant", default="core", help="Phase 2 manifest variant")
    parser.add_argument("--results-dir", default=None, help="Optional explicit results directory override")
    parser.add_argument("--feature-type", default="cls", help="Feature family to aggregate")
    args = parser.parse_args()
    feature_type = normalize_feature_type(args.feature_type)

    if args.results_dir:
        results_dir = Path(args.results_dir).resolve()
    else:
        results_dir = get_output_paths(args.analysis_name, args.manifest_variant)["results"]

    archived_root_metrics = _archive_root_primary_metrics(results_dir) if feature_type == "cls" else None

    rows = _collect_rows(results_dir, feature_type)
    if not rows:
        raise FileNotFoundError(
            f"No per-checkpoint Phase 2 metric files found under {results_dir} for feature type {feature_type}"
        )

    ranked_rows = _rank_rows(rows)
    csv_path = get_checkpoint_comparison_csv_path(results_dir, feature_type)
    json_path = get_checkpoint_comparison_json_path(results_dir, feature_type)

    _write_csv(csv_path, ranked_rows)
    _write_json(
        json_path,
        {
            "results_dir": str(results_dir),
            "n_checkpoints": len(ranked_rows),
            "archived_stale_root_primary_metrics": archived_root_metrics,
            "rows": ranked_rows,
            "best_by_mean_margin": _best_row(ranked_rows, "mean_margin"),
            "best_by_retrieval_top@1": _best_row(ranked_rows, "retrieval_top@1"),
            "best_by_retrieval_top@5": _best_row(ranked_rows, "retrieval_top@5"),
            "best_by_retrieval_map": _best_row(ranked_rows, "retrieval_map"),
            "best_by_modality_ilisi": _best_row(ranked_rows, "modality_ilisi"),
            "best_by_organ_clisi": _best_row(ranked_rows, "organ_clisi", reverse=False),
        },
    )

    if archived_root_metrics:
        print(f"Archived stale root metrics to {archived_root_metrics}")
    print(f"Wrote CSV summary to {csv_path}")
    print(f"Wrote JSON summary to {json_path}")


if __name__ == "__main__":
    main()