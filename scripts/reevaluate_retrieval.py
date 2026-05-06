#!/usr/bin/env python3
"""
Re-evaluate experiment traces for retrieval quality.

Uses schema-aware metrics with the SchemaGraph for subclass-aware
matching (path coherence, triple-level comparison) and ACCEPTABLE handling.

Usage:
    python scripts/reevaluate_retrieval.py EXPERIMENT_NAME
    python scripts/reevaluate_retrieval.py EXPERIMENT_NAME --field final_schema_triples
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from data.queries.experimental_corpus import get_experimental_query
from src.evaluation.retrieval_ground_truth import build_schema_gt_for_query
from src.evaluation.retrieval_metrics import calculate_retrieval_metrics
from src.config import RESULTS_DIR
from src.validation.schema_graph import get_combined_schema


def calc_retrieval_summary(metrics_list: list[dict]) -> dict:
    """Calculate mean retrieval metrics across a list of results."""
    if not metrics_list:
        return {"count": 0}

    n = len(metrics_list)
    keys = [
        "schema_path_coherence",
        "schema_triple_precision", "schema_triple_recall", "schema_triple_f1",
    ]
    summary = {"count": n}
    for key in keys:
        values = [m.get(key, 0) for m in metrics_list]
        summary[f"{key}_mean"] = sum(values) / n
        if n > 1:
            mean = summary[f"{key}_mean"]
            summary[f"{key}_std"] = (sum((v - mean) ** 2 for v in values) / n) ** 0.5

    return summary


def reevaluate_retrieval(
    experiment_name: str,
    triples_field: str = "all_retrieved_triples",
) -> dict:
    """Re-evaluate all traces for retrieval quality."""
    traces_dir = RESULTS_DIR / "experiments" / experiment_name / "traces"

    if not traces_dir.exists():
        print(f"Error: Traces directory not found: {traces_dir}")
        sys.exit(1)

    trace_files = sorted(traces_dir.glob("*.json"))
    print(f"Found {len(trace_files)} trace files in {traces_dir}")
    print(f"Using triples field: {triples_field}")

    results: dict[str, dict] = {}
    skipped = 0
    all_metrics: list[dict] = []
    metrics_by_approach: dict[str, list[dict]] = defaultdict(list)
    metrics_by_query_set: dict[str, list[dict]] = defaultdict(list)
    metrics_by_dataset: dict[str, list[dict]] = defaultdict(list)

    for trace_file in tqdm(trace_files, desc="Evaluating retrieval"):
        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                trace = json.load(f)
        except Exception as e:
            print(f"  Warning: Failed to load {trace_file.name}: {e}")
            skipped += 1
            continue

        # Get query metadata
        query_meta = trace.get("query_metadata", {})
        query_id = query_meta.get("query_id", "")
        query_set = query_meta.get("query_set", "")
        dataset = query_meta.get("dataset", "")
        approach = trace.get("approach", "")

        if not query_id:
            skipped += 1
            continue

        # Look up ground truth
        gt_query = get_experimental_query(query_id)
        if gt_query is None:
            skipped += 1
            continue

        # Build schema-aware GT (triple level)
        schema_gt = build_schema_gt_for_query(gt_query)
        if not schema_gt.triples:
            skipped += 1
            continue

        dataset_ids = gt_query.datasets or [gt_query.dataset]
        schema = get_combined_schema(dataset_ids)

        # Get retrieved triples
        turtle_strings = trace.get(triples_field, [])

        # Calculate schema-aware metrics
        metrics = calculate_retrieval_metrics(turtle_strings, schema, schema_gt)
        metrics_dict = metrics.to_dict()

        trace_name = trace_file.stem
        results[trace_name] = {
            "query_id": query_id,
            "query_set": query_set,
            "dataset": dataset,
            "approach": approach,
            "retrieval_metrics": metrics_dict,
            "schema_gt": schema_gt.to_dict(),
        }

        # Aggregate for summary
        all_metrics.append(metrics_dict)
        metrics_by_approach[approach].append(metrics_dict)
        metrics_by_query_set[query_set].append(metrics_dict)
        # Strip _LARGE suffix for dataset aggregation
        base_dataset = dataset.replace("_LARGE", "")
        metrics_by_dataset[base_dataset].append(metrics_dict)

    # Build summary
    summary = {
        "overall": calc_retrieval_summary(all_metrics),
        "by_approach": {
            k: calc_retrieval_summary(v) for k, v in sorted(metrics_by_approach.items())
        },
        "by_query_set": {
            k: calc_retrieval_summary(v) for k, v in sorted(metrics_by_query_set.items())
        },
        "by_dataset": {
            k: calc_retrieval_summary(v) for k, v in sorted(metrics_by_dataset.items())
        },
    }

    output = {
        "experiment_name": experiment_name,
        "evaluation_timestamp": datetime.now().isoformat(),
        "triples_field": triples_field,
        "total_traces": len(trace_files),
        "evaluated_traces": len(all_metrics),
        "skipped_traces": skipped,
        "traces": results,
        "summary": summary,
    }

    # Save results
    output_file = RESULTS_DIR / "experiments" / experiment_name / "retrieval_evaluation.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    print(f"Evaluated: {len(all_metrics)} traces, Skipped: {skipped}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"RETRIEVAL EVALUATION SUMMARY: {experiment_name}")
    print(f"{'='*80}")

    overall = summary["overall"]
    print(f"\nOverall ({overall['count']} traces):")
    print(f"  PathCoher - Score: {overall.get('schema_path_coherence_mean', 0):.3f}")
    print(f"  TripleMet - P: {overall.get('schema_triple_precision_mean', 0):.3f}  "
          f"R: {overall.get('schema_triple_recall_mean', 0):.3f}  "
          f"F1: {overall.get('schema_triple_f1_mean', 0):.3f}")

    print(f"\nBy Approach:")
    for approach, stats in sorted(summary["by_approach"].items()):
        print(f"  {approach:25s} - "
              f"PathCoh: {stats.get('schema_path_coherence_mean', 0):.3f}  "
              f"TripF1: {stats.get('schema_triple_f1_mean', 0):.3f}  "
              f"TripP: {stats.get('schema_triple_precision_mean', 0):.3f}  "
              f"TripR: {stats.get('schema_triple_recall_mean', 0):.3f}  "
              f"(n={stats['count']})")

    print(f"\nBy Query Set:")
    for qset, stats in sorted(summary["by_query_set"].items()):
        print(f"  {qset:10s} - "
              f"PathCoh: {stats.get('schema_path_coherence_mean', 0):.3f}  "
              f"TripF1: {stats.get('schema_triple_f1_mean', 0):.3f}  "
              f"(n={stats['count']})")

    print(f"\nBy Dataset:")
    for ds, stats in sorted(summary["by_dataset"].items()):
        print(f"  {ds:10s} - "
              f"PathCoh: {stats.get('schema_path_coherence_mean', 0):.3f}  "
              f"TripF1: {stats.get('schema_triple_f1_mean', 0):.3f}  "
              f"(n={stats['count']})")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate experiment traces for retrieval quality."
    )
    parser.add_argument("experiment", help="Experiment name (e.g., deepseek_test_8)")
    parser.add_argument(
        "--field",
        default="all_retrieved_triples",
        choices=["all_retrieved_triples", "final_schema_triples"],
        help="Which trace field to evaluate (default: all_retrieved_triples)",
    )
    args = parser.parse_args()
    reevaluate_retrieval(args.experiment, args.field)


if __name__ == "__main__":
    main()
