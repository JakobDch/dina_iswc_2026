#!/usr/bin/env python3
"""
Re-evaluate experiment with problem categorization and ambiguity analysis.

This script extends adaptive_evaluation.json with:
- Problem detection per trace
- Ambiguity type parsing from query notes
- Grouped statistics by ambiguity type, query set, dataset
- Correlation analysis between ambiguity types and problem categories
- Actionable insights

Usage:
    python scripts/reevaluate_with_problems.py EXPERIMENT_NAME

Output:
    Extends existing adaptive_evaluation.json with problem_analysis per trace
    and adds problem_summary section.
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path BEFORE any other imports
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from tqdm import tqdm


def _import_module_directly(module_name: str, file_path: Path):
    """Import a module directly from file, bypassing package __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Register so relative imports work
    spec.loader.exec_module(module)
    return module


# First, import modules that have no problematic dependencies
from data.queries.use_case_queries_tiered_tuples import (
    AdaptiveGroundTruth,
    get_adaptive_query,
)
from src.config import RESULTS_DIR

# Import problem_categorization directly (no dependencies on tiered_metrics)
_cat_module = _import_module_directly(
    "problem_categorization",
    _project_root / "src" / "evaluation" / "problem_categorization.py",
)
AmbiguityType = _cat_module.AmbiguityType
EvaluationReport = _cat_module.EvaluationReport
ProblemAnalysisResult = _cat_module.ProblemAnalysisResult
parse_ambiguity_type = _cat_module.parse_ambiguity_type
DetectedProblem = _cat_module.DetectedProblem
ProblemCategory = _cat_module.ProblemCategory
Severity = _cat_module.Severity
CategoryStats = _cat_module.CategoryStats
ProblemStats = _cat_module.ProblemStats
CorrelationEntry = _cat_module.CorrelationEntry

# Import tiered_metrics_tuples directly (bypasses __init__.py that imports tiered_metrics)
_metrics_module = _import_module_directly(
    "tiered_metrics_tuples",
    _project_root / "src" / "evaluation" / "tiered_metrics_tuples.py",
)
AdaptiveMetricsResult = _metrics_module.AdaptiveMetricsResult

# Also register the categorization module so problem_detector can import it
sys.modules["src.evaluation.problem_categorization"] = _cat_module

# Now import problem_detector (it uses the registered categorization module)
_detector_module = _import_module_directly(
    "problem_detector",
    _project_root / "src" / "evaluation" / "problem_detector.py",
)
detect_problems = _detector_module.detect_problems
generate_report = _detector_module.generate_report


def load_adaptive_evaluation(experiment_name: str) -> dict | None:
    """Load existing adaptive_evaluation.json."""
    eval_file = RESULTS_DIR / "experiments" / experiment_name / "adaptive_evaluation.json"

    if not eval_file.exists():
        print(f"Error: adaptive_evaluation.json not found at {eval_file}")
        print("Run 'python scripts/reevaluate_tiered.py EXPERIMENT_NAME --adaptive' first.")
        return None

    with open(eval_file, "r", encoding="utf-8") as f:
        return json.load(f)


def metrics_from_dict(d: dict) -> AdaptiveMetricsResult:
    """Reconstruct AdaptiveMetricsResult from dict."""
    # Extract the adaptive_metrics sub-dict if present
    metrics_dict = d.get("adaptive_metrics", d)

    result = AdaptiveMetricsResult()
    result.best_recall = metrics_dict.get("best_recall", 0.0)
    result.best_precision = metrics_dict.get("best_precision", 0.0)
    result.best_f1 = metrics_dict.get("best_f1", 0.0)
    result.essential_active = metrics_dict.get("essential_active", False)
    result.preferred_active = metrics_dict.get("preferred_active", False)
    result.acceptable_active = metrics_dict.get("acceptable_active", False)
    result.essential_matches = metrics_dict.get("essential_matches", 0)
    result.preferred_matches = metrics_dict.get("preferred_matches", 0)
    result.acceptable_matches = metrics_dict.get("acceptable_matches", 0)
    result.llm_row_count = metrics_dict.get("llm_row_count", 0)
    result.llm_columns = metrics_dict.get("llm_columns", [])
    result.schema_recall = metrics_dict.get("schema_recall", 0.0)
    result.schema_precision = metrics_dict.get("schema_precision", 0.0)
    result.schema_f1 = metrics_dict.get("schema_f1", 0.0)
    result.schema_expected_count = metrics_dict.get("schema_expected_count", 0)
    result.schema_matched_count = metrics_dict.get("schema_matched_count", 0)
    result.schema_llm_columns = metrics_dict.get("schema_llm_columns", 0)
    result.schema_llm_matched = metrics_dict.get("schema_llm_matched", 0)
    result.best_recall_variant = metrics_dict.get("best_recall_variant", "")
    result.best_precision_variant = metrics_dict.get("best_precision_variant", "")
    result.variants_evaluated = metrics_dict.get("variants_evaluated", 0)
    result.best_recall_gt_size = metrics_dict.get("best_recall_gt_size", 0)
    result.best_precision_gt_size = metrics_dict.get("best_precision_gt_size", 0)

    return result


def analyze_experiment(experiment_name: str) -> dict | None:
    """Analyze experiment with problem detection."""
    print(f"\nAnalyzing experiment: {experiment_name}")

    # Load existing adaptive evaluation
    evaluation = load_adaptive_evaluation(experiment_name)
    if not evaluation:
        return None

    traces = evaluation.get("traces", {})
    print(f"Found {len(traces)} traces to analyze")

    # Analyze each trace
    analyses: list[ProblemAnalysisResult] = []
    metrics_by_trace: dict[str, AdaptiveMetricsResult] = {}

    for trace_key, trace_data in tqdm(traces.items(), desc="Analyzing problems"):
        # Get query ID from trace
        query_id = trace_data.get("query_id")
        if not query_id:
            # Try to extract from trace_key (format: QUERYID_approach_hash)
            parts = trace_key.split("_")
            if parts:
                query_id = parts[0]

        if not query_id:
            continue

        # Get ground truth
        ground_truth = get_adaptive_query(query_id)
        if not ground_truth:
            continue

        # Parse ambiguity type from notes
        ambiguity_type = parse_ambiguity_type(
            ground_truth.notes or "",
            ground_truth.query_set,
        )

        # Reconstruct metrics from saved data
        metrics = metrics_from_dict(trace_data)
        metrics_by_trace[trace_key] = metrics

        # Detect problems
        analysis = detect_problems(
            adaptive_metrics=metrics,
            ground_truth=ground_truth,
            ambiguity_type=ambiguity_type,
            trace_key=trace_key,
        )
        analyses.append(analysis)

        # Add problem analysis to trace data
        trace_data["problem_analysis"] = analysis.to_dict()

    # Generate report
    report = generate_report(
        analyses=analyses,
        metrics_by_trace=metrics_by_trace,
        experiment_name=experiment_name,
        timestamp=datetime.now().isoformat(),
    )

    # Add problem summary to evaluation
    evaluation["problem_summary"] = report.to_dict()
    evaluation["problem_analysis_timestamp"] = datetime.now().isoformat()

    return evaluation


def print_report(report_dict: dict):
    """Print a summary of the problem analysis."""
    print("\n" + "=" * 70)
    print("PROBLEM ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nTotal traces: {report_dict.get('total_traces', 0)}")
    print(f"Overall success rate: {report_dict.get('overall_success_rate', 0):.1%}")
    print(f"Overall avg F1: {report_dict.get('overall_avg_f1', 0):.3f}")

    # By Ambiguity Type
    print("\n" + "-" * 70)
    print("BY AMBIGUITY TYPE:")
    print("-" * 70)
    print(f"  {'Type':<25} {'Count':>6} {'Avg F1':>10} {'Success':>10}")
    print(f"  {'-'*55}")

    by_ambiguity = report_dict.get("by_ambiguity_type", {})
    sorted_ambiguity = sorted(by_ambiguity.items(), key=lambda x: x[1].get("avg_f1", 0))

    for amb_type, stats in sorted_ambiguity:
        count = stats.get("total", 0)
        avg_f1 = stats.get("avg_f1", 0)
        success = stats.get("success_rate", 0)
        marker = " <-- HARDEST" if avg_f1 < 0.3 and count > 1 else ""
        print(f"  {amb_type:<25} {count:>6} {avg_f1:>10.1%} {success:>10.0%}{marker}")

    # By Query Set
    print("\n" + "-" * 70)
    print("BY QUERY SET:")
    print("-" * 70)
    print(f"  {'Set':<10} {'Count':>6} {'Avg F1':>10} {'Success':>10}")
    print(f"  {'-'*40}")

    by_query_set = report_dict.get("by_query_set", {})
    for qs in ["A", "B", "C", "D", "E"]:
        if qs in by_query_set:
            stats = by_query_set[qs]
            count = stats.get("total", 0)
            avg_f1 = stats.get("avg_f1", 0)
            success = stats.get("success_rate", 0)
            print(f"  {qs:<10} {count:>6} {avg_f1:>10.1%} {success:>10.0%}")

    # Top Problems
    print("\n" + "-" * 70)
    print("TOP PROBLEMS:")
    print("-" * 70)

    most_common = report_dict.get("most_common_problems", [])[:10]
    for i, item in enumerate(most_common, 1):
        problem = item.get("problem", "unknown")
        count = item.get("count", 0)
        print(f"  {i:>2}. {problem:<40} ({count} occurrences)")

    # Insights
    insights = report_dict.get("insights", [])
    if insights:
        print("\n" + "-" * 70)
        print("INSIGHTS:")
        print("-" * 70)
        for insight in insights:
            print(f"  - {insight}")

    # Hardest ambiguity types
    hardest = report_dict.get("hardest_ambiguity_types", [])[:5]
    if hardest:
        print("\n" + "-" * 70)
        print("HARDEST AMBIGUITY TYPES (lowest F1):")
        print("-" * 70)
        for item in hardest:
            amb_type = item.get("type", "unknown")
            avg_f1 = item.get("avg_f1", 0)
            print(f"  - {amb_type}: avg F1 = {avg_f1:.1%}")


def save_results(evaluation: dict, experiment_name: str):
    """Save extended evaluation back to file."""
    output_file = RESULTS_DIR / "experiments" / experiment_name / "adaptive_evaluation.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)

    print(f"\nSaved extended evaluation to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate experiment with problem categorization"
    )
    parser.add_argument(
        "experiment_name",
        help="Name of the experiment directory in results/experiments/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and print report without saving",
    )

    args = parser.parse_args()

    # Run analysis
    evaluation = analyze_experiment(args.experiment_name)
    if not evaluation:
        sys.exit(1)

    # Print report
    problem_summary = evaluation.get("problem_summary", {})
    print_report(problem_summary)

    # Save unless dry-run
    if not args.dry_run:
        save_results(evaluation, args.experiment_name)
    else:
        print("\n[DRY RUN] Results not saved.")


if __name__ == "__main__":
    main()
