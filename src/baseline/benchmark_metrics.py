"""
Metric calculation functions for benchmark evaluation.

This module contains all metric calculation logic extracted from the original tasks.py,
including syntax metrics, model selection metrics, and statistical aggregations.
"""

import logging
import statistics
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


def calc_stats(values: list[float]) -> dict[str, Any]:
    """
    Calculate basic statistics for a list of values.

    Args:
        values: List of numeric values to analyze.

    Returns:
        Dictionary with avg, variance, and raw values.
    """
    if not values:
        return {"avg": None, "variance": None, "values": []}

    avg = sum(values) / len(values)
    variance = statistics.variance(values) if len(values) > 1 else 0.0

    return {
        "avg": round(avg, 4),
        "variance": round(variance, 6),
        "values": [round(v, 4) for v in values]
    }


def calculate_syntax_metrics(successful_runs: list[dict]) -> dict[str, Any]:
    """
    Calculate aggregated syntax metrics from a list of successful benchmark runs.

    This function analyzes syntax errors found during benchmark runs, including
    runs with uncorrected syntax errors (error_type="syntax_error").

    Args:
        successful_runs: List of run objects (including runs with syntax errors).

    Returns:
        Dictionary with aggregated syntax metrics:
        - syntax_corrections_applied: Number of runs where syntax was corrected
        - syntax_correction_rate: Percentage of runs with syntax corrections
        - total_syntax_errors_found: Total count of syntax errors found
        - uncorrected_syntax_errors: Number of runs with unfixable syntax errors
        - avg_syntax_iterations: Average correction iterations per run
        - most_common_syntax_errors: Top 5 most frequent error types
        - syntax_error_distribution: Top 10 errors with counts
    """
    if not successful_runs:
        return {
            "syntax_corrections_applied": 0,
            "syntax_correction_rate": 0,
            "total_syntax_errors_found": 0,
            "avg_syntax_iterations": 0,
            "most_common_syntax_errors": [],
            "syntax_error_distribution": {},
            "uncorrected_syntax_errors": 0
        }

    total_runs = len(successful_runs)
    runs_with_syntax_correction = 0
    runs_with_uncorrected_syntax_errors = 0
    total_syntax_errors = 0
    all_syntax_errors: list[str] = []
    syntax_iterations: list[int] = []

    for run in successful_runs:
        # Check for uncorrected syntax errors
        if run.get("error_type") == "syntax_error" and run.get("initial_syntax_error"):
            runs_with_uncorrected_syntax_errors += 1
            initial_error = run.get("initial_syntax_error", "")
            if initial_error:
                all_syntax_errors.append(initial_error)
                total_syntax_errors += 1

        # Check for successful syntax corrections
        elif run.get("syntax_correction_applied"):
            runs_with_syntax_correction += 1
            syntax_errors_found = run.get("syntax_errors_found", [])
            if syntax_errors_found:
                all_syntax_errors.extend(syntax_errors_found)
                total_syntax_errors += len(syntax_errors_found)

            iterations_used = run.get("syntax_iterations", 0)
            if iterations_used > 0:
                syntax_iterations.append(iterations_used)

    # Calculate most common errors
    error_counts = Counter(all_syntax_errors)
    most_common_errors = [error for error, _ in error_counts.most_common(5)]
    error_distribution = dict(error_counts.most_common(10))

    # Calculate averages
    avg_syntax_iterations = sum(syntax_iterations) / len(syntax_iterations) if syntax_iterations else 0
    syntax_correction_rate = (runs_with_syntax_correction / total_runs * 100) if total_runs > 0 else 0

    return {
        "syntax_corrections_applied": runs_with_syntax_correction,
        "syntax_correction_rate": round(syntax_correction_rate, 1),
        "total_syntax_errors_found": total_syntax_errors,
        "uncorrected_syntax_errors": runs_with_uncorrected_syntax_errors,
        "avg_syntax_iterations": round(avg_syntax_iterations, 1),
        "most_common_syntax_errors": most_common_errors,
        "syntax_error_distribution": error_distribution
    }


def calculate_best_model_selection_metrics(
    groundtruth_semantic_models: list,
    final_models_used: list[str],
    request_id: str = "metrics"
) -> dict[str, Any]:
    """
    Calculate model selection metrics for all possible solutions and select the best one.

    The groundtruth_semantic_models can be in two formats:
    1. List of lists: [[solution1], [solution2], ...] - Multiple possible solutions
    2. Flat list: [model1, model2, ...] - Single solution (backwards compatible)

    For each possible solution, Precision, Recall, and F1-Score are calculated.
    The solution with the highest F1-Score is selected as the best solution.

    Args:
        groundtruth_semantic_models: Either list of lists or flat list of model names.
        final_models_used: List of models selected by the system.
        request_id: ID for logging purposes.

    Returns:
        Dictionary with best metrics and additional information:
        - precision, recall, f1_score: Best solution metrics
        - true_positives, false_positives, false_negatives: Counts
        - groundtruth_models_count, selected_models_count: Model counts
        - best_solution_index: Index of best solution (if multiple)
        - total_solutions_evaluated: Number of solutions evaluated (if multiple)
        - all_solutions_metrics: Metrics for all solutions (if multiple)
    """

    def normalize_model_name(model_name: str) -> str:
        """Normalize model names for comparison."""
        name = model_name.replace("_semantic_model_transformed.ttl", "")
        name = name.replace("_semantic_model_transformed", "")
        name = name.replace(".ttl", "")
        return name.lower()

    # Normalize selected models once
    selected_models_normalized = {normalize_model_name(m) for m in final_models_used}

    # Check if groundtruth is list of lists or flat list
    is_list_of_lists = (
        isinstance(groundtruth_semantic_models, list) and
        len(groundtruth_semantic_models) > 0 and
        isinstance(groundtruth_semantic_models[0], list)
    )

    if not is_list_of_lists:
        # Convert flat list to list of lists for uniform processing
        groundtruth_semantic_models = [groundtruth_semantic_models]
        logger.info(f"[{request_id}] Groundtruth in flat format, converting to list of lists")

    # Calculate metrics for all possible solutions
    all_solutions_metrics: list[dict] = []

    for solution_index, gt_solution in enumerate(groundtruth_semantic_models):
        gt_models_normalized = {normalize_model_name(m) for m in gt_solution}

        # Calculate True Positives, False Positives, False Negatives
        true_positives = len(gt_models_normalized.intersection(selected_models_normalized))
        false_positives = len(selected_models_normalized - gt_models_normalized)
        false_negatives = len(gt_models_normalized - selected_models_normalized)

        # Calculate Precision, Recall, and F1-Score
        precision = true_positives / len(selected_models_normalized) if selected_models_normalized else 0.0
        recall = true_positives / len(gt_models_normalized) if gt_models_normalized else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        solution_metrics = {
            "solution_index": solution_index,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "groundtruth_models_count": len(gt_models_normalized),
            "selected_models_count": len(selected_models_normalized),
            "groundtruth_models": sorted(list(gt_models_normalized))
        }

        all_solutions_metrics.append(solution_metrics)

        logger.info(
            f"[{request_id}] Solution {solution_index}: "
            f"P={precision:.2%}, R={recall:.2%}, F1={f1_score:.2%} "
            f"(GT: {sorted(list(gt_models_normalized))}, Selected: {sorted(list(selected_models_normalized))})"
        )

    # Select best solution based on F1-Score (tie-breaker: Precision, then Recall)
    best_solution = max(
        all_solutions_metrics,
        key=lambda x: (x["f1_score"], x["precision"], x["recall"])
    )

    logger.info(
        f"[{request_id}] Best solution: Index {best_solution['solution_index']} "
        f"with F1={best_solution['f1_score']:.2%} "
        f"(P={best_solution['precision']:.2%}, R={best_solution['recall']:.2%})"
    )

    # Build result dictionary
    result = {
        "precision": best_solution["precision"],
        "recall": best_solution["recall"],
        "f1_score": best_solution["f1_score"],
        "true_positives": best_solution["true_positives"],
        "false_positives": best_solution["false_positives"],
        "false_negatives": best_solution["false_negatives"],
        "groundtruth_models_count": best_solution["groundtruth_models_count"],
        "selected_models_count": best_solution["selected_models_count"]
    }

    # Add additional information if multiple solutions were evaluated
    if len(all_solutions_metrics) > 1:
        result["best_solution_index"] = best_solution["solution_index"]
        result["total_solutions_evaluated"] = len(all_solutions_metrics)
        result["all_solutions_metrics"] = all_solutions_metrics
        result["best_groundtruth_models"] = best_solution["groundtruth_models"]

    return result


def calculate_template_summary(
    template_scores: list[float],
    run_results: list[dict],
    num_runs: int
) -> dict[str, Any]:
    """
    Calculate summary statistics for a single template's benchmark results.

    Args:
        template_scores: List of scores from successful runs.
        run_results: List of all run result dictionaries.
        num_runs: Total number of runs attempted.

    Returns:
        Dictionary with template summary including:
        - average_score, successful_runs, failed_runs, scores
        - result_metrics (execution_accuracy, precision, recall, f1_score)
        - variable_level_metrics (if available)
        - model_selection_metrics (if available)
        - syntax metrics
    """
    successful_runs_count = len(template_scores)
    avg_score = sum(template_scores) / successful_runs_count if successful_runs_count > 0 else None

    # Filter runs for metric calculations
    successful_run_data = [r for r in run_results if not r.get("error")]
    runs_for_syntax = [r for r in run_results if not r.get("error") or r.get("error_type") == "syntax_error"]

    # Calculate syntax metrics
    syntax_metrics = calculate_syntax_metrics(runs_for_syntax)

    # Collect result-based metrics
    execution_accuracies = [r.get("execution_accuracy", 0.0) for r in successful_run_data if "execution_accuracy" in r]
    precisions = [r.get("precision", 0.0) for r in successful_run_data if "precision" in r]
    recalls = [r.get("recall", 0.0) for r in successful_run_data if "recall" in r]
    f1_scores = [r.get("f1_score", 0.0) for r in successful_run_data if "f1_score" in r]

    result_metrics = {
        "execution_accuracy": calc_stats(execution_accuracies),
        "precision": calc_stats(precisions),
        "recall": calc_stats(recalls),
        "f1_score": calc_stats(f1_scores)
    }

    # Collect variable-level metrics
    variable_precisions = [r.get("variable_precision", 0.0) for r in successful_run_data if "variable_precision" in r]
    variable_recalls = [r.get("variable_recall", 0.0) for r in successful_run_data if "variable_recall" in r]
    variable_f1s = [r.get("variable_f1", 0.0) for r in successful_run_data if "variable_f1" in r]

    variable_level_metrics = None
    if variable_precisions or variable_recalls or variable_f1s:
        variable_level_metrics = {
            "variable_precision": calc_stats(variable_precisions),
            "variable_recall": calc_stats(variable_recalls),
            "variable_f1": calc_stats(variable_f1s)
        }

    # Collect model selection metrics
    model_precisions = [
        r.get("model_selection_metrics", {}).get("precision", 0.0)
        for r in successful_run_data if "model_selection_metrics" in r
    ]
    model_recalls = [
        r.get("model_selection_metrics", {}).get("recall", 0.0)
        for r in successful_run_data if "model_selection_metrics" in r
    ]
    model_f1_scores = [
        r.get("model_selection_metrics", {}).get("f1_score", 0.0)
        for r in successful_run_data if "model_selection_metrics" in r
    ]

    model_selection_metrics = None
    if model_precisions:
        model_selection_metrics = {
            "precision": calc_stats(model_precisions),
            "recall": calc_stats(model_recalls),
            "f1_score": calc_stats(model_f1_scores)
        }

    return {
        "average_score": round(avg_score, 2) if avg_score is not None else None,
        "successful_runs": successful_runs_count,
        "failed_runs": num_runs - successful_runs_count,
        "scores": [round(s, 2) for s in template_scores],
        "result_metrics": result_metrics,
        "variable_level_metrics": variable_level_metrics,
        "model_selection_metrics": model_selection_metrics,
        **syntax_metrics
    }
