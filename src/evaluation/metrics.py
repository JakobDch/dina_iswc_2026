"""
Metrics calculation and aggregation for experiments.
"""

import statistics
from typing import TypeVar

from src.models.evaluation import (
    ExperimentResult,
    QueryAggregatedResult,
)

T = TypeVar("T", int, float)


def safe_mean(values: list[T]) -> float:
    """Calculate mean, returning 0.0 for empty lists."""
    return statistics.mean(values) if values else 0.0


def safe_stdev(values: list[T]) -> float:
    """Calculate standard deviation, returning 0.0 for lists with < 2 elements."""
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def calculate_metrics(result: ExperimentResult) -> dict:
    """
    Calculate summary metrics from experiment results.

    Returns dict with metrics grouped by approach and model.
    """
    summary = {
        "by_approach": {},
        "by_model": {},
        "overall": {},
    }

    # Group results
    by_approach: dict[str, list[QueryAggregatedResult]] = {}
    by_model: dict[str, list[QueryAggregatedResult]] = {}

    for qr in result.results:
        by_approach.setdefault(qr.approach, []).append(qr)
        by_model.setdefault(qr.llm_model, []).append(qr)

    # Calculate per-approach metrics
    for approach, results in by_approach.items():
        exec_accs = [r.mean_execution_accuracy for r in results]
        f1_scores = [r.mean_f1_score for r in results]
        edit_costs = [r.mean_edit_cost for r in results]
        success_rates = [r.success_rate for r in results]

        summary["by_approach"][approach] = {
            "execution_accuracy": {
                "mean": safe_mean(exec_accs),
                "std": safe_stdev(exec_accs),
            },
            "f1_score": {
                "mean": safe_mean(f1_scores),
                "std": safe_stdev(f1_scores),
            },
            "edit_cost": {
                "mean": safe_mean(edit_costs),
                "std": safe_stdev(edit_costs),
            },
            "success_rate": {
                "mean": safe_mean(success_rates),
                "std": safe_stdev(success_rates),
            },
            "num_queries": len(results),
        }

    # Calculate per-model metrics
    for model, results in by_model.items():
        exec_accs = [r.mean_execution_accuracy for r in results]
        f1_scores = [r.mean_f1_score for r in results]

        summary["by_model"][model] = {
            "execution_accuracy": {
                "mean": safe_mean(exec_accs),
                "std": safe_stdev(exec_accs),
            },
            "f1_score": {
                "mean": safe_mean(f1_scores),
                "std": safe_stdev(f1_scores),
            },
            "num_queries": len(results),
        }

    # Overall metrics
    all_exec_accs = [r.mean_execution_accuracy for r in result.results]
    all_f1_scores = [r.mean_f1_score for r in result.results]

    summary["overall"] = {
        "execution_accuracy": {
            "mean": safe_mean(all_exec_accs),
            "std": safe_stdev(all_exec_accs),
        },
        "f1_score": {
            "mean": safe_mean(all_f1_scores),
            "std": safe_stdev(all_f1_scores),
        },
        "total_queries": result.total_queries,
        "total_runs": result.total_runs,
    }

    return summary


def generate_latex_table(result: ExperimentResult) -> str:
    """
    Generate a LaTeX table from experiment results.

    Returns LaTeX code for a results comparison table.
    """
    metrics = calculate_metrics(result)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Experiment Results: " + result.experiment_name + "}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Approach & Exec. Acc. & F1 Score & Edit Cost & Success Rate \\",
        r"\midrule",
    ]

    for approach, data in metrics["by_approach"].items():
        exec_acc = data["execution_accuracy"]
        f1 = data["f1_score"]
        edit = data["edit_cost"]
        success = data["success_rate"]

        line = (
            f"{approach} & "
            f"{exec_acc['mean']:.3f} $\\pm$ {exec_acc['std']:.3f} & "
            f"{f1['mean']:.3f} $\\pm$ {f1['std']:.3f} & "
            f"{edit['mean']:.3f} $\\pm$ {edit['std']:.3f} & "
            f"{success['mean']:.3f} $\\pm$ {success['std']:.3f} \\\\"
        )
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:results}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_csv_summary(result: ExperimentResult) -> str:
    """
    Generate a CSV summary of experiment results.
    """
    lines = [
        "query_id,approach,llm_model,exec_accuracy,f1_score,edit_cost,success_rate"
    ]

    for qr in result.results:
        line = (
            f"{qr.query_id},{qr.approach},{qr.llm_model},"
            f"{qr.mean_execution_accuracy:.4f},{qr.mean_f1_score:.4f},"
            f"{qr.mean_edit_cost:.4f},{qr.success_rate:.4f}"
        )
        lines.append(line)

    return "\n".join(lines)
