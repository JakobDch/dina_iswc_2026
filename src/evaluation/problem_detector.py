"""
Problem Detection for VKGQA Evaluation.

This module implements metric-based problem detection by analyzing
AdaptiveMetricsResult to identify specific failure modes.

The detection is primarily based on:
- Schema metrics (concept coverage via semantic_concept)
- Tuple metrics (recall, precision)
- Ambiguity type context

No SPARQL parsing is performed - all detection uses existing metrics.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

# Try multiple import strategies for problem_categorization
# This handles both direct module loading and package imports
try:
    # When loaded directly via importlib
    from problem_categorization import (
        AmbiguityType,
        CategoryStats,
        CorrelationEntry,
        DetectedProblem,
        EvaluationReport,
        ProblemAnalysisResult,
        ProblemCategory,
        ProblemStats,
        Severity,
        parse_ambiguity_type,
    )
except ImportError:
    try:
        # When loaded as part of src.evaluation package
        from .problem_categorization import (
            AmbiguityType,
            CategoryStats,
            CorrelationEntry,
            DetectedProblem,
            EvaluationReport,
            ProblemAnalysisResult,
            ProblemCategory,
            ProblemStats,
            Severity,
            parse_ambiguity_type,
        )
    except ImportError:
        # Fallback: absolute import
        from src.evaluation.problem_categorization import (
            AmbiguityType,
            CategoryStats,
            CorrelationEntry,
            DetectedProblem,
            EvaluationReport,
            ProblemAnalysisResult,
            ProblemCategory,
            ProblemStats,
            Severity,
            parse_ambiguity_type,
        )

# Type aliases - actual types are injected at runtime by the caller
# This avoids importing modules that cause conflicts
AdaptiveMetricsResult = Any  # from .tiered_metrics_tuples
AdaptiveGroundTruth = Any  # from data.queries.use_case_queries_tiered_tuples


# Thresholds for problem detection
RECALL_CRITICAL_THRESHOLD = 0.0  # Empty results
RECALL_MAJOR_THRESHOLD = 0.3  # Very low recall
RECALL_MINOR_THRESHOLD = 0.7  # Moderate recall issues

SCHEMA_RECALL_THRESHOLD = 0.8  # Schema coverage threshold

SUCCESS_F1_THRESHOLD = 0.8  # F1 >= 0.8 counts as success


def detect_problems(
    adaptive_metrics: AdaptiveMetricsResult,
    ground_truth: AdaptiveGroundTruth,
    ambiguity_type: AmbiguityType,
    trace_key: str,
) -> ProblemAnalysisResult:
    """Detect problems in a single trace based on metrics.

    Args:
        adaptive_metrics: The adaptive evaluation metrics for this trace
        ground_truth: The ground truth query definition
        ambiguity_type: The ambiguity type of this query
        trace_key: Unique identifier for this trace

    Returns:
        ProblemAnalysisResult with all detected problems
    """
    result = ProblemAnalysisResult(
        trace_key=trace_key,
        query_id=ground_truth.query_id,
        query_set=ground_truth.query_set,
        dataset=ground_truth.dataset,
        ambiguity_type=ambiguity_type,
    )

    problems: list[DetectedProblem] = []

    # === 1. Empty Results Detection ===
    if adaptive_metrics.llm_row_count == 0:
        problems.append(DetectedProblem(
            category=ProblemCategory.EMPTY_OVER_CONSTRAINED,
            severity=Severity.CRITICAL,
            description="Query returned 0 rows - likely over-constrained",
            evidence={
                "llm_row_count": 0,
                "expected_gt_size": adaptive_metrics.best_recall_gt_size,
            },
        ))
        result.has_empty_result = True

    # === 2. Schema/Concept Coverage Detection ===
    if adaptive_metrics.schema_recall < 1.0:
        # Find which concepts are missing by comparing expected vs matched
        missing_concepts = _find_missing_concepts(adaptive_metrics, ground_truth)
        result.missing_concepts = missing_concepts

        if adaptive_metrics.schema_recall == 0.0:
            problems.append(DetectedProblem(
                category=ProblemCategory.CONCEPT_MISSING,
                severity=Severity.CRITICAL,
                description="No semantic concepts from Ground Truth were covered",
                evidence={
                    "schema_recall": 0.0,
                    "expected_concepts": adaptive_metrics.schema_expected_count,
                    "missing_concepts": missing_concepts,
                },
            ))
            result.has_concept_failure = True
        elif adaptive_metrics.schema_recall < SCHEMA_RECALL_THRESHOLD:
            problems.append(DetectedProblem(
                category=ProblemCategory.CONCEPT_PARTIAL,
                severity=Severity.MAJOR,
                description=f"Only {adaptive_metrics.schema_recall:.1%} of concepts covered",
                evidence={
                    "schema_recall": adaptive_metrics.schema_recall,
                    "schema_matched": adaptive_metrics.schema_matched_count,
                    "schema_expected": adaptive_metrics.schema_expected_count,
                    "missing_concepts": missing_concepts,
                },
            ))
            result.has_concept_failure = True

    # === 3. Measurement Column Detection ===
    measurement_indices = ground_truth.get_measurement_column_indices()
    if measurement_indices:
        # Check if measurement values are present
        # Measurement columns have is_measurement=True
        cols = ground_truth.get_columns_for_query()
        measurement_concepts = [
            cols[i].semantic_concept
            for i in measurement_indices
            if i < len(cols) and cols[i].semantic_concept
        ]

        # If schema recall is low and we have measurements, check if they're the issue
        if adaptive_metrics.schema_matched_count < len(measurement_concepts):
            missing_measurements = [
                c for c in measurement_concepts
                if c in result.missing_concepts
            ]
            if missing_measurements:
                problems.append(DetectedProblem(
                    category=ProblemCategory.MEASUREMENT_MISSING,
                    severity=Severity.MAJOR,
                    description=f"Measurement columns missing: {', '.join(missing_measurements)}",
                    evidence={
                        "missing_measurements": missing_measurements,
                        "expected_measurements": measurement_concepts,
                    },
                ))
                result.has_measurement_failure = True

    # === 4. Instance Matching Detection (based on ambiguity type) ===
    if adaptive_metrics.best_recall < RECALL_MINOR_THRESHOLD and not result.has_empty_result:
        severity = Severity.CRITICAL if adaptive_metrics.best_recall < RECALL_MAJOR_THRESHOLD else Severity.MAJOR

        if ambiguity_type == AmbiguityType.CASE_VARIATION:
            problems.append(DetectedProblem(
                category=ProblemCategory.INSTANCE_CASE_MISMATCH,
                severity=severity,
                description="Low recall likely due to case variation not handled",
                evidence={
                    "recall": adaptive_metrics.best_recall,
                    "ambiguity_type": ambiguity_type.value,
                },
            ))
            result.has_instance_failure = True

        elif ambiguity_type == AmbiguityType.SYNONYM:
            problems.append(DetectedProblem(
                category=ProblemCategory.INSTANCE_SYNONYM_NOT_RESOLVED,
                severity=severity,
                description="Low recall likely due to synonym not resolved",
                evidence={
                    "recall": adaptive_metrics.best_recall,
                    "notes": ground_truth.notes,
                },
            ))
            result.has_instance_failure = True

        elif ambiguity_type == AmbiguityType.TEMPORAL:
            problems.append(DetectedProblem(
                category=ProblemCategory.INSTANCE_TEMPORAL_NOT_RESOLVED,
                severity=severity,
                description="Low recall likely due to temporal name change not resolved",
                evidence={
                    "recall": adaptive_metrics.best_recall,
                    "notes": ground_truth.notes,
                },
            ))
            result.has_instance_failure = True

        elif ambiguity_type == AmbiguityType.FUZZY_TYPO:
            problems.append(DetectedProblem(
                category=ProblemCategory.INSTANCE_FUZZY_NOT_RESOLVED,
                severity=severity,
                description="Low recall likely due to fuzzy/typo match not resolved",
                evidence={
                    "recall": adaptive_metrics.best_recall,
                    "notes": ground_truth.notes,
                },
            ))
            result.has_instance_failure = True

        elif ambiguity_type == AmbiguityType.CROSS_DATASET:
            problems.append(DetectedProblem(
                category=ProblemCategory.CROSS_MISSING_QUERY,
                severity=severity,
                description="Low recall in cross-dataset query - likely missing sub-query",
                evidence={
                    "recall": adaptive_metrics.best_recall,
                    "datasets": ground_truth.datasets,
                },
            ))
            result.has_cross_dataset_failure = True

        elif ambiguity_type == AmbiguityType.MULTI_STEP:
            problems.append(DetectedProblem(
                category=ProblemCategory.CROSS_MISSING_QUERY,
                severity=severity,
                description="Low recall in multi-step query - steps may not be executed",
                evidence={
                    "recall": adaptive_metrics.best_recall,
                    "notes": ground_truth.notes,
                },
            ))
            result.has_cross_dataset_failure = True

        else:
            # Generic low recall
            problems.append(DetectedProblem(
                category=ProblemCategory.LOW_RECALL,
                severity=severity,
                description=f"Low recall ({adaptive_metrics.best_recall:.1%}) - cause not determined",
                evidence={
                    "recall": adaptive_metrics.best_recall,
                    "ambiguity_type": ambiguity_type.value,
                },
            ))

    # === 5. Aggregation Detection ===
    # Check if this is an aggregation query (has measurement columns with GROUP BY expected)
    if measurement_indices and result.has_measurement_failure:
        # If measurements are missing in an aggregation query, likely GROUP BY issue
        problems.append(DetectedProblem(
            category=ProblemCategory.AGG_MISSING_GROUP_BY,
            severity=Severity.MAJOR,
            description="Aggregation query missing measurement values - possible GROUP BY issue",
            evidence={
                "measurement_count": len(measurement_indices),
            },
        ))
        result.has_aggregation_failure = True

    result.problems = problems
    return result


def _find_missing_concepts(
    adaptive_metrics: AdaptiveMetricsResult,
    ground_truth: AdaptiveGroundTruth,
) -> list[str]:
    """Find semantic concepts that are missing from LLM results.

    Args:
        adaptive_metrics: The metrics result
        ground_truth: The ground truth definition

    Returns:
        List of missing semantic_concept values
    """
    # Get all unique semantic concepts from ground truth
    all_concepts = set()
    for col in ground_truth.get_columns_for_query():
        if col.semantic_concept:
            all_concepts.add(col.semantic_concept)

    # We can't directly know which concepts matched without re-running evaluation,
    # but we can infer from schema_matched_count vs schema_expected_count
    # For now, return all concepts if schema_recall < 1.0
    if adaptive_metrics.schema_recall < 1.0:
        # Best effort: return concept names
        return sorted(all_concepts)

    return []


def generate_report(
    analyses: list[ProblemAnalysisResult],
    metrics_by_trace: dict[str, AdaptiveMetricsResult],
    experiment_name: str,
    timestamp: str,
) -> EvaluationReport:
    """Generate a comprehensive evaluation report from all analyses.

    Args:
        analyses: List of ProblemAnalysisResult for each trace
        metrics_by_trace: Mapping of trace_key to AdaptiveMetricsResult
        experiment_name: Name of the experiment
        timestamp: Timestamp of the report generation

    Returns:
        Complete EvaluationReport with all statistics
    """
    report = EvaluationReport(
        experiment_name=experiment_name,
        timestamp=timestamp,
        total_traces=len(analyses),
        overall_success_rate=0.0,
        overall_avg_f1=0.0,
    )

    # Aggregate statistics
    by_query_set: dict[str, list[tuple[ProblemAnalysisResult, AdaptiveMetricsResult]]] = defaultdict(list)
    by_ambiguity: dict[str, list[tuple[ProblemAnalysisResult, AdaptiveMetricsResult]]] = defaultdict(list)
    by_dataset: dict[str, list[tuple[ProblemAnalysisResult, AdaptiveMetricsResult]]] = defaultdict(list)
    by_problem: dict[ProblemCategory, list[ProblemAnalysisResult]] = defaultdict(list)

    all_f1_scores = []

    for analysis in analyses:
        metrics = metrics_by_trace.get(analysis.trace_key)
        if not metrics:
            continue

        f1 = metrics.best_f1
        all_f1_scores.append(f1)

        # Group by dimensions
        by_query_set[analysis.query_set].append((analysis, metrics))
        by_ambiguity[analysis.ambiguity_type.value].append((analysis, metrics))
        by_dataset[analysis.dataset].append((analysis, metrics))

        # Track problems
        for problem in analysis.problems:
            by_problem[problem.category].append(analysis)

    # Calculate overall stats
    if all_f1_scores:
        report.overall_avg_f1 = sum(all_f1_scores) / len(all_f1_scores)
        report.overall_success_rate = sum(1 for f in all_f1_scores if f >= SUCCESS_F1_THRESHOLD) / len(all_f1_scores)

    # Calculate per-category stats
    report.by_query_set = {
        k: _calculate_category_stats(k, v) for k, v in by_query_set.items()
    }
    report.by_ambiguity_type = {
        k: _calculate_category_stats(k, v) for k, v in by_ambiguity.items()
    }
    report.by_dataset = {
        k: _calculate_category_stats(k, v) for k, v in by_dataset.items()
    }

    # Calculate problem stats
    for problem, affected in by_problem.items():
        stats = ProblemStats(
            problem=problem,
            occurrence_count=len(affected),
            affected_queries=list({a.query_id for a in affected}),
            affected_traces=[a.trace_key for a in affected],
        )
        report.by_problem[problem.value] = stats

    # Calculate correlations
    report.correlations = _calculate_correlations(analyses)

    # Generate insights
    report.insights = _generate_insights(report)

    # Find hardest ambiguity types (lowest avg F1)
    ambiguity_f1 = [
        (k, v.avg_f1) for k, v in report.by_ambiguity_type.items()
    ]
    report.hardest_ambiguity_types = sorted(ambiguity_f1, key=lambda x: x[1])[:5]

    # Find most common problems
    problem_counts = [
        (k, v.occurrence_count) for k, v in report.by_problem.items()
    ]
    report.most_common_problems = sorted(problem_counts, key=lambda x: -x[1])[:10]

    return report


def _calculate_category_stats(
    name: str,
    items: list[tuple[ProblemAnalysisResult, AdaptiveMetricsResult]],
) -> CategoryStats:
    """Calculate statistics for a category."""
    stats = CategoryStats(name=name, total=len(items))

    if not items:
        return stats

    f1_scores = []
    recall_scores = []
    precision_scores = []

    for _, metrics in items:
        f1 = metrics.best_f1
        f1_scores.append(f1)
        recall_scores.append(metrics.best_recall)
        precision_scores.append(metrics.best_precision)

        if f1 >= SUCCESS_F1_THRESHOLD:
            stats.success_count += 1
        elif f1 == 0:
            stats.failure_count += 1
        else:
            stats.partial_count += 1

    stats.avg_f1 = sum(f1_scores) / len(f1_scores)
    stats.avg_recall = sum(recall_scores) / len(recall_scores)
    stats.avg_precision = sum(precision_scores) / len(precision_scores)

    return stats


def _calculate_correlations(analyses: list[ProblemAnalysisResult]) -> list[CorrelationEntry]:
    """Calculate correlations between ambiguity types and problem categories."""
    correlations = []

    # Count occurrences per ambiguity type
    ambiguity_counts: dict[AmbiguityType, int] = defaultdict(int)
    for analysis in analyses:
        ambiguity_counts[analysis.ambiguity_type] += 1

    # Count co-occurrences
    co_occurrences: dict[tuple[AmbiguityType, ProblemCategory], int] = defaultdict(int)
    for analysis in analyses:
        for problem in analysis.problems:
            co_occurrences[(analysis.ambiguity_type, problem.category)] += 1

    # Calculate conditional probabilities
    for (ambiguity, problem), count in co_occurrences.items():
        total_with_ambiguity = ambiguity_counts[ambiguity]
        if total_with_ambiguity > 0:
            prob = count / total_with_ambiguity
            correlations.append(CorrelationEntry(
                ambiguity_type=ambiguity,
                problem_category=problem,
                co_occurrence_count=count,
                conditional_probability=prob,
            ))

    # Sort by probability descending
    correlations.sort(key=lambda x: -x.conditional_probability)

    return correlations


def _generate_insights(report: EvaluationReport) -> list[str]:
    """Generate actionable insights from the report."""
    insights = []

    # Compare SET A/B baseline to SET C instance matching
    baseline_f1 = None
    instance_f1 = None

    if "A" in report.by_query_set and "B" in report.by_query_set:
        baseline_f1 = (report.by_query_set["A"].avg_f1 + report.by_query_set["B"].avg_f1) / 2
    if "C" in report.by_query_set:
        instance_f1 = report.by_query_set["C"].avg_f1

    if baseline_f1 is not None and instance_f1 is not None:
        diff = baseline_f1 - instance_f1
        if diff > 0.1:
            insights.append(
                f"Instance matching (SET C) has {diff:.1%} lower F1 than baseline (SET A/B)"
            )

    # Check for synonym failures
    if "synonym" in report.by_ambiguity_type:
        stats = report.by_ambiguity_type["synonym"]
        if stats.success_rate() < 0.3:
            insights.append(
                f"Synonym resolution fails in {1 - stats.success_rate():.0%} of cases"
            )

    # Check for cross-dataset failures
    if "cross_dataset" in report.by_ambiguity_type:
        stats = report.by_ambiguity_type["cross_dataset"]
        if stats.success_rate() < 0.2:
            insights.append(
                "Cross-dataset queries need multi-step execution support"
            )

    # Check for common problems
    if report.most_common_problems:
        top_problem, count = report.most_common_problems[0]
        if count > len(report.by_problem) * 0.2:
            insights.append(
                f"Most common failure: {top_problem} ({count} occurrences)"
            )

    # Check measurement/aggregation issues
    measurement_stats = report.by_problem.get(ProblemCategory.MEASUREMENT_MISSING.value)
    if measurement_stats and measurement_stats.occurrence_count > 5:
        insights.append(
            f"Measurement columns often missing ({measurement_stats.occurrence_count} cases) - check aggregation handling"
        )

    return insights
