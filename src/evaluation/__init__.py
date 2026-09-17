"""Evaluation framework for experiments."""

import importlib as _importlib

def __getattr__(name):
    """Lazy-load ExperimentRunner and metrics to avoid triggering langchain import chain."""
    if name == "ExperimentRunner":
        mod = _importlib.import_module(".runner", __name__)
        return mod.ExperimentRunner
    if name == "calculate_metrics":
        mod = _importlib.import_module(".metrics", __name__)
        return mod.calculate_metrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
from .sparql_metrics import (
    calculate_sparql_metrics,
    calculate_query_execution_success,
    extract_result_set,
)
# Adaptive evaluation (tuple-based with NULL variant handling and semantic concepts)
from .tiered_metrics_tuples import (
    # Result data structures
    TieredMetricsResult,
    AdaptiveMetricsResult,
    ColumnMetadata,
    # Adaptive evaluation (primary API)
    evaluate_adaptive_ground_truth,
    calculate_adaptive_metrics,
    detect_active_levels,
    generate_null_variants,
    # Schema metrics
    calculate_schema_metrics,
    extract_llm_column_values,
    # Value/row extraction utilities
    extract_result_values,
    normalize_value,
    is_trivial_value,
    # Tuple matching
    calculate_metrics_tuple_match,
)

__all__ = [
    "ExperimentRunner",
    "calculate_metrics",
    "calculate_sparql_metrics",
    "calculate_query_execution_success",
    "extract_result_set",
    # Result data structures
    "TieredMetricsResult",
    "AdaptiveMetricsResult",
    "ColumnMetadata",
    # Adaptive evaluation (primary API)
    "evaluate_adaptive_ground_truth",
    "calculate_adaptive_metrics",
    "detect_active_levels",
    "generate_null_variants",
    # Schema metrics
    "calculate_schema_metrics",
    "extract_llm_column_values",
    # Value/row extraction utilities
    "extract_result_values",
    "normalize_value",
    "is_trivial_value",
    # Tuple matching
    "calculate_metrics_tuple_match",
]
