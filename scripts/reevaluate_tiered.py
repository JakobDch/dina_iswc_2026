#!/usr/bin/env python3
"""
Re-evaluate experiment traces against the tiered/adaptive ground truth.

This script loads existing experiment traces and recalculates metrics
using either:
- Tiered relevance system (legacy): PREFERRED, ACCEPTABLE value sets
- Adaptive evaluation (new): Single SPARQL query with NULL variant handling

Usage:
    # Legacy tiered evaluation
    python scripts/reevaluate_tiered.py EXPERIMENT_NAME

    # New adaptive evaluation
    python scripts/reevaluate_tiered.py EXPERIMENT_NAME --adaptive
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from data.queries.experimental_corpus import (
    ALL_EXPERIMENTAL_QUERIES,
    get_experimental_query,
)
from data.queries.use_case_queries_tiered_tuples import (
    AdaptiveGroundTruth,
    ValueSet,
)
from src.evaluation.retrieval_ground_truth import build_schema_gt_for_query
from src.evaluation.retrieval_metrics import calculate_retrieval_metrics
from src.validation.schema_graph import get_combined_schema
from src.evaluation.tiered_metrics_tuples import (
    TieredMetricsResult,
    AdaptiveMetricsResult,
    evaluate_adaptive_ground_truth,
    detect_active_levels,
    generate_null_variants,
    calculate_adaptive_metrics,
    calculate_schema_metrics,
    extract_result_values,
    extract_llm_column_values,
    normalize_value,
    is_trivial_value,
)
from data.queries.use_case_queries_tiered_tuples import RelevanceLevel
from src.config import RESULTS_DIR, ONTOP_SINGLE_ENDPOINTS, CACHE_DIR

# Use centralized endpoint configuration from src/config.py
# This ensures consistency and avoids hardcoded ports.
# Note: Reevaluation uses the original containers (8080-8085), not slot containers (8100+),
# since reevaluation runs after the experiment when slot context is not active.
# Includes both small and large endpoints for SET_LARGE queries.
ENDPOINTS = {
    # Small datasets (default)
    "EDU": ONTOP_SINGLE_ENDPOINTS.get("edu", ""),
    "TRN": ONTOP_SINGLE_ENDPOINTS.get("trn", ""),
    "NRG": ONTOP_SINGLE_ENDPOINTS.get("nrg", ""),
    "BSBM": ONTOP_SINGLE_ENDPOINTS.get("bsbm", ""),
    "LCA": ONTOP_SINGLE_ENDPOINTS.get("lca", ""),
    # Large datasets (for SET_LARGE queries)
    "EDU_LARGE": ONTOP_SINGLE_ENDPOINTS.get("edu-large", ""),
    "TRN_LARGE": ONTOP_SINGLE_ENDPOINTS.get("trn-large", ""),
    "NPD_LARGE": ONTOP_SINGLE_ENDPOINTS.get("nrg-large", ""),
}

# Ground Truth cache file (extracted by extract_ground_truth_cache.py)
GT_CACHE_FILE = CACHE_DIR / "ground_truth_full.json"

# Global cache storage (loaded once at startup)
_gt_cache: dict | None = None


def load_gt_cache() -> dict | None:
    """Load the Ground Truth cache if available.

    Returns:
        Cache dict or None if not available.
    """
    global _gt_cache
    if _gt_cache is not None:
        return _gt_cache

    if GT_CACHE_FILE.exists():
        try:
            with open(GT_CACHE_FILE, "r", encoding="utf-8") as f:
                _gt_cache = json.load(f)
            print(f"Loaded GT cache: {len(_gt_cache.get('queries', {}))} queries")
            return _gt_cache
        except Exception as e:
            print(f"Warning: Failed to load GT cache: {e}")
            return None
    return None


def get_cached_gt(query_id: str) -> dict | None:
    """Get cached GT data for a specific query.

    Returns:
        Cached GT dict or None if not cached.
    """
    cache = load_gt_cache()
    if cache:
        return cache.get("queries", {}).get(query_id)
    return None


def preload_gt_internal_cache(ground_truth: AdaptiveGroundTruth, cached_gt: dict) -> None:
    """Preload the internal cache of an AdaptiveGroundTruth object.

    This allows evaluate_adaptive_ground_truth to skip endpoint calls
    by using the pre-populated internal cache.

    Supports both legacy (single-variant) and new (multi-variant) cache formats.

    Args:
        ground_truth: The AdaptiveGroundTruth object to preload
        cached_gt: Cached GT data from ground_truth_full.json
    """
    if not cached_gt or not cached_gt.get("success"):
        return

    # Load all query variants if available
    variants = cached_gt.get("query_variants", [])
    if variants:
        for variant in variants:
            qi = variant.get("query_index", 0)
            if variant.get("success"):
                gt_tuples = {tuple(t) for t in variant.get("gt_tuples", [])}
                gt_columns = variant.get("gt_columns", [])
                ground_truth._cached_tuples[qi] = gt_tuples
                ground_truth._cached_column_names[qi] = gt_columns
    else:
        # Legacy format: single variant at index 0
        gt_tuples = {tuple(t) for t in cached_gt.get("gt_tuples", [])}
        gt_columns = cached_gt.get("gt_columns", [])
        ground_truth._cached_tuples[0] = gt_tuples
        ground_truth._cached_column_names[0] = gt_columns


def is_cross_query(ground_truth: AdaptiveGroundTruth) -> bool:
    """Check if this is a cross-dataset query."""
    return len(ground_truth.datasets) > 1


def parse_cross_sparql(sparql: str, datasets: list[str]) -> dict[str, str]:
    """Parse a CROSS query SPARQL into parts by dataset.

    DEPRECATED: This function is no longer used. CROSS queries now use UNION
    patterns where the same query is sent to all endpoints. Each endpoint
    returns results only for the data it has.

    This function was designed for a comment-marker format:
        # Part 1: EDU (8080)
        PREFIX ... SELECT ...

        # Part 2: TRN (8081)
        PREFIX ... SELECT ...

    Returns:
        Dict mapping dataset name to its SPARQL query part
    """
    parts: dict[str, str] = {}
    lines = sparql.split('\n')

    current_dataset = None
    current_lines: list[str] = []

    for line in lines:
        # Check for Part marker: "# Part X: DATASET"
        if line.strip().startswith('# Part') and ':' in line:
            # Save previous part if exists
            if current_dataset and current_lines:
                parts[current_dataset] = '\n'.join(current_lines).strip()
                current_lines = []

            # Extract dataset name from comment
            # Format: "# Part 1: EDU (8080)" or "# Part 2: TRN"
            part_info = line.split(':', 1)[1].strip()
            # Extract just the dataset name (before any parenthesis)
            dataset_name = part_info.split('(')[0].strip().upper()

            # Match to datasets list
            for ds in datasets:
                if ds.upper() == dataset_name:
                    current_dataset = ds
                    break
        elif current_dataset:
            # Skip comment lines at start of part
            if not line.strip().startswith('#'):
                current_lines.append(line)

    # Save last part
    if current_dataset and current_lines:
        parts[current_dataset] = '\n'.join(current_lines).strip()

    return parts


def execute_cross_query(
    ground_truth: AdaptiveGroundTruth,
    timeout: float = 60.0
) -> tuple[set[tuple[str, ...]], list[str]]:
    """Execute a CROSS query on multiple endpoints and merge results.

    CROSS queries use UNION patterns that reference multiple datasets. Since each
    VKG endpoint only has data for one dataset, we send the SAME query to ALL
    endpoints. Each endpoint returns results only for the UNION branches it can
    answer (branches referencing other datasets return empty results).

    Example: A query with UNION { ?x a eduo:Student } UNION { ?y a eno:Field }
    - EDU endpoint returns Student results (Field branch is empty)
    - NRG endpoint returns Field results (Student branch is empty)
    - Merged results contain both Students and Fields

    Returns:
        Tuple of (merged_tuples, column_names)
    """
    import httpx

    if not ground_truth.sparql_queries:
        return set(), []

    sparql = ground_truth.sparql_queries[0]

    all_tuples: set[tuple[str, ...]] = set()
    all_columns: list[str] = []

    # Send the SAME query to ALL endpoints and merge results
    for dataset in ground_truth.datasets:
        endpoint = ENDPOINTS.get(dataset.upper())
        if not endpoint:
            # Try without case conversion
            endpoint = ENDPOINTS.get(dataset)
        if not endpoint:
            print(f"  Warning: No endpoint for dataset {dataset}")
            continue

        try:
            response = httpx.post(
                endpoint,
                data={"query": sparql},
                headers={"Accept": "application/sparql-results+json"},
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()

            bindings = result.get("results", {}).get("bindings", [])
            vars_list = result.get("head", {}).get("vars", [])

            # Set columns from first successful response
            if not all_columns and vars_list:
                all_columns = vars_list

            # Extract tuples and normalize values
            for binding in bindings:
                row_values = []
                for var in all_columns:
                    val = binding.get(var, {})
                    if isinstance(val, dict):
                        # Normalize: strip whitespace, lowercase for comparison
                        row_values.append(val.get("value", "").strip().lower())
                    else:
                        row_values.append(str(val).strip().lower() if val else "")
                if row_values and any(v for v in row_values):
                    all_tuples.add(tuple(row_values))

        except Exception as e:
            print(f"  Error executing CROSS query on {dataset}: {e}")

    return all_tuples, all_columns


def load_trace(trace_file: Path) -> dict | None:
    """Load a single trace file."""
    try:
        with open(trace_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {trace_file}: {e}")
        return None


def extract_row_dict(binding: dict) -> dict:
    """Extract a clean row dict from a SPARQL binding."""
    row = {}
    for var, val in binding.items():
        if var.startswith("_"):
            continue
        if isinstance(val, dict):
            row[var] = val.get("value", "")
        else:
            row[var] = str(val)
    return row


def evaluate_trace_adaptive(
    trace: dict, ground_truth: AdaptiveGroundTruth, endpoint: str
) -> dict:
    """Evaluate a single trace against adaptive ground truth (new system).

    This enhanced version also captures:
    - GT sample rows AFTER column projection (for dashboard comparison)
    - Active columns information
    - Which columns were filtered out

    For CROSS queries (multiple datasets), executes on all endpoints and merges results.
    """
    generated_results = trace.get("final_results", [])
    # Extract LLM SPARQL query for signature-based column matching
    llm_sparql = trace.get("final_sparql", None)
    gen_values = extract_result_values(generated_results)

    # Check if this is a CROSS query (multiple datasets)
    is_cross = is_cross_query(ground_truth)

    # Try to use cached GT data first (faster, no endpoint dependency)
    cached_gt = get_cached_gt(ground_truth.query_id)

    if cached_gt and cached_gt.get("success"):
        # Use cached data
        gt_tuples = {tuple(t) for t in cached_gt.get("gt_tuples", [])}
        gt_col_names = cached_gt.get("gt_columns", [])
        columns_with_nulls = cached_gt.get("null_columns", [])
        values_by_level = cached_gt.get("values_by_level", {})
        essential_values = set(values_by_level.get("PREFERRED", []))
        preferred_values = essential_values
        acceptable_values = set(values_by_level.get("ACCEPTABLE", []))
    else:
        # Fallback: Execute GT query to get raw tuples and column info
        try:
            if is_cross:
                # CROSS query: execute on multiple endpoints and merge results
                gt_tuples, gt_col_names = execute_cross_query(ground_truth, timeout=60.0)
                columns_with_nulls = []  # Not tracked for CROSS queries
            else:
                # Normal query: execute on single endpoint
                gt_tuples, gt_col_names, columns_with_nulls = ground_truth.execute_query(endpoint, timeout=60.0)
        except Exception as e:
            print(f"  Error executing GT query: {e}")
            gt_tuples, gt_col_names, columns_with_nulls = set(), [], []

        # Get values by level for detection
        # NOTE: essential_values and preferred_values are now identical (ESSENTIAL removed - now only PREFERRED)
        try:
            if is_cross:
                # CROSS query: get values from all endpoints
                essential_values: set[str] = set()
                acceptable_values: set[str] = set()
                for dataset in ground_truth.datasets:
                    ds_endpoint = ENDPOINTS.get(dataset.upper()) or ENDPOINTS.get(dataset)
                    if ds_endpoint:
                        try:
                            ess = ground_truth.get_values_by_level(RelevanceLevel.PREFERRED, ds_endpoint, timeout=60.0)
                            acc = ground_truth.get_values_by_level(RelevanceLevel.ACCEPTABLE, ds_endpoint, timeout=60.0)
                            essential_values.update(ess)
                            acceptable_values.update(acc)
                        except Exception:
                            pass
                preferred_values = essential_values
            else:
                # Normal query: single endpoint
                essential_values = ground_truth.get_values_by_level(RelevanceLevel.PREFERRED, endpoint, timeout=60.0)
                preferred_values = essential_values  # Same as essential_values
                acceptable_values = ground_truth.get_values_by_level(RelevanceLevel.ACCEPTABLE, endpoint, timeout=60.0)
        except Exception as e:
            print(f"  Error getting values by level: {e}")
            essential_values, preferred_values, acceptable_values = set(), set(), set()

    # Detect active levels
    (
        essential_active,
        preferred_active,
        acceptable_active,
        essential_matches,
        preferred_matches,
        acceptable_matches,
    ) = detect_active_levels(gen_values, essential_values, preferred_values, acceptable_values)

    # Build column values for active column detection
    num_cols = len(gt_col_names)
    column_values: list[set[str]] = [set() for _ in range(num_cols)]
    for gt_tuple in gt_tuples:
        for col_idx in range(min(len(gt_tuple), num_cols)):
            val = gt_tuple[col_idx]
            if val and val.strip():
                column_values[col_idx].add(val)

    # Get column metadata (use query variant 0 for display)
    gt_cols = ground_truth.get_columns_for_query(0)
    gt_columns_full = [
        {"var_name": col.var_name, "level": col.level.name, "description": col.description}
        for col in gt_cols
    ]

    # Calculate adaptive metrics directly from cached values (fast path)
    # This avoids re-iterating through all tuples which is slow for LARGE queries
    if cached_gt and cached_gt.get("success"):
        # Use cached column_values if available (much faster for LARGE queries)
        if "column_values" in cached_gt:
            column_values = [set(cv) for cv in cached_gt["column_values"]]
        else:
            # Fallback: Build column_values from cached tuples
            num_cols = len(gt_col_names)
            column_values: list[set[str]] = [set() for _ in range(num_cols)]
            for gt_tuple in gt_tuples:
                for col_idx in range(min(len(gt_tuple), num_cols)):
                    val = gt_tuple[col_idx]
                    if val and val.strip():
                        column_values[col_idx].add(val)

        # Get measurement column indices
        measurement_columns = ground_truth.get_measurement_column_indices()

        # Calculate metrics directly (no endpoint calls needed)
        try:
            # Get GT SPARQL and dataset for signature extraction
            gt_sparql = ground_truth.sparql_queries[0] if ground_truth.sparql_queries else None
            dataset_id = ground_truth.dataset if hasattr(ground_truth, "dataset") else None

            metrics = calculate_adaptive_metrics(
                llm_results=generated_results,
                gt_tuples=gt_tuples,
                columns_with_nulls=columns_with_nulls,
                essential_values=essential_values,
                preferred_values=preferred_values,
                acceptable_values=acceptable_values,
                column_values=column_values,
                measurement_columns=measurement_columns,
                gt_sparql=gt_sparql,
                llm_sparql=llm_sparql,
                gt_col_names=gt_col_names,
                dataset_id=dataset_id,
            )
            metrics.llm_columns = [k for k in (generated_results[0].keys() if generated_results else []) if not k.startswith("_")]

            # Calculate schema metrics (same as in _evaluate_single_query)
            llm_col_vals = extract_llm_column_values(generated_results)
            gt_cols_for_schema = ground_truth.get_columns_for_query(0)
            gt_col_levels = [col.level for col in gt_cols_for_schema]
            gt_col_is_measurement = [col.is_measurement for col in gt_cols_for_schema]
            gt_semantic_concepts = [col.semantic_concept for col in gt_cols_for_schema]

            (
                metrics.schema_recall, metrics.schema_precision,
                metrics.schema_expected_count, metrics.schema_matched_count,
                metrics.schema_llm_columns, metrics.schema_llm_matched,
            ) = calculate_schema_metrics(
                llm_col_vals, column_values, gt_col_levels, gt_col_is_measurement, gt_semantic_concepts
            )

            if metrics.schema_recall + metrics.schema_precision > 0:
                metrics.schema_f1 = (2 * metrics.schema_recall * metrics.schema_precision /
                                    (metrics.schema_recall + metrics.schema_precision))
        except Exception as e:
            print(f"  Error in cached adaptive evaluation: {e}")
            metrics = AdaptiveMetricsResult()
    else:
        # Fallback: Use full evaluation with endpoint calls
        preload_gt_internal_cache(ground_truth, cached_gt) if cached_gt else None
        try:
            metrics = evaluate_adaptive_ground_truth(
                llm_results=generated_results,
                ground_truth=ground_truth,
                endpoint=endpoint,
                timeout=60.0,
                llm_sparql=llm_sparql,
            )
        except Exception as e:
            print(f"  Error in adaptive evaluation: {e}")
            metrics = AdaptiveMetricsResult()

    # --- Build dashboard display data from column mapping ---
    col_mapping = metrics.column_mapping  # list of (gt_col_idx, llm_col_name)
    matched_gt_indices = [gt_idx for gt_idx, _ in col_mapping]
    matched_llm_names = set(llm_name for _, llm_name in col_mapping)

    # Build active columns info with level
    col_name_to_level = {col.var_name: col.level.name for col in gt_cols}
    active_columns_info = [
        {
            "var_name": gt_col_names[gt_idx],
            "original_index": gt_idx,
            "level": col_name_to_level.get(gt_col_names[gt_idx], "UNKNOWN"),
            "matched_llm_column": llm_name,
        }
        for gt_idx, llm_name in col_mapping
    ]

    # Project GT tuples to matched columns for display
    projected_col_names = [gt_col_names[i] for i in matched_gt_indices] if matched_gt_indices else gt_col_names
    projected_tuples: set[tuple[str, ...]] = set()
    if matched_gt_indices:
        for gt_tuple in gt_tuples:
            projected = tuple(gt_tuple[i] for i in matched_gt_indices if i < len(gt_tuple))
            if projected and any(v.strip() for v in projected if v):
                projected_tuples.add(projected)
    else:
        projected_tuples = gt_tuples

    # Build LLM projected tuples (same column order as GT projection) for TP detection
    llm_projected_tuples: set[tuple[str, ...]] = set()
    matched_llm_names_ordered = [llm_name for _, llm_name in col_mapping]
    if matched_llm_names_ordered and generated_results:
        for binding in generated_results:
            row = tuple(
                normalize_value(binding.get(col, "")) or ""
                for col in matched_llm_names_ordered
            )
            if any(v for v in row):
                llm_projected_tuples.add(row)

    # Compute TP tuples (intersection of projected GT and LLM tuples)
    tp_tuples = projected_tuples & llm_projected_tuples

    # GT sample rows (ALL columns) with TP marking
    sorted_gt_tuples = sorted(gt_tuples, key=lambda t: str(t[0]).lower() if t else "")
    gt_sample_rows: list[dict] = []
    for tup in sorted_gt_tuples[:50]:
        row: dict = {}
        for i, col_name in enumerate(gt_col_names):
            if i < len(tup):
                row[col_name] = tup[i] if tup[i] else ""
        # Check if this row's projection is a TP
        if matched_gt_indices:
            projected = tuple(tup[i] for i in matched_gt_indices if i < len(tup))
            row["_is_tp"] = projected in tp_tuples
        else:
            row["_is_tp"] = False
        if row:
            gt_sample_rows.append(row)

    # Extract LLM sample rows (ALL columns) with TP marking
    llm_sample_rows = []
    llm_columns = []
    if generated_results:
        for binding in generated_results[:1]:
            llm_columns = [k for k in binding.keys() if not k.startswith("_")]
        first_col = llm_columns[0] if llm_columns else None
        if first_col:
            def get_sort_val(b):
                val = b.get(first_col, "")
                if isinstance(val, dict):
                    val = val.get("value", "")
                return str(val).lower()
            sorted_results = sorted(generated_results, key=get_sort_val)
        else:
            sorted_results = generated_results
        for binding in sorted_results[:50]:
            row_dict = extract_row_dict(binding)
            # Check if this row's projection is a TP
            if matched_llm_names_ordered:
                projected = tuple(
                    normalize_value(binding.get(col, "")) or ""
                    for col in matched_llm_names_ordered
                )
                row_dict["_is_tp"] = projected in tp_tuples
            else:
                row_dict["_is_tp"] = False
            llm_sample_rows.append(row_dict)

    return {
        "evaluation_mode": "adaptive",
        "adaptive_metrics": metrics.to_dict(),
        # Level activity
        "essential_active": metrics.essential_active,
        "preferred_active": metrics.preferred_active,
        "acceptable_active": metrics.acceptable_active,
        "essential_matches": metrics.essential_matches,
        "preferred_matches": metrics.preferred_matches,
        "acceptable_matches": metrics.acceptable_matches,
        # Variant info
        "best_recall_variant": metrics.best_recall_variant,
        "best_precision_variant": metrics.best_precision_variant,
        "variants_evaluated": metrics.variants_evaluated,
        # GT SPARQL query (for comparison with LLM generated)
        "gt_sparql": ground_truth.sparql_queries[0] if ground_truth.sparql_queries else "",
        # GT column definitions (original, all columns)
        "gt_columns": gt_columns_full,
        "gt_original_columns": gt_col_names,
        # Column mapping (GT col idx → LLM col name)
        "column_mapping": [
            {"gt_col_idx": gt_idx, "gt_col_name": gt_col_names[gt_idx], "llm_col_name": llm_name}
            for gt_idx, llm_name in col_mapping
        ],
        "active_columns": active_columns_info,
        "projected_columns": projected_col_names,
        # GT sample rows (ALL columns, with _is_tp flag)
        "gt_sample_rows": gt_sample_rows,
        "gt_total_rows": len(gt_tuples),
        # LLM generated results (ALL columns, with _is_tp flag)
        "generated_sample_rows": llm_sample_rows,
        "generated_columns": llm_columns,
        "generated_total_rows": len(generated_results),
    }


def calc_tiered_summary(metrics_list: list[dict]) -> dict:
    """Calculate summary statistics for tiered evaluation."""
    if not metrics_list:
        return {}
    n = len(metrics_list)
    return {
        "count": n,
        "preferred_recall_mean": sum(m.get("preferred_recall", 0) for m in metrics_list) / n,
        "preferred_precision_mean": sum(m.get("preferred_precision", 0) for m in metrics_list) / n,
        "preferred_f1_mean": sum(m.get("preferred_f1", 0) for m in metrics_list) / n,
        "essential_recall_mean": sum(m.get("essential_recall", 0) for m in metrics_list) / n,
        "essential_precision_mean": sum(m.get("essential_precision", 0) for m in metrics_list) / n,
        "essential_f1_mean": sum(m.get("essential_f1", 0) for m in metrics_list) / n,
        "acceptable_recall_mean": sum(m.get("acceptable_recall", 0) for m in metrics_list) / n,
        "acceptable_precision_mean": sum(m.get("acceptable_precision", 0) for m in metrics_list) / n,
        "acceptable_f1_mean": sum(m.get("acceptable_f1", 0) for m in metrics_list) / n,
        "recall_mean": sum(m.get("recall", 0) for m in metrics_list) / n,
        "precision_mean": sum(m.get("precision", 0) for m in metrics_list) / n,
        "f1_score_mean": sum(m.get("f1_score", 0) for m in metrics_list) / n,
        "tuple_recall_mean": sum(m.get("tuple_recall", 0) for m in metrics_list) / n,
        "tuple_precision_mean": sum(m.get("tuple_precision", 0) for m in metrics_list) / n,
        "tuple_f1_score_mean": sum(m.get("tuple_f1_score", 0) for m in metrics_list) / n,
        "noise_count_mean": sum(m.get("noise_count", 0) for m in metrics_list) / n,
        # Schema Metrics (SELECT clause evaluation)
        "schema_recall_mean": sum(m.get("schema_recall", 0) for m in metrics_list) / n,
        "schema_precision_mean": sum(m.get("schema_precision", 0) for m in metrics_list) / n,
        "schema_f1_mean": sum(m.get("schema_f1", 0) for m in metrics_list) / n,
    }


def calc_adaptive_summary(metrics_list: list[dict]) -> dict:
    """Calculate summary statistics for adaptive evaluation."""
    if not metrics_list:
        return {}
    n = len(metrics_list)
    return {
        "count": n,
        "best_recall_mean": sum(m.get("best_recall", 0) for m in metrics_list) / n,
        "best_precision_mean": sum(m.get("best_precision", 0) for m in metrics_list) / n,
        "best_f1_mean": sum(m.get("best_f1", 0) for m in metrics_list) / n,
        "essential_active_rate": sum(1 for m in metrics_list if m.get("essential_active")) / n,
        "preferred_active_rate": sum(1 for m in metrics_list if m.get("preferred_active")) / n,
        "acceptable_active_rate": sum(1 for m in metrics_list if m.get("acceptable_active")) / n,
        "essential_matches_mean": sum(m.get("essential_matches", 0) for m in metrics_list) / n,
        "preferred_matches_mean": sum(m.get("preferred_matches", 0) for m in metrics_list) / n,
        "acceptable_matches_mean": sum(m.get("acceptable_matches", 0) for m in metrics_list) / n,
        "variants_evaluated_mean": sum(m.get("variants_evaluated", 0) for m in metrics_list) / n,
        "llm_row_count_mean": sum(m.get("llm_row_count", 0) for m in metrics_list) / n,
        # Schema Metrics (SELECT clause evaluation)
        "schema_recall_mean": sum(m.get("schema_recall", 0) for m in metrics_list) / n,
        "schema_precision_mean": sum(m.get("schema_precision", 0) for m in metrics_list) / n,
        "schema_f1_mean": sum(m.get("schema_f1", 0) for m in metrics_list) / n,
        # Retrieval Metrics (schema-aware)
        "retrieval_path_coherence_mean": sum(m.get("retrieval_path_coherence", 0) for m in metrics_list) / n,
        "retrieval_triple_f1_mean": sum(m.get("retrieval_triple_f1", 0) for m in metrics_list) / n,
        "retrieval_triple_precision_mean": sum(m.get("retrieval_triple_precision", 0) for m in metrics_list) / n,
        "retrieval_triple_recall_mean": sum(m.get("retrieval_triple_recall", 0) for m in metrics_list) / n,
    }


def print_tiered_summary(results: dict):
    """Print tiered evaluation summary."""
    print("\n" + "=" * 60)
    print("TIERED EVALUATION SUMMARY")
    print("=" * 60)

    overall = results["summary"]["overall"]
    print(f"\nOverall ({overall.get('count', 0)} traces):")
    print(f"  {'Level':<25} {'Recall':>10} {'Precision':>10} {'F1':>10}")
    print(f"  {'-'*55}")
    print(f"  {'L1 PREFERRED':<25} {overall.get('preferred_recall_mean', 0):>10.2%} {overall.get('preferred_precision_mean', 0):>10.2%} {overall.get('preferred_f1_mean', 0):>10.2%}")
    print(f"  {'L2 PREFERRED':<25} {overall.get('essential_recall_mean', 0):>10.2%} {overall.get('essential_precision_mean', 0):>10.2%} {overall.get('essential_f1_mean', 0):>10.2%}")
    print(f"  {'L3 ACCEPTABLE':<25} {overall.get('acceptable_recall_mean', 0):>10.2%} {overall.get('acceptable_precision_mean', 0):>10.2%} {overall.get('acceptable_f1_mean', 0):>10.2%}")
    print(f"  {'-'*55}")
    print(f"  {'Combined (L1+L2)':<25} {overall.get('recall_mean', 0):>10.2%} {overall.get('precision_mean', 0):>10.2%} {overall.get('f1_score_mean', 0):>10.2%}")
    print(f"\n  Tuple F1: {overall.get('tuple_f1_score_mean', 0):.2%}, Noise: {overall.get('noise_count_mean', 0):.1f}")
    print(f"  Schema Metrics: Recall={overall.get('schema_recall_mean', 0):.2%}, Prec={overall.get('schema_precision_mean', 0):.2%}, F1={overall.get('schema_f1_mean', 0):.2%}")

    print("\nBy Approach:")
    for approach, summary in results["summary"]["by_approach"].items():
        label = approach.replace("agentic_", "").upper()
        print(f"  {label}: Results F1={summary.get('f1_score_mean', 0):.0%}, Schema F1={summary.get('schema_f1_mean', 0):.0%}")


def print_adaptive_summary(results: dict):
    """Print adaptive evaluation summary."""
    print("\n" + "=" * 60)
    print("ADAPTIVE EVALUATION SUMMARY")
    print("=" * 60)

    overall = results["summary"]["overall"]
    print(f"\nOverall ({overall.get('count', 0)} traces):")
    print(f"  Results Metrics (WHERE clause):")
    print(f"    Recall:    {overall.get('best_recall_mean', 0):.2%}")
    print(f"    Precision: {overall.get('best_precision_mean', 0):.2%}")
    print(f"    F1:        {overall.get('best_f1_mean', 0):.2%}")
    print(f"  Schema Metrics (SELECT clause):")
    print(f"    Recall:    {overall.get('schema_recall_mean', 0):.2%}")
    print(f"    Precision: {overall.get('schema_precision_mean', 0):.2%}")
    print(f"    F1:        {overall.get('schema_f1_mean', 0):.2%}")
    print(f"  Retrieval Metrics (schema-aware):")
    print(f"    PathCoherence: {overall.get('retrieval_path_coherence_mean', 0):.2%}")
    print(f"    Triple:  P={overall.get('retrieval_triple_precision_mean', 0):.2%}  R={overall.get('retrieval_triple_recall_mean', 0):.2%}  F1={overall.get('retrieval_triple_f1_mean', 0):.2%}")
    print(f"\n  PREFERRED active: {overall.get('essential_active_rate', 0):.0%}")
    print(f"  PREFERRED active: {overall.get('preferred_active_rate', 0):.0%}")
    print(f"  Avg variants:     {overall.get('variants_evaluated_mean', 0):.1f}")

    print("\nBy Approach:")
    for approach, summary in results["summary"]["by_approach"].items():
        label = approach.replace("agentic_", "").upper()
        print(f"  {label}:")
        print(f"    Results:   Recall={summary.get('best_recall_mean', 0):.0%}, Prec={summary.get('best_precision_mean', 0):.0%}, F1={summary.get('best_f1_mean', 0):.0%}")
        print(f"    Schema:    Recall={summary.get('schema_recall_mean', 0):.0%}, Prec={summary.get('schema_precision_mean', 0):.0%}, F1={summary.get('schema_f1_mean', 0):.0%}")
        print(f"    Retrieval: PathCoh={summary.get('retrieval_path_coherence_mean', 0):.0%}, TripF1={summary.get('retrieval_triple_f1_mean', 0):.0%}")


def reevaluate_experiment(experiment_name: str, use_adaptive: bool = False) -> dict:
    """Re-evaluate all traces in an experiment."""
    mode = "adaptive" if use_adaptive else "tiered"
    print(f"\nRe-evaluating experiment: {experiment_name} (mode: {mode})")

    # Check GT cache availability
    cache = load_gt_cache()
    if cache and cache.get("queries"):
        cache_queries = len(cache.get("queries", {}))
        cache_date = cache.get("extracted_at", "unknown")[:10]
        print(f"Using GT cache: {cache_queries} queries (extracted: {cache_date})")
    else:
        print("No GT cache available, using live endpoint queries")
        print("  Hint: Run 'python scripts/extract_ground_truth_cache.py' to create cache")

    experiment_dir = RESULTS_DIR / "experiments" / experiment_name
    traces_dir = experiment_dir / "traces"

    if not traces_dir.exists():
        print(f"Error: Traces directory not found: {traces_dir}")
        sys.exit(1)

    trace_files = list(traces_dir.glob("*.json"))
    print(f"Found {len(trace_files)} traces to evaluate")

    results = {
        "experiment_name": experiment_name,
        "evaluation_mode": mode,
        "reevaluation_timestamp": datetime.now().isoformat(),
        "total_traces": len(trace_files),
        "traces": {},
        "summary": {"by_approach": {}, "by_query_set": {}, "overall": {}},
    }

    all_metrics = []
    metrics_by_approach = {}
    metrics_by_query_set = {}
    skipped = 0
    skip_reasons: dict[str, list[str]] = {
        "load_failed": [],
        "no_query_id": [],
        "no_ground_truth": [],
        "no_endpoint": [],
        "eval_error": [],
    }

    for trace_file in tqdm(trace_files, desc=f"Re-evaluating ({mode})"):
        trace = load_trace(trace_file)
        if not trace:
            skipped += 1
            skip_reasons["load_failed"].append(trace_file.name)
            continue

        query_id = trace.get("query_metadata", {}).get("query_id") or trace.get("ground_truth_query_id")
        if not query_id:
            skipped += 1
            skip_reasons["no_query_id"].append(trace_file.name)
            continue

        # Always use adaptive ground truth (legacy tiered format deprecated)
        ground_truth = get_experimental_query(query_id)
        if not ground_truth:
            skipped += 1
            skip_reasons["no_ground_truth"].append(f"{trace_file.name} (query_id={query_id})")
            continue

        endpoint = ENDPOINTS.get(ground_truth.dataset.upper()) or ENDPOINTS.get(ground_truth.dataset)
        if not endpoint:
            skipped += 1
            skip_reasons["no_endpoint"].append(f"{trace_file.name} (dataset={ground_truth.dataset})")
            continue

        try:
            # Always use adaptive evaluation (legacy tiered format deprecated)
            eval_result = evaluate_trace_adaptive(trace, ground_truth, endpoint)
        except Exception as e:
            print(f"Error evaluating {trace_file.name}: {e}")
            skipped += 1
            skip_reasons["eval_error"].append(f"{trace_file.name}: {e}")
            continue

        # Retrieval evaluation (schema-aware metrics)
        retrieval_metrics_dict = None
        try:
            schema_gt = build_schema_gt_for_query(ground_truth)
            if schema_gt.triples:
                turtle_strings = trace.get("all_retrieved_triples", [])
                dataset_ids = ground_truth.datasets or [ground_truth.dataset]
                schema = get_combined_schema(dataset_ids)
                retrieval_result = calculate_retrieval_metrics(
                    turtle_strings, schema, schema_gt,
                )
                retrieval_metrics_dict = retrieval_result.to_dict()
        except Exception as e:
            logger.debug(f"Retrieval evaluation failed for {trace_file.name}: {e}")

        trace_key = trace_file.stem
        approach = trace.get("approach", "unknown")
        query_set = ground_truth.query_set if ground_truth else (query_id[0] if query_id else "X")

        # Derive baseline_id for paired grouping (SYN01->BASE01, TYPO01->BASE01, LARGE01->BASE01)
        baseline_id = None
        if query_set in ("SYN", "TYPO", "LARGE") and query_id:
            num_part = query_id[len(query_set):]  # "SYN01" -> "01"
            baseline_id = f"BASE{num_part}"

        trace_entry = {
            "query_id": query_id,
            "approach": approach,
            "query_set": query_set,
            "baseline_id": baseline_id,
            "dataset": ground_truth.dataset,
            "query_text": trace.get("query_metadata", {}).get("query_text", ""),
            "final_sparql": trace.get("final_sparql", ""),
            "final_sql": trace.get("final_sql", ""),
            "success": trace.get("success", False),
            **eval_result,
        }
        if retrieval_metrics_dict is not None:
            trace_entry["retrieval_metrics"] = retrieval_metrics_dict

        results["traces"][trace_key] = trace_entry

        metrics = eval_result.get("adaptive_metrics", {})
        if retrieval_metrics_dict is not None:
            metrics["retrieval_path_coherence"] = retrieval_metrics_dict.get("schema_path_coherence", 0)
            metrics["retrieval_triple_f1"] = retrieval_metrics_dict.get("schema_triple_f1", 0)
            metrics["retrieval_triple_precision"] = retrieval_metrics_dict.get("schema_triple_precision", 0)
            metrics["retrieval_triple_recall"] = retrieval_metrics_dict.get("schema_triple_recall", 0)
        all_metrics.append(metrics)

        if approach not in metrics_by_approach:
            metrics_by_approach[approach] = []
        metrics_by_approach[approach].append(metrics)

        if query_set not in metrics_by_query_set:
            metrics_by_query_set[query_set] = []
        metrics_by_query_set[query_set].append(metrics)

    print(f"Evaluated {len(all_metrics)} traces, skipped {skipped}")

    # Print skip reasons if any
    if skipped > 0:
        print("\nSkip reasons:")
        for reason, files in skip_reasons.items():
            if files:
                print(f"  {reason}: {len(files)}")
                for f in files[:5]:  # Show first 5
                    print(f"    - {f}")
                if len(files) > 5:
                    print(f"    ... and {len(files) - 5} more")

    # Always use adaptive summary (legacy tiered format deprecated)
    results["summary"]["overall"] = calc_adaptive_summary(all_metrics)
    for approach, mlist in metrics_by_approach.items():
        results["summary"]["by_approach"][approach] = calc_adaptive_summary(mlist)
    for qset, mlist in metrics_by_query_set.items():
        results["summary"]["by_query_set"][qset] = calc_adaptive_summary(mlist)

    output_file = experiment_dir / "adaptive_evaluation.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    print_adaptive_summary(results)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate experiment traces against tiered/adaptive ground truth"
    )
    parser.add_argument("experiment", help="Name of the experiment to re-evaluate")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive evaluation")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable GT cache, use live endpoint queries (requires running containers)"
    )

    args = parser.parse_args()

    # Disable cache if requested
    if args.no_cache:
        global _gt_cache
        _gt_cache = {}  # Empty dict = no cache hits
        print("GT cache disabled, using live endpoint queries")

    reevaluate_experiment(args.experiment, use_adaptive=args.adaptive)


if __name__ == "__main__":
    main()
