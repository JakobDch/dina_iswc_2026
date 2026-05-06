"""
Evaluation metrics for SPARQL result comparison.

Adapted from chat2sparql-main comparison_utils.py.
"""

import collections
import decimal
import logging
from typing import Any

logger = logging.getLogger(__name__)


def normalize_value(value_dict: dict[str, Any] | None) -> str | None:
    """
    Normalize a SPARQL binding value for comparison.

    For numeric values, normalizes by removing trailing zeros.
    For non-numeric values, converts to lowercase string.
    """
    if not value_dict or "value" not in value_dict:
        return None

    value_str = str(value_dict["value"])

    # Try to normalize as a numeric value
    try:
        numeric_value = decimal.Decimal(value_str)
        normalized = numeric_value.normalize()
        normalized_str = str(normalized)

        if "E" in normalized_str or "e" in normalized_str:
            normalized_str = format(numeric_value, "f")
            if "." in normalized_str:
                normalized_str = normalized_str.rstrip("0").rstrip(".")

        return normalized_str.lower()
    except (decimal.InvalidOperation, ValueError, decimal.DecimalException):
        pass

    return value_str.lower()


def canonicalize_results(sparql_results_json: dict) -> collections.Counter | None:
    """Convert SPARQL JSON results to a canonical multiset for comparison."""
    if not isinstance(sparql_results_json, dict):
        logger.error(f"Invalid SPARQL result format: Expected dict, got {type(sparql_results_json)}")
        return None

    # Handle ASK queries (boolean results)
    if "boolean" in sparql_results_json:
        try:
            normalized_bool_str = str(sparql_results_json["boolean"]).lower()
            return collections.Counter({frozenset({normalized_bool_str})})
        except Exception as e:
            logger.error(f"Error processing boolean SPARQL result: {e}")
            return None

    results_part = sparql_results_json.get("results")
    if not isinstance(results_part, dict):
        if sparql_results_json.get("head", {}).get("vars") is not None:
            return collections.Counter()
        logger.error("Invalid SPARQL result format: 'results' part missing")
        return None

    bindings = results_part.get("bindings")
    if not isinstance(bindings, list):
        if sparql_results_json.get("head", {}).get("vars") is not None:
            return collections.Counter()
        logger.error("Invalid SPARQL result format: 'bindings' part missing")
        return None

    list_of_row_value_frozensets = []
    try:
        for binding in bindings:
            if not isinstance(binding, dict):
                continue

            values_in_row_set = set()
            for var_name, var_value_dict in binding.items():
                normalized_val = normalize_value(var_value_dict)
                if normalized_val is not None:
                    values_in_row_set.add(normalized_val)

            list_of_row_value_frozensets.append(frozenset(values_in_row_set))

    except Exception as e:
        logger.error(f"Error processing bindings: {e}")
        return None

    return collections.Counter(list_of_row_value_frozensets)


def calculate_result_metrics(
    actual_results: dict,
    expected_results: dict,
) -> dict[str, float]:
    """
    Calculate precision, recall, F1, and execution accuracy.

    Returns dict with execution_accuracy, precision, recall, f1_score.
    """
    actual_multiset = canonicalize_results(actual_results)
    expected_multiset = canonicalize_results(expected_results)

    if actual_multiset is None or expected_multiset is None:
        return {
            "execution_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

    # Execution accuracy: exact match
    execution_accuracy = 1.0 if actual_multiset == expected_multiset else 0.0

    # Calculate precision and recall
    if len(actual_multiset) == 0 and len(expected_multiset) == 0:
        return {
            "execution_accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
        }

    # True positives: intersection
    intersection = actual_multiset & expected_multiset
    tp = sum(intersection.values())

    # False positives: in actual but not expected
    fp = sum((actual_multiset - expected_multiset).values())

    # False negatives: in expected but not actual
    fn = sum((expected_multiset - actual_multiset).values())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "execution_accuracy": execution_accuracy,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
    }


# Re-export calculate_edit_cost_score from sparql_analysis for convenience
from .sparql_analysis import calculate_edit_cost_score, calculate_diff_cost, calculate_max_edit_cost


def calculate_edit_cost(
    generated_query: str,
    corrected_query: str,
    normalize_variables: bool = True,
    request_id: str = "",
) -> dict[str, Any]:
    """
    Calculate the edit cost between a generated and corrected SPARQL query.

    Uses algebraic SPARQL analysis with Hungarian algorithm for optimal
    triple matching. Returns a normalized score where 100 = identical.

    Args:
        generated_query: The generated SPARQL query
        corrected_query: The corrected/ground truth SPARQL query
        normalize_variables: Whether to normalize variable names (?x -> ?v1)
        request_id: Optional request ID for logging

    Returns:
        Dictionary with:
            - actual_cost: The actual edit cost
            - max_cost: The maximum possible edit cost
            - normalized_score: Score from 0-100 (100 = identical)
            - edit_details: List of individual edits
    """
    return calculate_edit_cost_score(
        generated_query=generated_query,
        corrected_query=corrected_query,
        normalize_variables=normalize_variables,
        request_id=request_id,
    )
