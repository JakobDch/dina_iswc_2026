"""
SPARQL result metrics for experiment evaluation.

Calculates F1-Score, Execution Accuracy, Precision, and Recall by comparing
generated query results against ground-truth expected results.

Key approach: VALUE-BY-VALUE comparison. Variable names are IGNORED.
All individual values from all result bindings are compared as a multiset.
This gives partial credit when some values match even if the query structure differs.
"""

from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any


def normalize_value(value: Any) -> str:
    """
    Normalize a SPARQL binding value for comparison.

    - Extracts value from SPARQL JSON format {"type": "uri", "value": "..."}
    - Normalizes numeric values (removes trailing zeros, handles scientific notation)
    - Converts everything to lowercase for case-insensitive comparison
    """
    # Extract actual value from SPARQL JSON format
    if isinstance(value, dict):
        raw_value = str(value.get("value", ""))
    else:
        raw_value = str(value)

    raw_value = raw_value.strip()

    # Try to normalize as numeric value
    try:
        decimal_val = Decimal(raw_value)
        # Normalize removes trailing zeros
        normalized = decimal_val.normalize()

        # Convert to string, handling scientific notation
        str_val = str(normalized)
        if 'E' in str_val or 'e' in str_val:
            # Convert scientific notation to fixed-point
            str_val = format(normalized, 'f')
            # Remove trailing zeros after decimal point
            if '.' in str_val:
                str_val = str_val.rstrip('0').rstrip('.')

        return str_val.lower()
    except (InvalidOperation, ValueError):
        # Not a number, return as lowercase string
        return raw_value.lower()


def extract_bindings(results: dict | list | None) -> list[dict]:
    """
    Extract bindings list from various SPARQL result formats.

    Handles:
    - {"bindings": [...]} (standard SPARQL JSON)
    - {"results": {"bindings": [...]}} (nested format)
    - [binding1, binding2, ...] (list of bindings)
    """
    if results is None:
        return []

    if isinstance(results, list):
        return results

    if isinstance(results, dict):
        # Try nested format first
        if "results" in results and isinstance(results["results"], dict):
            return results["results"].get("bindings", [])
        if "bindings" in results:
            return results["bindings"]
        # Single result as dict
        return [results]

    return []


def extract_all_values(results: dict | list | None) -> Counter[str]:
    """
    Extract ALL individual values from ALL bindings as a multiset (Counter).

    This is the key function for value-by-value comparison:
    - Each binding contributes its individual values to the Counter
    - Variable names are completely ignored
    - Duplicate values are counted (multiset semantics)

    Example:
        Input: [{"s": "A", "p": "B"}, {"s": "A", "p": "C"}]
        Output: Counter({"a": 2, "b": 1, "c": 1})
    """
    bindings = extract_bindings(results)

    all_values: Counter[str] = Counter()
    for binding in bindings:
        if isinstance(binding, dict):
            for var, value in binding.items():
                # Skip internal/metadata fields
                if var.startswith("_"):
                    continue
                normalized = normalize_value(value)
                if normalized:  # Skip empty values
                    all_values[normalized] += 1

    return all_values


def calculate_sparql_metrics(
    generated_results: dict | list | None,
    expected_results: dict | list | None,
) -> dict[str, float]:
    """
    Calculate F1-Score, Precision, Recall, and Execution Accuracy.

    Uses VALUE-BY-VALUE comparison:
    - All individual values from all bindings are extracted
    - Variable names are completely ignored
    - Multiset comparison gives partial credit for overlapping values

    Example:
        Generated: [{student: S1, name: "Alice"}]  -> values: {s1, alice}
        Expected:  [{student: S1, course: C1}]     -> values: {s1, c1}
        Overlap: {s1} -> Precision=0.5, Recall=0.5, F1=0.5

    Args:
        generated_results: Results from the generated SPARQL query
        expected_results: Ground-truth expected results

    Returns:
        Dict with precision, recall, f1_score, and execution_accuracy
    """
    generated_values = extract_all_values(generated_results)
    expected_values = extract_all_values(expected_results)

    generated_count = sum(generated_values.values())
    expected_count = sum(expected_values.values())

    # Handle empty cases
    if expected_count == 0 and generated_count == 0:
        # Both empty = perfect match
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
            "execution_accuracy": 1.0,
            "generated_count": 0,
            "expected_count": 0,
            "true_positives": 0,
        }

    if expected_count == 0:
        # Expected is empty but we got results = all false positives
        return {
            "precision": 0.0,
            "recall": 1.0,  # Nothing to recall
            "f1_score": 0.0,
            "execution_accuracy": 0.0,
            "generated_count": generated_count,
            "expected_count": 0,
            "true_positives": 0,
        }

    if generated_count == 0:
        # We got nothing but expected results = all false negatives
        return {
            "precision": 0.0,  # Nothing predicted = no true positives possible
            "recall": 0.0,
            "f1_score": 0.0,
            "execution_accuracy": 0.0,
            "generated_count": 0,
            "expected_count": expected_count,
            "true_positives": 0,
        }

    # Calculate true positives using multiset intersection
    # Counter & gives minimum count for each element
    tp_counter = generated_values & expected_values
    tp_count = sum(tp_counter.values())

    # Precision: of all generated values, how many were in expected?
    precision = tp_count / generated_count

    # Recall: of all expected values, how many did we generate?
    recall = tp_count / expected_count

    # F1-Score: harmonic mean of precision and recall
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    # Execution Accuracy: exact match (1.0 if value multisets are identical)
    execution_accuracy = 1.0 if generated_values == expected_values else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "execution_accuracy": execution_accuracy,
        "generated_count": generated_count,
        "expected_count": expected_count,
        "true_positives": tp_count,
    }


def calculate_query_execution_success(
    generated_results: dict | list | None,
    success: bool,
) -> dict[str, Any]:
    """
    Calculate basic execution success metrics.

    Args:
        generated_results: Results from query execution
        success: Whether the query executed without errors

    Returns:
        Dict with execution_success and result_count
    """
    all_values = extract_all_values(generated_results)
    result_count = sum(all_values.values())

    return {
        "execution_success": success,
        "result_count": result_count,
        "has_results": result_count > 0,
    }


def extract_result_set(results: dict | list | None) -> set[str]:
    """
    Extract a set of unique values from SPARQL results.

    Returns a set (not Counter) of all unique normalized values.
    """
    all_values = extract_all_values(results)
    return set(all_values.keys())


__all__ = [
    "calculate_sparql_metrics",
    "calculate_query_execution_success",
    "extract_result_set",
    "extract_all_values",
    "normalize_value",
]