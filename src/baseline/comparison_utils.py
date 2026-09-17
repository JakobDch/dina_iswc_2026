import collections
import decimal
import json 
import logging

logger = logging.getLogger(__name__)

def normalize_value(value_dict):
    """
    Normalizes a SPARQL binding value for comparison, IGNORING datatypes
    to focus purely on the string representation of the value.

    For numeric values (integers, decimals, floats), normalizes by:
    - Converting to Decimal for precision
    - Removing trailing zeros and unnecessary decimal points
    - Converting to lowercase string

    For non-numeric values, converts to lowercase string.
    """
    if not value_dict or 'value' not in value_dict:
        return None

    value_str = str(value_dict['value'])

    # Try to normalize as a numeric value
    try:
        # Parse as Decimal to handle arbitrary precision
        numeric_value = decimal.Decimal(value_str)

        # Normalize by removing trailing zeros and unnecessary decimal point
        # e.g., "376.500000" -> "376.5", "100.00" -> "100"
        normalized = numeric_value.normalize()

        # Convert to string using 'f' format to avoid scientific notation
        # Check if normalized uses scientific notation (contains 'E' or 'e')
        normalized_str = str(normalized)
        if 'E' in normalized_str or 'e' in normalized_str:
            # Use fixed-point notation to avoid scientific notation
            # This preserves the exact value without exponent
            normalized_str = format(numeric_value, 'f')
            # Remove trailing zeros after decimal point
            if '.' in normalized_str:
                normalized_str = normalized_str.rstrip('0').rstrip('.')

        # Convert back to string (lowercase for consistency)
        return normalized_str.lower()
    except (decimal.InvalidOperation, ValueError, decimal.DecimalException):
        # Not a numeric value, treat as string
        pass

    # Treat all non-numeric values as lowercased strings for robust comparison
    return value_str.lower()

def canonicalize_results_value_multiset(sparql_results_json):
    """Converts SPARQL JSON results to a canonical multiset of value frozensets for comparison."""
    if not isinstance(sparql_results_json, dict):
         logger.error(f"Invalid SPARQL result format: Expected dict, got {type(sparql_results_json)}")
         return None

    
    if "boolean" in sparql_results_json:
        try:
            normalized_bool_str = str(sparql_results_json["boolean"]).lower()
            return collections.Counter({frozenset({normalized_bool_str})})
        except Exception as e:
            logger.error(f"Error processing boolean SPARQL result: {e}", exc_info=True)
            return None
    

    results_part = sparql_results_json.get("results")
    if not isinstance(results_part, dict):
        if sparql_results_json.get("head", {}).get("vars") is not None:
             logger.debug("'results' part missing but 'head.vars' exists. Treating as empty result set.")
             return collections.Counter() 
        logger.error("Invalid SPARQL result format: 'results' part is missing or not a dictionary.")
        return None

    bindings = results_part.get("bindings")
    if not isinstance(bindings, list):
        if sparql_results_json.get("head", {}).get("vars") is not None:
             logger.debug("'bindings' part missing but 'head.vars' exists. Treating as empty result set.")
             return collections.Counter()
        logger.error("Invalid SPARQL result format: 'bindings' part is missing or not a list.")
        return None

    head_part = sparql_results_json.get("head")
    if not isinstance(head_part, dict) or not isinstance(head_part.get("vars"), list):
         if bindings:
              logger.error("Invalid SPARQL result format: 'head' or 'vars' missing/invalid despite bindings being present.")
              return None
         else:
              logger.debug("No bindings and no valid 'head.vars'. Treating as empty or non-standard result set.")
              return collections.Counter()

    list_of_row_value_frozensets = []
    try:
        for binding in bindings:
            if not isinstance(binding, dict):
                logger.warning(f"Skipping invalid binding (not a dict): {binding}")
                continue
            
            values_in_row_set = set()
            
            for var_name, var_value_dict in binding.items():
                normalized_val = normalize_value(var_value_dict)
                if normalized_val is not None:
                    values_in_row_set.add(normalized_val)
                else:
                    logger.warning(f"Normalization returned None for variable '{var_name}' with data {var_value_dict} in a binding.")

            list_of_row_value_frozensets.append(frozenset(values_in_row_set))

    except Exception as e:
        logger.error(f"Error processing bindings in canonicalize_results_value_multiset: {e}", exc_info=True)
        return None

    return collections.Counter(list_of_row_value_frozensets)


def calculate_variable_level_metrics(actual_results_json: dict, expected_results_json: dict, request_id: str = "metrics") -> dict:
    """
    Calculate precision and recall at the variable level (per variable, not per binding).

    This enables finer-grained evaluation: if 3 out of 4 variables in a binding
    are correct, those 3 count as true positives.
    """
    log_prefix = f"[{request_id}]"

    actual_bindings = actual_results_json.get("results", {}).get("bindings", [])
    expected_bindings = expected_results_json.get("results", {}).get("bindings", [])

    if not actual_bindings and not expected_bindings:
        logger.debug(f"{log_prefix} Both result sets are empty. Variable-level metrics: 1.0")
        return {
            "variable_precision": 1.0,
            "variable_recall": 1.0,
            "variable_f1": 1.0,
            "total_variables_actual": 0,
            "total_variables_expected": 0,
            "matched_variables": 0
        }

    def normalize_binding(binding: dict) -> dict:
        """Normalize a binding to {var_name: normalized_value}."""
        normalized = {}
        for var_name, var_value_dict in binding.items():
            norm_val = normalize_value(var_value_dict)
            if norm_val is not None:
                normalized[var_name] = norm_val
        return normalized

    norm_actual_bindings = [normalize_binding(b) for b in actual_bindings]
    norm_expected_bindings = [normalize_binding(b) for b in expected_bindings]

    # Greedy matching: find best-matching expected binding for each actual binding
    # Compare VALUES only (ignore variable names) for consistency with binding-level metrics
    matched_expected_indices = set()
    total_matched_variables = 0
    total_actual_variables = 0
    total_expected_variables = 0

    for exp_binding in norm_expected_bindings:
        total_expected_variables += len(exp_binding)

    for act_binding in norm_actual_bindings:
        total_actual_variables += len(act_binding)

        if not norm_expected_bindings:
            continue

        act_values = set(act_binding.values())

        best_match_idx = None
        best_match_count = 0

        for idx, exp_binding in enumerate(norm_expected_bindings):
            if idx in matched_expected_indices:
                continue

            exp_values = set(exp_binding.values())
            overlap = act_values & exp_values
            match_count = len(overlap)

            if match_count > best_match_count:
                best_match_count = match_count
                best_match_idx = idx

        if best_match_idx is not None:
            matched_expected_indices.add(best_match_idx)
            total_matched_variables += best_match_count

    tp_count = total_matched_variables
    fp_count = total_actual_variables - tp_count
    fn_count = total_expected_variables - tp_count

    if total_actual_variables > 0:
        precision = tp_count / total_actual_variables
    else:
        precision = 1.0 if total_expected_variables == 0 else 0.0

    if total_expected_variables > 0:
        recall = tp_count / total_expected_variables
    else:
        recall = 1.0

    # F1 Score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    logger.info(f"{log_prefix} Variable-level metrics: "
                f"Precision={precision:.3f} ({tp_count}/{total_actual_variables}), "
                f"Recall={recall:.3f} ({tp_count}/{total_expected_variables}), "
                f"F1={f1_score:.3f}")
    logger.debug(f"{log_prefix} Variable-level details: TP={tp_count}, FP={fp_count}, FN={fn_count}")

    return {
        "variable_precision": round(precision, 4),
        "variable_recall": round(recall, 4),
        "variable_f1": round(f1_score, 4),
        "total_variables_actual": total_actual_variables,
        "total_variables_expected": total_expected_variables,
        "matched_variables": tp_count
    }
