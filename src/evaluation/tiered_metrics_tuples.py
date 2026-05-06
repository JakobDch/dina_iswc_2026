"""
Tiered relevance metrics for VKGQA evaluation - WITH TUPLE SUPPORT.

This version extends the original tiered_metrics.py with tuple-based matching.
When values belong together (e.g., "DeptA" with "50 students"), tuple matching
ensures that values are checked IN THE SAME ROW, not just anywhere in results.

TUPLE MATCHING (Subset-Tuple-Matching):
- Groundtruth tuple (DeptA, 50) matches if generated row contains BOTH values
- Extra columns in generated row are tolerated (ACCEPTABLE or Noise)
- Example: GT tuple ("DeptA", "50") matches generated ("DeptA", "50", "Prof X")

THREE-TIER evaluation (user-centric, each level measured separately):

Level 1 - PREFERRED (most important for users):
    Human-readable names/labels - what users actually want to see
    -> preferred_recall: How many names were found?

Level 2 - PREFERRED (technically correct):
    Entity URIs - correct but less user-friendly
    -> essential_recall: How many URIs were found?

Level 3 - ACCEPTABLE (bonus information):
    Related attributes - not directly asked for but useful
    -> acceptable_recall: How much bonus info was included?

Combined Metrics:
- recall: Overall recall across PREFERRED (direct answers)
- precision: Of generated values, how many are valid (any level)?
- f1_score: Harmonic mean of recall and precision
- noise_count: Irrelevant values returned

NEW Tuple Metrics:
- tuple_recall: How many expected tuples were found as complete rows?
- tuple_precision: Of generated rows, how many match expected tuples?

================================================================================
ADAPTIVE EVALUATION (NEW - handles OPTIONAL/column ambiguity)
================================================================================

Problem: Different LLMs may extract different columns (URIs vs names vs both).
Ground Truth uses OPTIONAL for all columns, but we need to match against what
the LLM actually returned.

STEP 1: Column-Level Detection
------------------------------
For each relevance level (PREFERRED, ACCEPTABLE), check if the LLM
results contain ANY matching values from that level's columns.

- PREFERRED columns match? → Level is ACTIVE (>0 matches)
- PREFERRED columns match? → Level is ACTIVE (>0 matches)
- If BOTH PREFERRED have 0 matches → Recall = 0, abort
  (ACCEPTABLE alone is not a valid answer)
- ACCEPTABLE columns match? → Only if PREFERRED is active

STEP 2: Filter GT to Active Columns
-----------------------------------
Remove columns from GT that are not active (no matches in LLM results).
This creates the "relevant GT subset" for comparison.

STEP 3: NULL Variant Generation (2^n combinations)
--------------------------------------------------
GT may have NULL values due to OPTIONAL. We don't know which columns the LLM
used OPTIONAL for, so we generate all possible variants:

- n = number of columns with NULL values
- Generate 2^n variants by including/excluding rows with NULLs per column
- Example with 2 NULL columns: 4 variants
  - Keep all rows (with NULLs)
  - Remove rows with NULL in column A
  - Remove rows with NULL in column B
  - Remove rows with NULL in both A and B

STEP 4: Calculate Metrics for Each Variant
------------------------------------------
For each GT variant, calculate Precision and Recall against LLM results.

STEP 5: Select Best Metrics
---------------------------
- Final Recall = max(Recall across all variants)
- Final Precision = max(Precision across all variants)
- This finds the "best interpretation" of the LLM's answer automatically

This ensures we don't penalize LLMs for choosing different valid approaches.
================================================================================
"""

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from data.queries.use_case_queries_tiered_tuples import (
    AdaptiveGroundTruth,
    RelevanceLevel,
)

logger = logging.getLogger(__name__)


# =============================================================================
# NEW: Data structures for Adaptive Evaluation
# =============================================================================


@dataclass
class ColumnMetadata:
    """Metadata for a single column in the Ground Truth.

    Attributes:
        name: Column name (e.g., "?university", "?name")
        level: Relevance level (PREFERRED, ACCEPTABLE)
        index: Column index in the tuple (0-based)
        has_nulls: Whether this column has NULL values in GT
    """
    name: str
    level: RelevanceLevel
    index: int
    has_nulls: bool = False


@dataclass
class AdaptiveMetricsResult:
    """Result of adaptive evaluation with all variant metrics.

    Stores the best metrics found across all GT variants, plus debug info
    about which variant produced the best results.
    """
    # Best metrics (max across all variants)
    best_recall: float = 0.0
    best_precision: float = 0.0
    best_f1: float = 0.0

    # Which levels were active (had matches)
    essential_active: bool = False
    preferred_active: bool = False
    acceptable_active: bool = False

    # Match counts per level
    essential_matches: int = 0
    preferred_matches: int = 0
    acceptable_matches: int = 0

    # Variant that produced best recall
    best_recall_variant: str = ""
    best_recall_gt_size: int = 0

    # Variant that produced best precision
    best_precision_variant: str = ""
    best_precision_gt_size: int = 0

    # Total variants evaluated
    variants_evaluated: int = 0

    # All variant results (for debugging)
    variant_results: list[dict] = field(default_factory=list)

    # LLM result info
    llm_row_count: int = 0
    llm_columns: list[str] = field(default_factory=list)

    # === Schema Metrics (SELECT clause - which columns were addressed?) ===
    schema_recall: float = 0.0  # Active (ESS+PREF) columns / Total (ESS+PREF) columns
    schema_precision: float = 0.0  # Matched LLM columns / All LLM columns
    schema_f1: float = 0.0  # Harmonic mean of schema metrics
    schema_expected_count: int = 0  # Number of critical GT columns (PREFERRED)
    schema_matched_count: int = 0  # Number of active critical GT columns
    schema_llm_columns: int = 0  # Number of LLM result columns
    schema_llm_matched: int = 0  # Number of LLM columns matching any GT column

    # === Column Mapping (GT → LLM column assignment) ===
    # Produced by STEP 2 of adaptive evaluation. Each entry is (gt_col_idx, llm_col_name).
    column_mapping: list[tuple[int, str]] = field(default_factory=list)
    # How the mapping was produced: "signatures" (strict ontological subgraph
    # matching) or "value_overlap" (legacy fallback). Useful for triage when
    # investigating evaluation regressions.
    column_mapping_source: str = "value_overlap"

    # === Prefix Matching (for ranked/top-N queries) ===
    # When requires_prefix_match=True, this captures how many contiguous rows
    # from the top of the GT results matched the LLM results
    detected_limit: int | None = None  # Number of prefix rows matched (None if not applicable)
    prefix_match_perfect: bool = False  # True if all LLM rows match GT prefix exactly

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "best_recall": self.best_recall,
            "best_precision": self.best_precision,
            "best_f1": self.best_f1,
            "essential_active": self.essential_active,
            "preferred_active": self.preferred_active,
            "acceptable_active": self.acceptable_active,
            "essential_matches": self.essential_matches,
            "preferred_matches": self.preferred_matches,
            "acceptable_matches": self.acceptable_matches,
            "best_recall_variant": self.best_recall_variant,
            "best_recall_gt_size": self.best_recall_gt_size,
            "best_precision_variant": self.best_precision_variant,
            "best_precision_gt_size": self.best_precision_gt_size,
            "variants_evaluated": self.variants_evaluated,
            "llm_row_count": self.llm_row_count,
            # Schema Metrics (SELECT clause)
            "schema_recall": self.schema_recall,
            "schema_precision": self.schema_precision,
            "schema_f1": self.schema_f1,
            "schema_expected_count": self.schema_expected_count,
            "schema_matched_count": self.schema_matched_count,
            "schema_llm_columns": self.schema_llm_columns,
            "schema_llm_matched": self.schema_llm_matched,
            # Column Mapping (GT → LLM)
            "column_mapping": [
                {"gt_col_idx": gt_idx, "llm_col_name": llm_name}
                for gt_idx, llm_name in self.column_mapping
            ],
            "column_mapping_source": self.column_mapping_source,
            # Prefix Matching (for ranked/top-N queries)
            "detected_limit": self.detected_limit,
            "prefix_match_perfect": self.prefix_match_perfect,
        }


@dataclass
class TieredMetricsResult:
    """Result of tiered relevance evaluation."""

    # === Per-Level Metrics (PRIMARY - user-centric hierarchy) ===
    # Level 1: PREFERRED - Human-readable names (what users want)
    preferred_recall: float = 0.0
    preferred_precision: float = 0.0
    preferred_f1: float = 0.0
    preferred_found: int = 0
    preferred_total: int = 0

    # Level 2: PREFERRED - Entity URIs (technically correct)
    essential_recall: float = 0.0
    essential_precision: float = 0.0
    essential_f1: float = 0.0
    essential_found: int = 0
    essential_total: int = 0

    # Level 3: ACCEPTABLE - Bonus information
    acceptable_recall: float = 0.0
    acceptable_precision: float = 0.0
    acceptable_f1: float = 0.0
    acceptable_found: int = 0
    acceptable_total: int = 0

    # === Combined Metrics ===
    recall: float = 0.0  # Combined recall (PREFERRED)
    precision: float = 0.0  # Valid values / Generated
    strict_precision: float = 0.0  # Direct answers only / Generated
    f1_score: float = 0.0  # Harmonic mean

    # Counts for CORRECT values (PREFERRED combined)
    correct_found: int = 0
    correct_total: int = 0

    # Noise (irrelevant values)
    noise_count: int = 0
    noise_ratio: float = 0.0

    # Total counts
    generated_count: int = 0

    # === NEW: Tuple-based Metrics (Results/WHERE clause) ===
    tuple_recall: float = 0.0  # How many expected tuples found?
    tuple_precision: float = 0.0  # How many generated rows are valid tuples?
    tuple_f1_score: float = 0.0  # Harmonic mean of tuple metrics
    tuples_found: int = 0
    tuples_expected: int = 0
    generated_rows: int = 0

    # === Schema Metrics (SELECT clause - which columns were addressed?) ===
    schema_recall: float = 0.0  # Active (ESS+PREF) columns / Total (ESS+PREF) columns
    schema_precision: float = 0.0  # Matched LLM columns / All LLM columns
    schema_f1: float = 0.0  # Harmonic mean of schema metrics
    schema_expected_count: int = 0  # Number of critical GT columns (PREFERRED)
    schema_matched_count: int = 0  # Number of active critical GT columns
    schema_llm_columns: int = 0  # Number of LLM result columns
    schema_llm_matched: int = 0  # Number of LLM columns matching any GT column

    # Debug info
    noise_values: set[str] | None = None
    missing_correct: set[str] | None = None
    missing_tuples: set[tuple[str, ...]] | None = None

    # Legacy
    weighted_recall: float = 0.0
    relevant_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without large sets)."""
        return {
            # Per-Level Metrics (PRIMARY)
            "preferred_recall": self.preferred_recall,
            "preferred_precision": self.preferred_precision,
            "preferred_f1": self.preferred_f1,
            "preferred_found": self.preferred_found,
            "preferred_total": self.preferred_total,
            "essential_recall": self.essential_recall,
            "essential_precision": self.essential_precision,
            "essential_f1": self.essential_f1,
            "essential_found": self.essential_found,
            "essential_total": self.essential_total,
            "acceptable_recall": self.acceptable_recall,
            "acceptable_precision": self.acceptable_precision,
            "acceptable_f1": self.acceptable_f1,
            "acceptable_found": self.acceptable_found,
            "acceptable_total": self.acceptable_total,
            # Combined Metrics
            "recall": self.recall,
            "precision": self.precision,
            "strict_precision": self.strict_precision,
            "f1_score": self.f1_score,
            "correct_found": self.correct_found,
            "correct_total": self.correct_total,
            # Noise
            "noise_count": self.noise_count,
            "noise_ratio": self.noise_ratio,
            # Totals
            "generated_count": self.generated_count,
            # Tuple Metrics (Results/WHERE clause)
            "tuple_recall": self.tuple_recall,
            "tuple_precision": self.tuple_precision,
            "tuple_f1_score": self.tuple_f1_score,
            "tuples_found": self.tuples_found,
            "tuples_expected": self.tuples_expected,
            "generated_rows": self.generated_rows,
            # Schema Metrics (SELECT clause)
            "schema_recall": self.schema_recall,
            "schema_precision": self.schema_precision,
            "schema_f1": self.schema_f1,
            "schema_expected_count": self.schema_expected_count,
            "schema_matched_count": self.schema_matched_count,
            "schema_llm_columns": self.schema_llm_columns,
            "schema_llm_matched": self.schema_llm_matched,
        }


def normalize_value(value: Any) -> str:
    """Normalize a value for comparison.

    - Extracts value from SPARQL JSON format {"type": "uri", "value": "..."}
    - Strips whitespace
    - Converts to lowercase for case-insensitive comparison
    """
    if isinstance(value, dict):
        raw_value = str(value.get("value", ""))
    else:
        raw_value = str(value)

    return raw_value.strip().lower()


# Trivial values that should not count as meaningful overlap for column detection
TRIVIAL_VALUES = frozenset({
    # Zeros and empty
    "0", "0.0", "0.00", "0.000", "0.0000", "0.00000", "0.000000",
    "", "null", "none", "n/a", "na", "nil",
    # Boolean
    "true", "false", "yes", "no", "1", "0",
    # Common placeholders
    "unknown", "undefined", "other", "-", "--", "...",
})


def is_trivial_value(value: str) -> bool:
    """Check if a value is trivial and should be excluded from column activity detection.

    Trivial values like 0, null, true/false should not count as meaningful overlap
    because they appear coincidentally in many datasets.
    """
    if not value:
        return True
    v = value.strip().lower()
    # Check exact matches
    if v in TRIVIAL_VALUES:
        return True
    # Check numeric zeros with various decimal places
    try:
        if float(v) == 0.0:
            return True
    except (ValueError, TypeError):
        pass
    return False


def infer_column_datatype(values: set[str], sample_size: int = 100) -> str | None:
    """Infer datatype from column values by examining their format.

    This function checks if all values in a column are consistently numeric
    or temporal (date/datetime). If mixed types or string-only, returns None.

    Args:
        values: Set of normalized column values (lowercase, stripped)
        sample_size: Maximum number of values to check (for performance)

    Returns:
        - "numeric" if all non-empty values are parsable as int/float
        - "temporal" if all non-empty values are ISO-8601 dates/datetimes
        - None if mixed types, string-only, or empty column

    Examples:
        >>> infer_column_datatype({"10", "20", "3.14"})
        'numeric'
        >>> infer_column_datatype({"2020-01-15", "2021-06-30"})
        'temporal'
        >>> infer_column_datatype({"hello", "world"})
        None
    """
    from datetime import datetime

    if not values:
        return None

    # Sample values for performance (check first N non-empty values)
    non_empty = [v for v in values if v and v.strip()]
    if not non_empty:
        return None

    sample = non_empty[:sample_size]

    # Check if all values are numeric
    all_numeric = True
    for v in sample:
        try:
            float(v)
        except (ValueError, TypeError):
            all_numeric = False
            break

    if all_numeric:
        return "numeric"

    # Check if all values are temporal (ISO-8601 date or datetime)
    all_temporal = True
    temporal_formats = [
        "%Y-%m-%d",           # 2020-01-15
        "%Y-%m-%dT%H:%M:%S",  # 2020-01-15T10:30:00
        "%Y-%m-%dT%H:%M:%SZ", # 2020-01-15T10:30:00Z
        "%Y-%m-%d %H:%M:%S",  # 2020-01-15 10:30:00
        "%H:%M:%S",           # 00:04:51 (time only)
        "%Y",                 # 2020 (year only)
        "%Y-%m",              # 2020-01 (year-month)
    ]

    for v in sample:
        parsed = False
        for fmt in temporal_formats:
            try:
                datetime.strptime(v, fmt)
                parsed = True
                break
            except (ValueError, TypeError):
                continue
        if not parsed:
            all_temporal = False
            break

    if all_temporal:
        return "temporal"

    # Mixed or string-only
    return None


def are_properties_compatible(
    gt_property: str | None,
    llm_property: str | None,
) -> bool:
    """Check if two SPARQL properties are the same for numeric/temporal columns.

    This function is called ONLY when both columns are inferred to be numeric
    or temporal from their values. It checks if both columns use the SAME property.

    The key insight: Numeric/temporal VALUES with overlap can come from DIFFERENT
    properties (e.g., routeName="10" via tro:abbreviation vs stationCode="10" via
    tro:code). These should NOT match despite value overlap.

    Args:
        gt_property: Ground truth property URI (from incoming_predicates or aggregation_inner)
        llm_property: LLM property URI (from incoming_predicates or aggregation_inner)

    Returns:
        True if:
        - Both properties are the same URI
        - Either property is missing (fail-open to avoid blocking valid matches)
        False if:
        - Properties are different URIs

    Examples:
        >>> are_properties_compatible("bsbm:productPropertyNumeric1", "bsbm:productPropertyNumeric1")
        True  # Same property
        >>> are_properties_compatible("tro:abbreviation", "tro:code")
        False  # Different properties (routeName vs stationCode)
        >>> are_properties_compatible(None, "tro:code")
        True  # Missing GT property, fail-open
    """
    # If either property is missing, fail-open (allow match)
    if gt_property is None or llm_property is None:
        return True

    # Direct comparison: properties must be identical
    return gt_property == llm_property


def _extract_bindings(results: dict | list | None) -> list[dict]:
    """Extract bindings list from SPARQL results in various formats.

    Handles nested SPARQL JSON format, flat binding lists, and single dicts.
    """
    if results is None:
        return []
    if isinstance(results, list):
        return [b for b in results if isinstance(b, dict)]
    if isinstance(results, dict):
        if "results" in results and isinstance(results["results"], dict):
            return results["results"].get("bindings", [])
        if "bindings" in results:
            return results["bindings"]
        return [results]
    return []


def extract_result_values(results: dict | list | None) -> set[str]:
    """Extract all unique values from SPARQL results.

    Handles various SPARQL result formats and normalizes values.
    """
    if results is None:
        return set()

    values = set()

    # Handle list of bindings
    if isinstance(results, list):
        bindings = results
    elif isinstance(results, dict):
        # Try nested format first
        if "results" in results and isinstance(results["results"], dict):
            bindings = results["results"].get("bindings", [])
        elif "bindings" in results:
            bindings = results["bindings"]
        else:
            bindings = [results]
    else:
        return set()

    for binding in bindings:
        if isinstance(binding, dict):
            for var, value in binding.items():
                if var.startswith("_"):
                    continue
                normalized = normalize_value(value)
                if normalized:
                    values.add(normalized)

    return values



def calculate_metrics_tuple_match(
    gt_tuples: set[tuple[str, ...]],
    llm_tuples: set[tuple[str, ...]],
) -> tuple[float, float, int, int, int]:
    """Standard Precision/Recall via exact projected tuple matching.

    Both GT and LLM tuples must be projected to the same matched columns
    before calling this function. Matching is exact: a tuple is a TP only
    if all values match positionally.

    Args:
        gt_tuples: Projected ground truth tuples (the solution set)
        llm_tuples: Projected LLM result tuples (same column order as GT)

    Returns:
        Tuple of (recall, precision, tp, fp, fn)
    """
    if not gt_tuples and not llm_tuples:
        return 0.0, 0.0, 0, 0, 0

    tp = len(gt_tuples & llm_tuples)
    fp = len(llm_tuples - gt_tuples)
    fn = len(gt_tuples - llm_tuples)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recall, precision, tp, fp, fn


# =============================================================================
# Schema Metrics Functions (SELECT clause evaluation)
# =============================================================================


def extract_llm_column_values(results: dict | list | None) -> dict[str, set[str]]:
    """Extract values per column from LLM SPARQL results for column-level matching.

    This function extracts values grouped by their variable/column name,
    enabling column-to-column matching between LLM results and Ground Truth.

    Args:
        results: SPARQL results in various formats (JSON, list of bindings)

    Returns:
        Dictionary mapping variable names to sets of normalized values
    """
    column_values: dict[str, set[str]] = {}

    if results is None:
        return column_values

    # Handle list of bindings
    if isinstance(results, list):
        bindings = results
    elif isinstance(results, dict):
        # Try nested format first
        if "results" in results and isinstance(results["results"], dict):
            bindings = results["results"].get("bindings", [])
        elif "bindings" in results:
            bindings = results["bindings"]
        else:
            bindings = [results]
    else:
        return column_values

    for binding in bindings:
        if isinstance(binding, dict):
            for var, value in binding.items():
                if var.startswith("_"):
                    continue
                normalized = normalize_value(value)
                if var not in column_values:
                    column_values[var] = set()
                if normalized:
                    column_values[var].add(normalized)

    return column_values


def extract_llm_tuples(results: dict | list | None) -> set[tuple[str, ...]]:
    """Extract tuples from LLM SPARQL results for tuple-based matching.

    Each binding/row becomes a tuple of normalized values (sorted by variable name
    for consistent ordering).

    Args:
        results: SPARQL results in various formats (JSON, list of bindings)

    Returns:
        Set of tuples, each representing one result row
    """
    tuples: set[tuple[str, ...]] = set()

    if results is None:
        return tuples

    # Handle list of bindings
    if isinstance(results, list):
        bindings = results
    elif isinstance(results, dict):
        # Try nested format first
        if "results" in results and isinstance(results["results"], dict):
            bindings = results["results"].get("bindings", [])
        elif "bindings" in results:
            bindings = results["bindings"]
        else:
            bindings = [results]
    else:
        return tuples

    for binding in bindings:
        if isinstance(binding, dict):
            # Sort by variable name for consistent tuple ordering
            sorted_vars = sorted(k for k in binding.keys() if not k.startswith("_"))
            row_values = []
            for var in sorted_vars:
                normalized = normalize_value(binding[var])
                row_values.append(normalized if normalized else "")
            if row_values:
                tuples.add(tuple(row_values))

    return tuples


def calculate_schema_metrics(
    llm_column_values: dict[str, set[str]],
    gt_column_values: list[set[str]],
    gt_column_levels: list[RelevanceLevel],
    gt_column_is_measurement: list[bool],
    gt_semantic_concepts: list[str] | None = None,
) -> tuple[float, float, int, int, int, int]:
    """Calculate Schema Precision/Recall via value overlap.

    Schema Recall measures how many semantic concepts from the Ground Truth were
    addressed by the LLM results. Columns with the same non-empty semantic_concept
    are grouped together - a concept is satisfied if AT LEAST ONE column from
    that group has matching values in the LLM results.

    This replaces the previous PREFERRED distinction: both levels are
    now treated equally, and columns are grouped by semantic_concept instead.

    Args:
        llm_column_values: Dict mapping LLM variable names to value sets
        gt_column_values: List of value sets per GT column
        gt_column_levels: Relevance level for each GT column
        gt_column_is_measurement: Whether each GT column is a measurement
        gt_semantic_concepts: Semantic concept labels for each GT column (optional).
            Columns with the same non-empty label are grouped together.

    Returns:
        Tuple of (schema_recall, schema_precision, expected_count, matched_count,
                  llm_columns, llm_matched)
    """
    # Collect all LLM values for overlap checking
    all_llm_values = (
        set().union(*llm_column_values.values()) if llm_column_values else set()
    )

    # Use semantic_concept grouping if provided
    if gt_semantic_concepts:
        # Build semantic concept groups for PREFERRED columns
        # Key: concept name, Value: list of (column_index, values, is_measurement)
        concept_groups: dict[str, list[tuple[int, set[str], bool]]] = {}

        for col_idx, (gt_vals, level, is_measurement, concept) in enumerate(zip(
            gt_column_values, gt_column_levels, gt_column_is_measurement, gt_semantic_concepts
        )):
            # Only PREFERRED count for Recall
            if level not in (RelevanceLevel.PREFERRED,):
                continue

            # Use concept as key, or unique key for empty concepts
            key = concept if concept else f"__col_{col_idx}__"
            if key not in concept_groups:
                concept_groups[key] = []
            concept_groups[key].append((col_idx, gt_vals, is_measurement))

        # Count satisfied concepts
        total_concepts = len(concept_groups)
        satisfied_concepts = 0

        for concept, columns in concept_groups.items():
            # A concept is satisfied if AT LEAST ONE column has a match
            concept_satisfied = False

            for col_idx, gt_vals, is_measurement in columns:
                # All columns (including measurements): check for non-trivial value overlap
                meaningful_overlap = {
                    v for v in (all_llm_values & gt_vals) if not is_trivial_value(v)
                }
                if meaningful_overlap:
                    concept_satisfied = True
                    break

            if concept_satisfied:
                satisfied_concepts += 1

        schema_recall = satisfied_concepts / total_concepts if total_concepts > 0 else 1.0
        # For backwards compatibility, use concept counts for expected/matched
        total_critical = total_concepts
        active_critical = satisfied_concepts

    else:
        # Fallback: original per-column logic (for TieredGroundTruth without semantic_concept)
        active_critical = 0
        total_critical = 0

        for gt_vals, level, is_measurement in zip(
            gt_column_values, gt_column_levels, gt_column_is_measurement
        ):
            # Only PREFERRED count for Recall
            if level in (RelevanceLevel.PREFERRED,):
                total_critical += 1

                # All columns (including measurements): check for non-trivial value overlap
                meaningful_overlap = {
                    v for v in (all_llm_values & gt_vals) if not is_trivial_value(v)
                }
                if meaningful_overlap:
                    active_critical += 1

        schema_recall = active_critical / total_critical if total_critical > 0 else 1.0

    # Schema Precision: LLM columns that match ANY GT column (incl. ACCEPTABLE)
    llm_matched = 0
    for llm_vals in llm_column_values.values():
        for gt_vals in gt_column_values:
            # Check for non-trivial overlap
            meaningful_overlap = {
                v for v in (llm_vals & gt_vals) if not is_trivial_value(v)
            }
            if meaningful_overlap:
                llm_matched += 1
                break

    schema_precision = (
        llm_matched / len(llm_column_values) if llm_column_values else 0.0
    )

    return (
        schema_recall,
        schema_precision,
        total_critical,
        active_critical,
        len(llm_column_values),
        llm_matched,
    )


# =============================================================================
# NEW: Adaptive Evaluation Functions
# =============================================================================


def detect_active_levels(
    llm_values: set[str],
    essential_values: set[str],
    preferred_values: set[str],
    acceptable_values: set[str],
) -> tuple[bool, bool, bool, int, int, int]:
    """Detect which relevance levels are active (have matches in LLM results).

    STEP 1 of adaptive evaluation: Determine which columns/levels the LLM
    actually extracted by checking for value matches.

    Args:
        llm_values: All values from LLM results (normalized)
        essential_values: All values from PREFERRED level columns
        preferred_values: All values from PREFERRED level columns
        acceptable_values: All values from ACCEPTABLE level columns

    Returns:
        Tuple of (essential_active, preferred_active, acceptable_active,
                  essential_matches, preferred_matches, acceptable_matches)
    """
    essential_matches = len(llm_values & essential_values)
    preferred_matches = len(llm_values & preferred_values)
    acceptable_matches = len(llm_values & acceptable_values)

    essential_active = essential_matches > 0
    preferred_active = preferred_matches > 0

    # ACCEPTABLE is only active if at least one of PREFERRED is active
    acceptable_active = acceptable_matches > 0 and (essential_active or preferred_active)

    return (
        essential_active,
        preferred_active,
        acceptable_active,
        essential_matches,
        preferred_matches,
        acceptable_matches,
    )


def generate_null_variants(
    gt_tuples: set[tuple[str, ...]],
    columns_with_nulls: list[int],
) -> list[tuple[set[tuple[str, ...]], str]]:
    """Generate all 2^n variants of GT by removing rows with NULLs in different columns.

    STEP 3 of adaptive evaluation: Since we don't know which columns the LLM
    used OPTIONAL for, we generate all possible combinations.

    Args:
        gt_tuples: Set of GT tuples (each tuple is a row, values are normalized)
        columns_with_nulls: List of column indices that have NULL values

    Returns:
        List of (filtered_tuples, variant_description) pairs
    """
    if not columns_with_nulls:
        # No NULL columns, return single variant with all tuples
        return [(gt_tuples, "all_rows")]

    variants = []
    n = len(columns_with_nulls)

    # Generate all 2^n combinations (which columns to exclude NULLs from)
    # Empty set = keep all rows, full set = remove all rows with any NULL
    for r in range(n + 1):
        for combo in combinations(columns_with_nulls, r):
            # Filter tuples: remove rows that have NULL/empty in ANY of the combo columns
            filtered = set()
            for tup in gt_tuples:
                keep = True
                for col_idx in combo:
                    if col_idx < len(tup):
                        # Check if value is NULL/empty (represented as empty string)
                        if not tup[col_idx] or tup[col_idx].strip() == "":
                            keep = False
                            break
                if keep:
                    filtered.add(tup)

            # Create description
            if not combo:
                desc = "all_rows"
            else:
                desc = f"no_null_in_col_{','.join(map(str, combo))}"

            variants.append((filtered, desc))

    return variants


def _column_mapping_by_value_overlap(
    column_values: list[set[str]],
    llm_col_vals: dict[str, set[str]],
    gt_col_names: list[str] | None = None,
    gt_sigs: dict | None = None,
    llm_sigs: dict | None = None,
) -> list[tuple[int, str]]:
    """Greedy 1:1 column mapping via non-trivial value overlap with property checking.

    For each (GT column, LLM column) pair, counts non-trivial value overlap.
    Then greedily assigns pairs by descending overlap count, 1:1.

    NEW: For numeric/temporal columns, also checks if both columns use the SAME
    SPARQL property. This prevents false matches like routeName (tro:abbreviation)
    vs stationCode (tro:code) despite having overlapping numeric values.

    Args:
        column_values: List of value sets per GT column
        llm_col_vals: Dict mapping LLM variable names to value sets
        gt_col_names: Optional list of GT column names (for signature lookup)
        gt_sigs: Optional dict of GT column signatures (var_name -> ColumnSignature)
        llm_sigs: Optional dict of LLM column signatures (var_name -> ColumnSignature)

    Returns:
        List of (gt_col_idx, llm_col_name) mapping pairs
    """
    from src.evaluation.column_signatures import ColumnSignature

    candidates: list[tuple[int, str, int]] = []
    for gt_idx, gt_vals in enumerate(column_values):
        for llm_name, llm_vals in llm_col_vals.items():
            overlap = {
                v for v in (gt_vals & llm_vals)
                if not is_trivial_value(v)
            }
            if not overlap:
                continue  # No overlap, skip this pair

            # STEP 1: Infer datatype of the OVERLAP (not the entire columns!)
            # This is critical: even if a column has mixed types (e.g., routeName='1','2','r'),
            # if the overlap is purely numeric (e.g., '8'), property checking should apply.
            overlap_type = infer_column_datatype(overlap)

            # STEP 2: If overlap is numeric or temporal, check property compatibility
            if overlap_type in ("numeric", "temporal"):
                # Both columns are the same numeric/temporal type
                # → Check if they use the SAME SPARQL property

                gt_property: str | None = None
                llm_property: str | None = None

                # Extract GT property
                if gt_sigs and gt_col_names and gt_idx < len(gt_col_names):
                    gt_var_name = gt_col_names[gt_idx].lstrip("?")
                    gt_sig: ColumnSignature | None = gt_sigs.get(gt_var_name)
                    if gt_sig:
                        # For aggregates (AVG, SUM, etc.), use aggregation_inner properties
                        # For regular literals, use incoming_predicates
                        predicates = (
                            gt_sig.aggregation_inner
                            if gt_sig.is_aggregated and gt_sig.aggregation_inner
                            else gt_sig.incoming_predicates
                        )

                        if predicates:
                            # Take first property (most columns have exactly one)
                            gt_property = next(iter(predicates))

                # Extract LLM property
                if llm_sigs:
                    llm_var_name = llm_name.lstrip("?")
                    llm_sig: ColumnSignature | None = llm_sigs.get(llm_var_name)
                    if llm_sig:
                        # For aggregates (AVG, SUM, etc.), use aggregation_inner properties
                        # For regular literals, use incoming_predicates
                        predicates = (
                            llm_sig.aggregation_inner
                            if llm_sig.is_aggregated and llm_sig.aggregation_inner
                            else llm_sig.incoming_predicates
                        )

                        if predicates:
                            # Take first property (most columns have exactly one)
                            llm_property = next(iter(predicates))

                # Check compatibility: properties must be identical
                if not are_properties_compatible(gt_property, llm_property):
                    # Different properties → BLOCK this match
                    # Do NOT add to candidates
                    logger.debug(
                        f"Property mismatch: GT col {gt_idx} (overlap_type={overlap_type}, prop={gt_property}) "
                        f"vs LLM col {llm_name} (overlap_type={overlap_type}, prop={llm_property}) - match blocked"
                    )
                    continue

            # If we reach here, the match is valid (either compatible or not numeric/temporal)
            candidates.append((gt_idx, llm_name, len(overlap)))

    candidates.sort(key=lambda x: x[2], reverse=True)
    used_gt: set[int] = set()
    used_llm: set[str] = set()
    mapping: list[tuple[int, str]] = []
    for gt_idx, llm_name, _ in candidates:
        if gt_idx not in used_gt and llm_name not in used_llm:
            mapping.append((gt_idx, llm_name))
            used_gt.add(gt_idx)
            used_llm.add(llm_name)
    return mapping


def calculate_adaptive_metrics(
    llm_results: dict | list | None,
    gt_tuples: set[tuple[str, ...]],
    columns_with_nulls: list[int],
    essential_values: set[str],
    preferred_values: set[str],
    acceptable_values: set[str],
    column_values: list[set[str]] | None = None,
    measurement_columns: list[int] | None = None,
    *,
    gt_sparql: str | None = None,
    llm_sparql: str | None = None,
    gt_col_names: list[str] | None = None,
    use_signature_matching: bool = False,
    dataset_id: str | None = None,
) -> AdaptiveMetricsResult:
    """Calculate adaptive metrics across all GT variants.

    This is the main function for the new adaptive evaluation algorithm.
    It implements all 5 steps of the algorithm.

    Args:
        llm_results: LLM SPARQL results (raw format)
        gt_tuples: Ground Truth tuples (with potential NULLs)
        columns_with_nulls: Column indices that have NULL values
        essential_values: Flattened PREFERRED values for level detection
        preferred_values: Flattened PREFERRED values for level detection
        acceptable_values: Flattened ACCEPTABLE values for level detection
        column_values: List of value sets per column (for column-level filtering)
        measurement_columns: Deprecated - no longer used for result metrics projection.
        gt_sparql: GT SPARQL query for signature extraction (enables datatype checking).
        llm_sparql: LLM SPARQL query for signature extraction (enables datatype checking).
        gt_col_names: GT column names for signature lookup.
        use_signature_matching: Unused, kept for API compatibility (always False).
        dataset_id: Dataset ID for schema loading (enables datatype property lookup).

    Returns:
        AdaptiveMetricsResult with best metrics across all variants
    """
    from src.evaluation.column_signatures import extract_column_signatures

    result = AdaptiveMetricsResult()

    # Extract LLM values (flat set for level detection)
    llm_values = extract_result_values(llm_results)
    # Extract LLM column values (per-column for column mapping)
    llm_col_vals = extract_llm_column_values(llm_results)

    # STEP 1: Detect active levels
    (
        essential_active,
        preferred_active,
        acceptable_active,
        essential_matches,
        preferred_matches,
        acceptable_matches,
    ) = detect_active_levels(llm_values, essential_values, preferred_values, acceptable_values)

    result.essential_active = essential_active
    result.preferred_active = preferred_active
    result.acceptable_active = acceptable_active
    result.essential_matches = essential_matches
    result.preferred_matches = preferred_matches
    result.acceptable_matches = acceptable_matches

    # Check if valid answer (PREFERRED must be active)
    if not essential_active and not preferred_active:
        # Neither PREFERRED columns found - invalid answer
        result.best_recall = 0.0
        result.best_precision = 0.0
        result.best_f1 = 0.0
        result.best_recall_variant = "no_valid_columns"
        result.best_precision_variant = "no_valid_columns"
        return result

    # STEP 2: Column mapping and projection
    # Greedy 1:1 mapping via non-trivial value overlap between GT and LLM columns.
    # NEW: With datatype compatibility checking for numeric/temporal columns.
    llm_tuples: set[tuple[str, ...]] = set()

    # Extract column signatures for property checking (if SPARQL provided)
    gt_sigs: dict = {}
    llm_sigs: dict = {}

    if gt_sparql:
        try:
            gt_sigs = extract_column_signatures(gt_sparql)
        except Exception as e:
            logger.debug(f"Failed to extract GT signatures: {e}")

    if llm_sparql:
        try:
            llm_sigs = extract_column_signatures(llm_sparql)
        except Exception as e:
            logger.debug(f"Failed to extract LLM signatures: {e}")

    if column_values and llm_col_vals:
        column_mapping = _column_mapping_by_value_overlap(
            column_values=column_values,
            llm_col_vals=llm_col_vals,
            gt_col_names=gt_col_names,
            gt_sigs=gt_sigs if gt_sigs else None,
            llm_sigs=llm_sigs if llm_sigs else None,
        )
        result.column_mapping_source = "value_overlap_with_property"

        if column_mapping:
            matched_gt_indices = [gt_idx for gt_idx, _ in column_mapping]
            matched_llm_names = [llm_name for _, llm_name in column_mapping]

            # Project GT tuples to matched columns
            projected_tuples: set[tuple[str, ...]] = set()
            for gt_tuple in gt_tuples:
                projected = tuple(
                    gt_tuple[i] for i in matched_gt_indices if i < len(gt_tuple)
                )
                if projected:
                    projected_tuples.add(projected)

            # Extract LLM tuples in the same column order
            bindings = _extract_bindings(llm_results)
            for binding in bindings:
                row = tuple(
                    normalize_value(binding.get(col, "")) or ""
                    for col in matched_llm_names
                )
                if any(v for v in row):
                    llm_tuples.add(row)

            # Update columns_with_nulls to reflect new column indices
            old_to_new_idx = {old: new for new, old in enumerate(matched_gt_indices)}
            columns_with_nulls = [
                old_to_new_idx[c] for c in columns_with_nulls if c in old_to_new_idx
            ]

            gt_tuples = projected_tuples
            result.column_mapping = column_mapping

    result.llm_row_count = len(llm_tuples)

    # STEP 3: Generate all GT variants (for NULL handling)
    variants = generate_null_variants(gt_tuples, columns_with_nulls)
    result.variants_evaluated = len(variants)

    # STEP 4: Calculate metrics for each variant using standard TP/FP/FN
    best_recall = 0.0
    best_precision = 0.0
    best_recall_variant = ""
    best_recall_gt_size = 0
    best_precision_variant = ""
    best_precision_gt_size = 0

    for gt_variant, variant_desc in variants:
        if not gt_variant:
            continue

        recall, precision, tp, fp, fn = calculate_metrics_tuple_match(
            gt_variant, llm_tuples
        )

        variant_result = {
            "variant": variant_desc,
            "gt_size": len(gt_variant),
            "recall": recall,
            "precision": precision,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        result.variant_results.append(variant_result)

        # STEP 5: Track best metrics
        if recall > best_recall:
            best_recall = recall
            best_recall_variant = variant_desc
            best_recall_gt_size = len(gt_variant)

        if precision > best_precision:
            best_precision = precision
            best_precision_variant = variant_desc
            best_precision_gt_size = len(gt_variant)

    # Set final best metrics
    result.best_recall = best_recall
    result.best_precision = best_precision
    result.best_recall_variant = best_recall_variant
    result.best_recall_gt_size = best_recall_gt_size
    result.best_precision_variant = best_precision_variant
    result.best_precision_gt_size = best_precision_gt_size

    # Calculate F1 from best recall and precision
    if best_recall + best_precision > 0:
        result.best_f1 = 2 * best_recall * best_precision / (best_recall + best_precision)
    else:
        result.best_f1 = 0.0

    return result


def _calculate_prefix_match(
    llm_tuples: set[tuple[str, ...]],
    gt_tuples_ordered: list[tuple[str, ...]],
) -> tuple[int, bool]:
    """Calculate prefix matching for ranked/top-N queries.

    For ranked queries (e.g., "Show top products"), the LLM results should match
    a contiguous prefix of the GT results (the first N rows).

    Args:
        llm_tuples: Set of tuples from LLM results
        gt_tuples_ordered: Ordered list of GT tuples (from ORDER BY query)

    Returns:
        Tuple of:
        - detected_limit: Number of contiguous GT rows matched from the start
        - perfect_match: True if all LLM tuples match the GT prefix exactly
    """
    if not llm_tuples or not gt_tuples_ordered:
        return 0, False

    # Count how many GT rows from the start are in LLM results
    detected_limit = 0
    for gt_tuple in gt_tuples_ordered:
        if gt_tuple in llm_tuples:
            detected_limit += 1
        else:
            # Break on first non-match (must be contiguous)
            break

    # Check if LLM results match prefix perfectly (no extra rows beyond GT prefix)
    gt_prefix_set = set(gt_tuples_ordered[:detected_limit])
    perfect_match = (llm_tuples == gt_prefix_set) if detected_limit > 0 else False

    return detected_limit, perfect_match


def _evaluate_single_query(
    llm_results: dict | list | None,
    ground_truth: AdaptiveGroundTruth,
    endpoint: str,
    timeout: float,
    query_index: int,
    *,
    llm_sparql: str | None = None,
    use_signature_matching: bool = False,
) -> AdaptiveMetricsResult:
    """Evaluate LLM results against a single GT query.

    Internal helper for evaluate_adaptive_ground_truth.
    """
    # Execute GT query to get tuples and metadata
    gt_tuples, col_names, columns_with_nulls = ground_truth.execute_query(
        endpoint, timeout, query_index=query_index
    )

    if not gt_tuples:
        # No GT data - return empty result
        return AdaptiveMetricsResult()

    # Extract values by level for level detection
    # NOTE: essential_values and preferred_values are now identical since PREFERRED
    # was merged into PREFERRED. Kept separate for backwards compatibility with
    # existing result files and metrics structure.
    essential_values = ground_truth.get_values_by_level(RelevanceLevel.PREFERRED, endpoint, timeout, query_index=query_index)
    preferred_values = essential_values  # Same as essential_values (no separate level anymore)
    acceptable_values = ground_truth.get_values_by_level(RelevanceLevel.ACCEPTABLE, endpoint, timeout, query_index=query_index)

    # === BIND-LABEL COLUMN FILTERING ===
    # BIND-label columns contain self-chosen strings (e.g., BIND("most_expensive" AS ?cat))
    # that are not part of the knowledge graph data. They must be excluded from both
    # result tuple matching and schema metrics to avoid penalizing LLMs for choosing
    # different label strings.
    query_columns = ground_truth.get_columns_for_query(query_index)
    bind_label_indices = {
        i for i, col in enumerate(query_columns) if getattr(col, "is_bind_label", False)
    }

    if bind_label_indices:
        keep_indices = [i for i in range(len(query_columns)) if i not in bind_label_indices]

        # Remove BIND-label values from level detection sets
        bind_values = set()
        for t in gt_tuples:
            for i in bind_label_indices:
                if i < len(t) and t[i]:
                    bind_values.add(t[i])
        acceptable_values = acceptable_values - bind_values

        # Project out BIND-label columns from GT tuples
        gt_tuples = {tuple(t[i] for i in keep_indices) for t in gt_tuples}
        col_names = [col_names[i] for i in keep_indices]

        # Remap columns_with_nulls to new indices
        old_to_new = {old: new for new, old in enumerate(keep_indices)}
        columns_with_nulls = [old_to_new[c] for c in columns_with_nulls if c in old_to_new]

        # Filter query_columns (used for schema metrics below)
        query_columns = [query_columns[i] for i in keep_indices]

    # STEP 2: Extract values per column for column-level filtering
    num_cols = len(col_names)
    column_values: list[set[str]] = [set() for _ in range(num_cols)]
    for gt_tuple in gt_tuples:
        for col_idx in range(min(len(gt_tuple), num_cols)):
            val = gt_tuple[col_idx]
            if val and val.strip():  # Skip NULL/empty
                column_values[col_idx].add(val)

    # Get measurement column indices (protected from projection)
    measurement_columns = ground_truth.get_measurement_column_indices(query_index=query_index)

    # Get GT SPARQL query for signature extraction
    gt_sparql = ground_truth.sparql_queries[query_index] if query_index < len(ground_truth.sparql_queries) else None

    # Get dataset ID for schema loading
    dataset_id = getattr(ground_truth, "dataset_id", None)

    # Calculate adaptive metrics (with datatype checking if SPARQL + dataset available)
    result = calculate_adaptive_metrics(
        llm_results=llm_results,
        gt_tuples=gt_tuples,
        columns_with_nulls=columns_with_nulls,
        essential_values=essential_values,
        preferred_values=preferred_values,
        acceptable_values=acceptable_values,
        column_values=column_values,
        measurement_columns=measurement_columns,
        gt_sparql=gt_sparql,
        llm_sparql=llm_sparql,
        gt_col_names=col_names,
        dataset_id=dataset_id,
    )

    result.llm_columns = col_names

    # === SCHEMA METRICS CALCULATION ===
    # query_columns is already filtered (BIND-label columns removed above)
    llm_col_vals = extract_llm_column_values(llm_results)
    gt_col_vals = column_values
    gt_col_levels = [col.level for col in query_columns]
    gt_col_is_measurement = [col.is_measurement for col in query_columns]
    gt_semantic_concepts = [col.semantic_concept for col in query_columns]

    (
        result.schema_recall, result.schema_precision,
        result.schema_expected_count, result.schema_matched_count,
        result.schema_llm_columns, result.schema_llm_matched,
    ) = calculate_schema_metrics(
        llm_col_vals, gt_col_vals, gt_col_levels, gt_col_is_measurement, gt_semantic_concepts
    )

    if result.schema_recall + result.schema_precision > 0:
        result.schema_f1 = (2 * result.schema_recall * result.schema_precision /
                           (result.schema_recall + result.schema_precision))

    return result


def evaluate_adaptive_ground_truth(
    llm_results: dict | list | None,
    ground_truth: AdaptiveGroundTruth,
    endpoint: str,
    timeout: float = 15.0,
    *,
    llm_sparql: str | None = None,
    use_signature_matching: bool = False,
) -> AdaptiveMetricsResult:
    """Evaluate LLM results against an AdaptiveGroundTruth.

    This is the main entry point for the adaptive evaluation algorithm.
    It handles the full pipeline:
    1. Execute all GT queries and find the one with best F1 score
    2. For prefix-match queries, calculate detected_limit
    3. Return best metrics across all GT query variants

    Args:
        llm_results: LLM SPARQL results (raw format)
        ground_truth: AdaptiveGroundTruth with multiple alternative queries
        endpoint: SPARQL endpoint URL
        timeout: Query timeout in seconds
        llm_sparql: LLM SPARQL query for signature extraction (enables datatype checking).
        use_signature_matching: Unused, kept for API compatibility (always False).

    Returns:
        AdaptiveMetricsResult with best metrics across all GT queries
    """
    if not ground_truth.sparql_queries:
        return AdaptiveMetricsResult()

    # Evaluate against all alternative queries and pick the best one
    best_result: AdaptiveMetricsResult | None = None
    best_f1 = -1.0
    best_query_idx = 0  # Track which query gave best result (for prefix matching)

    for query_index in range(len(ground_truth.sparql_queries)):
        result = _evaluate_single_query(
            llm_results=llm_results,
            ground_truth=ground_truth,
            endpoint=endpoint,
            timeout=timeout,
            query_index=query_index,
            llm_sparql=llm_sparql,
        )

        # Track best result by F1 score
        if result.best_f1 > best_f1:
            best_f1 = result.best_f1
            best_result = result
            best_query_idx = query_index

    if best_result is None:
        return AdaptiveMetricsResult()

    # === PREFIX MATCHING (for ranked/top-N queries) ===
    if ground_truth.requires_prefix_match:
        # Re-execute the best query to get ordered results (as list, not set)
        # We need to preserve order for prefix matching
        import httpx

        query = ground_truth.sparql_queries[best_query_idx]

        try:
            response = httpx.post(
                endpoint,
                data={"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            var_names = data.get("head", {}).get("vars", [])
            bindings = data.get("results", {}).get("bindings", [])

            # Build ordered list of GT tuples
            gt_tuples_ordered: list[tuple[str, ...]] = []
            for binding in bindings:
                row_values = []
                for var_name in var_names:
                    if var_name in binding:
                        value = binding[var_name].get("value", "")
                        row_values.append(value.strip().lower())
                    else:
                        row_values.append("")
                gt_tuples_ordered.append(tuple(row_values))

            # Extract LLM tuples
            llm_tuples = extract_llm_tuples(llm_results)

            # Calculate prefix match
            detected_limit, perfect_match = _calculate_prefix_match(
                llm_tuples, gt_tuples_ordered
            )

            best_result.detected_limit = detected_limit
            best_result.prefix_match_perfect = perfect_match

        except Exception:
            # If prefix matching fails, leave as None
            pass

    return best_result


__all__ = [
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
    # Tuple matching
    "calculate_metrics_tuple_match",
    # Trivial value filtering
    "is_trivial_value",
    "TRIVIAL_VALUES",
    # Datatype inference and property checking (NEW)
    "infer_column_datatype",
    "are_properties_compatible",
]
