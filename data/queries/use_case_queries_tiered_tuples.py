"""
Tiered Relevance Query Corpus for VKGQA Evaluation.

This module implements a tiered relevance system for evaluating query results,
replacing the combinatorial explosion of AcceptableInterpretation objects with
a more elegant, dimension-based approach.

Instead of listing all possible valid SPARQL combinations, we define:
- Expected VALUES at different relevance levels
- Evaluation compares actual results against these value sets
- Recall per level + weighted overall score

Value Population Strategy (Overhead-optimized):
- Large unfiltered sets: Store SPARQL query, load at runtime
- Small/filtered sets: Store values directly
- Aggregates: Store values directly

================================================================================
SEMANTIC CONCEPT GROUPING FOR SCHEMA EVALUATION
================================================================================

Columns are grouped by their semantic_concept attribute for Schema Recall
calculation. A semantic concept is satisfied if AT LEAST ONE column from
that group has matching values in the LLM results.

This means:
- URI and name columns for the same entity share a semantic_concept
- Example: "student" URI and "studentName" both have semantic_concept="student"
- Schema Recall = satisfied concepts / total concepts
- 100% recall if at least one column per concept is found

RELEVANCE LEVELS (for Results Recall, not Schema Recall):
- PREFERRED (Level 2): Entity URIs - technically correct
- PREFERRED (Level 1): Human-readable names/labels - user-friendly
- ACCEPTABLE (Level 3): Bonus information - not directly requested

For Schema Recall, PREFERRED are treated equally - both
contribute to the semantic concept. A concept with URI column (PREFERRED)
OR name column (PREFERRED) is satisfied if either is found.

ACCEPTABLE columns do NOT count for Schema Recall but are valid for Precision.

Example - Query A01 "Which students are taking courses?":
  Columns with semantic_concept:
    - student (URI) -> semantic_concept="student"
    - name -> semantic_concept="student"
    - course (URI) -> semantic_concept="course"
    - courseName -> semantic_concept="course"

  Total concepts for Schema Recall: 2 (student, course)
  If LLM returns only student URIs: 1/2 = 50% Schema Recall
  If LLM returns student names: 1/2 = 50% Schema Recall
  If LLM returns both student URI AND name: still 1/2 = 50% (same concept)
  If LLM returns student names + course names: 2/2 = 100% Schema Recall
================================================================================
"""

import re
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Literal

import httpx


class SPARQLComplexity(Enum):
    """SPARQL complexity levels (for compatibility with UseCaseQuery)."""
    SIMPLE = 1      # BGP, basic FILTER, ORDER BY, LIMIT
    MEDIUM = 2      # GROUP BY, Aggregates, OPTIONAL
    HARD = 3        # Subqueries, Negation (OPTIONAL+FILTER), Property Paths
    VERY_HARD = 4   # SERVICE, complex combinations

# Limit for Ground Truth SPARQL queries - must match EXECUTION_RESULT_LIMIT in sparql_tools.py
GROUND_TRUTH_RESULT_LIMIT = 1000


class RelevanceLevel(IntEnum):
    """Relevance level for result values.

    Two-tier system for Schema Recall evaluation:
    - PREFERRED: Core columns that should be returned (URIs or labels)
    - ACCEPTABLE: Bonus information (not directly requested but useful)

    Columns with the same semantic_concept are grouped together.
    A semantic concept is satisfied if AT LEAST ONE column from
    that group is found in LLM results (regardless of URI vs label).
    """

    PREFERRED = 1   # Core columns (URIs or labels) - count for Schema Recall
    ACCEPTABLE = 2  # Bonus info (related attributes, not directly asked)


# =============================================================================
# NEW: Adaptive Ground Truth Structure
# =============================================================================


@dataclass
class ColumnDef:
    """Definition of a column in an Adaptive Ground Truth query.

    Maps a SPARQL variable to a relevance level and semantic concept.

    Attributes:
        var_name: SPARQL variable name (e.g., "university", "name" - without ?)
        level: Relevance level for this column (PREFERRED, ACCEPTABLE)
        description: Human-readable description
        is_measurement: If True, this column represents an aggregated/computed
            value (COUNT, SUM, AVG, etc.) and must NEVER be removed during
            column projection. Measurement columns are always required together
            with their category columns (e.g., field + totalOil).
        semantic_concept: Groups columns representing the same entity/concept.
            For Schema Recall: A concept is satisfied if AT LEAST ONE column
            from that group is found in LLM results. This replaces the
            PREFERRED distinction - both levels are now treated
            equally within a semantic concept group.

            Example: "student" URI and "studentName" both have
            semantic_concept="student". If LLM returns either, the
            "student" concept is satisfied for Schema Recall.
        is_bind_label: If True, this column contains a self-chosen BIND label
            (e.g., BIND("most_expensive" AS ?priceCategory)). BIND-label
            columns are excluded from result tuple matching because their
            values are arbitrary strings chosen by the query author, not
            data from the knowledge graph. LLMs should not be penalized
            for choosing different label strings. The column is also
            excluded from schema metrics evaluation.
    """
    var_name: str
    level: RelevanceLevel
    description: str = ""
    is_measurement: bool = False
    semantic_concept: str = ""
    is_bind_label: bool = False


@dataclass
class AdaptiveGroundTruth:
    """Ground Truth with multiple alternative SPARQL queries and column-level metadata.

    This structure supports multiple equivalent SPARQL interpretations of a natural
    language query. During evaluation, all queries are executed and the one yielding
    the best score is used. This handles semantic variance where multiple query
    formulations are equally valid (e.g., filtering by totalDepth vs finalVerticalDepth).

    The adaptive evaluation algorithm uses this to:
    1. Execute all alternative SPARQL queries
    2. Detect which levels are present in LLM results
    3. Generate GT variants by removing rows with NULLs
    4. Find the best Precision/Recall across all variants and all GT queries

    Example:
        AdaptiveGroundTruth(
            query_id="A01",
            query="Which students are taking courses?",
            sparql_queries=['''
                SELECT ?student ?studentName WHERE {
                    ?student a eduo:Student .
                    ?student eduo:takesCourse ?course .
                    OPTIONAL { ?student eduo:name ?studentName }
                }
            '''],
            columns=[
                ColumnDef("student", RelevanceLevel.PREFERRED, "Student URI"),
                ColumnDef("studentName", RelevanceLevel.PREFERRED, "Student name"),
            ],
            dataset="EDU",
        )
    """

    # Identification
    query_id: str  # e.g., "A01", "B05"
    query_set: Literal["A", "B", "C", "D", "E"] = "A"
    description: str = ""
    query: str = ""  # Natural language query
    dataset: str = ""  # Primary dataset (e.g., "EDU", "NRG")
    datasets: list[str] = field(default_factory=list)

    # List of alternative SPARQL queries - all are valid interpretations
    # During evaluation, the query yielding the best score is used
    sparql_queries: list[str] = field(default_factory=list)

    # Column definitions with relevance levels.
    # Can be either:
    #   - list[ColumnDef]: same columns for all sparql_queries (single variant)
    #   - list[list[ColumnDef]]: per-query columns (one list per sparql_queries entry)
    columns: list[ColumnDef] | list[list[ColumnDef]] = field(default_factory=list)

    # Notes for documentation
    notes: str = ""

    # Prefix matching for ranked/top-N queries (e.g., "Show top products")
    # When True, GT results are ordered and LLM results must match a contiguous
    # prefix (first N rows). The detected_limit metric captures how many rows matched.
    requires_prefix_match: bool = False

    # Cache for loaded tuples (per query_index)
    _cached_tuples: dict[int, set[tuple[str, ...]]] = field(default_factory=dict, repr=False)
    _cached_column_names: dict[int, list[str]] = field(default_factory=dict, repr=False)

    def get_columns_for_query(self, query_index: int = 0) -> list[ColumnDef]:
        """Get column definitions for a specific query variant.

        If columns is a flat list[ColumnDef], returns it for all query indices.
        If columns is list[list[ColumnDef]], returns columns[query_index].
        """
        if not self.columns:
            return []
        if isinstance(self.columns[0], list):
            # Per-query columns
            if query_index < len(self.columns):
                return self.columns[query_index]
            return self.columns[0]  # Fallback to first
        # Flat list — same columns for all queries
        return self.columns

    def get_column_by_level(self, level: RelevanceLevel, query_index: int = 0) -> list[ColumnDef]:
        """Get all columns for a specific relevance level."""
        return [col for col in self.get_columns_for_query(query_index) if col.level == level]

    def get_column_indices_by_level(self, level: RelevanceLevel, query_index: int = 0) -> list[int]:
        """Get column indices for a specific relevance level."""
        return [i for i, col in enumerate(self.get_columns_for_query(query_index)) if col.level == level]

    def get_columns_with_nulls(self, query_index: int = 0) -> list[int]:
        """Get indices of columns that may have NULL values (OPTIONAL columns).

        By convention, PREFERRED columns should not have NULLs (no OPTIONAL).
        PREFERRED and ACCEPTABLE columns may have NULLs.
        """
        cols = self.get_columns_for_query(query_index)
        return [i for i, col in enumerate(cols)
                if col.level in (RelevanceLevel.PREFERRED, RelevanceLevel.ACCEPTABLE)]

    def get_measurement_column_indices(self, query_index: int = 0) -> list[int]:
        """Get indices of columns that are measurements (protected from projection).

        Measurement columns represent aggregated/computed values (COUNT, SUM, AVG)
        and must NEVER be removed during column projection, even if their values
        don't match. They are always required together with their category columns.
        """
        cols = self.get_columns_for_query(query_index)
        return [i for i, col in enumerate(cols) if col.is_measurement]

    def _add_limit_to_query(self, query: str) -> str:
        """Return query unchanged. GT queries without explicit LIMIT are
        fetched complete so LLMs returning all rows aren't penalized by a
        truncated GT."""
        return query

    def execute_query(
        self,
        endpoint: str,
        timeout: float = 60.0,
        query_index: int = 0,
    ) -> tuple[set[tuple[str, ...]], list[str], list[int]]:
        """Execute a SPARQL query and return tuples with metadata.

        Args:
            endpoint: SPARQL endpoint URL
            timeout: Request timeout in seconds
            query_index: Index of the query to execute (default: 0, first query)

        Returns:
            Tuple of:
            - Set of result tuples (each tuple is one row, values normalized)
            - List of column names (in order)
            - List of column indices that have NULL values
        """
        # Use per-index cache
        if query_index in self._cached_tuples and query_index in self._cached_column_names:
            null_cols = self._find_null_columns(self._cached_tuples[query_index])
            return self._cached_tuples[query_index], self._cached_column_names[query_index], null_cols

        if not self.sparql_queries or query_index >= len(self.sparql_queries):
            return set(), [], []

        try:
            query_with_limit = self._add_limit_to_query(self.sparql_queries[query_index])
            response = httpx.post(
                endpoint,
                data={"query": query_with_limit},
                headers={"Accept": "application/sparql-results+json"},
                timeout=timeout,
            )
            response.raise_for_status()

            data = response.json()
            var_names = data.get("head", {}).get("vars", [])
            bindings = data.get("results", {}).get("bindings", [])

            tuples = set()
            for binding in bindings:
                row_values = []
                for var_name in var_names:
                    if var_name in binding:
                        value = binding[var_name].get("value", "")
                        # Normalize: strip whitespace, lowercase
                        row_values.append(value.strip().lower())
                    else:
                        # NULL value (OPTIONAL not bound)
                        row_values.append("")

                tuples.add(tuple(row_values))

            self._cached_tuples[query_index] = tuples
            self._cached_column_names[query_index] = var_names

            # Find columns with NULLs
            null_cols = self._find_null_columns(tuples)

            return tuples, var_names, null_cols

        except Exception as e:
            print(f"Warning: Failed to execute AdaptiveGroundTruth query: {e}")
            return set(), [], []

    def _find_null_columns(self, tuples: set[tuple[str, ...]]) -> list[int]:
        """Find column indices that have at least one NULL/empty value."""
        if not tuples:
            return []

        num_cols = len(next(iter(tuples))) if tuples else 0
        null_cols = []

        for col_idx in range(num_cols):
            for tup in tuples:
                if col_idx < len(tup) and (not tup[col_idx] or tup[col_idx].strip() == ""):
                    null_cols.append(col_idx)
                    break

        return null_cols

    def get_values_by_level(
        self,
        level: RelevanceLevel,
        endpoint: str,
        timeout: float = 60.0,
        query_index: int = 0,
    ) -> set[str]:
        """Get flattened values for a specific relevance level.

        Useful for level detection in the adaptive algorithm.
        """
        tuples, _, _ = self.execute_query(endpoint, timeout, query_index=query_index)
        level_indices = self.get_column_indices_by_level(level, query_index=query_index)

        values = set()
        for tup in tuples:
            for idx in level_indices:
                if idx < len(tup) and tup[idx]:
                    values.add(tup[idx])

        return values

    def clear_cache(self) -> None:
        """Clear cached query results."""
        self._cached_tuples = {}
        self._cached_column_names = {}

    # =========================================================================
    # Compatibility properties (for UseCaseQuery interface)
    # =========================================================================

    @property
    def id(self) -> str:
        """Alias for query_id (UseCaseQuery compatibility)."""
        return self.query_id

    @property
    def expected_sparql(self) -> str:
        """Returns first SPARQL query (UseCaseQuery compatibility)."""
        return self.sparql_queries[0] if self.sparql_queries else ""

    @property
    def complexity(self) -> "SPARQLComplexity":
        """Infer complexity from query structure (UseCaseQuery compatibility)."""
        q = self.sparql_queries[0].upper() if self.sparql_queries else ""
        if "SERVICE" in q:
            return SPARQLComplexity.VERY_HARD
        if any(kw in q for kw in ["HAVING", "NOT EXISTS", "MINUS"]):
            return SPARQLComplexity.HARD
        if "FILTER(!BOUND" in q.replace(" ", ""):
            return SPARQLComplexity.HARD
        if any(kw in q for kw in ["GROUP BY", "COUNT", "SUM", "AVG", "OPTIONAL"]):
            return SPARQLComplexity.MEDIUM
        return SPARQLComplexity.SIMPLE


@dataclass
class ValueSet:
    """A set of expected values at a specific relevance level.

    Either `values` or `sparql_query` should be provided:
    - Use `values` for small/filtered sets (stored directly)
    - Use `sparql_query` for large sets (loaded at runtime to reduce file overhead)

    TUPLE MODE (is_tuple_based=True):
    When values belong together (e.g., "DeptA" with "50 students"), use tuple mode.
    This ensures that the evaluation checks if values appear IN THE SAME ROW,
    not just anywhere in the results.

    Example: "How many students per department?"
    - Without tuples: Checks if "DeptA", "DeptB", "50", "30" exist anywhere
    - With tuples: Checks if ("DeptA", "50") and ("DeptB", "30") exist as rows
    """

    level: RelevanceLevel
    description: str  # Human-readable description, e.g., "Student Names"

    # Option 1: Direct values (for small/filtered sets)
    values: set[str] = field(default_factory=set)

    # Option 2: SPARQL query to load values (for large sets)
    sparql_query: str | None = None

    # NEW: Tuple mode - when True, all columns from SPARQL query are treated as a tuple
    # that must appear together in the same row of generated results
    is_tuple_based: bool = False

    # Dataset this ValueSet queries (for cross-dataset queries)
    # If None, uses the primary dataset of the AdaptiveGroundTruth
    dataset: str | None = None

    # Cache for loaded values
    _cached_values: set[str] | None = field(default=None, repr=False)

    # Cache for loaded tuples (when is_tuple_based=True)
    _cached_tuples: set[tuple[str, ...]] | None = field(default=None, repr=False)

    def get_values(self, endpoint: str | None = None, timeout: float = 30.0) -> set[str]:
        """Return values - either direct or loaded via SPARQL.

        For tuple-based ValueSets, this returns the flattened set of all values
        (for backwards compatibility). Use get_tuples() for tuple matching.

        Args:
            endpoint: SPARQL endpoint URL (required if using sparql_query)
            timeout: HTTP request timeout in seconds

        Returns:
            Set of string values
        """
        # Direct values take precedence
        if self.values:
            return self.values

        # Try to use cached values
        if self._cached_values is not None:
            return self._cached_values

        # Load via SPARQL
        if self.sparql_query and endpoint:
            if self.is_tuple_based:
                # For tuple mode, load tuples and flatten for backwards compatibility
                tuples = self.get_tuples(endpoint, timeout)
                self._cached_values = {val for tup in tuples for val in tup}
            else:
                self._cached_values = self._execute_sparql_single(endpoint, timeout)
            return self._cached_values

        return set()

    def get_tuples(self, endpoint: str | None = None, timeout: float = 30.0) -> set[tuple[str, ...]]:
        """Return tuples - all columns from SPARQL query as tuples.

        Only meaningful when is_tuple_based=True. Each row from the SPARQL
        result becomes a tuple of values that must appear together.

        Args:
            endpoint: SPARQL endpoint URL (required if using sparql_query)
            timeout: HTTP request timeout in seconds

        Returns:
            Set of tuples, where each tuple contains values from one result row
        """
        if not self.is_tuple_based:
            # For non-tuple ValueSets, wrap single values in 1-tuples
            values = self.get_values(endpoint, timeout)
            return {(v,) for v in values}

        # Try to use cached tuples
        if self._cached_tuples is not None:
            return self._cached_tuples

        # Load via SPARQL
        if self.sparql_query and endpoint:
            self._cached_tuples = self._execute_sparql_tuples(endpoint, timeout)
            return self._cached_tuples

        return set()

    def _add_limit_to_query(self, query: str) -> str:
        """Return query unchanged. GT queries without explicit LIMIT are
        fetched complete so LLMs returning all rows aren't penalized by a
        truncated GT."""
        return query

    def _execute_sparql_single(self, endpoint: str, timeout: float) -> set[str]:
        """Execute SPARQL query and extract values from first column."""
        try:
            # Add LIMIT to match generated query limits for fair comparison
            query_with_limit = self._add_limit_to_query(self.sparql_query)
            response = httpx.post(
                endpoint,
                data={"query": query_with_limit},
                headers={"Accept": "application/sparql-results+json"},
                timeout=timeout,
            )
            response.raise_for_status()

            data = response.json()
            bindings = data.get("results", {}).get("bindings", [])

            values = set()
            for binding in bindings:
                # Extract first variable's value
                for var_name, var_data in binding.items():
                    value = var_data.get("value", "")
                    if value:
                        values.add(value)
                    break  # Only first column

            return values

        except Exception as e:
            print(f"Warning: Failed to execute SPARQL query: {e}")
            return set()

    def _execute_sparql_tuples(self, endpoint: str, timeout: float) -> set[tuple[str, ...]]:
        """Execute SPARQL query and extract ALL columns as tuples."""
        try:
            # Add LIMIT to match generated query limits for fair comparison
            query_with_limit = self._add_limit_to_query(self.sparql_query)
            response = httpx.post(
                endpoint,
                data={"query": query_with_limit},
                headers={"Accept": "application/sparql-results+json"},
                timeout=timeout,
            )
            response.raise_for_status()

            data = response.json()
            var_names = data.get("head", {}).get("vars", [])
            bindings = data.get("results", {}).get("bindings", [])

            tuples = set()
            for binding in bindings:
                # Extract ALL variables in order, normalize values
                row_values = []
                for var_name in var_names:
                    if var_name in binding:
                        value = binding[var_name].get("value", "")
                        # Normalize: strip whitespace, lowercase for comparison
                        row_values.append(value.strip().lower())
                    else:
                        row_values.append("")  # Handle missing optional values

                # Only add non-empty tuples
                if any(v for v in row_values):
                    tuples.add(tuple(row_values))

            return tuples

        except Exception as e:
            print(f"Warning: Failed to execute SPARQL query for tuples: {e}")
            return set()

    def clear_cache(self) -> None:
        """Clear cached values to force reload on next access."""
        self._cached_values = None
        self._cached_tuples = None


# =============================================================================
# TIERED QUERIES - SET A (Single-Mapping Baseline)
# =============================================================================


# =============================================================================
# Query lookup functions
# (All query definitions moved to experimental_corpus.py)
# =============================================================================

def get_adaptive_query(query_id: str) -> AdaptiveGroundTruth | None:
    """Get an adaptive query by ID from experimental corpus."""
    from data.queries.experimental_corpus import get_experimental_query
    return get_experimental_query(query_id)


def get_adaptive_queries_by_set(query_set: str) -> list[AdaptiveGroundTruth]:
    """Get all adaptive queries for a specific set from experimental corpus."""
    from data.queries.experimental_corpus import get_queries_by_set
    return get_queries_by_set(query_set)


# =============================================================================
# LARGE Ground Truth Cache Loading
# =============================================================================

# Global cache for LARGE ground truth (loaded once, reused across queries)
_LARGE_GROUND_TRUTH_CACHE: dict[str, dict] | None = None


def _load_large_ground_truth_file() -> dict[str, dict]:
    """Load the large_ground_truth_combined.json file.

    Returns cached data if already loaded.
    """
    global _LARGE_GROUND_TRUTH_CACHE

    if _LARGE_GROUND_TRUTH_CACHE is not None:
        return _LARGE_GROUND_TRUTH_CACHE

    import json
    from pathlib import Path

    cache_file = Path(__file__).parent.parent / "ground_truth" / "large_ground_truth_combined.json"

    if not cache_file.exists():
        print(f"Warning: LARGE ground truth cache not found at {cache_file}")
        _LARGE_GROUND_TRUTH_CACHE = {}
        return _LARGE_GROUND_TRUTH_CACHE

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            _LARGE_GROUND_TRUTH_CACHE = json.load(f)
        print(f"Loaded LARGE ground truth cache: {len(_LARGE_GROUND_TRUTH_CACHE)} queries")
    except Exception as e:
        print(f"Warning: Failed to load LARGE ground truth cache: {e}")
        _LARGE_GROUND_TRUTH_CACHE = {}

    return _LARGE_GROUND_TRUTH_CACHE


def populate_large_query_cache(query: "AdaptiveGroundTruth") -> bool:
    """Populate the cache of a LARGE query from pre-computed ground truth.

    This function loads pre-computed results for EDU and TRN LARGE queries
    from the large_ground_truth_combined.json file, avoiding expensive live
    SPARQL execution during evaluation.

    NRG LARGE queries are skipped (they execute quickly without caching).

    Args:
        query: An AdaptiveGroundTruth object (should be a LARGE query)

    Returns:
        True if cache was populated, False otherwise
    """
    # Only process LARGE queries
    if not query.query_id.startswith("LARGE"):
        return False

    # Skip NRG queries - they execute quickly without caching
    if "NRG" in query.dataset.upper():
        return False

    # Already cached?
    if query._cached_tuples is not None:
        return True

    # Load the ground truth file
    gt_cache = _load_large_ground_truth_file()

    if query.query_id not in gt_cache:
        return False

    entry = gt_cache[query.query_id]
    bindings = entry.get("bindings", [])

    if not bindings:
        return False

    # Extract column names from first binding
    var_names = list(bindings[0].keys()) if bindings else []

    # Convert bindings to normalized tuples
    tuples = set()
    for binding in bindings:
        row_values = []
        for var_name in var_names:
            if var_name in binding:
                value = binding[var_name].get("value", "")
                # Normalize: strip whitespace, lowercase (same as execute_query)
                row_values.append(value.strip().lower())
            else:
                row_values.append("")
        tuples.add(tuple(row_values))

    # Populate the cache
    query._cached_tuples = tuples
    query._cached_column_names = var_names

    return True


def load_large_ground_truth_for_queries(queries: list["AdaptiveGroundTruth"]) -> int:
    """Load pre-computed ground truth for all LARGE queries in the list.

    Call this before running experiments to avoid live SPARQL execution
    for EDU and TRN LARGE queries.

    Args:
        queries: List of AdaptiveGroundTruth objects

    Returns:
        Number of queries with cache populated
    """
    count = 0
    for query in queries:
        if populate_large_query_cache(query):
            count += 1

    if count > 0:
        print(f"Pre-loaded ground truth cache for {count} LARGE queries")

    return count


__all__ = [
    # Data structures
    "RelevanceLevel",
    "ColumnDef",
    "AdaptiveGroundTruth",
    "SPARQLComplexity",
    "ValueSet",
    # Query lookup functions (queries defined in experimental_corpus.py)
    "get_adaptive_query",
    "get_adaptive_queries_by_set",
    # Ground truth caching for LARGE queries
    "populate_large_query_cache",
    "load_large_ground_truth_for_queries",
]
