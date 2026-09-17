"""
Problem Categorization for VKGQA Evaluation.

This module defines enums and data structures for categorizing
query failures and ambiguity types to enable systematic error analysis.

The categorization enables grouping of recurring problems and
correlation analysis between ambiguity types and failure modes.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProblemCategory(str, Enum):
    """Categories of problems detected in LLM-generated SPARQL queries.

    Organized by failure type to enable systematic analysis of
    where and why queries fail.
    """

    # Schema/Concept Failures (semantic_concept not covered)
    CONCEPT_MISSING = "concept_missing"
    """A semantic_concept from Ground Truth is completely missing in LLM results."""

    CONCEPT_PARTIAL = "concept_partial"
    """A semantic_concept has some matching values but low recall (<0.8)."""

    MEASUREMENT_MISSING = "measurement_missing"
    """A measurement column (is_measurement=True) is missing from results."""

    # Filter Failures
    FILTER_MISSING = "filter_missing"
    """Required FILTER clause is missing from the query."""

    FILTER_WRONG_VALUE = "filter_wrong_value"
    """FILTER exists but uses wrong comparison value."""

    # Aggregation Failures
    AGG_MISSING_GROUP_BY = "agg_missing_group_by"
    """Query should have GROUP BY but doesn't."""

    AGG_WRONG_FUNCTION = "agg_wrong_function"
    """Wrong aggregation function used (e.g., SUM instead of COUNT)."""

    # Instance Matching Failures (primarily SET C)
    INSTANCE_CASE_MISMATCH = "instance_case_mismatch"
    """Entity not found due to case variation (e.g., 'troll' vs 'TROLL')."""

    INSTANCE_SYNONYM_NOT_RESOLVED = "instance_synonym_not_resolved"
    """Entity not found because synonym was not resolved (e.g., Statoil vs Equinor)."""

    INSTANCE_TEMPORAL_NOT_RESOLVED = "instance_temporal_not_resolved"
    """Entity not found due to temporal name change (e.g., BP -> BP Norge)."""

    INSTANCE_FUZZY_NOT_RESOLVED = "instance_fuzzy_not_resolved"
    """Entity not found due to typo/fuzzy match failure (e.g., Snore vs Snorre)."""

    # Cross-Dataset Failures (SET D)
    CROSS_WRONG_ENDPOINT = "cross_wrong_endpoint"
    """Query sent to wrong SPARQL endpoint."""

    CROSS_MISSING_QUERY = "cross_missing_query"
    """Multi-dataset query missing required sub-query for one dataset."""

    # Empty Result Failures
    EMPTY_OVER_CONSTRAINED = "empty_over_constrained"
    """Query returns 0 rows due to over-constraining (extra triples/filters)."""

    # Low Recall (generic)
    LOW_RECALL = "low_recall"
    """Query has low recall but specific cause not determined."""


class AmbiguityType(str, Enum):
    """Types of ambiguity present in queries.

    Parsed from the `notes` field of AdaptiveGroundTruth to enable
    correlation analysis between ambiguity types and failure modes.
    """

    # No ambiguity (SET A, B baseline)
    NONE = "none"
    """No specific ambiguity - baseline queries."""

    # Instance Matching Ambiguity (SET C)
    CASE_VARIATION = "case_variation"
    """Case sensitivity issues (e.g., 'ekofisk' vs 'EKOFISK'). Queries: C01, C02."""

    PARTIAL_MATCH = "partial_match"
    """Partial string matching required. Queries: C03, C05, C10."""

    SYNONYM = "synonym"
    """Synonym resolution needed (e.g., Statoil -> Equinor). Queries: C06."""

    TEMPORAL = "temporal"
    """Temporal name changes (e.g., BP -> BP Norge, Elf -> Total). Queries: C07, C12, C15."""

    FUZZY_TYPO = "fuzzy_typo"
    """Fuzzy matching for typos (e.g., Snore -> Snorre). Queries: C13, C14."""

    ABBREVIATION = "abbreviation"
    """Abbreviation handling (e.g., AS/A/S for Aksjeselskap). Queries: C11."""

    NUMERIC_FILTER = "numeric_filter"
    """Numeric value interpretation (e.g., 'affordable' = < 50). Queries: C08."""

    MULTI_STEP = "multi_step"
    """Query requires multiple execution steps. Queries: C04, C09."""

    # Cross-Dataset Ambiguity (SET D)
    CROSS_DATASET = "cross_dataset"
    """Query spans multiple datasets/endpoints. All SET D queries."""

    # Semantic Ambiguity (SET E)
    SEMANTIC_VARIANCE = "semantic_variance"
    """Multiple valid interpretations exist (e.g., depth = totalDepth or verticalDepth). Queries: E05."""

    ROLE_AMBIGUITY = "role_ambiguity"
    """Ambiguous role terms (e.g., 'responsible' = operator vs owner). Queries: E13, E18."""

    UNDERSPECIFIED = "underspecified"
    """Vague terms requiring interpretation (e.g., 'top', 'popular', 'large'). Queries: E17, E22, E23, E24."""

    ACTIVITY_AMBIGUITY = "activity_ambiguity"
    """Ambiguous activity terms (e.g., 'active' = teaching or publishing). Queries: E20."""


class Severity(str, Enum):
    """Severity level of a detected problem."""

    CRITICAL = "critical"
    """Causes empty results or completely wrong output."""

    MAJOR = "major"
    """Significantly affects recall/precision (>50% impact)."""

    MINOR = "minor"
    """Minor impact on metrics (<50% impact)."""


@dataclass
class DetectedProblem:
    """A single detected problem in an LLM-generated query.

    Captures what went wrong, how severe it is, and provides
    evidence for debugging and analysis.
    """

    category: ProblemCategory
    """The type of problem detected."""

    severity: Severity
    """How severe the problem is."""

    description: str
    """Human-readable description of the problem."""

    evidence: dict[str, Any] = field(default_factory=dict)
    """Context: expected vs actual values, affected concepts, etc."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "description": self.description,
            "evidence": self.evidence,
        }


@dataclass
class ProblemAnalysisResult:
    """Complete problem analysis for a single trace.

    Aggregates all detected problems and provides summary flags
    for easy filtering and grouping.
    """

    trace_key: str
    """Unique identifier for the trace (e.g., 'A01_agentic_grep_abc123')."""

    query_id: str
    """Query identifier (e.g., 'A01', 'C06')."""

    query_set: str
    """Query set (A, B, C, D, or E)."""

    dataset: str
    """Primary dataset (EDU, TRN, NRG, BSBM)."""

    ambiguity_type: AmbiguityType
    """Parsed ambiguity type from query notes."""

    problems: list[DetectedProblem] = field(default_factory=list)
    """List of all detected problems."""

    missing_concepts: list[str] = field(default_factory=list)
    """List of semantic_concepts that were not found in LLM results."""

    # Aggregate flags for easy filtering
    has_concept_failure: bool = False
    """True if any concept-related problem detected."""

    has_measurement_failure: bool = False
    """True if measurement columns are missing."""

    has_filter_failure: bool = False
    """True if filter-related problem detected."""

    has_aggregation_failure: bool = False
    """True if aggregation-related problem detected."""

    has_instance_failure: bool = False
    """True if instance matching problem detected."""

    has_cross_dataset_failure: bool = False
    """True if cross-dataset problem detected."""

    has_empty_result: bool = False
    """True if query returned 0 rows."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trace_key": self.trace_key,
            "query_id": self.query_id,
            "query_set": self.query_set,
            "dataset": self.dataset,
            "ambiguity_type": self.ambiguity_type.value,
            "detected_problems": [p.category.value for p in self.problems],
            "problems_detail": [p.to_dict() for p in self.problems],
            "missing_concepts": self.missing_concepts,
            "problem_flags": {
                "has_concept_failure": self.has_concept_failure,
                "has_measurement_failure": self.has_measurement_failure,
                "has_filter_failure": self.has_filter_failure,
                "has_aggregation_failure": self.has_aggregation_failure,
                "has_instance_failure": self.has_instance_failure,
                "has_cross_dataset_failure": self.has_cross_dataset_failure,
                "has_empty_result": self.has_empty_result,
            },
        }


@dataclass
class CategoryStats:
    """Statistics for a category (query set, ambiguity type, dataset, etc.)."""

    name: str
    """Category name."""

    total: int = 0
    """Total number of traces in this category."""

    success_count: int = 0
    """Number of traces with F1 >= 0.8."""

    partial_count: int = 0
    """Number of traces with 0 < F1 < 0.8."""

    failure_count: int = 0
    """Number of traces with F1 == 0."""

    avg_f1: float = 0.0
    """Average F1 score."""

    avg_recall: float = 0.0
    """Average recall."""

    avg_precision: float = 0.0
    """Average precision."""

    def success_rate(self) -> float:
        """Calculate success rate (F1 >= 0.8)."""
        return self.success_count / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "total": self.total,
            "success_count": self.success_count,
            "partial_count": self.partial_count,
            "failure_count": self.failure_count,
            "success_rate": round(self.success_rate(), 3),
            "avg_f1": round(self.avg_f1, 3),
            "avg_recall": round(self.avg_recall, 3),
            "avg_precision": round(self.avg_precision, 3),
        }


@dataclass
class ProblemStats:
    """Statistics for a specific problem category."""

    problem: ProblemCategory
    """The problem category."""

    occurrence_count: int = 0
    """Number of times this problem occurred."""

    affected_queries: list[str] = field(default_factory=list)
    """List of query_ids affected by this problem."""

    affected_traces: list[str] = field(default_factory=list)
    """List of trace_keys affected by this problem."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "problem": self.problem.value,
            "occurrence_count": self.occurrence_count,
            "affected_queries": sorted(set(self.affected_queries)),
            "affected_trace_count": len(self.affected_traces),
        }


@dataclass
class CorrelationEntry:
    """Correlation between ambiguity type and problem category."""

    ambiguity_type: AmbiguityType
    problem_category: ProblemCategory
    co_occurrence_count: int
    """Number of traces where both ambiguity and problem occur."""

    conditional_probability: float
    """P(problem | ambiguity) - probability of problem given ambiguity type."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ambiguity_type": self.ambiguity_type.value,
            "problem_category": self.problem_category.value,
            "co_occurrence_count": self.co_occurrence_count,
            "conditional_probability": round(self.conditional_probability, 3),
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report with problem categorization.

    Provides multi-dimensional analysis of query failures
    for identifying patterns and improvement opportunities.
    """

    experiment_name: str
    timestamp: str

    total_traces: int
    overall_success_rate: float
    overall_avg_f1: float

    # Per-category statistics
    by_query_set: dict[str, CategoryStats] = field(default_factory=dict)
    by_ambiguity_type: dict[str, CategoryStats] = field(default_factory=dict)
    by_problem: dict[str, ProblemStats] = field(default_factory=dict)
    by_dataset: dict[str, CategoryStats] = field(default_factory=dict)

    # Correlation matrix
    correlations: list[CorrelationEntry] = field(default_factory=list)

    # Insights
    insights: list[str] = field(default_factory=list)
    hardest_ambiguity_types: list[tuple[str, float]] = field(default_factory=list)
    most_common_problems: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "total_traces": self.total_traces,
            "overall_success_rate": round(self.overall_success_rate, 3),
            "overall_avg_f1": round(self.overall_avg_f1, 3),
            "by_query_set": {k: v.to_dict() for k, v in self.by_query_set.items()},
            "by_ambiguity_type": {k: v.to_dict() for k, v in self.by_ambiguity_type.items()},
            "by_problem": {k: v.to_dict() for k, v in self.by_problem.items()},
            "by_dataset": {k: v.to_dict() for k, v in self.by_dataset.items()},
            "correlations": [c.to_dict() for c in self.correlations],
            "insights": self.insights,
            "hardest_ambiguity_types": [
                {"type": t, "avg_f1": round(f, 3)} for t, f in self.hardest_ambiguity_types
            ],
            "most_common_problems": [
                {"problem": p, "count": c} for p, c in self.most_common_problems
            ],
        }


def parse_ambiguity_type(notes: str, query_set: str) -> AmbiguityType:
    """Parse ambiguity type from query notes field.

    Args:
        notes: The notes field from AdaptiveGroundTruth
        query_set: The query set (A, B, C, D, E)

    Returns:
        The detected AmbiguityType
    """
    # SET A and B are baseline - no ambiguity
    if query_set in ("A", "B"):
        return AmbiguityType.NONE

    # SET D is always cross-dataset
    if query_set == "D":
        return AmbiguityType.CROSS_DATASET

    notes_lower = notes.lower()

    # Instance Matching (SET C)
    if "case variation" in notes_lower:
        return AmbiguityType.CASE_VARIATION
    if "partial match" in notes_lower:
        return AmbiguityType.PARTIAL_MATCH
    if "synonym" in notes_lower:
        return AmbiguityType.SYNONYM
    if "temporal" in notes_lower:
        return AmbiguityType.TEMPORAL
    if "fuzzy" in notes_lower or "typo" in notes_lower:
        return AmbiguityType.FUZZY_TYPO
    if "abbreviation" in notes_lower:
        return AmbiguityType.ABBREVIATION
    if "numeric filter" in notes_lower:
        return AmbiguityType.NUMERIC_FILTER
    if "multi-step" in notes_lower:
        return AmbiguityType.MULTI_STEP

    # Semantic Variance (SET E)
    if "semantic variance" in notes_lower:
        return AmbiguityType.SEMANTIC_VARIANCE
    if "ambiguity" in notes_lower:
        # Check for specific types
        if "role" in notes_lower or "owns" in notes_lower or "responsible" in notes_lower:
            return AmbiguityType.ROLE_AMBIGUITY
        if "active" in notes_lower:
            return AmbiguityType.ACTIVITY_AMBIGUITY
        return AmbiguityType.UNDERSPECIFIED
    if "underspecified" in notes_lower:
        return AmbiguityType.UNDERSPECIFIED

    # Default based on query set
    if query_set == "C":
        return AmbiguityType.PARTIAL_MATCH  # Default for SET C
    if query_set == "E":
        return AmbiguityType.SEMANTIC_VARIANCE  # Default for SET E

    return AmbiguityType.NONE
