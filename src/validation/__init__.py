"""Semantic SPARQL validation package.

This package provides semantic validation for SPARQL queries against
combined schema graphs built from class semantic models and OWL ontologies.
"""

from src.validation.schema_graph import (
    SchemaGraph,
    load_schema_graph,
    get_combined_schema,
)
from src.validation.pattern_extractor import (
    TriplePattern,
    ExtractedPatterns,
    extract_query_patterns,
)
from src.validation.semantic_validator import (
    ValidationIssue,
    SemanticValidationResult,
    validate_sparql_semantics,
)

__all__ = [
    # Schema graph
    "SchemaGraph",
    "load_schema_graph",
    "get_combined_schema",
    # Pattern extraction
    "TriplePattern",
    "ExtractedPatterns",
    "extract_query_patterns",
    # Validation
    "ValidationIssue",
    "SemanticValidationResult",
    "validate_sparql_semantics",
]
