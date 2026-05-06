"""Semantic SPARQL query validator.

This module validates SPARQL queries against combined schema graphs,
checking for unknown properties/classes and domain/range compatibility.
"""

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

from src.validation.schema_graph import SchemaGraph, get_combined_schema
from src.validation.pattern_extractor import (
    ExtractedPatterns,
    TriplePattern,
    extract_query_patterns,
    get_all_variables,
)

logger = logging.getLogger(__name__)

# RDF type predicate
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


@dataclass
class ValidationIssue:
    """A single validation issue found in the query.

    Attributes:
        severity: ERROR (query definitely wrong) or WARNING (possible issue)
        code: Issue type code (e.g., UNKNOWN_PROPERTY, DOMAIN_MISMATCH)
        message: Human-readable description
        triple_pattern: The offending triple pattern as string
        suggestions: List of similar valid alternatives
    """

    severity: Literal["ERROR", "WARNING"]
    code: str
    message: str
    triple_pattern: str = ""
    suggestions: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        result = f"{self.severity} [{self.code}]: {self.message}"
        if self.triple_pattern:
            result += f"\n  Pattern: {self.triple_pattern}"
        if self.suggestions:
            result += f"\n  Suggestions: {', '.join(self.suggestions)}"
        return result


@dataclass
class SemanticValidationResult:
    """Result of semantic validation.

    Attributes:
        is_valid: True if no ERRORs were found (WARNINGs are OK)
        errors: List of ERROR-level issues
        warnings: List of WARNING-level issues
        inferred_types: Mapping of variable -> inferred types
        validated_patterns: Number of patterns checked
        parse_error: Error if query could not be parsed
    """

    is_valid: bool = True
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    inferred_types: dict[str, set[str]] = field(default_factory=dict)
    validated_patterns: int = 0
    parse_error: Optional[str] = None

    def add_error(self, issue: ValidationIssue) -> None:
        """Add an error and mark result as invalid."""
        self.errors.append(issue)
        self.is_valid = False

    def add_warning(self, issue: ValidationIssue) -> None:
        """Add a warning (doesn't affect is_valid)."""
        self.warnings.append(issue)

    @property
    def all_issues(self) -> list[ValidationIssue]:
        """Get all issues (errors + warnings)."""
        return self.errors + self.warnings

    def format_issues(self) -> str:
        """Format all issues as a readable string."""
        lines = []
        for issue in self.errors:
            lines.append(str(issue))
        for issue in self.warnings:
            lines.append(str(issue))
        return "\n".join(lines)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _get_local_name(uri: str) -> str:
    """Extract local name from URI (after # or last /)."""
    if "#" in uri:
        return uri.split("#")[-1]
    if "/" in uri:
        return uri.split("/")[-1]
    return uri


def _find_similar_uris(
    unknown_uri: str,
    known_uris: set[str],
    max_suggestions: int = 3,
    max_distance: int = 5,
) -> list[str]:
    """Find similar URIs using edit distance on local names."""
    unknown_local = _get_local_name(unknown_uri).lower()

    candidates = []
    for uri in known_uris:
        local = _get_local_name(uri).lower()
        distance = _levenshtein_distance(unknown_local, local)
        if distance <= max_distance:
            candidates.append((distance, uri))

    # Sort by distance and return top suggestions
    candidates.sort(key=lambda x: x[0])
    return [uri for _, uri in candidates[:max_suggestions]]


def _infer_types_from_patterns(
    patterns: ExtractedPatterns,
    schema: SchemaGraph,
) -> dict[str, set[str]]:
    """Infer types for variables from explicit constraints and property usage.

    Returns:
        Mapping of variable name -> set of possible types
    """
    inferred: dict[str, set[str]] = {}

    # Start with explicit type constraints
    for var, types in patterns.type_constraints.items():
        inferred[var] = types.copy()

    # Infer from property domains (if ?x uses property P, ?x might be in domain of P)
    for pattern in patterns.triple_patterns:
        if pattern.predicate == RDF_TYPE:
            continue

        # If subject is a variable and predicate has known domains
        if pattern.is_subject_var and pattern.predicate in schema.property_domains:
            var = pattern.subject
            if var not in inferred:
                inferred[var] = set()
            inferred[var].update(schema.property_domains[pattern.predicate])

        # If object is a variable and predicate has known range (and it's an object property)
        if pattern.is_object_var and pattern.predicate in schema.property_ranges:
            range_type = schema.property_ranges[pattern.predicate]
            # Only infer if range is a class (not XSD type)
            if not range_type.startswith("http://www.w3.org/2001/XMLSchema#"):
                var = pattern.object
                if var not in inferred:
                    inferred[var] = set()
                inferred[var].add(range_type)

    return inferred


def _validate_property_existence(
    pattern: TriplePattern,
    schema: SchemaGraph,
    result: SemanticValidationResult,
    include_suggestions: bool,
) -> bool:
    """Check if property exists in schema.

    Returns True if valid, False if error.
    """
    # Skip rdf:type - it's always valid
    if pattern.predicate == RDF_TYPE:
        return True

    # Skip variable predicates
    if pattern.is_predicate_var:
        return True

    if not schema.property_exists(pattern.predicate):
        suggestions = []
        if include_suggestions:
            suggestions = _find_similar_uris(pattern.predicate, schema.properties)

        result.add_error(ValidationIssue(
            severity="ERROR",
            code="UNKNOWN_PROPERTY",
            message=f"Property '{_get_local_name(pattern.predicate)}' does not exist in the schema.",
            triple_pattern=str(pattern),
            suggestions=suggestions,
        ))
        return False

    return True


def _validate_class_existence(
    pattern: TriplePattern,
    schema: SchemaGraph,
    result: SemanticValidationResult,
    include_suggestions: bool,
) -> bool:
    """Check if class in rdf:type triple exists.

    Returns True if valid, False if error.
    """
    if pattern.predicate != RDF_TYPE:
        return True

    # Skip if object is a variable
    if pattern.is_object_var:
        return True

    class_uri = pattern.object
    if not schema.class_exists(class_uri):
        suggestions = []
        if include_suggestions:
            suggestions = _find_similar_uris(class_uri, schema.classes)

        result.add_error(ValidationIssue(
            severity="ERROR",
            code="UNKNOWN_CLASS",
            message=f"Class '{_get_local_name(class_uri)}' does not exist in the schema.",
            triple_pattern=str(pattern),
            suggestions=suggestions,
        ))
        return False

    return True


def _validate_domain_compatibility(
    pattern: TriplePattern,
    inferred_types: dict[str, set[str]],
    schema: SchemaGraph,
    result: SemanticValidationResult,
) -> None:
    """Check if subject type is compatible with property domain."""
    # Skip special predicates
    if pattern.predicate == RDF_TYPE or pattern.is_predicate_var:
        return

    # Only check if subject is a variable with known type
    if not pattern.is_subject_var:
        return

    subject_types = inferred_types.get(pattern.subject, set())
    if not subject_types:
        # No type info - can't validate
        return

    # Check if any of the subject's types can have this property
    property_domains = schema.property_domains.get(pattern.predicate, set())
    if not property_domains:
        # No domain info - assume OK
        return

    # Check if any subject type (including parents) is in the domain
    compatible = False
    for subject_type in subject_types:
        all_types = schema.get_class_with_parents(subject_type)
        if all_types & property_domains:
            compatible = True
            break

    if not compatible:
        # Find which classes CAN use this property
        valid_classes = [_get_local_name(c) for c in property_domains][:3]

        result.add_warning(ValidationIssue(
            severity="WARNING",
            code="DOMAIN_MISMATCH",
            message=(
                f"Property '{_get_local_name(pattern.predicate)}' expects domain "
                f"{valid_classes}, but '{pattern.subject}' has type(s) "
                f"{[_get_local_name(t) for t in subject_types]}."
            ),
            triple_pattern=str(pattern),
            suggestions=[f"Add type constraint: {pattern.subject} a {c}" for c in valid_classes],
        ))


def _validate_class_property_compatibility(
    pattern: TriplePattern,
    inferred_types: dict[str, set[str]],
    schema: SchemaGraph,
    result: SemanticValidationResult,
    include_suggestions: bool,
) -> None:
    """Check if the subject's class can actually have this property.

    This is the critical subgraph validation - it ensures that the triple pattern
    forms a valid subgraph of the schema. A class can only have properties that
    are explicitly defined in its class semantic model.

    This catches cases where:
    - Property exists in schema but NOT for the given class
    - Class exists but the property belongs to a different class
    """
    # Skip rdf:type patterns
    if pattern.predicate == RDF_TYPE or pattern.is_predicate_var:
        return

    # Only check if subject is a variable with known type
    if not pattern.is_subject_var:
        return

    subject_types = inferred_types.get(pattern.subject, set())
    if not subject_types:
        # No type info - can't validate class-property compatibility
        return

    # Check if ANY of the subject's types can have this property
    can_have_property = False
    for subject_type in subject_types:
        if schema.can_class_have_property(subject_type, pattern.predicate):
            can_have_property = True
            break

    if not can_have_property:
        # Find which classes CAN have this property
        valid_classes = []
        for cls, props in schema.class_properties.items():
            if pattern.predicate in props:
                valid_classes.append(_get_local_name(cls))

        suggestions = []
        if include_suggestions and valid_classes:
            suggestions = [f"Use class: {c}" for c in valid_classes[:3]]

        subject_type_names = [_get_local_name(t) for t in subject_types]

        result.add_error(ValidationIssue(
            severity="ERROR",
            code="CLASS_PROPERTY_MISMATCH",
            message=(
                f"Class(es) {subject_type_names} cannot have property "
                f"'{_get_local_name(pattern.predicate)}'. "
                f"This property is only valid for: {valid_classes[:5] if valid_classes else 'unknown classes'}."
            ),
            triple_pattern=str(pattern),
            suggestions=suggestions,
        ))


def _validate_range_compatibility(
    pattern: TriplePattern,
    inferred_types: dict[str, set[str]],
    schema: SchemaGraph,
    result: SemanticValidationResult,
) -> None:
    """Check if object type is compatible with property range."""
    # Skip special predicates and variable predicates
    if pattern.predicate == RDF_TYPE or pattern.is_predicate_var:
        return

    expected_range = schema.get_property_range(pattern.predicate)
    if not expected_range:
        # No range info - assume OK
        return

    # If object is a literal, check against XSD types
    if not pattern.is_object_var and not pattern.object.startswith("http"):
        # It's a literal - check if range is a datatype
        if expected_range.startswith("http://www.w3.org/2001/XMLSchema#"):
            # OK - property expects datatype and we have a literal
            return
        else:
            # Property expects object but got literal
            result.add_warning(ValidationIssue(
                severity="WARNING",
                code="RANGE_MISMATCH",
                message=(
                    f"Property '{_get_local_name(pattern.predicate)}' expects "
                    f"object of type '{_get_local_name(expected_range)}', "
                    f"but got literal value."
                ),
                triple_pattern=str(pattern),
            ))
            return

    # If object is a variable with known type
    if pattern.is_object_var:
        object_types = inferred_types.get(pattern.object, set())
        if not object_types:
            # No type info - can't validate
            return

        # Check if any object type matches expected range
        if expected_range.startswith("http://www.w3.org/2001/XMLSchema#"):
            # Property expects datatype but we have object types
            result.add_warning(ValidationIssue(
                severity="WARNING",
                code="RANGE_MISMATCH",
                message=(
                    f"Property '{_get_local_name(pattern.predicate)}' expects "
                    f"datatype '{_get_local_name(expected_range)}', but "
                    f"'{pattern.object}' appears to be a resource."
                ),
                triple_pattern=str(pattern),
            ))
            return

        # Check if object type (or parents) matches expected range
        compatible = False
        for obj_type in object_types:
            all_types = schema.get_class_with_parents(obj_type)
            if expected_range in all_types:
                compatible = True
                break

        if not compatible:
            result.add_warning(ValidationIssue(
                severity="WARNING",
                code="RANGE_MISMATCH",
                message=(
                    f"Property '{_get_local_name(pattern.predicate)}' expects range "
                    f"'{_get_local_name(expected_range)}', but '{pattern.object}' "
                    f"has type(s) {[_get_local_name(t) for t in object_types]}."
                ),
                triple_pattern=str(pattern),
            ))


def validate_sparql_semantics(
    query: str,
    dataset_ids: list[str] | None = None,
    strict_mode: bool = False,
    include_suggestions: bool = True,
) -> SemanticValidationResult:
    """Validate SPARQL query semantics against combined schema graph.

    This function checks if the graph pattern in the WHERE clause is
    semantically compatible with the available schema. It detects:
    - Unknown properties (ERROR)
    - Unknown classes (ERROR)
    - Class-property mismatches (ERROR) - when a class cannot have a property
    - Domain mismatches (WARNING)
    - Range mismatches (WARNING)

    The class-property validation ensures that the query forms a valid subgraph
    of the schema - i.e., each triple pattern must correspond to a valid
    relationship defined in the class semantic models.

    Args:
        query: The SPARQL query to validate
        dataset_ids: Datasets to validate against (None = all datasets)
        strict_mode: If True, treat warnings as errors
        include_suggestions: If True, generate correction suggestions

    Returns:
        SemanticValidationResult with issues and suggestions
    """
    result = SemanticValidationResult()

    # Parse query
    patterns = extract_query_patterns(query)
    if not patterns.is_valid_parse:
        result.is_valid = False
        result.parse_error = patterns.parse_error
        return result

    # Load schema
    try:
        schema = get_combined_schema(dataset_ids)
    except Exception as e:
        result.is_valid = False
        result.parse_error = f"Failed to load schema: {e}"
        return result

    # Infer types for variables
    result.inferred_types = _infer_types_from_patterns(patterns, schema)

    # Validate each triple pattern
    for pattern in patterns.triple_patterns:
        result.validated_patterns += 1

        # Check property existence
        _validate_property_existence(pattern, schema, result, include_suggestions)

        # Check class existence (for rdf:type patterns)
        _validate_class_existence(pattern, schema, result, include_suggestions)

        # Check class-property compatibility (subgraph validation)
        # This ensures the triple pattern forms a valid subgraph of the schema
        _validate_class_property_compatibility(
            pattern, result.inferred_types, schema, result, include_suggestions
        )

        # Check domain compatibility
        _validate_domain_compatibility(pattern, result.inferred_types, schema, result)

        # Check range compatibility
        _validate_range_compatibility(pattern, result.inferred_types, schema, result)

    # In strict mode, warnings become errors
    if strict_mode and result.warnings:
        for warning in result.warnings:
            warning.severity = "ERROR"
        result.errors.extend(result.warnings)
        result.warnings = []
        result.is_valid = len(result.errors) == 0

    return result


def format_validation_for_llm(result: SemanticValidationResult) -> str:
    """Format validation result for LLM correction prompt.

    Returns a structured string suitable for including in an LLM prompt
    to help the model correct the query.
    """
    if result.is_valid and not result.warnings:
        return "Query is semantically valid."

    lines = ["Semantic validation found issues:\n"]

    for error in result.errors:
        lines.append(f"ERROR [{error.code}]: {error.message}")
        if error.triple_pattern:
            lines.append(f"  In pattern: {error.triple_pattern}")
        if error.suggestions:
            lines.append(f"  Did you mean: {', '.join(error.suggestions[:3])}")
        lines.append("")

    for warning in result.warnings:
        lines.append(f"WARNING [{warning.code}]: {warning.message}")
        if warning.triple_pattern:
            lines.append(f"  In pattern: {warning.triple_pattern}")
        if warning.suggestions:
            lines.append(f"  Suggestions: {', '.join(warning.suggestions[:3])}")
        lines.append("")

    if result.inferred_types:
        lines.append("Inferred variable types:")
        for var, types in result.inferred_types.items():
            type_names = [_get_local_name(t) for t in types]
            lines.append(f"  {var}: {', '.join(type_names)}")

    return "\n".join(lines)
