"""SPARQL query pattern extraction for semantic validation.

This module parses SPARQL queries and extracts triple patterns,
type constraints, and variable bindings for semantic validation.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from rdflib import Variable
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.term import URIRef, Literal, BNode

from src.validation.schema_graph import COMMON_PREFIXES

logger = logging.getLogger(__name__)


@dataclass
class TriplePattern:
    """A single triple pattern extracted from a SPARQL query.

    Attributes:
        subject: Variable name (e.g., "?x") or URI
        predicate: Predicate URI (fully expanded)
        object: Variable name, URI, or literal value
        is_subject_var: True if subject is a variable
        is_predicate_var: True if predicate is a variable
        is_object_var: True if object is a variable
        raw_subject: Original subject before prefix expansion
        raw_predicate: Original predicate before prefix expansion
        raw_object: Original object before prefix expansion
    """

    subject: str
    predicate: str
    object: str
    is_subject_var: bool = False
    is_predicate_var: bool = False
    is_object_var: bool = False
    raw_subject: str = ""
    raw_predicate: str = ""
    raw_object: str = ""

    def __str__(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"


@dataclass
class ExtractedPatterns:
    """Patterns extracted from a SPARQL query for semantic validation.

    Attributes:
        triple_patterns: List of triple patterns from WHERE clause
        type_constraints: Mapping of variable -> set of declared types
        prefixes: Mapping of prefix -> namespace URI
        select_variables: Variables in SELECT clause
        is_valid_parse: Whether parsing succeeded
        parse_error: Error message if parsing failed
    """

    triple_patterns: list[TriplePattern] = field(default_factory=list)
    type_constraints: dict[str, set[str]] = field(default_factory=dict)
    prefixes: dict[str, str] = field(default_factory=dict)
    select_variables: set[str] = field(default_factory=set)
    is_valid_parse: bool = True
    parse_error: Optional[str] = None

    def get_type_for_variable(self, var_name: str) -> Optional[set[str]]:
        """Get declared types for a variable."""
        # Normalize variable name
        if not var_name.startswith("?"):
            var_name = "?" + var_name
        return self.type_constraints.get(var_name)


def _clean_query(query: str) -> str:
    """Clean SPARQL query for parsing."""
    # Remove zero-width spaces
    query = query.replace("\u200B", "")

    # Remove markdown code blocks
    query = re.sub(r"```(?:sparql)?\s*\n?", "", query)
    query = re.sub(r"\n?```", "", query)

    # Fix common spacing issues
    query = re.sub(r"SELECT\s+DISTINCT\?", "SELECT DISTINCT ?", query)
    query = re.sub(r"SELECT\?", "SELECT ?", query)
    query = re.sub(r"(\w+:[\w_]+)\?", r"\1 ?", query)

    return query.strip()


def _term_to_string(term: Any, prefixes: dict[str, str]) -> tuple[str, bool, str]:
    """Convert an RDF term to string representation.

    Returns:
        Tuple of (expanded_uri, is_variable, raw_form)
    """
    if isinstance(term, Variable):
        var_name = f"?{term}"
        return var_name, True, var_name

    if isinstance(term, URIRef):
        uri_str = str(term)
        raw = uri_str
        # Try to create prefixed form for raw
        for prefix, namespace in prefixes.items():
            if uri_str.startswith(namespace):
                raw = f"{prefix}:{uri_str[len(namespace):]}"
                break
        return uri_str, False, raw

    if isinstance(term, Literal):
        return str(term), False, repr(term)

    if isinstance(term, BNode):
        return f"_:{term}", False, f"_:{term}"

    return str(term), False, str(term)


def _extract_bgp_triples(
    algebra: CompValue,
    patterns: list[TriplePattern],
    prefixes: dict[str, str],
    visited: set[int],
) -> None:
    """Recursively extract BGP triples from SPARQL algebra."""
    if not isinstance(algebra, CompValue) or id(algebra) in visited:
        return

    visited.add(id(algebra))

    # Handle BGP (Basic Graph Pattern)
    if algebra.name == "BGP" and "triples" in algebra:
        for triple in algebra["triples"]:
            if len(triple) == 3:
                subj, pred, obj = triple

                subj_str, is_subj_var, raw_subj = _term_to_string(subj, prefixes)
                pred_str, is_pred_var, raw_pred = _term_to_string(pred, prefixes)
                obj_str, is_obj_var, raw_obj = _term_to_string(obj, prefixes)

                pattern = TriplePattern(
                    subject=subj_str,
                    predicate=pred_str,
                    object=obj_str,
                    is_subject_var=is_subj_var,
                    is_predicate_var=is_pred_var,
                    is_object_var=is_obj_var,
                    raw_subject=raw_subj,
                    raw_predicate=raw_pred,
                    raw_object=raw_obj,
                )
                patterns.append(pattern)

    # Recurse into child structures
    for child in algebra.values():
        if isinstance(child, CompValue):
            _extract_bgp_triples(child, patterns, prefixes, visited)
        elif isinstance(child, (list, tuple)):
            for item in child:
                if isinstance(item, CompValue):
                    _extract_bgp_triples(item, patterns, prefixes, visited)


def _extract_type_constraints(patterns: list[TriplePattern]) -> dict[str, set[str]]:
    """Extract rdf:type constraints from triple patterns.

    Looks for patterns like: ?x rdf:type Class or ?x a Class
    """
    type_constraints: dict[str, set[str]] = {}

    RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

    for pattern in patterns:
        if pattern.predicate == RDF_TYPE and pattern.is_subject_var:
            var_name = pattern.subject
            if var_name not in type_constraints:
                type_constraints[var_name] = set()

            if not pattern.is_object_var:
                type_constraints[var_name].add(pattern.object)

    return type_constraints


def _extract_select_variables(algebra: CompValue) -> set[str]:
    """Extract variables from SELECT clause."""
    variables = set()

    def _find_project(comp: CompValue, visited: set[int]) -> None:
        if not isinstance(comp, CompValue) or id(comp) in visited:
            return
        visited.add(id(comp))

        if comp.name == "Project" and "PV" in comp:
            for var in comp["PV"]:
                variables.add(f"?{var}")

        for child in comp.values():
            if isinstance(child, CompValue):
                _find_project(child, visited)
            elif isinstance(child, (list, tuple)):
                for item in child:
                    if isinstance(item, CompValue):
                        _find_project(item, visited)

    _find_project(algebra, set())
    return variables


def _extract_query_prefixes(query: str) -> dict[str, str]:
    """Extract PREFIX declarations from query string."""
    prefixes = COMMON_PREFIXES.copy()

    prefix_pattern = re.compile(
        r"PREFIX\s+(\w+):\s*<([^>]+)>",
        re.IGNORECASE
    )

    for match in prefix_pattern.finditer(query):
        prefix = match.group(1)
        namespace = match.group(2)
        prefixes[prefix] = namespace

    return prefixes


def expand_prefixed_uri(uri: str, prefixes: dict[str, str]) -> str:
    """Expand a prefixed URI to full form.

    Args:
        uri: Prefixed URI (e.g., "trn:Trip") or full URI
        prefixes: Mapping of prefix -> namespace

    Returns:
        Fully expanded URI
    """
    if uri.startswith("http://") or uri.startswith("https://"):
        return uri

    if ":" in uri:
        parts = uri.split(":", 1)
        if len(parts) == 2:
            prefix, local = parts
            if prefix in prefixes:
                return prefixes[prefix] + local

    return uri


def extract_query_patterns(query: str) -> ExtractedPatterns:
    """Parse SPARQL query and extract patterns for validation.

    Args:
        query: SPARQL query string

    Returns:
        ExtractedPatterns with all relevant information
    """
    result = ExtractedPatterns()

    try:
        # Clean and prepare query
        cleaned_query = _clean_query(query)

        # Extract prefixes from query text
        result.prefixes = _extract_query_prefixes(cleaned_query)

        # Parse query
        parsed = parseQuery(cleaned_query)
        algebra = translateQuery(parsed)

        # Extract triple patterns
        _extract_bgp_triples(algebra.algebra, result.triple_patterns, result.prefixes, set())

        # Extract type constraints
        result.type_constraints = _extract_type_constraints(result.triple_patterns)

        # Extract SELECT variables
        result.select_variables = _extract_select_variables(algebra.algebra)

        result.is_valid_parse = True

    except Exception as e:
        result.is_valid_parse = False
        result.parse_error = str(e)
        logger.debug(f"Failed to parse query: {e}")

    return result


def get_all_variables(patterns: ExtractedPatterns) -> set[str]:
    """Get all variables used in the query patterns."""
    variables = set()

    for pattern in patterns.triple_patterns:
        if pattern.is_subject_var:
            variables.add(pattern.subject)
        if pattern.is_predicate_var:
            variables.add(pattern.predicate)
        if pattern.is_object_var:
            variables.add(pattern.object)

    return variables


def get_variable_positions(patterns: ExtractedPatterns) -> dict[str, list[str]]:
    """Get positions where each variable is used.

    Returns:
        Mapping of variable -> list of positions (e.g., ["subject", "object"])
    """
    positions: dict[str, list[str]] = {}

    for i, pattern in enumerate(patterns.triple_patterns):
        if pattern.is_subject_var:
            if pattern.subject not in positions:
                positions[pattern.subject] = []
            positions[pattern.subject].append(f"pattern_{i}_subject")

        if pattern.is_predicate_var:
            if pattern.predicate not in positions:
                positions[pattern.predicate] = []
            positions[pattern.predicate].append(f"pattern_{i}_predicate")

        if pattern.is_object_var:
            if pattern.object not in positions:
                positions[pattern.object] = []
            positions[pattern.object].append(f"pattern_{i}_object")

    return positions
