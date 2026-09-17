"""Retrieval Ground Truth extraction from GT SPARQL queries.

Parses GT SPARQL queries and extracts the classes, properties, and data
instances the retrieval agent SHOULD have found. Elements from OPTIONAL
blocks are marked ACCEPTABLE (no penalty if missing). Elements from MINUS
blocks are REQUIRED (central to the query semantics).

The algebra traversal tracks context (optional vs required) and resolves
variable types for FILTER IN and instance extraction.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rdflib import Variable
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.term import BNode, Literal, URIRef

from src.validation.schema_graph import COMMON_PREFIXES

logger = logging.getLogger(__name__)

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# Namespaces that indicate schema/ontology URIs (not instance URIs)
_SCHEMA_NAMESPACES = set(COMMON_PREFIXES.values())


class RetrievalNecessity(str, Enum):
    """Whether a schema element is required or acceptable (forgivable if missing)."""

    REQUIRED = "required"
    ACCEPTABLE = "acceptable"


@dataclass
class InstanceReference:
    """A data instance referenced in the GT SPARQL query.

    Represents a concrete value (URI, string, number, time) that the
    retrieval agent should extract from the NL query and connect to
    the correct class and property.
    """

    value: str  # "TROLL", "http://www.university0.edu", "3000"
    value_type: str  # "uri", "string", "numeric", "temporal"
    associated_class: str | None = None  # Class URI (e.g., eno:Field)
    associated_property: str | None = None  # Property URI (e.g., eno:name)
    necessity: RetrievalNecessity = RetrievalNecessity.REQUIRED

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "value_type": self.value_type,
            "associated_class": self.associated_class,
            "associated_property": self.associated_property,
            "necessity": self.necessity.value,
        }


@dataclass
class RetrievalGroundTruth:
    """Ground truth schema elements and instances for retrieval evaluation."""

    classes: dict[str, RetrievalNecessity] = field(default_factory=dict)
    properties: dict[str, RetrievalNecessity] = field(default_factory=dict)
    instances: list[InstanceReference] = field(default_factory=list)
    query_id: str = ""
    parse_error: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.parse_error is None

    @property
    def required_classes(self) -> set[str]:
        return {c for c, n in self.classes.items() if n == RetrievalNecessity.REQUIRED}

    @property
    def acceptable_classes(self) -> set[str]:
        return {c for c, n in self.classes.items() if n == RetrievalNecessity.ACCEPTABLE}

    @property
    def required_properties(self) -> set[str]:
        return {p for p, n in self.properties.items() if n == RetrievalNecessity.REQUIRED}

    @property
    def acceptable_properties(self) -> set[str]:
        return {p for p, n in self.properties.items() if n == RetrievalNecessity.ACCEPTABLE}

    @property
    def all_classes(self) -> set[str]:
        return set(self.classes.keys())

    @property
    def all_properties(self) -> set[str]:
        return set(self.properties.keys())

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "classes": {uri: n.value for uri, n in self.classes.items()},
            "properties": {uri: n.value for uri, n in self.properties.items()},
            "instances": [inst.to_dict() for inst in self.instances],
            "parse_error": self.parse_error,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _add_element(
    target: dict[str, RetrievalNecessity],
    uri: str,
    necessity: RetrievalNecessity,
) -> None:
    """Add element, never downgrading REQUIRED to ACCEPTABLE."""
    if uri in target and target[uri] == RetrievalNecessity.REQUIRED:
        return
    target[uri] = necessity


def _is_schema_namespace(uri: str) -> bool:
    """Check if a URI belongs to a known schema/ontology namespace."""
    for ns in _SCHEMA_NAMESPACES:
        if uri.startswith(ns):
            return True
    return False


def _classify_literal(lit: Literal) -> str:
    """Classify a literal's value type."""
    dt = str(lit.datatype) if lit.datatype else ""
    if "integer" in dt or "decimal" in dt or "float" in dt or "double" in dt:
        return "numeric"
    val = str(lit)
    # Time pattern: HH:MM:SS
    if re.match(r"^\d{2}:\d{2}:\d{2}", val):
        return "temporal"
    # Date pattern: YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}", val):
        return "temporal"
    return "string"


def _resolve_variable_context(
    var_name: str,
    bgp_triples: list[tuple[Any, Any, Any]],
    prefixes: dict[str, str],
    type_map: dict[str, set[str]],
) -> tuple[str | None, str | None]:
    """Resolve a variable to its associated class and property.

    Given a variable like ?name, look through BGP patterns to find
    a triple like ?field eno:name ?name, and resolve:
    - associated_class: type of ?field (from type_map)
    - associated_property: eno:name

    Returns:
        (associated_class_uri, associated_property_uri)
    """
    for s, p, o in bgp_triples:
        p_str = str(p) if isinstance(p, URIRef) else None
        if p_str == RDF_TYPE:
            continue

        # Check if our variable is the object of this triple
        if isinstance(o, Variable) and f"?{o}" == var_name:
            # The subject's type is the associated class
            if isinstance(s, Variable):
                subj_var = f"?{s}"
                classes = type_map.get(subj_var, set())
                assoc_class = next(iter(classes), None) if classes else None
            elif isinstance(s, URIRef):
                assoc_class = str(s)
            else:
                assoc_class = None
            return assoc_class, p_str

        # Check if our variable is the subject
        if isinstance(s, Variable) and f"?{s}" == var_name:
            classes = type_map.get(var_name, set())
            assoc_class = next(iter(classes), None) if classes else None
            return assoc_class, p_str

    return None, None


# ---------------------------------------------------------------------------
# Filter expression traversal for instance extraction
# ---------------------------------------------------------------------------


def _extract_filter_instances(
    expr: Any,
    bgp_triples: list[tuple[Any, Any, Any]],
    prefixes: dict[str, str],
    type_map: dict[str, set[str]],
    necessity: RetrievalNecessity,
) -> list[InstanceReference]:
    """Extract data instances from a FILTER expression."""
    instances: list[InstanceReference] = []

    if not isinstance(expr, CompValue):
        return instances

    if expr.name == "RelationalExpression":
        inner_expr = expr.get("expr")
        other = expr.get("other")

        # Handle: ?var op literal  or  BUILTIN(?var) op literal
        if isinstance(other, Literal):
            var_name = _get_variable_from_expr(inner_expr)
            if var_name:
                assoc_class, assoc_prop = _resolve_variable_context(
                    var_name, bgp_triples, prefixes, type_map
                )
                vtype = _classify_literal(other)
                instances.append(
                    InstanceReference(
                        value=str(other),
                        value_type=vtype,
                        associated_class=assoc_class,
                        associated_property=assoc_prop,
                        necessity=necessity,
                    )
                )

        # Handle: literal op ?var  (less common)
        elif isinstance(inner_expr, Literal) and not isinstance(other, Literal):
            var_name = _get_variable_from_expr(other)
            if var_name:
                assoc_class, assoc_prop = _resolve_variable_context(
                    var_name, bgp_triples, prefixes, type_map
                )
                vtype = _classify_literal(inner_expr)
                instances.append(
                    InstanceReference(
                        value=str(inner_expr),
                        value_type=vtype,
                        associated_class=assoc_class,
                        associated_property=assoc_prop,
                        necessity=necessity,
                    )
                )

        # Handle: FILTER(?var IN (val1, val2, ...)) - already handled for classes
        # but also extract non-class URIs as instances
        if expr.get("op") == "IN" and isinstance(other, list):
            var_name = _get_variable_from_expr(inner_expr)
            for item in other:
                if isinstance(item, Literal):
                    if var_name:
                        assoc_class, assoc_prop = _resolve_variable_context(
                            var_name, bgp_triples, prefixes, type_map
                        )
                    else:
                        assoc_class, assoc_prop = None, None
                    vtype = _classify_literal(item)
                    instances.append(
                        InstanceReference(
                            value=str(item),
                            value_type=vtype,
                            associated_class=assoc_class,
                            associated_property=assoc_prop,
                            necessity=necessity,
                        )
                    )

    elif expr.name == "Builtin_REGEX":
        text_var = expr.get("text")
        pattern = expr.get("pattern")
        if isinstance(pattern, Literal):
            var_name = _get_variable_from_expr(text_var)
            assoc_class, assoc_prop = None, None
            if var_name:
                assoc_class, assoc_prop = _resolve_variable_context(
                    var_name, bgp_triples, prefixes, type_map
                )
            instances.append(
                InstanceReference(
                    value=str(pattern),
                    value_type="string",
                    associated_class=assoc_class,
                    associated_property=assoc_prop,
                    necessity=necessity,
                )
            )

    elif expr.name == "ConditionalAndExpression":
        # AND expression: recurse into expr and other list
        inner = expr.get("expr")
        if isinstance(inner, CompValue):
            instances.extend(
                _extract_filter_instances(inner, bgp_triples, prefixes, type_map, necessity)
            )
        other_list = expr.get("other", [])
        if isinstance(other_list, list):
            for item in other_list:
                if isinstance(item, CompValue):
                    instances.extend(
                        _extract_filter_instances(item, bgp_triples, prefixes, type_map, necessity)
                    )

    elif expr.name == "ConditionalOrExpression":
        inner = expr.get("expr")
        if isinstance(inner, CompValue):
            instances.extend(
                _extract_filter_instances(inner, bgp_triples, prefixes, type_map, necessity)
            )
        other_list = expr.get("other", [])
        if isinstance(other_list, list):
            for item in other_list:
                if isinstance(item, CompValue):
                    instances.extend(
                        _extract_filter_instances(item, bgp_triples, prefixes, type_map, necessity)
                    )

    return instances


def _get_variable_from_expr(expr: Any) -> str | None:
    """Extract variable name from an expression (handles builtins like UCASE)."""
    if isinstance(expr, Variable):
        return f"?{expr}"
    if isinstance(expr, CompValue):
        # Builtins like UCASE, LCASE, STR have an 'arg' field
        arg = expr.get("arg")
        if isinstance(arg, Variable):
            return f"?{arg}"
        # Some builtins have 'text' field
        text = expr.get("text")
        if isinstance(text, Variable):
            return f"?{text}"
    return None


# ---------------------------------------------------------------------------
# Core algebra traversal
# ---------------------------------------------------------------------------


def _collect_all_bgp_triples(
    node: Any,
    visited: set[int],
) -> list[tuple[Any, Any, Any]]:
    """Collect ALL BGP triples from the entire algebra tree (for variable resolution)."""
    triples = []
    if not isinstance(node, CompValue) or id(node) in visited:
        return triples
    visited.add(id(node))

    if node.name == "BGP" and "triples" in node:
        for triple in node["triples"]:
            if len(triple) == 3:
                triples.append(triple)

    for child in node.values():
        if isinstance(child, CompValue):
            triples.extend(_collect_all_bgp_triples(child, visited))
        elif isinstance(child, (list, tuple)):
            for item in child:
                if isinstance(item, CompValue):
                    triples.extend(_collect_all_bgp_triples(item, visited))

    return triples


def _build_type_map(bgp_triples: list[tuple[Any, Any, Any]]) -> dict[str, set[str]]:
    """Build variable -> set of class URIs from rdf:type patterns."""
    type_map: dict[str, set[str]] = {}
    for s, p, o in bgp_triples:
        if isinstance(p, URIRef) and str(p) == RDF_TYPE and isinstance(s, Variable):
            var_name = f"?{s}"
            if var_name not in type_map:
                type_map[var_name] = set()
            if isinstance(o, URIRef):
                type_map[var_name].add(str(o))
            elif isinstance(o, Variable):
                # Variable type (like ?courseType) - tracked for FILTER IN
                type_map[var_name] = set()  # Will be filled by FILTER IN
    return type_map


def _traverse_algebra(
    node: CompValue,
    classes: dict[str, RetrievalNecessity],
    properties: dict[str, RetrievalNecessity],
    instances: list[InstanceReference],
    prefixes: dict[str, str],
    is_optional: bool,
    all_bgp_triples: list[tuple[Any, Any, Any]],
    type_map: dict[str, set[str]],
    type_variables: dict[str, str],
    visited: set[int],
) -> None:
    """Recursively traverse SPARQL algebra extracting classes, properties, instances.

    Args:
        node: Current algebra node
        classes: Accumulator for class URIs -> necessity
        properties: Accumulator for property URIs -> necessity
        instances: Accumulator for instance references
        prefixes: PREFIX namespace mappings
        is_optional: True if inside OPTIONAL block (makes elements ACCEPTABLE)
        all_bgp_triples: All BGP triples from entire tree (for variable resolution)
        type_map: Variable -> class URIs from rdf:type patterns
        type_variables: Variable -> variable mapping for ?x rdf:type ?typeVar
        visited: Visited node IDs to prevent cycles
    """
    if not isinstance(node, CompValue) or id(node) in visited:
        return
    visited.add(id(node))

    necessity = RetrievalNecessity.ACCEPTABLE if is_optional else RetrievalNecessity.REQUIRED

    if node.name == "BGP" and "triples" in node:
        for triple in node["triples"]:
            if len(triple) != 3:
                continue
            s, p, o = triple

            if isinstance(p, URIRef):
                p_str = str(p)
                if p_str == RDF_TYPE:
                    # rdf:type → object is a class
                    if isinstance(o, URIRef):
                        _add_element(classes, str(o), necessity)
                    elif isinstance(o, Variable):
                        # Track: ?x rdf:type ?typeVar  (for FILTER IN resolution)
                        type_variables[f"?{o}"] = f"?{s}"
                else:
                    # Non-type predicate → property
                    _add_element(properties, p_str, necessity)

                    # Check for instance URIs in subject/object
                    if isinstance(s, URIRef) and not _is_schema_namespace(str(s)):
                        instances.append(
                            InstanceReference(
                                value=str(s),
                                value_type="uri",
                                associated_class=None,
                                associated_property=p_str,
                                necessity=necessity,
                            )
                        )
                    if isinstance(o, URIRef) and not _is_schema_namespace(str(o)):
                        instances.append(
                            InstanceReference(
                                value=str(o),
                                value_type="uri",
                                associated_class=None,
                                associated_property=p_str,
                                necessity=necessity,
                            )
                        )
                    # Literal objects in BGP are also data instances
                    # e.g., ?uni eduo:name "University0"
                    if isinstance(o, Literal):
                        subj_class = None
                        if isinstance(s, Variable):
                            subj_var = f"?{s}"
                            cls_set = type_map.get(subj_var, set())
                            subj_class = next(iter(cls_set), None) if cls_set else None
                        instances.append(
                            InstanceReference(
                                value=str(o),
                                value_type=_classify_literal(o),
                                associated_class=subj_class,
                                associated_property=p_str,
                                necessity=necessity,
                            )
                        )

    elif node.name == "LeftJoin":
        # OPTIONAL: p1 = required, p2 = optional
        p1 = node.get("p1")
        p2 = node.get("p2")
        if isinstance(p1, CompValue):
            _traverse_algebra(
                p1, classes, properties, instances, prefixes,
                is_optional, all_bgp_triples, type_map, type_variables, visited,
            )
        if isinstance(p2, CompValue):
            _traverse_algebra(
                p2, classes, properties, instances, prefixes,
                True, all_bgp_triples, type_map, type_variables, visited,
            )
        return  # Don't recurse into children again

    elif node.name == "Minus":
        # MINUS: both p1 and p2 are REQUIRED (MINUS is central to query semantics)
        p1 = node.get("p1")
        p2 = node.get("p2")
        if isinstance(p1, CompValue):
            _traverse_algebra(
                p1, classes, properties, instances, prefixes,
                is_optional, all_bgp_triples, type_map, type_variables, visited,
            )
        if isinstance(p2, CompValue):
            _traverse_algebra(
                p2, classes, properties, instances, prefixes,
                is_optional, all_bgp_triples, type_map, type_variables, visited,
            )
        return

    elif node.name in ("Union", "Join"):
        p1 = node.get("p1")
        p2 = node.get("p2")
        if isinstance(p1, CompValue):
            _traverse_algebra(
                p1, classes, properties, instances, prefixes,
                is_optional, all_bgp_triples, type_map, type_variables, visited,
            )
        if isinstance(p2, CompValue):
            _traverse_algebra(
                p2, classes, properties, instances, prefixes,
                is_optional, all_bgp_triples, type_map, type_variables, visited,
            )
        return

    elif node.name == "Filter":
        # Recurse into the pattern
        p = node.get("p")
        if isinstance(p, CompValue):
            _traverse_algebra(
                p, classes, properties, instances, prefixes,
                is_optional, all_bgp_triples, type_map, type_variables, visited,
            )

        # Extract instances and classes from FILTER expression
        expr = node.get("expr")
        if isinstance(expr, CompValue):
            # Check for FILTER(?typeVar IN (Class1, Class2, ...))
            _extract_filter_classes(expr, classes, type_variables, necessity)

            # Extract data instances from FILTER
            filter_instances = _extract_filter_instances(
                expr, all_bgp_triples, prefixes, type_map, necessity
            )
            instances.extend(filter_instances)
        return

    # Default: recurse into all CompValue children
    for child in node.values():
        if isinstance(child, CompValue):
            _traverse_algebra(
                child, classes, properties, instances, prefixes,
                is_optional, all_bgp_triples, type_map, type_variables, visited,
            )
        elif isinstance(child, (list, tuple)):
            for item in child:
                if isinstance(item, CompValue):
                    _traverse_algebra(
                        item, classes, properties, instances, prefixes,
                        is_optional, all_bgp_triples, type_map, type_variables, visited,
                    )


def _extract_filter_classes(
    expr: CompValue,
    classes: dict[str, RetrievalNecessity],
    type_variables: dict[str, str],
    necessity: RetrievalNecessity,
) -> None:
    """Extract classes from FILTER(?typeVar IN (Class1, Class2, ...)) expressions."""
    if not isinstance(expr, CompValue):
        return

    if expr.name == "RelationalExpression" and expr.get("op") == "IN":
        inner = expr.get("expr")
        other = expr.get("other")
        if isinstance(inner, Variable) and isinstance(other, list):
            var_name = f"?{inner}"
            if var_name in type_variables:
                for item in other:
                    if isinstance(item, URIRef):
                        _add_element(classes, str(item), necessity)


# ---------------------------------------------------------------------------
# Clean query helper (reused from pattern_extractor.py)
# ---------------------------------------------------------------------------


def _clean_query(query: str) -> str:
    """Clean SPARQL query for parsing."""
    query = query.replace("\u200B", "")
    query = re.sub(r"```(?:sparql)?\s*\n?", "", query)
    query = re.sub(r"\n?```", "", query)
    query = re.sub(r"SELECT\s+DISTINCT\?", "SELECT DISTINCT ?", query)
    query = re.sub(r"SELECT\?", "SELECT ?", query)
    query = re.sub(r"(\w+:[\w_]+)\?", r"\1 ?", query)
    return query.strip()


def _extract_query_prefixes(query: str) -> dict[str, str]:
    """Extract PREFIX declarations from query string."""
    prefixes = COMMON_PREFIXES.copy()
    prefix_pattern = re.compile(r"PREFIX\s+(\w+):\s*<([^>]+)>", re.IGNORECASE)
    for match in prefix_pattern.finditer(query):
        prefixes[match.group(1)] = match.group(2)
    return prefixes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_retrieval_ground_truth(
    sparql: str,
    query_id: str = "",
) -> RetrievalGroundTruth:
    """Extract retrieval ground truth from a GT SPARQL query.

    Parses the SPARQL algebra tree, tracking OPTIONAL/MINUS context.
    Elements in OPTIONAL blocks are labeled ACCEPTABLE.
    Elements in MINUS blocks are REQUIRED (central to query semantics).

    Classes: URIs from rdf:type patterns + FILTER(?var IN (...)) patterns
    Properties: Non-rdf:type predicate URIs
    Instances: URI constants, string/numeric/temporal literals from FILTER

    Args:
        sparql: Ground truth SPARQL query string
        query_id: Query ID for tracking

    Returns:
        RetrievalGroundTruth with classes, properties, and instances
    """
    gt = RetrievalGroundTruth(query_id=query_id)

    try:
        cleaned = _clean_query(sparql)
        prefixes = _extract_query_prefixes(cleaned)

        parsed = parseQuery(cleaned)
        algebra = translateQuery(parsed)

        # Collect all BGP triples for variable resolution
        all_bgp_triples = _collect_all_bgp_triples(algebra.algebra, set())
        type_map = _build_type_map(all_bgp_triples)
        type_variables: dict[str, str] = {}  # ?typeVar -> ?subjectVar

        # Traverse algebra with context tracking
        _traverse_algebra(
            algebra.algebra,
            gt.classes,
            gt.properties,
            gt.instances,
            prefixes,
            is_optional=False,
            all_bgp_triples=all_bgp_triples,
            type_map=type_map,
            type_variables=type_variables,
            visited=set(),
        )

    except Exception as e:
        gt.parse_error = str(e)
        logger.debug(f"Failed to parse query {query_id}: {e}")

    return gt


def extract_retrieval_gt_for_query(
    ground_truth: "AdaptiveGroundTruth",
) -> RetrievalGroundTruth:
    """Extract retrieval GT from an AdaptiveGroundTruth.

    If multiple SPARQL queries exist, extracts from each and merges.
    An element is REQUIRED if REQUIRED in ANY variant.
    """
    from data.queries.use_case_queries_tiered_tuples import AdaptiveGroundTruth

    merged = RetrievalGroundTruth(query_id=ground_truth.query_id)
    any_success = False

    for sparql in ground_truth.sparql_queries:
        gt = extract_retrieval_ground_truth(sparql, ground_truth.query_id)
        if gt.is_valid:
            any_success = True
            for uri, nec in gt.classes.items():
                _add_element(merged.classes, uri, nec)
            for uri, nec in gt.properties.items():
                _add_element(merged.properties, uri, nec)
            # Merge instances (deduplicate by value)
            existing_values = {inst.value for inst in merged.instances}
            for inst in gt.instances:
                if inst.value not in existing_values:
                    merged.instances.append(inst)
                    existing_values.add(inst.value)

    if not any_success:
        merged.parse_error = f"All {len(ground_truth.sparql_queries)} SPARQL variants failed to parse"

    return merged


# ---------------------------------------------------------------------------
# Schema-level retrieval ground truth
# ---------------------------------------------------------------------------


@dataclass
class RetrievalSchemaGT:
    """Schema-level ground truth: ideal retrieval as concrete schema triples.

    Built by resolving SPARQL query paths to actual semantic model triples
    from the SchemaGraph.  Each triple carries a necessity label (REQUIRED
    or ACCEPTABLE) inherited from the SPARQL structure.
    """

    # Each tuple: (subject_uri, predicate_uri, object_uri, necessity)
    triples: list[tuple[str, str, str, RetrievalNecessity]] = field(default_factory=list)
    # Deduplicated (domain_class_or_None, property_uri, necessity) paths
    paths: list[tuple[str | None, str, RetrievalNecessity]] = field(default_factory=list)
    query_id: str = ""
    build_warnings: list[str] = field(default_factory=list)

    def to_turtle(self, prefixes: dict[str, str] | None = None) -> str:
        """Serialize GT triples as a valid Turtle string."""
        if not self.triples:
            return ""

        # Collect used namespaces
        all_uris = set()
        for s, p, o, _n in self.triples:
            all_uris.update([s, p, o])

        # Build prefix block
        prefix_lines = []
        if prefixes:
            for pfx, ns in sorted(prefixes.items()):
                if any(uri.startswith(ns) for uri in all_uris):
                    prefix_lines.append(f"@prefix {pfx}: <{ns}> .")

        # Build triple lines grouped by subject
        from collections import OrderedDict

        by_subject: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
        for s, p, o, _n in self.triples:
            by_subject.setdefault(s, []).append((p, o))

        def _shorten(uri: str) -> str:
            if not prefixes:
                return f"<{uri}>"
            for pfx, ns in prefixes.items():
                if uri.startswith(ns):
                    return f"{pfx}:{uri[len(ns):]}"
            return f"<{uri}>"

        triple_lines = []
        for subj, po_pairs in by_subject.items():
            pairs_str = " ;\n    ".join(f"{_shorten(p)} {_shorten(o)}" for p, o in po_pairs)
            triple_lines.append(f"{_shorten(subj)} {pairs_str} .")

        parts = []
        if prefix_lines:
            parts.append("\n".join(prefix_lines))
        parts.append("\n\n".join(triple_lines))
        return "\n\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "triples": [
                {"subject": s, "predicate": p, "object": o, "necessity": n.value}
                for s, p, o, n in self.triples
            ],
            "paths": [
                {"domain_class": d, "property": p, "necessity": n.value}
                for d, p, n in self.paths
            ],
            "build_warnings": self.build_warnings,
        }


def build_schema_gt(
    sparql: str,
    dataset_ids: list[str],
    query_id: str = "",
) -> RetrievalSchemaGT:
    """Build schema-level retrieval GT from a GT SPARQL query.

    Resolves SPARQL triple-pattern paths to concrete schema triples from
    the semantic model files via the SchemaGraph.

    For each (domain_class, property) path in the SPARQL:
    - If domain_class is specified: look up the class (and its properties)
      in the SchemaGraph, get range(s) from class_property_ranges.
    - If domain_class is None: find all classes with the property via
      property_domains.
    - Necessity (REQUIRED/ACCEPTABLE) is inherited from the SPARQL structure.

    Args:
        sparql: Ground truth SPARQL query string
        dataset_ids: Dataset identifiers to load schema from
        query_id: For tracking

    Returns:
        RetrievalSchemaGT with concrete triples and paths
    """
    from src.evaluation.retrieval_metrics import extract_gt_paths
    from src.validation.schema_graph import SchemaGraph, get_combined_schema

    result = RetrievalSchemaGT(query_id=query_id)

    # Step 1: Load schema (also used to type untyped variables below)
    schema = get_combined_schema(dataset_ids)

    # Step 2: Extract paths and necessity info from SPARQL
    try:
        paths = extract_gt_paths(sparql, schema=schema)
    except Exception as e:
        result.build_warnings.append(f"Failed to extract paths: {e}")
        return result

    retrieval_gt = extract_retrieval_ground_truth(sparql, query_id)
    if not retrieval_gt.is_valid:
        result.build_warnings.append(f"Failed to parse SPARQL: {retrieval_gt.parse_error}")
        return result

    # Step 3: Build triples for each path
    seen_triples: set[tuple[str, str, str]] = set()

    for domain_class, prop_uri in paths:
        # Determine necessity for this property
        necessity = retrieval_gt.properties.get(prop_uri, RetrievalNecessity.REQUIRED)

        # Find candidate classes that have this property.
        #
        # Where the domain is unknown, an earlier version expanded to every
        # class carrying the property. On a large schema that is ruinous: NRG
        # defines `designation` on 113 classes, so one untyped variable turned a
        # four-triple ground truth into sixty, and an agent scored a hit for
        # retrieving any of them — including classes that denote entirely
        # different entities (a UnitFacility is not a Deposit; the two share no
        # instances). The GT therefore leaves the domain open instead, which the
        # metrics already handle: an open path only requires the property to
        # appear, and no spurious class-level triples are invented.
        candidate_classes: set[str] = set()
        if domain_class is not None:
            if prop_uri in schema.class_properties.get(domain_class, set()):
                candidate_classes.add(domain_class)
            else:
                result.build_warnings.append(
                    f"Property {prop_uri} not declared on class {domain_class}; "
                    f"leaving the domain open rather than expanding"
                )
        else:
            result.build_warnings.append(
                f"Untyped subject for property {prop_uri}; "
                f"leaving the domain open rather than expanding"
            )

        # Build triples from candidate classes
        for cls in sorted(candidate_classes):
            ranges = schema.class_property_ranges.get(cls, {}).get(prop_uri, set())
            if not ranges:
                # Fallback to global property_ranges
                global_range = schema.property_ranges.get(prop_uri)
                if global_range:
                    ranges = {global_range}

            for range_uri in sorted(ranges):
                triple_key = (cls, prop_uri, range_uri)
                if triple_key not in seen_triples:
                    seen_triples.add(triple_key)
                    result.triples.append((cls, prop_uri, range_uri, necessity))

        # Record the path with necessity
        result.paths.append((domain_class, prop_uri, necessity))

    return result


def build_schema_gt_for_query(
    ground_truth: "AdaptiveGroundTruth",
) -> RetrievalSchemaGT:
    """Build schema GT from an AdaptiveGroundTruth.

    If multiple SPARQL variants exist, builds from each and merges
    (union of triples, REQUIRED wins over ACCEPTABLE).
    """
    from data.queries.use_case_queries_tiered_tuples import AdaptiveGroundTruth

    dataset_ids = ground_truth.datasets or [ground_truth.dataset]
    merged = RetrievalSchemaGT(query_id=ground_truth.query_id)
    seen_triples: set[tuple[str, str, str]] = set()

    for sparql in ground_truth.sparql_queries:
        gt = build_schema_gt(sparql, dataset_ids, ground_truth.query_id)
        merged.build_warnings.extend(gt.build_warnings)

        for s, p, o, nec in gt.triples:
            key = (s, p, o)
            if key not in seen_triples:
                seen_triples.add(key)
                merged.triples.append((s, p, o, nec))
            else:
                # If already exists, upgrade to REQUIRED if this variant says REQUIRED
                if nec == RetrievalNecessity.REQUIRED:
                    for i, (es, ep, eo, en) in enumerate(merged.triples):
                        if (es, ep, eo) == key and en != RetrievalNecessity.REQUIRED:
                            merged.triples[i] = (es, ep, eo, RetrievalNecessity.REQUIRED)
                            break

        # Merge paths (deduplicate, REQUIRED wins)
        seen_paths = {(d, p) for d, p, _n in merged.paths}
        for d, p, nec in gt.paths:
            if (d, p) not in seen_paths:
                seen_paths.add((d, p))
                merged.paths.append((d, p, nec))

    return merged
