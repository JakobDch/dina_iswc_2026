"""Column Signature Matching for SPARQL result evaluation.

Core idea
---------
Each SELECT variable of a SPARQL query is described by a *signature* extracted
from the query's BGP / algebra: which predicates use the variable as subject
or object, which rdf:type classes apply, and — for projected expressions —
which aggregation chain produced the value.

Two variables (one from GT, one from LLM) are considered "the same column"
iff their signatures overlap on an ontology level. This replaces the previous
value-overlap heuristic in ``tiered_metrics_tuples.calculate_adaptive_metrics``
which can produce spurious cross-column matches when unrelated literal columns
happen to share a few numeric values (e.g. ``routeName="10"`` vs.
``stationCode="10"`` in BASE04).

Matching is agnostic to variable names and to result values. It operates
purely on the structural role a variable plays in its defining SPARQL graph.

Phase 1 scope
-------------
- URI-vs-literal distinction via "variable ever used as subject in BGP?"
- Direct rdf:type constraints
- Incoming / outgoing predicate sets (including OPTIONAL blocks)
- Aggregation detection via ``Extend`` nodes
- Greedy 1:1 matching with Jaccard-based similarity score

Deferred (not needed for BASE corpus):
- rdfs:range / rdfs:domain chase through external TBox
- Property-path expansion (tro:a/tro:b style)
- Equivalence between properties (rdfs:label ≈ foaf:name)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from rdflib import Variable
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.term import URIRef

logger = logging.getLogger(__name__)

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class ColumnSignature:
    """Structural description of one SELECT variable.

    Attributes:
        var_name: Variable name without leading "?" (e.g. "routeName").
        is_uri: True if the variable is ever used as subject of a BGP triple
            (strong indicator it binds to a URI/BNode, not a literal). False
            if it appears only as object of triples (likely literal, but could
            also be URI). None if variable is computed (aggregation) and
            doesn't appear in any BGP.
        direct_types: Class URIs from ``?var a <Class>`` patterns.
        incoming_predicates: Predicate URIs where var is object of triple
            ``?s <p> ?var``.
        outgoing_predicates: Predicate URIs where var is subject of triple
            ``?var <p> ?o``.
        is_aggregated: True if variable is introduced by ``(agg(...) AS ?var)``.
        aggregation_kind: Aggregation function name (SUM, COUNT, AVG, MIN,
            MAX, SAMPLE, GROUP_CONCAT) if known.
        aggregation_inner: Signature of the inner variable(s) being aggregated,
            for nested aggregations like SUM(COUNT(?x)). Simplified to a flat
            list of predicate URIs that the inner variable depends on.
        subject_context: Disambiguation context for literal columns that share
            the same producing property (e.g. ``eno:designation`` on both
            Deposit and Company). Contains direction-prefixed predicates of the
            *subject* variable(s) from the triple ``?subject <prop> ?this_var``.
            Format: ``{"out:<uri>", "in:<uri>", "type:<uri>", ...}``.
    """

    var_name: str
    is_uri: bool | None = None
    direct_types: set[str] = field(default_factory=set)
    incoming_predicates: set[str] = field(default_factory=set)
    outgoing_predicates: set[str] = field(default_factory=set)
    is_aggregated: bool = False
    aggregation_kind: str | None = None
    aggregation_inner: set[str] = field(default_factory=set)
    subject_context: set[str] = field(default_factory=set)

    @property
    def is_literal(self) -> bool:
        """Strong literal indicator: never used as subject, not aggregated."""
        return self.is_uri is False and not self.is_aggregated

    def to_dict(self) -> dict[str, Any]:
        return {
            "var_name": self.var_name,
            "is_uri": self.is_uri,
            "direct_types": sorted(self.direct_types),
            "incoming_predicates": sorted(self.incoming_predicates),
            "outgoing_predicates": sorted(self.outgoing_predicates),
            "is_aggregated": self.is_aggregated,
            "aggregation_kind": self.aggregation_kind,
            "aggregation_inner": sorted(self.aggregation_inner),
            "subject_context": sorted(self.subject_context),
        }

    def is_empty(self) -> bool:
        """True if the signature contains no discriminative information."""
        return (
            not self.direct_types
            and not self.incoming_predicates
            and not self.outgoing_predicates
            and not self.is_aggregated
        )


# =============================================================================
# Query parsing helpers
# =============================================================================


def _clean_query(query: str) -> str:
    """Minor cleanup so rdflib's parser accepts common LLM-emitted queries."""
    query = query.replace("\u200B", "")
    query = re.sub(r"```(?:sparql)?\s*\n?", "", query)
    query = re.sub(r"\n?```", "", query)
    query = re.sub(r"SELECT\s+DISTINCT\?", "SELECT DISTINCT ?", query)
    query = re.sub(r"SELECT\?", "SELECT ?", query)
    query = re.sub(r"(\w+:[\w_]+)\?", r"\1 ?", query)
    return query.strip()


def _collect_all_bgp_triples(
    node: Any,
    visited: set[int],
) -> list[tuple[Any, Any, Any]]:
    """Collect all BGP triples from the algebra tree, across OPTIONAL/UNION."""
    triples: list[tuple[Any, Any, Any]] = []
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


def _collect_projected_vars(algebra: Any) -> list[str]:
    """Extract names of variables appearing in the outermost SELECT clause.

    Returns names without leading "?" in SELECT order.
    """
    # Outer algebra is typically Project or SelectQuery wrapping Project.
    # rdflib's translated algebra exposes ``PV`` on Project nodes.
    def _find_project(node: Any) -> CompValue | None:
        if isinstance(node, CompValue):
            if node.name == "Project":
                return node
            for child in node.values():
                if isinstance(child, CompValue):
                    found = _find_project(child)
                    if found is not None:
                        return found
                elif isinstance(child, (list, tuple)):
                    for item in child:
                        if isinstance(item, CompValue):
                            found = _find_project(item)
                            if found is not None:
                                return found
        return None

    project = _find_project(algebra)
    if project is None or "PV" not in project:
        return []
    return [str(v) for v in project["PV"] if isinstance(v, Variable)]


def _nearest_aggregate_scope(
    node: Any,
    visited: set[int] | None = None,
) -> dict[str, CompValue]:
    """Return the aux-var → Aggregate map of the nearest enclosed AggregateJoin.

    Walks down the algebra from ``node`` but does NOT descend through
    ``ToMultiSet`` (which introduces a new query scope and its own
    ``__agg_N__`` naming, colliding with the outer scope).
    """
    if visited is None:
        visited = set()
    if not isinstance(node, CompValue) or id(node) in visited:
        return {}
    visited.add(id(node))

    if node.name == "AggregateJoin" and "A" in node:
        result: dict[str, CompValue] = {}
        for agg in node["A"]:
            if isinstance(agg, CompValue):
                res = agg.get("res")
                if isinstance(res, Variable):
                    result[str(res)] = agg
        return result

    # Don't cross subquery boundaries — inner __agg_N__ variables are separate.
    if node.name == "ToMultiSet":
        return {}

    for child in node.values():
        if isinstance(child, CompValue):
            found = _nearest_aggregate_scope(child, visited)
            if found:
                return found
        elif isinstance(child, (list, tuple)):
            for item in child:
                if isinstance(item, CompValue):
                    found = _nearest_aggregate_scope(item, visited)
                    if found:
                        return found

    return {}


def _collect_extend_assignments_with_scope(
    node: Any,
    assignments: dict[str, tuple[Any, dict[str, CompValue]]],
    visited: set[int],
) -> None:
    """Collect Extend var → (expr, aggregate_scope) mappings.

    For each ``Extend`` node, records the assigned expression together with
    the aggregate-scope map of the AggregateJoin that the Extend's child
    tree points to. This ensures that when the expr is an auxiliary
    ``__agg_N__`` variable, we resolve it against the *correct* scope even
    if the same name exists inside a nested subquery.
    """
    if not isinstance(node, CompValue) or id(node) in visited:
        return
    visited.add(id(node))

    if node.name == "Extend":
        var = node.get("var")
        expr = node.get("expr")
        if isinstance(var, Variable):
            child_p = node.get("p")
            scope = _nearest_aggregate_scope(child_p, set()) if child_p else {}
            assignments[str(var)] = (expr, scope)

    for child in node.values():
        if isinstance(child, CompValue):
            _collect_extend_assignments_with_scope(child, assignments, visited)
        elif isinstance(child, (list, tuple)):
            for item in child:
                if isinstance(item, CompValue):
                    _collect_extend_assignments_with_scope(item, assignments, visited)


def _resolve_aggregation(
    expr: Any,
    aggregates: dict[str, CompValue],
) -> tuple[str | None, set[str]]:
    """Resolve an Extend expression to (aggregation_kind, inner_var_names).

    If ``expr`` is a Variable referencing an aggregate's result, we look up
    the aggregate's function name and the variables it consumes. Returns
    (None, set()) if the expression is not an aggregation.
    """
    if isinstance(expr, Variable):
        agg = aggregates.get(str(expr))
        if agg is None:
            return None, set()
        # Aggregate node name is like "Aggregate_Count_", "Aggregate_Sum_", etc.
        name = agg.name or ""
        kind = None
        match = re.match(r"Aggregate_([A-Za-z_]+)_?$", name)
        if match:
            kind = match.group(1).upper().rstrip("_")
        inner_vars: set[str] = set()
        inner_expr = agg.get("vars")
        if isinstance(inner_expr, Variable):
            inner_vars.add(str(inner_expr))
        elif isinstance(inner_expr, CompValue):
            # Nested expression — walk it shallowly looking for Variables.
            for v in _collect_vars_from_expr(inner_expr):
                inner_vars.add(v)
        return kind, inner_vars

    if isinstance(expr, CompValue):
        # Aggregation could be wrapped, e.g. in a Builtin. Walk shallowly.
        for v in _collect_vars_from_expr(expr):
            agg = aggregates.get(v)
            if agg is not None:
                return _resolve_aggregation(Variable(v), aggregates)

    return None, set()


def _collect_vars_from_expr(expr: Any) -> list[str]:
    """Collect Variable names appearing anywhere inside an expression tree."""
    found: list[str] = []
    if isinstance(expr, Variable):
        found.append(str(expr))
        return found
    if isinstance(expr, CompValue):
        for child in expr.values():
            if isinstance(child, (Variable, CompValue)):
                found.extend(_collect_vars_from_expr(child))
            elif isinstance(child, (list, tuple)):
                for item in child:
                    found.extend(_collect_vars_from_expr(item))
    return found


# =============================================================================
# Signature extraction
# =============================================================================


def extract_column_signatures(sparql: str) -> dict[str, ColumnSignature]:
    """Extract a ColumnSignature per projected SELECT variable.

    Args:
        sparql: SPARQL query string.

    Returns:
        Mapping from variable name (without "?") to ColumnSignature. Empty
        dict if the query cannot be parsed.
    """
    try:
        cleaned = _clean_query(sparql)
        parsed = parseQuery(cleaned)
        algebra = translateQuery(parsed)
    except Exception as e:
        logger.debug("extract_column_signatures: parse failed: %s", e)
        return {}

    root = algebra.algebra
    projected = _collect_projected_vars(root)
    bgp_triples = _collect_all_bgp_triples(root, set())

    # Per-Extend scope: each (var -> (expr, local_aggregate_scope_map)).
    # Using scoped aggregates is essential because rdflib reuses ``__agg_N__``
    # inside subqueries, so a single global map would conflate the outer
    # ``(SUM(?x) AS ?y)`` with an inner ``(COUNT(?z) AS ?y)`` in a nested
    # SELECT.
    extend_assignments: dict[str, tuple[Any, dict[str, CompValue]]] = {}
    _collect_extend_assignments_with_scope(root, extend_assignments, set())

    signatures: dict[str, ColumnSignature] = {}
    for var_name in projected:
        sig = ColumnSignature(var_name=var_name)

        # Aggregation check: is this variable assigned via Extend?
        if var_name in extend_assignments:
            expr, scope = extend_assignments[var_name]
            kind, inner_vars = _resolve_aggregation(expr, scope)
            # rdflib wraps every GROUP BY variable as an implicit
            # ``Aggregate_Sample`` — ignore those, they carry no semantic
            # aggregation information and would otherwise mask the variable's
            # BGP-derived signature.
            if kind is not None and kind != "SAMPLE":
                sig.is_aggregated = True
                sig.aggregation_kind = kind
                # Inner variable's predicates contribute to the signature
                # so that SUM(COUNT(?stopTime)) and COUNT(?stopEvent) can
                # match when the inner variables have the same predicate set.
                for inner in inner_vars:
                    inner_preds = _predicates_of(inner, bgp_triples)
                    sig.aggregation_inner.update(inner_preds)
                    # If the aggregate's inner variable is also assigned via
                    # another Extend (nested subquery aggregation), recurse
                    # using that Extend's *own* aggregate scope.
                    if inner in extend_assignments:
                        inner_expr, inner_scope = extend_assignments[inner]
                        inner_kind, inner_inner = _resolve_aggregation(
                            inner_expr, inner_scope
                        )
                        if inner_kind is not None and inner_kind != "SAMPLE":
                            sig.aggregation_inner.add(f"_nested_{inner_kind}")
                            for ii in inner_inner:
                                sig.aggregation_inner.update(
                                    _predicates_of(ii, bgp_triples)
                                )

        # BGP-based signature
        used_as_subject = False
        used_as_object = False
        for s, p, o in bgp_triples:
            if not isinstance(p, URIRef):
                continue
            p_str = str(p)

            if isinstance(s, Variable) and str(s) == var_name:
                used_as_subject = True
                if p_str == RDF_TYPE:
                    if isinstance(o, URIRef):
                        sig.direct_types.add(str(o))
                else:
                    sig.outgoing_predicates.add(p_str)

            if isinstance(o, Variable) and str(o) == var_name:
                used_as_object = True
                if p_str != RDF_TYPE:
                    sig.incoming_predicates.add(p_str)

        if used_as_subject:
            sig.is_uri = True
        elif used_as_object:
            sig.is_uri = False
        else:
            # Only in SELECT, never in BGP → pure computed variable
            sig.is_uri = None

        # Subject context: for variables that appear as objects (literals and
        # URI refs alike), collect the signature of the *subject* variable(s)
        # from ``?subject <pred> ?this_var``. This disambiguates when the same
        # producing property (e.g. eno:designation) is used on different entity
        # types (Deposit vs Company).
        if used_as_object:
            subject_vars: set[str] = set()
            for s, p, o in bgp_triples:
                if (
                    isinstance(o, Variable)
                    and str(o) == var_name
                    and isinstance(s, Variable)
                ):
                    subject_vars.add(str(s))
            for subj in subject_vars:
                for s, p, o in bgp_triples:
                    if not isinstance(p, URIRef):
                        continue
                    p_str = str(p)
                    if isinstance(s, Variable) and str(s) == subj:
                        if p_str == RDF_TYPE and isinstance(o, URIRef):
                            sig.subject_context.add(f"type:{str(o)}")
                        else:
                            sig.subject_context.add(f"out:{p_str}")
                    if isinstance(o, Variable) and str(o) == subj:
                        sig.subject_context.add(f"in:{p_str}")

        signatures[var_name] = sig

    return signatures


def _predicates_of(
    var_name: str,
    bgp_triples: list[tuple[Any, Any, Any]],
) -> set[str]:
    """Collect all non-type predicates attached to ``?var_name`` in either role."""
    preds: set[str] = set()
    for s, p, o in bgp_triples:
        if not isinstance(p, URIRef):
            continue
        p_str = str(p)
        if p_str == RDF_TYPE:
            continue
        if isinstance(s, Variable) and str(s) == var_name:
            preds.add(p_str)
        if isinstance(o, Variable) and str(o) == var_name:
            preds.add(p_str)
    return preds


# =============================================================================
# Matching
# =============================================================================


def signature_similarity(a: ColumnSignature, b: ColumnSignature) -> float:
    """Compute a similarity score in [0, 1] between two ColumnSignatures.

    The score answers the schema-level question "do these two variables
    represent the same data column?" — i.e. the same (Class, Property) pair
    in ontology terms.  It intentionally ignores WHERE-clause differences
    (filters, OPTIONALs, join structure) which belong to result metrics.

    Hard compatibility check:
      - Aggregation flag must agree (aggregated ↔ aggregated only).

    Score = direction-aware Jaccard over predicate sets:
      - ``in_j``:  Jaccard of incoming predicates (the *producing property*
        for literal columns — strongest signal).
      - ``out_j``: Jaccard of outgoing predicates (structural context for
        URI columns that act as subjects).
      - ``type_j``: Jaccard of direct ``rdf:type`` classes.

    Incoming and outgoing predicates are compared **separately** (never
    merged) so that a URI column with ``outgoing={label}`` cannot spuriously
    match a literal column with ``incoming={label}``.

    The ``is_uri`` flag is deliberately NOT used as a hard gate because LLM
    queries often omit outgoing triples for a URI variable (e.g.
    ``?trip tro:line ?route`` without ``?route a tro:Line``), which makes
    the variable appear object-only (``is_uri=False``) even though it binds
    URIs.  The direction-aware predicate Jaccard is sufficient to prevent
    URI↔literal false matches.
    """
    # Aggregation handling
    if a.is_aggregated != b.is_aggregated:
        return 0.0
    if a.is_aggregated and b.is_aggregated:
        if a.aggregation_kind != b.aggregation_kind:
            # Allow substitution between SUM and COUNT since the agent may
            # have misinterpreted a "total number of X" as SUM vs COUNT.
            compat_pairs = {frozenset({"SUM", "COUNT"})}
            pair = frozenset({a.aggregation_kind or "", b.aggregation_kind or ""})
            if pair not in compat_pairs:
                return 0.0
        return _jaccard(a.aggregation_inner, b.aggregation_inner)

    # Direction-aware Jaccard over predicates and types.
    # Incoming and outgoing are compared separately so that e.g.
    # dept(out={label}) does NOT match courseLabel(in={label}).
    in_j = _jaccard(a.incoming_predicates, b.incoming_predicates)
    out_j = _jaccard(a.outgoing_predicates, b.outgoing_predicates)
    type_j = _jaccard(a.direct_types, b.direct_types)

    # Subject-context refinement: when two columns share the same producing
    # property (e.g. both have in={eno:designation}), disambiguate by how
    # similar their subject entities are. This scales the incoming Jaccard
    # down for mismatched subjects, breaking ties that would otherwise cause
    # the greedy matcher to pick an arbitrary (often wrong) assignment.
    if in_j > 0 and a.subject_context and b.subject_context:
        subj_j = _jaccard(a.subject_context, b.subject_context)
        in_j *= 0.7 + 0.3 * subj_j
    elif in_j > 0 and (a.subject_context or b.subject_context):
        # One side has context, the other doesn't → mild penalty
        in_j *= 0.85

    score = max(in_j, out_j)
    if a.direct_types and b.direct_types:
        score = max(score, type_j)

    return score


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


@dataclass
class SignatureMatch:
    """One matched (gt_var, llm_var) pair with debug info."""

    gt_var: str
    llm_var: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "gt_var": self.gt_var,
            "llm_var": self.llm_var,
            "score": self.score,
        }


def match_columns(
    gt_sigs: dict[str, ColumnSignature],
    llm_sigs: dict[str, ColumnSignature],
    min_score: float = 0.3,
) -> list[SignatureMatch]:
    """Greedy 1:1 bipartite matching between GT and LLM signatures.

    Only pairs with ``score >= min_score`` are considered. The algorithm
    repeatedly picks the highest-scoring unmatched pair until no pair above
    threshold remains.

    Args:
        gt_sigs: ground truth signatures by var name (no "?")
        llm_sigs: LLM query signatures by var name (no "?")
        min_score: minimum Jaccard score for a mapping to be accepted.

    Returns:
        List of SignatureMatch objects, sorted by descending score.
    """
    candidates: list[SignatureMatch] = []
    for gt_name, gt_sig in gt_sigs.items():
        if gt_sig.is_empty():
            continue
        for llm_name, llm_sig in llm_sigs.items():
            if llm_sig.is_empty():
                continue
            score = signature_similarity(gt_sig, llm_sig)
            if score >= min_score:
                candidates.append(SignatureMatch(gt_name, llm_name, score))

    candidates.sort(key=lambda m: m.score, reverse=True)

    used_gt: set[str] = set()
    used_llm: set[str] = set()
    final: list[SignatureMatch] = []
    for cand in candidates:
        if cand.gt_var in used_gt or cand.llm_var in used_llm:
            continue
        final.append(cand)
        used_gt.add(cand.gt_var)
        used_llm.add(cand.llm_var)

    return final


__all__ = [
    "ColumnSignature",
    "SignatureMatch",
    "extract_column_signatures",
    "signature_similarity",
    "match_columns",
]