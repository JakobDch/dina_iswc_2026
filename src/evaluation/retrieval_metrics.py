"""Retrieval metrics: parse retrieved triples and evaluate against GT.

Parses the agent's Turtle-format schema triples and calculates
schema-aware retrieval precision/recall/F1 using the SchemaGraph
for subclass-aware matching and ACCEPTABLE handling.
"""

import logging
import re
from dataclasses import dataclass, field

from rdflib import Graph, URIRef, Variable
from rdflib.namespace import OWL, RDF, RDFS, XSD
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parser import parseQuery

from src.evaluation.retrieval_ground_truth import (
    RetrievalNecessity,
    _build_type_map,
    _clean_query,
    _collect_all_bgp_triples,
)

logger = logging.getLogger(__name__)

# Namespaces to exclude from extracted classes (these are schema/meta URIs, not domain classes)
_EXCLUDED_OBJECT_NAMESPACES = {
    str(XSD),
    str(RDF),
    str(RDFS),
    str(OWL),
}

# Predicates to exclude from extracted properties (ontology structure, not domain predicates)
_EXCLUDED_PREDICATES = {
    str(RDFS.subClassOf),
    str(RDF.type),
    str(RDFS.subPropertyOf),
    str(OWL.equivalentClass),
    str(OWL.equivalentProperty),
}


@dataclass
class RetrievalMetricsResult:
    """Result of schema-aware retrieval evaluation.

    Uses the SchemaGraph for subclass-aware matching and ACCEPTABLE handling.
    """

    # Path coherence: fraction of GT (domain_class, property) edges
    # matched in the retrieved Turtle, using SchemaGraph for subclass resolution.
    schema_path_coherence: float = 0.0
    schema_paths_matched: int = 0
    schema_paths_total: int = 0
    schema_paths_details: list[dict] = field(default_factory=list)

    # Triple-level metrics: GT schema triples vs retrieved triples,
    # with subclass-aware matching and ACCEPTABLE handling.
    schema_triple_precision: float = 0.0
    schema_triple_recall: float = 0.0
    schema_triple_f1: float = 0.0
    schema_triples_matched: int = 0
    schema_triples_total: int = 0

    # GT schema as Turtle for debugging/inspection
    gt_schema_turtle: str = ""

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        return {
            "schema_path_coherence": self.schema_path_coherence,
            "schema_paths_matched": self.schema_paths_matched,
            "schema_paths_total": self.schema_paths_total,
            "schema_paths_details": self.schema_paths_details,
            "schema_triple_precision": self.schema_triple_precision,
            "schema_triple_recall": self.schema_triple_recall,
            "schema_triple_f1": self.schema_triple_f1,
            "schema_triples_matched": self.schema_triples_matched,
            "schema_triples_total": self.schema_triples_total,
            "gt_schema_turtle": self.gt_schema_turtle,
        }


def _is_excluded_namespace(uri: str) -> bool:
    """Check if URI belongs to an excluded namespace."""
    for ns in _EXCLUDED_OBJECT_NAMESPACES:
        if uri.startswith(ns):
            return True
    return False


def _extract_turtle_blocks(text: str) -> list[str]:
    """Extract Turtle code blocks from Markdown-formatted text.

    LLM responses often contain Turtle code in Markdown code blocks like:
    ```turtle
    PREFIX ex: <http://example.org/>
    ex:Class ex:property ex:Value .
    ```

    This function extracts all such blocks, as well as any standalone Turtle
    that looks valid (starts with PREFIX or a URI).

    Args:
        text: Raw text potentially containing Turtle code blocks

    Returns:
        List of extracted Turtle code snippets
    """
    blocks = []

    # Extract Markdown code blocks: ```turtle ... ``` or ``` ... ```
    # Pattern matches:
    # - Optional language identifier (turtle, ttl, or empty)
    # - Code content (non-greedy)
    # - Closing backticks
    pattern = r'```(?:turtle|ttl)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    blocks.extend(matches)

    # If no code blocks found, check if the entire text looks like Turtle
    # (starts with PREFIX or a URI)
    if not blocks:
        stripped = text.strip()
        if stripped.startswith(('PREFIX', '@prefix', 'http://', 'https://')):
            blocks.append(stripped)

    return blocks


def _add_missing_prefixes(turtle_code: str) -> str:
    """Add commonly missing prefixes to Turtle code.

    LLMs sometimes forget to declare common prefixes like foaf, dcterms, schema.
    This function adds standard prefix declarations if they're used but not declared.

    Args:
        turtle_code: Turtle code that may be missing prefix declarations

    Returns:
        Turtle code with missing prefixes added
    """
    # Common prefixes that LLMs often forget to declare
    common_prefixes = {
        'foaf': 'http://xmlns.com/foaf/0.1/',
        'dcterms': 'http://purl.org/dc/terms/',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'schema': 'http://schema.org/',
        'skos': 'http://www.w3.org/2004/02/skos/core#',
        'dbo': 'http://dbpedia.org/ontology/',
        'dbr': 'http://dbpedia.org/resource/',
    }

    # Extract already declared prefixes
    declared_prefixes = set()
    for line in turtle_code.split('\n'):
        if line.strip().startswith(('PREFIX', '@prefix')):
            # Extract prefix name (e.g., "foaf:" from "PREFIX foaf: <...>")
            parts = line.split()
            if len(parts) >= 2:
                prefix_name = parts[1].rstrip(':')
                declared_prefixes.add(prefix_name)

    # Find used but undeclared prefixes
    missing_prefixes = []
    for prefix, namespace in common_prefixes.items():
        # Check if prefix is used in the code (e.g., "foaf:name")
        if f'{prefix}:' in turtle_code and prefix not in declared_prefixes:
            missing_prefixes.append(f'PREFIX {prefix}: <{namespace}>')

    # Add missing prefixes at the beginning
    if missing_prefixes:
        prefix_block = '\n'.join(missing_prefixes) + '\n\n'
        return prefix_block + turtle_code

    return turtle_code


def _regex_fallback_parse(turtle_code: str) -> tuple[set[str], set[str]]:
    """Fallback parser using regex when RDFLib fails.

    Extracts classes and properties using pattern matching when strict parsing fails.
    This is more lenient and handles syntactically invalid but semantically meaningful Turtle.

    Args:
        turtle_code: Turtle code to parse

    Returns:
        Tuple of (class_uris, property_uris)
    """
    classes: set[str] = set()
    properties: set[str] = set()

    # Build prefix map from declarations
    prefix_map = {}
    for line in turtle_code.split('\n'):
        if line.strip().startswith(('PREFIX', '@prefix')):
            # Parse: PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            match = re.match(r'(?:PREFIX|@prefix)\s+(\w+):\s*<([^>]+)>', line, re.IGNORECASE)
            if match:
                prefix_map[match.group(1)] = match.group(2)

    # Extract triple patterns: subject predicate object .
    # Pattern: prefix:Name prefix:property prefix:Name .
    triple_pattern = r'(\w+):(\w+)\s+(\w+):(\w+)\s+(\w+):(\w+)\s*\.'
    for match in re.finditer(triple_pattern, turtle_code):
        subj_prefix, subj_name = match.group(1), match.group(2)
        pred_prefix, pred_name = match.group(3), match.group(4)
        obj_prefix, obj_name = match.group(5), match.group(6)

        # Expand to full URIs
        if subj_prefix in prefix_map:
            subj_uri = prefix_map[subj_prefix] + subj_name
            classes.add(subj_uri)

        if pred_prefix in prefix_map:
            pred_uri = prefix_map[pred_prefix] + pred_name
            if pred_uri not in _EXCLUDED_PREDICATES:
                properties.add(pred_uri)

        if obj_prefix in prefix_map and not _is_excluded_namespace(prefix_map[obj_prefix]):
            obj_uri = prefix_map[obj_prefix] + obj_name
            classes.add(obj_uri)

    return classes, properties


def parse_retrieved_triples(
    turtle_strings: list[str],
) -> tuple[set[str], set[str]]:
    """Parse Turtle-format retrieved triples into classes and properties.

    The agent stores retrieved schema triples as Turtle strings where:
    - Subjects are classes (domain)
    - Predicates are domain properties (or rdfs:subClassOf for hierarchy)
    - Objects are classes (range) or XSD types

    We extract:
    - Classes: All URIRef subjects + URIRef objects that aren't XSD/RDF/RDFS/OWL
    - Properties: All predicates except rdfs:subClassOf, rdf:type, etc.

    Args:
        turtle_strings: List of Turtle-format strings from trace (may contain
            Markdown-formatted text with embedded ```turtle code blocks)

    Returns:
        Tuple of (class_uris, property_uris)
    """
    classes: set[str] = set()
    properties: set[str] = set()

    for turtle_str in turtle_strings:
        if not turtle_str or not turtle_str.strip():
            continue

        # Extract Turtle code blocks from Markdown-formatted text
        turtle_blocks = _extract_turtle_blocks(turtle_str)

        # If no blocks extracted, try parsing the whole string (backward compatibility)
        if not turtle_blocks:
            turtle_blocks = [turtle_str]

        for turtle_code in turtle_blocks:
            # Try to add missing common prefixes
            enhanced_code = _add_missing_prefixes(turtle_code)

            try:
                g = Graph()
                g.parse(data=enhanced_code, format="turtle")
                for s, p, o in g:
                    s_str = str(s)
                    p_str = str(p)
                    o_str = str(o)

                    # Subject is always a class in schema triples
                    if hasattr(s, "n3") and not s_str.startswith("_:"):
                        classes.add(s_str)

                    # Object: add as class if it's a URI and not an excluded namespace
                    if hasattr(o, "n3") and not o_str.startswith("_:"):
                        from rdflib.term import URIRef as _URIRef

                        if isinstance(o, _URIRef) and not _is_excluded_namespace(o_str):
                            classes.add(o_str)

                    # Predicate: add as property if not excluded
                    if p_str not in _EXCLUDED_PREDICATES:
                        properties.add(p_str)

            except Exception as e:
                logger.debug(f"RDFLib parsing failed: {e}. Trying regex fallback...")
                # Use regex fallback parser for syntactically invalid but semantically meaningful Turtle
                fallback_classes, fallback_props = _regex_fallback_parse(turtle_code)
                classes.update(fallback_classes)
                properties.update(fallback_props)
                logger.debug(f"Fallback parser extracted {len(fallback_classes)} classes, {len(fallback_props)} properties")

    return classes, properties


def _f1(precision: float, recall: float) -> float:
    """Calculate F1 score."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _safe_div(numerator: float, denominator: float) -> float:
    """Safe division."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


RDF_TYPE_URI = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def extract_gt_paths(
    gt_sparql: str,
    prefix_map: dict[str, str] | None = None,
    schema: "SchemaGraph | None" = None,
) -> list[tuple[str | None, str]]:
    """Extract ``(domain_class_uri, property_uri)`` pairs from a GT SPARQL query.

    For every triple pattern ``?s p ?o`` (across all BGPs — including ones
    nested in ``OPTIONAL`` / ``UNION`` / ``MINUS`` / ``GRAPH`` / subqueries),
    emit one ``(domain_class, property)`` pair per type the subject is bound
    to. The subject's type set is taken from any ``?s rdf:type C`` / ``?s a C``
    pattern anywhere in the algebra tree — which matches the existing
    retrieval-GT extractor's semantics in
    :mod:`src.evaluation.retrieval_ground_truth`.

    - ``rdf:type`` predicates are skipped (no path is generated for them)
    - If ``?s`` has no type assertion but appears as the object of a triple
      whose predicate has a class-valued range in ``schema``, that range is
      used as the subject's type. Without this, an untyped variable makes
      :func:`build_schema_gt` expand to every class carrying the property,
      which on a large schema inflates the GT by two orders of magnitude.
    - If ``?s`` still has **no** type, ``domain_class`` is ``None`` —
      in that case the path-coherence check falls back to property-only
      presence (same as ``property_recall``)
    - Uses rdflib's SPARQL parser & algebra (not regex) — robust to
      indentation, ``;``/``,``-chained triples, and arbitrary nesting

    Args:
        gt_sparql: Ground-truth SPARQL query string
        prefix_map: Optional seed ``prefix → namespace`` mapping. Unused by
            this implementation (rdflib expands prefixes internally), kept
            for API compatibility.

    Returns:
        Deduplicated list of ``(domain_class_full_uri_or_None, property_full_uri)``
    """
    _ = prefix_map  # unused — parseQuery handles prefix expansion itself

    try:
        cleaned = _clean_query(gt_sparql)
        parsed = parseQuery(cleaned)
        algebra = translateQuery(parsed)
    except Exception as e:
        logger.debug(f"extract_gt_paths: failed to parse SPARQL: {e}")
        return []

    # Collect every BGP triple from the whole algebra tree (covers OPTIONAL /
    # UNION / MINUS / GRAPH / subqueries).
    all_triples = _collect_all_bgp_triples(algebra.algebra, set())
    if not all_triples:
        return []

    # Variable → set of class URIs (from ?v rdf:type C across the whole tree)
    type_map = _build_type_map(all_triples)

    # Infer the type of variables that carry no rdf:type assertion but occur as
    # the object of a property with a class-valued range.
    if schema is not None:
        for s, p, o in all_triples:
            if not isinstance(o, Variable) or not isinstance(p, URIRef):
                continue
            var_name = f"?{o}"
            if type_map.get(var_name):
                continue  # already typed explicitly
            range_uri = schema.property_ranges.get(str(p))
            if range_uri and range_uri in schema.class_properties:
                type_map[var_name] = {range_uri}
            _ = s

    edges: list[tuple[str | None, str]] = []
    seen: set[tuple[str, str]] = set()

    for s, p, _o in all_triples:
        # Only URI predicates; skip rdf:type
        if not isinstance(p, URIRef):
            continue
        p_str = str(p)
        if p_str == RDF_TYPE_URI:
            continue

        # Subject can be a Variable (most common) or a URIRef constant
        subj_classes: list[str | None]
        if isinstance(s, Variable):
            cls_set = type_map.get(f"?{s}") or set()
            subj_classes = sorted(cls_set) if cls_set else [None]
        elif isinstance(s, URIRef):
            subj_classes = [str(s)]
        else:
            continue  # BNode subject — skip

        for dom in subj_classes:
            key = (dom or "", p_str)
            if key in seen:
                continue
            seen.add(key)
            edges.append((dom, p_str))

    return edges


def calculate_schema_path_coherence(
    schema_gt: "RetrievalSchemaGT",
    turtle_strings: list[str],
    schema: "SchemaGraph",
) -> tuple[float, int, int, list[dict]]:
    """Compute path coherence with subclass-expansion matching.

    Matching rule:
    - If the retrieved set contains the GT domain class itself for the
      property → direct match (1/1).
    - Otherwise, if at least one concrete subclass of the GT domain class
      with that property in the schema was retrieved, the GT path is
      expanded into one expected path per such subclass and recall is
      measured over how many of those subclass-level paths the agent
      retrieved (k/n).
    - No match at all → 0/1 (single GT path, missed).

    ACCEPTABLE handling: ACCEPTABLE paths only contribute to the
    effective GT denominator by what was actually matched, so missing an
    ACCEPTABLE path never penalises recall.

    Returns:
        ``(score, matched, effective_total, details)``
    """
    from src.evaluation.retrieval_ground_truth import RetrievalNecessity, RetrievalSchemaGT

    if not schema_gt.paths:
        return 0.0, 0, 0, []

    # Parse retrieved Turtle into edge index
    g = Graph()
    for blob in turtle_strings or []:
        if not blob or not blob.strip():
            continue
        try:
            g.parse(data=blob, format="turtle")
        except Exception:
            pass

    edge_index: set[tuple[str, str]] = {
        (str(s), str(p))
        for s, p, _o in g
        if isinstance(s, URIRef) and isinstance(p, URIRef)
    }
    predicate_index: set[str] = {p for _s, p in edge_index}

    matched = 0
    details: list[dict] = []
    required_total = 0
    acceptable_denom = 0

    for domain_class, prop_uri, necessity in schema_gt.paths:
        if domain_class is None:
            ok = prop_uri in predicate_index
            m, e = (1, 1) if ok else (0, 1)
        elif (domain_class, prop_uri) in edge_index:
            # Direct match on the GT class itself
            m, e = 1, 1
        else:
            # Expand to concrete subclasses with this property in the schema
            descendants = schema.get_all_descendants(domain_class)
            relevant_subs = {
                c for c in descendants
                if c in schema.class_property_ranges
                and prop_uri in schema.class_property_ranges[c]
            }
            if not relevant_subs:
                m, e = 0, 1
            else:
                matched_subs = sum(
                    1 for sub in relevant_subs
                    if (sub, prop_uri) in edge_index
                )
                if matched_subs == 0:
                    m, e = 0, 1
                else:
                    m, e = matched_subs, len(relevant_subs)

        matched += m

        if necessity == RetrievalNecessity.REQUIRED:
            required_total += e
        elif necessity == RetrievalNecessity.ACCEPTABLE:
            # ACCEPTABLE: never penalises recall, only adds to numerator/denom what matched
            acceptable_denom += m

        details.append({
            "domain": domain_class,
            "property": prop_uri,
            "necessity": necessity.value,
            "matched": m,
            "expected": e,
        })

    # Effective GT = required (with expansion) + acceptable that matched
    effective_total = required_total + acceptable_denom
    score = matched / effective_total if effective_total > 0 else 0.0

    return score, matched, effective_total, details


def calculate_schema_triple_metrics(
    schema_gt: "RetrievalSchemaGT",
    turtle_strings: list[str],
    schema: "SchemaGraph",
) -> tuple[float, float, float, int, int]:
    """Compare GT schema triples against retrieved triples.

    Matching rule (subclass-expansion):
    - The primary expectation is the GT class itself. A retrieved triple
      matches a GT triple ``(S_gt, P_gt, O_gt)`` directly if its subject
      equals ``S_gt``, its predicate equals ``P_gt``, and its object is
      ``O_gt`` (or, for class ranges, a descendant of ``O_gt``). XSD
      ranges are matched exactly.
    - If the GT class itself is missing from the retrieved set, the GT
      triple is expanded into one expected triple per concrete subclass
      of ``S_gt`` that has ``P_gt`` in the schema. Recall is then measured
      over how many of these subclass-level triples are covered.
    - If no direct match and no subclass match exists, the GT triple
      counts as 0/1 (one expected concept missed).

    Precision is computed against any retrieved triple whose predicate
    appears in the GT and that matches some GT triple by direct or
    subclass coverage (so retrieving a subclass triple is never punished
    as a false positive on its own).

    ACCEPTABLE handling:
    - ACCEPTABLE triples only contribute to the effective GT denominator
      by what was actually matched, so missing an ACCEPTABLE concept (or
      missing some of its subclass variants) never penalises recall.

    Returns:
        ``(precision, recall, f1, matched_count, effective_gt_count)``
    """
    from src.evaluation.retrieval_ground_truth import RetrievalNecessity, RetrievalSchemaGT

    if not schema_gt.triples:
        return 0.0, 0.0, 0.0, 0, 0

    # Parse retrieved Turtle
    g = Graph()
    for blob in turtle_strings or []:
        if not blob or not blob.strip():
            continue
        try:
            g.parse(data=blob, format="turtle")
        except Exception:
            pass

    retrieved_triples: set[tuple[str, str, str]] = {
        (str(s), str(p), str(o)) for s, p, o in g
    }

    # Pre-compute descendant sets for GT classes (cached)
    _descendant_cache: dict[str, set[str]] = {}

    def _get_valid_set(uri: str) -> set[str]:
        if uri not in _descendant_cache:
            _descendant_cache[uri] = {uri} | schema.get_all_descendants(uri)
        return _descendant_cache[uri]

    XSD_PREFIX = "http://www.w3.org/2001/XMLSchema#"

    def _valid_objects(gt_o: str) -> set[str]:
        if gt_o.startswith(XSD_PREFIX):
            return {gt_o}
        return _get_valid_set(gt_o)

    def _matches_for_triple(gt_s: str, gt_p: str, gt_o: str) -> tuple[int, int, set[tuple[str, str, str]]]:
        """Apply the subclass-expansion rule.

        Returns ``(matched_count, expected_count, matched_retrieved_triples)``.
        """
        v_objs = _valid_objects(gt_o)

        direct = {
            (rs, rp, ro)
            for rs, rp, ro in retrieved_triples
            if rs == gt_s and rp == gt_p and ro in v_objs
        }
        if direct:
            return 1, 1, direct

        descendants = schema.get_all_descendants(gt_s)
        relevant_subs = {
            c for c in descendants
            if c in schema.class_property_ranges
            and gt_p in schema.class_property_ranges[c]
        }
        if not relevant_subs:
            return 0, 1, set()

        matched_subs: set[str] = set()
        matched_set: set[tuple[str, str, str]] = set()
        for sub in relevant_subs:
            sub_hits = {
                (rs, rp, ro)
                for rs, rp, ro in retrieved_triples
                if rs == sub and rp == gt_p and ro in v_objs
            }
            if sub_hits:
                matched_subs.add(sub)
                matched_set.update(sub_hits)

        if not matched_subs:
            return 0, 1, set()
        return len(matched_subs), len(relevant_subs), matched_set

    required_matched = 0
    required_total = 0
    acceptable_matched = 0
    acceptable_denom = 0
    total_matched = 0

    matched_retrieved: set[tuple[str, str, str]] = set()

    for gt_s, gt_p, gt_o, necessity in schema_gt.triples:
        m, e, ms = _matches_for_triple(gt_s, gt_p, gt_o)
        matched_retrieved.update(ms)
        total_matched += m

        if necessity == RetrievalNecessity.REQUIRED:
            required_matched += m
            required_total += e
        elif necessity == RetrievalNecessity.ACCEPTABLE:
            acceptable_matched += m
            acceptable_denom += m

    # Effective GT = required (with expansion) + acceptable that were matched
    effective_gt = required_total + acceptable_denom

    # Precision: fraction of retrieved triples (with GT predicates) that contribute to TP
    gt_predicates = {p for _s, p, _o, _n in schema_gt.triples}
    relevant_retrieved = {
        (s, p, o) for s, p, o in retrieved_triples if p in gt_predicates
    }
    precision_tp = len(matched_retrieved)
    precision = _safe_div(precision_tp, len(relevant_retrieved)) if relevant_retrieved else 0.0

    recall = _safe_div(total_matched, effective_gt) if effective_gt > 0 else 0.0
    f1 = _f1(precision, recall)

    return precision, recall, f1, total_matched, effective_gt


def calculate_retrieval_metrics(
    turtle_strings: list[str],
    schema: "SchemaGraph",
    schema_gt: "RetrievalSchemaGT",
) -> RetrievalMetricsResult:
    """Calculate schema-aware retrieval metrics.

    Uses the SchemaGraph for subclass-aware matching and ACCEPTABLE handling.

    Args:
        turtle_strings: Raw Turtle strings from agent's retrieved context
        schema: SchemaGraph for the query's dataset(s)
        schema_gt: Schema-level ground truth (from build_schema_gt_for_query)

    Returns:
        RetrievalMetricsResult with path coherence and triple-level metrics
    """
    result = RetrievalMetricsResult()

    if not turtle_strings or not schema_gt.triples:
        return result

    # Path coherence (uses SchemaGraph for subclass resolution)
    spc, s_matched, s_total, s_details = calculate_schema_path_coherence(
        schema_gt, turtle_strings, schema
    )
    result.schema_path_coherence = spc
    result.schema_paths_matched = s_matched
    result.schema_paths_total = s_total
    result.schema_paths_details = s_details

    # Triple-level metrics
    st_prec, st_rec, st_f1, st_matched, st_total = calculate_schema_triple_metrics(
        schema_gt, turtle_strings, schema
    )
    result.schema_triple_precision = st_prec
    result.schema_triple_recall = st_rec
    result.schema_triple_f1 = st_f1
    result.schema_triples_matched = st_matched
    result.schema_triples_total = st_total

    # Store GT Turtle for debugging
    result.gt_schema_turtle = schema_gt.to_turtle(schema.prefixes)

    return result


def calculate_retrieval_metrics_best_variant(
    turtle_strings: list[str],
    ground_truth: "AdaptiveGroundTruth",
) -> RetrievalMetricsResult:
    """Score retrieval candidate-wise against every admissible reading.

    Mirrors the result-level evaluation: the retrieval ground truth R_Q is built
    per admissible query Q, the retrieved schema triples are scored against each
    R_Q, and the reading the agent covers best (highest triple recall, ties by
    path coherence) is reported. Recall and path coherence therefore ask whether
    the agent surfaced the schema of *one* admissible reading in full, not the
    union of all of them. Precision is measured against the union of all R_Q,
    so that a triple belonging to any admissible reading is never a false
    positive. F1 combines the reported recall with that precision.

    With a single reading this reduces to calculate_retrieval_metrics on the
    (then identical) union.
    """
    from src.evaluation.retrieval_ground_truth import build_schema_gt, build_schema_gt_for_query
    from src.validation.schema_graph import get_combined_schema

    dataset_ids = ground_truth.datasets or [ground_truth.dataset]
    schema = get_combined_schema(dataset_ids)
    union_gt = build_schema_gt_for_query(ground_truth)

    variants = list(ground_truth.sparql_queries or [])
    if len(variants) <= 1:
        return calculate_retrieval_metrics(turtle_strings, schema, union_gt)

    per_reading = []
    for sparql in variants:
        gt = build_schema_gt(sparql, dataset_ids, ground_truth.query_id)
        if gt.triples:
            per_reading.append(calculate_retrieval_metrics(turtle_strings, schema, gt))
    if not per_reading:
        return calculate_retrieval_metrics(turtle_strings, schema, union_gt)

    best = max(per_reading, key=lambda r: (r.schema_triple_recall, r.schema_path_coherence))
    union_res = calculate_retrieval_metrics(turtle_strings, schema, union_gt)
    best.schema_triple_precision = union_res.schema_triple_precision
    best.schema_triple_f1 = _f1(best.schema_triple_precision, best.schema_triple_recall)
    return best
