"""
SPARQL query analysis and edit cost calculation.

This module provides functions for parsing SPARQL queries and calculating
edit distances between generated and reference queries.
"""

import json
import logging
import re
import collections
from typing import Optional, Dict, Any

from rdflib.plugins.sparql.parser import parseQuery as parse_sparql_query
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parserutils import CompValue
from scipy.optimize import linear_sum_assignment
import numpy as np

logger = logging.getLogger(__name__)

# Cost rules for algebraic components
ALGEBRA_COST_RULES: Dict[str, int] = {
    'LeftJoin': 2, 'Filter': 2, 'Extend': 2, 'Distinct': 1, 'Group': 2,
    'Aggregate': 3, 'Union': 1, 'Values': 2, 'Service': 3, 'Minus': 2,
    'OrderBy': 1, 'Slice': 1
}
PROJECTION_VAR_COST = 1
TRIPLE_COST = 3

SIMPLE_CLAUSE_RULES = {
    'Distinct': {
        'regex': re.compile(r'\bSELECT\s+DISTINCT\b', re.IGNORECASE),
        'cost_add': 1,
        'cost_modify': 0
    },
    'OrderBy': {
        'regex': re.compile(r'\bORDER\s+BY\s+(.*?)(?=\b(GROUP BY|HAVING|LIMIT|OFFSET|$))', re.IGNORECASE | re.DOTALL),
        'cost_add': 2,
        'cost_modify': 1
    },
    'LimitOffset': {
        'regex': re.compile(r'\b(LIMIT\s+\d+|OFFSET\s+\d+)\b', re.IGNORECASE),
        'cost_add': 1,
        'cost_modify': 1
    }
}


def robust_json_parse(llm_response: str, request_id: str = "") -> Optional[dict]:
    """
    Parse JSON from LLM responses with multiple fallback strategies.

    Handles markdown code blocks, double braces, and extra text around JSON.
    """
    if not llm_response or not llm_response.strip():
        logger.error(f"[{request_id}] Empty LLM response")
        return None

    text = llm_response.strip()

    markdown_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text, re.DOTALL)
    if markdown_match:
        text = markdown_match.group(1).strip()

    json_match = re.search(r'\{[\s\S]*\}', text)
    if not json_match:
        logger.error(f"[{request_id}] No JSON object found in response")
        return None

    json_str = json_match.group(0).strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.debug(f"[{request_id}] Initial JSON parse failed: {e}")

    if json_str.startswith('{{') and json_str.endswith('}}'):
        json_str_fixed = json_str[1:-1].strip()
        try:
            return json.loads(json_str_fixed)
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    try:
        obj, idx = decoder.raw_decode(json_str)
        return obj
    except json.JSONDecodeError as e:
        logger.error(f"[{request_id}] All JSON parsing attempts failed: {e}")
        return None


def _clean_sparql_query(query_string: str) -> str:
    """Clean SPARQL query by removing formatting artifacts and fixing syntax errors."""
    query_string = query_string.replace('\u200B', '')

    if query_string.strip().startswith('{') and query_string.strip().endswith('}'):
        try:
            json_data = json.loads(query_string)
            if isinstance(json_data, dict) and 'corrected_query' in json_data:
                query_string = json_data['corrected_query']
            elif isinstance(json_data, dict) and 'query' in json_data:
                query_string = json_data['query']
        except:
            pass

    query_string = re.sub(r'```(?:sparql)?\s*\n?', '', query_string)
    query_string = re.sub(r'\n?```', '', query_string)
    query_string = re.sub(r'SELECT\s+DISTINCT\?', 'SELECT DISTINCT ?', query_string)
    query_string = re.sub(r'SELECT\?', 'SELECT ?', query_string)
    query_string = re.sub(r'(\w+:[\w_]+)\?', r'\1 ?', query_string)

    return query_string.strip()


def _serialize_value(value: Any) -> str:
    """Serialize a value to a deterministic string representation."""
    if value is None:
        return "None"

    if isinstance(value, set):
        sorted_elements = sorted(_serialize_value(elem) for elem in value)
        return '{' + ', '.join(sorted_elements) + '}'

    if hasattr(value, 'n3'):
        return value.n3()

    type_name = type(value).__name__
    if type_name in ['Variable', 'URIRef', 'Literal', 'BNode']:
        return str(value)

    if isinstance(value, (str, int, float, bool)):
        return repr(value)

    return str(value)


def _serialize_algebra_component(comp: Any) -> str:
    """
    Convert an algebra component to a stable semantic string representation.

    Creates deterministic representations avoiding Python object IDs.
    """
    if not isinstance(comp, CompValue):
        return _serialize_value(comp)

    comp_dict = dict(comp.items())
    comp_name = comp.name

    FIELDS_TO_SERIALIZE = {
        'Filter': ['expr'],
        'Group': ['expr'],
        'Extend': ['var', 'expr'],
    }

    if comp_name in FIELDS_TO_SERIALIZE:
        keys_to_serialize = FIELDS_TO_SERIALIZE[comp_name]
        comp_dict = {k: v for k, v in comp_dict.items() if k in keys_to_serialize}

    serialized_items = []
    for key, value in sorted(comp_dict.items()):
        if isinstance(value, CompValue):
            serialized_value = _serialize_algebra_component(value)
        elif isinstance(value, (list, tuple)):
            serialized_elements = [
                _serialize_algebra_component(v) if isinstance(v, CompValue) else _serialize_value(v)
                for v in value
            ]
            serialized_value = '[' + ', '.join(serialized_elements) + ']'
        else:
            serialized_value = _serialize_value(value)

        serialized_items.append(f"{key}={serialized_value}")

    return f"{comp.name}({', '.join(serialized_items)})"


def _extract_algebra_components(part: Any, components: Dict[str, Any], visited: set):
    """Extract algebraic components recursively from a parsed query."""
    if not isinstance(part, CompValue) or id(part) in visited:
        return
    visited.add(id(part))

    part_name = part.name

    if part_name in ALGEBRA_COST_RULES:
        components[part_name].append(_serialize_algebra_component(part))

    if part_name == 'Project' and 'PV' in part:
        components['ProjectionVars'].update(str(v) for v in part['PV'])

    elif part_name == 'BGP' and 'triples' in part:
        for t in part['triples']:
            components['Triples'].append(tuple(v.n3() for v in t))

    for child in part.values():
        if isinstance(child, (list, tuple)):
            for item in child:
                _extract_algebra_components(item, components, visited)
        elif isinstance(child, CompValue):
            _extract_algebra_components(child, components, visited)


def _normalize_query_variables(components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize variable names and blank nodes to ?v1, ?v2, etc.

    Ensures semantically equivalent queries with different variable names
    map to the same normalized form.
    """
    var_pattern = re.compile(r'\?(\w+)')
    blank_node_pattern = re.compile(r'_:[A-Za-z0-9]+')

    seen_identifiers = []
    identifier_mapping = {}

    def collect_identifiers(text: str):
        for match in var_pattern.finditer(text):
            identifier = '?' + match.group(1)
            if identifier not in seen_identifiers:
                seen_identifiers.append(identifier)

        for match in blank_node_pattern.finditer(text):
            identifier = match.group(0)
            if identifier not in seen_identifiers:
                seen_identifiers.append(identifier)

    if 'ProjectionVars' in components:
        for var in sorted(components['ProjectionVars']):
            collect_identifiers(var)

    if 'Triples' in components:
        for triple in components['Triples']:
            for component in triple:
                collect_identifiers(component)

    for comp_type, comp_data in components.items():
        if comp_type in ['ProjectionVars', 'Triples']:
            continue
        if isinstance(comp_data, collections.Counter):
            for item in comp_data.elements():
                collect_identifiers(str(item))

    for i, identifier in enumerate(seen_identifiers, 1):
        identifier_mapping[identifier] = f'?v{i}'

    var_mapping = identifier_mapping
    normalized_components = {}

    if 'ProjectionVars' in components:
        normalized_components['ProjectionVars'] = set()
        for var in components['ProjectionVars']:
            normalized_var = var_mapping.get(var, var)
            normalized_components['ProjectionVars'].add(normalized_var)

    if 'Triples' in components:
        normalized_components['Triples'] = []
        for triple in components['Triples']:
            normalized_triple = tuple(
                _apply_var_mapping_to_string(component, var_mapping)
                for component in triple
            )
            normalized_components['Triples'].append(normalized_triple)

    for comp_type, comp_data in components.items():
        if comp_type in ['ProjectionVars', 'Triples']:
            continue
        if isinstance(comp_data, collections.Counter):
            normalized_counter = collections.Counter()
            for item, count in comp_data.items():
                normalized_item = _apply_var_mapping_to_string(str(item), var_mapping)
                normalized_counter[normalized_item] = count
            normalized_components[comp_type] = normalized_counter
        else:
            normalized_components[comp_type] = comp_data

    return normalized_components


def _apply_var_mapping_to_string(text: str, var_mapping: Dict[str, str]) -> str:
    """Apply variable mapping to a string."""
    result = text
    for old_var in sorted(var_mapping.keys(), key=len, reverse=True):
        result = result.replace(old_var, var_mapping[old_var])
    return result


def _get_query_components(query_string: str, normalize_variables: bool = True) -> Optional[Dict[str, Any]]:
    """
    Parse a query and return a structured list of its components.

    Args:
        query_string: The SPARQL query to parse
        normalize_variables: If True, normalize variable names to ?v1, ?v2, etc.

    Returns:
        Dictionary with query components or None on error
    """
    try:
        cleaned_query = _clean_sparql_query(query_string)
        parsed_query = parse_sparql_query(cleaned_query)
        algebra_query = translateQuery(parsed_query)

        components = collections.defaultdict(list)
        components['ProjectionVars'] = set()

        _extract_algebra_components(algebra_query.algebra, components, set())

        all_triples = components.pop('Triples', [])

        for key, value in components.items():
            if key != 'ProjectionVars':
                components[key] = collections.Counter(value)

        components['Triples'] = all_triples

        if normalize_variables:
            components = _normalize_query_variables(components)

        return components
    except Exception as e:
        logger.error(f"Failed to parse query: {str(e)}")
        return None


def _compare_simple_clauses_textually(
    generated_query_str: str,
    corrected_query_str: str,
    request_id: str
) -> tuple[int, list]:
    """Compare simple SPARQL clauses textually and return cost and edits."""
    total_cost = 0
    edits = []

    for clause_name, rules in SIMPLE_CLAUSE_RULES.items():
        gen_match = rules['regex'].search(generated_query_str)
        corr_match = rules['regex'].search(corrected_query_str)

        gen_content = gen_match.group(1).strip() if gen_match and rules['regex'].groups > 0 else (gen_match is not None)
        corr_content = corr_match.group(1).strip() if corr_match and rules['regex'].groups > 0 else (corr_match is not None)

        if not gen_content and corr_content:
            cost = rules['cost_add']
            total_cost += cost
            edits.append({
                "change": f"Clause '{clause_name}' added",
                "cost": cost,
                "detail": f"Added: {rules['regex'].pattern.split('(')[0].strip()}"
            })
        elif gen_content and corr_content and gen_content != corr_content:
            cost = rules['cost_modify']
            if cost > 0:
                total_cost += cost
                edits.append({
                    "change": f"Clause '{clause_name}' content modified",
                    "cost": cost,
                    "detail": f"FROM: {gen_content} -> TO: {corr_content}"
                })
        elif gen_content and not corr_content:
            edits.append({
                "change": f"Clause '{clause_name}' removed",
                "cost": 0,
                "detail": f"Removed: {gen_content}"
            })

    return total_cost, edits


def calculate_diff_cost_deterministically(
    generated_query_str: str,
    corrected_query_str: str,
    request_id: str
) -> Optional[Dict[str, Any]]:
    """
    Calculate the edit cost between a generated and corrected SPARQL query.

    Uses Hungarian algorithm for optimal triple matching and compares
    algebraic structures for comprehensive edit cost calculation.
    """
    log_prefix = f"[{request_id}]"
    logger.info(f"{log_prefix} Starting deterministic diff_cost calculation.")

    cand_comps = _get_query_components(generated_query_str)
    corr_comps = _get_query_components(corrected_query_str)

    if not cand_comps or not corr_comps:
        logger.error(f"{log_prefix} Could not extract components from one or both queries.")
        return None

    total_cost = 0
    edits_breakdown = []

    textual_cost, textual_edits = _compare_simple_clauses_textually(
        generated_query_str, corrected_query_str, request_id
    )
    total_cost += textual_cost
    edits_breakdown.extend(textual_edits)

    # Triple comparison using Hungarian algorithm
    cand_triples = list(cand_comps.get('Triples', []))
    corr_triples = list(corr_comps.get('Triples', []))

    if cand_triples and corr_triples:
        cost_matrix = np.zeros((len(cand_triples), len(corr_triples)))
        for i, t_cand in enumerate(cand_triples):
            for j, t_corr in enumerate(corr_triples):
                cost_matrix[i, j] = sum(a != b for a, b in zip(t_cand, t_corr))
        cand_indices, corr_indices = linear_sum_assignment(cost_matrix)
        matched_cand_indices = set(cand_indices)
        matched_corr_indices = set(corr_indices)
        for i in range(len(cand_indices)):
            c_idx, r_idx = cand_indices[i], corr_indices[i]
            match_cost = int(cost_matrix[c_idx, r_idx])
            if match_cost > 0:
                total_cost += match_cost
                t_cand_str = " ".join(cand_triples[c_idx])
                t_corr_str = " ".join(corr_triples[r_idx])
                edits_breakdown.append({
                    "change": f"Triple modified (cost: {match_cost})",
                    "cost": match_cost,
                    "detail": f"'{t_cand_str}' -> '{t_corr_str}'"
                })
    else:
        matched_cand_indices = set()
        matched_corr_indices = set()

    added_triples_details = [" ".join(t_corr) for i, t_corr in enumerate(corr_triples) if i not in matched_corr_indices]
    if added_triples_details:
        cost = len(added_triples_details) * TRIPLE_COST
        total_cost += cost
        edits_breakdown.append({
            "change": f"{len(added_triples_details)} Triple(s) added",
            "cost": cost,
            "detail": "\n".join([f"    - {t}" for t in added_triples_details])
        })

    removed_triples_details = [" ".join(t_cand) for i, t_cand in enumerate(cand_triples) if i not in matched_cand_indices]
    if removed_triples_details:
        edits_breakdown.append({
            "change": f"{len(removed_triples_details)} Triple(s) removed (no cost)",
            "cost": 0,
            "detail": "\n".join([f"    - {t}" for t in removed_triples_details])
        })

    # Compare algebraic structures
    algebra_types_to_check = {k: v for k, v in ALGEBRA_COST_RULES.items() if k not in ['Distinct', 'OrderBy', 'Slice']}

    for comp_type, rule_cost in algebra_types_to_check.items():
        if rule_cost <= 0:
            continue

        cand_counter = cand_comps.get(comp_type, collections.Counter())
        corr_counter = corr_comps.get(comp_type, collections.Counter())

        added_or_changed = corr_counter - cand_counter
        num_additions_or_changes = sum(added_or_changed.values())

        if num_additions_or_changes > 0:
            cost = num_additions_or_changes * rule_cost
            total_cost += cost
            edits_breakdown.append({
                "change": f"Structure '{comp_type}' added or modified",
                "cost": cost,
                "detail": f"{num_additions_or_changes} instance(s)"
            })

        removed_or_changed = cand_counter - corr_counter
        num_removals_or_changes = sum(removed_or_changed.values())
        if num_removals_or_changes > 0:
            edits_breakdown.append({
                "change": f"Structure '{comp_type}' removed or modified",
                "cost": 0,
                "detail": f"{num_removals_or_changes} instance(s) (no cost)"
            })

    # Projection variables
    cand_vars = cand_comps.get('ProjectionVars', set())
    corr_vars = corr_comps.get('ProjectionVars', set())
    missing_vars = corr_vars - cand_vars
    if missing_vars:
        cost = len(missing_vars) * PROJECTION_VAR_COST
        total_cost += cost
        edits_breakdown.append({
            "change": f"{len(missing_vars)} Projection Variable(s) added",
            "cost": cost,
            "detail": f"Added: {', '.join(missing_vars)}"
        })

    removed_vars = cand_vars - corr_vars
    if removed_vars:
        edits_breakdown.append({
            "change": f"{len(removed_vars)} Projection Variable(s) removed (no cost)",
            "cost": 0,
            "detail": f"Removed: {', '.join(removed_vars)}"
        })

    return {"total_edits": int(total_cost), "edits": edits_breakdown}


def calculate_max_edit_cost_deterministically(query_string: str, request_id: str) -> Optional[Dict[str, Any]]:
    """
    Calculate the maximum possible edit cost for a query.

    This represents the cost of constructing the query from scratch.
    """
    log_prefix = f"[{request_id}]"
    logger.info(f"{log_prefix} Starting max_edit_cost calculation.")

    corr_comps = _get_query_components(query_string, normalize_variables=False)
    if not corr_comps:
        return None

    total_cost = 0
    cost_breakdown = []

    num_triples = len(corr_comps.get('Triples', []))
    if num_triples > 0:
        cost = num_triples * TRIPLE_COST
        total_cost += cost
        cost_breakdown.append({"component": "Triples", "cost": cost, "detail": f"{num_triples} triples"})

    num_vars = len(corr_comps.get('ProjectionVars', set()))
    if num_vars > 0:
        cost = num_vars * PROJECTION_VAR_COST
        total_cost += cost
        cost_breakdown.append({"component": "Projection Variables", "cost": cost, "detail": f"{num_vars} variables"})

    for comp_type, rule_cost in ALGEBRA_COST_RULES.items():
        if rule_cost > 0:
            count = len(corr_comps.get(comp_type, []))
            if count > 0:
                cost = count * rule_cost
                total_cost += cost
                cost_breakdown.append({"component": f"Structure: {comp_type}", "cost": cost, "detail": f"{count} instance(s)"})

    return {"max_edit_cost": total_cost, "cost_breakdown": cost_breakdown}


def calculate_diff_cost_no_normalization(
    generated_query_str: str,
    corrected_query_str: str,
    request_id: str
) -> Optional[Dict[str, Any]]:
    """
    Calculate edit cost WITHOUT variable normalization.

    Used for queries where variables are already normalized.
    """
    log_prefix = f"[{request_id}]"
    logger.info(f"{log_prefix} Starting diff_cost calculation without normalization.")

    cand_comps = _get_query_components(generated_query_str, normalize_variables=False)
    corr_comps = _get_query_components(corrected_query_str, normalize_variables=False)

    if not cand_comps or not corr_comps:
        logger.error(f"{log_prefix} Could not extract components from one or both queries.")
        return None

    total_cost = 0
    edits_breakdown = []

    textual_cost, textual_edits = _compare_simple_clauses_textually(
        generated_query_str, corrected_query_str, request_id
    )
    total_cost += textual_cost
    edits_breakdown.extend(textual_edits)

    cand_triples = list(cand_comps.get('Triples', []))
    corr_triples = list(corr_comps.get('Triples', []))

    if cand_triples and corr_triples:
        cost_matrix = np.zeros((len(cand_triples), len(corr_triples)))
        for i, t_cand in enumerate(cand_triples):
            for j, t_corr in enumerate(corr_triples):
                cost_matrix[i, j] = sum(a != b for a, b in zip(t_cand, t_corr))
        cand_indices, corr_indices = linear_sum_assignment(cost_matrix)
        matched_cand_indices = set(cand_indices)
        matched_corr_indices = set(corr_indices)
        for i in range(len(cand_indices)):
            c_idx, r_idx = cand_indices[i], corr_indices[i]
            match_cost = int(cost_matrix[c_idx, r_idx])
            if match_cost > 0:
                total_cost += match_cost
                t_cand_str = " ".join(cand_triples[c_idx])
                t_corr_str = " ".join(corr_triples[r_idx])
                edits_breakdown.append({
                    "change": f"Triple modified (cost: {match_cost})",
                    "cost": match_cost,
                    "detail": f"'{t_cand_str}' -> '{t_corr_str}'"
                })
    else:
        matched_cand_indices = set()
        matched_corr_indices = set()

    added_triples_details = [" ".join(t_corr) for i, t_corr in enumerate(corr_triples) if i not in matched_corr_indices]
    if added_triples_details:
        cost = len(added_triples_details) * TRIPLE_COST
        total_cost += cost
        edits_breakdown.append({
            "change": f"{len(added_triples_details)} Triple(s) added",
            "cost": cost,
            "detail": "\n".join([f"    - {t}" for t in added_triples_details])
        })

    removed_triples_details = [" ".join(t_cand) for i, t_cand in enumerate(cand_triples) if i not in matched_cand_indices]
    if removed_triples_details:
        edits_breakdown.append({
            "change": f"{len(removed_triples_details)} Triple(s) removed (no cost)",
            "cost": 0,
            "detail": "\n".join([f"    - {t}" for t in removed_triples_details])
        })

    algebra_types_to_check = {k: v for k, v in ALGEBRA_COST_RULES.items() if k not in ['Distinct', 'OrderBy', 'Slice']}

    for comp_type, rule_cost in algebra_types_to_check.items():
        if rule_cost <= 0:
            continue

        cand_counter = cand_comps.get(comp_type, collections.Counter())
        corr_counter = corr_comps.get(comp_type, collections.Counter())

        added_or_changed = corr_counter - cand_counter
        num_additions_or_changes = sum(added_or_changed.values())

        if num_additions_or_changes > 0:
            cost = num_additions_or_changes * rule_cost
            total_cost += cost
            edits_breakdown.append({
                "change": f"Structure '{comp_type}' added or modified",
                "cost": cost,
                "detail": f"{num_additions_or_changes} instance(s)"
            })

        removed_or_changed = cand_counter - corr_counter
        num_removals_or_changes = sum(removed_or_changed.values())
        if num_removals_or_changes > 0:
            edits_breakdown.append({
                "change": f"Structure '{comp_type}' removed or modified",
                "cost": 0,
                "detail": f"{num_removals_or_changes} instance(s) (no cost)"
            })

    cand_vars = cand_comps.get('ProjectionVars', set())
    corr_vars = corr_comps.get('ProjectionVars', set())
    missing_vars = corr_vars - cand_vars
    if missing_vars:
        cost = len(missing_vars) * PROJECTION_VAR_COST
        total_cost += cost
        edits_breakdown.append({
            "change": f"{len(missing_vars)} Projection Variable(s) added",
            "cost": cost,
            "detail": f"Added: {', '.join(missing_vars)}"
        })

    removed_vars = cand_vars - corr_vars
    if removed_vars:
        edits_breakdown.append({
            "change": f"{len(removed_vars)} Projection Variable(s) removed (no cost)",
            "cost": 0,
            "detail": f"Removed: {', '.join(removed_vars)}"
        })

    return {"total_edits": int(total_cost), "edits": edits_breakdown}
