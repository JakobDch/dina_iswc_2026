"""
Multi-Step Query Tools for decomposition and transformation.

These tools are used by the Multi-Step Agent to:
1. Analyze failed queries to identify bottlenecks and suggest decomposition
2. Execute individual step queries against endpoints
3. Test Python transformation scripts
4. Validate step connections (join keys)
"""

import json
import logging
import re
import traceback
from collections import defaultdict
from typing import Any

import httpx
from langchain_core.tools import tool

from src.tools.sparql_tools import get_endpoint_for_dataset, QUERY_TIMEOUT
from src.tracing import TraceEvent, TraceEventType, get_tracer

logger = logging.getLogger(__name__)


# =============================================================================
# Query Analysis Functions (for analyze_failed_query tool)
# =============================================================================

def _remove_nested_construct(text: str, keyword: str) -> str:
    """Remove FILTER(...) or OPTIONAL{...} including nested parens/braces."""
    result = []
    i = 0
    while i < len(text):
        if text[i:i+len(keyword)].upper() == keyword.upper():
            j = i + len(keyword)
            # Skip whitespace
            while j < len(text) and text[j] in ' \t\n':
                j += 1
            if j < len(text) and text[j] in '({':
                open_char = text[j]
                close_char = ')' if open_char == '(' else '}'
                count = 1
                k = j + 1
                while k < len(text) and count > 0:
                    if text[k] == open_char:
                        count += 1
                    elif text[k] == close_char:
                        count -= 1
                    k += 1
                i = k
                continue
        result.append(text[i])
        i += 1
    return ''.join(result)


def _extract_filters(where_clause: str) -> list[str]:
    """Extract FILTER expressions from WHERE clause."""
    filters = []
    i = 0
    text = where_clause
    while True:
        idx = text.upper().find('FILTER', i)
        if idx == -1:
            break
        j = idx + 6
        while j < len(text) and text[j] in ' \t\n':
            j += 1
        if j < len(text) and text[j] == '(':
            count = 1
            k = j + 1
            while k < len(text) and count > 0:
                if text[k] == '(':
                    count += 1
                elif text[k] == ')':
                    count -= 1
                k += 1
            filters.append(text[j+1:k-1])
            i = k
        else:
            i = idx + 1
    return filters


def _parse_sparql_query(query: str) -> dict:
    """Parse SPARQL query and extract structure."""
    # Extract prefixes
    prefixes = dict(re.findall(r'PREFIX\s+(\w+):\s*<([^>]+)>', query, re.I))

    # Extract WHERE clause
    match = re.search(r'WHERE\s*\{(.+)\}', query, re.I | re.DOTALL)
    if not match:
        return {'error': 'No WHERE clause found'}
    where_clause = match.group(1)

    # Extract filters before removing them
    filters = _extract_filters(where_clause)

    # Remove OPTIONAL and FILTER for pattern extraction
    clean = _remove_nested_construct(where_clause, 'OPTIONAL')
    clean = _remove_nested_construct(clean, 'FILTER')

    # Parse triple patterns
    patterns = []
    for part in clean.split('.'):
        part = ' '.join(part.split())  # Normalize whitespace
        # Match: ?var predicate ?var/prefix:local/"literal"
        m = re.match(r'(\?\w+)\s+(a|[\w:]+)\s+(\?\w+|[\w:]+|"[^"]+")', part)
        if m:
            patterns.append({
                'subject': m.group(1),
                'predicate': m.group(2),
                'object': m.group(3)
            })

    # Build variable dependency graph
    var_usage = {}  # var -> list of pattern indices where it appears
    for i, p in enumerate(patterns):
        for term in [p['subject'], p['object']]:
            if term.startswith('?'):
                var = term[1:]
                if var not in var_usage:
                    var_usage[var] = []
                var_usage[var].append(i)

    return {
        'prefixes': prefixes,
        'patterns': patterns,
        'filters': filters,
        'var_usage': var_usage
    }


def _get_predicate_cardinalities(endpoint: str, prefixes: dict, predicates: list[str], timeout: float = 10.0) -> dict:
    """Get cardinality for each predicate. Timeout means VERY large (>1M)."""
    prefix_str = '\n'.join(f'PREFIX {k}: <{v}>' for k, v in prefixes.items())
    counts = {}

    for pred in predicates:
        if pred == 'a':
            continue
        query = f'{prefix_str}\nSELECT (COUNT(*) AS ?c) WHERE {{ ?s {pred} ?o }}'
        try:
            resp = httpx.post(
                endpoint,
                data={'query': query},
                headers={'Accept': 'application/sparql-results+json'},
                timeout=timeout
            )
            if resp.status_code == 200:
                bindings = resp.json().get('results', {}).get('bindings', [])
                if bindings:
                    counts[pred] = int(bindings[0]['c']['value'])
            else:
                counts[pred] = f'ERROR:{resp.status_code}'
        except httpx.ReadTimeout:
            counts[pred] = 'TIMEOUT'  # Timeout = very large table (>1M rows)
        except Exception as e:
            counts[pred] = f'ERROR:{type(e).__name__}'

    return counts


@tool
def analyze_failed_query(query: str, dataset: str) -> str:
    """
    Analyze a failed SPARQL query to provide facts about its structure.

    This tool provides RAW DATA only - no decomposition suggestions.
    The agent must decide how to use this information.

    Returns:
        - patterns: Triple patterns with their cardinalities (TIMEOUT = very large)
        - filters: FILTER expressions found in the query
        - var_usage: Which variables appear in which patterns
    """
    endpoint = get_endpoint_for_dataset(dataset)
    if not endpoint:
        return json.dumps({
            "success": False,
            "error": f"Unknown dataset: {dataset}. Available: edu, trn, nrg, bsbm",
        })

    # Parse the query
    parsed = _parse_sparql_query(query)
    if 'error' in parsed:
        return json.dumps({"success": False, "error": parsed['error']})

    # Get unique predicates
    predicates = list(set(p['predicate'] for p in parsed['patterns']))

    # Get cardinalities
    cardinalities = _get_predicate_cardinalities(endpoint, parsed['prefixes'], predicates)

    # Annotate patterns with cardinalities
    annotated_patterns = []
    for i, p in enumerate(parsed['patterns']):
        pred = p['predicate']
        card = cardinalities.get(pred, 'unknown')

        annotated_patterns.append({
            'index': i,
            'subject': p['subject'],
            'predicate': p['predicate'],
            'object': p['object'],
            'cardinality': card if card != 'TIMEOUT' else '>1M (TIMEOUT)',
        })

    return json.dumps({
        "success": True,
        "dataset": dataset,
        "patterns": annotated_patterns,
        "filters": parsed['filters'],
        "var_usage": parsed['var_usage'],
    }, indent=2)


# Global tracer instance
_tracer = None

def _get_tracer():
    """Get or initialize the tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = get_tracer()
    return _tracer

# Module-level storage for large query results (to avoid token explosion)
# Results are stored here and only summaries are returned to the LLM
_STEP_RESULTS_CACHE: dict[str, list[dict]] = {}


@tool
def execute_step_query(query: str, dataset: str, step_name: str, timeout: float = 30.0) -> str:
    """
    Execute a single SPARQL step query against a specific dataset endpoint.

    Use this tool to test individual step queries before combining them.
    Each step should be a simple query without complex JOINs.

    Args:
        query: Complete SPARQL query with PREFIX declarations
        dataset: Dataset ID (edu, trn, nrg, bsbm)
        step_name: Unique name for this step (e.g., "stops", "trips", "routes").
                   This name will be used in the transformation script to access results.
        timeout: Query timeout in seconds (default: 30)

    Returns:
        JSON string with:
        - success: True/False
        - step_name: The step name for reference
        - results: List of result bindings (if success)
        - count: Number of results
        - variables: List of variables in results
        - error: Error message (if failed)
        - sample: First 5 results for preview
    """
    endpoint = get_endpoint_for_dataset(dataset)
    if not endpoint:
        return json.dumps({
            "success": False,
            "error": f"Unknown dataset: {dataset}. Available: edu, trn, nrg, bsbm",
        })

    try:
        response = httpx.post(
            endpoint,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=timeout,
        )

        if response.status_code == 400:
            return json.dumps({
                "success": False,
                "error": f"SPARQL syntax error: {response.text[:500]}",
            })

        response.raise_for_status()
        result = response.json()
        bindings = result.get("results", {}).get("bindings", [])
        variables = result.get("head", {}).get("vars", [])

        # Store full results in cache using step_name as key
        _STEP_RESULTS_CACHE[step_name] = bindings

        # Emit tracing event for successful step query
        tracer = _get_tracer()
        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.MULTI_STEP_STEP,
                phase="generation",
                agent="multi_step_agent",
                data={
                    "step_name": step_name,
                    "query": query,
                    "dataset": dataset,
                    "result_count": len(bindings),
                    "variables": variables,
                    "sample_results": bindings[:10],
                },
            )
        )

        # Only return summary + sample to LLM
        return json.dumps({
            "success": True,
            "step_name": step_name,  # Echo back for confirmation
            "count": len(bindings),
            "variables": variables,
            "sample": bindings[:10],  # Show first 10 for context
        })

    except httpx.ReadTimeout:
        return json.dumps({
            "success": False,
            "error": f"Query timeout after {timeout}s. The query may still be too complex - try simplifying further.",
        })
    except httpx.HTTPStatusError as e:
        return json.dumps({
            "success": False,
            "error": f"HTTP error {e.response.status_code}: {e.response.text[:300]}",
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Execution error: {str(e)}",
        })


@tool
def test_transform_script(script: str, step_results_json: str) -> str:
    """
    Test a Python transformation script with actual step results.

    The script should define a `transform(step_results)` function that:
    - Takes a dict mapping step names to result lists
    - Returns a list of aggregated result dicts

    Example script:
    ```python
    from collections import defaultdict

    def transform(step_results):
        # Build index from step 1
        index = defaultdict(set)
        for row in step_results["step1"]:
            key = row["course"]["value"]
            val = row["prof"]["value"]
            index[key].add(val)

        # Aggregate from step 2
        counts = defaultdict(int)
        for row in step_results["step2"]:
            course = row["course"]["value"]
            for prof in index.get(course, []):
                counts[prof] += 1

        # Format results
        return [{"prof": p, "count": c} for p, c in counts.items()]
    ```

    Args:
        script: Python code defining transform(step_results) function
        step_results_json: JSON string with step results, e.g.:
            {"step1": [{"var": {"type": "uri", "value": "..."}}, ...], "step2": [...]}

    Returns:
        JSON string with:
        - success: True/False
        - results: Transformed results (if success)
        - count: Number of result rows
        - error: Error message (if failed)
        - traceback: Full traceback (if failed)
    """
    try:
        step_results = json.loads(step_results_json)
    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON for step_results: {str(e)}",
        })

    # Create execution environment with common imports pre-loaded
    import re as re_module
    safe_globals = {
        "__builtins__": {
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "range": range,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "True": True,
            "False": False,
            "None": None,
            "print": print,  # For debugging
        },
        # Pre-imported modules and classes (no need for import statements)
        "defaultdict": defaultdict,
        "collections": type("collections", (), {"defaultdict": defaultdict})(),
        "re": re_module,
    }
    safe_locals: dict[str, Any] = {}

    # Pre-process script: remove 'from collections import defaultdict' since it's pre-loaded
    import re
    script = re.sub(r'from\s+collections\s+import\s+defaultdict\s*\n?', '', script)
    script = re.sub(r'import\s+collections\s*\n?', '', script)

    try:
        # Execute the script to define the transform function
        exec(script, safe_globals, safe_locals)

        if "transform" not in safe_locals:
            return json.dumps({
                "success": False,
                "error": "Script must define a 'transform(step_results)' function",
            })

        transform_fn = safe_locals["transform"]

        # Execute the transform function
        results = transform_fn(step_results)

        if not isinstance(results, list):
            return json.dumps({
                "success": False,
                "error": f"transform() must return a list, got {type(results).__name__}",
            })

        # Emit tracing event for successful transformation
        tracer = _get_tracer()
        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.MULTI_STEP_TRANSFORM,
                phase="generation",
                agent="multi_step_agent",
                data={
                    "transform_script": script,
                    "input_steps": list(step_results.keys()),
                    "input_counts": {k: len(v) for k, v in step_results.items()},
                    "output_count": len(results),
                    "sample_output": results[:15],
                },
            )
        )

        return json.dumps({
            "success": True,
            "results": results,
            "count": len(results),
            "sample": results[:5] if len(results) > 5 else results,
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })


@tool
def validate_step_connection(
    step1_name: str,
    step1_vars: str,
    step2_name: str,
    step2_vars: str,
    join_key: str,
) -> str:
    """
    Validate that two steps can be connected via a join key.

    This checks that both steps have the join_key variable in their output.

    Args:
        step1_name: Name of the first step
        step1_vars: Comma-separated list of variables from step 1
        step2_name: Name of the second step
        step2_vars: Comma-separated list of variables from step 2
        join_key: Variable name to use for joining

    Returns:
        JSON string with:
        - valid: True/False
        - message: Explanation
        - step1_has_key: Whether step 1 has the join key
        - step2_has_key: Whether step 2 has the join key
    """
    vars1 = [v.strip() for v in step1_vars.split(",")]
    vars2 = [v.strip() for v in step2_vars.split(",")]

    step1_has = join_key in vars1
    step2_has = join_key in vars2

    if step1_has and step2_has:
        return json.dumps({
            "valid": True,
            "message": f"Both {step1_name} and {step2_name} have '{join_key}' - can be joined.",
            "step1_has_key": True,
            "step2_has_key": True,
        })
    elif not step1_has and not step2_has:
        return json.dumps({
            "valid": False,
            "message": f"Neither step has '{join_key}'. {step1_name} has: {vars1}. {step2_name} has: {vars2}.",
            "step1_has_key": False,
            "step2_has_key": False,
        })
    else:
        missing = step1_name if not step1_has else step2_name
        present = step1_name if step1_has else step2_name
        return json.dumps({
            "valid": False,
            "message": f"'{join_key}' is in {present} but not in {missing}. Add it to {missing}'s SELECT clause.",
            "step1_has_key": step1_has,
            "step2_has_key": step2_has,
        })


def get_cached_results(step_name: str) -> list[dict] | None:
    """Retrieve full results from cache by step_name."""
    return _STEP_RESULTS_CACHE.get(step_name)


def get_all_cached_results() -> dict[str, list[dict]]:
    """Retrieve all cached step results.

    Returns:
        Dictionary mapping step_name -> list of result bindings
    """
    return dict(_STEP_RESULTS_CACHE)


def clear_results_cache() -> None:
    """Clear the results cache (call after processing is complete)."""
    _STEP_RESULTS_CACHE.clear()


# Collect all tools for the Multi-Step Agent
# NOTE: analyze_failed_query is NOT in this list - it's called automatically
# before the LLM loop starts (see multi_step_agent.py decompose_and_transform)
MULTI_STEP_TOOLS = [
    execute_step_query,
    test_transform_script,
    validate_step_connection,
]


__all__ = [
    "analyze_failed_query",
    "execute_step_query",
    "test_transform_script",
    "validate_step_connection",
    "MULTI_STEP_TOOLS",
    "get_cached_results",
    "clear_results_cache",
]
