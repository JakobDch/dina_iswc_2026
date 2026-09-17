"""
SPARQL query execution against OnTop endpoint.

Provides functions for executing SPARQL queries with proper error handling
and syntax validation.
"""

import asyncio
import logging
import re
from typing import Optional, Dict, Any

import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)


def _add_limit_to_sparql_query(query: str, limit: int = 5) -> str:
    """Add or replace a LIMIT clause in a SPARQL query."""
    limit_regex = re.compile(r'(LIMIT\s+\d+)(\s+OFFSET\s+\d+)?\s*$', re.IGNORECASE | re.DOTALL)
    if limit_regex.search(query):
        return limit_regex.sub(f'LIMIT {limit}\\2', query)
    else:
        return query.strip() + f"\nLIMIT {limit}"


async def validate_sparql_syntax_with_ontop(query: str) -> bool:
    """
    Validate SPARQL syntax by sending query to OnTop with LIMIT 0.

    Returns True if syntax is valid, False otherwise.
    """
    settings = get_settings()
    ontop_url = settings.ontop_sparql_url

    # Add LIMIT 0 to avoid returning data while validating syntax
    validation_query = _add_limit_to_sparql_query(query, limit=0)

    headers = {"Accept": "application/sparql-results+json"}
    params = {"query": validation_query}

    try:
        async with httpx.AsyncClient(timeout=settings.ontop_timeout_seconds) as client:
            response = await client.get(ontop_url, params=params, headers=headers)
            if response.status_code == 200:
                return True
            elif response.status_code == 400:
                logger.warning(f"Syntax validation failed: {response.text[:200]}")
                return False
            else:
                logger.error(f"Unexpected status {response.status_code} during syntax validation")
                return False
    except httpx.RequestError as e:
        logger.error(f"Network error during syntax validation: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during syntax validation: {e}")
        return False


async def execute_sparql_in_ontop(
    client: httpx.AsyncClient,
    query: str,
    delay: float = 0.0,
    timeout: float = None
) -> Optional[Dict]:
    """
    Execute SPARQL query against OnTop endpoint.

    Args:
        client: HTTP client instance
        query: SPARQL query string
        delay: Optional delay before execution (seconds)
        timeout: Optional timeout override

    Returns:
        Query results as dict, or dict with error_details for syntax errors, or None for other errors
    """
    settings = get_settings()

    if delay > 0:
        await asyncio.sleep(delay)

    ontop_url = settings.ontop_sparql_url
    effective_timeout = timeout if timeout is not None else settings.ontop_timeout_seconds
    headers = {"Accept": "application/sparql-results+json"}
    params = {"query": query}

    try:
        response = await client.get(
            ontop_url,
            params=params,
            headers=headers,
            timeout=effective_timeout
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as e:
        logger.error(f"OnTop query failed with status {e.response.status_code}")
        error_body = e.response.text

        if e.response.status_code == 400:
            return {
                "error_details": {
                    "error_type": "syntax_error",
                    "status_code": 400,
                    "error_message": error_body.strip(),
                    "query_fragment": query[:200] if query else ""
                }
            }
        return None

    except httpx.TimeoutException as e:
        logger.error(f"OnTop query timed out after {effective_timeout}s: {e}")
        return {
            "error_details": {
                "error_type": "timeout",
                "error_message": f"Query timed out after {effective_timeout} seconds",
                "query_fragment": query[:200] if query else ""
            }
        }

    except httpx.RequestError as e:
        logger.error(f"OnTop request failed: {e}")
        return None


async def execute_sparql_query(
    query: str,
    delay: float = 0.0,
    timeout: float = None
) -> Optional[Dict]:
    """
    Execute SPARQL query against OnTop endpoint.

    Convenience function that creates its own client.

    Args:
        query: SPARQL query string
        delay: Optional delay before execution (seconds)
        timeout: Optional timeout override

    Returns:
        Query results as dict, or dict with error_details, or None for errors
    """
    settings = get_settings()
    effective_timeout = timeout if timeout is not None else settings.ontop_timeout_seconds

    async with httpx.AsyncClient(timeout=effective_timeout) as client:
        return await execute_sparql_in_ontop(client, query, delay, timeout)


async def execute_sparql_count_query(
    query: str,
    timeout: float = None
) -> Optional[int]:
    """
    Execute a SPARQL query and return the count of results.

    Wraps the query in a SELECT (COUNT(*) AS ?count) if not already a count query.

    Args:
        query: SPARQL query string
        timeout: Optional timeout override

    Returns:
        Number of results, or None on error
    """
    # Check if query is already a COUNT query
    if not re.search(r'\bCOUNT\s*\(', query, re.IGNORECASE):
        # Wrap in COUNT
        # Find the SELECT clause and variables
        select_match = re.match(r'(.*?\bSELECT\s+)(DISTINCT\s+)?(.+?)(\bWHERE\b.*)', query, re.IGNORECASE | re.DOTALL)
        if select_match:
            prefix = select_match.group(1)
            distinct = select_match.group(2) or ""
            where_clause = select_match.group(4)
            query = f"{prefix}(COUNT(*) AS ?count) {where_clause}"

    result = await execute_sparql_query(query, timeout=timeout)

    if result and "results" in result:
        bindings = result.get("results", {}).get("bindings", [])
        if bindings:
            count_value = bindings[0].get("count", {}).get("value")
            if count_value is not None:
                return int(count_value)

    return None


def extract_results_as_list(sparql_result: Dict[str, Any]) -> list:
    """
    Extract SPARQL results as a list of dictionaries.

    Args:
        sparql_result: Raw SPARQL JSON result

    Returns:
        List of result rows, each as a dict mapping variable names to values
    """
    if not sparql_result or "results" not in sparql_result:
        return []

    bindings = sparql_result.get("results", {}).get("bindings", [])
    results = []

    for binding in bindings:
        row = {}
        for var_name, var_data in binding.items():
            row[var_name] = var_data.get("value")
        results.append(row)

    return results
