"""
SPARQL generation and execution tools with multi-endpoint support for Dataspaces.
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path

import httpx
from langchain_core.tools import tool

from src.config import USE_ONTOP_REPLICAS, get_ontop_endpoint, get_settings
from src.utils.query_validator import validate_query, ValidationSeverity
from src.utils.query_metadata import extract_query_metadata

# Execution result limit to prevent OutOfMemoryError on OnTop

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 30.0

# Query timeout in seconds
# OnTop queries are either fast (<2s) or will timeout anyway (60s MySQL limit)
# Using 15s client-side timeout saves ~45s per problematic query
QUERY_TIMEOUT = 15.0

# Exceptions that should trigger retry (transient failures)
# Note: ReadTimeout is NOT retryable - it means the query is too expensive, not a transient issue
RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    ConnectionResetError,
    ConnectionRefusedError,
)

# Timeout exceptions that should NOT be retried (query is too expensive)
NON_RETRYABLE_TIMEOUT = (httpx.ReadTimeout,)


def is_retryable_http_status(status_code: int) -> bool:
    """Check if an HTTP status code is retryable (5xx server errors, 429 rate limit).

    Note: HTTP 500 is excluded because OnTop returns 500 for unsupported SPARQL features
    (Property Paths with *, +, ?, SERVICE queries) - these are query errors, not transient failures.
    """
    return status_code in (429, 502, 503, 504)  # 500 excluded - usually query syntax issues

# Singleton AsyncClient for connection pooling
_async_client: httpx.AsyncClient | None = None


def get_async_client() -> httpx.AsyncClient:
    """Get or create the async HTTP client singleton."""
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(timeout=QUERY_TIMEOUT)
    return _async_client


async def close_async_client() -> None:
    """Close the async HTTP client (call on application shutdown)."""
    global _async_client
    if _async_client is not None:
        await _async_client.aclose()
        _async_client = None


def load_dataset_registry() -> dict:
    """Load the dataset registry from JSON file."""
    registry_path = Path(__file__).parent.parent.parent / "data" / "datasets" / "registry.json"
    if registry_path.exists():
        with open(registry_path) as f:
            return json.load(f)
    return {"datasets": []}


def get_endpoint_for_dataset(dataset_id: str, use_load_balancing: bool = True) -> str | None:
    """
    Get the SPARQL endpoint URL for a specific dataset ID (case-insensitive).

    Respects the current dataset_size setting (small/large) via get_ontop_endpoint().
    When USE_ONTOP_REPLICAS is enabled and use_load_balancing is True,
    returns the next endpoint from the round-robin pool for parallel execution.

    Args:
        dataset_id: Dataset identifier (as defined in the dataset registry)
        use_load_balancing: If True and replicas are configured, use load balancing

    Returns:
        Endpoint URL or None if dataset not found.
    """
    dataset_id_lower = dataset_id.lower()

    # Always use get_ontop_endpoint which respects dataset_size context
    # This ensures small/large endpoint selection works correctly
    endpoint = get_ontop_endpoint(dataset_id_lower)
    if endpoint:
        return endpoint

    # Fallback to registry lookup (for datasets not in ONTOP_SINGLE_ENDPOINTS)
    registry = load_dataset_registry()
    for dataset in registry.get("datasets", []):
        if dataset["id"].lower() == dataset_id_lower:
            return dataset["sparql_endpoint"]
    return None


def get_all_endpoints() -> list[str]:
    """Get all available SPARQL endpoints from the registry."""
    registry = load_dataset_registry()
    return [d["sparql_endpoint"] for d in registry.get("datasets", [])]


def get_endpoints_for_query(query: str) -> list[str]:
    """
    Determine which endpoints to query based on the prefixes used in a SPARQL query.

    Analyzes PREFIX declarations in the query and matches them against
    the ontology_prefix values in the dataset registry.

    When USE_ONTOP_REPLICAS is enabled, returns load-balanced endpoints
    from the replica pool for each matched dataset.

    Args:
        query: The SPARQL query to analyze

    Returns:
        List of endpoint URLs that match the prefixes used in the query.
        Returns empty list if no matches found (caller should use default endpoint).
    """
    import re

    # Extract all PREFIX declarations from the query
    # Matches: PREFIX name: <uri>
    prefix_pattern = re.compile(r'PREFIX\s+(\w+):\s*<([^>]+)>', re.IGNORECASE)
    declared_prefixes = prefix_pattern.findall(query)

    if not declared_prefixes:
        return []

    # Get URIs from declared prefixes
    declared_uris = {uri for _, uri in declared_prefixes}

    # Load registry and match against ontology_prefix
    # Respect dataset_size_context to prevent small queries hitting large endpoints
    from src.config import get_dataset_size
    current_size = get_dataset_size()

    registry = load_dataset_registry()
    matched_dataset_ids = set()

    for dataset in registry.get("datasets", []):
        ontology_prefix = dataset.get("ontology_prefix", "")
        if not ontology_prefix:
            continue

        dataset_id = dataset["id"]

        # STRICT: When size is "small", NEVER include "-large" datasets
        if current_size == "small" and ("-large" in dataset_id or "_large" in dataset_id):
            continue

        # Check if any declared prefix URI matches or starts with the ontology prefix
        for uri in declared_uris:
            # Exact match or the declared URI starts with the ontology prefix
            if uri == ontology_prefix or uri.startswith(ontology_prefix.rstrip('#/')):
                matched_dataset_ids.add(dataset_id)
                break

    # Get endpoints with load balancing support
    endpoints = []
    for dataset_id in matched_dataset_ids:
        endpoint = get_endpoint_for_dataset(dataset_id, use_load_balancing=True)
        if endpoint:
            endpoints.append(endpoint)

    return endpoints


@tool
def execute_sparql(
    query: str,
    endpoints: list[str] | None = None,
    endpoint_url: str | None = None,
    dataset_ids: list[str] | None = None,
) -> dict:
    """
    Execute a SPARQL query against one or multiple OnTop endpoints.

    Args:
        query: The SPARQL query to execute
        endpoints: List of endpoint URLs to query (preferred for multi-endpoint)
        endpoint_url: Single endpoint URL (backwards compatible)
        dataset_ids: List of dataset IDs to query (alternative to endpoints)

    Returns:
        Dict with success status, merged results from all endpoints,
        and information about which endpoints were queried
    """
    settings = get_settings()

    # Determine which endpoints to query
    urls = []
    if endpoints:
        urls = endpoints
    elif dataset_ids:
        for dataset_id in dataset_ids:
            endpoint = get_endpoint_for_dataset(dataset_id)
            if endpoint:
                urls.append(endpoint)
    elif endpoint_url:
        urls = [endpoint_url]
    else:
        urls = [settings.ontop_sparql_url]

    if not urls:
        return {
            "success": False,
            "error_type": "no_endpoints",
            "error_message": "No valid endpoints found to query",
        }

    # Pre-execution validation: Check for known problematic patterns
    validation_result = validate_query(query)
    if validation_result.severity == ValidationSeverity.ERROR:
        # Query will definitely fail - don't waste time executing
        return {
            "success": False,
            "error_type": "validation_error",
            "error_message": f"VALIDATION ERROR: {validation_result.message}",
            "validation_pattern": validation_result.matched_pattern,
            "suggestion": validation_result.suggestion,
            "queried_endpoints": [],
            "skipped_execution": True,
        }

    all_results = []
    errors = []
    successful_endpoints = []
    total_retries = 0

    # Include validation warning in response if present
    validation_warning = None
    if validation_result.severity == ValidationSeverity.WARNING:
        validation_warning = {
            "pattern": validation_result.matched_pattern,
            "message": validation_result.message,
            "suggestion": validation_result.suggestion,
        }

    headers = {"Accept": "application/sparql-results+json"}
    params = {"query": query}

    for url in urls:
        last_error = None
        wait_time = INITIAL_BACKOFF
        endpoint_success = False

        for attempt in range(MAX_RETRIES):
            try:
                with httpx.Client(timeout=QUERY_TIMEOUT) as client:
                    response = client.get(url, params=params, headers=headers)
                    response.raise_for_status()
                    result = response.json()
                    bindings = result.get("results", {}).get("bindings", [])
                    # Tag results with source endpoint
                    for binding in bindings:
                        binding["_source_endpoint"] = url
                    all_results.extend(bindings)
                    successful_endpoints.append(url)
                    endpoint_success = True
                    break  # Success, exit retry loop

            except NON_RETRYABLE_TIMEOUT as e:
                # Query timeout - no point retrying, query is too expensive
                errors.append({
                    "endpoint": url,
                    "error_type": "query_timeout",
                    "error_message": f"Query timed out after {QUERY_TIMEOUT}s - query is too expensive for OnTop",
                })
                break  # Don't retry

            except RETRYABLE_EXCEPTIONS as e:
                total_retries += 1
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"Retry {attempt + 1}/{MAX_RETRIES} for {url}: {type(e).__name__}: {e}"
                    )
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 2, MAX_BACKOFF)
                continue

            except httpx.HTTPStatusError as e:
                if is_retryable_http_status(e.response.status_code) and attempt < MAX_RETRIES - 1:
                    total_retries += 1
                    last_error = e
                    logger.warning(
                        f"Retry {attempt + 1}/{MAX_RETRIES} for {url} after HTTP {e.response.status_code}"
                    )
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 2, MAX_BACKOFF)
                    continue
                # Non-retryable error
                errors.append({
                    "endpoint": url,
                    "error_type": "http_error",
                    "error_message": f"HTTP {e.response.status_code}: {e.response.text[:500]}",
                })
                break

            except httpx.RequestError as e:
                total_retries += 1
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"Retry {attempt + 1}/{MAX_RETRIES} for {url}: {type(e).__name__}: {e}"
                    )
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 2, MAX_BACKOFF)
                continue

        # If all retries exhausted without success
        if not endpoint_success and last_error and not any(e["endpoint"] == url for e in errors):
            errors.append({
                "endpoint": url,
                "error_type": "connection_error",
                "error_message": f"Failed after {MAX_RETRIES} attempts: {last_error}",
            })

    # Determine overall success
    success = len(all_results) > 0 or (len(errors) == 0 and len(urls) > 0)

    result = {
        "success": success,
        "results": {
            "bindings": all_results,
            "total_count": len(all_results),
        },
        "queried_endpoints": urls,
        "successful_endpoints": successful_endpoints,
        "errors": errors if errors else None,
        "total_retries": total_retries,
        "timeout_seconds": QUERY_TIMEOUT,  # Track timeout used for this execution
    }

    # Add validation warning if present
    if validation_warning:
        result["validation_warning"] = validation_warning

    return result


async def execute_sparql_async(
    query: str,
    endpoints: list[str] | None = None,
    endpoint_url: str | None = None,
    dataset_ids: list[str] | None = None,
) -> dict:
    """
    Async version of execute_sparql for parallel endpoint queries.

    Args:
        query: The SPARQL query to execute
        endpoints: List of endpoint URLs to query (preferred for multi-endpoint)
        endpoint_url: Single endpoint URL (backwards compatible)
        dataset_ids: List of dataset IDs to query (alternative to endpoints)

    Returns:
        Dict with success status, merged results from all endpoints,
        and information about which endpoints were queried
    """
    settings = get_settings()

    # Determine which endpoints to query
    urls = []
    if endpoints:
        urls = endpoints
    elif dataset_ids:
        for dataset_id in dataset_ids:
            endpoint = get_endpoint_for_dataset(dataset_id)
            if endpoint:
                urls.append(endpoint)
    elif endpoint_url:
        urls = [endpoint_url]
    else:
        urls = [settings.ontop_sparql_url]

    if not urls:
        return {
            "success": False,
            "error_type": "no_endpoints",
            "error_message": "No valid endpoints found to query",
        }

    # Pre-execution validation: Check for known problematic patterns
    validation_result = validate_query(query)
    if validation_result.severity == ValidationSeverity.ERROR:
        # Query will definitely fail - don't waste time executing
        return {
            "success": False,
            "error_type": "validation_error",
            "error_message": f"VALIDATION ERROR: {validation_result.message}",
            "validation_pattern": validation_result.matched_pattern,
            "suggestion": validation_result.suggestion,
            "queried_endpoints": [],
            "skipped_execution": True,
        }

    # Include validation warning in response if present
    validation_warning = None
    if validation_result.severity == ValidationSeverity.WARNING:
        validation_warning = {
            "pattern": validation_result.matched_pattern,
            "message": validation_result.message,
            "suggestion": validation_result.suggestion,
        }

    headers = {"Accept": "application/sparql-results+json"}
    params = {"query": query}
    client = get_async_client()

    async def query_endpoint(url: str) -> tuple[str, list | None, dict | None, int]:
        """Query a single endpoint with retry, return (url, bindings, error, retries)."""
        last_error = None
        wait_time = INITIAL_BACKOFF
        retries = 0

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                result = response.json()
                bindings = result.get("results", {}).get("bindings", [])
                # Tag results with source endpoint
                for binding in bindings:
                    binding["_source_endpoint"] = url
                return (url, bindings, None, retries)

            except RETRYABLE_EXCEPTIONS as e:
                retries += 1
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"Retry {attempt + 1}/{MAX_RETRIES} for {url}: {type(e).__name__}: {e}"
                    )
                    await asyncio.sleep(wait_time)
                    wait_time = min(wait_time * 2, MAX_BACKOFF)
                continue

            except httpx.HTTPStatusError as e:
                if is_retryable_http_status(e.response.status_code) and attempt < MAX_RETRIES - 1:
                    retries += 1
                    last_error = e
                    logger.warning(
                        f"Retry {attempt + 1}/{MAX_RETRIES} for {url} after HTTP {e.response.status_code}"
                    )
                    await asyncio.sleep(wait_time)
                    wait_time = min(wait_time * 2, MAX_BACKOFF)
                    continue
                return (url, None, {
                    "endpoint": url,
                    "error_type": "http_error",
                    "error_message": f"HTTP {e.response.status_code}: {e.response.text[:500]}",
                }, retries)

            except httpx.RequestError as e:
                retries += 1
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"Retry {attempt + 1}/{MAX_RETRIES} for {url}: {type(e).__name__}: {e}"
                    )
                    await asyncio.sleep(wait_time)
                    wait_time = min(wait_time * 2, MAX_BACKOFF)
                continue

        # All retries exhausted
        return (url, None, {
            "endpoint": url,
            "error_type": "connection_error",
            "error_message": f"Failed after {MAX_RETRIES} attempts: {last_error}",
        }, retries)

    # Query all endpoints in parallel
    tasks = [query_endpoint(url) for url in urls]
    results = await asyncio.gather(*tasks)

    # Aggregate results
    all_results = []
    errors = []
    successful_endpoints = []
    total_retries = 0

    for url, bindings, error, retries in results:
        total_retries += retries
        if bindings is not None:
            all_results.extend(bindings)
            successful_endpoints.append(url)
        if error:
            errors.append(error)

    # Determine overall success
    success = len(all_results) > 0 or (len(errors) == 0 and len(urls) > 0)

    result = {
        "success": success,
        "results": {
            "bindings": all_results,
            "total_count": len(all_results),
        },
        "queried_endpoints": urls,
        "successful_endpoints": successful_endpoints,
        "errors": errors if errors else None,
        "total_retries": total_retries,
        "timeout_seconds": QUERY_TIMEOUT,  # Track timeout used for this execution
    }

    # Add validation warning if present
    if validation_warning:
        result["validation_warning"] = validation_warning

    return result


@tool
def execute_sparql_on_datasets(
    query: str,
    dataset_ids: list[str],
) -> dict:
    """
    Execute a SPARQL query on specific datasets by their IDs.

    Args:
        query: The SPARQL query to execute
        dataset_ids: List of dataset IDs (as provided in the task context)

    Returns:
        Dict with success status and merged results
    """
    return execute_sparql.invoke({
        "query": query,
        "dataset_ids": dataset_ids,
    })


@tool
def list_available_datasets() -> dict:
    """
    List all available datasets in the registry with their endpoints and descriptions.

    Returns:
        Dict with list of datasets and their metadata
    """
    registry = load_dataset_registry()
    datasets = []
    for d in registry.get("datasets", []):
        datasets.append({
            "id": d["id"],
            "name": d["name"],
            "sparql_endpoint": d["sparql_endpoint"],
            "description": d["description"],
            "keywords": d.get("keywords", []),
            "concepts": d.get("concepts", []),
        })
    return {
        "success": True,
        "datasets": datasets,
        "total_count": len(datasets),
    }


@tool
def validate_sparql_syntax(
    query: str,
    endpoint_url: str | None = None,
    dataset_id: str | None = None,
) -> dict:
    """
    Validate the syntax of a SPARQL query.

    Args:
        query: The SPARQL query to validate
        endpoint_url: Optional custom endpoint URL
        dataset_id: Optional dataset ID to use for validation

    Returns:
        Dict with is_valid boolean and optional error message
    """
    # Add LIMIT 0 to avoid actual execution
    limited_query = query.strip()
    if "LIMIT" not in limited_query.upper():
        limited_query += "\nLIMIT 0"

    # Determine endpoint
    if dataset_id:
        endpoint = get_endpoint_for_dataset(dataset_id)
        if endpoint:
            endpoint_url = endpoint

    result = execute_sparql.invoke({
        "query": limited_query,
        "endpoint_url": endpoint_url,
    })

    if result.get("success"):
        return {"is_valid": True, "error_message": None}
    elif "syntax" in result.get("error_message", "").lower():
        return {"is_valid": False, "error_message": result.get("error_message")}
    else:
        # Connection errors etc. - assume syntax is valid
        return {"is_valid": True, "error_message": None}


# Standard SPARQL prefixes for the experiment
STANDARD_PREFIXES = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
"""

# Dataset-specific prefixes
DATASET_PREFIXES = {
    "edu": "PREFIX eduo: <http://example.org/ontology/education#>",
    "trn": "PREFIX tro: <http://example.org/ontology/transport#>",
    "nrg": "PREFIX eno: <http://example.org/ontology/energy#>",
    "bsbm": "PREFIX bsbm: <http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/>",
    "bgee": "PREFIX genex: <http://purl.org/genex#>",
}


@tool
def add_prefixes(
    query: str,
    additional_prefixes: str = "",
    dataset_ids: list[str] | None = None,
) -> str:
    """
    Add standard and dataset-specific prefixes to a SPARQL query if not present.

    Args:
        query: The SPARQL query
        additional_prefixes: Additional prefixes to add
        dataset_ids: List of dataset IDs to add prefixes for

    Returns:
        Query with prefixes added
    """
    # Check if query already has prefixes
    has_prefix = query.strip().upper().startswith("PREFIX")

    if has_prefix:
        return query

    prefixes = STANDARD_PREFIXES

    # Add dataset-specific prefixes
    if dataset_ids:
        for dataset_id in dataset_ids:
            if dataset_id in DATASET_PREFIXES:
                prefixes += "\n" + DATASET_PREFIXES[dataset_id]

    if additional_prefixes:
        prefixes += "\n" + additional_prefixes

    return prefixes + "\n" + query


@tool
def analyze_query_metadata(
    query: str,
    error_message: str | None = None,
) -> dict:
    """
    Analyze a SPARQL query and extract raw metadata facts.

    Use this tool to understand why a query might be problematic or inefficient.
    The metadata includes:
    - Classes used and their instance counts in the database
    - Properties used in the query
    - Query structure (LIMIT, FILTER, GROUP BY, etc.)
    - Error classification if an error message is provided

    Args:
        query: The SPARQL query to analyze
        error_message: Optional error message from a failed execution

    Returns:
        Dict with raw metadata facts (no interpretation - decide yourself what to do)
    """
    settings = get_settings()
    try:
        metadata = extract_query_metadata(
            query=query,
            error_message=error_message,
            endpoint_url=settings.ontop_sparql_url,
            fetch_counts=True,
        )
        return {
            "success": True,
            "metadata": metadata.to_dict(),
            "feedback": metadata.to_feedback_string(),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


SPARQL_TOOLS = [
    execute_sparql,
    execute_sparql_on_datasets,
    list_available_datasets,
    validate_sparql_syntax,
    add_prefixes,
    analyze_query_metadata,
]

__all__ = [
    "execute_sparql",
    "execute_sparql_async",
    "execute_sparql_on_datasets",
    "list_available_datasets",
    "validate_sparql_syntax",
    "add_prefixes",
    "analyze_query_metadata",
    "load_dataset_registry",
    "get_endpoint_for_dataset",
    "get_all_endpoints",
    "get_endpoints_for_query",
    "get_async_client",
    "close_async_client",
    "STANDARD_PREFIXES",
    "DATASET_PREFIXES",
    "SPARQL_TOOLS",
    "USE_ONTOP_REPLICAS",
]
