"""
OnTop SPARQL endpoint integration.

Provides async functions for executing SPARQL queries against an OnTop endpoint.
Includes load balancing support for multiple OnTop replicas.
"""

import asyncio
import logging
import re
import threading
from dataclasses import dataclass
from typing import Iterator

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 30.0

# Query timeout - OnTop queries are either fast (<2s) or will timeout at MySQL level (60s)
# Using 15s client-side timeout saves ~45s per problematic query
QUERY_TIMEOUT = 15.0

# Exceptions that should trigger retry (transient failures)
RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    ConnectionResetError,
    ConnectionRefusedError,
)

# Timeout exceptions that should NOT be retried (query is too expensive, not a transient issue)
NON_RETRYABLE_TIMEOUT = (httpx.ReadTimeout,)


def is_retryable_http_status(status_code: int) -> bool:
    """Check if an HTTP status code is retryable (5xx server errors, 429 rate limit).

    Note: HTTP 500 is excluded because OnTop returns 500 for unsupported SPARQL features
    (Property Paths with *, +, ?, SERVICE queries) - these are query errors, not transient failures.
    """
    return status_code in (429, 502, 503, 504)  # 500 excluded - usually query syntax issues


@dataclass
class QueryResult:
    """Result from a SPARQL query execution."""

    success: bool
    results: dict | None = None
    error_type: str | None = None
    error_message: str | None = None
    retries: int = 0


class OnTopEndpoint:
    """Client for interacting with an OnTop SPARQL endpoint."""

    def __init__(self, endpoint_url: str | None = None, max_retries: int = MAX_RETRIES):
        """Initialize the OnTop endpoint client."""
        settings = get_settings()
        self.endpoint_url = endpoint_url or settings.ontop_sparql_url
        self.max_retries = max_retries

    async def execute_query(
        self,
        query: str,
        timeout: float = QUERY_TIMEOUT,
    ) -> QueryResult:
        """
        Execute a SPARQL query against the OnTop endpoint with retry logic.

        Args:
            query: The SPARQL query to execute
            timeout: Request timeout in seconds

        Returns:
            QueryResult with success status and results or error details
        """
        headers = {"Accept": "application/sparql-results+json"}
        params = {"query": query}

        retries = 0
        last_error = None
        wait_time = INITIAL_BACKOFF

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(
                        self.endpoint_url,
                        params=params,
                        headers=headers,
                    )
                    response.raise_for_status()
                    return QueryResult(
                        success=True,
                        results=response.json(),
                        retries=retries,
                    )

            except NON_RETRYABLE_TIMEOUT:
                # Query timeout - don't retry, query is too expensive
                logger.warning(f"Query timeout after {timeout}s for {self.endpoint_url} - not retrying")
                return QueryResult(
                    success=False,
                    error_type="query_timeout",
                    error_message=f"Query timed out after {timeout}s. Query is too expensive for OnTop.",
                    retries=0,
                )

            except RETRYABLE_EXCEPTIONS as e:
                retries += 1
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_retries} for {self.endpoint_url}: {type(e).__name__}: {e}"
                    )
                    await asyncio.sleep(wait_time)
                    wait_time = min(wait_time * 2, MAX_BACKOFF)
                continue

            except httpx.HTTPStatusError as e:
                error_body = e.response.text

                # Retry on server errors (5xx) and rate limiting (429)
                if is_retryable_http_status(e.response.status_code) and attempt < self.max_retries - 1:
                    retries += 1
                    last_error = e
                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_retries} after HTTP {e.response.status_code}"
                    )
                    await asyncio.sleep(wait_time)
                    wait_time = min(wait_time * 2, MAX_BACKOFF)
                    continue

                # Non-retryable HTTP errors
                if e.response.status_code == 400:
                    return QueryResult(
                        success=False,
                        error_type="syntax_error",
                        error_message=error_body.strip(),
                        retries=retries,
                    )
                query_preview = query[:500] + "..." if len(query) > 500 else query
                logger.error(
                    f"OnTop query failed with status {e.response.status_code}\n"
                    f"Query: {query_preview}\n"
                    f"Response: {error_body[:1000]}"
                )
                return QueryResult(
                    success=False,
                    error_type="http_error",
                    error_message=f"HTTP {e.response.status_code}: {error_body}",
                    retries=retries,
                )

            except httpx.RequestError as e:
                retries += 1
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_retries} for {self.endpoint_url}: {type(e).__name__}: {e}"
                    )
                    await asyncio.sleep(wait_time)
                    wait_time = min(wait_time * 2, MAX_BACKOFF)
                continue

        # All retries exhausted (only for transient connection errors)
        logger.error(f"OnTop request failed after {self.max_retries} attempts: {last_error}")
        return QueryResult(
            success=False,
            error_type="connection_error",
            error_message=str(last_error) or f"Connection failed after {self.max_retries} attempts",
            retries=retries,
        )

    async def validate_syntax(self, query: str) -> tuple[bool, str | None]:
        """
        Validate SPARQL query syntax.

        Returns tuple of (is_valid, error_message).
        """
        # Add LIMIT 0 to avoid actual execution
        limited_query = self._add_limit(query, limit=0)
        result = await self.execute_query(limited_query)

        if result.success:
            return True, None
        elif result.error_type == "syntax_error":
            return False, result.error_message
        else:
            # Connection errors etc. - assume syntax is valid
            return True, None

    @staticmethod
    def _add_limit(query: str, limit: int = 5) -> str:
        """Add or replace a LIMIT clause in a SPARQL query."""
        limit_regex = re.compile(
            r"(LIMIT\s+\d+)(\s+OFFSET\s+\d+)?\s*$",
            re.IGNORECASE | re.DOTALL,
        )
        if limit_regex.search(query):
            return limit_regex.sub(f"LIMIT {limit}\\2", query)
        return query.strip() + f"\nLIMIT {limit}"


# Convenience function for simple usage
async def execute_sparql(query: str, endpoint_url: str | None = None) -> QueryResult:
    """Execute a SPARQL query against the configured OnTop endpoint."""
    endpoint = OnTopEndpoint(endpoint_url)
    return await endpoint.execute_query(query)


class OnTopEndpointPool:
    """
    Round-Robin Load Balancer for OnTop replicas.

    Distributes queries across multiple OnTop instances to achieve
    parallel query execution (bypassing OnTop's internal query queue).

    Thread-safe for concurrent access from multiple async tasks.

    Example:
        pool = OnTopEndpointPool([
            "http://localhost:8080/sparql",
            "http://localhost:8180/sparql",
            "http://localhost:8280/sparql",
        ])
        endpoint = pool.get_next()  # Returns next endpoint in round-robin order
    """

    def __init__(self, replicas: list[str]):
        """
        Initialize the endpoint pool.

        Args:
            replicas: List of OnTop endpoint URLs to load balance across.
                      Should have at least one endpoint.
        """
        if not replicas:
            raise ValueError("At least one replica endpoint is required")
        self.replicas = replicas
        self._index = 0
        self._lock = threading.Lock()

    def get_next(self) -> str:
        """
        Get the next endpoint in round-robin order.

        Thread-safe: Can be called concurrently from multiple threads/tasks.

        Returns:
            The next endpoint URL to use.
        """
        with self._lock:
            endpoint = self.replicas[self._index % len(self.replicas)]
            self._index += 1
            return endpoint

    def get_all(self) -> list[str]:
        """Get all replica endpoints (useful for health checks)."""
        return list(self.replicas)

    def __iter__(self) -> Iterator[str]:
        """Iterate over all replica endpoints."""
        return iter(self.replicas)

    def __len__(self) -> int:
        """Return the number of replicas in the pool."""
        return len(self.replicas)

    @property
    def count(self) -> int:
        """Number of replicas in the pool."""
        return len(self.replicas)


# Global endpoint pools (lazy-initialized per dataset)
_endpoint_pools: dict[str, OnTopEndpointPool] = {}
_pools_lock = threading.Lock()


def get_endpoint_pool(dataset_id: str) -> OnTopEndpointPool | None:
    """
    Get or create an endpoint pool for a dataset.

    Uses lazy initialization with thread-safe singleton pattern.
    Returns None if no replicas are configured for this dataset.

    Args:
        dataset_id: Dataset identifier (e.g., "edu", "trn", "nrg")

    Returns:
        OnTopEndpointPool for the dataset, or None if not configured.
    """
    from src.config import ONTOP_REPLICAS

    dataset_key = dataset_id.lower()

    with _pools_lock:
        if dataset_key not in _endpoint_pools:
            replicas = ONTOP_REPLICAS.get(dataset_key, [])
            if replicas:
                _endpoint_pools[dataset_key] = OnTopEndpointPool(replicas)
                logger.info(
                    f"Created endpoint pool for {dataset_key} with {len(replicas)} replicas"
                )
            else:
                return None
        return _endpoint_pools.get(dataset_key)


def reset_endpoint_pools() -> None:
    """Reset all endpoint pools (useful for testing)."""
    global _endpoint_pools
    with _pools_lock:
        _endpoint_pools = {}
