"""
Query Metadata Extraction - Extract raw metadata from SPARQL queries.

This module provides functions to extract metadata from failed SPARQL queries
to help the agent understand why a query failed and how to fix it.

The metadata is RAW FACTS only - no interpretation or suggestions.
The LLM decides what to do with the information.
"""

import re
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)

# Timeout for COUNT queries (should be fast)
COUNT_QUERY_TIMEOUT = 5.0


@dataclass
class QueryMetadata:
    """Raw metadata extracted from a query - NO interpretation."""

    # Classes used in the query with their instance counts
    classes: dict[str, int]  # e.g. {"eduo:Publication": 523847, "eduo:Author": -1}

    # Properties used in the query
    properties: list[str]  # e.g. ["eduo:author", "rdfs:label"]

    # Query structure facts (booleans)
    has_limit: bool
    limit_value: Optional[int]
    has_filter: bool
    has_group_by: bool
    has_order_by: bool
    has_distinct: bool
    has_optional: bool
    has_union: bool
    has_subquery: bool

    # Aggregations used
    aggregations: list[str]  # e.g. ["COUNT", "SUM", "GROUP_CONCAT"]

    # The error that occurred (if any)
    error_type: Optional[str]
    error_message: Optional[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_feedback_string(self) -> str:
        """Format as a feedback string for the agent."""
        lines = ["QUERY METADATA:"]

        if self.error_type:
            lines.append(f"  Error Type: {self.error_type}")

        if self.classes:
            lines.append("  Classes:")
            for cls, count in self.classes.items():
                count_str = f"{count:,}" if count >= 0 else "unknown"
                lines.append(f"    - {cls}: {count_str} instances")

        if self.properties:
            lines.append(f"  Properties: {', '.join(self.properties)}")

        lines.append(f"  Has LIMIT: {self.has_limit}" + (f" (value: {self.limit_value})" if self.limit_value else ""))
        lines.append(f"  Has FILTER: {self.has_filter}")
        lines.append(f"  Has GROUP BY: {self.has_group_by}")
        lines.append(f"  Has DISTINCT: {self.has_distinct}")

        if self.aggregations:
            lines.append(f"  Aggregations: {', '.join(self.aggregations)}")

        return "\n".join(lines)


def extract_classes_from_query(query: str) -> list[str]:
    """Extract all class URIs from a SPARQL query (patterns like ?x a prefix:Class)."""
    patterns = [
        r'\?\w+\s+a\s+([\w]+:[\w]+)',
        r'\?\w+\s+rdf:type\s+([\w]+:[\w]+)',
    ]
    classes = set()
    for pattern in patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        classes.update(matches)
    return list(classes)


def extract_properties_from_query(query: str) -> list[str]:
    """Extract all property URIs from a SPARQL query."""
    pattern = r'\?\w+\s+([\w]+:[\w]+)\s+\?\w+'
    properties = set()
    matches = re.findall(pattern, query, re.IGNORECASE)
    for prop in matches:
        if prop.lower() not in ['rdf:type', 'a']:
            properties.add(prop)
    return list(properties)


def extract_limit_value(query: str) -> Optional[int]:
    """Extract LIMIT value if present."""
    match = re.search(r'LIMIT\s+(\d+)', query, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def extract_aggregations(query: str) -> list[str]:
    """Extract aggregation functions used in the query."""
    aggs = []
    if re.search(r'\bCOUNT\s*\(', query, re.IGNORECASE):
        aggs.append("COUNT")
    if re.search(r'\bSUM\s*\(', query, re.IGNORECASE):
        aggs.append("SUM")
    if re.search(r'\bAVG\s*\(', query, re.IGNORECASE):
        aggs.append("AVG")
    if re.search(r'\bMIN\s*\(', query, re.IGNORECASE):
        aggs.append("MIN")
    if re.search(r'\bMAX\s*\(', query, re.IGNORECASE):
        aggs.append("MAX")
    if re.search(r'\bGROUP_CONCAT\s*\(', query, re.IGNORECASE):
        aggs.append("GROUP_CONCAT")
    return aggs


def classify_error_type(error_message: str) -> str:
    """Classify the error type from the error message."""
    if not error_message:
        return "UNKNOWN"

    error_lower = error_message.lower()

    if "timeout" in error_lower or "execution time exceeded" in error_lower:
        return "TIMEOUT"
    elif "SQLSerializationException" in error_message:
        return "SQL_SERIALIZATION_ERROR"
    elif "NullPointerException" in error_message:
        return "INTERNAL_ERROR"
    elif "ArbitraryLengthPath" in error_message or "Unsupported arbitrary length path" in error_message:
        return "UNSUPPORTED_PROPERTY_PATH"
    elif "Multiple entries with same key: UNION" in error_message:
        return "UNION_MAPPING_BUG"
    elif "does not appear in columnIDs" in error_message:
        return "COLUMN_ID_ERROR"
    elif "syntax" in error_lower or "parse" in error_lower:
        return "SYNTAX_ERROR"
    else:
        return "UNKNOWN"


def get_class_count(class_uri: str, endpoint_url: str) -> int:
    """
    Get the instance count for a class by executing a COUNT query.

    Args:
        class_uri: The class URI (e.g., "eduo:Publication")
        endpoint_url: The SPARQL endpoint URL

    Returns:
        Instance count, or -1 if the query failed
    """
    # Build COUNT query
    count_query = f"SELECT (COUNT(?x) AS ?count) WHERE {{ ?x a {class_uri} }}"

    try:
        with httpx.Client(timeout=COUNT_QUERY_TIMEOUT) as client:
            response = client.get(
                endpoint_url,
                params={"query": count_query},
                headers={"Accept": "application/sparql-results+json"}
            )
            response.raise_for_status()
            result = response.json()
            bindings = result.get("results", {}).get("bindings", [])
            if bindings:
                count_value = bindings[0].get("count", {}).get("value", "0")
                return int(count_value)
    except Exception as e:
        logger.debug(f"Failed to get count for {class_uri}: {e}")

    return -1


def get_class_counts(classes: list[str], endpoint_url: str | None = None) -> dict[str, int]:
    """
    Get instance counts for all classes in the list.

    Args:
        classes: List of class URIs
        endpoint_url: Optional endpoint URL. If None, uses default from settings.

    Returns:
        Dictionary mapping class URIs to their instance counts (-1 if unknown)
    """
    if not endpoint_url:
        settings = get_settings()
        endpoint_url = settings.ontop_sparql_url

    result = {}
    for cls in classes:
        result[cls] = get_class_count(cls, endpoint_url)

    return result


def extract_query_metadata(
    query: str,
    error_message: str | None = None,
    endpoint_url: str | None = None,
    fetch_counts: bool = True,
) -> QueryMetadata:
    """
    Extract raw metadata from a SPARQL query.

    Args:
        query: The SPARQL query
        error_message: Optional error message if the query failed
        endpoint_url: Optional endpoint URL for fetching class counts
        fetch_counts: If True, execute COUNT queries to get class sizes

    Returns:
        QueryMetadata with extracted facts (no interpretation)
    """
    # Extract classes
    classes = extract_classes_from_query(query)

    # Get class counts if requested
    if fetch_counts and classes:
        class_counts = get_class_counts(classes, endpoint_url)
    else:
        class_counts = {cls: -1 for cls in classes}

    # Extract properties
    properties = extract_properties_from_query(query)

    # Extract limit
    limit_value = extract_limit_value(query)

    # Extract aggregations
    aggregations = extract_aggregations(query)

    # Classify error type
    error_type = classify_error_type(error_message) if error_message else None

    return QueryMetadata(
        classes=class_counts,
        properties=properties,
        has_limit=limit_value is not None,
        limit_value=limit_value,
        has_filter=bool(re.search(r'\bFILTER\s*\(', query, re.IGNORECASE)),
        has_group_by=bool(re.search(r'\bGROUP\s+BY\b', query, re.IGNORECASE)),
        has_order_by=bool(re.search(r'\bORDER\s+BY\b', query, re.IGNORECASE)),
        has_distinct=bool(re.search(r'\bDISTINCT\b', query, re.IGNORECASE)),
        has_optional=bool(re.search(r'\bOPTIONAL\s*\{', query, re.IGNORECASE)),
        has_union=bool(re.search(r'\bUNION\s*\{', query, re.IGNORECASE)),
        has_subquery=bool(re.search(r'\{\s*SELECT\b', query, re.IGNORECASE)),
        aggregations=aggregations,
        error_type=error_type,
        error_message=error_message[:500] if error_message else None,
    )


__all__ = [
    "QueryMetadata",
    "extract_query_metadata",
    "extract_classes_from_query",
    "extract_properties_from_query",
    "get_class_counts",
    "classify_error_type",
]
