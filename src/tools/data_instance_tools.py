"""
Data Instance Retrieval Tools.

Tools for querying specific data values from the knowledge graph.
The agent specifies (Class, DatatypeProperty) combinations and the
tool handles the SPARQL query construction and execution.
"""

import asyncio
import logging

from langchain_core.tools import tool

from src.endpoints.ontop import OnTopEndpoint
from src.tools.retrieval_tools import get_property_value_index

logger = logging.getLogger(__name__)


# Common prefix URIs for different datasets
PREFIX_MAP = {
    "ub": "http://example.org/ontology/education#",
    "npdv": "http://example.org/ontology/energy#",
    "trn": "http://example.org/ontology/transport#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
}


def _get_prefix_declarations(prefixes: set[str]) -> str:
    """Generate PREFIX declarations for a set of prefixes."""
    declarations = []
    for p in prefixes:
        if p in PREFIX_MAP:
            declarations.append(f"PREFIX {p}: <{PREFIX_MAP[p]}>")
        else:
            declarations.append(f"PREFIX {p}: <http://example.org/{p}#>")
    return "\n".join(declarations)


def _build_values_query(
    class_uri: str,
    property_uri: str,
    limit: int = 100,
) -> str:
    """
    Build a SPARQL query to retrieve values for a class+property combination.
    """
    # Extract prefixes
    prefixes = set()
    for uri in [class_uri, property_uri]:
        if ":" in uri and not uri.startswith("<"):
            prefixes.add(uri.split(":")[0])

    return f"""
{_get_prefix_declarations(prefixes)}

SELECT DISTINCT ?value WHERE {{
    ?subject a {class_uri} .
    ?subject {property_uri} ?value .
}}
LIMIT {limit}
""".strip()


@tool
def fetch_property_values(
    class_name: str,
    property_name: str,
    prefix: str = "ub",
    limit: int = 100,
) -> dict:
    """
    Fetch all values for a specific class+property combination from the data.

    The agent specifies which class and datatype property to query,
    and this tool constructs and executes the appropriate SPARQL query.

    Args:
        class_name: The class name without prefix (e.g., "Company", "DoctoralCandidate")
        property_name: The property name without prefix (e.g., "name", "emailAddress")
        prefix: The ontology prefix (e.g., "ub", "npdv", "trn")
        limit: Maximum number of values to retrieve

    Returns:
        Dict with values list and metadata

    Example:
        fetch_property_values("Company", "name", "npdv", 50)
        fetch_property_values("DoctoralCandidate", "emailAddress", "ub", 100)
    """
    class_uri = f"{prefix}:{class_name}"
    property_uri = f"{prefix}:{property_name}"

    query = _build_values_query(class_uri, property_uri, limit)

    endpoint = OnTopEndpoint()
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(endpoint.execute_query(query))
    finally:
        loop.close()

    if not result.success:
        return {
            "success": False,
            "error": result.error_message,
            "class": class_name,
            "property": property_name,
            "values": [],
        }

    # Extract values from results
    bindings = result.results.get("results", {}).get("bindings", [])
    values = []
    for binding in bindings:
        if "value" in binding:
            val_obj = binding["value"]
            val = val_obj.get("value", "") if isinstance(val_obj, dict) else str(val_obj)
            if val:
                values.append(val)

    return {
        "success": True,
        "class": class_name,
        "property": property_name,
        "prefix": prefix,
        "count": len(values),
        "values": values,
    }


@tool
def index_and_search_values(
    class_name: str,
    property_name: str,
    search_term: str,
    prefix: str = "ub",
    top_k: int = 5,
) -> list[dict]:
    """
    Fetch, index, and semantically search values for a class+property combination.

    This is the main tool for finding specific data values. It:
    1. Fetches all values for the class+property from the database
    2. Creates embeddings and indexes them
    3. Searches for values matching the search term

    Args:
        class_name: The class name (e.g., "Company", "Discovery")
        property_name: The property name (e.g., "name", "status")
        search_term: What to search for semantically
        prefix: The ontology prefix
        top_k: Number of results to return

    Returns:
        List of matching values with similarity scores

    Example:
        index_and_search_values("Company", "name", "German energy", "npdv")
        index_and_search_values("Discovery", "hydrocarbonType", "oil", "npdv")
    """
    # Step 1: Fetch values
    fetch_result = fetch_property_values.invoke({
        "class_name": class_name,
        "property_name": property_name,
        "prefix": prefix,
        "limit": 500,
    })

    if not fetch_result.get("success") or not fetch_result.get("values"):
        return [{
            "error": f"Could not fetch values for {class_name}.{property_name}",
            "details": fetch_result.get("error", "No values found"),
        }]

    # Step 2: Index the values
    index = get_property_value_index()
    indexed_count = index.index_values(
        class_name=class_name,
        property_name=property_name,
        values=fetch_result["values"],
    )

    if indexed_count == 0:
        return [{
            "error": "Failed to index values",
            "values_fetched": len(fetch_result["values"]),
        }]

    # Step 3: Search
    results = index.search(class_name, property_name, search_term, top_k)

    return results


@tool
def format_instance_triple(
    class_name: str,
    property_name: str,
    value: str,
    prefix: str = "ub",
) -> str:
    """
    Format a data instance as a triple for schema injection.

    Returns a formatted triple that can be injected into the schema context
    to replace the generic type with an actual value.

    Args:
        class_name: The class name
        property_name: The property name
        value: The actual value found
        prefix: The ontology prefix

    Returns:
        Formatted triple string

    Example:
        format_instance_triple("Company", "name", "Bayer Group AG", "npdv")
        -> 'eno:Company eno:name "Bayer Group AG" .'
    """
    class_uri = f"{prefix}:{class_name}"
    property_uri = f"{prefix}:{property_name}"

    # Escape quotes in value
    escaped_value = value.replace('"', '\\"')

    return f'{class_uri} {property_uri} "{escaped_value}" .'


@tool
def find_matching_instances(
    class_name: str,
    property_name: str,
    search_term: str,
    prefix: str = "ub",
    return_as_triples: bool = True,
) -> dict:
    """
    Find data instances matching a search term and return them as injectable triples.

    This is the primary tool for the Context Agent. It:
    1. Searches for matching values
    2. Formats them as triples ready for schema injection

    Args:
        class_name: The class to search
        property_name: The datatype property to search
        search_term: What to search for
        prefix: The ontology prefix
        return_as_triples: If True, return formatted triples

    Returns:
        Dict with matching values and/or triples

    Example:
        find_matching_instances("Company", "name", "oil company", "npdv")
    """
    # Search for matching values
    search_results = index_and_search_values.invoke({
        "class_name": class_name,
        "property_name": property_name,
        "search_term": search_term,
        "prefix": prefix,
        "top_k": 5,
    })

    if not search_results or "error" in search_results[0]:
        return {
            "success": False,
            "error": search_results[0].get("error", "Search failed") if search_results else "No results",
            "triples": [],
            "values": [],
        }

    # Format as triples if requested
    triples = []
    values = []

    for result in search_results:
        value = result.get("value", "")
        if value:
            values.append({
                "value": value,
                "score": result.get("score", 0),
            })

            if return_as_triples:
                triple = format_instance_triple.invoke({
                    "class_name": class_name,
                    "property_name": property_name,
                    "value": value,
                    "prefix": prefix,
                })
                triples.append(triple)

    return {
        "success": True,
        "class": class_name,
        "property": property_name,
        "triples": triples,
        "values": values,
    }


DATA_INSTANCE_TOOLS = [
    fetch_property_values,
    index_and_search_values,
    format_instance_triple,
    find_matching_instances,
]

__all__ = [
    "fetch_property_values",
    "index_and_search_values",
    "format_instance_triple",
    "find_matching_instances",
    "DATA_INSTANCE_TOOLS",
]
