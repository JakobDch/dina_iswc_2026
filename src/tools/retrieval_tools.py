"""
Retrieval tools for finding relevant schema information.

Provides:
- Generic grep search across mapping files
- Structured semantic search with Class+Property combinations
"""

import asyncio
import os
import re
import json
import logging
from pathlib import Path
from io import StringIO

# Fix OpenMP conflict on Windows (faiss vs numpy/scipy)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import faiss
from rdflib import Graph, XSD, RDFS
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

from src.config import (
    get_settings,
    MAPPINGS_DIR,
    MAPPINGS_DIR_ROOT,
    get_semantic_models_path,
    EMBEDDINGS_CACHE_DIR,
    AVAILABLE_DATASETS,
    DATA_DIR,
)
from src.utils.ontop_hints import get_ontop_timeout_hints
from src.tracing import TraceEvent, TraceEventType, get_tracer
# Use the slot/replica-aware endpoint resolver from sparql_tools so value-lookup
# tools route to the same live containers as execute_sparql. The previous local
# copy only read the static registry (ports 8080-8085) and thus failed whenever
# USE_SLOT_CONTAINERS=True was active at experiment time.
from src.tools.sparql_tools import get_endpoint_for_dataset

logger = logging.getLogger(__name__)

# Global tracer instance for tools
_tracer = None

def _get_tracer():
    """Get or initialize the tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = get_tracer()
    return _tracer


def load_dataset_registry() -> dict:
    """Load the dataset registry from JSON file."""
    registry_path = DATA_DIR / "datasets" / "registry.json"
    if registry_path.exists():
        with open(registry_path, encoding="utf-8") as f:
            return json.load(f)
    return {"datasets": []}


# =============================================================================
# GREP-BASED RETRIEVAL
# =============================================================================


@tool
def grep_schema(
    search_term: str,
    dataset: str | None = None,
) -> list[dict]:
    """
    Search for any term in semantic model mapping files across all datasets.

    Use this to find classes, properties, or any schema elements by keyword.
    The search is case-insensitive and searches all .ttl files.
    Results include the SPARQL endpoint for each match.

    Args:
        search_term: The term to search for (a class or property name)
        dataset: Optional - specific dataset to search. If None, searches all datasets.

    Returns:
        List of matches with file name, matching lines, dataset, and sparql_endpoint

    Example:
        grep_schema("Book")            # searches all datasets
        grep_schema("title", dataset=<some_dataset>)
    """
    datasets_to_search = [dataset] if dataset else AVAILABLE_DATASETS
    results = []
    pattern = re.compile(re.escape(search_term), re.IGNORECASE)

    for ds in datasets_to_search:
        search_dir = get_semantic_models_path(ds)
        if not search_dir.exists():
            continue

        endpoint = get_endpoint_for_dataset(ds)

        for file_path in search_dir.glob("**/*.ttl"):
            try:
                content = file_path.read_text(encoding="utf-8")
                matching_lines = []

                for i, line in enumerate(content.split("\n"), 1):
                    if pattern.search(line) and not line.strip().startswith("@prefix"):
                        matching_lines.append({
                            "line_num": i,
                            "content": line.strip(),
                        })

                if matching_lines:
                    results.append({
                        "file": file_path.name,
                        "dataset": ds,
                        "sparql_endpoint": endpoint,
                        "matches": matching_lines,
                    })
            except Exception as e:
                logger.warning(f"Failed to search {file_path}: {e}")

    return results


@tool
def grep_classes(
    search_term: str,
    dataset: str | None = None,
) -> list[dict]:
    """
    Search for CLASS definitions by keyword across all datasets.

    Only returns classes (subjects in TTL files) whose local name matches
    the search term. Use this to find classes by their literal name
    (case-insensitive substring match).

    NOTE: only exact lexical matches are returned — the search term must
    appear in the class's local name. Synonyms, paraphrases or domain
    hypernyms will NOT match.

    Args:
        search_term: A class name to look for (e.g., "Book", "Author")
        dataset: Optional - specific dataset to search. If None, searches all datasets.

    Returns:
        List of matching classes with:
        - class: Full prefixed class name (e.g., "lib:Paperback")
        - file: The mapping file where this class is defined
        - dataset: The dataset this class belongs to
        - sparql_endpoint: The SPARQL endpoint for this dataset
        - properties_preview: First 3 properties as preview
        - prefix_declaration: The PREFIX declaration for this class

    Example:
        grep_classes("Book")                     # searches all datasets
        grep_classes("Author", dataset=<some_dataset>)
    """
    datasets_to_search = [dataset] if dataset else AVAILABLE_DATASETS
    results = []
    pattern = re.compile(re.escape(search_term), re.IGNORECASE)

    for ds in datasets_to_search:
        search_dir = get_semantic_models_path(ds)
        if not search_dir.exists():
            continue

        endpoint = get_endpoint_for_dataset(ds)

        for file_path in search_dir.glob("**/*.ttl"):
            try:
                content = file_path.read_text(encoding="utf-8")
                lines = content.split("\n")

                # Parse prefixes
                prefixes = {}
                for line in lines:
                    if line.strip().startswith("@prefix"):
                        match = re.match(r'@prefix\s+(\w+):\s+<([^>]+)>', line.strip())
                        if match:
                            prefixes[match.group(1)] = match.group(2)

                # Find class definitions (lines starting with prefix:ClassName)
                class_pattern = re.compile(r'^(\w+):(\w+)\s+', re.MULTILINE)

                for match in class_pattern.finditer(content):
                    prefix, class_name = match.groups()
                    full_class = f"{prefix}:{class_name}"

                    # Check if this class matches the search term
                    if pattern.search(class_name):
                        # Extract properties preview from class block
                        properties_preview = _extract_class_properties_preview(
                            content, full_class, max_props=3
                        )

                        results.append({
                            "class": full_class,
                            "file": file_path.name,
                            "dataset": ds,
                            "sparql_endpoint": endpoint,
                            "properties_preview": properties_preview,
                            "prefix_declaration": f"PREFIX {prefix}: <{prefixes.get(prefix, '')}>",
                        })

            except Exception as e:
                logger.warning(f"Failed to search {file_path}: {e}")

    # Deduplicate by (class, dataset)
    seen = set()
    unique_results = []
    for r in results:
        key = (r["class"], r["dataset"])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    # Enrich with hierarchy information from OWL ontologies
    hierarchy_index = get_hierarchy_index()
    for result in unique_results:
        # Extract class name without prefix (e.g., "eduo:DoctoralCandidate" -> "DoctoralCandidate")
        class_name = result["class"].split(":")[-1] if ":" in result["class"] else result["class"]
        ds = result["dataset"]

        # Get hierarchy info
        parents = hierarchy_index.get_parents(class_name, ds)
        ancestors = hierarchy_index.get_ancestors(class_name, ds)
        children = hierarchy_index.get_children(class_name, ds)

        # Add prefix to parent/child names for consistency
        prefix = result["class"].split(":")[0] if ":" in result["class"] else ""
        result["parent_classes"] = [f"{prefix}:{p}" for p in parents] if prefix else parents
        result["ancestor_chain"] = [f"{prefix}:{a}" for a in ancestors] if prefix else ancestors
        result["child_classes"] = [f"{prefix}:{c}" for c in children] if prefix else children
        result["hierarchy_depth"] = len(ancestors)

    return unique_results


def _extract_class_properties_preview(
    content: str,
    class_name: str,
    max_props: int = 3,
) -> list[str]:
    """Extract first N property names from a class block."""
    lines = content.split("\n")
    in_block = False
    properties = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(class_name + " "):
            in_block = True
            continue
        elif in_block:
            if line.startswith(" ") or line.startswith("\t"):
                # Extract property name (prefix:propertyName)
                prop_match = re.match(r'\s*(\w+:\w+)\s+', line)
                if prop_match and len(properties) < max_props:
                    properties.append(prop_match.group(1))
            elif stripped and not stripped.startswith("@prefix"):
                break  # New subject starts

    return properties


@tool
def get_class_hierarchy(
    class_name: str,
    dataset: str | None = None,
    direction: str = "both",
) -> dict:
    """
    Get the class hierarchy (rdfs:subClassOf) for a class.

    Use this to understand inheritance relationships between classes —
    i.e., to find out which more specific subclasses a general concept has,
    or which more abstract superclasses a specific class inherits from.

    Args:
        class_name: The class name to look up.
                   Can include prefix (e.g., "lib:Book") or just the local name.
        dataset: Optional - specific dataset. If None, searches all datasets.
        direction: "up" for ancestors only, "down" for descendants only, "both" for all.

    Returns:
        Dict with:
        - class: The input class name (with prefix if found)
        - dataset: The dataset where this class was found
        - parents: Immediate parent classes (direct superclasses)
        - children: Immediate child classes (direct subclasses)
        - ancestors: Full path to root class(es) - only if direction is "up" or "both"
        - descendants: All subclasses recursively - only if direction is "down" or "both"
        - hierarchy_depth: Distance from root (number of ancestors)
        - sparql_hint: SPARQL pattern hint for querying this class and subclasses

    Example (generic library schema):
        get_class_hierarchy("Book", dataset=<some_dataset>)
        # Might return something like:
        # {
        #   "class": "<prefix>:Book",
        #   "parents": ["<prefix>:Publication"],
        #   "children": ["<prefix>:Paperback", "<prefix>:Hardcover"],
        #   "sparql_hint": "?x a/rdfs:subClassOf* <prefix>:Book"
        # }
        # — real classes/prefixes/datasets depend on the ontologies you
        #   query; treat the shape above as illustrative only.
    """
    # Remove prefix if present
    local_name = class_name.split(":")[-1] if ":" in class_name else class_name

    hierarchy_index = get_hierarchy_index()
    datasets_to_search = [dataset] if dataset else AVAILABLE_DATASETS

    for ds in datasets_to_search:
        hierarchy_index.ensure_indexed(ds)
        info = hierarchy_index.get_hierarchy_info(local_name, ds)

        # Check if we found any hierarchy info for this class
        if info["parents"] or info["children"] or (local_name, ds) in hierarchy_index._hierarchy:
            # Determine prefix from dataset
            prefix_map = {
                "EDU": "ub",
                "NRG": "npdv",
                "TRN": "trn",
                "BSBM": "bsbm",
                "BGEE": "bgee",
            }
            prefix = prefix_map.get(ds, "")
            full_class = f"{prefix}:{local_name}" if prefix else local_name

            result = {
                "class": full_class,
                "dataset": ds,
                "parents": [f"{prefix}:{p}" for p in info["parents"]] if prefix else info["parents"],
                "children": [f"{prefix}:{c}" for c in info["children"]] if prefix else info["children"],
                "hierarchy_depth": info["hierarchy_depth"],
            }

            # Add ancestors/descendants based on direction
            if direction in ("up", "both"):
                result["ancestors"] = [f"{prefix}:{a}" for a in info["ancestors"]] if prefix else info["ancestors"]
            if direction in ("down", "both"):
                result["descendants"] = [f"{prefix}:{d}" for d in info["descendants"]] if prefix else info["descendants"]

            # Add SPARQL hint if class has subclasses
            if info["children"]:
                result["sparql_hint"] = (
                    f"To query ALL {local_name}s including subclasses: "
                    f"?x a/rdfs:subClassOf* {full_class}"
                )
            else:
                result["sparql_hint"] = f"This class has no subclasses. Use: ?x a {full_class}"

            return result

    # Class not found in hierarchy
    return {
        "class": class_name,
        "dataset": dataset or "all",
        "error": f"Class '{class_name}' not found in ontology hierarchy",
        "note": "The class may exist but has no rdfs:subClassOf relations defined",
    }


@tool
def grep_properties(
    search_term: str,
    dataset: str | None = None,
) -> list[dict]:
    """
    Search for PROPERTY definitions by keyword across all datasets.

    Returns properties (predicates in TTL files) whose local name matches
    the search term (case-insensitive substring match). Use this to find
    a relationship by its literal name.

    NOTE: only exact lexical matches are returned — synonyms or paraphrased
    property names will NOT match.

    Args:
        search_term: A property name to look for (e.g., "title", "writtenBy")
        dataset: Optional - specific dataset to search. If None, searches all datasets.

    Returns:
        List of matching properties with:
        - property: Full prefixed property name (e.g., "lib:writtenBy")
        - domain_class: The class that has this property
        - range: What this property points to (class or xsd type)
        - is_object_property: True if connects to another class
        - file: The mapping file where this property is defined
        - dataset: The dataset this property belongs to
        - sparql_endpoint: The SPARQL endpoint for this dataset

    Example:
        grep_properties("writtenBy")
        grep_properties("title", dataset=<some_dataset>)
    """
    datasets_to_search = [dataset] if dataset else AVAILABLE_DATASETS
    results = []
    pattern = re.compile(re.escape(search_term), re.IGNORECASE)

    for ds in datasets_to_search:
        search_dir = get_semantic_models_path(ds)
        if not search_dir.exists():
            continue

        endpoint = get_endpoint_for_dataset(ds)

        for file_path in search_dir.glob("**/*.ttl"):
            try:
                content = file_path.read_text(encoding="utf-8")
                g = Graph()
                g.parse(data=content, format="turtle")

                # Get all subjects (classes)
                for subject in set(g.subjects()):
                    try:
                        subject_qname = g.namespace_manager.qname(str(subject))
                    except Exception:
                        continue

                    # Get all predicates for this subject
                    for predicate, obj in g.predicate_objects(subject):
                        try:
                            pred_qname = g.namespace_manager.qname(str(predicate))
                            obj_qname = g.namespace_manager.qname(str(obj))
                        except Exception:
                            continue

                        # Extract property name (after the colon)
                        prop_name = pred_qname.split(":")[-1] if ":" in pred_qname else pred_qname

                        # Check if property matches search term
                        if pattern.search(prop_name):
                            is_datatype = "XMLSchema" in str(obj) or obj_qname.startswith("xsd:")

                            results.append({
                                "property": pred_qname,
                                "domain_class": subject_qname,
                                "range": obj_qname,
                                "is_object_property": not is_datatype,
                                "file": file_path.name,
                                "dataset": ds,
                                "sparql_endpoint": endpoint,
                            })

            except Exception as e:
                logger.warning(f"Failed to parse {file_path}: {e}")

    # Deduplicate by (property, domain, range, dataset)
    seen = set()
    unique_results = []
    for r in results:
        key = (r["property"], r["domain_class"], r["range"], r["dataset"])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    return unique_results


@tool
def get_class_schema(
    class_name: str,
    dataset: str | None = None,
) -> dict:
    """
    Get the complete schema definition for a specific class.

    Returns all properties (datatype and object) defined for the class.
    Searches across all datasets if no specific dataset is provided.
    Result includes the SPARQL endpoint.

    Args:
        class_name: The class name (e.g., "Book")
        dataset: Optional - specific dataset to search. If None, searches all datasets.

    Returns:
        Dict with class name, file, dataset, sparql_endpoint, and all property definitions

    Example:
        get_class_schema("Book")   # searches all datasets
    """
    datasets_to_search = [dataset] if dataset else AVAILABLE_DATASETS

    for ds in datasets_to_search:
        search_dir = get_semantic_models_path(ds)
        if not search_dir.exists():
            continue

        endpoint = get_endpoint_for_dataset(ds)

        # Search for the class in all files
        for file_path in search_dir.glob("**/*.ttl"):
            try:
                content = file_path.read_text(encoding="utf-8")
                lines = content.split("\n")

                # Find class definition block
                class_pattern = re.compile(rf'\w+:{class_name}\s+', re.IGNORECASE)
                in_class_block = False
                class_triples = []
                prefixes = []

                for line in lines:
                    stripped = line.strip()

                    # Collect prefixes
                    if stripped.startswith("@prefix"):
                        prefixes.append(stripped)
                        continue

                    if not stripped:
                        continue

                    # Check if this line starts the class definition
                    if class_pattern.match(stripped):
                        in_class_block = True
                        class_triples.append(stripped)
                    elif in_class_block:
                        # Class block ends when we hit a new subject (no leading whitespace)
                        if not line.startswith(" ") and not line.startswith("\t") and stripped:
                            break
                        if stripped:
                            class_triples.append(stripped)

                if class_triples:
                    return {
                        "class": class_name,
                        "file": file_path.name,
                        "dataset": ds,
                        "sparql_endpoint": endpoint,
                        "prefixes": prefixes,
                        "schema": "\n".join(class_triples),
                        "properties": _extract_properties(class_triples),
                    }

            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")

    searched = dataset if dataset else "all datasets"
    return {"error": f"Class '{class_name}' not found in {searched}"}


def _extract_properties(triples: list[str]) -> list[dict]:
    """Extract property definitions from a class block."""
    properties = []

    for triple in triples:
        # Match property definitions like "eduo:name xsd:string" or "eduo:advisor eduo:Lecturer"
        matches = re.findall(r'(\w+:\w+)\s+(\w+:\w+)', triple)
        for prop, range_type in matches:
            # Skip the class definition itself
            if ":" in prop and not triple.strip().startswith(prop):
                continue
            properties.append({
                "property": prop,
                "range": range_type,
                "is_datatype": range_type.startswith("xsd:"),
            })

    return properties


@tool
def list_all_classes(dataset: str | None = None) -> list[dict]:
    """
    List all classes defined in semantic models across all datasets.

    Args:
        dataset: Optional - specific dataset to list. If None, lists from all datasets.

    Returns:
        List of dicts with class name, dataset, and sparql_endpoint

    Example:
        list_all_classes()                     # all datasets
        list_all_classes(dataset=<some_dataset>)
    """
    datasets_to_search = [dataset] if dataset else AVAILABLE_DATASETS
    results = []

    # Pattern to match class definitions (subject at start of line)
    class_pattern = re.compile(r'^(\w+:\w+)\s+\w+:', re.MULTILINE)

    for ds in datasets_to_search:
        search_dir = get_semantic_models_path(ds)
        if not search_dir.exists():
            continue

        endpoint = get_endpoint_for_dataset(ds)
        classes_in_dataset = set()

        for file_path in search_dir.glob("**/*.ttl"):
            try:
                content = file_path.read_text(encoding="utf-8")
                matches = class_pattern.findall(content)
                classes_in_dataset.update(matches)
            except Exception:
                continue

        for class_name in sorted(classes_in_dataset):
            results.append({
                "class": class_name,
                "dataset": ds,
                "sparql_endpoint": endpoint,
            })

    return results


@tool
def read_semantic_model(
    file_name: str,
    dataset: str,
) -> dict:
    """
    Read the entire content of a semantic model file with highlighted prefixes.

    Returns the file content along with extracted prefix declarations
    that MUST be used in SPARQL queries.

    Args:
        file_name: Name of the .ttl file (as returned by a prior class/property
                   search tool call in the "file" field).
        dataset: Dataset the file belongs to (as returned by a prior search
                 tool call in the "dataset" field).

    Returns:
        Dict with:
        - content: The complete file content with highlighted prefix section
        - prefixes: Dict mapping prefix names to URIs
        - sparql_prefixes: Ready-to-use SPARQL PREFIX declarations
        - classes_defined: List of classes defined in this file
        - file: The file name
        - dataset: The dataset name
        - sparql_endpoint: The SPARQL endpoint for this dataset
        - note: Reminder to use these prefixes

    Example:
        read_semantic_model("book.ttl", "LIBRARY")
    """
    file_path = get_semantic_models_path(dataset) / file_name
    if not file_path.exists():
        return {"error": f"File '{file_name}' not found in {dataset}"}

    try:
        content = file_path.read_text(encoding="utf-8")

        # Extract prefixes
        prefixes = {}
        prefix_lines = []
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("@prefix"):
                match = re.match(r'@prefix\s+(\w+):\s+<([^>]+)>', stripped)
                if match:
                    prefix_name, uri = match.groups()
                    prefixes[prefix_name] = uri
                    prefix_lines.append(f"PREFIX {prefix_name}: <{uri}>")

        # Extract classes defined in this file (subjects at start of line)
        class_pattern = re.compile(r'^(\w+:\w+)\s+\w+:', re.MULTILINE)
        classes = list(set(class_pattern.findall(content)))

        sparql_prefixes = "\n".join(prefix_lines)

        # Format content with highlighted prefix section
        highlighted_content = f"""
# ============================================================
# PREFIX DECLARATIONS - USE THESE IN YOUR SPARQL QUERIES!
# ============================================================
{sparql_prefixes}

# ============================================================
# SCHEMA DEFINITIONS
# ============================================================
{content}
"""

        endpoint = get_endpoint_for_dataset(dataset)

        return {
            "content": highlighted_content,
            "prefixes": prefixes,
            "sparql_prefixes": sparql_prefixes,
            "classes_defined": sorted(classes),
            "file": file_name,
            "dataset": dataset,
            "sparql_endpoint": endpoint,
            "note": "IMPORTANT: Use the PREFIX declarations above in your SPARQL queries!",
        }

    except Exception as e:
        return {"error": f"Error reading file: {e}"}


# =============================================================================
# SEMANTIC (EMBEDDING-BASED) DATA INSTANCE SEARCH
# =============================================================================


class PropertyValueIndex:
    """
    Index for semantic search over property values.

    Indexes values for specific (Class, DatatypeProperty) combinations
    to enable semantic search over data instances.
    """

    def __init__(self):
        """Initialize the property value index."""
        settings = get_settings()
        self.embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
        # Separate index per (class, property) combination
        self._indices: dict[tuple[str, str], dict] = {}

    def index_values(
        self,
        class_name: str,
        property_name: str,
        values: list[str],
    ) -> int:
        """
        Index values for a specific class+property combination.

        Args:
            class_name: The class (e.g., "Company")
            property_name: The datatype property (e.g., "name")
            values: List of string values to index

        Returns:
            Number of values indexed
        """
        if not values:
            return 0

        key = (class_name, property_name)

        # Remove duplicates and empty values
        unique_values = list(set(v.strip() for v in values if v and v.strip()))
        if not unique_values:
            return 0

        logger.info(f"Indexing {len(unique_values)} values for {class_name}.{property_name}")

        try:
            embeddings_list = self.embeddings.embed_documents(unique_values)
            embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
            faiss.normalize_L2(embeddings_matrix)

            dimension = embeddings_matrix.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings_matrix)

            self._indices[key] = {
                "index": index,
                "values": unique_values,
                "embeddings": embeddings_matrix,
            }

            return len(unique_values)
        except Exception as e:
            logger.error(f"Failed to index values: {e}")
            return 0

    def search(
        self,
        class_name: str,
        property_name: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search for values matching a query.

        Args:
            class_name: The class to search in
            property_name: The property to search
            query: The search query
            top_k: Number of results

        Returns:
            List of matching values with scores
        """
        key = (class_name, property_name)
        if key not in self._indices:
            return []

        index_data = self._indices[key]
        index = index_data["index"]
        values = index_data["values"]

        try:
            query_embedding = np.array(
                [self.embeddings.embed_query(query)], dtype=np.float32
            )
            faiss.normalize_L2(query_embedding)

            k = min(top_k, len(values))
            scores, indices = index.search(query_embedding, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    results.append({
                        "value": values[idx],
                        "score": float(score),
                        "class": class_name,
                        "property": property_name,
                    })

            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def has_index(self, class_name: str, property_name: str) -> bool:
        """Check if an index exists for this class+property."""
        return (class_name, property_name) in self._indices


# Singleton instance
_property_value_index: PropertyValueIndex | None = None


def get_property_value_index() -> PropertyValueIndex:
    """Get or create the property value index singleton."""
    global _property_value_index
    if _property_value_index is None:
        _property_value_index = PropertyValueIndex()
    return _property_value_index


@tool
def search_property_values(
    class_name: str,
    property_name: str,
    search_term: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Semantically search for data values of a specific class+property combination.

    The agent specifies which class and datatype property to search,
    and the tool returns values that semantically match the search term.

    Args:
        class_name: The class (as previously discovered by a class search).
        property_name: The datatype property (as previously discovered by a property search).
        search_term: What to search for (free-form — paraphrases allowed).
        top_k: Number of results to return

    Returns:
        List of matching values with similarity scores

    Example:
        search_property_values("Author", "name", "English playwright")
    """
    index = get_property_value_index()

    # Check if we have this combination indexed
    if not index.has_index(class_name, property_name):
        return [{
            "error": f"No index for {class_name}.{property_name}",
            "hint": "Use index_property_values first to index the data",
        }]

    return index.search(class_name, property_name, search_term, top_k)


# =============================================================================
# SCHEMA-LEVEL SEMANTIC SEARCH
# =============================================================================

# Common prefix URIs for different datasets (used by SchemaIndex and tools)
PREFIX_MAP = {
    "ub": "http://example.org/ontology/education#",
    "npdv": "http://example.org/ontology/energy#",
    "trn": "http://example.org/ontology/transport#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
}


# =============================================================================
# HierarchyIndex: Class hierarchy from OWL ontologies (rdfs:subClassOf)
# =============================================================================


class HierarchyIndex:
    """Index for class hierarchies from OWL ontology files.

    Parses rdfs:subClassOf relations from ontology.owl files and provides:
    - Parent class lookup (direct superclasses)
    - Child class lookup (direct subclasses)
    - Ancestor chain (transitive closure of parents)
    - Descendant tree (transitive closure of children)

    This helps the LLM understand class relationships without additional tool calls.
    """

    _instance: "HierarchyIndex | None" = None

    def __init__(self):
        """Initialize the hierarchy index."""
        # Map: (class_name, dataset) -> {"parents": set, "children": set}
        self._hierarchy: dict[tuple[str, str], dict[str, set[str]]] = {}
        self._indexed_datasets: set[str] = set()
        # Cache for transitive closures
        self._ancestor_cache: dict[tuple[str, str], list[str]] = {}
        self._descendant_cache: dict[tuple[str, str], list[str]] = {}

    @classmethod
    def get_instance(cls) -> "HierarchyIndex":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def index_dataset(self, dataset: str) -> int:
        """Parse ontology.owl for a dataset and extract subClassOf relations.

        Args:
            dataset: Dataset name (e.g., "EDU", "NRG")

        Returns:
            Number of class hierarchy relations found
        """
        from src.config import get_ontology_path

        if dataset in self._indexed_datasets:
            return 0

        ontology_path = get_ontology_path(dataset)
        if not ontology_path.exists():
            logger.warning(f"No ontology.owl found for {dataset} at {ontology_path}")
            self._indexed_datasets.add(dataset)
            return 0

        try:
            g = Graph()
            g.parse(str(ontology_path), format="xml")

            count = 0
            # Extract subClassOf relations
            for subclass, superclass in g.subject_objects(RDFS.subClassOf):
                # Get local names (without namespace)
                sub_name = self._extract_local_name(str(subclass))
                super_name = self._extract_local_name(str(superclass))

                if not sub_name or not super_name:
                    continue

                # Initialize entries if needed
                sub_key = (sub_name, dataset)
                super_key = (super_name, dataset)

                if sub_key not in self._hierarchy:
                    self._hierarchy[sub_key] = {"parents": set(), "children": set()}
                if super_key not in self._hierarchy:
                    self._hierarchy[super_key] = {"parents": set(), "children": set()}

                # Add bidirectional relation
                self._hierarchy[sub_key]["parents"].add(super_name)
                self._hierarchy[super_key]["children"].add(sub_name)
                count += 1

            self._indexed_datasets.add(dataset)
            # Clear caches for this dataset
            self._ancestor_cache = {k: v for k, v in self._ancestor_cache.items() if k[1] != dataset}
            self._descendant_cache = {k: v for k, v in self._descendant_cache.items() if k[1] != dataset}

            logger.info(f"Indexed {count} subClassOf relations for {dataset}")
            return count

        except Exception as e:
            logger.error(f"Failed to parse ontology for {dataset}: {e}")
            self._indexed_datasets.add(dataset)
            return 0

    def _extract_local_name(self, uri: str) -> str:
        """Extract local name from URI (e.g., 'http://...#Student' -> 'Student')."""
        if "#" in uri:
            return uri.split("#")[-1]
        elif "/" in uri:
            return uri.split("/")[-1]
        return uri

    def index_all_datasets(self) -> int:
        """Index all available datasets."""
        total = 0
        for dataset in AVAILABLE_DATASETS:
            total += self.index_dataset(dataset)
        return total

    def ensure_indexed(self, dataset: str) -> None:
        """Ensure a dataset is indexed."""
        if dataset not in self._indexed_datasets:
            self.index_dataset(dataset)

    def get_parents(self, class_name: str, dataset: str) -> list[str]:
        """Get immediate parent classes (direct superclasses).

        Args:
            class_name: Class name without prefix (e.g., "DoctoralCandidate")
            dataset: Dataset name (e.g., "EDU")

        Returns:
            List of parent class names
        """
        self.ensure_indexed(dataset)
        key = (class_name, dataset)
        if key in self._hierarchy:
            return sorted(self._hierarchy[key]["parents"])
        return []

    def get_children(self, class_name: str, dataset: str) -> list[str]:
        """Get immediate child classes (direct subclasses).

        Args:
            class_name: Class name without prefix (e.g., "Student")
            dataset: Dataset name (e.g., "EDU")

        Returns:
            List of child class names
        """
        self.ensure_indexed(dataset)
        key = (class_name, dataset)
        if key in self._hierarchy:
            return sorted(self._hierarchy[key]["children"])
        return []

    def get_ancestors(self, class_name: str, dataset: str) -> list[str]:
        """Get all ancestors (transitive closure of parents).

        Returns ancestors in order from immediate parent to root.

        Args:
            class_name: Class name without prefix
            dataset: Dataset name

        Returns:
            List of ancestor class names (closest first)
        """
        self.ensure_indexed(dataset)
        key = (class_name, dataset)

        # Check cache
        if key in self._ancestor_cache:
            return self._ancestor_cache[key]

        # Compute transitive closure
        ancestors = []
        visited = set()
        queue = list(self.get_parents(class_name, dataset))

        while queue:
            parent = queue.pop(0)
            if parent not in visited:
                visited.add(parent)
                ancestors.append(parent)
                queue.extend(self.get_parents(parent, dataset))

        self._ancestor_cache[key] = ancestors
        return ancestors

    def get_descendants(self, class_name: str, dataset: str) -> list[str]:
        """Get all descendants (transitive closure of children).

        Args:
            class_name: Class name without prefix
            dataset: Dataset name

        Returns:
            List of descendant class names
        """
        self.ensure_indexed(dataset)
        key = (class_name, dataset)

        # Check cache
        if key in self._descendant_cache:
            return self._descendant_cache[key]

        # Compute transitive closure
        descendants = []
        visited = set()
        queue = list(self.get_children(class_name, dataset))

        while queue:
            child = queue.pop(0)
            if child not in visited:
                visited.add(child)
                descendants.append(child)
                queue.extend(self.get_children(child, dataset))

        self._descendant_cache[key] = descendants
        return descendants

    def get_hierarchy_info(self, class_name: str, dataset: str) -> dict:
        """Get complete hierarchy info for a class.

        Args:
            class_name: Class name without prefix
            dataset: Dataset name

        Returns:
            Dict with parents, children, ancestors, descendants, and depth
        """
        self.ensure_indexed(dataset)
        ancestors = self.get_ancestors(class_name, dataset)

        return {
            "class": class_name,
            "dataset": dataset,
            "parents": self.get_parents(class_name, dataset),
            "children": self.get_children(class_name, dataset),
            "ancestors": ancestors,
            "descendants": self.get_descendants(class_name, dataset),
            "hierarchy_depth": len(ancestors),
        }

    def get_all_classes_with_hierarchy(self, dataset: str) -> list[dict]:
        """Get all classes with their hierarchy info for a dataset.

        Useful for building connection graphs that include subClassOf edges.
        """
        self.ensure_indexed(dataset)
        result = []
        for (class_name, ds), info in self._hierarchy.items():
            if ds == dataset:
                result.append({
                    "class": class_name,
                    "parents": sorted(info["parents"]),
                    "children": sorted(info["children"]),
                })
        return result


def get_hierarchy_index() -> HierarchyIndex:
    """Get the singleton HierarchyIndex instance."""
    return HierarchyIndex.get_instance()


class SchemaIndex:
    """Index for semantic search over schema elements with persistent caching.

    Supports indexing all datasets with endpoint metadata for Dataspace scenarios.
    Uses a two-pass approach to deduplicate classes and collect connection metadata.
    """

    def __init__(self):
        """Initialize the schema index."""
        settings = get_settings()
        self.embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
        self._index: faiss.IndexFlatIP | None = None
        self._elements: list[dict] = []
        self._indexed_datasets: set[str] = set()
        # Registries for two-pass parsing (collect then deduplicate)
        self._class_registry: dict[tuple[str, str], dict] = {}  # (class_name, dataset) -> metadata
        self._property_registry: dict[tuple[str, str, str, str], dict] = {}  # (prop_name, domain, range, dataset) -> metadata
        # Prefix registry: prefix -> URI mapping
        self._prefix_registry: dict[str, str] = {}

    def _get_cache_paths(self, dataset: str) -> tuple[Path, Path]:
        """Get cache file paths for a dataset."""
        cache_dir = EMBEDDINGS_CACHE_DIR / dataset
        cache_dir.mkdir(parents=True, exist_ok=True)
        return (
            cache_dir / "schema_index.faiss",
            cache_dir / "schema_elements.json",
        )

    def _get_global_cache_paths(self) -> tuple[Path, Path]:
        """Get cache file paths for the global (all datasets) index."""
        cache_dir = EMBEDDINGS_CACHE_DIR / "_global"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return (
            cache_dir / "schema_index.faiss",
            cache_dir / "schema_elements.json",
        )

    def _load_from_cache(self, dataset: str) -> bool:
        """
        Try to load index from cache.

        Returns:
            True if loaded successfully, False otherwise
        """
        faiss_path, elements_path = self._get_cache_paths(dataset)

        if not faiss_path.exists() or not elements_path.exists():
            logger.info(f"No cache found for {dataset}")
            return False

        try:
            # Load FAISS index
            self._index = faiss.read_index(str(faiss_path))

            # Load elements metadata
            with open(elements_path, "r", encoding="utf-8") as f:
                self._elements = json.load(f)

            self._indexed_datasets = {dataset}
            logger.info(f"Loaded {len(self._elements)} elements from cache for {dataset}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load cache for {dataset}: {e}")
            return False

    def _load_global_cache(self) -> bool:
        """
        Try to load the global (all datasets) index from cache.

        Returns:
            True if loaded successfully, False otherwise
        """
        faiss_path, elements_path = self._get_global_cache_paths()
        prefixes_path = self._get_global_cache_paths()[0].parent / "prefixes.json"

        if not faiss_path.exists() or not elements_path.exists():
            logger.info("No global cache found")
            return False

        try:
            self._index = faiss.read_index(str(faiss_path))
            with open(elements_path, "r", encoding="utf-8") as f:
                self._elements = json.load(f)

            # Load prefixes if available
            if prefixes_path.exists():
                with open(prefixes_path, "r", encoding="utf-8") as f:
                    self._prefix_registry = json.load(f)

            # Extract indexed datasets from elements
            self._indexed_datasets = {e.get("dataset", "") for e in self._elements if e.get("dataset")}

            # Warn if cache contains datasets no longer in AVAILABLE_DATASETS (stale cache).
            # Elements from removed datasets are kept in memory (FAISS index ordering must
            # match) but will return stale endpoints. Recommend rebuild via
            # index_all_datasets(force_reindex=True).
            stale = self._indexed_datasets - set(AVAILABLE_DATASETS)
            if stale:
                logger.warning(
                    f"Global schema cache contains datasets not in AVAILABLE_DATASETS: {stale}. "
                    f"Recommend rebuild: rm -rf {faiss_path.parent}"
                )

            logger.info(f"Loaded {len(self._elements)} elements from global cache ({len(self._indexed_datasets)} datasets)")
            return True
        except Exception as e:
            logger.warning(f"Failed to load global cache: {e}")
            return False

    def _save_global_cache(self) -> bool:
        """Save the global index to cache."""
        if self._index is None or not self._elements:
            return False

        faiss_path, elements_path = self._get_global_cache_paths()
        prefixes_path = faiss_path.parent / "prefixes.json"

        try:
            faiss.write_index(self._index, str(faiss_path))
            with open(elements_path, "w", encoding="utf-8") as f:
                json.dump(self._elements, f, ensure_ascii=False, indent=2)

            # Save prefixes
            with open(prefixes_path, "w", encoding="utf-8") as f:
                json.dump(self._prefix_registry, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved {len(self._elements)} elements to global cache")
            return True
        except Exception as e:
            logger.error(f"Failed to save global cache: {e}")
            return False

    def _save_to_cache(self, dataset: str) -> bool:
        """
        Save index to cache.

        Returns:
            True if saved successfully, False otherwise
        """
        if self._index is None or not self._elements:
            return False

        faiss_path, elements_path = self._get_cache_paths(dataset)

        try:
            # Save FAISS index
            faiss.write_index(self._index, str(faiss_path))

            # Save elements metadata
            with open(elements_path, "w", encoding="utf-8") as f:
                json.dump(self._elements, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved {len(self._elements)} elements to cache for {dataset}")
            return True
        except Exception as e:
            logger.error(f"Failed to save cache for {dataset}: {e}")
            return False

    def index_dataset(self, dataset: str, force_reindex: bool = False) -> int:
        """
        Index all schema elements from a dataset.

        First tries to load from cache, only creates new embeddings if needed.

        Args:
            dataset: Dataset name (EDU, NRG, etc.)
            force_reindex: If True, ignore cache and rebuild index

        Returns:
            Number of elements indexed
        """
        # Check if already loaded for this dataset
        if not force_reindex and dataset in self._indexed_datasets and self._elements:
            return len(self._elements)

        # Try to load from cache first
        if not force_reindex and self._load_from_cache(dataset):
            return len(self._elements)

        # Build index from scratch
        logger.info(f"Building new index for {dataset} (this may take a while)...")

        search_dir = get_semantic_models_path(dataset)
        if not search_dir.exists():
            return 0

        # Clear and initialize registries for two-pass parsing
        self._class_registry.clear()
        self._property_registry.clear()
        endpoint = get_endpoint_for_dataset(dataset)

        # Pass 1: Parse all files and collect into registries
        for file_path in search_dir.glob("**/*.ttl"):
            try:
                content = file_path.read_text(encoding="utf-8")
                self._parse_schema_file(content, file_path.name, dataset, endpoint)
            except Exception as e:
                logger.warning(f"Failed to parse {file_path}: {e}")

        # Pass 2: Finalize - deduplicate and convert to elements
        self._finalize_elements()

        if not self._elements:
            return 0

        # Create embeddings
        texts = [e["searchable_text"] for e in self._elements]

        try:
            logger.info(f"Creating embeddings for {len(texts)} schema elements...")
            embeddings_list = self.embeddings.embed_documents(texts)
            embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
            faiss.normalize_L2(embeddings_matrix)

            dimension = embeddings_matrix.shape[1]
            self._index = faiss.IndexFlatIP(dimension)
            self._index.add(embeddings_matrix)

            self._indexed_datasets = {dataset}
            logger.info(f"Indexed {len(self._elements)} schema elements from {dataset}")

            # Save to cache for next time
            self._save_to_cache(dataset)

            return len(self._elements)
        except Exception as e:
            logger.error(f"Failed to create schema index: {e}")
            return 0

    def index_all_datasets(self, force_reindex: bool = False) -> int:
        """
        Index schema elements from ALL available datasets.

        Each element will include dataset and sparql_endpoint metadata.

        Args:
            force_reindex: If True, ignore cache and rebuild index

        Returns:
            Total number of elements indexed across all datasets
        """
        # Try to load global cache first
        if not force_reindex and self._load_global_cache():
            return len(self._elements)

        logger.info("Building global index for all datasets...")

        # Clear and initialize registries for two-pass parsing
        self._class_registry.clear()
        self._property_registry.clear()

        # Pass 1: Parse all files from all datasets into registries
        for dataset in AVAILABLE_DATASETS:
            search_dir = get_semantic_models_path(dataset)
            if not search_dir.exists():
                logger.warning(f"No semantic models directory for {dataset}")
                continue

            endpoint = get_endpoint_for_dataset(dataset)

            for file_path in search_dir.glob("**/*.ttl"):
                try:
                    content = file_path.read_text(encoding="utf-8")
                    self._parse_schema_file(content, file_path.name, dataset, endpoint)
                except Exception as e:
                    logger.warning(f"Failed to parse {file_path}: {e}")

        # Pass 2: Finalize - deduplicate and convert to elements
        self._finalize_elements()

        if not self._elements:
            logger.warning("No schema elements found in any dataset")
            return 0

        # Create embeddings
        texts = [e["searchable_text"] for e in self._elements]

        try:
            logger.info(f"Creating embeddings for {len(texts)} schema elements across all datasets...")
            embeddings_list = self.embeddings.embed_documents(texts)
            embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
            faiss.normalize_L2(embeddings_matrix)

            dimension = embeddings_matrix.shape[1]
            self._index = faiss.IndexFlatIP(dimension)
            self._index.add(embeddings_matrix)

            self._indexed_datasets = set(AVAILABLE_DATASETS)
            logger.info(f"Indexed {len(self._elements)} schema elements from {len(self._indexed_datasets)} datasets")

            # Save to global cache
            self._save_global_cache()

            return len(self._elements)
        except Exception as e:
            logger.error(f"Failed to create global schema index: {e}")
            return 0

    def _get_qname(self, g: Graph, uri: str) -> tuple[str, str]:
        """Get prefix and local name from URI using namespace manager.

        Returns:
            Tuple of (prefix, local_name)
        """
        qname = g.namespace_manager.qname(uri)
        prefix, local_name = qname.split(":", 1)
        return prefix, local_name

    def _parse_schema_file(
        self,
        content: str,
        file_name: str,
        dataset: str | None = None,
        sparql_endpoint: str | None = None,
    ) -> None:
        """Parse a schema file and collect metadata into registries.

        Uses a two-pass approach:
        1. Collect class and property info into registries (this method)
        2. Deduplicate and finalize into _elements (in _finalize_elements)

        Extracts connection metadata:
        - Classes: defined_in, referenced_in, connects_to, connected_from
        - Properties: domain_class, range_class, connects, is_object_property
        """
        g = Graph()
        g.parse(data=content, format="turtle")

        # Extract and store prefix declarations
        for prefix, namespace in g.namespaces():
            if prefix:  # Skip empty prefix
                self._prefix_registry[prefix] = str(namespace)

        subjects = set(g.subjects())

        for subject in subjects:
            subject_str = str(subject)
            class_prefix, class_name = self._get_qname(g, subject_str)
            full_class_name = f"{class_prefix}:{class_name}"

            # Register or update class
            class_key = (class_name, dataset)
            if class_key not in self._class_registry:
                self._class_registry[class_key] = {
                    "type": "class",
                    "name": class_name,
                    "prefix": class_prefix,
                    "full_name": full_class_name,
                    "uri": subject_str,
                    "defined_in": [],           # Files where class is the subject
                    "referenced_in": set(),     # Files where class is referenced as range
                    "connects_to": set(),       # Classes this class connects TO via object properties
                    "connected_from": set(),    # Classes that connect TO this class
                    "dataset": dataset,
                    "sparql_endpoint": sparql_endpoint,
                    "searchable_text": self._expand_term(class_name),
                }

            # This class is defined (as subject) in this file
            if file_name not in self._class_registry[class_key]["defined_in"]:
                self._class_registry[class_key]["defined_in"].append(file_name)

            # Process all predicate-object pairs
            for predicate, obj in g.predicate_objects(subject):
                pred_str = str(predicate)
                obj_str = str(obj)

                prop_prefix, prop_name = self._get_qname(g, pred_str)
                full_prop_name = f"{prop_prefix}:{prop_name}"

                range_prefix, range_name = self._get_qname(g, obj_str)
                full_range = f"{range_prefix}:{range_name}"

                is_datatype = str(XSD) in obj_str
                is_object_property = not is_datatype

                # Register or update property
                prop_key = (prop_name, class_name, range_name, dataset)
                if prop_key not in self._property_registry:
                    self._property_registry[prop_key] = {
                        "type": "property",
                        "name": prop_name,
                        "prefix": prop_prefix,
                        "full_name": full_prop_name,
                        "uri": pred_str,
                        "domain_class": full_class_name,
                        "domain_class_name": class_name,
                        "range_class": full_range,
                        "range_class_name": range_name,
                        "is_datatype": is_datatype,
                        "is_object_property": is_object_property,
                        "defined_in": [],           # Files where this property is defined
                        "connects": [class_name, range_name] if is_object_property else [],
                        "dataset": dataset,
                        "sparql_endpoint": sparql_endpoint,
                        "searchable_text": f"{self._expand_term(prop_name)} of {self._expand_term(class_name)}",
                    }

                # This property is defined in this file
                if file_name not in self._property_registry[prop_key]["defined_in"]:
                    self._property_registry[prop_key]["defined_in"].append(file_name)

                # Track connections for object properties
                if is_object_property:
                    # This class connects TO the range class
                    self._class_registry[class_key]["connects_to"].add(range_name)

                    # Register or update the target class (range) - it's referenced here
                    target_key = (range_name, dataset)
                    if target_key not in self._class_registry:
                        self._class_registry[target_key] = {
                            "type": "class",
                            "name": range_name,
                            "prefix": range_prefix,
                            "full_name": full_range,
                            "uri": obj_str,
                            "defined_in": [],
                            "referenced_in": set(),
                            "connects_to": set(),
                            "connected_from": set(),
                            "dataset": dataset,
                            "sparql_endpoint": sparql_endpoint,
                            "searchable_text": self._expand_term(range_name),
                        }

                    # The target class is referenced in this file and connected FROM this class
                    self._class_registry[target_key]["referenced_in"].add(file_name)
                    self._class_registry[target_key]["connected_from"].add(class_name)

    def _finalize_elements(self) -> None:
        """Convert registries to deduplicated elements list.

        Call this after all files have been parsed.
        Converts sets to sorted lists for JSON serialization.
        """
        self._elements = []

        # Add classes (deduplicated)
        for class_key, class_data in self._class_registry.items():
            elem = class_data.copy()
            # Convert sets to sorted lists for JSON serialization
            elem["referenced_in"] = sorted(elem["referenced_in"])
            elem["connects_to"] = sorted(elem["connects_to"])
            elem["connected_from"] = sorted(elem["connected_from"])
            elem["content"] = elem["full_name"]
            self._elements.append(elem)

        # Add properties (deduplicated by domain+range combination)
        for prop_key, prop_data in self._property_registry.items():
            elem = prop_data.copy()
            elem["content"] = f"{elem['domain_class']} {elem['full_name']} {elem['range_class']}"
            self._elements.append(elem)

        # Clear registries to free memory
        self._class_registry.clear()
        self._property_registry.clear()

    def _expand_term(self, term: str) -> str:
        """Expand camelCase to separate words."""
        expanded = re.sub(r'([a-z])([A-Z])', r'\1 \2', term)
        return f"{term} {expanded}" if expanded != term else term

    def _slot_aware_endpoint(self, elem: dict) -> str | None:
        """Resolve the element's endpoint lazily via get_endpoint_for_dataset.

        The cached ``sparql_endpoint`` field stored on disk is a static URL from
        the time the index was built (e.g. ``http://127.0.0.1:8084/sparql``). Under
        slot-based container isolation those ports do not exist. Always resolve
        from the dataset name via ``get_endpoint_for_dataset`` so the current
        ``slot_context`` / ``dataset_size_context`` is respected.
        """
        ds = elem.get("dataset")
        if not ds:
            return elem.get("sparql_endpoint")
        resolved = get_endpoint_for_dataset(ds)
        return resolved or elem.get("sparql_endpoint")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search for schema elements matching a query.

        Returns results with dataset, sparql_endpoint, and connection metadata.
        For classes: defined_in, referenced_in, connects_to, connected_from
        For properties: domain_class, range_class, connects, is_object_property, defined_in
        """
        if not self._elements or self._index is None:
            return []

        try:
            query_embedding = np.array(
                [self.embeddings.embed_query(query)], dtype=np.float32
            )
            faiss.normalize_L2(query_embedding)

            k = min(top_k, len(self._elements))
            scores, indices = self._index.search(query_embedding, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    elem = self._elements[idx]

                    if elem["type"] == "class":
                        results.append({
                            "type": "class",
                            "name": elem["full_name"],
                            "defined_in": elem.get("defined_in", []),
                            "referenced_in": elem.get("referenced_in", []),
                            "connects_to": elem.get("connects_to", []),
                            "connected_from": elem.get("connected_from", []),
                            "dataset": elem.get("dataset"),
                            "sparql_endpoint": self._slot_aware_endpoint(elem),
                            "content": elem["content"],
                            "score": float(score),
                        })
                    else:  # property
                        results.append({
                            "type": "property",
                            "name": elem["full_name"],
                            "domain_class": elem.get("domain_class"),
                            "range_class": elem.get("range_class"),
                            "is_object_property": elem.get("is_object_property", False),
                            "connects": elem.get("connects", []),
                            "defined_in": elem.get("defined_in", []),
                            "dataset": elem.get("dataset"),
                            "sparql_endpoint": self._slot_aware_endpoint(elem),
                            "content": elem["content"],
                            "score": float(score),
                        })

            return results
        except Exception as e:
            logger.error(f"Schema search failed: {e}")
            return []

    def search_by_type(
        self,
        query: str,
        element_type: str,
        top_k: int = 10,
    ) -> list[dict]:
        """Search for schema elements of a specific type only.

        Args:
            query: Natural language search query
            element_type: Either "class" or "property"
            top_k: Number of results to return

        Returns:
            List of matching elements filtered by type
        """
        if not self._elements or self._index is None:
            return []

        if element_type not in ("class", "property"):
            logger.warning(f"Invalid element_type: {element_type}")
            return []

        try:
            query_embedding = np.array(
                [self.embeddings.embed_query(query)], dtype=np.float32
            )
            faiss.normalize_L2(query_embedding)

            # Search more than needed, then filter by type
            search_k = min(top_k * 3, len(self._elements))
            scores, indices = self._index.search(query_embedding, search_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and len(results) < top_k:
                    elem = self._elements[idx]

                    # Filter by type
                    if elem.get("type") != element_type:
                        continue

                    if element_type == "class":
                        results.append({
                            "type": "class",
                            "name": elem["full_name"],
                            "defined_in": elem.get("defined_in", []),
                            "referenced_in": elem.get("referenced_in", []),
                            "connects_to": elem.get("connects_to", []),
                            "connected_from": elem.get("connected_from", []),
                            "dataset": elem.get("dataset"),
                            "sparql_endpoint": self._slot_aware_endpoint(elem),
                            "score": float(score),
                        })
                    else:  # property
                        results.append({
                            "type": "property",
                            "name": elem["full_name"],
                            "domain_class": elem.get("domain_class"),
                            "range_class": elem.get("range_class"),
                            "is_object_property": elem.get("is_object_property", False),
                            "connects": elem.get("connects", []),
                            "defined_in": elem.get("defined_in", []),
                            "dataset": elem.get("dataset"),
                            "sparql_endpoint": self._slot_aware_endpoint(elem),
                            "score": float(score),
                        })

            return results
        except Exception as e:
            logger.error(f"Schema search by type failed: {e}")
            return []

    def is_global_indexed(self) -> bool:
        """Check if all datasets are indexed."""
        return len(self._indexed_datasets) == len(AVAILABLE_DATASETS)

    def get_sparql_prefixes(self, used_prefixes: list[str] | None = None) -> str:
        """Get prefix declarations in SPARQL format.

        Args:
            used_prefixes: Optional list of prefix names to include.
                          If None, returns all known prefixes.

        Returns:
            SPARQL prefix declarations as a string
        """
        prefixes_to_include = used_prefixes or list(self._prefix_registry.keys())
        lines = []
        for prefix in prefixes_to_include:
            if prefix in self._prefix_registry:
                uri = self._prefix_registry[prefix]
                lines.append(f"PREFIX {prefix}: <{uri}>")
        return "\n".join(lines)

    def get_all_prefixes(self) -> dict[str, str]:
        """Get all known prefix to URI mappings."""
        return self._prefix_registry.copy()

    def _build_connection_graph(
        self,
        dataset: str | None = None,
        include_hierarchy: bool = True,
    ) -> dict[str, dict[str, list[dict]]]:
        """Build a graph of class connections from indexed elements.

        Args:
            dataset: Optional dataset filter
            include_hierarchy: If True, include rdfs:subClassOf edges from ontology

        Returns:
            Dict mapping class_name -> {target_class: [property_info, ...]}
        """
        graph: dict[str, dict[str, list[dict]]] = {}

        # Add object property edges (existing behavior)
        for elem in self._elements:
            if elem.get("type") != "property":
                continue
            if not elem.get("is_object_property"):
                continue
            if dataset and elem.get("dataset") != dataset:
                continue

            domain = elem.get("domain_class_name") or elem.get("domain_class", "").split(":")[-1]
            range_cls = elem.get("range_class_name") or elem.get("range_class", "").split(":")[-1]

            if not domain or not range_cls:
                continue

            if domain not in graph:
                graph[domain] = {}
            if range_cls not in graph[domain]:
                graph[domain][range_cls] = []

            graph[domain][range_cls].append({
                "property": elem.get("full_name") or elem.get("name"),
                "domain_full": elem.get("domain_class"),
                "range_full": elem.get("range_class"),
                "dataset": elem.get("dataset"),
                "prefix": elem.get("prefix"),
                "edge_type": "object_property",
            })

        # Add rdfs:subClassOf edges from hierarchy (NEW)
        if include_hierarchy:
            hierarchy_index = get_hierarchy_index()
            datasets_to_check = [dataset] if dataset else AVAILABLE_DATASETS

            for ds in datasets_to_check:
                hierarchy_index.ensure_indexed(ds)
                prefix_map = {
                    "EDU": "ub",
                    "NRG": "npdv",
                    "TRN": "trn",
                    "BSBM": "bsbm",
                    "BGEE": "bgee",
                }
                prefix = prefix_map.get(ds, "")

                for class_info in hierarchy_index.get_all_classes_with_hierarchy(ds):
                    class_name = class_info["class"]
                    for parent in class_info["parents"]:
                        # Add subClassOf edge: subclass -> superclass
                        if class_name not in graph:
                            graph[class_name] = {}
                        if parent not in graph[class_name]:
                            graph[class_name][parent] = []

                        graph[class_name][parent].append({
                            "property": "rdfs:subClassOf",
                            "domain_full": f"{prefix}:{class_name}" if prefix else class_name,
                            "range_full": f"{prefix}:{parent}" if prefix else parent,
                            "dataset": ds,
                            "prefix": prefix,
                            "edge_type": "hierarchy",
                        })

        return graph

    def find_shortest_path(
        self,
        class_names: list[str],
        dataset: str | None = None,
    ) -> dict:
        """Find the shortest path connecting multiple classes.

        Uses BFS to find connections between classes via object properties.

        Args:
            class_names: List of class names to connect (without prefix)
            dataset: Optional dataset filter

        Returns:
            Dict with path edges, tripels, and connection status
        """
        from collections import deque

        if len(class_names) < 2:
            return {
                "error": "Need at least 2 classes to find a path",
                "classes": class_names,
            }

        # Normalize class names (remove prefix if present)
        normalized = []
        for c in class_names:
            name = c.split(":")[-1] if ":" in c else c
            normalized.append(name)

        # Build bidirectional graph
        graph = self._build_connection_graph(dataset)

        # Add reverse edges for bidirectional search
        bi_graph: dict[str, dict[str, list[dict]]] = {}
        for src, targets in graph.items():
            if src not in bi_graph:
                bi_graph[src] = {}
            for tgt, props in targets.items():
                if tgt not in bi_graph:
                    bi_graph[tgt] = {}
                # Forward edge
                if tgt not in bi_graph[src]:
                    bi_graph[src][tgt] = []
                bi_graph[src][tgt].extend(props)
                # Reverse edge (for traversal, but mark as reverse)
                if src not in bi_graph[tgt]:
                    bi_graph[tgt][src] = []
                for p in props:
                    bi_graph[tgt][src].append({**p, "reverse": True})

        # Find paths between all pairs and collect edges
        all_edges = []
        connected = True

        for i in range(len(normalized) - 1):
            start = normalized[i]
            end = normalized[i + 1]

            path = self._bfs_path(bi_graph, start, end)
            if path is None:
                connected = False
                continue

            for edge in path:
                if edge not in all_edges:
                    all_edges.append(edge)

        # Also try to find connections from any class to any other
        if not all_edges:
            # Try all pairs
            for i, src in enumerate(normalized):
                for j, tgt in enumerate(normalized):
                    if i >= j:
                        continue
                    path = self._bfs_path(bi_graph, src, tgt)
                    if path:
                        for edge in path:
                            if edge not in all_edges:
                                all_edges.append(edge)
                        break
                if all_edges:
                    break

        if not all_edges:
            return {
                "connected": False,
                "classes": normalized,
                "error": f"No path found between classes: {normalized}",
                "available_classes": list(bi_graph.keys())[:20],
            }

        # Generate tripels from edges
        tripels = []
        prefixes_used = set()

        for edge in all_edges:
            domain = edge["domain_full"]
            prop = edge["property"]
            range_cls = edge["range_full"]

            tripel = f"{domain} {prop} {range_cls} ."
            if tripel not in tripels:
                tripels.append(tripel)

            # Collect prefixes
            for val in [domain, prop, range_cls]:
                if ":" in val:
                    prefixes_used.add(val.split(":")[0])

        # Get prefix declarations (fallback to PREFIX_MAP if registry empty)
        prefix_lines = []
        for p in sorted(prefixes_used):
            if p in self._prefix_registry:
                prefix_lines.append(f"PREFIX {p}: <{self._prefix_registry[p]}>")
            elif p in PREFIX_MAP:
                prefix_lines.append(f"PREFIX {p}: <{PREFIX_MAP[p]}>")

        return {
            "connected": connected,
            "classes": normalized,
            "path": all_edges,
            "tripels": tripels,
            "prefixes": "\n".join(prefix_lines),
            "dataset": all_edges[0].get("dataset") if all_edges else dataset,
        }

    def _bfs_path(
        self,
        graph: dict[str, dict[str, list[dict]]],
        start: str,
        end: str,
    ) -> list[dict] | None:
        """BFS to find shortest path between two classes."""
        from collections import deque

        if start == end:
            return []

        if start not in graph:
            return None

        queue = deque([(start, [])])
        visited = {start}

        while queue:
            current, path = queue.popleft()

            if current not in graph:
                continue

            for neighbor, props in graph[current].items():
                if neighbor in visited:
                    continue

                # Pick first property for this edge
                prop_info = props[0]
                edge = {
                    "from": current,
                    "to": neighbor,
                    "property": prop_info["property"],
                    "domain_full": prop_info["domain_full"],
                    "range_full": prop_info["range_full"],
                    "reverse": prop_info.get("reverse", False),
                    "dataset": prop_info.get("dataset"),
                }

                new_path = path + [edge]

                if neighbor == end:
                    return new_path

                visited.add(neighbor)
                queue.append((neighbor, new_path))

        return None


# Singleton
_schema_index: SchemaIndex | None = None


def get_schema_index() -> SchemaIndex:
    """Get or create the schema index singleton."""
    global _schema_index
    if _schema_index is None:
        _schema_index = SchemaIndex()
    return _schema_index


@tool
def semantic_schema_search(
    query: str,
    dataset: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Semantically search for schema elements (classes and properties) across all datasets.

    Use natural language to find relevant classes and properties.
    Results include the SPARQL endpoint for each element.

    IMPORTANT: The first result contains PREFIX DECLARATIONS that MUST be used
    in SPARQL queries. Do not invent prefixes - use only the provided ones!

    Args:
        query: Free-form natural language description of what you want to find.
        dataset: Optional - specific dataset to search. If None, searches all datasets.
        top_k: Number of results

    Returns:
        List of matching schema elements with scores, dataset, and sparql_endpoint.
        First entry contains prefix declarations.

    Example:
        semantic_schema_search("contact information")
        semantic_schema_search("authorship relation")
    """
    index = get_schema_index()

    if dataset:
        # Search specific dataset
        count = index.index_dataset(dataset)
        if count == 0:
            return [{"error": f"Failed to index dataset {dataset}"}]
    else:
        # Search all datasets
        count = index.index_all_datasets()
        if count == 0:
            return [{"error": "Failed to index any datasets"}]

    results = index.search(query, top_k)

    # Collect prefixes used in results
    used_prefixes = set()
    for r in results:
        name = r.get("name", "")
        if ":" in name:
            used_prefixes.add(name.split(":")[0])
        # Also check domain/range for properties
        domain = r.get("domain_class", "")
        if domain and ":" in domain:
            used_prefixes.add(domain.split(":")[0])
        range_class = r.get("range_class", "")
        if range_class and ":" in range_class:
            used_prefixes.add(range_class.split(":")[0])

    # Prepend prefix declarations as first result
    prefixes = index.get_sparql_prefixes(list(used_prefixes))
    if prefixes:
        prefix_entry = {
            "type": "prefixes",
            "content": prefixes,
            "note": "USE THESE EXACT PREFIXES in your SPARQL queries! Do NOT invent prefixes.",
        }
        results = [prefix_entry] + results

    return results


@tool
def search_classes(
    search_term: str,
    dataset: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Semantically search for CLASS definitions across all datasets.

    Uses embedding-based similarity to find relevant classes. Good for
    conceptual / paraphrased queries where the user's wording may not
    match the class's literal name.

    Returns the SAME format as grep_classes for fair comparison.

    Args:
        search_term: Free-form conceptual description of the class to look for.
        dataset: Optional - specific dataset to search. If None, searches all datasets.
        top_k: Number of results to return (default 5). Increase if you need more results.

    Returns:
        List of matching classes with:
        - class: Full prefixed class name (e.g., "lib:Paperback")
        - file: The mapping file where this class is defined
        - dataset: The dataset this class belongs to
        - sparql_endpoint: The SPARQL endpoint for this dataset
        - properties_preview: First 3 properties as preview
        - prefix_declaration: The PREFIX declaration for this class

    Example:
        search_classes("printed publication")
        search_classes("person who wrote a book", top_k=10)
    """
    index = get_schema_index()

    # Ensure index is built
    if dataset:
        count = index.index_dataset(dataset)
        if count == 0:
            return [{"error": f"Failed to index dataset {dataset}"}]
    else:
        count = index.index_all_datasets()
        if count == 0:
            return [{"error": "Failed to index any datasets"}]

    # Search only classes using semantic similarity
    raw_results = index.search_by_type(search_term, element_type="class", top_k=top_k)

    # Convert to same format as grep_classes
    results = []
    for r in raw_results:
        full_class = r.get("name", "")
        ds = r.get("dataset")
        endpoint = r.get("sparql_endpoint")
        defined_in = r.get("defined_in", [])
        file_name = defined_in[0] if defined_in else ""

        # Extract prefix
        prefix = full_class.split(":")[0] if ":" in full_class else ""

        # Get properties preview and prefix URI by reading the semantic model file
        # (same approach as grep_classes for consistency)
        properties_preview = []
        prefix_uri = ""
        if file_name and ds:
            search_dir = get_semantic_models_path(ds)
            file_path = search_dir / file_name
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding="utf-8")
                    properties_preview = _extract_class_properties_preview(content, full_class, max_props=3)

                    # Parse prefix from file (same as grep_classes)
                    for line in content.split("\n"):
                        if line.strip().startswith("@prefix"):
                            match = re.match(r'@prefix\s+(\w+):\s+<([^>]+)>', line.strip())
                            if match and match.group(1) == prefix:
                                prefix_uri = match.group(2)
                                break
                except Exception:
                    pass

        results.append({
            "class": full_class,
            "file": file_name,
            "dataset": ds,
            "sparql_endpoint": endpoint,
            "properties_preview": properties_preview,
            "prefix_declaration": f"PREFIX {prefix}: <{prefix_uri}>" if prefix_uri else "",
        })

    # Deduplicate by (class, dataset) - same as grep_classes
    seen = set()
    unique_results = []
    for r in results:
        key = (r["class"], r["dataset"])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    # Enrich with hierarchy information from OWL ontologies (same as grep_classes)
    hierarchy_index = get_hierarchy_index()
    for result in unique_results:
        # Extract class name without prefix (e.g., "eduo:DoctoralCandidate" -> "DoctoralCandidate")
        class_name = result["class"].split(":")[-1] if ":" in result["class"] else result["class"]
        ds = result["dataset"]

        # Get hierarchy info
        parents = hierarchy_index.get_parents(class_name, ds)
        ancestors = hierarchy_index.get_ancestors(class_name, ds)
        children = hierarchy_index.get_children(class_name, ds)

        # Add prefix to parent/child names for consistency
        prefix = result["class"].split(":")[0] if ":" in result["class"] else ""
        result["parent_classes"] = [f"{prefix}:{p}" for p in parents] if prefix else parents
        result["ancestor_chain"] = [f"{prefix}:{a}" for a in ancestors] if prefix else ancestors
        result["child_classes"] = [f"{prefix}:{c}" for c in children] if prefix else children
        result["hierarchy_depth"] = len(ancestors)

    return unique_results


@tool
def search_properties(
    search_term: str,
    dataset: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Semantically search for PROPERTY definitions across all datasets.

    Uses embedding-based similarity to find relevant properties. Good
    for conceptual / paraphrased queries where the user's wording may
    not match the property's literal name.

    Returns the SAME format as grep_properties for fair comparison.

    Args:
        search_term: Free-form conceptual description of the relationship.
        dataset: Optional - specific dataset to search. If None, searches all datasets.
        top_k: Number of top results to return (default 5). Increase if you need more matches.

    Returns:
        List of matching properties with:
        - property: Full prefixed property name (e.g., "lib:writtenBy")
        - domain_class: The class that has this property
        - range: What this property points to (class or xsd type)
        - is_object_property: True if connects to another class
        - file: The mapping file where this property is defined
        - dataset: The dataset this property belongs to
        - sparql_endpoint: The SPARQL endpoint for this dataset

    Example:
        search_properties("authorship")
        search_properties("membership in a collection")
    """
    index = get_schema_index()

    # Ensure index is built
    if dataset:
        count = index.index_dataset(dataset)
        if count == 0:
            return [{"error": f"Failed to index dataset {dataset}"}]
    else:
        count = index.index_all_datasets()
        if count == 0:
            return [{"error": "Failed to index any datasets"}]

    # Search only properties using semantic similarity
    raw_results = index.search_by_type(search_term, element_type="property", top_k=top_k)

    # Convert to same format as grep_properties
    results = []
    for r in raw_results:
        defined_in = r.get("defined_in", [])
        file_name = defined_in[0] if defined_in else ""

        results.append({
            "property": r.get("name", ""),
            "domain_class": r.get("domain_class", ""),
            "range": r.get("range_class", ""),
            "is_object_property": r.get("is_object_property", False),
            "file": file_name,
            "dataset": r.get("dataset"),
            "sparql_endpoint": r.get("sparql_endpoint"),
        })

    # Deduplicate by (property, domain, range, dataset) - same as grep_properties
    seen = set()
    unique_results = []
    for r in results:
        key = (r["property"], r["domain_class"], r["range"], r["dataset"])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    return unique_results


@tool
def find_shortest_path(
    classes: list[str],
    dataset: str | None = None,
) -> dict:
    """
    Find the shortest path connecting multiple classes in the schema.

    Use this after identifying central classes to understand how they connect.
    Returns the object properties that link the classes together.

    Args:
        classes: List of class names to connect (e.g., ["Book", "Author"]).
                 Can include prefix (e.g., "lib:Book") or just the local name.
        dataset: Optional - specific dataset. If None, searches all datasets.

    Returns:
        Dict with:
        - connected: True if all classes are connected
        - classes: The input class names (normalized)
        - path: List of edges with from/to/property information
        - tripels: Ready-to-use tripels in Turtle format
        - prefixes: PREFIX declarations for the tripels
        - dataset: The dataset where connections were found

    Example (shape only — real class/property/prefix names depend on the
    ontology being queried):
        find_shortest_path(["Book", "Author"])
        # Might return:
        # {
        #   "connected": True,
        #   "path": [{"from": "Book", "property": "<prefix>:writtenBy", "to": "Author"}],
        #   "tripels": ["<prefix>:Book <prefix>:writtenBy <prefix>:Author ."],
        #   "prefixes": "PREFIX <prefix>: <http://example.org/library#>"
        # }
    """
    index = get_schema_index()

    # Ensure index is built
    if dataset:
        count = index.index_dataset(dataset)
        if count == 0:
            return {"error": f"Failed to index dataset {dataset}", "classes": classes}
    else:
        count = index.index_all_datasets()
        if count == 0:
            return {"error": "Failed to index any datasets", "classes": classes}

    return index.find_shortest_path(classes, dataset)


# =============================================================================
# DATA INSTANCE RETRIEVAL
# =============================================================================


_PREFIX_CACHE_BY_DATASET: dict[str, dict[str, str]] = {}


def _load_prefixes_from_mapping(dataset: str | None) -> dict[str, str]:
    """Load prefix declarations from mapping.ttl files.

    Uses absolute paths via ``MAPPINGS_DIR_ROOT`` so resolution is independent
    of the caller's CWD. Results are cached per dataset on first load.

    Args:
        dataset: Dataset name (e.g., "trn", "nrg", "edu"). If ``None``, loads
            from the generic ``mappings/mapping.ttl`` (if present).

    Returns:
        Dict mapping prefix names to URIs (empty if no mapping file found)
    """
    cache_key = (dataset or "").lower()
    if cache_key in _PREFIX_CACHE_BY_DATASET:
        return _PREFIX_CACHE_BY_DATASET[cache_key]

    prefixes: dict[str, str] = {}
    if dataset:
        mapping_paths = [
            MAPPINGS_DIR_ROOT / dataset.lower() / "mapping.ttl",
            MAPPINGS_DIR_ROOT / dataset / "mapping.ttl",
        ]
    else:
        mapping_paths = [MAPPINGS_DIR_ROOT / "mapping.ttl"]

    prefix_pattern = r"@prefix\s+(\w+):\s*<([^>]+)>\s*\."
    for mapping_path in mapping_paths:
        if mapping_path.exists():
            try:
                content = mapping_path.read_text(encoding="utf-8")
                for match in re.finditer(prefix_pattern, content):
                    prefix_name, prefix_uri = match.groups()
                    prefixes[prefix_name] = prefix_uri
                break  # Use first found mapping file
            except Exception as e:
                logger.warning(
                    f"_load_prefixes_from_mapping: failed to read {mapping_path}: {e}"
                )
                continue
    if not prefixes:
        logger.warning(
            f"_load_prefixes_from_mapping: no mapping.ttl found for dataset={dataset!r} "
            f"(searched under {MAPPINGS_DIR_ROOT})"
        )

    _PREFIX_CACHE_BY_DATASET[cache_key] = prefixes
    return prefixes


def _get_prefix_declarations_for_query(
    prefixes: set[str],
    dataset: str | None = None,
    custom_prefix_uris: dict[str, str] | None = None,
) -> str:
    """Generate PREFIX declarations for a set of prefixes.

    Resolution order (first match wins):
    1. ``custom_prefix_uris`` passed by the agent
    2. The dataset's ``mapping.ttl`` (the authoritative source — this is what
       OnTop uses at query time)
    3. Static ``PREFIX_MAP`` for common prefixes (xsd/rdf/rdfs/etc.)

    If a prefix cannot be resolved, it is **omitted** (no silent fallback
    URI) — a missing PREFIX causes a clear SPARQL parse error rather than a
    successful query against a nonexistent namespace returning 0 matches.
    """
    mapping_prefixes = _load_prefixes_from_mapping(dataset) if dataset else {}

    declarations = []
    unresolved: list[str] = []
    for p in prefixes:
        if custom_prefix_uris and p in custom_prefix_uris:
            declarations.append(f"PREFIX {p}: <{custom_prefix_uris[p]}>")
        elif p in mapping_prefixes:
            declarations.append(f"PREFIX {p}: <{mapping_prefixes[p]}>")
        elif p in PREFIX_MAP:
            declarations.append(f"PREFIX {p}: <{PREFIX_MAP[p]}>")
        else:
            unresolved.append(p)

    if unresolved:
        logger.warning(
            f"_get_prefix_declarations_for_query: could not resolve prefixes "
            f"{unresolved} for dataset={dataset!r}. Pass them via prefix_uris=."
        )

    return "\n".join(declarations)


@tool
def grep_data_values(
    class_uri: str,
    property_uri: str,
    keyword: str,
    dataset: str | None = None,
    limit: int = 50,
    prefix_uris: dict[str, str] | None = None,
) -> dict:
    """
    Search for data values using keyword/regex matching via SPARQL.

    Executes a SPARQL query with FILTER regex to find values matching the keyword.
    Use this to verify that a specific entity value (a person/place/product name)
    actually exists as data in the endpoint.

    Args:
        class_uri: Full prefixed class URI (as previously discovered by a class search)
        property_uri: Full prefixed property URI (as previously discovered by a property search)
        keyword: The keyword to search for (case-insensitive regex)
        dataset: Optional - specific dataset. If None, uses default endpoint.
        limit: Maximum number of values to retrieve
        prefix_uris: Optional dict mapping prefix names to full URIs for custom prefixes

    Returns:
        Dict with matching values and metadata

    Example (shape only — use URIs you discovered from the schema):
        grep_data_values("<prefix>:Author", "<prefix>:name", "Shakespeare")
    """
    from src.endpoints.ontop import OnTopEndpoint

    # Extract prefixes from URIs
    prefixes_used = set()
    for uri in [class_uri, property_uri]:
        if ":" in uri:
            prefixes_used.add(uri.split(":")[0])
    prefixes_used.add("xsd")  # Always include xsd for regex

    # Build prefixes
    prefix_str = _get_prefix_declarations_for_query(
        prefixes_used,
        dataset=dataset,
        custom_prefix_uris=prefix_uris,
    )

    # Build query with FILTER regex
    query = f"""
{prefix_str}

SELECT DISTINCT ?value WHERE {{
    ?subject a {class_uri} .
    ?subject {property_uri} ?value .
    FILTER(regex(str(?value), "{keyword}", "i"))
}}
LIMIT {limit}
""".strip()

    # Get endpoint for dataset
    endpoint_url = None
    if dataset:
        endpoint_url = get_endpoint_for_dataset(dataset)

    endpoint = OnTopEndpoint(endpoint_url=endpoint_url) if endpoint_url else OnTopEndpoint()

    # Handle both sync and async contexts
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context - run in thread pool to avoid loop conflict
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, endpoint.execute_query(query))
            result = future.result()
    except RuntimeError:
        # No running loop - safe to create one
        result = asyncio.run(endpoint.execute_query(query))

    if not result.success:
        return {
            "success": False,
            "error": result.error_message,
            "class_uri": class_uri,
            "property_uri": property_uri,
            "keyword": keyword,
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
        "class_uri": class_uri,
        "property_uri": property_uri,
        "keyword": keyword,
        "dataset": dataset,
        "sparql_endpoint": endpoint_url or "default",
        "count": len(values),
        "values": values,
    }


@tool
def search_data_values(
    class_uri: str,
    property_uri: str,
    keyword: str,
    dataset: str | None = None,
    limit: int = 50,
    prefix_uris: dict[str, str] | None = None,
) -> dict:
    """
    Semantically search for data values using embeddings.

    First fetches values for the class+property, then uses semantic similarity
    to find values matching the keyword. Returns SAME format as grep_data_values.

    Args:
        class_uri: Full prefixed class URI (as previously discovered by a class search)
        property_uri: Full prefixed property URI (as previously discovered by a property search)
        keyword: What to search for semantically (free-form — paraphrases allowed)
        dataset: Optional - specific dataset. If None, uses default endpoint.
        limit: Maximum number of values to retrieve
        prefix_uris: Optional dict mapping prefix names to full URIs for custom prefixes

    Returns:
        Dict with matching values (same format as grep_data_values)

    Example (shape only — use URIs you discovered from the schema):
        search_data_values("<prefix>:Author", "<prefix>:name", "English playwright")
    """
    from src.endpoints.ontop import OnTopEndpoint

    # Extract prefixes from URIs
    prefixes_used = set()
    for uri in [class_uri, property_uri]:
        if ":" in uri:
            prefixes_used.add(uri.split(":")[0])

    # Extract class and property names for indexing
    class_name = class_uri.split(":")[-1] if ":" in class_uri else class_uri
    property_name = property_uri.split(":")[-1] if ":" in property_uri else property_uri

    # Build query to fetch all values
    prefix_str = _get_prefix_declarations_for_query(
        prefixes_used,
        dataset=dataset,
        custom_prefix_uris=prefix_uris,
    )

    query = f"""
{prefix_str}

SELECT DISTINCT ?value WHERE {{
    ?subject a {class_uri} .
    ?subject {property_uri} ?value .
}}
LIMIT 500
""".strip()

    # Get endpoint for dataset
    endpoint_url = None
    if dataset:
        endpoint_url = get_endpoint_for_dataset(dataset)

    endpoint = OnTopEndpoint(endpoint_url=endpoint_url) if endpoint_url else OnTopEndpoint()

    # Handle both sync and async contexts
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context - run in thread pool to avoid loop conflict
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, endpoint.execute_query(query))
            result = future.result()
    except RuntimeError:
        # No running loop - safe to create one
        result = asyncio.run(endpoint.execute_query(query))

    if not result.success:
        return {
            "success": False,
            "error": result.error_message,
            "class_uri": class_uri,
            "property_uri": property_uri,
            "values": [],
        }

    # Extract values
    bindings = result.results.get("results", {}).get("bindings", [])
    all_values = []
    for binding in bindings:
        if "value" in binding:
            val_obj = binding["value"]
            val = val_obj.get("value", "") if isinstance(val_obj, dict) else str(val_obj)
            if val and val.strip():
                all_values.append(val.strip())

    if not all_values:
        return {
            "success": True,
            "class_uri": class_uri,
            "property_uri": property_uri,
            "keyword": keyword,
            "values": [],
            "note": "No values found for this class+property combination",
        }

    # Index and search using PropertyValueIndex
    # Use full URIs as keys to avoid collisions between different prefixes
    index = get_property_value_index()
    indexed_count = index.index_values(
        class_name=class_uri,
        property_name=property_uri,
        values=all_values,
    )

    if indexed_count == 0:
        return {
            "success": False,
            "error": "Failed to index values for semantic search",
            "class_uri": class_uri,
            "property_uri": property_uri,
            "values": [],
        }

    # Search using full URIs as keys (limit results)
    search_results = index.search(class_uri, property_uri, keyword, min(limit, 50))

    # Extract just the value strings (same format as grep_data_values)
    value_strings = [r["value"] if isinstance(r, dict) else str(r) for r in search_results]

    return {
        "success": True,
        "class_uri": class_uri,
        "property_uri": property_uri,
        "keyword": keyword,
        "dataset": dataset,
        "sparql_endpoint": endpoint_url or "default",
        "count": len(value_strings),
        "values": value_strings,
    }


# =============================================================================
# SPARQL GENERATION SUPPORT TOOLS
# =============================================================================


@tool
def sample_property_values(
    class_uri: str,
    property_uri: str,
    dataset: str | None = None,
    limit: int = 10,
    prefix_uris: dict[str, str] | None = None,
) -> dict:
    """
    Get sample values for a property to understand data format, units, and range.

    Use this BEFORE generating FILTER clauses to:
    - Check units (meters vs kilometers, EUR vs cents)
    - Check date formats (ISO vs custom)
    - Check enum values (exact strings used)
    - Check numeric ranges (min/max values)

    Args:
        class_uri: Full prefixed class URI (as previously discovered by a class search)
        property_uri: Full prefixed property URI (as previously discovered by a property search)
        dataset: Optional - specific dataset ID
        limit: Number of sample values to retrieve (default: 10)
        prefix_uris: Optional dict mapping prefix names to full URIs for custom prefixes

    Returns:
        Dict with sample values, detected datatype, and statistics (min/max for numerics)

    Example (shape only — use URIs you discovered from the schema):
        sample_property_values("<prefix>:Book", "<prefix>:publicationYear")
        → Shows actual year values like "1998", "2001", ...
    """
    from src.endpoints.ontop import OnTopEndpoint

    # Extract prefixes from URIs
    prefixes_used = set()
    for uri in [class_uri, property_uri]:
        if ":" in uri:
            prefixes_used.add(uri.split(":")[0])
    prefixes_used.add("xsd")  # Always include xsd for datatype

    # Build prefixes
    prefix_str = _get_prefix_declarations_for_query(
        prefixes_used,
        dataset=dataset,
        custom_prefix_uris=prefix_uris,
    )

    # Query to get sample values with their datatypes
    query = f"""
{prefix_str}

SELECT DISTINCT ?value (DATATYPE(?value) AS ?datatype) WHERE {{
    ?subject a {class_uri} .
    ?subject {property_uri} ?value .
}}
LIMIT {limit}
""".strip()

    # Get endpoint for dataset
    endpoint_url = None
    if dataset:
        endpoint_url = get_endpoint_for_dataset(dataset)

    endpoint = OnTopEndpoint(endpoint_url=endpoint_url) if endpoint_url else OnTopEndpoint()

    # Handle both sync and async contexts
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, endpoint.execute_query(query))
            result = future.result()
    except RuntimeError:
        result = asyncio.run(endpoint.execute_query(query))

    if not result.success:
        return {
            "success": False,
            "error": result.error_message,
            "class_uri": class_uri,
            "property_uri": property_uri,
            "samples": [],
        }

    # Extract values and analyze
    bindings = result.results.get("results", {}).get("bindings", [])
    samples = []
    datatypes = set()
    numeric_values = []

    for binding in bindings:
        if "value" in binding:
            val_obj = binding["value"]
            val = val_obj.get("value", "") if isinstance(val_obj, dict) else str(val_obj)
            datatype = None
            if "datatype" in binding:
                dt_obj = binding["datatype"]
                datatype = dt_obj.get("value", "") if isinstance(dt_obj, dict) else str(dt_obj)
                if datatype:
                    datatypes.add(datatype)

            if val:
                samples.append({"value": val, "datatype": datatype})
                # Try to parse as numeric for statistics
                try:
                    numeric_values.append(float(val))
                except (ValueError, TypeError):
                    pass

    # Build response with statistics
    response = {
        "success": True,
        "class_uri": class_uri,
        "property_uri": property_uri,
        "dataset": dataset,
        "sparql_endpoint": endpoint_url or "default",
        "sample_count": len(samples),
        "samples": samples,
        "detected_datatypes": list(datatypes),
    }

    # Add numeric statistics if applicable
    if numeric_values:
        response["numeric_stats"] = {
            "min": min(numeric_values),
            "max": max(numeric_values),
            "is_numeric": True,
        }
        # Hint about units based on range
        val_range = max(numeric_values) - min(numeric_values)
        if val_range > 0:
            response["numeric_stats"]["range"] = val_range
            response["numeric_stats"]["hint"] = (
                "Check if these values match expected units in your FILTER clause"
            )

    return response


@tool
def test_candidate_query(
    query: str,
    dataset: str,
    limit: int = 5,
    timeout_seconds: int = 30,
) -> dict:
    """
    Test a candidate SPARQL query before finalizing it.

    Use this tool to verify your query BEFORE outputting the final JSON:
    - Pre-validates against known OnTop anti-patterns
    - Check if syntax is correct
    - Check if query returns results (not empty)
    - Preview sample results to validate query logic
    - Provides detailed error feedback with optimization hints

    Args:
        query: Complete SPARQL query with PREFIX declarations
        dataset: Dataset ID to query (as provided in the task context) - REQUIRED
        limit: Maximum results to return (default: 5, max: 10)
        timeout_seconds: Query timeout in seconds (default: 30, min: 15, max: 120).
            ONLY increase beyond 30s if the Mapping Optimizer has already optimized
            the mappings and the generated SQL has no more self-join issues.
            Use this as a last resort when the query simply needs more execution time
            rather than optimization.

    Returns:
        Dict with:
        - success: Whether query executed successfully
        - syntax_valid: Whether query syntax is valid
        - result_count: Number of results returned
        - has_results: True if at least one result
        - sample_results: List of result rows (up to limit)
        - variables: Variable names in SELECT
        - error_message: Error details if failed
        - hints: Suggestions if query has issues
        - validation_feedback: Pre-execution validation warnings (if any)
        - ontop_error_analysis: Detailed error analysis for OnTop errors

    Example (shape only — use PREFIX/classes you discovered from the schema):
        test_candidate_query(
            query=\"\"\"
            PREFIX <prefix>: <http://example.org/library#>
            SELECT ?book WHERE {
                ?book a <prefix>:Book .
            }
            \"\"\",
            dataset=<dataset_id>,
            limit=5
        )
    """
    import concurrent.futures
    import time
    from src.endpoints.ontop import OnTopEndpoint
    from src.tools.sparql_tools import get_endpoint_for_dataset as get_endpoint
    from src.utils.query_validator import validate_query_with_feedback, get_feedback_for_ontop_error
    from src.utils.query_metadata import extract_query_metadata

    # Validate and enforce limit
    limit = min(max(1, limit), 10)

    # Validate and enforce timeout (15-120 seconds)
    timeout_seconds = min(max(15, timeout_seconds), 120)

    # Get endpoint for dataset
    endpoint_url = get_endpoint(dataset)
    if not endpoint_url:
        return {
            "success": False,
            "syntax_valid": True,
            "result_count": 0,
            "has_results": False,
            "sample_results": [],
            "variables": [],
            "error_message": f"Unknown dataset: {dataset}. Valid datasets: edu, trn, nrg, bsbm, etc.",
            "execution_time_ms": 0,
            "endpoint_used": None,
            "hints": {"empty_result": False, "suggestion": "Check dataset ID spelling."}
        }

    # === NEW: Pre-validation against known OnTop anti-patterns ===
    should_execute, validation_feedback = validate_query_with_feedback(query)
    if not should_execute:
        # Query has ERROR-level anti-pattern - don't even try to execute
        return {
            "success": False,
            "syntax_valid": True,  # It's not a syntax error, it's an OnTop limitation
            "result_count": 0,
            "has_results": False,
            "sample_results": [],
            "variables": [],
            "error_message": "Query blocked by pre-validation (known OnTop limitation).",
            "execution_time_ms": 0,
            "endpoint_used": endpoint_url,
            "validation_feedback": validation_feedback,
            "hints": {
                "empty_result": False,
                "suggestion": "Fix the issue described in validation_feedback before testing again.",
                "blocked_by_validator": True
            }
        }

    endpoint = OnTopEndpoint(endpoint_url=endpoint_url)

    # Ensure query has appropriate LIMIT for testing
    test_query = _ensure_query_limit(query, limit)

    # Execute query with timeout
    start_time = time.time()
    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, endpoint.execute_query(test_query))
            result = future.result(timeout=float(timeout_seconds))
    except RuntimeError:
        result = asyncio.run(endpoint.execute_query(test_query))
    except concurrent.futures.TimeoutError:
        # === Extract metadata for timeout errors ===
        timeout_error_msg = f"Query timed out after {timeout_seconds} seconds."
        try:
            metadata = extract_query_metadata(
                query=test_query,
                error_message=timeout_error_msg,
                endpoint_url=endpoint_url,
                fetch_counts=True,
            )
            metadata_feedback = metadata.to_feedback_string()
        except Exception:
            metadata_feedback = ""

        # === Timeout feedback - includes hint about timeout parameter ===
        if timeout_seconds < 120:
            timeout_hint = f"""
If the Mapping Optimizer has already optimized the mappings and the generated SQL
has no self-join issues, you can increase timeout_seconds (up to 120s) as a last resort.
Current timeout: {timeout_seconds}s."""
        else:
            timeout_hint = "\nMaximum timeout (120s) already used. Query optimization is required."

        timeout_with_instruction = f"""TIMEOUT ERROR: Query timed out after {timeout_seconds} seconds.

Try ONE optimization:
- Remove OPTIONAL blocks
- Simplify JOINs
- Consider delegating to Mapping Optimizer (for self-join issues) or Multi-Step Agent (for complex queries)
{timeout_hint}

Note: After 3 timeouts, the system will automatically delegate to Multi-Step Agent."""

        return {
            "success": False,
            "syntax_valid": True,  # Timeout doesn't mean syntax error
            "result_count": 0,
            "has_results": False,
            "sample_results": [],
            "variables": [],
            "error_message": timeout_with_instruction,
            "execution_time_ms": float(timeout_seconds * 1000),
            "endpoint_used": endpoint_url or "default",
            "timeout_seconds_used": timeout_seconds,
            "hints": {
                "empty_result": False,
                "suggestion": "Try simplifying the query. Auto-delegation after 3 timeouts.",
            }
        }

    execution_time_ms = (time.time() - start_time) * 1000

    # Handle execution failure
    if not result.success:
        error_msg = result.error_message or "Unknown error"
        is_syntax_error = "syntax" in error_msg.lower() or result.error_type == "syntax_error"
        is_timeout = result.error_type == "query_timeout" or "timeout" in error_msg.lower()

        # === Detailed error analysis for OnTop errors ===
        ontop_analysis = get_feedback_for_ontop_error(error_msg, test_query)

        # Extract metadata for non-syntax errors
        metadata_feedback = ""
        if not is_syntax_error:
            try:
                metadata = extract_query_metadata(
                    query=test_query,
                    error_message=error_msg,
                    endpoint_url=endpoint_url,
                    fetch_counts=True,
                )
                metadata_feedback = metadata.to_feedback_string()
            except Exception:
                pass

        # === For timeout errors, provide optimization hints ===
        if is_timeout:
            error_msg = f"""TIMEOUT ERROR: {error_msg}

Try ONE optimization:
- Remove OPTIONAL blocks
- Simplify JOINs
- Consider delegating to Mapping Optimizer (for self-join issues) or Multi-Step Agent (for complex queries)

Note: After 3 timeouts, the system will automatically delegate to Multi-Step Agent."""

        response = {
            "success": False,
            "syntax_valid": not is_syntax_error,
            "result_count": 0,
            "has_results": False,
            "sample_results": [],
            "variables": [],
            "error_message": error_msg,
            "execution_time_ms": execution_time_ms,
            "endpoint_used": endpoint_url or "default",
            "validation_feedback": validation_feedback,  # Include any warnings
            "ontop_error_analysis": ontop_analysis,
            "query_metadata": metadata_feedback,
            "hints": {
                "empty_result": False,
                "suggestion": "Try simplifying the query. Auto-delegation after 3 timeouts."
                if is_timeout else (
                    "See ontop_error_analysis for specific fix suggestions."
                    if ontop_analysis else (
                        "Check query syntax, PREFIX declarations, and class/property names."
                        if is_syntax_error else "Check endpoint availability and query constraints."
                    )
                )
            }
        }

        # Emit detailed tracing event for failed SPARQL query test
        tracer = _get_tracer()
        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.SPARQL_QUERY_TEST,
                phase="generation",
                agent="sparql_agent",
                data={
                    "query": test_query,
                    "dataset": dataset,
                    "endpoint": endpoint_url,
                    "success": False,
                    "is_timeout": is_timeout,
                    "is_syntax_error": is_syntax_error,
                    "error_message": error_msg[:500] if error_msg else None,
                    "execution_time_ms": execution_time_ms,
                },
            )
        )

        return response

    # Process successful results
    bindings = result.results.get("results", {}).get("bindings", [])
    variables = result.results.get("head", {}).get("vars", [])

    # Format sample results (truncate long values)
    sample_results = []
    for binding in bindings[:limit]:
        row = {}
        for var in variables:
            if var in binding:
                val_obj = binding[var]
                val = val_obj.get("value", str(val_obj)) if isinstance(val_obj, dict) else str(val_obj)
                # Truncate long values
                row[var] = val[:200] + "..." if len(val) > 200 else val
            else:
                row[var] = None
        sample_results.append(row)

    has_results = len(bindings) > 0
    hints = {
        "empty_result": not has_results,
        "suggestion": None
    }
    if not has_results:
        hints["suggestion"] = (
            "Query returned no results. Verify class/property names match the schema, "
            "check FILTER conditions, and use sample_property_values to inspect actual data values."
        )

    response = {
        "success": True,
        "syntax_valid": True,
        "result_count": len(bindings),
        "has_results": has_results,
        "sample_results": sample_results,
        "variables": variables,
        "error_message": None,
        "execution_time_ms": execution_time_ms,
        "endpoint_used": endpoint_url or "default",
        "hints": hints
    }

    # Include validation warnings if any (query succeeded but had warnings)
    if validation_feedback:
        response["validation_feedback"] = validation_feedback
        response["hints"]["validation_warning"] = True

    # Emit detailed tracing event for SPARQL query test
    tracer = _get_tracer()
    tracer.emit(
        TraceEvent(
            event_type=TraceEventType.SPARQL_QUERY_TEST,
            phase="generation",
            agent="sparql_agent",
            data={
                "query": test_query,
                "dataset": dataset,
                "endpoint": endpoint_url,
                "success": True,
                "result_count": len(bindings),
                "variables": variables,
                "sample_results": sample_results[:10],  # Show up to 10 sample rows
                "execution_time_ms": execution_time_ms,
            },
        )
    )

    return response


def _ensure_query_limit(query: str, limit: int) -> str:
    """Ensure query has appropriate LIMIT for testing."""
    # Find and replace existing LIMIT or add new one
    limit_pattern = re.compile(r'LIMIT\s+\d+', re.IGNORECASE)
    if limit_pattern.search(query):
        return limit_pattern.sub(f'LIMIT {limit}', query)
    return query.strip() + f'\nLIMIT {limit}'


# =============================================================================
# MULTI-STEP DELEGATION TOOL
# =============================================================================

def _sanitize_for_json(obj):
    """Konvertiere alle Werte zu JSON-serialisierbaren Typen.

    Handles rdflib URIRef/Literal, datetime, and other non-serializable objects
    by converting them to strings.
    """
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        # Convert rdflib URIRef, Literal, datetime, etc. to string
        return str(obj)


# Timeout for delegation tools (increased to 500s for Mapping Optimizer with multiple template changes)
DELEGATION_TIMEOUT_SECONDS = 500

# Apply nest_asyncio to allow nested event loops
# This is needed when calling async agents from sync tools within an async context
try:
    import nest_asyncio
    nest_asyncio.apply()
    _NEST_ASYNCIO_APPLIED = True
    logger.info("nest_asyncio applied - nested event loops enabled")
except ImportError:
    _NEST_ASYNCIO_APPLIED = False
    logger.warning("nest_asyncio not available - falling back to thread-based execution")


def _run_async_agent(coro_func, llm_model: str, timeout: int = DELEGATION_TIMEOUT_SECONDS):
    """
    Run an async agent from a sync context, handling event loop issues.

    This helper properly handles the case where we're called from within
    an already-running event loop (e.g., from the dashboard).

    Uses nest_asyncio to allow nested event loops (preferred) or falls back
    to thread-based execution if nest_asyncio is not available.

    Args:
        coro_func: A callable that takes llm_model and returns a coroutine
        llm_model: The LLM model to use (passed explicitly to avoid ContextVar issues)
        timeout: Timeout in seconds

    Returns:
        The result from the coroutine

    Raises:
        Exception: If the coroutine fails or times out
    """
    import concurrent.futures

    async def _run_with_timeout():
        """Run the coroutine with a timeout."""
        return await asyncio.wait_for(coro_func(llm_model), timeout=timeout)

    # Check if we're in an async context
    try:
        loop = asyncio.get_running_loop()
        is_async_context = True
    except RuntimeError:
        is_async_context = False

    if is_async_context and _NEST_ASYNCIO_APPLIED:
        # Use nest_asyncio - allows nested asyncio.run() calls
        logger.info(f"Running async agent with nest_asyncio (timeout={timeout}s, model={llm_model})")
        try:
            return asyncio.run(_run_with_timeout())
        except asyncio.TimeoutError:
            logger.error(f"Async agent timed out after {timeout}s")
            raise TimeoutError(f"Agent execution timed out after {timeout} seconds")

    elif is_async_context:
        # Fallback: run in a thread (less reliable for HTTP clients)
        logger.warning(f"Running async agent in thread - nest_asyncio not available (timeout={timeout}s)")

        def _thread_target():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(_run_with_timeout())
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_thread_target)
            try:
                return future.result(timeout=timeout + 10)  # Extra buffer for thread overhead
            except concurrent.futures.TimeoutError:
                logger.error(f"Async agent thread timed out after {timeout}s")
                raise TimeoutError(f"Agent execution timed out after {timeout} seconds")
    else:
        # Not in async context - can run directly
        logger.info(f"Running async agent directly (model={llm_model})")
        try:
            return asyncio.run(_run_with_timeout())
        except asyncio.TimeoutError:
            logger.error(f"Async agent timed out after {timeout}s")
            raise TimeoutError(f"Agent execution timed out after {timeout} seconds")


@tool
def delegate_to_multi_step_agent(
    failed_query: str,
    user_query: str,
    schema_context: str,
    error_message: str,
    query_history: str,
    dataset: str,
) -> str:
    """
    Delegate a complex query to the Multi-Step Agent for decomposition.

    Use this tool when:
    - A query repeatedly times out (OnTop cannot process the JOINs)
    - The query has chain patterns like A→B→C→D
    - You've tried optimizing but the query is fundamentally too complex

    The Multi-Step Agent will:
    1. Analyze the query to identify bottleneck patterns
    2. Decompose the query into simpler step queries
    3. Test each step individually
    4. Write a Python transformation script to combine results
    5. Return the aggregated final results

    Args:
        failed_query: The SPARQL query that timed out
        user_query: The original natural language question
        schema_context: The schema triples being used (as string)
        error_message: The timeout/error message received
        query_history: JSON array of all queries tested by SPARQL Agent
                       [{"query": str, "status": "success"|"timeout"|"error", "count": int, "error_msg": str|null}]
        dataset: The dataset ID (edu, trn, nrg, bsbm)

    Returns:
        JSON with:
        - success: True/False
        - steps_count: Number of decomposed steps
        - final_results: The aggregated results (if successful)
        - transform_script: The Python aggregation code
        - error: Error message (if failed)
    """
    import asyncio
    import json

    # Lazy import to avoid circular dependency
    from src.agents.generation.multi_step_agent import MultiStepAgent
    from src.tracing import TraceEvent, TraceEventType, get_tracer

    # Emit delegation start event
    tracer = get_tracer()
    tracer.emit(
        TraceEvent(
            event_type=TraceEventType.DELEGATION_START,
            phase="generation",
            data={
                "from_agent": "sparql_generation",
                "to_agent": "multi_step",
                "reason": f"Delegating complex query to Multi-Step Agent",
            },
        )
    )

    # Capture LLM model BEFORE spawning thread (ContextVar doesn't propagate to threads)
    from src.config import get_llm_model
    current_llm_model = get_llm_model()

    async def _run_agent(llm_model: str):
        logger.info(f"delegate_to_multi_step_agent: Using LLM model: {llm_model}")
        agent = MultiStepAgent(llm_model=llm_model)
        result = await agent.decompose_and_transform(
            failed_query=failed_query,
            user_query=user_query,
            schema_context=schema_context,
            error_message=error_message,
            query_history=query_history,
            dataset=dataset,
        )
        return result.to_dict()

    # Run the async agent using the helper
    try:
        result = _run_async_agent(_run_agent, current_llm_model)

        logger.info(f"delegate_to_multi_step_agent: Result keys: {list(result.keys())}")
        logger.info(f"delegate_to_multi_step_agent: steps count: {len(result.get('steps', []))}")
        logger.info(f"delegate_to_multi_step_agent: final_results count: {len(result.get('final_results', []))}")

        # Remove large 'results' arrays from steps to avoid JSON serialization issues
        # Keep only metadata (name, query, dataset, result_count) for display
        if 'steps' in result:
            clean_steps = []
            for step in result['steps']:
                clean_step = {
                    'name': step.get('name', ''),
                    'query': step.get('query', ''),
                    'dataset': step.get('dataset', ''),
                    'result_count': step.get('result_count', len(step.get('results', []))),
                }
                clean_steps.append(clean_step)
            result['steps'] = clean_steps

        # Sanitize result to ensure JSON serialization works
        # (converts rdflib URIRef/Literal, datetime, etc. to strings)
        sanitized_result = _sanitize_for_json(result)

        # Emit delegation end event
        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.DELEGATION_END,
                phase="generation",
                data={
                    "from_agent": "multi_step",
                    "to_agent": "sparql_generation",
                    "reason": f"Multi-Step completed: success={result.get('success', False)}",
                },
            )
        )
        return json.dumps(sanitized_result, indent=2)

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"delegate_to_multi_step_agent FAILED: {e}\n{error_trace}")

        # Emit delegation end event (with error)
        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.DELEGATION_END,
                phase="generation",
                data={
                    "from_agent": "multi_step",
                    "to_agent": "sparql_generation",
                    "reason": f"Multi-Step failed: {str(e)}",
                },
            )
        )
        return json.dumps({
            "success": False,
            "error": f"Multi-Step Agent failed: {str(e)}",
        })


@tool
def delegate_to_mapping_optimizer(
    sparql_query: str,
    dataset: str,
    sparql_agent_reasoning: str,
    generated_sql: str = "",
    previous_attempts: str = "[]",
    feedback_on_failure: str | None = None,
) -> str:
    """
    Delegate to the Mapping Optimizer Agent to analyze and fix mapping issues.

    Use this tool when a query times out and you suspect the problem is in the
    R2RML mapping (self-joins, inefficient IRI templates, etc.).

    The Mapping Optimizer Agent will:
    1. ANALYZE the generated SQL for self-joins and mapping issues
    2. DECIDE if mapping optimization can help (returns can_help=false if not)
    3. If it can help: optimize IRI templates to use primary keys

    After the agent returns, this tool automatically restarts OnTop if changes
    were made (single restart after all changes, not per-change).

    Args:
        sparql_query: The SPARQL query that timed out
        dataset: Dataset ID (e.g., "trn", "edu", "nrg")
        sparql_agent_reasoning: YOUR analysis of why mapping optimization might help
        generated_sql: SQL from OnTop /reformulate endpoint (optional - agent can fetch)
        previous_attempts: JSON array of previous optimization attempts (optional)
        feedback_on_failure: What went wrong in previous attempt (optional)

    Returns:
        JSON with:
        - success: True/False (did the agent complete successfully)
        - can_help: True/False (can mapping optimization fix this issue)
        - analysis: Agent's SQL analysis (especially if can_help=false)
        - optimized_tables: List of tables that were optimized
        - changes_made: Details of template changes
        - reasoning: Explanation from the Mapping Optimizer Agent
        - error: Error message (if failed)

    If can_help=false, consider using delegate_to_multi_step_agent instead.
    """
    import asyncio
    import json

    # Lazy import to avoid circular dependency
    from src.agents.generation.mapping_optimizer_agent import (
        MappingOptimizerAgent,
        MappingOptimizationRequest,
    )
    from src.tools.sparql_tools import get_endpoint_for_dataset
    from src.config import MAPPINGS_DIR_ROOT, get_dataset_from_endpoint

    # Parse previous attempts
    try:
        prev_attempts = json.loads(previous_attempts) if previous_attempts else []
    except json.JSONDecodeError:
        prev_attempts = []

    # Get endpoint and derive the actual dataset name (e.g., trn -> trn-large)
    endpoint_url = get_endpoint_for_dataset(dataset) or ""
    # Use inverse lookup to get the actual dataset name (includes -large/-small suffix)
    actual_dataset = get_dataset_from_endpoint(endpoint_url) or dataset

    # Determine mapping path - prefer actual_dataset dir, fallback to base dataset dir
    mapping_path_actual = MAPPINGS_DIR_ROOT / actual_dataset.lower() / "mapping.ttl"
    mapping_path_base = MAPPINGS_DIR_ROOT / dataset.lower() / "mapping.ttl"
    mapping_path = str(mapping_path_actual if mapping_path_actual.exists() else mapping_path_base)

    # If no SQL provided, try to get it via reformulate endpoint
    sql_to_use = generated_sql
    if not sql_to_use and endpoint_url:
        try:
            import httpx
            reformulate_url = endpoint_url.replace("/sparql", "/ontop/reformulate")
            with httpx.Client(timeout=30.0) as client:
                response = client.get(reformulate_url, params={"query": sparql_query})
                if response.status_code == 200:
                    sql_to_use = response.text
        except Exception as e:
            logger.warning(f"Could not fetch SQL via reformulate: {e}")
            sql_to_use = "(SQL not available - agent should analyze query structure)"

    # Build request - agent analyzes SQL itself
    # Use actual_dataset (e.g., "trn-large") instead of base dataset (e.g., "trn")
    # This ensures the agent can find the correct database
    request = MappingOptimizationRequest(
        sparql_query=sparql_query,
        generated_sql=sql_to_use,
        dataset=actual_dataset,
        endpoint_url=endpoint_url,
        mapping_path=mapping_path,
        sparql_agent_reasoning=sparql_agent_reasoning,
        previous_attempts=prev_attempts,
        feedback_on_failure=feedback_on_failure,
    )

    # Emit delegation start event
    from src.tracing import TraceEvent, TraceEventType, get_tracer
    tracer = get_tracer()
    tracer.emit(
        TraceEvent(
            event_type=TraceEventType.DELEGATION_START,
            phase="generation",
            data={
                "from_agent": "sparql_generation",
                "to_agent": "mapping_optimizer",
                "reason": f"Delegating to Mapping Optimizer for {dataset}",
            },
        )
    )

    # Capture LLM model BEFORE spawning thread (ContextVar doesn't propagate to threads)
    from src.config import get_llm_model
    current_llm_model = get_llm_model()

    async def _run_agent(llm_model: str):
        logger.info(f"delegate_to_mapping_optimizer: Using LLM model: {llm_model}")
        agent = MappingOptimizerAgent(llm_model=llm_model)
        result = await agent.optimize(request)
        return result.to_dict()

    # Run the async agent using the helper
    try:
        result = _run_async_agent(_run_agent, current_llm_model)

        logger.info(f"delegate_to_mapping_optimizer: success={result.get('success')}, can_help={result.get('can_help')}")
        logger.info(f"delegate_to_mapping_optimizer: optimized_tables={result.get('optimized_tables')}")

        # Safety check: if MO made edits but failed (can_help=False OR success=False),
        # restore immediately and restart the container
        # (--dev mode auto-restarts OnTop on file change, which can crash if mapping is broken)
        if not result.get('can_help', False) or not result.get('success', False):
            from src.tools.mapping_optimizer_tools import get_mapping_context
            mapping_context = get_mapping_context()
            if mapping_context.modified:
                reason = "can_help=False" if not result.get('can_help', False) else "success=False"
                logger.warning(f"MO returned {reason} but mapping was modified - restoring original mapping")
                mapping_context.restore_all()
                result['mappings_restored'] = True
                # Restart container to recover from broken mapping state
                try:
                    from src.tools.mapping_optimizer_tools import restart_ontop_container
                    restart_result_json = restart_ontop_container.invoke({
                        "dataset": actual_dataset,
                        "wait_for_health": True,
                    })
                    import json as _json
                    restart_data = _json.loads(restart_result_json)
                    logger.info(f"Container restarted after MO failure: healthy={restart_data.get('healthy')}")
                except Exception as e:
                    logger.error(f"Container restart after MO failure failed: {e}")

        # Save a snapshot of the optimized mapping for post-hoc analysis
        if result.get('success', False) and result.get('can_help', False):
            try:
                from src.tools.mapping_optimizer_tools import _get_mapping_path
                mapping_path = _get_mapping_path(actual_dataset)
                if mapping_path and mapping_path.exists():
                    snapshot_path = mapping_path.parent / "mapping_optimized_snapshot.ttl"
                    import shutil
                    shutil.copy2(str(mapping_path), str(snapshot_path))
                    logger.info(f"Saved optimized mapping snapshot to {snapshot_path}")
            except Exception as e:
                logger.warning(f"Could not save mapping snapshot: {e}")

            # Log the changes made
            for change in result.get('changes', []):
                logger.info(f"MO change: table={change.get('table')}, "
                            f"old_template={change.get('old_template', '')[:100]}, "
                            f"new_template={change.get('new_template', '')[:100]}, "
                            f"reasoning={change.get('reasoning', '')[:100]}")

        # Restart OnTop if mapping was modified (single restart after all changes)
        if result.get('requires_restart', False):
            logger.info(f"delegate_to_mapping_optimizer: Restarting OnTop for {actual_dataset} after mapping changes")
            try:
                from src.tools.mapping_optimizer_tools import (
                    restart_ontop_container,
                    get_mapping_context,
                )
                restart_result = restart_ontop_container.invoke({
                    "dataset": actual_dataset,
                    "wait_for_health": True,
                })
                restart_data = json.loads(restart_result)
                result['ontop_restarted'] = restart_data.get('healthy', False)

                if not result['ontop_restarted']:
                    # Check for mapping errors
                    error_type = restart_data.get('error_type')
                    error_message = restart_data.get('error_message', 'OnTop did not become healthy')

                    if error_type == 'mapping_error':
                        # Mapping error detected - restore original mappings and report error
                        logger.error(f"Mapping error detected: {error_message[:300]}")

                        # Restore original mappings
                        mapping_context = get_mapping_context()
                        if mapping_context.modified:
                            logger.info("Restoring original mappings due to mapping error...")
                            mapping_context.restore_all()
                            result['mappings_restored'] = True

                        # Include detailed error for the SPARQL agent to potentially retry
                        result['mapping_error'] = True
                        result['restart_error'] = error_message
                        result['error_type'] = error_type
                        result['can_help'] = False  # Override - optimization failed
                        result['success'] = False
                        result['reasoning'] = (
                            f"Mapping optimization failed: The modified mappings caused OnTop to crash. "
                            f"Error: {error_message[:500]}. "
                            f"Original mappings have been restored. "
                            f"This typically happens when IRI templates reference columns not available in the source query. "
                            f"Consider using delegate_to_multi_step_agent instead."
                        )
                    else:
                        result['restart_error'] = error_message
                        logger.warning(f"OnTop restart issue: {error_message}")
                else:
                    logger.info(f"delegate_to_mapping_optimizer: OnTop restarted successfully")
            except Exception as e:
                result['ontop_restarted'] = False
                result['restart_error'] = str(e)
                logger.error(f"delegate_to_mapping_optimizer: OnTop restart failed: {e}")

        # Emit delegation end event
        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.DELEGATION_END,
                phase="generation",
                data={
                    "from_agent": "mapping_optimizer",
                    "to_agent": "sparql_generation",
                    "reason": f"Mapping Optimizer completed: can_help={result.get('can_help', False)}",
                },
            )
        )
        return json.dumps(result, indent=2)

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"delegate_to_mapping_optimizer FAILED: {e}\n{error_trace}")

        # Build a useful error message even if str(e) is empty
        error_msg = str(e) if str(e) else f"{type(e).__name__}"
        # Include last line of traceback for context
        tb_lines = error_trace.strip().split('\n')
        if len(tb_lines) > 1:
            error_msg += f" | {tb_lines[-1]}"

        # Emit delegation end event (with error)
        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.DELEGATION_END,
                phase="generation",
                data={
                    "from_agent": "mapping_optimizer",
                    "to_agent": "sparql_generation",
                    "reason": f"Mapping Optimizer failed: {error_msg}",
                },
            )
        )
        return json.dumps({
            "success": False,
            "error": f"Mapping Optimizer Agent failed: {error_msg}",
        })


# Tools for SPARQL generation agent
# Agent generates query, receives execution feedback from system, then decides next action
SPARQL_GENERATION_TOOLS = [
    sample_property_values,  # Check data values before writing FILTERs
    # delegate_to_mapping_optimizer,  # DISABLED: Mapping Optimizer removed from evaluated pipeline
    # delegate_to_multi_step_agent,   # DISABLED: Multi-Step Agent temporarily removed
]


# =============================================================================
# EXPORTS
# =============================================================================

GREP_TOOLS = [
    grep_classes,
    grep_properties,  # Re-added for fair comparison with search_properties
    read_semantic_model,
    grep_data_values,
    find_shortest_path,
    get_class_hierarchy,
    # list_all_classes removed: enumerates the whole schema and undermines
    # the retrieval comparison (especially on SYN queries where grep fails).
]

SEMANTIC_TOOLS = [
    search_classes,
    search_properties,  # Re-added: allows direct property search
    read_semantic_model,
    search_data_values,
    find_shortest_path,
    get_class_hierarchy,
    # list_all_classes removed: see GREP_TOOLS comment above.
]

# Legacy tools (kept for backwards compatibility)
LEGACY_GREP_TOOLS = [
    grep_schema,
    get_class_schema,
    list_all_classes,
    grep_properties,  # Moved to legacy
]

LEGACY_SEMANTIC_TOOLS = [
    semantic_schema_search,
    search_property_values,
    search_properties,  # Moved to legacy
]

__all__ = [
    # Core Grep tools
    "grep_classes",
    "grep_data_values",
    "read_semantic_model",
    "find_shortest_path",
    # Core Semantic tools
    "search_classes",
    "search_data_values",
    # SPARQL generation support tools
    "sample_property_values",
    "test_candidate_query",
    # Delegation tools
    "delegate_to_multi_step_agent",
    "delegate_to_mapping_optimizer",
    # Legacy tools (backwards compatibility)
    "grep_schema",
    "grep_properties",
    "get_class_schema",
    "list_all_classes",
    "semantic_schema_search",
    "search_property_values",
    "search_properties",
    # Index classes
    "PropertyValueIndex",
    "get_property_value_index",
    "SchemaIndex",
    "get_schema_index",
    # Tool lists
    "GREP_TOOLS",
    "SEMANTIC_TOOLS",
    "SPARQL_GENERATION_TOOLS",
    "LEGACY_GREP_TOOLS",
    "LEGACY_SEMANTIC_TOOLS",
]
