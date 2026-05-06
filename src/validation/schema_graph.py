"""Schema graph loading and management for semantic SPARQL validation.

This module loads schema information from class semantic models (TTL) and
OWL ontologies to build a combined schema graph for validation.
"""

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

from src.config import (
    AVAILABLE_DATASETS,
    get_ontology_path,
    get_semantic_models_path,
)

logger = logging.getLogger(__name__)

# Common namespace prefixes used across datasets
COMMON_PREFIXES: dict[str, str] = {
    "rdf": str(RDF),
    "rdfs": str(RDFS),
    "owl": str(OWL),
    "xsd": str(XSD),
    # Dataset-specific prefixes
    "ub": "http://example.org/ontology/education#",
    "trn": "http://example.org/ontology/transport#",
    "npdv": "http://example.org/ontology/energy#",
    "bsbm": "http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/",  # Corrected namespace
    "genex": "http://purl.org/genex#",
    "orth": "http://purl.org/net/orth#",
    "oriont": "https://orienting.eu/oriont#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "dc": "http://purl.org/dc/elements/1.1/",  # Dublin Core Elements
    "dct": "http://purl.org/dc/terms/",        # Dublin Core Terms
    "dcterms": "http://purl.org/dc/terms/",
    "schema1": "http://schema.org/",
    "geo1": "http://www.w3.org/2003/01/geo/wgs84_pos#",
    "rev": "http://purl.org/stuff/rev#",       # Review vocabulary
}


@dataclass
class SchemaGraph:
    """Combined schema information for semantic validation.

    Attributes:
        classes: All known class URIs in the schema
        properties: All known property URIs in the schema
        datatype_properties: Properties with literal/XSD ranges
        object_properties: Properties with class ranges
        class_properties: Mapping of class -> set of properties it has
        property_domains: Mapping of property -> set of valid domain classes
        property_ranges: Mapping of property -> range (class URI or XSD type)
        class_property_ranges: Mapping of class -> property -> set of range URIs
        subclass_of: Mapping of class -> set of parent classes
        prefixes: Mapping of prefix -> namespace URI
        dataset_id: The dataset this schema was loaded from
    """

    classes: set[str] = field(default_factory=set)
    properties: set[str] = field(default_factory=set)
    datatype_properties: set[str] = field(default_factory=set)
    object_properties: set[str] = field(default_factory=set)
    class_properties: dict[str, set[str]] = field(default_factory=dict)
    property_domains: dict[str, set[str]] = field(default_factory=dict)
    property_ranges: dict[str, str] = field(default_factory=dict)
    class_property_ranges: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    subclass_of: dict[str, set[str]] = field(default_factory=dict)
    prefixes: dict[str, str] = field(default_factory=dict)
    dataset_id: str = ""

    def get_class_with_parents(self, class_uri: str) -> set[str]:
        """Get a class and all its parent classes (transitive)."""
        result = {class_uri}
        to_process = [class_uri]

        while to_process:
            current = to_process.pop()
            parents = self.subclass_of.get(current, set())
            for parent in parents:
                if parent not in result:
                    result.add(parent)
                    to_process.append(parent)

        return result

    def get_all_properties_for_class(self, class_uri: str) -> set[str]:
        """Get all properties available for a class (including inherited)."""
        result = set()
        all_classes = self.get_class_with_parents(class_uri)

        for cls in all_classes:
            result.update(self.class_properties.get(cls, set()))

        return result

    def property_exists(self, property_uri: str) -> bool:
        """Check if a property exists in the schema."""
        return property_uri in self.properties

    def class_exists(self, class_uri: str) -> bool:
        """Check if a class exists in the schema."""
        return class_uri in self.classes

    def can_class_have_property(self, class_uri: str, property_uri: str) -> bool:
        """Check if a class can have a given property."""
        all_props = self.get_all_properties_for_class(class_uri)
        return property_uri in all_props

    def get_property_range(self, property_uri: str) -> Optional[str]:
        """Get the range of a property."""
        return self.property_ranges.get(property_uri)

    def get_all_descendants(self, class_uri: str) -> set[str]:
        """Get all descendant classes (transitive) via inverse subclass_of.

        Returns all C' such that C' rdfs:subClassOf* class_uri.
        Does NOT include class_uri itself.
        """
        children_of: dict[str, set[str]] = {}
        for child, parents in self.subclass_of.items():
            for parent in parents:
                children_of.setdefault(parent, set()).add(child)

        result: set[str] = set()
        frontier = [class_uri]
        while frontier:
            current = frontier.pop()
            for child in children_of.get(current, set()):
                if child not in result:
                    result.add(child)
                    frontier.append(child)
        return result


def _parse_semantic_model_file(file_path: Path, schema: SchemaGraph) -> None:
    """Parse a single class semantic model TTL file.

    Format: ClassName property Range .
    Example: trn:Trip trn:route trn:Route .
    """
    try:
        g = Graph()
        g.parse(file_path, format="turtle")

        # Extract prefixes
        for prefix, namespace in g.namespaces():
            if prefix:
                schema.prefixes[prefix] = str(namespace)

        # Extract triples - format is: Class property Range
        for subj, pred, obj in g:
            subj_str = str(subj)
            pred_str = str(pred)
            obj_str = str(obj)

            # Subject is the class
            schema.classes.add(subj_str)

            # Predicate is a property
            schema.properties.add(pred_str)

            # Add to class_properties
            if subj_str not in schema.class_properties:
                schema.class_properties[subj_str] = set()
            schema.class_properties[subj_str].add(pred_str)

            # Add to property_domains
            if pred_str not in schema.property_domains:
                schema.property_domains[pred_str] = set()
            schema.property_domains[pred_str].add(subj_str)

            # Object is the range (global last-writer-wins + per-class tracking)
            schema.property_ranges[pred_str] = obj_str
            schema.class_property_ranges.setdefault(subj_str, {}).setdefault(
                pred_str, set()
            ).add(obj_str)

            # Classify property type based on range
            if obj_str.startswith(str(XSD)) or obj_str.startswith("http://www.w3.org/2001/XMLSchema#"):
                schema.datatype_properties.add(pred_str)
            else:
                schema.object_properties.add(pred_str)
                # Object is also a class
                schema.classes.add(obj_str)

    except Exception as e:
        logger.warning(f"Failed to parse semantic model {file_path}: {e}")


def _is_valid_uri(uri_str: str) -> bool:
    """Check if a string is a valid URI (not a blank node)."""
    return uri_str.startswith("http://") or uri_str.startswith("https://")


def _parse_ontology_file(file_path: Path, schema: SchemaGraph) -> None:
    """Parse an OWL ontology file to extract class hierarchy and property types."""
    if not file_path.exists():
        logger.debug(f"Ontology file not found: {file_path}")
        return

    try:
        g = Graph()
        g.parse(file_path, format="xml")

        # Extract classes (filter out blank nodes)
        for subj in g.subjects(RDF.type, OWL.Class):
            subj_str = str(subj)
            if _is_valid_uri(subj_str):
                schema.classes.add(subj_str)

        # Extract subclass relationships (filter out blank nodes)
        for subj, _, obj in g.triples((None, RDFS.subClassOf, None)):
            subj_str = str(subj)
            obj_str = str(obj)
            if _is_valid_uri(subj_str) and _is_valid_uri(obj_str):
                if subj_str not in schema.subclass_of:
                    schema.subclass_of[subj_str] = set()
                schema.subclass_of[subj_str].add(obj_str)

        # Extract datatype properties (filter out blank nodes)
        for subj in g.subjects(RDF.type, OWL.DatatypeProperty):
            prop_str = str(subj)
            if _is_valid_uri(prop_str):
                schema.properties.add(prop_str)
                schema.datatype_properties.add(prop_str)

        # Extract object properties (filter out blank nodes)
        for subj in g.subjects(RDF.type, OWL.ObjectProperty):
            prop_str = str(subj)
            if _is_valid_uri(prop_str):
                schema.properties.add(prop_str)
                schema.object_properties.add(prop_str)

        # Extract domain/range constraints (if any) - only for valid URIs
        for prop in list(schema.properties):
            if not _is_valid_uri(prop):
                continue
            prop_uri = URIRef(prop)

            # Domain
            for _, _, domain in g.triples((prop_uri, RDFS.domain, None)):
                domain_str = str(domain)
                if _is_valid_uri(domain_str):
                    if prop not in schema.property_domains:
                        schema.property_domains[prop] = set()
                    schema.property_domains[prop].add(domain_str)

            # Range
            for _, _, range_val in g.triples((prop_uri, RDFS.range, None)):
                range_str = str(range_val)
                if _is_valid_uri(range_str):
                    schema.property_ranges[prop] = range_str

    except Exception as e:
        logger.warning(f"Failed to parse ontology {file_path}: {e}")


def load_schema_graph(dataset_id: str) -> SchemaGraph:
    """Load schema graph for a specific dataset.

    Args:
        dataset_id: Dataset identifier (e.g., "TRN", "EDU")

    Returns:
        SchemaGraph with all schema information for the dataset
    """
    schema = SchemaGraph(
        prefixes=COMMON_PREFIXES.copy(),
        dataset_id=dataset_id,
    )

    # Load class semantic models
    models_path = get_semantic_models_path(dataset_id)
    if models_path.exists():
        for ttl_file in models_path.glob("*.ttl"):
            _parse_semantic_model_file(ttl_file, schema)
        logger.debug(
            f"Loaded {len(schema.classes)} classes and "
            f"{len(schema.properties)} properties from {dataset_id} semantic models"
        )
    else:
        logger.warning(f"Semantic models path not found: {models_path}")

    # Load ontology for class hierarchy
    ontology_path = get_ontology_path(dataset_id)
    _parse_ontology_file(ontology_path, schema)

    return schema


class _SchemaCache:
    """Thread-safe cache for schema graphs."""

    _instance: Optional["_SchemaCache"] = None
    _lock = threading.RLock()  # RLock allows reentrant locking (same thread can acquire multiple times)

    def __new__(cls) -> "_SchemaCache":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._cache: dict[str, SchemaGraph] = {}
                    cls._instance._combined_cache: dict[str, SchemaGraph] = {}
        return cls._instance

    def get_or_load(self, dataset_id: str) -> SchemaGraph:
        """Get schema graph for dataset, loading if necessary."""
        dataset_id = dataset_id.upper()

        if dataset_id not in self._cache:
            with self._lock:
                if dataset_id not in self._cache:
                    self._cache[dataset_id] = load_schema_graph(dataset_id)

        return self._cache[dataset_id]

    def get_combined(self, dataset_ids: list[str]) -> SchemaGraph:
        """Get merged schema graph for multiple datasets."""
        # Normalize and sort for consistent cache key
        normalized_ids = sorted([d.upper() for d in dataset_ids])
        cache_key = "|".join(normalized_ids)

        if cache_key not in self._combined_cache:
            with self._lock:
                if cache_key not in self._combined_cache:
                    self._combined_cache[cache_key] = _merge_schemas(
                        [self.get_or_load(d) for d in normalized_ids]
                    )

        return self._combined_cache[cache_key]

    def clear(self) -> None:
        """Clear all cached schemas."""
        with self._lock:
            self._cache.clear()
            self._combined_cache.clear()


def _merge_schemas(schemas: list[SchemaGraph]) -> SchemaGraph:
    """Merge multiple schema graphs into one."""
    merged = SchemaGraph(
        prefixes=COMMON_PREFIXES.copy(),
        dataset_id="|".join(s.dataset_id for s in schemas),
    )

    for schema in schemas:
        merged.classes.update(schema.classes)
        merged.properties.update(schema.properties)
        merged.datatype_properties.update(schema.datatype_properties)
        merged.object_properties.update(schema.object_properties)
        merged.prefixes.update(schema.prefixes)

        # Merge class_properties
        for cls, props in schema.class_properties.items():
            if cls not in merged.class_properties:
                merged.class_properties[cls] = set()
            merged.class_properties[cls].update(props)

        # Merge property_domains
        for prop, domains in schema.property_domains.items():
            if prop not in merged.property_domains:
                merged.property_domains[prop] = set()
            merged.property_domains[prop].update(domains)

        # Merge property_ranges (last wins if conflict)
        merged.property_ranges.update(schema.property_ranges)

        # Merge class_property_ranges (set union per class+property)
        for cls, prop_ranges in schema.class_property_ranges.items():
            if cls not in merged.class_property_ranges:
                merged.class_property_ranges[cls] = {}
            for prop, ranges in prop_ranges.items():
                merged.class_property_ranges[cls].setdefault(prop, set()).update(ranges)

        # Merge subclass_of
        for cls, parents in schema.subclass_of.items():
            if cls not in merged.subclass_of:
                merged.subclass_of[cls] = set()
            merged.subclass_of[cls].update(parents)

    return merged


def get_combined_schema(dataset_ids: list[str] | None = None) -> SchemaGraph:
    """Get combined schema for specified datasets (or all datasets).

    Args:
        dataset_ids: List of dataset IDs, or None for all available datasets

    Returns:
        Combined SchemaGraph
    """
    if dataset_ids is None:
        dataset_ids = AVAILABLE_DATASETS

    cache = _SchemaCache()
    return cache.get_combined(dataset_ids)


def get_schema_for_dataset(dataset_id: str) -> SchemaGraph:
    """Get schema for a single dataset (cached).

    Args:
        dataset_id: Dataset identifier

    Returns:
        SchemaGraph for the dataset
    """
    cache = _SchemaCache()
    return cache.get_or_load(dataset_id)


def clear_schema_cache() -> None:
    """Clear the schema cache (useful for testing)."""
    cache = _SchemaCache()
    cache.clear()
