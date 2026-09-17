"""
Utility functions for baseline pipeline.

Simplified version - removed web app specific utilities.
"""

import asyncio
import codecs
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from collections import Counter
from rdflib import Graph, URIRef, Literal, BNode, RDFS

from .config import DATA_DIR

logger = logging.getLogger(__name__)


def parse_ttl_to_graph_json(file_path: Path, request_id: str) -> Optional[Dict[str, List[Dict]]]:
    """
    Parses a Turtle file and converts it into a JSON structure
    suitable for graph visualization libraries like React Flow.
    Returns a dictionary with 'nodes' and 'edges' or None on failure.
    """
    if not file_path.exists():
        logger.error(f"[{request_id}] File not found for graph parsing: {file_path}")
        return None

    try:
        g = Graph()
        g.parse(str(file_path), format="turtle")

        nodes = {}
        edges = []
        edge_counter = Counter()

        def get_node_id(node: Union[URIRef, BNode, Literal]) -> str:
            if isinstance(node, URIRef):
                return str(node).split('#')[-1].split('/')[-1]
            elif isinstance(node, BNode):
                return str(node)
            return str(node)

        for s, p, o in g:
            source_id = get_node_id(s)
            target_id = get_node_id(o)

            edge_counter[source_id] += 1
            edge_counter[target_id] += 1

            for node_obj in [s, o]:
                node_id = get_node_id(node_obj)
                if node_id not in nodes:
                    nodes[node_id] = {
                        "id": node_id,
                        "data": {"label": node_id},
                        "position": {"x": 0, "y": 0}
                    }

            predicate_label = get_node_id(p)

            edge_id = f"edge-{source_id}-{target_id}-{predicate_label}"
            edges.append({
                "id": edge_id,
                "source": source_id,
                "target": target_id,
                "label": predicate_label,
                "type": "default"
            })

        if edge_counter:
            root_node_id = edge_counter.most_common(1)[0][0]
            if root_node_id in nodes:
                nodes[root_node_id]['type'] = 'custom'

        return {"nodes": list(nodes.values()), "edges": edges}

    except Exception as e:
        logger.error(f"[{request_id}] Failed to parse TTL file {file_path} into graph JSON: {e}", exc_info=True)
        return None


def parse_ttl_to_clean_triples(file_path: Path, request_id: str) -> Optional[str]:
    """
    Parses a Turtle (.ttl) file and returns a clean, readable string of its triples.
    Returns None if parsing fails.
    """
    if not file_path.exists():
        logger.error(f"[{request_id}] File not found for parsing: {file_path}")
        return None

    try:
        g = Graph()
        g.parse(str(file_path), format="turtle")

        if len(g) == 0:
            logger.warning(f"[{request_id}] Parsed graph from {file_path} is empty.")
            return "# The model is empty."

        def get_node_representation(node):
            if isinstance(node, URIRef):
                label = g.value(subject=node, predicate=RDFS.label)
                if label:
                    return f'"{label}"'
                return node.n3(g.namespace_manager)
            return node.n3()

        triple_strings = []
        for s, p, o in g:
            subj_repr = get_node_representation(s)
            pred_repr = get_node_representation(p)
            obj_repr = get_node_representation(o)
            triple_strings.append(f"{subj_repr} {pred_repr} {obj_repr} .")

        return "\n".join(triple_strings)

    except Exception as e:
        logger.error(f"[{request_id}] Failed to parse TTL file {file_path}: {e}", exc_info=True)
        return f"# Fehler beim Parsen des Modells: {e}"


def decode_unicode_escapes(data: Union[str, List[Any], Dict[Any, Any], None]) -> Union[str, List[Any], Dict[Any, Any], None]:
    """
    Recursively decodes Python-style unicode escape sequences in strings,
    lists, or dictionaries. E.g., "\\u00fc" becomes "ü".
    """
    if isinstance(data, str):
        if '\\u' in data or '\\U' in data:
            try:
                return codecs.decode(data, 'unicode_escape')
            except UnicodeDecodeError as e:
                logger.warning(f"UnicodeDecodeError while decoding '{data[:100]}...': {e}. Returning original string.")
                return data
            except Exception as e:
                logger.error(f"Unexpected error in decode_unicode_escapes for string '{data[:100]}...': {e}", exc_info=True)
                return data
        else:
            return data
    elif isinstance(data, list):
        return [decode_unicode_escapes(item) for item in data]
    elif isinstance(data, dict):
        return {
            decode_unicode_escapes(k) if isinstance(k, str) else k: decode_unicode_escapes(v)
            for k, v in data.items()
        }
    return data


def load_templates_for_workspace(workspace_id: str, request_id_str: str) -> List[Dict[str, Any]]:
    """Load request templates for a workspace."""
    workspace_template_dir = DATA_DIR / workspace_id / "request_templates"
    logger.info(f"[{workspace_id}] Attempting to load templates from directory {workspace_template_dir}. Request: {request_id_str}")

    if not workspace_template_dir.exists() or not workspace_template_dir.is_dir():
        logger.warning(f"[{workspace_id}] Templates directory not found at {workspace_template_dir}. Returning empty list. Request: {request_id_str}")
        return []

    json_files = list(workspace_template_dir.glob("*.json"))
    if not json_files:
        logger.warning(f"[{workspace_id}] No .json files found in {workspace_template_dir}. Returning empty list. Request: {request_id_str}")
        return []

    templates_file_path_to_load = json_files[0]
    logger.info(f"[{workspace_id}] Found template file to load: {templates_file_path_to_load}. Request: {request_id_str}")

    try:
        with open(templates_file_path_to_load, 'r', encoding='utf-8') as f:
            templates_data = json.load(f)
            if not isinstance(templates_data, list):
                logger.error(f"[{workspace_id}] Templates file at {templates_file_path_to_load} does not contain a JSON list. Content type: {type(templates_data)}. Request: {request_id_str}")
                return []
            logger.info(f"[{workspace_id}] Successfully loaded templates. Number of templates: {len(templates_data)}. First template ID (if any): {templates_data[0].get('id') if templates_data and len(templates_data) > 0 else 'N/A'}. Request: {request_id_str}")
            return templates_data
    except json.JSONDecodeError as e:
        logger.error(f"[{workspace_id}] Error decoding JSON from {templates_file_path_to_load}: {e}. Request: {request_id_str}")
        return []
    except Exception as e:
        logger.error(f"[{workspace_id}] An unexpected error occurred while loading templates from {templates_file_path_to_load}: {e}. Request: {request_id_str}")
        return []


def analyze_property_types_from_triples(model_path: Path, request_id: str = "unknown") -> str:
    """
    Analyzes property types from a Turtle file by examining triple objects.

    Logic:
    - If object ends with or contains xsd: datatypes -> DatatypeProperty
    - If object is a URI/class (not a datatype) -> ObjectProperty

    Returns:
    - "only_datatype": Only datatype properties found
    - "only_object": Only object properties found
    - "both": Both types found
    - "none": No properties found or error parsing
    """
    if not model_path.exists():
        logger.warning(f"[{request_id}] Model file not found for property analysis: {model_path}")
        return "none"

    try:
        g = Graph()
        g.parse(str(model_path), format="turtle")

        if len(g) == 0:
            logger.warning(f"[{request_id}] Empty graph in {model_path}")
            return "none"

        has_datatype_properties = False
        has_object_properties = False

        for s, p, o in g:
            obj_repr = o.n3(g.namespace_manager)

            if isinstance(o, Literal):
                has_datatype_properties = True
            elif 'xsd:' in obj_repr or obj_repr.startswith('xsd:'):
                has_datatype_properties = True
            else:
                has_object_properties = True

        if has_datatype_properties and has_object_properties:
            return "both"
        elif has_datatype_properties:
            return "only_datatype"
        elif has_object_properties:
            return "only_object"
        else:
            return "none"

    except Exception as e:
        logger.error(f"[{request_id}] Failed to analyze property types from {model_path}: {e}", exc_info=True)
        return "none"
