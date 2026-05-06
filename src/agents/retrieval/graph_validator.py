"""
Graph validation for retrieval agent output.

Validates that the extracted tripels form a coherent, connected graph.
"""

import re
import logging
from dataclasses import dataclass
from rdflib import Graph

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of graph validation."""

    is_valid: bool
    parsed_graph: Graph | None
    syntax_error: str | None
    tripel_count: int
    is_connected: bool
    feedback_message: str | None


def extract_turtle_from_response(response_text: str) -> str:
    """
    Extract Turtle content from LLM response.

    Handles responses with markdown code blocks or plain Turtle.
    Strips all explanation text before/after the Turtle content.
    """
    # Try to extract from code block (with or without language tag)
    code_block_pattern = r"```(?:turtle|ttl|sparql|rdf)?\s*\n(.*?)```"
    matches = re.findall(code_block_pattern, response_text, re.DOTALL)

    if matches:
        # Use the longest code block (likely the main content)
        turtle_content = max(matches, key=len)
        return _clean_turtle_content(turtle_content)

    # Fallback: extract lines that look like Turtle
    if "@prefix" in response_text.lower() or "prefix" in response_text.upper():
        return _extract_turtle_lines(response_text)

    return response_text


def _clean_turtle_content(content: str) -> str:
    """Remove non-Turtle lines from extracted content."""
    lines = []
    for line in content.split("\n"):
        stripped = line.strip()
        # Skip empty lines at start
        if not lines and not stripped:
            continue
        # Skip obvious explanation text
        if stripped.startswith(("Here", "This", "The ", "I ", "Let me", "Actually", "Wait")):
            continue
        if stripped.startswith(("**", "- ", "* ", "1.", "2.", "3.")):
            continue
        lines.append(line)
    return "\n".join(lines)


def _extract_turtle_lines(response_text: str) -> str:
    """Extract Turtle lines from mixed content."""
    lines = []
    in_turtle = False

    for line in response_text.split("\n"):
        stripped = line.strip()

        # Start capturing at PREFIX declaration
        if stripped.upper().startswith("PREFIX") or stripped.startswith("@prefix"):
            in_turtle = True

        if in_turtle:
            # Stop at obvious non-Turtle content
            if stripped.startswith(("Here", "This", "The ", "I ", "Let me", "Actually")):
                break
            if stripped.startswith(("**", "- ", "* ")) and ":" not in stripped:
                continue
            # Keep Turtle-looking lines
            if not stripped or stripped.startswith("#"):
                lines.append(line)
            elif any(c in stripped for c in [":", ".", ";"]):
                lines.append(line)

    return "\n".join(lines)


def _build_adjacency(g: Graph) -> dict[str, set[str]]:
    """Build undirected adjacency map from an RDF graph."""
    adjacency: dict[str, set[str]] = {}
    for s, p, o in g:
        s_str = str(s)
        o_str = str(o)
        if s_str not in adjacency:
            adjacency[s_str] = set()
        if o_str not in adjacency:
            adjacency[o_str] = set()
        adjacency[s_str].add(o_str)
        adjacency[o_str].add(s_str)
    return adjacency


def _is_graph_connected(g: Graph) -> bool:
    """
    Check if the RDF graph is connected.

    A graph is connected if all nodes can be reached from any starting node
    when treating the graph as undirected.
    """
    if len(g) == 0:
        return True

    adjacency = _build_adjacency(g)
    if not adjacency:
        return True

    # BFS from first node
    start = next(iter(adjacency.keys()))
    visited = {start}
    queue = [start]

    while queue:
        node = queue.pop(0)
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == len(adjacency)


def _get_connected_components(g: Graph) -> list[set[str]]:
    """Get all connected components as sets of node strings."""
    adjacency = _build_adjacency(g)
    if not adjacency:
        return []

    remaining = set(adjacency.keys())
    components: list[set[str]] = []

    while remaining:
        start = next(iter(remaining))
        visited = {start}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neighbor in adjacency.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(visited)
        remaining -= visited

    return components


def _namespace_of(uri: str) -> str:
    """Extract namespace from a URI (everything up to and including last # or /)."""
    for sep in ("#", "/"):
        idx = uri.rfind(sep)
        if idx != -1:
            return uri[: idx + 1]
    return uri


def _is_cross_dataset_graph(g: Graph, components: list[set[str]]) -> bool:
    """Check if disconnected components originate from different ontology namespaces.

    For each component, collect the namespaces of RDF subjects that appear in
    triples belonging to that component.  If at least two components have
    *disjoint* namespace sets the graph is a valid cross-dataset schema.
    """
    # Map each component's nodes to the subject namespaces they contain
    component_namespaces: list[set[str]] = []
    for comp_nodes in components:
        namespaces: set[str] = set()
        for s, _p, _o in g:
            if str(s) in comp_nodes:
                ns = _namespace_of(str(s))
                namespaces.add(ns)
        component_namespaces.append(namespaces)

    # Cross-dataset if at least two components have non-overlapping namespaces
    for i in range(len(component_namespaces)):
        for j in range(i + 1, len(component_namespaces)):
            if component_namespaces[i].isdisjoint(component_namespaces[j]):
                return True
    return False


def validate_tripel_graph(turtle_content: str) -> ValidationResult:
    """
    Validate that extracted tripels form a coherent, connected graph.

    Checks:
    1. Turtle syntax is valid (can be parsed by rdflib)
    2. Graph is connected (all nodes reachable)

    Args:
        turtle_content: Turtle-formatted tripels

    Returns:
        ValidationResult with validation status and feedback
    """
    # Extract turtle from potential markdown
    clean_turtle = extract_turtle_from_response(turtle_content)

    if not clean_turtle.strip():
        return ValidationResult(
            is_valid=False,
            parsed_graph=None,
            syntax_error="No Turtle content found in response",
            tripel_count=0,
            is_connected=False,
            feedback_message="ERROR: No valid Turtle tripels found. Please provide tripels in this format:\n"
            "PREFIX prefix: <uri>\n"
            "Class property Range .",
        )

    # Try to parse with rdflib
    g = Graph()
    try:
        g.parse(data=clean_turtle, format="turtle")
    except Exception as e:
        error_msg = str(e)
        return ValidationResult(
            is_valid=False,
            parsed_graph=None,
            syntax_error=error_msg,
            tripel_count=0,
            is_connected=False,
            feedback_message=f"SYNTAX ERROR: Invalid Turtle syntax.\n"
            f"Error: {error_msg}\n\n"
            f"Please fix the syntax and provide valid Turtle tripels.",
        )

    tripel_count = len(g)

    if tripel_count == 0:
        return ValidationResult(
            is_valid=False,
            parsed_graph=g,
            syntax_error=None,
            tripel_count=0,
            is_connected=False,
            feedback_message="ERROR: No tripels found after parsing. "
            "Please provide at least one tripel in the format:\n"
            "Class property Range .",
        )

    # Check if graph is connected
    is_connected = _is_graph_connected(g)

    if not is_connected:
        # Cross-dataset queries intentionally have disconnected components
        # from different ontology namespaces (e.g. EDU + TRN).  Accept these.
        components = _get_connected_components(g)
        if _is_cross_dataset_graph(g, components):
            logger.info(
                f"Accepting disconnected graph as cross-dataset "
                f"({len(components)} components from different namespaces)"
            )
            return ValidationResult(
                is_valid=True,
                parsed_graph=g,
                syntax_error=None,
                tripel_count=tripel_count,
                is_connected=False,
                feedback_message=None,
            )

        return ValidationResult(
            is_valid=False,
            parsed_graph=g,
            syntax_error=None,
            tripel_count=tripel_count,
            is_connected=False,
            feedback_message="DISCONNECTED GRAPH: The tripels form multiple disconnected components. "
            "Please ensure all classes are connected through Object Properties. "
            "Add the missing connections between the components.",
        )

    # All valid
    return ValidationResult(
        is_valid=True,
        parsed_graph=g,
        syntax_error=None,
        tripel_count=tripel_count,
        is_connected=True,
        feedback_message=None,
    )


def serialize_validated_graph(validation_result: ValidationResult) -> str:
    """
    Serialize a validated graph back to clean Turtle format.

    Args:
        validation_result: A successful validation result

    Returns:
        Clean Turtle string
    """
    if not validation_result.parsed_graph:
        return ""

    return validation_result.parsed_graph.serialize(format="turtle")