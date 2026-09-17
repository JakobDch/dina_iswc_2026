"""
Semantic Model Context Agent.

This agent processes semantic model schemas and data instances to generate
an optimized context for SPARQL query generation. It:

1. Takes a complete schema (TTL format) as input
2. Takes data instance information as (Class, Property, Value) tuples
3. Filters out irrelevant triples based on the user query
4. Injects found data instances directly into the schema

Example transformation:
    Input Schema:
        eno:Company eno:name xsd:string ;
            eno:companyGroup xsd:string .

    Data Instance:
        ("Company", "name", "Bayer Group AG")

    Output:
        eno:Company eno:name "Bayer Group AG" ;
            eno:companyGroup xsd:string .
"""

import logging
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class DataInstance:
    """A data instance to inject into the schema."""
    class_name: str
    property_name: str
    value: str
    prefix: str = "ub"

    @property
    def class_uri(self) -> str:
        return f"{self.prefix}:{self.class_name}"

    @property
    def property_uri(self) -> str:
        return f"{self.prefix}:{self.property_name}"

    def to_triple(self) -> str:
        """Format as a triple string."""
        escaped = self.value.replace('"', '\\"')
        return f'{self.class_uri} {self.property_uri} "{escaped}" .'


FILTER_PROMPT = """You are an expert at analyzing RDF schemas for SPARQL query generation.

Given a user's natural language query and a schema, identify:
1. Which classes are RELEVANT to answering the query
2. Which properties are RELEVANT to answering the query

Output a JSON object with:
- relevant_classes: list of class names (without prefix)
- relevant_properties: list of property names (without prefix)
- reasoning: brief explanation

Be conservative - only include what's truly needed. Exclude administrative properties like dateSynchronized, factPageURL, etc. unless explicitly asked for.

IMPORTANT: Output ONLY valid JSON, no markdown."""


class SemanticModelContextAgent:
    """
    Agent that optimizes schema context for SPARQL generation.

    Main responsibilities:
    1. Filter irrelevant triples from schema
    2. Inject data instance values into schema
    3. Produce clean, focused context for SPARQL generation
    """

    def __init__(self, llm_model: str = "deepseek-chat"):
        """Initialize the context agent."""
        settings = get_settings()

        if "claude" in llm_model.lower():
            self.llm = ChatAnthropic(
                model=llm_model,
                api_key=settings.anthropic_api_key,
            )
        elif "deepseek" in llm_model.lower():
            self.llm = ChatOpenAI(
                model=llm_model,
                api_key=settings.deepseek_api_key,
                base_url="https://api.deepseek.com",
            )
        else:
            self.llm = ChatOpenAI(
                model=llm_model,
                api_key=settings.openai_api_key,
            )

    async def analyze_relevance(
        self,
        user_query: str,
        schema: str,
    ) -> dict:
        """
        Analyze which schema elements are relevant to the query.

        Args:
            user_query: The natural language query
            schema: The complete schema as TTL string

        Returns:
            Dict with relevant_classes and relevant_properties
        """
        prompt = f"""User Query: {user_query}

Schema:
{schema}

Analyze which classes and properties are needed to answer this query."""

        messages = [
            SystemMessage(content=FILTER_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            import json
            # Try to parse JSON from response
            content = response.content.strip()
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = re.sub(r'^```\w*\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
            return json.loads(content)
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "relevant_classes": [],
                "relevant_properties": [],
                "reasoning": f"Analysis failed: {e}",
            }

    def filter_schema(
        self,
        schema: str,
        relevant_classes: list[str],
        relevant_properties: list[str],
    ) -> str:
        """
        Filter schema to only include relevant classes and their properties.

        Args:
            schema: Complete schema as TTL string
            relevant_classes: List of relevant class names
            relevant_properties: List of relevant property names

        Returns:
            Filtered schema string
        """
        if not relevant_classes and not relevant_properties:
            return schema

        lines = schema.split("\n")
        filtered_lines = []
        current_class = None
        current_class_relevant = False
        current_block_lines = []

        # Normalize names for comparison
        relevant_classes_lower = {c.lower() for c in relevant_classes}
        relevant_props_lower = {p.lower() for p in relevant_properties}

        for line in lines:
            stripped = line.strip()

            # Always keep prefix declarations
            if stripped.startswith("@prefix"):
                filtered_lines.append(line)
                continue

            # Empty lines
            if not stripped:
                if current_class_relevant and current_block_lines:
                    filtered_lines.extend(current_block_lines)
                    filtered_lines.append("")
                current_block_lines = []
                continue

            # Check if this is a class definition (starts a new block)
            class_match = re.match(r'^(\w+):(\w+)\s+', stripped)
            if class_match:
                # Save previous block if relevant
                if current_class_relevant and current_block_lines:
                    filtered_lines.extend(current_block_lines)
                    filtered_lines.append("")

                # Start new block
                current_class = class_match.group(2)
                current_class_relevant = current_class.lower() in relevant_classes_lower
                current_block_lines = []

                if current_class_relevant:
                    # Filter properties within this class
                    filtered_line = self._filter_class_line(
                        stripped, relevant_props_lower
                    )
                    if filtered_line:
                        current_block_lines.append(filtered_line)
            elif current_class_relevant:
                # Continuation of a relevant class block
                # Check if any property on this line is relevant
                filtered_line = self._filter_property_line(
                    stripped, relevant_props_lower
                )
                if filtered_line:
                    current_block_lines.append("    " + filtered_line)

        # Don't forget the last block
        if current_class_relevant and current_block_lines:
            filtered_lines.extend(current_block_lines)

        return "\n".join(filtered_lines)

    def _filter_class_line(
        self,
        line: str,
        relevant_props: set[str],
    ) -> str:
        """Filter a class definition line to only include relevant properties."""
        # Parse class URI and properties
        parts = line.split(";")
        if not parts:
            return ""

        # First part contains class URI and first property
        first_part = parts[0].strip()
        match = re.match(r'^(\w+:\w+)\s+(.+)$', first_part)
        if not match:
            return ""

        class_uri = match.group(1)
        first_prop = match.group(2).strip()

        # Check if first property is relevant
        filtered_props = []
        prop_match = re.match(r'(\w+):(\w+)', first_prop)
        if prop_match and prop_match.group(2).lower() in relevant_props:
            filtered_props.append(first_prop)

        # Check remaining properties
        for part in parts[1:]:
            part = part.strip()
            if not part or part == ".":
                continue
            prop_match = re.match(r'(\w+):(\w+)', part)
            if prop_match and prop_match.group(2).lower() in relevant_props:
                filtered_props.append(part)

        if not filtered_props:
            return ""

        # Reconstruct the line
        if len(filtered_props) == 1:
            return f"{class_uri} {filtered_props[0]} ."
        else:
            return f"{class_uri} {filtered_props[0]} ;"

    def _filter_property_line(
        self,
        line: str,
        relevant_props: set[str],
    ) -> str:
        """Filter a property line (continuation line)."""
        stripped = line.strip()
        if not stripped or stripped == ".":
            return ""

        # Check if property is relevant
        prop_match = re.match(r'(\w+):(\w+)', stripped)
        if prop_match and prop_match.group(2).lower() in relevant_props:
            return stripped
        return ""

    def inject_instances(
        self,
        schema: str,
        instances: list[DataInstance],
    ) -> str:
        """
        Inject data instance values into the schema.

        Replaces generic type declarations (e.g., xsd:string) with
        actual values where matching instances are provided.

        Args:
            schema: Schema as TTL string
            instances: List of DataInstance objects to inject

        Returns:
            Schema with injected values
        """
        if not instances:
            return schema

        # Group instances by (class, property)
        instance_map: dict[tuple[str, str], str] = {}
        for inst in instances:
            key = (inst.class_name.lower(), inst.property_name.lower())
            # Use first matching instance
            if key not in instance_map:
                instance_map[key] = inst.value

        lines = schema.split("\n")
        result_lines = []
        current_class = None

        for line in lines:
            stripped = line.strip()

            # Keep prefixes and empty lines as-is
            if stripped.startswith("@prefix") or not stripped:
                result_lines.append(line)
                continue

            # Track current class
            class_match = re.match(r'^(\w+):(\w+)\s+', stripped)
            if class_match:
                current_class = class_match.group(2)

            # Try to inject instance values
            if current_class:
                modified_line = self._inject_into_line(
                    line, current_class, instance_map
                )
                result_lines.append(modified_line)
            else:
                result_lines.append(line)

        return "\n".join(result_lines)

    def _inject_into_line(
        self,
        line: str,
        current_class: str,
        instance_map: dict[tuple[str, str], str],
    ) -> str:
        """Inject instance value into a single line if applicable."""
        # Find property declarations with xsd: types
        # Pattern: property xsd:type ; or property xsd:type .
        def replace_value(match):
            prefix = match.group(1)
            prop_name = match.group(2)
            xsd_type = match.group(3)
            terminator = match.group(4)  # ; or .

            key = (current_class.lower(), prop_name.lower())
            if key in instance_map:
                value = instance_map[key]
                escaped = value.replace('"', '\\"')
                return f'{prefix}:{prop_name} "{escaped}" {terminator}'
            return match.group(0)

        # Match patterns like "eno:name xsd:string ;" or "eduo:emailAddress xsd:string ."
        pattern = r'(\w+):(\w+)\s+xsd:\w+\s*([;.])'
        return re.sub(pattern, replace_value, line)

    async def generate_context(
        self,
        user_query: str,
        schema: str,
        instances: list[DataInstance] | None = None,
        filter_irrelevant: bool = True,
    ) -> dict:
        """
        Generate optimized context for SPARQL generation.

        Args:
            user_query: The natural language query
            schema: Complete schema as TTL string
            instances: Optional list of data instances to inject
            filter_irrelevant: Whether to filter irrelevant triples

        Returns:
            Dict with optimized schema and metadata
        """
        result_schema = schema
        analysis = None

        # Step 1: Analyze and filter if requested
        if filter_irrelevant:
            analysis = await self.analyze_relevance(user_query, schema)
            result_schema = self.filter_schema(
                schema,
                analysis.get("relevant_classes", []),
                analysis.get("relevant_properties", []),
            )

        # Step 2: Inject instances if provided
        if instances:
            result_schema = self.inject_instances(result_schema, instances)

        # Step 3: Extract prefixes
        prefixes = self._extract_prefixes(result_schema)

        return {
            "optimized_schema": result_schema,
            "prefixes": prefixes,
            "analysis": analysis,
            "instances_injected": len(instances) if instances else 0,
            "stats": {
                "original_lines": len(schema.split("\n")),
                "filtered_lines": len(result_schema.split("\n")),
            },
        }

    def _extract_prefixes(self, schema: str) -> list[str]:
        """Extract prefix declarations from schema."""
        prefixes = []
        for line in schema.split("\n"):
            if line.strip().startswith("@prefix"):
                prefixes.append(line.strip())
        return prefixes


async def build_optimized_context(
    user_query: str,
    schema: str,
    instances: list[DataInstance] | None = None,
    filter_irrelevant: bool = True,
    llm_model: str = "deepseek-chat",
) -> dict:
    """
    Convenience function to build optimized SPARQL context.

    Args:
        user_query: Natural language query
        schema: Complete schema as TTL string
        instances: Optional data instances to inject
        filter_irrelevant: Whether to filter irrelevant triples
        llm_model: LLM model to use

    Returns:
        Optimized context dict
    """
    agent = SemanticModelContextAgent(llm_model=llm_model)
    return await agent.generate_context(
        user_query=user_query,
        schema=schema,
        instances=instances,
        filter_irrelevant=filter_irrelevant,
    )


# Keep old name for backwards compatibility
build_sparql_context = build_optimized_context
