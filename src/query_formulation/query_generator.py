"""
User Query Formulation Module.

This module implements the query formulation process described in the experiment plan:
1. Define target data (what data should be retrieved)
2. Provide textual description of the data
3. External person or LLM formulates natural language queries
4. Generate variations of queries for robustness testing
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from src.config import get_settings, DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class TargetDataSpec:
    """Specification of target data for query formulation."""

    target_description: str  # What data should be retrieved
    expected_result_type: str  # e.g., "list of names", "count", "yes/no"
    relevant_classes: list[str] = field(default_factory=list)
    relevant_properties: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    example_results: list[dict] = field(default_factory=list)


@dataclass
class FormulatedQuery:
    """A formulated natural language query with metadata."""

    query: str
    complexity: str  # simple, medium, complex
    query_type: str  # lookup, aggregation, filter, join
    variation_of: str | None = None


QUERY_FORMULATION_PROMPT = """You are helping formulate natural language queries for a Text-to-SPARQL experiment.

Given:
1. A description of target data to be retrieved
2. Optional information about relevant schema elements
3. Optional example results

Your task is to formulate natural language queries that would retrieve this target data.
The queries should be:
- Natural and realistic (as a real user might ask)
- Clear about what data is needed
- Varied in phrasing and complexity

Output a JSON array of query objects, each with:
- query: The natural language query string
- complexity: "simple", "medium", or "complex"
- query_type: "lookup", "aggregation", "filter", "join", or "complex"

Generate diverse queries that cover different ways users might ask for the same data."""


VARIATION_PROMPT = """You are generating variations of a natural language query for robustness testing.

Original query: {original_query}

Generate {num_variations} variations that:
1. Ask for the same information
2. Use different phrasing, vocabulary, or sentence structure
3. Range from casual to formal
4. Include some with slight rephrasing that might trip up an NL system

Output a JSON array of query strings."""


class QueryFormulator:
    """Generates natural language queries for target data specifications."""

    def __init__(self, llm_model: str = "deepseek-chat"):
        """Initialize the query formulator."""
        settings = get_settings()

        if "claude" in llm_model.lower():
            self.llm = ChatAnthropic(
                model=llm_model,
                api_key=settings.anthropic_api_key,
            )
        else:
            self.llm = ChatOpenAI(
                model=llm_model,
                api_key=settings.openai_api_key,
            )

    async def formulate_queries(
        self,
        target_spec: TargetDataSpec,
        num_queries: int = 5,
    ) -> list[FormulatedQuery]:
        """
        Formulate natural language queries for a target data specification.

        Args:
            target_spec: Specification of target data
            num_queries: Number of queries to generate

        Returns:
            List of formulated queries
        """
        # Build context from target spec
        context_parts = [
            f"Target Data Description: {target_spec.target_description}",
            f"Expected Result Type: {target_spec.expected_result_type}",
        ]

        if target_spec.relevant_classes:
            context_parts.append(
                f"Relevant Classes: {', '.join(target_spec.relevant_classes)}"
            )

        if target_spec.relevant_properties:
            context_parts.append(
                f"Relevant Properties: {', '.join(target_spec.relevant_properties)}"
            )

        if target_spec.constraints:
            context_parts.append(
                f"Constraints: {', '.join(target_spec.constraints)}"
            )

        if target_spec.example_results:
            context_parts.append(
                f"Example Results: {json.dumps(target_spec.example_results[:3])}"
            )

        prompt = f"""{chr(10).join(context_parts)}

Generate {num_queries} natural language queries that would retrieve this target data.
Output as a JSON array."""

        messages = [
            SystemMessage(content=QUERY_FORMULATION_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            queries_data = json.loads(response.content)

            return [
                FormulatedQuery(
                    query=q["query"],
                    complexity=q.get("complexity", "medium"),
                    query_type=q.get("query_type", "lookup"),
                )
                for q in queries_data
            ]
        except Exception as e:
            logger.error(f"Query formulation failed: {e}")
            return []

    async def generate_variations(
        self,
        original_query: str,
        num_variations: int = 5,
    ) -> list[str]:
        """
        Generate variations of a natural language query.

        Args:
            original_query: The original query to create variations of
            num_variations: Number of variations to generate

        Returns:
            List of query variations
        """
        prompt = VARIATION_PROMPT.format(
            original_query=original_query,
            num_variations=num_variations,
        )

        messages = [
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            return json.loads(response.content)
        except Exception as e:
            logger.error(f"Variation generation failed: {e}")
            return [original_query]

    async def formulate_from_sparql(
        self,
        sparql_query: str,
        schema_context: str = "",
    ) -> list[FormulatedQuery]:
        """
        Formulate natural language queries from a SPARQL query (reverse engineering).

        This is useful for creating test sets from existing SPARQL benchmarks.

        Args:
            sparql_query: A SPARQL query to reverse-engineer
            schema_context: Optional schema context

        Returns:
            List of formulated queries
        """
        prompt = f"""Given this SPARQL query and optional schema context, formulate natural language queries that would produce this SPARQL query.

SPARQL Query:
{sparql_query}

{f"Schema Context:{chr(10)}{schema_context}" if schema_context else ""}

Generate 3-5 natural language queries that a user might ask to retrieve this data.
Output as a JSON array with 'query', 'complexity', and 'query_type' fields."""

        messages = [
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            queries_data = json.loads(response.content)

            return [
                FormulatedQuery(
                    query=q["query"],
                    complexity=q.get("complexity", "medium"),
                    query_type=q.get("query_type", "lookup"),
                )
                for q in queries_data
            ]
        except Exception as e:
            logger.error(f"Reverse formulation failed: {e}")
            return []


async def generate_query_variations(
    original_query: str,
    num_variations: int = 5,
    llm_model: str = "deepseek-chat",
) -> list[str]:
    """
    Convenience function to generate query variations.

    Args:
        original_query: Original query string
        num_variations: Number of variations to generate
        llm_model: LLM model to use

    Returns:
        List of query variations including original
    """
    formulator = QueryFormulator(llm_model=llm_model)
    variations = await formulator.generate_variations(original_query, num_variations)
    return [original_query] + variations


async def generate_queries_for_target_data(
    target_description: str,
    expected_result_type: str = "list",
    relevant_classes: list[str] | None = None,
    relevant_properties: list[str] | None = None,
    num_queries: int = 5,
    llm_model: str = "deepseek-chat",
) -> list[FormulatedQuery]:
    """
    Convenience function to generate queries for target data.

    Args:
        target_description: Description of target data
        expected_result_type: Type of expected results
        relevant_classes: Optional list of relevant classes
        relevant_properties: Optional list of relevant properties
        num_queries: Number of queries to generate
        llm_model: LLM model to use

    Returns:
        List of formulated queries
    """
    spec = TargetDataSpec(
        target_description=target_description,
        expected_result_type=expected_result_type,
        relevant_classes=relevant_classes or [],
        relevant_properties=relevant_properties or [],
    )

    formulator = QueryFormulator(llm_model=llm_model)
    return await formulator.formulate_queries(spec, num_queries)


def load_query_set(file_path: Path) -> list[dict]:
    """
    Load a query set from a JSON file.

    Expected format:
    [
        {
            "id": "q1",
            "query": "Find all graduate students",
            "expected_sparql": "SELECT ?s WHERE { ?s a eduo:DoctoralCandidate }",
            "expected_results": {...},
            "variations": ["List graduate students", ...]
        },
        ...
    ]
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load query set from {file_path}: {e}")
        return []


def save_query_set(queries: list[dict], file_path: Path) -> bool:
    """
    Save a query set to a JSON file.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(queries, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Failed to save query set to {file_path}: {e}")
        return False
