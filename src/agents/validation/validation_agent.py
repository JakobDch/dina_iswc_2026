"""
Validation agent for SPARQL query results.

Validates and selects the best query based on execution results.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from src.config import get_settings
from src.tools.validation_tools import VALIDATION_TOOLS
from src.tools.sparql_tools import execute_sparql


SYSTEM_PROMPT = """You are a validation agent for SPARQL queries. Your task is to:

1. Execute each query variant against the endpoint
2. Compare the results across variants
3. Calculate confidence based on result agreement
4. Select the best query based on:
   - Syntax validity
   - Non-empty results (when expected)
   - Agreement with other variants
   - Reasonable result size

Use the available tools to compare results and calculate confidence.
Return the index of the best query and your reasoning."""


class ValidationAgent:
    """Agent for validating and selecting SPARQL queries."""

    def __init__(self, llm_model: str = "deepseek-chat"):
        """Initialize the validation agent."""
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

        self.llm_with_tools = self.llm.bind_tools(VALIDATION_TOOLS)

    async def validate_and_select(
        self,
        queries: list[str],
        user_query: str,
    ) -> dict:
        """
        Validate queries and select the best one.

        Args:
            queries: List of SPARQL queries to validate
            user_query: Original user query for context

        Returns:
            Dict with selected_index, final_query, confidence, and results
        """
        # Execute all queries
        results = []
        syntax_valid = []

        for query in queries:
            exec_result = execute_sparql.invoke({"query": query})
            results.append(exec_result)
            syntax_valid.append(exec_result.get("success", False))

        # Calculate confidence
        from src.tools.validation_tools import calculate_confidence, select_best_query

        # Get successful results for comparison
        successful_results = [
            r.get("results", {})
            for r in results
            if r.get("success", False)
        ]

        confidence = calculate_confidence.invoke({
            "results_list": successful_results,
            "syntax_valid_list": syntax_valid,
        })

        # Select best query
        selection = select_best_query.invoke({
            "queries": queries,
            "results_list": results,
        })

        selected_idx = selection["selected_index"]

        return {
            "selected_index": selected_idx,
            "final_query": queries[selected_idx] if selected_idx >= 0 else None,
            "confidence_score": confidence,
            "query_results": results,
            "reasoning": selection.get("reasoning", ""),
        }

    async def back_translate(self, sparql_query: str) -> str:
        """
        Translate a SPARQL query back to natural language.

        Used for similarity comparison with original query.
        """
        messages = [
            SystemMessage(
                content="Translate this SPARQL query to a natural language question. "
                "Be concise and capture the intent of the query."
            ),
            HumanMessage(content=f"SPARQL Query:\n{sparql_query}"),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content
