"""Pydantic models for query handling."""

from typing import Literal

from pydantic import BaseModel, Field


class NaturalLanguageQuery(BaseModel):
    """A natural language query from the user."""

    id: str = Field(..., description="Unique identifier for the query")
    text: str = Field(..., description="The natural language query text")
    expected_sparql: str | None = Field(
        default=None,
        description="Ground truth SPARQL query (if available)",
    )
    expected_results: list[dict] | None = Field(
        default=None,
        description="Expected query results (if available)",
    )


class RetrievedContext(BaseModel):
    """Context retrieved from mapping files."""

    mapping_file: str = Field(..., description="Source mapping file")
    triples: list[str] = Field(
        default_factory=list,
        description="Retrieved RDF triples from the mapping",
    )
    relevance_score: float = Field(
        default=0.0,
        description="Relevance score from retrieval",
    )


class GeneratedSPARQL(BaseModel):
    """A generated SPARQL query with metadata."""

    query: str = Field(..., description="The generated SPARQL query")
    reasoning: str = Field(
        default="",
        description="LLM reasoning for this query",
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence score for this query",
    )


class QueryExecutionResult(BaseModel):
    """Result from executing a SPARQL query."""

    query: str = Field(..., description="The executed SPARQL query")
    success: bool = Field(..., description="Whether execution was successful")
    results: list[dict] = Field(
        default_factory=list,
        description="Query results as list of bindings",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if execution failed",
    )
    execution_time_ms: float = Field(
        default=0.0,
        description="Query execution time in milliseconds",
    )


class OrchestratorState(BaseModel):
    """State for the orchestrator agent."""

    user_query: str = Field(..., description="Original user query")
    current_phase: Literal["retrieval", "generation", "validation"] = Field(
        default="retrieval",
        description="Current processing phase",
    )
    retrieved_context: list[RetrievedContext] = Field(
        default_factory=list,
        description="Retrieved mapping triples",
    )
    generated_queries: list[GeneratedSPARQL] = Field(
        default_factory=list,
        description="Generated SPARQL queries (k variants)",
    )
    query_results: list[QueryExecutionResult] = Field(
        default_factory=list,
        description="Execution results for generated queries",
    )
    confidence_score: float = Field(
        default=0.0,
        description="Overall confidence based on result variance",
    )
    iteration_count: int = Field(
        default=0,
        description="Number of iterations through phases",
    )
    final_query: str | None = Field(
        default=None,
        description="Selected final SPARQL query",
    )
    error_messages: list[str] = Field(
        default_factory=list,
        description="Error messages encountered during processing",
    )


__all__ = [
    "NaturalLanguageQuery",
    "RetrievedContext",
    "GeneratedSPARQL",
    "QueryExecutionResult",
    "OrchestratorState",
]
