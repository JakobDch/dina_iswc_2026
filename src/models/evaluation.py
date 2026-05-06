"""Pydantic models for evaluation results."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from src.config import ApproachType


class ResultMetrics(BaseModel):
    """Data-based evaluation metrics."""

    execution_accuracy: float = Field(
        ...,
        description="1.0 if results match exactly, 0.0 otherwise",
    )
    precision: float = Field(
        ...,
        description="TP / (TP + FP)",
    )
    recall: float = Field(
        ...,
        description="TP / (TP + FN)",
    )
    f1_score: float = Field(
        ...,
        description="Harmonic mean of precision and recall",
    )


class QueryMetrics(BaseModel):
    """Query-based evaluation metrics using algebraic SPARQL analysis."""

    edit_cost: float = Field(
        ...,
        description="Normalized edit cost score (0-100, where 100 = identical)",
    )
    actual_edit_cost: int = Field(
        default=0,
        description="Actual edit cost (number of edits needed)",
    )
    max_edit_cost: int = Field(
        default=0,
        description="Maximum possible edit cost (cost to build from scratch)",
    )
    edit_details: list[dict] = Field(
        default_factory=list,
        description="Detailed breakdown of individual edits",
    )
    corrected_query: str | None = Field(
        default=None,
        description="Agent-corrected query used for edit cost calculation",
    )
    syntax_valid: bool = Field(
        ...,
        description="Whether the generated query is syntactically valid",
    )
    # Iterative correction tracking
    correction_success: bool = Field(
        default=False,
        description="True if corrected query produces groundtruth results",
    )
    correction_iterations: int = Field(
        default=0,
        description="Number of correction iterations performed",
    )
    correction_attempts: list[dict] = Field(
        default_factory=list,
        description="All correction attempts with feedback",
    )


class AgentMetrics(BaseModel):
    """Agent-specific metrics."""

    iteration_count: int = Field(
        ...,
        description="Number of iterations through agent phases",
    )
    retrieval_precision: float = Field(
        default=0.0,
        description="Precision of retrieved triples vs. relevant triples",
    )
    retrieval_recall: float = Field(
        default=0.0,
        description="Recall of retrieved triples vs. relevant triples",
    )
    confidence_score: float = Field(
        default=0.0,
        description="Agent's confidence in the result",
    )
    # Note: confidence_correlation is calculated post-hoc during data analysis,
    # not during the experiment run


class TokenUsage(BaseModel):
    """Token usage tracking."""

    prompt_tokens: int = Field(default=0, description="Input tokens")
    completion_tokens: int = Field(default=0, description="Output tokens")
    total_tokens: int = Field(default=0, description="Total tokens")
    estimated_cost_usd: float = Field(default=0.0, description="Estimated cost in USD")


class SingleRunResult(BaseModel):
    """Result from a single experiment run."""

    query_id: str = Field(..., description="ID of the query")
    approach: ApproachType = Field(..., description="Approach used")
    llm_model: str = Field(..., description="LLM model used")
    run_number: int = Field(..., description="Run number (1-indexed)")

    # Generated output
    generated_sparql: str = Field(..., description="Generated SPARQL query")
    final_results: list[dict] = Field(
        default_factory=list,
        description="Query execution results",
    )

    # Metrics
    result_metrics: ResultMetrics | None = Field(
        default=None,
        description="Data-based metrics (if ground truth available)",
    )
    query_metrics: QueryMetrics | None = Field(
        default=None,
        description="Query-based metrics (if ground truth available)",
    )
    agent_metrics: AgentMetrics | None = Field(
        default=None,
        description="Agent-specific metrics (for agentic approaches)",
    )
    token_usage: TokenUsage = Field(
        default_factory=TokenUsage,
        description="Token usage for this run",
    )

    # Timing
    total_time_ms: float = Field(..., description="Total processing time")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this run was executed",
    )

    # Errors
    success: bool = Field(..., description="Whether the run completed successfully")
    error_message: str | None = Field(
        default=None,
        description="Error message if run failed",
    )


class QueryAggregatedResult(BaseModel):
    """Aggregated results for a single query across multiple runs."""

    query_id: str = Field(..., description="ID of the query")
    query_text: str = Field(..., description="Original query text")
    approach: ApproachType = Field(..., description="Approach used")
    llm_model: str = Field(..., description="LLM model used")
    num_runs: int = Field(..., description="Number of runs")

    # Aggregated metrics (mean + std)
    mean_execution_accuracy: float = Field(default=0.0)
    std_execution_accuracy: float = Field(default=0.0)
    mean_f1_score: float = Field(default=0.0)
    std_f1_score: float = Field(default=0.0)
    mean_edit_cost: float = Field(default=0.0)
    std_edit_cost: float = Field(default=0.0)

    # Success rate
    success_rate: float = Field(default=0.0, description="Proportion of successful runs")

    # Individual runs
    runs: list[SingleRunResult] = Field(default_factory=list)


class ExperimentResult(BaseModel):
    """Complete experiment result."""

    experiment_name: str = Field(..., description="Name of the experiment")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the experiment was run",
    )
    config: dict = Field(
        default_factory=dict,
        description="Experiment configuration",
    )

    # Results per approach/model/query
    results: list[QueryAggregatedResult] = Field(default_factory=list)

    # Overall summary
    total_queries: int = Field(default=0)
    total_runs: int = Field(default=0)
    total_token_usage: TokenUsage = Field(default_factory=TokenUsage)


__all__ = [
    "ResultMetrics",
    "QueryMetrics",
    "AgentMetrics",
    "TokenUsage",
    "SingleRunResult",
    "QueryAggregatedResult",
    "ExperimentResult",
]
