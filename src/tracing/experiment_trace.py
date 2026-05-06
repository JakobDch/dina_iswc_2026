"""
Detailed experiment tracing models for VKGQA evaluation.

This module provides comprehensive tracing for scientific evaluation,
capturing all relevant data during an experiment run without redundant
information (no raw prompts, no edit cost, no semantic similarity).

Key design decisions:
- Query metadata from UseCaseQuery is captured for post-hoc correlation analysis
- LLM calls track tokens and timing, not full prompts (templates are static)
- Strategy (grep/semantic) is fixed per run for fair comparison
- Confidence can optionally be requested from the LLM
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# =============================================================================
# Query Metadata (captured from AdaptiveGroundTruth for correlation analysis)
# =============================================================================


class QueryMetadata(BaseModel):
    """
    Metadata about the query being evaluated.

    Captured from AdaptiveGroundTruth for post-hoc analysis.
    """

    # Identification
    query_id: str = Field(..., description="Query ID (e.g., 'BASE01', 'SYN01')")
    query_set: Literal["BASE", "SYN", "TYPO", "UNDER", "CROSS"] = Field(
        ..., description="Query set category"
    )
    query_text: str = Field(..., description="Natural language query")

    # Dataset info
    dataset: str = Field(..., description="Primary dataset (e.g., 'EDU', 'NRG')")
    datasets: list[str] = Field(
        default_factory=list, description="All involved datasets"
    )

    # Ground truth
    expected_sparql: str = Field(default="", description="Ground truth SPARQL")
    expected_results_count: int | None = Field(
        default=None, description="Expected result count"
    )


# =============================================================================
# LLM Call Tracing
# =============================================================================


class LLMCallTrace(BaseModel):
    """
    Trace of a single LLM invocation.

    Does NOT include full prompts (templates are static).
    Tracks tokens, timing, and context size for cost/efficiency analysis.
    """

    timestamp: datetime = Field(default_factory=datetime.now)
    phase: Literal["retrieval", "generation", "orchestrator"] = Field(
        ..., description="Which phase made this call"
    )
    purpose: str = Field(
        ...,
        description="Purpose of the call (e.g., 'generate_queries', 'evaluate_retrieval')",
    )

    # Model info
    model: str = Field(..., description="LLM model used")

    # Token usage
    input_tokens: int = Field(default=0, description="Prompt tokens")
    output_tokens: int = Field(default=0, description="Completion tokens")
    total_tokens: int = Field(default=0, description="Total tokens")

    # Context size (what varied in the prompt)
    context_triples_count: int = Field(
        default=0, description="Number of schema triples in context"
    )
    context_char_count: int = Field(
        default=0, description="Character count of dynamic context"
    )

    # Timing
    duration_ms: float = Field(default=0.0, description="LLM inference time")

    # Cost estimation
    estimated_cost_usd: float = Field(default=0.0, description="Estimated cost in USD")


# =============================================================================
# SPARQL Execution Tracing
# =============================================================================


class SPARQLExecutionTrace(BaseModel):
    """Trace of a single SPARQL query execution."""

    timestamp: datetime = Field(default_factory=datetime.now)
    query: str = Field(..., description="The executed SPARQL query")
    query_index: int = Field(
        default=0, description="Index in the batch of generated queries"
    )

    # Execution result
    success: bool = Field(..., description="Whether execution succeeded")
    result_count: int | None = Field(
        default=None, description="Number of result rows"
    )
    results_sample: list[dict] = Field(
        default_factory=list,
        description="First 5 result rows (for debugging)",
    )

    # Error info
    error_type: str | None = Field(default=None, description="Error type if failed")
    error_message: str | None = Field(
        default=None, description="Error message if failed"
    )

    # Timing
    duration_ms: float = Field(default=0.0, description="Query execution time")
    timeout_seconds: float = Field(
        default=15.0, description="Timeout used for this query execution"
    )

    # Endpoint info
    endpoint: str = Field(default="", description="SPARQL endpoint used")


# =============================================================================
# Phase-Specific Traces
# =============================================================================


class RetrievalPhaseTrace(BaseModel):
    """Trace of a single retrieval phase execution."""

    timestamp: datetime = Field(default_factory=datetime.now)
    iteration: int = Field(default=1, description="Iteration number (1-indexed)")

    # Strategy (fixed per experiment run!)
    strategy: Literal["grep", "semantic"] = Field(
        ..., description="Retrieval strategy used (fixed per run)"
    )

    # Output
    triples_found: int = Field(default=0, description="Number of triples retrieved")
    retrieved_triples: list[str] = Field(
        default_factory=list,
        description="Actual triple strings retrieved in this phase",
    )

    # LLM calls in this phase
    llm_calls: list[LLMCallTrace] = Field(
        default_factory=list, description="LLM calls made during retrieval"
    )

    # Timing
    duration_ms: float = Field(default=0.0, description="Total phase duration")


class GenerationPhaseTrace(BaseModel):
    """Trace of a single generation phase execution."""

    timestamp: datetime = Field(default_factory=datetime.now)
    iteration: int = Field(default=1, description="Iteration number (1-indexed)")

    # Input context
    schema_triples_count: int = Field(
        default=0, description="Number of schema triples used as context"
    )
    execution_feedback: str = Field(
        default="", description="Feedback from query execution (results passed to SPARQL agent for refinement)"
    )

    # Output
    queries_generated: int = Field(default=0, description="Number of queries generated")
    queries_with_syntax_errors: int = Field(
        default=0, description="Queries with syntax errors"
    )
    queries_executed_successfully: int = Field(
        default=0, description="Queries that executed without error"
    )
    queries_with_results: int = Field(
        default=0, description="Queries that returned non-empty results"
    )

    # SPARQL executions
    sparql_executions: list[SPARQLExecutionTrace] = Field(
        default_factory=list, description="All SPARQL executions in this phase"
    )

    # LLM calls in this phase
    llm_calls: list[LLMCallTrace] = Field(
        default_factory=list, description="LLM calls made during generation"
    )

    # Timing
    duration_ms: float = Field(default=0.0, description="Total phase duration")
    llm_time_ms: float = Field(default=0.0, description="Time spent on LLM calls")
    sparql_time_ms: float = Field(
        default=0.0, description="Time spent on SPARQL execution"
    )

    # Orchestrator decision after this phase
    orchestrator_decision: Literal[
        "retrieval", "generation", "complete", "unanswerable"
    ] = Field(default="complete", description="What orchestrator decided")
    orchestrator_reasoning: str = Field(
        default="", description="Orchestrator's reasoning"
    )
    selected_query_index: int = Field(
        default=-1, description="Index of selected query (-1 if none)"
    )


# =============================================================================
# Evaluation Metrics
# =============================================================================


class ResultMetrics(BaseModel):
    """
    Data-based evaluation metrics (comparing results to ground truth).

    IMPORTANT: Variable names are IGNORED when comparing results.
    Only the actual data values are compared row-by-row.
    This is the standard approach for Text2SPARQL evaluation.
    """

    # Core metrics
    execution_accuracy: float = Field(
        default=0.0, description="1.0 if result sets match exactly, 0.0 otherwise"
    )
    precision: float = Field(default=0.0, description="TP / (TP + FP)")
    recall: float = Field(default=0.0, description="TP / (TP + FN)")
    f1_score: float = Field(
        default=0.0, description="Harmonic mean of precision and recall"
    )

    # Result counts for analysis
    true_positives: int = Field(default=0, description="Matching result rows")

    # Actual vs expected
    actual_result_count: int = Field(default=0, description="Actual result count")
    expected_result_count: int = Field(default=0, description="Expected result count")


class TokenUsage(BaseModel):
    """Aggregated token usage across all LLM calls."""

    prompt_tokens: int = Field(default=0, description="Total input tokens")
    completion_tokens: int = Field(default=0, description="Total output tokens")
    total_tokens: int = Field(default=0, description="Total tokens")
    estimated_cost_usd: float = Field(default=0.0, description="Estimated total cost")

    # Per-phase breakdown
    retrieval_tokens: int = Field(default=0, description="Tokens in retrieval phase")
    generation_tokens: int = Field(default=0, description="Tokens in generation phase")
    orchestrator_tokens: int = Field(
        default=0, description="Tokens in orchestrator calls"
    )


# =============================================================================
# Tool and Agent Usage Tracking (NEW)
# =============================================================================


class ToolInvocation(BaseModel):
    """Single tool invocation record with effectiveness tracking."""

    timestamp: datetime = Field(default_factory=datetime.now)
    tool_name: str = Field(..., description="Name of the tool (e.g., 'grep_classes')")
    agent: str = Field(..., description="Agent that invoked the tool")
    duration_ms: float = Field(default=0.0, description="Execution time")
    success: bool = Field(default=True, description="Whether invocation succeeded")
    error_message: str | None = Field(default=None, description="Error if failed")
    # Tool-specific context (for analysis)
    input_summary: str = Field(default="", description="Brief summary of input args")
    output_size: int = Field(default=0, description="Size of output (chars or items)")
    # Tool effectiveness fields
    result_count: int = Field(default=0, description="Number of results returned by tool")
    result_useful: bool | None = Field(
        default=None,
        description="Whether results contributed to final answer (None=unknown)",
    )
    tool_category: str = Field(
        default="",
        description="Tool category: retrieval, sparql, validation, data_access, other",
    )


class ToolUsageMetrics(BaseModel):
    """
    Aggregated tool usage metrics for a specific tool within an agent.

    Enables analysis like: "grep_classes was called 5 times by grep_retrieval_agent,
    averaging 120ms per call with a 95% success rate and returned 42 results."
    """

    tool_name: str = Field(..., description="Name of the tool")
    agent: str = Field(..., description="Agent that used this tool")
    tool_category: str = Field(default="", description="Tool category")
    call_count: int = Field(default=0, description="Number of invocations")
    total_duration_ms: float = Field(default=0.0, description="Total time in this tool")
    avg_duration_ms: float = Field(default=0.0, description="Average time per call")
    min_duration_ms: float = Field(default=0.0, description="Fastest invocation")
    max_duration_ms: float = Field(default=0.0, description="Slowest invocation")
    success_count: int = Field(default=0, description="Successful invocations")
    error_count: int = Field(default=0, description="Failed invocations")
    success_rate: float = Field(default=1.0, description="Success rate (0.0-1.0)")
    # Effectiveness metrics
    total_results: int = Field(default=0, description="Total results across all invocations")
    avg_results_per_call: float = Field(default=0.0, description="Average results per call")
    useful_count: int = Field(default=0, description="Invocations marked as useful")


class AgentTimingMetrics(BaseModel):
    """
    Detailed timing metrics for a specific agent.

    Enables analysis like: "sparql_generation_agent ran 3 times for 15.2s total,
    spending 8.1s in LLM calls and 4.3s in tool execution."
    """

    agent_name: str = Field(..., description="Name of the agent")
    phase: str = Field(
        default="", description="Phase this agent operates in (retrieval/generation)"
    )
    invocation_count: int = Field(default=0, description="Number of times invoked")
    total_duration_ms: float = Field(default=0.0, description="Total time in agent")
    avg_duration_ms: float = Field(default=0.0, description="Average per invocation")
    min_duration_ms: float = Field(default=0.0, description="Fastest invocation")
    max_duration_ms: float = Field(default=0.0, description="Slowest invocation")
    # Sub-timing breakdown
    llm_time_ms: float = Field(default=0.0, description="Time spent in LLM calls")
    tool_execution_time_ms: float = Field(
        default=0.0, description="Time spent executing tools"
    )
    # Context metrics
    total_llm_calls: int = Field(default=0, description="LLM calls made by this agent")
    total_tool_calls: int = Field(default=0, description="Tool calls made by this agent")
    avg_context_chars: int = Field(
        default=0, description="Average context size in characters"
    )


class DelegationMetrics(BaseModel):
    """
    Metrics for agent delegation (when one agent hands off to another).

    Tracks: SPARQL agent → MultiStep agent, SPARQL agent → MappingOptimizer, etc.
    """

    from_agent: str = Field(..., description="Agent that delegated")
    to_agent: str = Field(..., description="Agent that received delegation")
    delegation_count: int = Field(default=0, description="Number of delegations")
    total_duration_ms: float = Field(
        default=0.0, description="Total time in delegated agent"
    )
    avg_duration_ms: float = Field(default=0.0, description="Average delegation time")
    success_count: int = Field(default=0, description="Successful delegations")
    failure_count: int = Field(default=0, description="Failed delegations")
    reasons: list[str] = Field(
        default_factory=list, description="Delegation reasons encountered"
    )


class TimingBreakdown(BaseModel):
    """Detailed timing breakdown."""

    total_time_ms: float = Field(default=0.0, description="Total run time")
    retrieval_time_ms: float = Field(
        default=0.0, description="Total time in retrieval phases"
    )
    generation_time_ms: float = Field(
        default=0.0, description="Total time in generation phases"
    )
    llm_inference_time_ms: float = Field(
        default=0.0, description="Total LLM inference time"
    )
    sparql_execution_time_ms: float = Field(
        default=0.0, description="Total SPARQL execution time"
    )
    tool_execution_time_ms: float = Field(
        default=0.0, description="Total time spent executing tools (excluding LLM)"
    )

    # Flexible agent-level timing (any agent can be tracked)
    agent_timing: dict[str, float] = Field(
        default_factory=dict,
        description="Timing per agent: agent_name -> total_ms. Dynamically populated based on which agents run.",
    )


# =============================================================================
# Lightweight Trace Summary (for Dashboard)
# =============================================================================


class TraceSummary(BaseModel):
    """
    Lightweight trace summary for dashboard display (~2-5 KB per trace).

    Contains essential metrics and preview data for visualization.
    Full trace details can be loaded on-demand via the trace_file reference.
    """

    # Identification
    run_id: str = Field(..., description="Unique run identifier")
    query_id: str = Field(..., description="Query ID (e.g., 'A01', 'B05')")
    approach: Literal["agentic_grep", "agentic_semantic"] = Field(
        ..., description="Approach used"
    )
    llm_model: str = Field(..., description="LLM model used")
    run_number: int = Field(default=1, description="Run number (for repeated runs)")

    # Query info for display
    query_text: str = Field(default="", description="Natural language query")
    expected_sparql: str = Field(default="", description="Ground truth SPARQL")

    # Essential metrics
    f1_score: float | None = Field(default=None, description="F1 score")
    precision: float | None = Field(default=None, description="Precision")
    recall: float | None = Field(default=None, description="Recall")
    execution_accuracy: float | None = Field(
        default=None, description="1.0 if exact match"
    )
    true_positives: int = Field(default=0, description="Number of matching results")
    actual_result_count: int = Field(default=0, description="Number of results returned")
    expected_result_count: int = Field(default=0, description="Number of expected results")

    # Minimal timing/cost info
    total_time_ms: float = Field(default=0.0, description="Total run time")
    total_tokens: int = Field(default=0, description="Total tokens used")

    # Status
    success: bool = Field(default=False, description="Whether run succeeded")
    is_unanswerable: bool = Field(default=False, description="Marked as unanswerable")

    # Generated output preview (first 5 results only)
    final_sparql: str | None = Field(default=None, description="Final SPARQL query")
    results_sample: list[dict] = Field(
        default_factory=list, description="First 5 result rows"
    )

    # File reference for full details
    trace_file: str = Field(default="", description="Filename of full trace JSON")

    # Tool/Agent summary (NEW - key metrics for quick analysis)
    total_tool_calls: int = Field(default=0, description="Total tool invocations")
    unique_tools_used: int = Field(default=0, description="Distinct tools used")
    total_delegation_count: int = Field(default=0, description="Agent delegations")
    # Per-agent time summary (top-level for quick access)
    retrieval_agent_time_ms: float = Field(
        default=0.0, description="Time in retrieval agent"
    )
    generation_agent_time_ms: float = Field(
        default=0.0, description="Time in generation agents"
    )


# =============================================================================
# Complete Experiment Run Trace
# =============================================================================


class ExperimentRunTrace(BaseModel):
    """
    Complete trace of a single experiment run.

    This is the top-level model that captures everything about one
    query evaluation run (one query + one approach + one LLM model).
    """

    # Identification
    run_id: str = Field(..., description="Unique run identifier")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Configuration
    approach: Literal["agentic_grep", "agentic_semantic"] = Field(
        ..., description="Approach used (determines retrieval strategy)"
    )
    llm_model: str = Field(..., description="LLM model used")
    run_number: int = Field(default=1, description="Run number (for repeated runs)")

    # Query metadata (for correlation analysis)
    query_metadata: QueryMetadata = Field(
        ..., description="Metadata about the query being evaluated"
    )

    # Phase traces (chronological)
    retrieval_phases: list[RetrievalPhaseTrace] = Field(
        default_factory=list, description="All retrieval phase executions"
    )
    generation_phases: list[GenerationPhaseTrace] = Field(
        default_factory=list, description="All generation phase executions"
    )

    # Retrieved triples tracking (for retrieval quality analysis)
    all_retrieved_triples: list[str] = Field(
        default_factory=list,
        description="All unique triples retrieved across all retrieval phases",
    )
    final_schema_triples: list[str] = Field(
        default_factory=list,
        description="Final triples passed to SPARQL generation (after filtering/selection)",
    )

    # Final output
    final_sparql: str | None = Field(
        default=None, description="Final selected SPARQL query"
    )
    final_results: list[dict] = Field(
        default_factory=list, description="Final query results"
    )
    final_result_count: int = Field(default=0, description="Number of final results")

    # Ground truth for comparison
    ground_truth_query_id: str | None = Field(
        default=None,
        description="Query ID for looking up ground truth from global cache",
    )

    # Termination status
    success: bool = Field(default=False, description="Whether run completed successfully")
    is_unanswerable: bool = Field(
        default=False, description="Query marked as unanswerable"
    )
    unanswerable_reason: str = Field(default="", description="Why query is unanswerable")
    error_message: str | None = Field(
        default=None, description="Error message if run failed"
    )

    # Aggregated metrics
    result_metrics: ResultMetrics | None = Field(
        default=None, description="Data-based evaluation metrics"
    )
    token_usage: TokenUsage = Field(
        default_factory=TokenUsage, description="Aggregated token usage"
    )
    timing: TimingBreakdown = Field(
        default_factory=TimingBreakdown, description="Timing breakdown"
    )

    # Agent behavior metrics
    total_iterations: int = Field(
        default=0, description="Total iterations through phases"
    )
    retrieval_iterations: int = Field(
        default=0, description="Number of retrieval phase executions"
    )
    generation_iterations: int = Field(
        default=0, description="Number of generation phase executions"
    )
    total_llm_calls: int = Field(default=0, description="Total LLM calls made")
    total_sparql_executions: int = Field(
        default=0, description="Total SPARQL queries executed"
    )

    # Strategy documentation (for transparency)
    strategy_used: Literal["grep", "semantic"] = Field(
        ..., description="Retrieval strategy used (derived from approach)"
    )

    # ==========================================================================
    # Detailed Tool and Agent Tracking (NEW - for scientific evaluation)
    # ==========================================================================

    # Tool usage per agent: maps agent_name -> list of tool metrics
    tool_usage_by_agent: dict[str, list[ToolUsageMetrics]] = Field(
        default_factory=dict,
        description="Tool usage metrics grouped by agent name",
    )

    # All individual tool invocations (chronological)
    tool_invocations: list[ToolInvocation] = Field(
        default_factory=list,
        description="All tool invocations in chronological order",
    )

    # Per-agent detailed timing
    agent_timing_details: list[AgentTimingMetrics] = Field(
        default_factory=list,
        description="Detailed timing metrics for each agent",
    )

    # Delegation tracking
    delegation_metrics: list[DelegationMetrics] = Field(
        default_factory=list,
        description="Metrics for inter-agent delegations",
    )

    # Summary counts for quick access
    total_tool_calls: int = Field(
        default=0, description="Total number of tool invocations"
    )
    unique_tools_used: int = Field(
        default=0, description="Number of distinct tools used"
    )
    total_delegation_count: int = Field(
        default=0, description="Total number of agent delegations"
    )

    def to_summary(self, trace_file: str = "") -> "TraceSummary":
        """
        Create a lightweight summary for dashboard display.

        Args:
            trace_file: Filename of the full trace JSON (set by save_trace)

        Returns:
            TraceSummary with essential metrics and preview data (~2-5 KB)
        """
        # Get first 5 results for preview
        results_sample = []
        if self.final_results and isinstance(self.final_results, list):
            results_sample = self.final_results[:5]

        return TraceSummary(
            run_id=self.run_id,
            query_id=self.query_metadata.query_id,
            approach=self.approach,
            llm_model=self.llm_model,
            run_number=self.run_number,
            # Query info
            query_text=self.query_metadata.query_text,
            expected_sparql=self.query_metadata.expected_sparql,
            # Metrics
            f1_score=self.result_metrics.f1_score if self.result_metrics else None,
            precision=self.result_metrics.precision if self.result_metrics else None,
            recall=self.result_metrics.recall if self.result_metrics else None,
            execution_accuracy=(
                self.result_metrics.execution_accuracy if self.result_metrics else None
            ),
            true_positives=(
                self.result_metrics.true_positives if self.result_metrics else 0
            ),
            actual_result_count=(
                self.result_metrics.actual_result_count if self.result_metrics else 0
            ),
            expected_result_count=(
                self.result_metrics.expected_result_count if self.result_metrics else 0
            ),
            # Timing
            total_time_ms=self.timing.total_time_ms,
            total_tokens=self.token_usage.total_tokens,
            # Status
            success=self.success,
            is_unanswerable=self.is_unanswerable,
            # Output preview
            final_sparql=self.final_sparql,
            results_sample=results_sample,
            trace_file=trace_file,
            # Tool/Agent summary
            total_tool_calls=self.total_tool_calls,
            unique_tools_used=self.unique_tools_used,
            total_delegation_count=self.total_delegation_count,
            # Calculate agent timing from flexible dict
            retrieval_agent_time_ms=sum(
                t for name, t in self.timing.agent_timing.items()
                if "retrieval" in name.lower()
            ),
            generation_agent_time_ms=sum(
                t for name, t in self.timing.agent_timing.items()
                if any(x in name.lower() for x in ["sparql", "generation", "multi_step", "mapping"])
            ),
        )


# =============================================================================
# Experiment Batch (multiple runs)
# =============================================================================


class ExperimentBatch(BaseModel):
    """
    A batch of experiment runs for statistical analysis.

    Groups multiple runs of the same query/approach/model combination.
    """

    # Identification
    batch_id: str = Field(..., description="Unique batch identifier")
    experiment_name: str = Field(..., description="Name of the experiment")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Configuration
    query_id: str = Field(..., description="Query being evaluated")
    approach: Literal["agentic_grep", "agentic_semantic"] = Field(
        ..., description="Approach used"
    )
    llm_model: str = Field(..., description="LLM model used")
    runs_per_query: int = Field(default=5, description="Number of runs for statistics")

    # Individual runs
    runs: list[ExperimentRunTrace] = Field(
        default_factory=list, description="All runs in this batch"
    )

    # Aggregated statistics (mean ± std)
    mean_f1_score: float = Field(default=0.0)
    std_f1_score: float = Field(default=0.0)
    mean_execution_accuracy: float = Field(default=0.0)
    std_execution_accuracy: float = Field(default=0.0)
    mean_total_time_ms: float = Field(default=0.0)
    std_total_time_ms: float = Field(default=0.0)
    mean_total_tokens: float = Field(default=0.0)
    std_total_tokens: float = Field(default=0.0)

    # Success rate
    success_rate: float = Field(default=0.0, description="Proportion of successful runs")
    unanswerable_rate: float = Field(
        default=0.0, description="Proportion marked unanswerable"
    )


__all__ = [
    # Metadata
    "QueryMetadata",
    # Traces
    "LLMCallTrace",
    "SPARQLExecutionTrace",
    "RetrievalPhaseTrace",
    "GenerationPhaseTrace",
    # Metrics
    "ResultMetrics",
    "TokenUsage",
    "TimingBreakdown",
    # Tool and Agent tracking
    "ToolInvocation",
    "ToolUsageMetrics",
    "AgentTimingMetrics",
    "DelegationMetrics",
    # Lightweight summary
    "TraceSummary",
    # Top-level
    "ExperimentRunTrace",
    "ExperimentBatch",
]