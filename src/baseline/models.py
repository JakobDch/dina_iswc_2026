"""
Pydantic and SQLModel models for baseline pipeline.

Simplified version for experiment - no web app models needed.
"""

from sqlmodel import SQLModel, Field
from pydantic import BaseModel, model_validator
from datetime import datetime, timezone
from typing import Optional, List, Literal, Dict, Any
from uuid import uuid4


# --- Search Query Model ---
class SearchQuery(SQLModel):
    query_text: str
    k: int = Field(default=30, gt=0)
    min_model_hits: int = Field(default=3, gt=0)
    candidate_margin: float = Field(default=0.1, ge=0.0, le=1.0)
    top_n_for_avg: int = Field(default=3, gt=0)

    @model_validator(mode="after")
    def check_top_n_for_avg(self):
        if self.top_n_for_avg > self.min_model_hits:
            pass
        return self


# --- API Request/Response Models ---

class ValidateSparqlRequest(SQLModel):
    query: str
    workspace_id: str


class ExecuteAndCompareRequest(SQLModel):
    sparql_query: str
    template_id: str
    workspace_id: str
    model_info_blocks: Optional[str] = None
    model_check_hints: Optional[str] = None


class FixSyntaxRequest(SQLModel):
    generated_query: str
    groundtruth_query: str
    syntax_error_details: str
    workspace_id: str
    llm_profile: Optional[str] = "deepseek_chat"


class DirectSparqlExecutionRequest(SQLModel):
    workspace_id: str
    sparql_query: str


class RequestDataForValidation(SQLModel):
    message: str
    workspace_id: str
    template_id: Optional[str] = None


# --- Benchmark Models ---

class BenchmarkRequest(SQLModel):
    llm_profile: str
    num_runs: int = Field(default=1, gt=0)
    benchmark_mode: str = Field(default="normal")  # "normal" or "sparql_only"
    agentic_reasoning_enabled: bool = False
    few_shot_prompting_enabled: bool = False
    adaptive_few_shot_enabled: bool = False
    rate_limiting_enabled: bool = False


class BenchmarkResumeRequest(SQLModel):
    """Request model for resuming a partially completed benchmark"""
    llm_profile: Optional[str] = None
    num_runs: Optional[int] = None
    benchmark_mode: Optional[str] = None
    agentic_reasoning_enabled: Optional[bool] = None
    few_shot_prompting_enabled: Optional[bool] = None
    adaptive_few_shot_enabled: Optional[bool] = None
    rate_limiting_enabled: Optional[bool] = None


# --- Pipeline Continuation Model ---
class ContinuePipelineRequest(SQLModel):
    original_user_query: str
    user_response: str
    llm_profile: str
    validated_models_json: str
    initial_llm_reasoning: str
    clarification_question: str
    workspace_id: str
    agentic_reasoning_enabled: bool = False
    few_shot_prompting_enabled: bool = False
    adaptive_few_shot_enabled: bool = False


# --- LLM Response Model ---
class LLMResponse(BaseModel):
    """Structured response from LLM including content and token usage."""
    content: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


# --- Experiment-specific models ---

class ExperimentQuery(BaseModel):
    """A single query for the experiment."""
    id: str
    text: str  # Natural language query
    expected_sparql: Optional[str] = None  # Ground truth SPARQL
    dataset: str = "unknown"  # Which dataset this query is for
    difficulty: Optional[str] = None  # easy, medium, hard
    tags: List[str] = []


class ExperimentResult(BaseModel):
    """Result of running a single query through an approach."""
    query_id: str
    approach: str  # "baseline", "agentic_grep", "agentic_semantic"
    llm_model: str
    run_number: int
    generated_sparql: str
    execution_result: Optional[Dict[str, Any]] = None
    execution_success: bool = False
    execution_time_ms: float = 0.0
    token_usage: Dict[str, int] = {}
    reasoning: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExperimentMetrics(BaseModel):
    """Aggregated metrics for an experiment."""
    approach: str
    llm_model: str
    total_queries: int
    successful_executions: int
    execution_accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_execution_time_ms: float
    total_tokens: int
    avg_tokens_per_query: float
