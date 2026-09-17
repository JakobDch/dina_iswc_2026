"""Tracing package for DINA agent debugging and experiment tracing."""

from src.tracing.console import RichTraceConsole
from src.tracing.models import TraceEvent, TraceEventType
from src.tracing.tracer import (
    TracingManager,
    get_tracer,
    reset_tracer,
    trace_run_context,
    get_current_run_id,
    set_current_run_id,
)
from src.tracing.experiment_trace import (
    # Metadata
    QueryMetadata,
    # Traces
    LLMCallTrace,
    SPARQLExecutionTrace,
    RetrievalPhaseTrace,
    GenerationPhaseTrace,
    # Metrics
    ResultMetrics,
    TokenUsage,
    TimingBreakdown,
    # Tool and Agent tracking
    ToolInvocation,
    ToolUsageMetrics,
    AgentTimingMetrics,
    DelegationMetrics,
    # Lightweight summary
    TraceSummary,
    # Top-level
    ExperimentRunTrace,
    ExperimentBatch,
)
from src.tracing.collector import (
    ExperimentTraceCollector,
    convert_use_case_query_to_metadata,
    create_experiment_batch,
)
from src.tracing.context_collector import (
    AgentContextCollector,
    AgentContextSnapshot,
    AgentTimelineTrace,
    DelegationEvent,
    save_context_trace,
)

__all__ = [
    # Existing
    "TraceEvent",
    "TraceEventType",
    "TracingManager",
    "get_tracer",
    "reset_tracer",
    "trace_run_context",
    "get_current_run_id",
    "set_current_run_id",
    "RichTraceConsole",
    # Experiment tracing models
    "QueryMetadata",
    "LLMCallTrace",
    "SPARQLExecutionTrace",
    "RetrievalPhaseTrace",
    "GenerationPhaseTrace",
    "ResultMetrics",
    "TokenUsage",
    "TimingBreakdown",
    # Tool and Agent tracking
    "ToolInvocation",
    "ToolUsageMetrics",
    "AgentTimingMetrics",
    "DelegationMetrics",
    # Summary and top-level
    "TraceSummary",
    "ExperimentRunTrace",
    "ExperimentBatch",
    # Collector
    "ExperimentTraceCollector",
    "convert_use_case_query_to_metadata",
    "create_experiment_batch",
    # Context visualization
    "AgentContextCollector",
    "AgentContextSnapshot",
    "AgentTimelineTrace",
    "DelegationEvent",
    "save_context_trace",
]
