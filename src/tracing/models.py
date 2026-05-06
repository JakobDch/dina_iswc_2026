"""Pydantic models for tracing events."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TraceEventType(str, Enum):
    """Types of trace events."""

    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ORCHESTRATOR_DECISION = "orchestrator_decision"
    SPARQL_EXECUTION = "sparql_execution"
    SPARQL_RESULT = "sparql_result"
    ERROR = "error"
    INFO = "info"  # Informational messages (timeouts, warnings, progress)
    QUERY_START = "query_start"
    QUERY_COMPLETE = "query_complete"
    AGENT_ACTION = "agent_action"
    SPARQL_AGENT_DECISION = "sparql_agent_decision"  # SPARQL agent DONE/NEED_TRIPLE/MULTI_STEP_RESULT decision
    # Agent-level timing events
    AGENT_START = "agent_start"  # Agent starts processing (sparql_agent, multi_step_agent, retrieval_agent)
    AGENT_END = "agent_end"  # Agent finishes processing with duration_ms
    # Detailed debugging events
    SCHEMA_CONTEXT = "schema_context"  # Schema triples passed to SPARQL agent
    SPARQL_QUERY_TEST = "sparql_query_test"  # Full SPARQL query being tested with results
    MULTI_STEP_INPUT = "multi_step_input"  # Input to multi-step agent
    MULTI_STEP_STEP = "multi_step_step"  # Individual step execution with query and data sample
    MULTI_STEP_TRANSFORM = "multi_step_transform"  # Transformation script and result sample
    # Agent context visualization events
    AGENT_MESSAGE = "agent_message"  # Individual LLM message (SystemMessage/HumanMessage/AIMessage/ToolMessage)
    DELEGATION_START = "delegation_start"  # Control passed to another agent
    DELEGATION_END = "delegation_end"  # Control returned from delegated agent


class TraceEvent(BaseModel):
    """A single trace event."""

    event_type: TraceEventType
    timestamp: datetime = Field(default_factory=datetime.now)
    phase: str | None = None
    agent: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float | None = None
    run_id: str | None = Field(
        default=None,
        description="Unique run identifier for filtering events in parallel execution"
    )
