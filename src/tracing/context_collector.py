"""
Agent Context Collector for visualization.

Collects trace events organized by agent timeline for the
agent context visualization dashboard.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.tracing.models import TraceEvent, TraceEventType
from src.tracing.tracer import TraceListener


class AgentContextSnapshot(BaseModel):
    """Snapshot of an agent's context at a point in time."""

    timestamp: str
    event_type: str  # "message", "waiting", "active"

    # Message content (for message events)
    message_type: str | None = None  # "SystemMessage", "HumanMessage", "AIMessage", "ToolMessage"
    content: str | None = None
    tool_calls: list[dict] | None = None  # For AIMessage with tool_calls
    tool_call_id: str | None = None  # For ToolMessage
    tool_name: str | None = None  # For ToolMessage

    # State
    is_active: bool = False
    duration_ms: float | None = None


class DelegationEvent(BaseModel):
    """Represents control passing between agents."""

    timestamp: str
    from_agent: str
    to_agent: str
    reason: str | None = None


class AgentTimelineTrace(BaseModel):
    """Complete timeline for visualization."""

    run_id: str
    query_text: str = ""
    approach: str = ""
    start_time: str = Field(default_factory=lambda: datetime.now().isoformat())
    end_time: str | None = None

    # Agent timelines - ordered list of context snapshots per agent
    retrieval_timeline: list[AgentContextSnapshot] = Field(default_factory=list)
    sparql_generation_timeline: list[AgentContextSnapshot] = Field(default_factory=list)
    multi_step_timeline: list[AgentContextSnapshot] = Field(default_factory=list)
    mapping_optimizer_timeline: list[AgentContextSnapshot] = Field(default_factory=list)

    # Delegation events (for drawing arrows)
    delegations: list[DelegationEvent] = Field(default_factory=list)


# Agent name mapping from trace events to timeline keys
AGENT_NAME_MAP = {
    "grep_retrieval_agent": "retrieval",
    "semantic_retrieval_agent": "retrieval",
    "retrieval_agent": "retrieval",
    "sparql_agent": "sparql_generation",
    "sparql_generation_agent": "sparql_generation",
    "multi_step_agent": "multi_step",
    "mapping_optimizer_agent": "mapping_optimizer",
    "mapping_optimizer": "mapping_optimizer",
}


class AgentContextCollector(TraceListener):
    """
    Collects trace events for agent context visualization.

    Implements the TraceListener protocol and organizes events by agent
    for the 4-column visualization dashboard.

    Usage:
        collector = AgentContextCollector(run_id="my-run-123")
        tracer = get_tracer()
        tracer.add_listener(collector)

        # ... run experiment ...

        timeline_data = collector.to_dict()
        tracer.remove_listener(collector)
    """

    def __init__(self, run_id: str, query_text: str = "", approach: str = ""):
        """
        Initialize the collector.

        Args:
            run_id: Unique identifier for this run (for filtering events)
            query_text: The natural language query being processed
            approach: The approach used ('agentic_grep' or 'agentic_semantic')
        """
        self.run_id = run_id
        self.query_text = query_text
        self.approach = approach
        self.start_time = datetime.now()
        self.end_time: datetime | None = None

        # Timelines per agent
        self.timelines: dict[str, list[AgentContextSnapshot]] = {
            "retrieval": [],
            "sparql_generation": [],
            "multi_step": [],
            "mapping_optimizer": [],
        }

        # Delegation tracking
        self.delegations: list[DelegationEvent] = []

        # Track current active agent
        self.active_agent: str | None = None
        self.last_active_agent: str | None = None

    def _get_agent_key(self, agent_name: str | None) -> str | None:
        """Map agent name from trace event to timeline key."""
        if agent_name is None:
            return None
        return AGENT_NAME_MAP.get(agent_name, agent_name)

    def _mark_others_waiting(self, active_agent: str, timestamp: datetime) -> None:
        """Mark all other agents as waiting when one becomes active."""
        for agent_key in self.timelines:
            if agent_key != active_agent:
                # Only add waiting marker if the agent has content and isn't already waiting
                timeline = self.timelines[agent_key]
                if timeline and timeline[-1].event_type != "waiting":
                    self.timelines[agent_key].append(AgentContextSnapshot(
                        timestamp=timestamp.isoformat(),
                        event_type="waiting",
                        is_active=False,
                    ))

    def on_event(self, event: TraceEvent) -> None:
        """Handle a trace event."""
        # Filter by run_id - ignore events from other runs
        if event.run_id is not None and event.run_id != self.run_id:
            return

        agent_key = self._get_agent_key(event.agent)

        # Handle agent start - marks agent as active
        if event.event_type == TraceEventType.AGENT_START:
            if agent_key and agent_key in self.timelines:
                # Record delegation if switching agents
                if self.active_agent and self.active_agent != agent_key:
                    self.delegations.append(DelegationEvent(
                        timestamp=event.timestamp.isoformat(),
                        from_agent=self.active_agent,
                        to_agent=agent_key,
                        reason=event.data.get("reason"),
                    ))

                self.last_active_agent = self.active_agent
                self.active_agent = agent_key
                self._mark_others_waiting(agent_key, event.timestamp)

        # Handle agent end
        elif event.event_type == TraceEventType.AGENT_END:
            if agent_key and agent_key in self.timelines:
                # If returning from delegated agent, restore previous
                if self.last_active_agent and agent_key != self.last_active_agent:
                    self.delegations.append(DelegationEvent(
                        timestamp=event.timestamp.isoformat(),
                        from_agent=agent_key,
                        to_agent=self.last_active_agent,
                        reason="completed",
                    ))

        # Handle agent messages (the key event for context visualization)
        elif event.event_type == TraceEventType.AGENT_MESSAGE:
            if agent_key and agent_key in self.timelines:
                self.timelines[agent_key].append(AgentContextSnapshot(
                    timestamp=event.timestamp.isoformat(),
                    event_type="message",
                    message_type=event.data.get("message_type"),
                    content=event.data.get("content"),
                    tool_calls=event.data.get("tool_calls"),
                    tool_call_id=event.data.get("tool_call_id"),
                    tool_name=event.data.get("tool_name"),
                    is_active=True,
                ))

        # Handle delegation events
        elif event.event_type == TraceEventType.DELEGATION_START:
            from_agent = self._get_agent_key(event.data.get("from_agent"))
            to_agent = self._get_agent_key(event.data.get("to_agent"))
            if from_agent and to_agent:
                self.delegations.append(DelegationEvent(
                    timestamp=event.timestamp.isoformat(),
                    from_agent=from_agent,
                    to_agent=to_agent,
                    reason=event.data.get("reason"),
                ))
                self.active_agent = to_agent
                self._mark_others_waiting(to_agent, event.timestamp)

        elif event.event_type == TraceEventType.DELEGATION_END:
            from_agent = self._get_agent_key(event.data.get("from_agent"))
            to_agent = self._get_agent_key(event.data.get("to_agent"))
            if from_agent and to_agent:
                self.delegations.append(DelegationEvent(
                    timestamp=event.timestamp.isoformat(),
                    from_agent=from_agent,
                    to_agent=to_agent,
                    reason="returned",
                ))

        # Handle tool calls - add to current active agent's timeline
        elif event.event_type == TraceEventType.TOOL_CALL:
            if agent_key and agent_key in self.timelines:
                self.timelines[agent_key].append(AgentContextSnapshot(
                    timestamp=event.timestamp.isoformat(),
                    event_type="tool_call",
                    tool_name=event.data.get("tool_name"),
                    content=str(event.data.get("args", {}))[:500],  # Truncate args
                    is_active=True,
                ))

        # Handle tool results
        elif event.event_type == TraceEventType.TOOL_RESULT:
            if agent_key and agent_key in self.timelines:
                result_preview = str(event.data.get("result", ""))[:500]
                self.timelines[agent_key].append(AgentContextSnapshot(
                    timestamp=event.timestamp.isoformat(),
                    event_type="tool_result",
                    tool_name=event.data.get("tool_name"),
                    content=result_preview,
                    is_active=True,
                ))

        # Handle query start - capture query info
        elif event.event_type == TraceEventType.QUERY_START:
            if not self.query_text:
                self.query_text = event.data.get("query", "")
            if not self.approach:
                self.approach = event.data.get("approach", "")

        # Handle query complete
        elif event.event_type == TraceEventType.QUERY_COMPLETE:
            self.end_time = event.timestamp

    def finalize(self) -> AgentTimelineTrace:
        """Finalize and return the collected timeline trace."""
        if self.end_time is None:
            self.end_time = datetime.now()

        return AgentTimelineTrace(
            run_id=self.run_id,
            query_text=self.query_text,
            approach=self.approach,
            start_time=self.start_time.isoformat(),
            end_time=self.end_time.isoformat() if self.end_time else None,
            retrieval_timeline=self.timelines["retrieval"],
            sparql_generation_timeline=self.timelines["sparql_generation"],
            multi_step_timeline=self.timelines["multi_step"],
            mapping_optimizer_timeline=self.timelines["mapping_optimizer"],
            delegations=self.delegations,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.finalize().model_dump()


def save_context_trace(output_dir, run_id: str, context_data: dict) -> None:
    """
    Save context trace to a JSON file.

    Args:
        output_dir: Directory to save the trace file
        run_id: The run identifier
        context_data: The context data dictionary
    """
    import json
    from pathlib import Path

    output_path = Path(output_dir)
    context_traces_dir = output_path / "context_traces"
    context_traces_dir.mkdir(parents=True, exist_ok=True)

    trace_file = context_traces_dir / f"{run_id}_context.json"
    with open(trace_file, "w", encoding="utf-8") as f:
        json.dump(context_data, f, indent=2, ensure_ascii=False)


__all__ = [
    "AgentContextSnapshot",
    "DelegationEvent",
    "AgentTimelineTrace",
    "AgentContextCollector",
    "save_context_trace",
    "AGENT_NAME_MAP",
]
