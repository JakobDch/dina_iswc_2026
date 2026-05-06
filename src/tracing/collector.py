"""
Trace collector for experiment runs.

This module provides a TraceCollector that implements the TraceListener protocol
and collects all events during an experiment run into the structured
ExperimentRunTrace format.
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from src.tracing.models import TraceEvent, TraceEventType
from src.tracing.tracer import TraceListener
from src.tracing.experiment_trace import (
    # Models
    QueryMetadata,
    LLMCallTrace,
    SPARQLExecutionTrace,
    RetrievalPhaseTrace,
    GenerationPhaseTrace,
    ResultMetrics,
    TokenUsage,
    TimingBreakdown,
    ExperimentRunTrace,
    ExperimentBatch,
    # Tool and Agent tracking
    ToolInvocation,
    ToolUsageMetrics,
    AgentTimingMetrics,
    DelegationMetrics,
)

if TYPE_CHECKING:
    from data.queries.use_case_queries import UseCaseQuery
    from data.queries.use_case_queries_tiered_tuples import AdaptiveGroundTruth


def convert_use_case_query_to_metadata(query: "UseCaseQuery | AdaptiveGroundTruth") -> QueryMetadata:
    """
    Convert an AdaptiveGroundTruth dataclass to QueryMetadata Pydantic model.

    This extracts the relevant fields for experiment tracing.
    """
    # Handle id/query_id difference between UseCaseQuery and AdaptiveGroundTruth
    query_id = getattr(query, 'id', None) or getattr(query, 'query_id', 'unknown')

    return QueryMetadata(
        query_id=query_id,
        query_set=query.query_set,
        query_text=query.query,
        dataset=query.dataset,
        datasets=list(query.datasets) if query.datasets else [query.dataset],
        expected_sparql=getattr(query, 'expected_sparql', '') or getattr(query, 'sparql_query', ''),
        expected_results_count=getattr(query, 'expected_results_count', None),
    )


class ExperimentTraceCollector(TraceListener):
    """
    Collects trace events during an experiment run.

    Implements the TraceListener protocol and builds up an ExperimentRunTrace
    as events are received.

    Usage:
        collector = ExperimentTraceCollector(
            query_metadata=metadata,
            approach="agentic_semantic",
            llm_model="gpt-4o",
        )
        tracer = get_tracer()
        tracer.add_listener(collector)

        # ... run experiment ...

        trace = collector.finalize()
        tracer.remove_listener(collector)
    """

    def __init__(
        self,
        query_metadata: QueryMetadata,
        approach: str,
        llm_model: str,
        run_number: int = 1,
    ):
        """
        Initialize the collector.

        Args:
            query_metadata: Metadata about the query being evaluated
            approach: The approach being used ('agentic_grep' or 'agentic_semantic')
            llm_model: The LLM model being used
            run_number: Run number (for repeated runs)
        """
        self.run_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now()
        self.query_metadata = query_metadata
        self.approach = approach
        self.llm_model = llm_model
        self.run_number = run_number

        # Derive strategy from approach
        self.strategy = "grep" if approach == "agentic_grep" else "semantic"

        # Current phase tracking
        self._current_phase: str | None = None
        self._current_phase_start: datetime | None = None
        self._current_retrieval_trace: RetrievalPhaseTrace | None = None
        self._current_generation_trace: GenerationPhaseTrace | None = None

        # Collected traces
        self.retrieval_phases: list[RetrievalPhaseTrace] = []
        self.generation_phases: list[GenerationPhaseTrace] = []

        # Aggregated data
        self.all_llm_calls: list[LLMCallTrace] = []
        self.all_sparql_executions: list[SPARQLExecutionTrace] = []

        # Final output
        self.final_sparql: str | None = None
        self.final_results: list[dict] = []
        self.ground_truth_query_id: str | None = None  # Reference to global cache
        self.success = False
        self.is_unanswerable = False
        self.unanswerable_reason = ""
        self.error_message: str | None = None

        # Retrieved triples tracking
        self.all_retrieved_triples: set[str] = set()  # Unique triples across all phases
        self.final_schema_triples: list[str] = []  # Triples used for final generation

        # Agent-level timing tracking
        self._agent_start_times: dict[str, datetime] = {}  # agent_name -> start time
        self.agent_timing: dict[str, float] = {}  # Flexible: agent_name -> total_ms

        # =====================================================================
        # Detailed tool and agent tracking
        # =====================================================================

        # Track current agent for tool attribution
        self._current_agent: str | None = None

        # Tool invocations (chronological, for detailed analysis)
        self.tool_invocations: list[ToolInvocation] = []

        # Per-agent timing details: agent_name -> list of (start_time, end_time, llm_time, tool_time)
        self._agent_invocation_records: dict[str, list[dict]] = {}

        # Track tool calls being made (pending completion)
        self._pending_tool_calls: dict[str, dict] = {}  # call_id -> {tool_name, agent, start_time}

        # Delegation tracking
        self._delegation_records: list[dict] = []  # All delegation events
        self._active_delegations: dict[str, dict] = {}  # to_agent -> {from_agent, start_time, reason}

        # Tool execution time accumulator
        self.total_tool_execution_time_ms: float = 0.0

    def on_event(self, event: TraceEvent) -> None:
        """Handle a trace event.

        Events are filtered by run_id to prevent cross-contamination
        during parallel execution. Only events with matching run_id
        (or no run_id for backwards compatibility) are processed.
        """
        # Filter by run_id to prevent cross-contamination in parallel execution
        if event.run_id is not None and event.run_id != self.run_id:
            return  # Ignore events from other runs

        if event.event_type == TraceEventType.PHASE_START:
            self._handle_phase_start(event)
        elif event.event_type == TraceEventType.PHASE_END:
            self._handle_phase_end(event)
        elif event.event_type == TraceEventType.TOOL_CALL:
            self._handle_tool_call(event)
        elif event.event_type == TraceEventType.TOOL_RESULT:
            self._handle_tool_result(event)
        elif event.event_type == TraceEventType.ORCHESTRATOR_DECISION:
            self._handle_orchestrator_decision(event)
        elif event.event_type == TraceEventType.SPARQL_EXECUTION:
            self._handle_sparql_execution(event)
        elif event.event_type == TraceEventType.SPARQL_RESULT:
            self._handle_sparql_result(event)
        elif event.event_type == TraceEventType.ERROR:
            self._handle_error(event)
        elif event.event_type == TraceEventType.QUERY_COMPLETE:
            self._handle_query_complete(event)
        elif event.event_type == TraceEventType.AGENT_START:
            self._handle_agent_start(event)
        elif event.event_type == TraceEventType.AGENT_END:
            self._handle_agent_end(event)
        elif event.event_type == TraceEventType.AGENT_MESSAGE:
            self._handle_agent_message(event)
        elif event.event_type == TraceEventType.DELEGATION_START:
            self._handle_delegation_start(event)
        elif event.event_type == TraceEventType.DELEGATION_END:
            self._handle_delegation_end(event)

    def _handle_phase_start(self, event: TraceEvent) -> None:
        """Handle phase start event."""
        phase = event.phase

        # Finalize any open phase before starting a new one
        # This handles the case where PHASE_END events are not emitted
        self._finalize_current_phase(event.timestamp)

        self._current_phase = phase
        self._current_phase_start = event.timestamp

        if phase == "retrieval":
            self._current_retrieval_trace = RetrievalPhaseTrace(
                timestamp=event.timestamp,
                iteration=len(self.retrieval_phases) + 1,
                strategy=self.strategy,  # type: ignore
            )
        elif phase == "generation":
            self._current_generation_trace = GenerationPhaseTrace(
                timestamp=event.timestamp,
                iteration=len(self.generation_phases) + 1,
                execution_feedback=event.data.get("feedback", ""),
            )
            # Track schema triples used for generation (final triples for SPARQL agent)
            schema_triples = event.data.get("schema_triples", [])
            if schema_triples:
                self.final_schema_triples = schema_triples

    def _finalize_current_phase(self, end_time: datetime | None = None) -> None:
        """
        Finalize any currently open phase.

        This is called when a new phase starts or when finalize() is called,
        to ensure phases are properly closed even without explicit PHASE_END events.
        """
        if end_time is None:
            end_time = datetime.now()

        # Finalize open retrieval phase
        if self._current_retrieval_trace is not None:
            if self._current_phase_start:
                duration_ms = (end_time - self._current_phase_start).total_seconds() * 1000
                self._current_retrieval_trace.duration_ms = duration_ms
            self.retrieval_phases.append(self._current_retrieval_trace)
            self._current_retrieval_trace = None

        # Finalize open generation phase
        if self._current_generation_trace is not None:
            if self._current_phase_start:
                duration_ms = (end_time - self._current_phase_start).total_seconds() * 1000
                self._current_generation_trace.duration_ms = duration_ms
            # Calculate sub-timings
            llm_time = sum(
                c.duration_ms for c in self._current_generation_trace.llm_calls
            )
            sparql_time = sum(
                e.duration_ms for e in self._current_generation_trace.sparql_executions
            )
            self._current_generation_trace.llm_time_ms = llm_time
            self._current_generation_trace.sparql_time_ms = sparql_time
            self.generation_phases.append(self._current_generation_trace)
            self._current_generation_trace = None

        self._current_phase = None
        self._current_phase_start = None

    def _handle_phase_end(self, event: TraceEvent) -> None:
        """Handle phase end event."""
        phase = event.phase
        duration_ms = event.duration_ms or 0.0

        # Use the provided duration if available
        if phase == "retrieval" and self._current_retrieval_trace:
            self._current_retrieval_trace.duration_ms = duration_ms
        elif phase == "generation" and self._current_generation_trace:
            self._current_generation_trace.duration_ms = duration_ms

        # Finalize the current phase
        self._finalize_current_phase(event.timestamp)

    def _handle_tool_call(self, event: TraceEvent) -> None:
        """Handle tool call event (including LLM calls)."""
        data = event.data
        tool_name = data.get("tool", "")

        # Check if this is an LLM-related tool call
        if "llm" in tool_name.lower() or "chat" in tool_name.lower():
            llm_trace = LLMCallTrace(
                timestamp=event.timestamp,
                phase=self._current_phase or "unknown",  # type: ignore
                purpose=data.get("purpose", tool_name),
                model=self.llm_model,
                context_triples_count=data.get("context_triples_count", 0),
                context_char_count=data.get("context_char_count", 0),
            )
            self.all_llm_calls.append(llm_trace)

            # Add to current phase
            if self._current_retrieval_trace:
                self._current_retrieval_trace.llm_calls.append(llm_trace)
            elif self._current_generation_trace:
                self._current_generation_trace.llm_calls.append(llm_trace)

    def _handle_tool_result(self, event: TraceEvent) -> None:
        """Handle tool result event (including LLM responses with token counts)."""
        data = event.data

        # Update the last LLM call with token usage
        if self.all_llm_calls and data.get("tokens"):
            last_call = self.all_llm_calls[-1]
            tokens = data["tokens"]
            last_call.input_tokens = tokens.get("input", 0)
            last_call.output_tokens = tokens.get("output", 0)
            last_call.total_tokens = tokens.get("total", 0)
            last_call.duration_ms = data.get("duration_ms", 0.0)
            last_call.estimated_cost_usd = data.get("cost_usd", 0.0)

    def _handle_orchestrator_decision(self, event: TraceEvent) -> None:
        """Handle orchestrator decision event."""
        data = event.data
        decision = data.get("next_phase", "complete")
        reasoning = data.get("reasoning", "")

        # Update current phase trace with decision
        if self._current_retrieval_trace:
            self._current_retrieval_trace.triples_found = data.get("triples_found", 0)

            # Extract and store retrieved triples
            retrieved_triples = data.get("retrieved_triples", [])
            if retrieved_triples:
                self._current_retrieval_trace.retrieved_triples = retrieved_triples
                # Add to cumulative set of all retrieved triples
                for triple in retrieved_triples:
                    self.all_retrieved_triples.add(triple)

        elif self._current_generation_trace:
            self._current_generation_trace.orchestrator_decision = decision
            self._current_generation_trace.orchestrator_reasoning = reasoning
            self._current_generation_trace.selected_query_index = data.get(
                "selected_query_index", -1
            )

    def _handle_sparql_execution(self, event: TraceEvent) -> None:
        """Handle SPARQL execution event."""
        data = event.data
        # This is just a marker that execution started
        if self._current_generation_trace:
            self._current_generation_trace.queries_generated = data.get(
                "query_count", 0
            )

    def _handle_sparql_result(self, event: TraceEvent) -> None:
        """Handle SPARQL result event."""
        data = event.data

        execution_trace = SPARQLExecutionTrace(
            timestamp=event.timestamp,
            query=data.get("query", ""),
            query_index=data.get("index", 0),
            success=data.get("success", False),
            error_type=data.get("error_type"),
            error_message=data.get("error_message"),
            duration_ms=data.get("duration_ms", 0.0),
            timeout_seconds=data.get("timeout_seconds", 15.0),
            endpoint=data.get("endpoint", ""),
        )

        # Extract result count
        results = data.get("results", {})
        if isinstance(results, dict):
            bindings = results.get("bindings", [])
            execution_trace.result_count = len(bindings)
            execution_trace.results_sample = bindings[:5]

        self.all_sparql_executions.append(execution_trace)

        # Add to current generation phase
        if self._current_generation_trace:
            self._current_generation_trace.sparql_executions.append(execution_trace)

            # Update counters
            if execution_trace.success:
                self._current_generation_trace.queries_executed_successfully += 1
                if execution_trace.result_count and execution_trace.result_count > 0:
                    self._current_generation_trace.queries_with_results += 1
            else:
                self._current_generation_trace.queries_with_syntax_errors += 1

    def _handle_error(self, event: TraceEvent) -> None:
        """Handle error event."""
        data = event.data
        self.error_message = data.get("message", str(data))

    def _handle_query_complete(self, event: TraceEvent) -> None:
        """Handle query complete event."""
        data = event.data
        self.final_sparql = data.get("final_query")
        self.final_results = data.get("final_results", [])
        self.success = data.get("success", False)
        self.is_unanswerable = data.get("is_unanswerable", False)
        self.unanswerable_reason = data.get("unanswerable_reason", "")

    def _handle_agent_start(self, event: TraceEvent) -> None:
        """Handle agent start event for timing tracking."""
        agent_name = event.agent or event.data.get("agent", "unknown")
        self._agent_start_times[agent_name] = event.timestamp
        self._current_agent = agent_name

        # Initialize invocation record tracking for this agent
        if agent_name not in self._agent_invocation_records:
            self._agent_invocation_records[agent_name] = []

        # Start a new invocation record
        self._agent_invocation_records[agent_name].append({
            "start_time": event.timestamp,
            "end_time": None,
            "duration_ms": 0.0,
            "llm_time_ms": 0.0,
            "tool_time_ms": 0.0,
            "llm_calls": 0,
            "tool_calls": 0,
            "context_chars": 0,
        })

    def _handle_agent_end(self, event: TraceEvent) -> None:
        """Handle agent end event for timing tracking."""
        agent_name = event.agent or event.data.get("agent", "unknown")
        duration_ms = event.duration_ms or 0.0

        # If duration not provided, calculate from start time
        if duration_ms == 0.0 and agent_name in self._agent_start_times:
            start_time = self._agent_start_times[agent_name]
            duration_ms = (event.timestamp - start_time).total_seconds() * 1000

        # Accumulate time using flexible dict (any agent can be tracked)
        self.agent_timing[agent_name] = self.agent_timing.get(agent_name, 0.0) + duration_ms

        # Update the invocation record
        if agent_name in self._agent_invocation_records and self._agent_invocation_records[agent_name]:
            record = self._agent_invocation_records[agent_name][-1]
            record["end_time"] = event.timestamp
            record["duration_ms"] = duration_ms

        # Clean up
        self._agent_start_times.pop(agent_name, None)
        if self._current_agent == agent_name:
            self._current_agent = None

    def _handle_agent_message(self, event: TraceEvent) -> None:
        """
        Handle agent message event - extract tool calls from AIMessage.tool_calls.

        This is where we capture which specific tools are being called by each agent.
        """
        data = event.data
        agent_name = event.agent or self._current_agent or "unknown"
        message_type = data.get("message_type", "")

        # AIMessage contains tool_calls list
        if message_type == "AIMessage" and "tool_calls" in data:
            tool_calls = data.get("tool_calls", [])
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "unknown")
                call_id = tool_call.get("id", str(uuid.uuid4())[:8])
                args = tool_call.get("args", {})

                # Record the pending tool call (waiting for result)
                self._pending_tool_calls[call_id] = {
                    "tool_name": tool_name,
                    "agent": agent_name,
                    "start_time": event.timestamp,
                    "input_summary": str(args)[:200],  # Truncate for storage
                }

                # Update agent invocation record
                if agent_name in self._agent_invocation_records and self._agent_invocation_records[agent_name]:
                    self._agent_invocation_records[agent_name][-1]["tool_calls"] += 1

        # ToolMessage indicates tool execution completed
        elif message_type == "ToolMessage":
            tool_call_id = data.get("tool_call_id")
            content = data.get("content", "")

            if tool_call_id and tool_call_id in self._pending_tool_calls:
                pending = self._pending_tool_calls.pop(tool_call_id)
                duration_ms = (event.timestamp - pending["start_time"]).total_seconds() * 1000
                tool_name = pending["tool_name"]

                # Determine tool category
                tool_name_lower = tool_name.lower()
                if any(x in tool_name_lower for x in ["search", "grep", "find", "retrieve"]):
                    tool_category = "retrieval"
                elif any(x in tool_name_lower for x in ["sparql", "query", "execute"]):
                    tool_category = "sparql"
                elif any(x in tool_name_lower for x in ["validate", "check", "verify"]):
                    tool_category = "validation"
                elif any(x in tool_name_lower for x in ["sample", "instance", "data"]):
                    tool_category = "data_access"
                else:
                    tool_category = "other"

                # Try to extract result count from content
                result_count = 0
                if isinstance(content, str):
                    # Look for common patterns in tool output
                    import re
                    # Pattern: "Found X results" or "X triples" or "X classes" etc.
                    match = re.search(r"(?:found|returned|retrieved)\s+(\d+)", content.lower())
                    if match:
                        result_count = int(match.group(1))
                    # Check if content looks like a list
                    elif content.startswith("[") and content.endswith("]"):
                        try:
                            import json
                            parsed = json.loads(content)
                            if isinstance(parsed, list):
                                result_count = len(parsed)
                        except:
                            pass

                # Create tool invocation record
                invocation = ToolInvocation(
                    timestamp=pending["start_time"],
                    tool_name=tool_name,
                    agent=pending["agent"],
                    duration_ms=duration_ms,
                    success="error" not in content.lower()[:100],
                    error_message=content[:500] if "error" in content.lower()[:100] else None,
                    input_summary=pending["input_summary"],
                    output_size=len(content) if isinstance(content, str) else 0,
                    result_count=result_count,
                    tool_category=tool_category,
                )
                self.tool_invocations.append(invocation)

                # Accumulate tool execution time
                self.total_tool_execution_time_ms += duration_ms

                # Update agent invocation record with tool time
                agent = pending["agent"]
                if agent in self._agent_invocation_records and self._agent_invocation_records[agent]:
                    self._agent_invocation_records[agent][-1]["tool_time_ms"] += duration_ms

    def _handle_delegation_start(self, event: TraceEvent) -> None:
        """Handle delegation start event (agent hands off to another agent)."""
        data = event.data
        from_agent = data.get("from_agent", "unknown")
        to_agent = data.get("to_agent", "unknown")
        reason = data.get("reason", "")

        self._active_delegations[to_agent] = {
            "from_agent": from_agent,
            "start_time": event.timestamp,
            "reason": reason,
        }

    def _handle_delegation_end(self, event: TraceEvent) -> None:
        """Handle delegation end event (delegated agent returns control)."""
        data = event.data
        to_agent = data.get("to_agent", "unknown")
        success = "success" in data.get("reason", "").lower()

        if to_agent in self._active_delegations:
            delegation = self._active_delegations.pop(to_agent)
            duration_ms = (event.timestamp - delegation["start_time"]).total_seconds() * 1000

            self._delegation_records.append({
                "from_agent": delegation["from_agent"],
                "to_agent": to_agent,
                "reason": delegation["reason"],
                "duration_ms": duration_ms,
                "success": success,
            })

    def finalize(
        self,
        result_metrics: ResultMetrics | None = None,
    ) -> ExperimentRunTrace:
        """
        Finalize and return the complete experiment run trace.

        Args:
            result_metrics: Optional pre-calculated result metrics

        Returns:
            Complete ExperimentRunTrace
        """
        end_time = datetime.now()

        # Finalize any open phases before creating the trace
        self._finalize_current_phase(end_time)

        total_time_ms = (end_time - self.start_time).total_seconds() * 1000

        # Calculate timing breakdown
        retrieval_time = sum(p.duration_ms for p in self.retrieval_phases)
        generation_time = sum(p.duration_ms for p in self.generation_phases)
        llm_time = sum(c.duration_ms for c in self.all_llm_calls)
        sparql_time = sum(e.duration_ms for e in self.all_sparql_executions)

        timing = TimingBreakdown(
            total_time_ms=total_time_ms,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            llm_inference_time_ms=llm_time,
            sparql_execution_time_ms=sparql_time,
            tool_execution_time_ms=self.total_tool_execution_time_ms,
            # Flexible agent-level timing
            agent_timing=self.agent_timing,
        )

        # Calculate token usage
        total_prompt = sum(c.input_tokens for c in self.all_llm_calls)
        total_completion = sum(c.output_tokens for c in self.all_llm_calls)
        total_cost = sum(c.estimated_cost_usd for c in self.all_llm_calls)

        # Per-phase tokens
        retrieval_tokens = sum(
            c.total_tokens
            for p in self.retrieval_phases
            for c in p.llm_calls
        )
        generation_tokens = sum(
            c.total_tokens
            for p in self.generation_phases
            for c in p.llm_calls
        )
        orchestrator_tokens = (total_prompt + total_completion) - retrieval_tokens - generation_tokens

        token_usage = TokenUsage(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
            estimated_cost_usd=total_cost,
            retrieval_tokens=retrieval_tokens,
            generation_tokens=generation_tokens,
            orchestrator_tokens=max(0, orchestrator_tokens),
        )

        # Calculate final result count
        final_count = 0
        if self.final_results:
            if isinstance(self.final_results, dict):
                bindings = self.final_results.get("bindings", [])
                final_count = len(bindings)
            elif isinstance(self.final_results, list):
                final_count = len(self.final_results)

        # =====================================================================
        # NEW: Aggregate tool usage metrics by agent
        # =====================================================================
        tool_usage_by_agent: dict[str, list[ToolUsageMetrics]] = {}

        # Group tool invocations by (agent, tool_name)
        tool_groups: dict[tuple[str, str], list[ToolInvocation]] = {}
        for inv in self.tool_invocations:
            key = (inv.agent, inv.tool_name)
            if key not in tool_groups:
                tool_groups[key] = []
            tool_groups[key].append(inv)

        # Helper to categorize tools
        def categorize_tool(name: str) -> str:
            name_lower = name.lower()
            if any(x in name_lower for x in ["search", "grep", "find", "retrieve"]):
                return "retrieval"
            elif any(x in name_lower for x in ["sparql", "query", "execute"]):
                return "sparql"
            elif any(x in name_lower for x in ["validate", "check", "verify"]):
                return "validation"
            elif any(x in name_lower for x in ["sample", "instance", "data"]):
                return "data_access"
            return "other"

        # Calculate metrics for each group
        for (agent, tool_name), invocations in tool_groups.items():
            durations = [inv.duration_ms for inv in invocations]
            success_count = sum(1 for inv in invocations if inv.success)
            total_results = sum(inv.result_count for inv in invocations)
            useful_count = sum(1 for inv in invocations if inv.result_useful is True)
            category = categorize_tool(tool_name)

            metrics = ToolUsageMetrics(
                tool_name=tool_name,
                agent=agent,
                tool_category=category,
                call_count=len(invocations),
                total_duration_ms=sum(durations),
                avg_duration_ms=sum(durations) / len(durations) if durations else 0.0,
                min_duration_ms=min(durations) if durations else 0.0,
                max_duration_ms=max(durations) if durations else 0.0,
                success_count=success_count,
                error_count=len(invocations) - success_count,
                success_rate=success_count / len(invocations) if invocations else 1.0,
                total_results=total_results,
                avg_results_per_call=total_results / len(invocations) if invocations else 0.0,
                useful_count=useful_count,
            )

            if agent not in tool_usage_by_agent:
                tool_usage_by_agent[agent] = []
            tool_usage_by_agent[agent].append(metrics)

        # =====================================================================
        # NEW: Aggregate agent timing metrics
        # =====================================================================
        agent_timing_details: list[AgentTimingMetrics] = []

        for agent_name, records in self._agent_invocation_records.items():
            if not records:
                continue

            durations = [r["duration_ms"] for r in records if r["duration_ms"] > 0]
            llm_times = [r.get("llm_time_ms", 0.0) for r in records]
            tool_times = [r.get("tool_time_ms", 0.0) for r in records]
            llm_calls = sum(r.get("llm_calls", 0) for r in records)
            tool_calls = sum(r.get("tool_calls", 0) for r in records)
            context_chars = [r.get("context_chars", 0) for r in records if r.get("context_chars", 0) > 0]

            # Determine phase from agent name
            phase = ""
            if "retrieval" in agent_name.lower():
                phase = "retrieval"
            elif any(x in agent_name.lower() for x in ["sparql", "generation", "multi_step", "mapping"]):
                phase = "generation"
            elif "orchestrator" in agent_name.lower():
                phase = "orchestrator"

            agent_timing = AgentTimingMetrics(
                agent_name=agent_name,
                phase=phase,
                invocation_count=len(records),
                total_duration_ms=sum(durations) if durations else 0.0,
                avg_duration_ms=sum(durations) / len(durations) if durations else 0.0,
                min_duration_ms=min(durations) if durations else 0.0,
                max_duration_ms=max(durations) if durations else 0.0,
                llm_time_ms=sum(llm_times),
                tool_execution_time_ms=sum(tool_times),
                total_llm_calls=llm_calls,
                total_tool_calls=tool_calls,
                avg_context_chars=sum(context_chars) // len(context_chars) if context_chars else 0,
            )
            agent_timing_details.append(agent_timing)

        # =====================================================================
        # NEW: Aggregate delegation metrics
        # =====================================================================
        delegation_metrics: list[DelegationMetrics] = []

        # Group delegations by (from_agent, to_agent)
        delegation_groups: dict[tuple[str, str], list[dict]] = {}
        for d in self._delegation_records:
            key = (d["from_agent"], d["to_agent"])
            if key not in delegation_groups:
                delegation_groups[key] = []
            delegation_groups[key].append(d)

        for (from_agent, to_agent), delegations in delegation_groups.items():
            durations = [d["duration_ms"] for d in delegations]
            success_count = sum(1 for d in delegations if d.get("success", False))
            reasons = list(set(d["reason"] for d in delegations if d.get("reason")))

            delegation_metrics.append(DelegationMetrics(
                from_agent=from_agent,
                to_agent=to_agent,
                delegation_count=len(delegations),
                total_duration_ms=sum(durations),
                avg_duration_ms=sum(durations) / len(durations) if durations else 0.0,
                success_count=success_count,
                failure_count=len(delegations) - success_count,
                reasons=reasons,
            ))

        # Calculate summary counts
        unique_tools = set(inv.tool_name for inv in self.tool_invocations)
        total_delegations = len(self._delegation_records)

        return ExperimentRunTrace(
            run_id=self.run_id,
            timestamp=self.start_time,
            approach=self.approach,  # type: ignore
            llm_model=self.llm_model,
            run_number=self.run_number,
            query_metadata=self.query_metadata,
            retrieval_phases=self.retrieval_phases,
            generation_phases=self.generation_phases,
            final_sparql=self.final_sparql,
            final_results=self.final_results if isinstance(self.final_results, list) else [],
            final_result_count=final_count,
            ground_truth_query_id=self.ground_truth_query_id,
            success=self.success,
            is_unanswerable=self.is_unanswerable,
            unanswerable_reason=self.unanswerable_reason,
            error_message=self.error_message,
            result_metrics=result_metrics,
            token_usage=token_usage,
            timing=timing,
            total_iterations=len(self.retrieval_phases) + len(self.generation_phases),
            retrieval_iterations=len(self.retrieval_phases),
            generation_iterations=len(self.generation_phases),
            total_llm_calls=len(self.all_llm_calls),
            total_sparql_executions=len(self.all_sparql_executions),
            strategy_used=self.strategy,  # type: ignore
            # Retrieved triples tracking
            all_retrieved_triples=list(self.all_retrieved_triples),
            final_schema_triples=self.final_schema_triples,
            # Detailed tool and agent tracking
            tool_usage_by_agent=tool_usage_by_agent,
            tool_invocations=self.tool_invocations,
            agent_timing_details=agent_timing_details,
            delegation_metrics=delegation_metrics,
            total_tool_calls=len(self.tool_invocations),
            unique_tools_used=len(unique_tools),
            total_delegation_count=total_delegations,
        )


def create_experiment_batch(
    runs: list[ExperimentRunTrace],
    experiment_name: str,
) -> ExperimentBatch:
    """
    Create an ExperimentBatch from a list of runs.

    Calculates aggregated statistics (mean, std) across runs.

    Args:
        runs: List of experiment run traces
        experiment_name: Name of the experiment

    Returns:
        ExperimentBatch with aggregated statistics
    """
    if not runs:
        raise ValueError("Cannot create batch from empty run list")

    # All runs should have same query/approach/model
    first = runs[0]

    # Calculate statistics
    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        m = mean(values)
        variance = sum((x - m) ** 2 for x in values) / len(values)
        return variance ** 0.5

    f1_scores = [
        r.result_metrics.f1_score for r in runs if r.result_metrics
    ]
    exec_accuracies = [
        r.result_metrics.execution_accuracy for r in runs if r.result_metrics
    ]
    times = [r.timing.total_time_ms for r in runs]
    tokens = [r.token_usage.total_tokens for r in runs]

    success_count = sum(1 for r in runs if r.success)
    unanswerable_count = sum(1 for r in runs if r.is_unanswerable)

    return ExperimentBatch(
        batch_id=str(uuid.uuid4())[:8],
        experiment_name=experiment_name,
        query_id=first.query_metadata.query_id,
        approach=first.approach,
        llm_model=first.llm_model,
        runs_per_query=len(runs),
        runs=runs,
        mean_f1_score=mean(f1_scores),
        std_f1_score=std(f1_scores),
        mean_execution_accuracy=mean(exec_accuracies),
        std_execution_accuracy=std(exec_accuracies),
        mean_total_time_ms=mean(times),
        std_total_time_ms=std(times),
        mean_total_tokens=mean(tokens),
        std_total_tokens=std(tokens),
        success_rate=success_count / len(runs) if runs else 0.0,
        unanswerable_rate=unanswerable_count / len(runs) if runs else 0.0,
    )


__all__ = [
    "convert_use_case_query_to_metadata",
    "ExperimentTraceCollector",
    "create_experiment_batch",
]