"""Central tracing manager using Observer pattern."""

import contextvars
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Protocol

from src.tracing.models import TraceEvent, TraceEventType


# Context variable for run_id - automatically propagates across async tasks
_current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_run_id", default=None
)


def get_current_run_id() -> str | None:
    """Get the current run_id from context."""
    return _current_run_id.get()


def set_current_run_id(run_id: str | None) -> contextvars.Token:
    """Set the current run_id in context. Returns a token to reset later."""
    return _current_run_id.set(run_id)


@contextmanager
def trace_run_context(run_id: str):
    """
    Context manager to set the run_id for all trace events in this context.

    This is CRITICAL for parallel execution - each concurrent run should
    use its own trace_run_context to ensure events are correctly associated.

    Usage:
        with trace_run_context("my-run-id"):
            # All trace events emitted here will have run_id="my-run-id"
            tracer.emit(TraceEvent(...))
    """
    token = set_current_run_id(run_id)
    try:
        yield
    finally:
        _current_run_id.reset(token)


class TraceListener(Protocol):
    """Protocol for trace listeners."""

    def on_event(self, event: TraceEvent) -> None:
        """Handle a trace event."""
        ...


class TracingManager:
    """
    Singleton tracing manager.

    Thread-safe, supports multiple listeners, configurable verbosity.
    """

    _instance: "TracingManager | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "TracingManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._listeners: list[TraceListener] = []
        self._enabled = True
        self._verbosity = 2  # 0=off, 1=phases, 2=+tools, 3=+details
        self._current_phase: str | None = None
        self._phase_start_time: datetime | None = None
        self._initialized = True

    def add_listener(self, listener: TraceListener) -> None:
        """Add a trace listener."""
        self._listeners.append(listener)

    def remove_listener(self, listener: TraceListener) -> None:
        """Remove a trace listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def clear_listeners(self) -> None:
        """Remove all listeners."""
        self._listeners.clear()

    def set_verbosity(self, level: int) -> None:
        """Set verbosity level (0-3)."""
        self._verbosity = max(0, min(3, level))

    def get_verbosity(self) -> int:
        """Get current verbosity level."""
        return self._verbosity

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable tracing."""
        self._enabled = enabled

    def is_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._enabled

    def emit(self, event: TraceEvent) -> None:
        """Emit an event to all listeners.

        Automatically attaches the current run_id from context if not already set.
        This ensures correct event routing in parallel execution scenarios.
        """
        if not self._enabled:
            return

        # Automatically attach run_id from context if not set
        if event.run_id is None:
            event.run_id = get_current_run_id()

        # Filter by verbosity
        if self._verbosity == 0:
            return

        # Level 1: Only phases, query start/complete, errors
        if self._verbosity == 1 and event.event_type not in [
            TraceEventType.PHASE_START,
            TraceEventType.PHASE_END,
            TraceEventType.QUERY_START,
            TraceEventType.QUERY_COMPLETE,
            TraceEventType.ERROR,
        ]:
            return

        # Level 2: Add tool calls, orchestrator decisions, sparql results
        # (everything except detailed LLM reasoning)

        for listener in self._listeners:
            try:
                listener.on_event(event)
            except Exception:
                pass  # Don't let listener errors break the flow

    @contextmanager
    def trace_phase(self, phase: str, agent: str | None = None):
        """Context manager for tracing a phase."""
        self._current_phase = phase
        self._phase_start_time = datetime.now()

        self.emit(
            TraceEvent(
                event_type=TraceEventType.PHASE_START,
                phase=phase,
                agent=agent,
            )
        )

        try:
            yield
        finally:
            duration = (datetime.now() - self._phase_start_time).total_seconds() * 1000
            self.emit(
                TraceEvent(
                    event_type=TraceEventType.PHASE_END,
                    phase=phase,
                    agent=agent,
                    duration_ms=duration,
                )
            )
            self._current_phase = None

    def get_current_phase(self) -> str | None:
        """Get the currently active phase."""
        return self._current_phase


# Global accessor
_tracer: TracingManager | None = None


def get_tracer() -> TracingManager:
    """Get the global tracing manager."""
    global _tracer
    if _tracer is None:
        _tracer = TracingManager()
    return _tracer


def reset_tracer() -> None:
    """Reset the global tracer (mainly for testing).

    WARNING: Do NOT call this during parallel execution!
    This clears ALL listeners, which will break event collection
    for other concurrent runs. Use trace_run_context() instead
    for run isolation in parallel scenarios.
    """
    import warnings
    warnings.warn(
        "reset_tracer() is deprecated for parallel execution. "
        "Use trace_run_context() for run isolation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    global _tracer
    if _tracer is not None:
        _tracer.clear_listeners()
        _tracer._enabled = True
        _tracer._verbosity = 2


__all__ = [
    "TraceListener",
    "TracingManager",
    "get_tracer",
    "reset_tracer",
    "get_current_run_id",
    "set_current_run_id",
    "trace_run_context",
]
