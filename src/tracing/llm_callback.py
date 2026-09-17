"""
LangChain callback handler for LLM tracing.

This module provides a callback handler that tracks all LLM calls
and emits TraceEvents with token usage and timing information.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from src.tracing.models import TraceEvent, TraceEventType
from src.tracing.tracer import get_tracer


class LLMTraceCallback(AsyncCallbackHandler):
    """
    Async callback handler that traces LLM calls.

    Emits TraceEvents for each LLM call with:
    - Token usage (input, output, total)
    - Duration
    - Model name
    - Phase context
    """

    def __init__(self, phase: str = "unknown", purpose: str = "llm_call"):
        """
        Initialize the callback.

        Args:
            phase: Current phase (retrieval, generation, orchestrator)
            purpose: Purpose of the LLM call
        """
        super().__init__()
        self.phase = phase
        self.purpose = purpose
        self._call_start_times: dict[UUID, datetime] = {}
        self._call_prompts: dict[UUID, str] = {}

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts."""
        self._call_start_times[run_id] = datetime.now()
        # Store prompt length for context tracking (not full prompt)
        self._call_prompts[run_id] = str(len(prompts[0])) if prompts else "0"

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM ends - emit trace event with token usage."""
        tracer = get_tracer()

        # Calculate duration
        start_time = self._call_start_times.pop(run_id, None)
        duration_ms = 0.0
        if start_time:
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Extract token usage from response
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        model_name = "unknown"

        if response.llm_output:
            # OpenAI format
            token_usage = response.llm_output.get("token_usage", {})
            if token_usage:
                input_tokens = token_usage.get("prompt_tokens", 0)
                output_tokens = token_usage.get("completion_tokens", 0)
                total_tokens = token_usage.get("total_tokens", 0)

            # Model name
            model_name = response.llm_output.get("model_name", "unknown")

        # Try to get from generation info (works for newer LangChain)
        if not total_tokens and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, "generation_info") and gen.generation_info:
                        info = gen.generation_info
                        # Anthropic format
                        if "usage" in info:
                            usage = info["usage"]
                            input_tokens = usage.get("input_tokens", 0)
                            output_tokens = usage.get("output_tokens", 0)
                            total_tokens = input_tokens + output_tokens
                        # OpenAI format in generation_info
                        if "token_usage" in info:
                            usage = info["token_usage"]
                            input_tokens = usage.get("prompt_tokens", 0)
                            output_tokens = usage.get("completion_tokens", 0)
                            total_tokens = usage.get("total_tokens", 0)

        # Clean up prompt tracking
        prompt_chars = self._call_prompts.pop(run_id, "0")

        # Emit trace event
        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.TOOL_CALL,
                phase=self.phase,
                data={
                    "tool_name": "llm_call",
                    "purpose": self.purpose,
                    "model": model_name,
                    "context_char_count": int(prompt_chars),
                },
            )
        )

        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.TOOL_RESULT,
                phase=self.phase,
                data={
                    "tool_name": "llm_call",
                    "tokens": {
                        "input": input_tokens,
                        "output": output_tokens,
                        "total": total_tokens,
                    },
                    "duration_ms": duration_ms,
                    "model": model_name,
                },
            )
        )

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors."""
        tracer = get_tracer()

        # Clean up
        self._call_start_times.pop(run_id, None)
        self._call_prompts.pop(run_id, None)

        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.ERROR,
                phase=self.phase,
                data={
                    "source": "llm_call",
                    "purpose": self.purpose,
                    "error": str(error),
                },
            )
        )


def create_llm_callback(phase: str, purpose: str = "llm_call") -> LLMTraceCallback:
    """
    Create an LLM trace callback for a specific phase.

    Args:
        phase: Current phase (retrieval, generation, orchestrator)
        purpose: Purpose of the LLM call

    Returns:
        Configured LLMTraceCallback
    """
    return LLMTraceCallback(phase=phase, purpose=purpose)


__all__ = ["LLMTraceCallback", "create_llm_callback"]
