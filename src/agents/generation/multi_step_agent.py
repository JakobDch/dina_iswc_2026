"""
Multi-Step Agent for Query Decomposition and Transformation.

This agent is invoked by the SPARQL Agent when a query repeatedly times out.
It decomposes complex queries into simpler steps and writes Python transformation
scripts to aggregate the results.

Workflow:
1. Analyze the failed query structure (JOINs, aggregations)
2. Decompose into individual step queries (each step = simple query)
3. Test each step against the endpoint
4. Generate Python transformation script for aggregation
5. Test the transformation script
6. Return aggregated results
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

from src.config import get_settings
from src.tools.multi_step_tools import MULTI_STEP_TOOLS, get_cached_results, get_all_cached_results, clear_results_cache
from src.tracing import TraceEvent, TraceEventType, get_tracer
from src.tracing.llm_callback import create_llm_callback

logger = logging.getLogger(__name__)


def emit_agent_message(tracer, msg: BaseMessage, agent_name: str = "multi_step_agent") -> None:
    """Emit an AGENT_MESSAGE event for context visualization.

    Args:
        tracer: The TracingManager instance
        msg: The LangChain message to emit
        agent_name: Name of the agent emitting the message
    """
    data = {
        "message_type": msg.__class__.__name__,
        "content": msg.content[:2000] if msg.content else "",
    }

    if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
        data["tool_calls"] = msg.tool_calls

    if isinstance(msg, ToolMessage):
        data["tool_call_id"] = getattr(msg, 'tool_call_id', '')
        data["tool_name"] = getattr(msg, 'name', '')

    tracer.emit(TraceEvent(
        event_type=TraceEventType.AGENT_MESSAGE,
        phase="generation",
        agent=agent_name,
        data=data,
    ))


@dataclass
class MultiStepResult:
    """Result from Multi-Step Agent decomposition and transformation."""

    success: bool
    steps: list[dict] = field(default_factory=list)  # [{name, query, dataset, results}]
    transform_script: str = ""
    final_results: list[dict] = field(default_factory=list)
    error: str = ""
    reasoning: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "steps": self.steps,
            "steps_count": len(self.steps),
            "transform_script": self.transform_script,
            "final_results": self.final_results,
            "result_count": len(self.final_results),
            "error": self.error,
            "reasoning": self.reasoning,
        }


SYSTEM_PROMPT = """You decompose complex SPARQL queries that timeout into simpler step queries.

## Your Task

A SPARQL query timed out. You receive:
- The failed query
- Query history: ALL queries the SPARQL Agent tested (with their results)
- An analysis showing cardinalities of each predicate (TIMEOUT = very large table)
- Variable usage information

You must:
1. Analyze the query structure and the cardinality data
2. Decide how to split the query into steps that won't timeout
3. Test your step queries with `execute_step_query`
4. Write a Python transformation script to combine the results
5. Test the script with `test_transform_script`

## Decomposition Strategy

**Key Principle: Split joins, don't limit data.**

- NEVER use LIMIT to avoid timeouts - it gives incomplete/wrong results
- Each step should query ONE relationship (one predicate)
- Pass URIs from step N to step N+1 using VALUES clause
- Do the join/aggregation in Python, not in SPARQL

**Example:** A query times out because ex:borrows has >1M rows:

```sparql
# FAILED QUERY (times out due to ex:borrows cardinality)
SELECT ?lib ?book WHERE {
  ?lib a ex:Library .
  ?lib ex:name "Central" .       # Small: only 1 library
  ?member ex:memberOf ?lib .     # Medium: ~500 members
  ?member ex:borrows ?book .     # TIMEOUT: >1M borrow records
}
```

**Correct decomposition:**

Step 1 - Find the library (small result):
```sparql
SELECT ?lib WHERE { ?lib a ex:Library . ?lib ex:name "Central" }
```
→ Returns 1 URI

Step 2 - Find members OF THAT library using VALUES:
```sparql
SELECT ?member WHERE { VALUES ?lib { <uri-from-step1> } ?member ex:memberOf ?lib }
```
→ Returns ~500 URIs

Step 3 - Find books borrowed BY THOSE members using VALUES:
```sparql
SELECT ?member ?book WHERE { VALUES ?member { <uri1> <uri2> ... } ?member ex:borrows ?book }
```
→ Returns all books (complete data, no LIMIT!)

Transform - Aggregate in Python:
```python
def transform(step_results):
    lib = step_results["step1"][0]["lib"]["value"]
    books = set(r["book"]["value"] for r in step_results["step3"])
    return [{"lib": lib, "book": b} for b in books]
```

**Important:** If step N returns a small result (<1000 URIs), include ALL of them in the VALUES clause for step N+1. Don't iterate one-by-one - that wastes tool calls!

## Available Tools

1. `execute_step_query(query, dataset, step_name, timeout)` - Execute a step query
2. `test_transform_script(script, step_results_json)` - Test transformation script

## Constraints

- Maximum 15 tool calls total
- Each step query must complete within 30 seconds
- The transformation script must define `transform(step_results)` returning a list

## Output Format

When done, respond with JSON:
```json
{
    "status": "SUCCESS",
    "steps": [{"name": "...", "query": "...", "dataset": "..."}],
    "transform_script": "def transform(step_results): ...",
    "reasoning": "..."
}
```

Or if decomposition fails:
```json
{
    "status": "FAILED",
    "error": "...",
    "reasoning": "..."
}
```
"""


class MultiStepAgent:
    """Agent for decomposing complex queries and generating transformation scripts."""

    def __init__(
        self,
        llm_model: str = "deepseek-chat",
        max_tool_calls: int = 15,
    ):
        """
        Initialize the Multi-Step Agent.

        Args:
            llm_model: The LLM model to use
            max_tool_calls: Maximum number of tool calls allowed
        """
        settings = get_settings()
        self.max_tool_calls = max_tool_calls

        # Initialize LLM based on model name
        ollama_patterns = ["qwen", "llama", "deepseek-r1", "mistral", "gemma", "phi"]
        is_ollama = any(p in llm_model.lower() for p in ollama_patterns)

        if llm_model.startswith("openrouter/"):
            self.llm = ChatOpenAI(
                model=llm_model[len("openrouter/"):],
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                temperature=0.1,
            )
        elif llm_model.startswith("openai/"):
            self.llm = ChatOpenAI(
                model=llm_model,
                api_key=os.getenv("LOCAL_PROXY_API_KEY", ""),
                base_url=os.getenv("LOCAL_PROXY_BASE_URL", "http://localhost:4000/v1"),
                temperature=0.1,
            )
        elif is_ollama:
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.llm = ChatOllama(
                model=llm_model,
                base_url=ollama_base_url,
                temperature=0.1,
            )
        elif "claude" in llm_model.lower():
            self.llm = ChatAnthropic(
                model=llm_model,
                api_key=settings.anthropic_api_key,
            )
        elif "deepseek" in llm_model.lower():
            self.llm = ChatOpenAI(
                model=llm_model,
                api_key=settings.deepseek_api_key,
                base_url="https://api.deepseek.com",
            )
        else:
            self.llm = ChatOpenAI(
                model=llm_model,
                api_key=settings.openai_api_key,
            )

        # Set up tools
        self.tools = MULTI_STEP_TOOLS
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.tracer = get_tracer()
        self.llm_callback = create_llm_callback("generation", "multi_step_agent")

    async def decompose_and_transform(
        self,
        failed_query: str,
        user_query: str,
        schema_context: str,
        error_message: str,
        query_history: str,
        dataset: str,
    ) -> MultiStepResult:
        """
        Decompose a failed query into steps and generate transformation script.

        This is the main entry point, called by the SPARQL Agent's delegation tool.

        Args:
            failed_query: The SPARQL query that timed out
            user_query: Original natural language query
            schema_context: Available schema triples
            error_message: The timeout/error message
            query_history: JSON array of all queries tested by SPARQL Agent
                           [{"query": str, "status": "success"|"timeout"|"error", "count": int}]
            dataset: Dataset ID (edu, trn, nrg, bsbm)

        Returns:
            MultiStepResult with steps, transformation script, and results
        """
        from datetime import datetime
        from src.tools.multi_step_tools import analyze_failed_query

        agent_start_time = datetime.now()

        # Emit agent start event for timing tracking
        self.tracer.emit(
            TraceEvent(
                event_type=TraceEventType.AGENT_START,
                phase="generation",
                agent="multi_step_agent",
                data={"failed_query_length": len(failed_query), "dataset": dataset},
            )
        )

        # === AUTOMATIC QUERY ANALYSIS ===
        # Run analyze_failed_query BEFORE the LLM loop to identify bottlenecks
        logger.info(f"Multi-Step Agent: Auto-analyzing query for dataset '{dataset}'...")
        analysis_result = analyze_failed_query.invoke({
            "query": failed_query,
            "dataset": dataset
        })
        logger.info(f"Multi-Step Agent: Query analysis complete")

        # Parse query_history JSON
        try:
            history_list = json.loads(query_history) if query_history else []
        except json.JSONDecodeError:
            history_list = []

        # Emit detailed input event for debugging
        self.tracer.emit(
            TraceEvent(
                event_type=TraceEventType.MULTI_STEP_INPUT,
                phase="generation",
                agent="multi_step_agent",
                data={
                    "failed_query": failed_query,
                    "user_query": user_query,
                    "dataset": dataset,
                    "query_analysis": analysis_result[:2000] if len(analysis_result) > 2000 else analysis_result,
                    "schema_context": schema_context[:3000] if len(schema_context) > 3000 else schema_context,
                    "error_message": error_message,
                    "query_history_count": len(history_list),
                },
            )
        )

        # Clear any cached results from previous runs
        clear_results_cache()

        # Format query history for display
        history_str = ""
        if history_list:
            history_str = "\n## Query History (all queries tested by SPARQL Agent)\n\n"
            for i, entry in enumerate(history_list, 1):
                status = entry.get("status", "unknown")
                count = entry.get("count", 0)
                error_msg = entry.get("error_msg", "")
                query = entry.get("query", "")

                status_display = f"**{status.upper()}**"
                if status == "success":
                    status_display += f" ({count} results)"
                elif status == "error" and error_msg:
                    status_display += f" - {error_msg}"

                history_str += f"### Query {i}: {status_display}\n```sparql\n{query}\n```\n\n"

        # Build the user message with facts only - no suggestions
        user_message = f"""## Failed Query (last attempt)

```sparql
{failed_query}
```

## Error
{error_message}

## Dataset
{dataset}
{history_str}
## Query Analysis

Cardinalities and structure of the query patterns:

```json
{analysis_result}
```

## Original User Question
{user_query}

## Schema Context
{schema_context}
"""

        sys_msg = SystemMessage(content=SYSTEM_PROMPT)
        human_msg = HumanMessage(content=user_message)
        messages: list[BaseMessage] = [sys_msg, human_msg]

        # Emit initial messages for context visualization
        emit_agent_message(self.tracer, sys_msg, "multi_step_agent")
        emit_agent_message(self.tracer, human_msg, "multi_step_agent")

        # Run the agent loop with tools
        tool_call_count = 0
        step_results: dict[str, list[dict]] = {}
        llm_call_count = 0

        while tool_call_count < self.max_tool_calls:
            llm_call_count += 1
            logger.info(f"Multi-Step Agent: LLM call #{llm_call_count} starting...")

            try:
                import asyncio
                response = await asyncio.wait_for(
                    self.llm_with_tools.ainvoke(messages),
                    timeout=180.0  # 3 minutes per LLM call (DeepSeek can be slow)
                )
            except asyncio.TimeoutError:
                logger.error(f"Multi-Step Agent: LLM call #{llm_call_count} timed out after 180s")
                return MultiStepResult(
                    success=False,
                    error="LLM call timed out after 180 seconds",
                    reasoning="The language model did not respond in time. Check API connectivity.",
                )

            logger.info(f"Multi-Step Agent: LLM call #{llm_call_count} completed")
            messages.append(response)
            emit_agent_message(self.tracer, response, "multi_step_agent")

            # If no tool calls, we have the final response
            if not response.tool_calls:
                break

            # Execute tool calls
            for tool_call in response.tool_calls:
                if tool_call_count >= self.max_tool_calls:
                    limit_msg = ToolMessage(
                        content="LIMIT REACHED: Maximum tool calls exceeded. Finalize your response NOW.",
                        tool_call_id=tool_call["id"],
                        name="system",
                    )
                    messages.append(limit_msg)
                    emit_agent_message(self.tracer, limit_msg, "multi_step_agent")
                    continue

                tool_call_count += 1
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                # Emit tracing event
                self.tracer.emit(
                    TraceEvent(
                        event_type=TraceEventType.TOOL_CALL,
                        phase="generation",
                        agent="multi_step_agent",
                        data={"tool_name": tool_name, "tool_args": tool_args},
                    )
                )

                # Find and execute tool
                tool_fn = next(
                    (t for t in self.tools if t.name == tool_name),
                    None,
                )

                if not tool_fn:
                    error_msg = f"ERROR: Tool '{tool_name}' not found."
                    not_found_msg = ToolMessage(content=error_msg, tool_call_id=tool_call["id"], name="error")
                    messages.append(not_found_msg)
                    emit_agent_message(self.tracer, not_found_msg, "multi_step_agent")
                    continue

                try:
                    result = tool_fn.invoke(tool_args)

                    # Track step results for later use
                    if tool_name == "execute_step_query":
                        result_data = json.loads(result)
                        if result_data.get("success"):
                            # Use the step_name from the tool call args
                            step_name = tool_args.get("step_name", f"step{len(step_results) + 1}")
                            # Results are cached by step_name in the tool
                            step_results[step_name] = True  # Mark as executed

                            # Emit detailed step event for debugging
                            self.tracer.emit(
                                TraceEvent(
                                    event_type=TraceEventType.MULTI_STEP_STEP,
                                    phase="generation",
                                    agent="multi_step_agent",
                                    data={
                                        "step_name": step_name,
                                        "query": tool_args.get("query", ""),
                                        "dataset": tool_args.get("dataset", ""),
                                        "result_count": result_data.get("count", 0),
                                        "variables": result_data.get("variables", []),
                                        "sample_results": result_data.get("sample", [])[:10],
                                    },
                                )
                            )

                    self.tracer.emit(
                        TraceEvent(
                            event_type=TraceEventType.TOOL_RESULT,
                            phase="generation",
                            agent="multi_step_agent",
                            data={"tool_name": tool_name, "success": True},
                        )
                    )

                    # Add remaining calls info with progressive warnings
                    remaining = self.max_tool_calls - tool_call_count
                    result_str = result

                    # For execute_step_query, remind about the step name for consistency
                    if tool_name == "execute_step_query":
                        step_name = tool_args.get("step_name", "")
                        if step_name:
                            result_str += f'\n\n[REMEMBER] This step is stored as "{step_name}". Use step_results["{step_name}"] in your transform script!'

                    if remaining == 1:
                        result_str += f"\n\n[CRITICAL] This is your LAST tool call! If you haven't tested your transform script with test_transform_script yet, do it NOW! Otherwise respond with your final JSON."
                    elif remaining == 2:
                        result_str += f"\n\n[WARNING] Only 2 tool calls remaining! You MUST call test_transform_script before returning SUCCESS! Test your script now!"
                    elif remaining <= 3:
                        result_str += f"\n\n[ATTENTION] Only {remaining} tool calls remaining. Have you tested your transform script? Call test_transform_script before finishing!"

                    result_tool_msg = ToolMessage(content=result_str, tool_call_id=tool_call["id"], name=tool_name)
                    messages.append(result_tool_msg)
                    emit_agent_message(self.tracer, result_tool_msg, "multi_step_agent")

                except Exception as e:
                    error_msg = f"Tool error: {str(e)}"
                    error_tool_msg = ToolMessage(content=error_msg, tool_call_id=tool_call["id"], name=tool_name)
                    messages.append(error_tool_msg)
                    emit_agent_message(self.tracer, error_tool_msg, "multi_step_agent")

        # Check if we exited due to tool limit without a final response
        # This happens when tool_call_count >= max_tool_calls AND the last message isn't a final AI response
        last_msg = messages[-1] if messages else None
        needs_finalization = (
            tool_call_count >= self.max_tool_calls and
            (isinstance(last_msg, ToolMessage) or (isinstance(last_msg, AIMessage) and last_msg.tool_calls))
        )
        if needs_finalization:
            # Agent ran out of tool calls - force a finalization call
            logger.warning("Multi-Step Agent: Tool limit reached, forcing finalization...")

            # Get the actual cached step names to enforce consistency
            cached_step_names = list(get_all_cached_results().keys())
            step_names_str = ", ".join(f'"{name}"' for name in cached_step_names) if cached_step_names else "(none cached)"

            finalize_human_msg = HumanMessage(content=f"""You have completed your tool calls. Now provide your final JSON response.

**CRITICAL: You executed steps with these EXACT names: {step_names_str}**
Your transform_script MUST use these exact names in step_results["..."], not different names!

If you successfully tested your step queries and transformation script, respond with:
```json
{{
    "status": "SUCCESS",
    "steps": [{{"name": "<use exact step names from above>", "query": "...", "dataset": "..."}}],
    "transform_script": "def transform(step_results): ... # use exact step names!",
    "reasoning": "..."
}}
```

If the decomposition failed, respond with status "FAILED" and explain why.""")
            messages.append(finalize_human_msg)
            emit_agent_message(self.tracer, finalize_human_msg, "multi_step_agent")

            try:
                import asyncio
                # Make one more call WITHOUT tools to get the final response
                finalization_response = await asyncio.wait_for(
                    self.llm.ainvoke(messages),  # Use llm without tools
                    timeout=120.0  # 2 minutes for finalization
                )
                messages.append(finalization_response)
                emit_agent_message(self.tracer, finalization_response, "multi_step_agent")
            except Exception as e:
                logger.error(f"Finalization call failed: {e}")

        # Parse final response
        final_response = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                final_response = msg.content
                break

        if not final_response:
            # Emit agent end event with timing
            agent_end_time = datetime.now()
            agent_duration_ms = (agent_end_time - agent_start_time).total_seconds() * 1000
            self.tracer.emit(
                TraceEvent(
                    event_type=TraceEventType.AGENT_END,
                    phase="generation",
                    agent="multi_step_agent",
                    duration_ms=agent_duration_ms,
                    data={"success": False, "error": "no_final_response"},
                )
            )
            return MultiStepResult(
                success=False,
                error="No final response from agent",
                reasoning="Agent did not produce a final response after tool calls",
            )

        # Try to parse JSON from response
        try:
            # Find JSON in response
            json_start = final_response.find("{")
            json_end = final_response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                result_json = json.loads(final_response[json_start:json_end])

                if result_json.get("status") == "SUCCESS":
                    # Execute the transformation to get final results
                    steps = result_json.get("steps", [])
                    transform_script = result_json.get("transform_script", "")
                    logger.info(f"Multi-Step Agent: Parsed SUCCESS response with {len(steps)} steps")
                    logger.info(f"Multi-Step Agent: step_results keys: {list(step_results.keys())}")

                    # Get ALL cached results (actual executed steps)
                    # This is more reliable than matching by names from JSON
                    all_cached = get_all_cached_results()
                    logger.info(f"Multi-Step Agent: all cached step names: {list(all_cached.keys())}")

                    # Collect step results - prefer cached results over JSON names
                    all_step_results = {}
                    enriched_steps = []

                    # First, try to match steps from JSON with cached results
                    matched_step_names = set()
                    for step in steps:
                        step_name = step.get("name")
                        enriched_step = dict(step)  # Copy the step
                        if step_name:
                            cached = get_cached_results(step_name)
                            if cached is not None:
                                all_step_results[step_name] = cached
                                enriched_step["results"] = cached
                                enriched_step["result_count"] = len(cached)
                                matched_step_names.add(step_name)
                            else:
                                logger.warning(f"No cached results found for step '{step_name}'")
                                enriched_step["results"] = []
                                enriched_step["result_count"] = 0
                        enriched_steps.append(enriched_step)

                    # Add any cached steps that weren't in the JSON (step name mismatch)
                    for cached_name, cached_results in all_cached.items():
                        if cached_name not in matched_step_names:
                            all_step_results[cached_name] = cached_results
                            # Also add to enriched_steps for display
                            enriched_steps.append({
                                "name": cached_name,
                                "results": cached_results,
                                "result_count": len(cached_results),
                                "query": "(cached step - name mismatch with JSON)",
                                "dataset": "",
                            })
                            logger.info(f"Multi-Step Agent: Added cached step '{cached_name}' (not in JSON)")

                    logger.info(f"Multi-Step Agent: Final all_step_results keys: {list(all_step_results.keys())}")

                    # Execute transformation if we have results
                    final_results = []

                    if all_step_results and transform_script:
                        try:
                            import re
                            from collections import defaultdict

                            # Strip import statements since modules are pre-loaded
                            clean_script = re.sub(r'from\s+collections\s+import\s+defaultdict\s*\n?', '', transform_script)
                            clean_script = re.sub(r'import\s+collections\s*\n?', '', clean_script)
                            clean_script = re.sub(r'import\s+re\s*\n?', '', clean_script)

                            exec_globals = {
                                "__builtins__": {
                                    "len": len, "str": str, "int": int, "float": float,
                                    "list": list, "dict": dict, "set": set, "tuple": tuple,
                                    "sorted": sorted, "enumerate": enumerate, "zip": zip,
                                    "range": range, "sum": sum, "min": min, "max": max,
                                    "True": True, "False": False, "None": None,
                                },
                                "defaultdict": defaultdict,
                                "re": re,
                            }
                            exec_locals: dict[str, Any] = {}
                            exec(clean_script, exec_globals, exec_locals)
                            if "transform" in exec_locals:
                                final_results = exec_locals["transform"](all_step_results)

                                # Emit detailed transformation event for debugging
                                self.tracer.emit(
                                    TraceEvent(
                                        event_type=TraceEventType.MULTI_STEP_TRANSFORM,
                                        phase="generation",
                                        agent="multi_step_agent",
                                        data={
                                            "transform_script": transform_script,
                                            "input_steps": list(all_step_results.keys()),
                                            "input_counts": {k: len(v) for k, v in all_step_results.items()},
                                            "output_count": len(final_results) if isinstance(final_results, list) else 0,
                                            "sample_output": final_results[:15] if isinstance(final_results, list) else [],
                                        },
                                    )
                                )
                        except Exception as e:
                            logger.warning(f"Transform execution failed: {e}")
                            # Emit error event for transform failure
                            self.tracer.emit(
                                TraceEvent(
                                    event_type=TraceEventType.MULTI_STEP_TRANSFORM,
                                    phase="generation",
                                    agent="multi_step_agent",
                                    data={
                                        "transform_script": transform_script,
                                        "error": str(e),
                                        "success": False,
                                    },
                                )
                            )

                    # Emit agent end event with timing (success)
                    agent_end_time = datetime.now()
                    agent_duration_ms = (agent_end_time - agent_start_time).total_seconds() * 1000
                    self.tracer.emit(
                        TraceEvent(
                            event_type=TraceEventType.AGENT_END,
                            phase="generation",
                            agent="multi_step_agent",
                            duration_ms=agent_duration_ms,
                            data={"success": True, "steps_count": len(enriched_steps)},
                        )
                    )
                    return MultiStepResult(
                        success=True,
                        steps=enriched_steps,  # Use enriched steps with results
                        transform_script=transform_script,
                        final_results=final_results if isinstance(final_results, list) else [],
                        reasoning=result_json.get("reasoning", ""),
                    )
                else:
                    # Emit agent end event with timing (failed status)
                    agent_end_time = datetime.now()
                    agent_duration_ms = (agent_end_time - agent_start_time).total_seconds() * 1000
                    self.tracer.emit(
                        TraceEvent(
                            event_type=TraceEventType.AGENT_END,
                            phase="generation",
                            agent="multi_step_agent",
                            duration_ms=agent_duration_ms,
                            data={"success": False, "error": "failed_status"},
                        )
                    )
                    return MultiStepResult(
                        success=False,
                        error=result_json.get("error", "Unknown error"),
                        reasoning=result_json.get("reasoning", ""),
                    )

        except json.JSONDecodeError:
            pass

        # Could not parse response
        # Emit agent end event with timing
        agent_end_time = datetime.now()
        agent_duration_ms = (agent_end_time - agent_start_time).total_seconds() * 1000
        self.tracer.emit(
            TraceEvent(
                event_type=TraceEventType.AGENT_END,
                phase="generation",
                agent="multi_step_agent",
                duration_ms=agent_duration_ms,
                data={"success": False, "error": "json_parse_error"},
            )
        )
        return MultiStepResult(
            success=False,
            error="Could not parse agent response as JSON",
            reasoning=final_response[:500],
        )


__all__ = ["MultiStepAgent", "MultiStepResult"]
