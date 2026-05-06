"""
SPARQL generation agent with endpoint selection for Dataspaces.

Generates SPARQL queries from natural language using retrieved schema context
and selects appropriate endpoints based on the query context.

Now supports tool calls to inspect actual data values before generating queries,
which helps with correct FILTER clauses (units, formats, enum values).
"""

import asyncio
import json
import re
from typing import Any

import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


def serialize_messages(messages: list[BaseMessage]) -> list[dict]:
    """Serialize LangChain messages to dicts for storage in GraphState.

    Preserves ALL message attributes including:
    - AIMessage.tool_calls (which tools were called)
    - ToolMessage (tool results with tool_call_id and name)
    """
    serialized = []
    for msg in messages:
        msg_dict = {
            "type": msg.__class__.__name__,
            "content": msg.content,
        }
        # Handle AIMessage with tool_calls
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            msg_dict["tool_calls"] = msg.tool_calls
        # Handle ToolMessage
        if isinstance(msg, ToolMessage):
            msg_dict["tool_call_id"] = getattr(msg, 'tool_call_id', '')
            msg_dict["name"] = getattr(msg, 'name', '')
        serialized.append(msg_dict)
    return serialized


def deserialize_messages(serialized: list[dict]) -> list[BaseMessage]:
    """Deserialize dicts back to LangChain messages.

    Restores ALL message types with their full attributes.
    """
    messages = []
    for msg_dict in serialized:
        msg_type = msg_dict.get("type", "HumanMessage")
        content = msg_dict.get("content", "")

        if msg_type == "SystemMessage":
            messages.append(SystemMessage(content=content))
        elif msg_type == "HumanMessage":
            messages.append(HumanMessage(content=content))
        elif msg_type == "AIMessage":
            tool_calls = msg_dict.get("tool_calls", [])
            messages.append(AIMessage(content=content, tool_calls=tool_calls))
        elif msg_type == "ToolMessage":
            messages.append(ToolMessage(
                content=content,
                tool_call_id=msg_dict.get("tool_call_id", ""),
                name=msg_dict.get("name", ""),
            ))
        else:
            # Default to HumanMessage
            messages.append(HumanMessage(content=content))
    return messages


def extract_json_from_response(text: str) -> dict:
    """Extract JSON object from LLM response that may contain extra text.

    LLMs sometimes output explanatory text before the JSON. This function
    finds and parses the JSON object from within the response.

    Args:
        text: Raw LLM response that may contain JSON

    Returns:
        Parsed JSON as dict

    Raises:
        ValueError: If no valid JSON found
    """
    # First try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in text
    # Look for { ... } pattern with balanced braces
    brace_count = 0
    start_idx = None

    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                # Found complete JSON object
                json_str = text[start_idx:i + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Continue searching for another JSON object
                    start_idx = None
                    continue

    # Try regex for JSON code block
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find by key patterns
    json_match = re.search(r'\{[^{}]*"query"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in response")

from src.config import get_settings, normalize_content
from src.tools.sparql_tools import load_dataset_registry, get_endpoint_for_dataset
from src.tools.retrieval_tools import SPARQL_GENERATION_TOOLS
from src.tracing import TraceEvent, TraceEventType, get_tracer
from src.tracing.llm_callback import create_llm_callback


def emit_agent_message(tracer, msg: BaseMessage, agent_name: str = "sparql_agent") -> None:
    """Emit an AGENT_MESSAGE event for context visualization.

    Args:
        tracer: The TracingManager instance
        msg: The LangChain message to emit
        agent_name: Name of the agent emitting the message
    """
    data = {
        "message_type": msg.__class__.__name__,
        "content": normalize_content(msg.content)[:2000] if msg.content else "",  # Truncate long content
    }

    # Handle AIMessage with tool_calls
    if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
        data["tool_calls"] = msg.tool_calls

    # Handle ToolMessage
    if isinstance(msg, ToolMessage):
        data["tool_call_id"] = getattr(msg, 'tool_call_id', '')
        data["tool_name"] = getattr(msg, 'name', '')

    tracer.emit(TraceEvent(
        event_type=TraceEventType.AGENT_MESSAGE,
        phase="generation",
        agent=agent_name,
        data=data,
    ))


class SPARQLOutputWithEndpoints(BaseModel):
    """Output format for SPARQL generation with endpoint selection."""

    status: str = Field(description="PENDING (query to test), DONE (results verified), NEED_TRIPLE (missing schema), MULTI_STEP_RESULT (delegated)")
    query: str = Field(default="", description="The generated SPARQL query")
    target_endpoints: list[str] = Field(default_factory=list, description="List of endpoint URLs to query")
    target_datasets: list[str] = Field(default_factory=list, description="List of dataset IDs (from the available datasets above)")
    reasoning: str = Field(description="Explanation of the query structure and endpoint selection")
    result_verification: str = Field(default="", description="REQUIRED for status=DONE: Explain why the results correctly answer the user's question")
    confidence: float = Field(default=0.0, description="Confidence score 0-1")
    needed_triple: str = Field(default="", description="Description of missing schema element (if status=NEED_TRIPLE)")
    multi_step_result: dict = Field(default_factory=dict, description="Result from Multi-Step Agent (if status=MULTI_STEP_RESULT)")


# Legacy output format for backwards compatibility
class SPARQLOutput(BaseModel):
    """Output format for SPARQL generation."""

    query: str = Field(description="The generated SPARQL query")
    reasoning: str = Field(description="Explanation of the query structure")
    confidence: float = Field(description="Confidence score 0-1")


SYSTEM_PROMPT_WITH_TOOLS = """You are an expert SPARQL query generator for a Dataspace environment.

Given a natural language question, relevant schema triples, and available dataset endpoints, you must:
1. Generate a valid SPARQL query
2. Select the appropriate endpoint(s) to execute the query against

Available Datasets:
{endpoints_registry}

## CRITICAL: OnTop 5.4.0 Limitations (SPARQL → SQL Translation)

**WILL FAIL:**
1. **No type casts in aggregations**: `AVG(xsd:decimal(?x))` → use `AVG(?x)` directly
2. **No property paths (*, +, ?)**: `?x rdfs:subClassOf+ ?y` → use explicit triple patterns
3. **No SERVICE, BNODE, STRUUID**

**WILL TIMEOUT:**
4. **Disconnected patterns create Cartesian products**: Ensure ALL triple patterns share variables
5. **Many SELECT variables cause N-1 self-joins**: Request only essential variables
6. **Multiple OPTIONALs = multiple LEFT JOINs**: Combine into single OPTIONAL block or remove

**TIP:** Use `CONTAINS()` instead of `REGEX()` for better performance.

## Available Tools

### test_candidate_query(query, dataset, limit?, timeout_seconds?)
Test a SPARQL query before finalizing it. Returns sample results or detailed error feedback.

Arguments:
- query: Complete SPARQL query with PREFIX declarations
- dataset: Dataset ID (as provided in the task context) - REQUIRED
- limit: Maximum results to return (default: 5, max: 10)
- timeout_seconds: Query timeout in seconds (default: 30, min: 15, max: 120)

**IMPORTANT about timeout_seconds:**
- Default is 30 seconds - sufficient for most optimized queries
- ONLY increase timeout (up to 120s) when ALL of these conditions are met:
  1. You have already delegated to the Mapping Optimizer
  2. The Mapping Optimizer confirmed it optimized the mappings (can_help: true, success: true)
  3. The generated SQL (from reformulate feedback) shows NO self-join issues
  4. The query simply needs more execution time due to large data volume
- Increasing timeout is a LAST RESORT, not a substitute for query optimization!

### sample_property_values(class_name, property_name, dataset?, prefix?)
Get sample values BEFORE writing FILTER clauses to understand data format, units, and range.

Arguments:
- class_name: Class name without prefix (as seen in the schema context)
- property_name: Property name without prefix
- dataset: Optional dataset ID
- prefix: Optional ontology prefix (auto-detected from dataset if omitted)

**Use for numeric comparisons** - data may use different units than expected.

### delegate_to_mapping_optimizer(sparql_query, generated_sql, dataset, sparql_agent_reasoning)
Delegate to the Mapping Optimizer Agent to fix self-join issues in the R2RML mapping.

Arguments:
- sparql_query: The query that timed out
- generated_sql: The SQL from the timeout feedback (REQUIRED - copy it exactly from the feedback)
- dataset: Dataset ID (as provided in the task context)
- sparql_agent_reasoning: YOUR analysis of why mapping optimization will help

Returns `can_help: false` if mapping optimization won't fix this particular query - try simplifying the query yourself.

## WORKFLOW

### Step 1: Analyze & Prepare
- Read the schema context and user question
- If you need to check data values (for FILTERs), use `sample_property_values`

### Step 2: Generate Query
- Generate a complete SPARQL query that answers the user's question
- Output with `status: "PENDING"` - the system will execute it and give you feedback

### Step 3: Receive Execution Feedback (automatic)
The system executes your query and returns one of:

| Feedback Type | What You Receive |
|---------------|------------------|
| **SUCCESS** | Sample results (first rows) with column names and values |
| **TIMEOUT** | Generated SQL query + self-join analysis + JOIN count |
| **SYNTAX ERROR** | Error message with details |
| **EMPTY RESULTS** | Confirmation that query returned 0 rows |

### Step 4: Decide Based on Feedback

**SUCCESS with sample results:**
- Carefully verify: Do these results correctly answer the user's original question?
- If YES → output `status: "DONE"` with `result_verification` explaining why results are correct
- If NO → generate a new query with `status: "PENDING"`

**TIMEOUT with SQL analysis:**
The feedback includes the generated SQL. Analyze it to decide which agent to delegate to.

**→ `delegate_to_mapping_optimizer`** if you see ANY of these patterns in the SQL:

1. **Self-Joins**: Same table appears multiple times with different aliases
   - Example: `FROM trips t1, trips t2` or `trips t1 JOIN trips t2`
   - Cause: IRI template doesn't use PRIMARY KEY columns

2. **Complex IRI Matching**: Many CONCAT or string operations in WHERE clause
   - Example: `WHERE CONCAT('http://.../', t1.col) = CONCAT('http://.../', t2.col)`
   - Cause: IRI template uses non-indexed columns requiring full table scans

3. **Cross-Table IRI Construction**: JOINs that only serve to build IRIs
   - Example: SELECT has many `CONCAT('http://...', t1.x, t2.y)` patterns
   - Cause: IRI template references columns from multiple tables

**If mapping optimizer cannot help (`can_help: false`):**
- Try simplifying the query yourself (fewer variables, simpler patterns)
- Consider breaking the query into smaller, focused sub-queries manually

**After successful mapping optimization (`can_help: true`, `success: true`):**
- Re-test your query with `test_candidate_query`
- If it STILL times out and the SQL shows NO self-join issues:
  → Increase `timeout_seconds` (up to 120s) as a last resort
  → Example: `test_candidate_query(query=..., dataset=..., timeout_seconds=90)`
- Only use increased timeout when mappings are already optimized!

**SYNTAX ERROR:**
- Fix the error in your query
- Generate corrected query with `status: "PENDING"`

**EMPTY RESULTS:**
- Check if FILTER is too restrictive
- Generate adjusted query with `status: "PENDING"`

## Guidelines for SPARQL Generation

1. Use only classes and properties from the provided schema context
2. Ensure all prefixes are properly declared
3. Use appropriate variable names that reflect the data
4. Apply filters and constraints as needed - USE sample_property_values to verify values first!

## Guidelines for Endpoint Selection

1. Match the query concepts to the dataset descriptions and keywords
2. If the query involves concepts from multiple datasets, select all relevant endpoints
3. Do NOT query endpoints that are clearly unrelated to the question
4. Prefer precision over recall - only select endpoints that are likely to have relevant data

## Multi-Dataset Queries

When a question requires data from MULTIPLE datasets:
1. Use UNION to combine sub-queries for each dataset into ONE query
2. Each UNION branch uses the classes/properties from ONE dataset's ontology
3. Use harmonized variable names across all branches (e.g., ?entity, ?name, ?count)

## Output Format

### When generating a query to test (status=PENDING):
```json
{{
  "status": "PENDING",
  "query": "SELECT ...",
  "target_endpoints": ["http://..."],
  "target_datasets": ["dataset_id"],
  "reasoning": "Why this query should answer the question"
}}
```

### When results are verified correct (status=DONE):
```json
{{
  "status": "DONE",
  "query": "SELECT ...",
  "target_endpoints": ["http://..."],
  "target_datasets": ["dataset_id"],
  "reasoning": "Query structure explanation",
  "result_verification": "The sample results show X, Y, Z which correctly answers the user's question about ..."
}}
```

### If you need more schema elements (status=NEED_TRIPLE):
```json
{{
  "status": "NEED_TRIPLE",
  "needed_triple": "Description of missing schema element",
  "reasoning": "Why this is needed"
}}
```

### When delegating (TIMEOUT with self-joins):
DO NOT output JSON. Instead, make a TOOL CALL to the delegation function.
You MUST invoke the tool - do not just write text saying you will delegate.

Example tool call:
```
delegate_to_mapping_optimizer(
    sparql_query="<the timed-out query>",
    dataset="<the dataset id>",
    sparql_agent_reasoning="Self-joins detected on TABLE_X (v1, v2 aliases)",
    generated_sql="<copy SQL from timeout feedback>"
)
```

**CRITICAL:**
- You can ONLY output `status: "DONE"` AFTER you have received and verified execution results!
- First query must always be `status: "PENDING"`
- `status: "DONE"` requires `result_verification` explaining why the results are correct

Be precise and avoid hallucinating properties or classes not in the schema.

IMPORTANT: You have a maximum of {max_tool_calls} tool calls. Use them wisely."""


SYSTEM_PROMPT_WITH_ENDPOINTS = """You are an expert SPARQL query generator for a Dataspace environment.

Given a natural language question, relevant schema triples, and available dataset endpoints, you must:
1. Generate a valid SPARQL query
2. Select the appropriate endpoint(s) to execute the query against

Available Datasets:
{endpoints_registry}

Guidelines for SPARQL Generation:
1. Use only classes and properties from the provided schema context
2. Ensure all prefixes are properly declared
3. Use appropriate variable names that reflect the data
4. Apply filters and constraints as needed
5. Consider whether SELECT, ASK, or CONSTRUCT is appropriate

Guidelines for Endpoint Selection:
1. Match the query concepts to the dataset descriptions and keywords
2. If the query involves concepts from multiple datasets, select all relevant endpoints
3. Do NOT query endpoints that are clearly unrelated to the question
4. Prefer precision over recall - only select endpoints that are likely to have relevant data

Output your response as JSON with these fields:
- query: The complete SPARQL query
- target_endpoints: List of endpoint URLs to query (from the available datasets above)
- target_datasets: List of dataset IDs (from the available datasets above)
- reasoning: Brief explanation of your query structure AND why you selected these endpoints
- confidence: Your confidence in this query (0.0 to 1.0)

Be precise and avoid hallucinating properties or classes not in the schema."""


SYSTEM_PROMPT = """You are an expert SPARQL query generator. Given a natural language question and relevant schema triples, generate a valid SPARQL query.

Guidelines:
1. Use only classes and properties from the provided schema context
2. Ensure all prefixes are properly declared
3. Use appropriate variable names that reflect the data
4. Apply filters and constraints as needed
5. Consider whether SELECT, ASK, or CONSTRUCT is appropriate

Output your response as JSON with these fields:
- query: The complete SPARQL query
- reasoning: Brief explanation of your query structure
- confidence: Your confidence in this query (0.0 to 1.0)

Be precise and avoid hallucinating properties or classes not in the schema."""


def format_registry_for_prompt() -> str:
    """Format the dataset registry for inclusion in the prompt.

    Uses get_endpoint_for_dataset() to respect the current size setting (small/large).
    """
    registry = load_dataset_registry()
    lines = []
    for dataset in registry.get("datasets", []):
        # Use get_endpoint_for_dataset to get the correct endpoint for current size setting
        # This respects DATASET_SIZE env var (small vs large)
        endpoint = get_endpoint_for_dataset(dataset['id']) or dataset['sparql_endpoint']
        lines.append(f"""
- ID: {dataset['id']}
  Name: {dataset['name']}
  Endpoint: {endpoint}
  Description: {dataset['description']}
  Keywords: {', '.join(dataset.get('keywords', []))}
  Concepts: {', '.join(dataset.get('concepts', []))}
""")
    return "\n".join(lines)


class SPARQLGenerationAgent:
    """Agent for generating SPARQL queries from natural language with endpoint selection.

    Now supports tool calls to inspect actual data values before generating queries.
    """

    def __init__(
        self,
        llm_model: str = "deepseek-chat",
        enable_endpoint_selection: bool = True,
        enable_tools: bool = True,
        max_tool_calls: int = 7,  # Allow enough calls for testing + multi-step delegation
    ):
        """
        Initialize the SPARQL generation agent.

        Args:
            llm_model: The LLM model to use
            enable_endpoint_selection: Whether to enable endpoint selection (Dataspace mode)
            enable_tools: Whether to enable tool calls for data inspection
            max_tool_calls: Maximum number of tool calls allowed per generation
        """
        settings = get_settings()
        self.enable_endpoint_selection = enable_endpoint_selection
        self.enable_tools = enable_tools
        self.max_tool_calls = max_tool_calls

        # Check if this is an Ollama model (qwen, llama, deepseek-r1 local, etc.)
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
            from src.config import get_openai_reasoning_kwargs
            self.llm = ChatOpenAI(
                model=llm_model,
                api_key=settings.openai_api_key,
                **get_openai_reasoning_kwargs(llm_model),
            )

        if enable_endpoint_selection:
            self.parser = JsonOutputParser(pydantic_object=SPARQLOutputWithEndpoints)
        else:
            self.parser = JsonOutputParser(pydantic_object=SPARQLOutput)

        # Set up tools
        self.tools = SPARQL_GENERATION_TOOLS if enable_tools else []
        self.llm_with_tools = self.llm.bind_tools(self.tools) if self.tools else self.llm

        self.tracer = get_tracer()
        self.llm_callback = create_llm_callback("generation", "sparql_generation")

    def _summarize_result(self, result: Any) -> dict:
        """Summarize tool result for tracing."""
        if isinstance(result, list):
            return {"type": "list", "count": len(result)}
        elif isinstance(result, dict):
            return {"type": "dict", "keys": list(result.keys())[:5]}
        elif isinstance(result, str):
            return {"type": "string", "length": len(result)}
        return {"type": type(result).__name__}

    async def _generate_with_tools(
        self,
        messages: list,
        variant_index: int = 0,
    ) -> str:
        """Generate a query with potential tool calls for data inspection.

        Legacy method - returns only the response content.
        Note: Does not support automatic multi-step delegation (no user_query/schema_context).
        """
        content, _, _ = await self._generate_with_tools_persistent(messages, variant_index)
        return content

    async def _generate_with_tools_persistent(
        self,
        messages: list[BaseMessage],
        variant_index: int = 0,
        user_query: str = "",
        schema_context: str = "",
    ) -> tuple[str, list[BaseMessage], dict | None]:
        """Generate a query with tool calls, returning updated message history.

        This version maintains the full conversation history for persistence
        across iterations, allowing the agent to learn from previous attempts.

        The agent decides on timeout handling:
        - Option 1: Try a different query
        - Option 2: Delegate to Mapping Optimizer Agent
        - Option 3: Delegate to Multi-Step Agent

        Returns:
            Tuple of (response_content, updated_messages, delegation_result)
            - delegation_result is None unless the agent called a delegation tool
        """
        # Work with a copy to avoid mutating the input
        messages = list(messages)
        tool_call_count = 0
        timeout_count = 0  # Track timeouts for context
        last_failed_query = ""  # Track the query that timed out

        # Track ALL queries tested (for context if agent delegates)
        # Each entry: {"query": str, "status": "success"|"timeout"|"error", "count": int, "error_msg": str|None}
        query_history: list[dict] = []

        while tool_call_count < self.max_tool_calls:
            response = await self.llm_with_tools.ainvoke(
                messages, config={"callbacks": [self.llm_callback]}
            )
            messages.append(response)
            emit_agent_message(self.tracer, response, "sparql_agent")

            # If no tool calls, we have the final response
            if not response.tool_calls:
                break

            # Execute tool calls
            for tool_call in response.tool_calls:
                if tool_call_count >= self.max_tool_calls:
                    limit_msg = ToolMessage(
                        content="LIMIT REACHED: Maximum tool calls exceeded. Generate your query NOW.",
                        tool_call_id=tool_call["id"],
                        name="system",
                    )
                    messages.append(limit_msg)
                    emit_agent_message(self.tracer, limit_msg, "sparql_agent")
                    continue

                tool_call_count += 1
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                print(f"[DEBUG] SPARQL Agent executing tool: '{tool_name}'")

                # Emit tool call event
                self.tracer.emit(
                    TraceEvent(
                        event_type=TraceEventType.TOOL_CALL,
                        phase="generation",
                        agent="sparql_agent",
                        data={
                            "tool_name": tool_name,
                            "tool_args": tool_args,
                            "variant_index": variant_index,
                        },
                    )
                )

                # Find and execute tool
                tool_fn = next(
                    (t for t in self.tools if t.name == tool_name),
                    None,
                )

                if not tool_fn:
                    available_tools = [t.name for t in self.tools]
                    error_msg = f"ERROR: Tool '{tool_name}' not available. Available: {available_tools}"
                    not_found_msg = ToolMessage(content=error_msg, tool_call_id=tool_call["id"], name="error")
                    messages.append(not_found_msg)
                    emit_agent_message(self.tracer, not_found_msg, "sparql_agent")
                    continue

                try:
                    result = tool_fn.invoke(tool_args)

                    # === QUERY HISTORY TRACKING & TIMEOUT DETECTION ===
                    if tool_name == "test_candidate_query":
                        result_str_check = str(result)
                        tested_query = tool_args.get("query", "")
                        is_timeout = "TIMEOUT" in result_str_check.upper() or "timed out" in result_str_check.lower()

                        # Track ALL test_candidate_query results for Multi-Step Agent
                        history_entry = {
                            "query": tested_query,
                            "status": "timeout" if is_timeout else "unknown",
                            "count": 0,
                            "error_msg": None,
                        }

                        # Try to extract result count from successful queries
                        if not is_timeout:
                            try:
                                import json as json_parse
                                result_data = json_parse.loads(result) if isinstance(result, str) else result
                                if result_data.get("success"):
                                    history_entry["status"] = "success"
                                    history_entry["count"] = result_data.get("result_count", len(result_data.get("results", [])))
                                else:
                                    history_entry["status"] = "error"
                                    history_entry["error_msg"] = result_data.get("error", "Unknown error")
                            except (json.JSONDecodeError, TypeError, AttributeError):
                                history_entry["status"] = "error"
                                history_entry["error_msg"] = "Could not parse result"

                        query_history.append(history_entry)

                        if is_timeout:
                            timeout_count += 1
                            last_failed_query = tested_query

                            self.tracer.emit(
                                TraceEvent(
                                    event_type=TraceEventType.INFO,
                                    phase="generation",
                                    agent="sparql_agent",
                                    data={
                                        "message": f"Timeout #{timeout_count} detected",
                                        "query_preview": last_failed_query[:100],
                                    },
                                )
                            )

                            # Agent decides what to do - we just track the timeout
                            # The agent will see this in the tool result and decide:
                            # 1. Try a different query
                            # 2. Call delegate_to_mapping_optimizer
                            # 3. Call delegate_to_multi_step_agent
                    # === END TIMEOUT DETECTION ===

                    self.tracer.emit(
                        TraceEvent(
                            event_type=TraceEventType.TOOL_RESULT,
                            phase="generation",
                            agent="sparql_agent",
                            data={
                                "tool_name": tool_name,
                                "result_summary": self._summarize_result(result),
                                "success": True,
                            },
                        )
                    )

                    # === DELEGATION TOOL DETECTION ===
                    # When a delegation tool is called, parse the result and return immediately
                    if tool_name == "delegate_to_multi_step_agent":
                        try:
                            delegation_result = json.loads(result) if isinstance(result, str) else result
                            logger.info(f"Multi-Step delegation completed: success={delegation_result.get('success')}")

                            # Add the tool result to messages for context
                            tool_msg = ToolMessage(content=str(result), tool_call_id=tool_call["id"], name=tool_name)
                            messages.append(tool_msg)
                            emit_agent_message(self.tracer, tool_msg, "sparql_agent")

                            # Return immediately with the delegation result
                            return "", messages, {"type": "multi_step", "result": delegation_result}
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse multi-step result: {e}")
                            # Continue with normal flow if parsing fails

                    if tool_name == "delegate_to_mapping_optimizer":
                        try:
                            delegation_result = json.loads(result) if isinstance(result, str) else result
                            can_help = delegation_result.get("can_help", False)
                            success = delegation_result.get("success", False)
                            logger.info(f"Mapping Optimizer delegation completed: success={success}, can_help={can_help}")

                            # Add the tool result to messages
                            tool_msg = ToolMessage(content=str(result), tool_call_id=tool_call["id"], name=tool_name)
                            messages.append(tool_msg)
                            emit_agent_message(self.tracer, tool_msg, "sparql_agent")

                            # Always return delegation result - let orchestrator decide what to do
                            # - can_help=True, success=True: Orchestrator triggers query retry
                            # - can_help=False: Orchestrator provides feedback, SPARQL agent continues
                            # - success=False: Error occurred, orchestrator handles it
                            return "", messages, {"type": "mapping_optimizer", "result": delegation_result}
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse mapping optimizer result: {e}")
                            # Return error result
                            return "", messages, {"type": "mapping_optimizer", "result": {
                                "success": False,
                                "can_help": False,
                                "error": f"Failed to parse result: {e}",
                                "raw_result": str(result)[:500]
                            }}
                    # === END DELEGATION TOOL DETECTION ===

                    # Format result for message
                    result_str = str(result)
                    if len(result_str) > 2000:
                        result_str = result_str[:2000] + "...[truncated]"

                    remaining = self.max_tool_calls - tool_call_count

                    # Add warning when running low on tool calls
                    if remaining == 0:
                        result_str += "\n\nWARNING: NO TOOL CALLS REMAINING - You MUST generate your final SPARQL query NOW!"
                    elif remaining == 1:
                        result_str += "\n\nWARNING: LAST TOOL CALL - After this, you must generate your final query. Use this call wisely (e.g., test your query with test_candidate_query)."
                    elif remaining == 2:
                        result_str += f"\n\nWARNING: Only {remaining} tool calls remaining. Start finalizing your approach - consider testing your query soon."
                    else:
                        result_str += f"\n\n[{remaining} tool calls remaining]"

                    tool_msg = ToolMessage(content=result_str, tool_call_id=tool_call["id"], name=tool_name)
                    messages.append(tool_msg)
                    emit_agent_message(self.tracer, tool_msg, "sparql_agent")

                except Exception as e:
                    self.tracer.emit(
                        TraceEvent(
                            event_type=TraceEventType.ERROR,
                            phase="generation",
                            agent="sparql_agent",
                            data={"tool_name": tool_name, "error": str(e)},
                        )
                    )
                    remaining = self.max_tool_calls - tool_call_count
                    error_msg = f"Error: {e}"
                    if remaining <= 1:
                        error_msg += f"\n\nWARNING: Only {remaining} tool call(s) remaining!"
                    error_tool_msg = ToolMessage(content=error_msg, tool_call_id=tool_call["id"], name=tool_name)
                    messages.append(error_tool_msg)
                    emit_agent_message(self.tracer, error_tool_msg, "sparql_agent")

        # Agent decides on delegation - no automatic fallback
        # The agent can call delegate_to_mapping_optimizer or delegate_to_multi_step_agent
        # as tools if it decides to delegate

        # Get final response content
        final_response = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                final_response = normalize_content(msg.content)
                break

        # If still no response after tool calls, force one
        if not final_response:
            force_message = "Generate your SPARQL query NOW as JSON. No more tool calls allowed."
            messages.append(HumanMessage(content=force_message))

            forced_response = await self.llm.ainvoke(
                messages, config={"callbacks": [self.llm_callback]}
            )
            messages.append(forced_response)
            final_response = normalize_content(forced_response.content)

        return final_response, messages, None  # No multi-step delegation occurred

    async def generate(
        self,
        user_query: str,
        schema_context: list[str],
        k: int = 1,  # Changed default to 1 for iterative mode
        orchestrator_feedback: str = "",
        failed_approaches: list[str] | None = None,
        previous_attempts: list[dict] | None = None,  # NEW: History of attempts
        semantic_validation_feedback: str = "",  # NEW: Semantic validation feedback
        conversation_history: list[dict] | None = None,  # NEW: Persistent LLM message history
    ) -> tuple[list[dict], list[dict]]:
        """
        Generate k SPARQL query variants with endpoint selection.

        In iterative mode (k=1), uses previous_attempts, semantic feedback,
        and persistent conversation history for better context.

        Args:
            user_query: Natural language query
            schema_context: List of relevant schema triples
            k: Number of variants to generate (default 1 for iterative mode)
            orchestrator_feedback: Feedback from OrchestratorLLM about previous attempts
            failed_approaches: List of previously failed query approaches to avoid
            previous_attempts: List of previous query attempts with feedback (NEW)
            semantic_validation_feedback: Feedback from semantic validation (NEW)
            conversation_history: Serialized LLM messages from previous iterations (NEW)

        Returns:
            Tuple of (results, updated_messages):
            - results: List of dicts with query, reasoning, confidence, and target_endpoints
            - updated_messages: Serialized LLM messages for next iteration
        """
        from datetime import datetime
        agent_start_time = datetime.now()

        # Emit agent start event for timing tracking
        self.tracer.emit(
            TraceEvent(
                event_type=TraceEventType.AGENT_START,
                phase="generation",
                agent="sparql_agent",
                data={"schema_triples_count": len(schema_context)},
            )
        )

        if failed_approaches is None:
            failed_approaches = []
        if previous_attempts is None:
            previous_attempts = []

        context_str = "\n".join(schema_context)

        # Emit schema context for detailed debugging
        self.tracer.emit(
            TraceEvent(
                event_type=TraceEventType.SCHEMA_CONTEXT,
                phase="generation",
                agent="sparql_agent",
                data={
                    "triples": schema_context,
                    "triples_count": len(schema_context),
                    "context_preview": context_str[:2000] if len(context_str) > 2000 else context_str,
                },
            )
        )

        # Build system prompt
        if self.enable_tools and self.tools:
            registry_str = format_registry_for_prompt()
            system_prompt = SYSTEM_PROMPT_WITH_TOOLS.format(
                endpoints_registry=registry_str,
                max_tool_calls=self.max_tool_calls,
            )
        elif self.enable_endpoint_selection:
            registry_str = format_registry_for_prompt()
            system_prompt = SYSTEM_PROMPT_WITH_ENDPOINTS.format(endpoints_registry=registry_str)
        else:
            system_prompt = SYSTEM_PROMPT

        # Track messages for this generation (will be returned for persistence)
        final_messages: list[BaseMessage] = []

        # Check if we have conversation history to continue from
        if conversation_history and len(conversation_history) > 0:
            # Continue from previous conversation - deserialize messages
            final_messages = deserialize_messages(conversation_history)

            # Add feedback as new human message to continue the conversation
            feedback_parts = []
            if orchestrator_feedback:
                feedback_parts.append(f"## Orchestrator Feedback:\n{orchestrator_feedback}")
            if semantic_validation_feedback:
                feedback_parts.append(f"## Semantic Validation:\n{semantic_validation_feedback}")

            if feedback_parts:
                feedback_content = "\n\n".join(feedback_parts)
                # Check if feedback suggests delegation (timeout case)
                if "delegate" in feedback_content.lower() and "timeout" in feedback_content.lower():
                    feedback_content += "\n\nFollow the delegation recommendation above, or if you believe the query can be optimized, generate an improved version."
                else:
                    feedback_content += "\n\nPlease generate an improved SPARQL query based on this feedback."
                feedback_human_msg = HumanMessage(content=feedback_content)
                final_messages.append(feedback_human_msg)
                emit_agent_message(self.tracer, feedback_human_msg, "sparql_agent")
        else:
            # First iteration - build fresh prompt
            prompt = f"""Schema context:
{context_str}

User question: {user_query}
"""

            # Add previous attempts history (legacy support for non-persistent mode)
            if previous_attempts:
                prompt += "\n## Previous Attempts (DO NOT repeat these mistakes):\n"
                for i, attempt in enumerate(previous_attempts[-3:], 1):
                    prompt += f"\n### Attempt {i}:\n"
                    query = attempt.get('query', 'N/A')
                    if len(query) > 200:
                        query = query[:200] + "..."
                    prompt += f"Query: {query}\n"
                    prompt += f"Error/Feedback: {attempt.get('feedback', 'N/A')}\n"

            # Add semantic validation feedback if available
            if semantic_validation_feedback:
                prompt += f"""
## Semantic Validation Feedback:
{semantic_validation_feedback}

Use this feedback to fix any semantic errors in your query.
"""

            # Add orchestrator feedback if available
            if orchestrator_feedback:
                prompt += f"""
## Orchestrator Feedback:
{orchestrator_feedback}

Take this feedback into account when generating the query.
"""

            # Add failed approaches to avoid (legacy support)
            if failed_approaches and not previous_attempts:
                prompt += "\n## AVOID THESE APPROACHES (they have failed before):\n"
                for i, approach in enumerate(failed_approaches[-3:], 1):
                    truncated = approach[:150] + "..." if len(approach) > 150 else approach
                    prompt += f"{i}. {truncated}\n"
                prompt += "\nGenerate a DIFFERENT approach that avoids these issues.\n"

            prompt += f"""
Generate a SPARQL query to answer this question using the provided schema.
{self.parser.get_format_instructions()}"""

            sys_msg = SystemMessage(content=system_prompt)
            human_msg = HumanMessage(content=prompt)
            final_messages = [sys_msg, human_msg]

            # Emit initial messages for context visualization
            emit_agent_message(self.tracer, sys_msg, "sparql_agent")
            emit_agent_message(self.tracer, human_msg, "sparql_agent")

        async def generate_single_variant(variant_index: int, messages: list[BaseMessage]) -> tuple[dict, list[BaseMessage]]:
            """Generate a single query variant and return updated messages."""
            working_messages = messages.copy()

            # Add variation hint for subsequent queries
            if variant_index > 0:
                working_messages.append(
                    HumanMessage(
                        content=f"Generate variant {variant_index + 1}: Try a SIGNIFICANTLY different approach or structure."
                    )
                )

            try:
                # Use tool-enabled generation if tools are available
                if self.enable_tools and self.tools:
                    response_content, working_messages, delegation_result = await self._generate_with_tools_persistent(
                        working_messages,
                        variant_index,
                        user_query=user_query,
                        schema_context=context_str,
                    )

                    # Check if delegation occurred (Multi-Step or Mapping Optimizer)
                    if delegation_result is not None:
                        delegation_type = delegation_result.get("type", "")
                        result_data = delegation_result.get("result", {})

                        if delegation_type == "multi_step":
                            # Return special result indicating Multi-Step delegation
                            return {
                                "query": "",
                                "target_endpoints": [],
                                "target_datasets": [],
                                "reasoning": "Query delegated to Multi-Step Agent for decomposition",
                                "confidence": 0.0,
                                "status": "MULTI_STEP_RESULT",
                                "multi_step_result": result_data,
                            }, working_messages

                        elif delegation_type == "mapping_optimizer":
                            # Return special result indicating Mapping Optimizer delegation
                            return {
                                "query": "",
                                "target_endpoints": [],
                                "target_datasets": [],
                                "reasoning": f"Mapping Optimizer optimized: {result_data.get('optimized_tables', [])}",
                                "confidence": 0.0,
                                "status": "MAPPING_OPTIMIZER_RESULT",
                                "mapping_optimizer_result": result_data,
                            }, working_messages
                else:
                    response = await self.llm.ainvoke(
                        working_messages, config={"callbacks": [self.llm_callback]}
                    )
                    working_messages.append(response)
                    response_content = normalize_content(response.content)

                # Try standard parser first, fall back to JSON extraction
                try:
                    parsed = self.parser.parse(response_content)
                except Exception:
                    # LLM may have output text before JSON - try extraction
                    parsed = extract_json_from_response(response_content)

                # Ensure backwards compatibility
                if not self.enable_endpoint_selection:
                    parsed["target_endpoints"] = []
                    parsed["target_datasets"] = []

                # Note: MULTI_STEP_RESULT status is only set by automatic delegation
                # after 3 timeouts (see timeout detection code above)

                return parsed, working_messages
            except Exception as e:
                self.tracer.emit(
                    TraceEvent(
                        event_type=TraceEventType.ERROR,
                        phase="generation",
                        agent="sparql_agent",
                        data={"error": str(e), "variant_index": variant_index},
                    )
                )
                return {
                    "query": "",
                    "target_endpoints": [],
                    "target_datasets": [],
                    "reasoning": f"Error: {e}",
                    "confidence": 0.0,
                }, working_messages

        # Generate single variant (k=1 for iterative mode with persistent history)
        # Note: For k>1, we'd need separate message tracks per variant
        if k == 1:
            result, updated_messages = await generate_single_variant(0, final_messages)

            # Emit agent end event with timing
            agent_end_time = datetime.now()
            agent_duration_ms = (agent_end_time - agent_start_time).total_seconds() * 1000
            self.tracer.emit(
                TraceEvent(
                    event_type=TraceEventType.AGENT_END,
                    phase="generation",
                    agent="sparql_agent",
                    duration_ms=agent_duration_ms,
                    data={"status": result.get("status", "DONE")},
                )
            )

            return [result], serialize_messages(updated_messages)
        else:
            # For k>1, run in parallel but each starts from same base messages
            # Messages from first successful variant are returned
            tasks = [generate_single_variant(i, final_messages) for i in range(k)]
            variant_results = await asyncio.gather(*tasks)
            results = [r[0] for r in variant_results]

            # Emit agent end event with timing
            agent_end_time = datetime.now()
            agent_duration_ms = (agent_end_time - agent_start_time).total_seconds() * 1000
            self.tracer.emit(
                TraceEvent(
                    event_type=TraceEventType.AGENT_END,
                    phase="generation",
                    agent="sparql_agent",
                    duration_ms=agent_duration_ms,
                    data={"variant_count": k},
                )
            )

            # Return messages from first variant
            return results, serialize_messages(variant_results[0][1])

    async def generate_single(
        self,
        user_query: str,
        schema_context: list[str],
    ) -> dict:
        """Generate a single SPARQL query with endpoint selection.

        Note: This is a convenience method that discards the message history.
        For iterative generation with persistent history, use generate() directly.
        """
        results, _ = await self.generate(user_query, schema_context, k=1)
        return results[0] if results else {
            "query": "",
            "target_endpoints": [],
            "target_datasets": [],
            "reasoning": "No result",
            "confidence": 0.0,
        }

    async def generate_with_explicit_datasets(
        self,
        user_query: str,
        schema_context: list[str],
        dataset_ids: list[str],
    ) -> dict:
        """
        Generate a SPARQL query for specific datasets.

        Args:
            user_query: Natural language query
            schema_context: List of relevant schema triples
            dataset_ids: List of dataset IDs to target

        Returns:
            Dict with query and metadata
        """
        result = await self.generate_single(user_query, schema_context)

        # Override target datasets with explicit ones
        # Use get_endpoint_for_dataset to respect current size setting (small/large)
        target_endpoints = []
        for dataset_id in dataset_ids:
            endpoint = get_endpoint_for_dataset(dataset_id)
            if endpoint:
                target_endpoints.append(endpoint)

        result["target_datasets"] = dataset_ids
        result["target_endpoints"] = target_endpoints

        return result
