"""
Grep-based retrieval agent.

Uses regex/keyword-based search to find relevant schema information.
This agent ONLY has access to grep-based tools for fair comparison.

In Dataspace mode, searches all datasets and extracts endpoint information.
"""

from typing import Any

import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

from src.config import get_settings, normalize_content
from src.config import get_openrouter_kwargs
from src.tools.retrieval_tools import GREP_TOOLS
from src.tracing import TraceEvent, TraceEventType, get_tracer
from src.tracing.llm_callback import create_llm_callback
from src.agents.retrieval.graph_validator import validate_tripel_graph, serialize_validated_graph


def emit_agent_message(tracer, msg: BaseMessage, agent_name: str = "grep_retrieval_agent") -> None:
    """Emit an AGENT_MESSAGE event for context visualization.

    Args:
        tracer: The TracingManager instance
        msg: The LangChain message to emit
        agent_name: Name of the agent emitting the message
    """
    data = {
        "message_type": msg.__class__.__name__,
        "content": normalize_content(msg.content)[:2000] if msg.content else "",
    }

    if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
        data["tool_calls"] = msg.tool_calls

    if isinstance(msg, ToolMessage):
        data["tool_call_id"] = getattr(msg, 'tool_call_id', '')
        data["tool_name"] = getattr(msg, 'name', '')

    tracer.emit(TraceEvent(
        event_type=TraceEventType.AGENT_MESSAGE,
        phase="retrieval",
        agent=agent_name,
        data=data,
    ))


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
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            msg_dict["tool_calls"] = msg.tool_calls
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
            messages.append(HumanMessage(content=content))
    return messages


SYSTEM_PROMPT = """You are a schema retrieval agent that extracts relevant tripels from semantic models using keyword search.

## Your Goal
Extract tripels (class-property-range) needed to answer the query. Output goes to SPARQL generation.

## Tools

1. **grep_classes(term, dataset?)** - Find classes. Returns hierarchy info (parent/child classes).
2. **grep_properties(term, dataset?)** - Find properties by keyword (e.g., "advisor", "name", "operator").
3. **find_shortest_path(classes, dataset?)** - Find object properties connecting classes.
4. **read_semantic_model(file, dataset)** - Read all properties for a class.
5. **grep_data_values(class, property, keyword, dataset?, prefix?)** - Find actual data values.
6. **get_class_hierarchy(class, dataset?, direction?)** - Explore class inheritance.
7. **list_all_classes(dataset?)** - List all classes in a dataset (for domain context).

## 4-Step Workflow

### Step 1: Analyze Query Terms

Categorize each term:
- **CONCEPTS** (students, books, trips) → grep_classes()
- **NAMED ENTITIES** (person names, company names, place names) → grep_data_values()
- **NUMERIC FILTERS** (under 50, in 2023) → schema only, SPARQL handles these

Named entities are NOT in schema - use grep_data_values() to find actual values!

### Step 2: Find Classes and Connections
```
grep_classes("Book")  → Find Book class
grep_classes("Author") → Find Author class
find_shortest_path(["Book", "Author"]) → Get connection tripels
```

### Step 3: Get Datatype Properties
```
read_semantic_model("book.ttl", "LIBRARY") → See all Book properties
```
Extract only the datatype properties needed for the query (e.g., title, year).

### Step 4: Find Entity Values (if needed)
```
grep_data_values("lib:Author", "lib:name", "Shakespeare", dataset="LIBRARY")
→ Returns: {"success": True, "count": 1, "values": ["William Shakespeare"]}
```
Inject found values into tripels.

## Output Format

Your final message must contain:

**Part 1: Matching Justification**
```
MATCHING:
- "books" → lib:Book (class match)
- "written by" → lib:writtenBy (property match)
- "Shakespeare" → "William Shakespeare" (entity value match)
```

**Part 2: Turtle Schema**
```turtle
PREFIX lib: <http://example.org/library#>
lib:Book lib:writtenBy lib:Author .
lib:Author lib:name "William Shakespeare" .
```

## Important Notes

- **Domain context**: If all classes belong to one domain, adjectives may be implicit.
- **Hierarchy**: grep_classes() returns parent/child classes. Use get_class_hierarchy() for details.
- **MAX 10 tool calls** - plan efficiently!
- **Final message must contain tripels** - never end with a tool call!
- Use ONLY prefixes and URIs from tool results."""


class GrepRetrievalAgent:
    """Agent that uses grep-based search for schema retrieval.

    In Dataspace mode, searches all available datasets and extracts
    endpoint information from the results.
    """

    def __init__(self, llm_model: str = "deepseek-chat"):
        """Initialize the grep retrieval agent."""
        settings = get_settings()

        # Check if this is an Ollama model (qwen, llama, deepseek-r1 local, etc.)
        ollama_patterns = ["qwen", "llama", "deepseek-r1", "mistral", "gemma", "phi"]
        is_ollama = any(p in llm_model.lower() for p in ollama_patterns)

        if llm_model.startswith("openrouter/"):
            self.llm = ChatOpenAI(
                model=llm_model[len("openrouter/"):],
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                temperature=0.1,
                **get_openrouter_kwargs(llm_model),
            )
        elif llm_model.startswith("openai/"):
            self.llm = ChatOpenAI(
                model=llm_model,
                api_key=os.getenv("KI4BUW_API_KEY", ""),
                base_url=os.getenv("KI4BUW_BASE_URL", "https://llm.ki4buw.de/v1"),
                temperature=0.1,
            )
        elif is_ollama:
            ollama_base_url = os.getenv("REMOTE_OLLAMA_LLAMA31_70B_BASE_URL", "http://localhost:11434")
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

        self.tools = GREP_TOOLS
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tracer = get_tracer()
        self.llm_callback = create_llm_callback("retrieval", "grep_search")

    def _summarize_result(self, result: Any) -> dict:
        """Summarize tool result for tracing."""
        if isinstance(result, list):
            return {"type": "list", "count": len(result)}
        elif isinstance(result, dict):
            return {"type": "dict", "keys": list(result.keys())[:5]}
        elif isinstance(result, str):
            return {"type": "string", "length": len(result)}
        return {"type": type(result).__name__}

    async def retrieve(
        self,
        user_query: str,
        orchestrator_feedback: str = "",
        needed_triple_description: str | None = None,  # NEW: For NEED_TRIPLE from SPARQL agent
        conversation_history: list[dict] | None = None,  # NEW: Persistent LLM message history
    ) -> tuple[dict, list[dict]]:
        """
        Retrieve relevant schema information for a query.

        Searches all datasets in Dataspace mode and extracts endpoint information.
        Validates output to ensure a coherent graph with all referenced classes defined.

        Args:
            user_query: Natural language query from user
            orchestrator_feedback: Optional feedback from OrchestratorLLM about what to search for
            needed_triple_description: Semantic description of missing triple from SPARQL agent
            conversation_history: Serialized LLM messages from previous iterations

        Returns:
            Tuple of (result_dict, updated_messages):
            - result_dict: Dict with schema, retrieved_triples, mapping_files_used, detected_endpoints
            - updated_messages: Serialized LLM messages for next iteration
        """
        from datetime import datetime
        agent_start_time = datetime.now()

        # Emit agent start event for timing tracking
        self.tracer.emit(
            TraceEvent(
                event_type=TraceEventType.AGENT_START,
                phase="retrieval",
                agent="grep_retrieval_agent",
                data={"query_length": len(user_query)},
            )
        )

        # Check if we have conversation history to continue from
        if conversation_history and len(conversation_history) > 0:
            # Continue from previous conversation
            messages = deserialize_messages(conversation_history)

            # Add feedback as new message to continue the conversation
            feedback_parts = []
            if needed_triple_description:
                feedback_parts.append(f"""## TARGETED SEARCH - SPARQL Agent braucht spezifisches Schema

Der SPARQL Agent hat analysiert dass ein bestimmtes Tripel/Schema-Element fehlt:

"{needed_triple_description}"

Suche nach Properties/Classes die diese Beziehung modellieren koennten.""")
            if orchestrator_feedback:
                feedback_parts.append(f"## Orchestrator Feedback:\n{orchestrator_feedback}")

            if feedback_parts:
                feedback_msg = "\n\n".join(feedback_parts)
                feedback_msg += "\n\nBitte suche basierend auf diesem Feedback weiter."
                messages.append(HumanMessage(content=feedback_msg))
        else:
            # First iteration - build fresh prompt
            user_content = f"Find relevant schema elements for: {user_query}"

            if needed_triple_description:
                user_content += f"""

## TARGETED SEARCH MODE - SPARQL Agent braucht spezifisches Schema

Der SPARQL Agent hat analysiert dass ein bestimmtes Tripel/Schema-Element fehlt:

"{needed_triple_description}"

Deine Aufgabe:
1. Verstehe was der SPARQL Agent semantisch braucht
2. Suche nach Properties/Classes die diese Beziehung modellieren koennten
3. Das gesuchte Element koennte anders benannt sein als erwartet

Beispiel: Wenn der Agent "Property um Mitarbeiter nach Land zu filtern" braucht,
suche nach: worksIn, employedIn, country, location, basedIn, etc.
"""
            elif orchestrator_feedback:
                user_content += f"\n\nWICHTIG - Hinweis vom Orchestrator:\n{orchestrator_feedback}"

            from src.config import UNDER_INSTRUCTED, UNDERSPECIFIED_INSTRUCTION_RETRIEVAL
            sys_msg = SystemMessage(
                content=SYSTEM_PROMPT + (UNDERSPECIFIED_INSTRUCTION_RETRIEVAL if UNDER_INSTRUCTED else "")
            )
            human_msg = HumanMessage(content=user_content)
            messages = [sys_msg, human_msg]

            # Emit initial messages for context visualization
            emit_agent_message(self.tracer, sys_msg, "grep_retrieval_agent")
            emit_agent_message(self.tracer, human_msg, "grep_retrieval_agent")

        mapping_files: set[str] = set()
        detected_endpoints: set[str] = set()
        validated_schema: str = ""

        # Reduced from 3 to 2 - with forced response mechanism, fewer retries needed
        max_validation_attempts = 2
        validation_attempt = 0

        while validation_attempt < max_validation_attempts:
            validation_attempt += 1

            # Run agent loop (max 10 TOTAL tool calls per validation attempt)
            max_tool_calls = 10
            tool_call_count = 0

            while tool_call_count < max_tool_calls:
                remaining_calls = max_tool_calls - tool_call_count
                response = await self.llm_with_tools.ainvoke(
                    messages, config={"callbacks": [self.llm_callback]}
                )
                messages.append(response)
                emit_agent_message(self.tracer, response, "grep_retrieval_agent")

                if not response.tool_calls:
                    break

                # Execute each tool call (but respect the limit)
                for tool_call in response.tool_calls:
                    # Stop executing more tools if limit reached
                    if tool_call_count >= max_tool_calls:
                        # Don't execute, but still need to respond to the tool call
                        limit_msg = ToolMessage(
                            content="LIMIT REACHED: You have used all 10 tool calls. Output your tripels NOW as text.",
                            tool_call_id=tool_call["id"],
                            name="system",
                        )
                        messages.append(limit_msg)
                        emit_agent_message(self.tracer, limit_msg, "grep_retrieval_agent")
                        continue

                    tool_call_count += 1
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    # Emit tool call event
                    self.tracer.emit(
                        TraceEvent(
                            event_type=TraceEventType.TOOL_CALL,
                            phase="retrieval",
                            agent="grep_agent",
                            data={
                                "tool_name": tool_name,
                                "tool_args": tool_args,
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
                        error_msg = f"ERROR: Tool '{tool_name}' is NOT available! You can ONLY use: {available_tools}"
                        self.tracer.emit(
                            TraceEvent(
                                event_type=TraceEventType.ERROR,
                                phase="retrieval",
                                agent="grep_agent",
                                data={"tool_name": tool_name, "error": error_msg},
                            )
                        )
                        not_found_msg = ToolMessage(content=error_msg, tool_call_id=tool_call["id"], name="error")
                        messages.append(not_found_msg)
                        emit_agent_message(self.tracer, not_found_msg, "grep_retrieval_agent")
                        continue

                    try:
                        result = tool_fn.invoke(tool_args)

                        self.tracer.emit(
                            TraceEvent(
                                event_type=TraceEventType.TOOL_RESULT,
                                phase="retrieval",
                                agent="grep_agent",
                                data={
                                    "tool_name": tool_name,
                                    "result_summary": self._summarize_result(result),
                                    "result_data": result[:20] if isinstance(result, list) else result,
                                    "success": True,
                                },
                            )
                        )

                        # Extract metadata from results
                        self._extract_metadata(
                            tool_name, result, mapping_files, detected_endpoints
                        )

                        # Add tool result to messages with remaining calls info
                        result_str = str(result)
                        if len(result_str) > 3000:
                            result_str = result_str[:3000] + "...[truncated]"

                        # Add remaining calls warning (based on actual tool call count)
                        calls_after_this = max_tool_calls - tool_call_count
                        if calls_after_this == 0:
                            result_str += "\n\nWARNING: NO TOOL CALLS REMAINING - Output your tripels NOW!"
                        elif calls_after_this == 1:
                            result_str += "\n\nWARNING: LAST TOOL CALL - Output your tripels after this!"
                        elif calls_after_this <= 3:
                            result_str += f"\n\nWARNING: Only {calls_after_this} tool calls remaining! Output your tripels soon."
                        else:
                            result_str += f"\n\n[{calls_after_this} tool calls remaining]"

                        result_tool_msg = ToolMessage(content=result_str, tool_call_id=tool_call["id"], name=tool_name)
                        messages.append(result_tool_msg)
                        emit_agent_message(self.tracer, result_tool_msg, "grep_retrieval_agent")

                    except Exception as e:
                        self.tracer.emit(
                            TraceEvent(
                                event_type=TraceEventType.ERROR,
                                phase="retrieval",
                                agent="grep_agent",
                                data={"tool_name": tool_name, "error": str(e)},
                            )
                        )
                        calls_after_this = max_tool_calls - tool_call_count
                        error_msg = f"Error: {e}\n\n[{calls_after_this} tool calls remaining]"
                        error_tool_msg = ToolMessage(content=error_msg, tool_call_id=tool_call["id"], name=tool_name)
                        messages.append(error_tool_msg)
                        emit_agent_message(self.tracer, error_tool_msg, "grep_retrieval_agent")

            # Get the final text response from the agent
            final_response = ""
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    final_response = normalize_content(msg.content)
                    break

            if not final_response:
                # No text response - FORCE the agent to make a decision
                # This happens when agent used all tool calls without outputting tripels
                force_message = """STOP! You have used all your tool calls and must now make a decision.

Based on the schema information you gathered from your tool calls, you MUST now do ONE of these:

OPTION 1 - If you found relevant classes and properties:
Output the tripels in Turtle format:
```
PREFIX ...
ClassName propertyName RangeType .
```

OPTION 2 - If you could NOT find relevant schema elements:
Output exactly: UNANSWERABLE: [brief reason why the schema doesn't support this query]

You CANNOT make more tool calls. Decide NOW based on what you already found."""
                from src.config import UNDER_INSTRUCTED, UNDERSPECIFIED_FORCE_OPTION_RETRIEVAL
                if UNDER_INSTRUCTED:
                    force_message = force_message.replace(
                        "\n\nYou CANNOT make more tool calls",
                        UNDERSPECIFIED_FORCE_OPTION_RETRIEVAL + "\n\nYou CANNOT make more tool calls",
                    )

                force_human_msg = HumanMessage(content=force_message)
                messages.append(force_human_msg)
                emit_agent_message(self.tracer, force_human_msg, "grep_retrieval_agent")

                # Call LLM WITHOUT tools to force a text response
                forced_response = await self.llm.ainvoke(
                    messages, config={"callbacks": [self.llm_callback]}
                )
                messages.append(forced_response)
                emit_agent_message(self.tracer, forced_response, "grep_retrieval_agent")

                if forced_response.content:
                    final_response = normalize_content(forced_response.content)
                    self.tracer.emit(
                        TraceEvent(
                            event_type=TraceEventType.AGENT_ACTION,
                            phase="retrieval",
                            agent="grep_agent",
                            data={"action": "forced_response", "has_content": bool(final_response)},
                        )
                    )

            if not final_response:
                # Still no response after forcing - give up on this attempt
                break

            # UNDER instructed ablation: an UNDERSPECIFIED stop is not a triple graph, skip validation
            if final_response.strip().upper().startswith("UNDERSPECIFIED:"):
                validated_schema = final_response
                break

            # Validate the output graph
            validation_result = validate_tripel_graph(final_response)

            self.tracer.emit(
                TraceEvent(
                    event_type=TraceEventType.TOOL_RESULT,
                    phase="retrieval",
                    agent="grep_agent",
                    data={
                        "validation_attempt": validation_attempt,
                        "is_valid": validation_result.is_valid,
                        "tripel_count": validation_result.tripel_count,
                        "is_connected": validation_result.is_connected,
                        "syntax_error": validation_result.syntax_error,
                    },
                )
            )

            if validation_result.is_valid:
                # Success - serialize the validated graph
                validated_schema = serialize_validated_graph(validation_result)
                break
            else:
                # Validation failed - give feedback to agent
                if validation_attempt < max_validation_attempts:
                    validation_feedback_msg = HumanMessage(content=validation_result.feedback_message)
                    messages.append(validation_feedback_msg)
                    emit_agent_message(self.tracer, validation_feedback_msg, "grep_retrieval_agent")
                else:
                    # Max attempts reached - use what we have
                    validated_schema = final_response

        # Check if agent signaled UNANSWERABLE (missing schema) or, with UNDER_INSTRUCTED=1,
        # UNDERSPECIFIED (the question does not fix a criterion the answer depends on)
        is_unanswerable = False
        unanswerable_reason = ""
        stop_kind = ""
        if validated_schema and validated_schema.strip().upper().startswith("UNANSWERABLE:"):
            is_unanswerable = True
            stop_kind = "missing_schema"
            unanswerable_reason = validated_schema.split(":", 1)[1].strip() if ":" in validated_schema else "Schema does not support this query"
            validated_schema = ""  # Clear schema since it's not valid triples
        elif validated_schema and validated_schema.strip().upper().startswith("UNDERSPECIFIED:"):
            is_unanswerable = True
            stop_kind = "underspecified"
            unanswerable_reason = validated_schema.split(":", 1)[1].strip() if ":" in validated_schema else "Question does not fix a required criterion"
            validated_schema = ""

        result = {
            "status": ("UNDERSPECIFIED" if stop_kind == "underspecified" else "UNANSWERABLE") if is_unanswerable else "FOUND",
            "schema": validated_schema,
            "retrieved_triples": [validated_schema] if validated_schema else [],
            "mapping_files_used": list(mapping_files),
            "detected_endpoints": list(detected_endpoints),
            "unanswerable_reason": unanswerable_reason,
            "stop_kind": stop_kind,
        }

        # Emit agent end event with timing
        agent_end_time = datetime.now()
        agent_duration_ms = (agent_end_time - agent_start_time).total_seconds() * 1000
        self.tracer.emit(
            TraceEvent(
                event_type=TraceEventType.AGENT_END,
                phase="retrieval",
                agent="grep_retrieval_agent",
                duration_ms=agent_duration_ms,
                data={
                    "status": result["status"],
                    "triples_count": len(result["retrieved_triples"]),
                },
            )
        )

        return result, serialize_messages(messages)

    def _extract_metadata(
        self,
        tool_name: str,
        result: Any,
        mapping_files: set[str],
        detected_endpoints: set[str],
    ) -> None:
        """Extract mapping files and endpoints from tool results."""
        if tool_name == "grep_classes" and isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    if item.get("file"):
                        mapping_files.add(item["file"])
                    if item.get("sparql_endpoint"):
                        detected_endpoints.add(item["sparql_endpoint"])

        elif tool_name == "grep_properties" and isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    if item.get("file"):
                        mapping_files.add(item["file"])
                    if item.get("sparql_endpoint"):
                        detected_endpoints.add(item["sparql_endpoint"])

        elif tool_name == "grep_data_values" and isinstance(result, dict):
            if result.get("sparql_endpoint"):
                detected_endpoints.add(result["sparql_endpoint"])

        elif tool_name == "read_semantic_model" and isinstance(result, dict):
            if result.get("file"):
                mapping_files.add(result["file"])
            if result.get("sparql_endpoint"):
                detected_endpoints.add(result["sparql_endpoint"])

        # Handler for list_all_classes tool
        elif tool_name == "list_all_classes" and isinstance(result, list):
            for item in result:
                if isinstance(item, dict) and item.get("sparql_endpoint"):
                    detected_endpoints.add(item["sparql_endpoint"])

        # Handler for get_class_hierarchy tool
        elif tool_name == "get_class_hierarchy" and isinstance(result, dict):
            if result.get("sparql_endpoint"):
                detected_endpoints.add(result["sparql_endpoint"])
