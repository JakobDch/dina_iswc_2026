"""
Orchestrator agent for Text-to-SPARQL generation.

Implements a LangGraph StateGraph with TWO phases:
1. RETRIEVAL - Retrieval Agent finds relevant schema triples
2. GENERATION - SPARQL Agent generates, tests, and finalizes queries

Architecture (simplified):
- Retrieval Agent: Searches for schema elements
- SPARQL Agent: Generates queries, tests them, decides DONE/NEED_TRIPLE
- StateGraph: Rule-based routing between phases (no LLM decisions)
"""

import logging
import time as _time
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from src.config import get_settings, MAPPINGS_DIR, EMBEDDINGS_CACHE_DIR, set_llm_model
from src.agents.retrieval.grep_agent import GrepRetrievalAgent
from src.agents.retrieval.semantic_agent import SemanticRetrievalAgent
from src.agents.generation.sparql_agent import SPARQLGenerationAgent
from src.agents.orchestrator_llm import count_actual_triples
from src.tracing import TraceEvent, TraceEventType, get_tracer
from src.tools.retrieval_tools import load_dataset_registry

logger = logging.getLogger(__name__)


# =============================================================================
# Agent Memory - Persistent LLM Conversation History
# =============================================================================
#
# Each of the three LLM agents maintains its complete conversation history:
# - Retrieval Agent: Searches for schema elements
# - Generation Agent (SPARQL): Generates SPARQL queries
# - Orchestrator Agent: Coordinates and makes decisions
#
# Memory is stored as serialized LangChain messages (list[dict]) containing:
# - System prompts
# - Human/User messages
# - AI responses
# - Tool calls and their results
#
# This gives each agent full context of what it has said, done, and learned.
# =============================================================================

# Type alias for agent memory - serialized LLM messages
AgentMemory = list[dict]


# =============================================================================
# Helper Functions
# =============================================================================


def _derive_endpoints_from_triples(retrieved_triples: list[str]) -> list[str]:
    """
    Derive SPARQL endpoints from prefix URIs used in retrieved triples.

    This is a fallback mechanism when _extract_metadata() didn't capture
    endpoint information from tool results. It analyzes the prefixes used
    in the retrieved schema triples and maps them to endpoints via:
    1. Global prefix cache (short prefix -> full URI)
    2. Dataset registry (ontology_prefix URI -> sparql_endpoint)

    Uses the same logic as get_endpoints_for_query() but works with
    short prefixes (trn:) instead of full PREFIX declarations.

    Args:
        retrieved_triples: List of schema triple strings in Turtle format

    Returns:
        List of SPARQL endpoint URLs derived from the triple prefixes
    """
    import re
    import json

    if not retrieved_triples:
        return []

    # 1. Load prefix registry (short prefix -> full URI) from global cache
    prefix_cache = EMBEDDINGS_CACHE_DIR / "_global" / "prefixes.json"
    prefix_map = {}
    if prefix_cache.exists():
        try:
            with open(prefix_cache, encoding="utf-8") as f:
                prefix_map = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load prefix cache: {e}")

    # Fallback: use SchemaIndex if cache doesn't exist
    if not prefix_map:
        try:
            from src.tools.retrieval_tools import get_schema_index
            index = get_schema_index()
            if index._prefix_registry:
                prefix_map = index.get_all_prefixes()
        except Exception as e:
            logger.warning(f"Failed to get prefixes from SchemaIndex: {e}")

    if not prefix_map:
        logger.warning("No prefix map available, cannot derive endpoints from triples")
        return []

    # 2. Extract all prefixes used in the triples
    #    Matches: "trn:Route", "eno:Field", etc.
    prefix_pattern = re.compile(r'\b(\w+):\w+')
    used_prefixes = set()

    for triple_block in retrieved_triples:
        matches = prefix_pattern.findall(triple_block)
        used_prefixes.update(matches)

    # 3. Convert short prefixes to full URIs
    prefix_uris = set()
    for prefix in used_prefixes:
        if prefix in prefix_map:
            prefix_uris.add(prefix_map[prefix])

    if not prefix_uris:
        return []

    # 4. Match URIs against dataset registry (same logic as get_endpoints_for_query)
    # NOTE: Resolve endpoint via get_endpoint_for_dataset so slot_context /
    # dataset_size_context is respected. The registry stores static legacy URLs
    # (e.g. 8084) which do NOT exist under slot-based container isolation.
    from src.tools.sparql_tools import get_endpoint_for_dataset
    registry = load_dataset_registry()
    endpoints = set()

    for dataset in registry.get("datasets", []):
        ontology_prefix = dataset.get("ontology_prefix", "")
        if not ontology_prefix:
            continue

        for uri in prefix_uris:
            # Exact match or URI starts with ontology prefix
            if uri == ontology_prefix or uri.startswith(ontology_prefix.rstrip('#/')):
                resolved = get_endpoint_for_dataset(dataset["id"])
                if resolved:
                    endpoints.add(resolved)
                break

    return list(endpoints)


def _get_sql_for_timeout(query: str, dataset: str) -> dict:
    """
    Get generated SQL via reformulate endpoint when a query times out.

    The SPARQL Agent will analyze the SQL and decide which agent to delegate to.

    Args:
        query: The SPARQL query that timed out
        dataset: Dataset ID (e.g., "trn", "edu")

    Returns:
        Dict with:
        - success: True if SQL was retrieved
        - generated_sql: The SQL query from OnTop
        - error: Error message if failed
    """
    import httpx
    from src.tools.sparql_tools import get_endpoint_for_dataset

    endpoint_url = get_endpoint_for_dataset(dataset)
    logger.info(f"[Reformulate] Dataset: {dataset}, Endpoint: {endpoint_url}")
    if not endpoint_url:
        return {
            "success": False,
            "error": f"Unknown dataset: {dataset}",
            "generated_sql": "",
        }

    # Build reformulate URL
    reformulate_url = endpoint_url.replace("/sparql", "/ontop/reformulate")
    logger.info(f"[Reformulate] URL: {reformulate_url}")

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(reformulate_url, params={"query": query})

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Reformulate API returned {response.status_code}",
                    "generated_sql": "",
                }

            sql = response.text
            # Keep more of the SQL - 6000 chars should capture the full structure
            # while avoiding context overflow
            return {
                "success": True,
                "generated_sql": sql[:6000] + ("..." if len(sql) > 6000 else ""),
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "generated_sql": "",
        }


def _format_execution_feedback(execution_result: dict, sql_result: dict | None = None) -> str:
    """
    Format execution result as feedback for the SPARQL agent.

    Args:
        execution_result: Result from execute_sparql
        sql_result: SQL from _get_sql_for_timeout (for timeouts)

    Returns:
        Formatted feedback string for the agent
    """
    if execution_result.get("success"):
        # Success - show sample results
        results = execution_result.get("results", {})
        bindings = results.get("bindings", [])
        total = results.get("total_count", len(bindings))

        if not bindings:
            return """## Execution Feedback: EMPTY RESULTS

Your query executed successfully but returned 0 results.

Possible causes:
- FILTER clause is too restrictive
- Property values don't match expected format
- Data doesn't exist for the query pattern

Try adjusting or removing FILTER clauses, or check property values with sample_property_values."""

        # Format sample results
        feedback = f"""## Execution Feedback: SUCCESS

Query returned {total} result(s). Sample results (first {min(len(bindings), 5)}):

"""
        # Get column names from first result
        if bindings:
            columns = list(bindings[0].keys())
            feedback += "| " + " | ".join(columns) + " |\n"
            feedback += "| " + " | ".join(["---"] * len(columns)) + " |\n"

            for row in bindings[:5]:
                values = []
                for col in columns:
                    val = row.get(col, {})
                    if isinstance(val, dict):
                        val = val.get("value", str(val))
                    values.append(str(val)[:50])  # Truncate long values
                feedback += "| " + " | ".join(values) + " |\n"

        feedback += """
**Your task:** Verify if these results correctly answer the user's original question.
- If YES: Output status="DONE" with result_verification explaining why
- If NO: Generate a new query with status="PENDING"
"""
        return feedback

    # Extract error message from various possible locations
    # execute_sparql returns errors in an "errors" list
    error_msg = execution_result.get("error_message", "") or execution_result.get("error", "")
    error_type = execution_result.get("error_type", "")

    # Check the errors list if direct fields are empty
    errors_list = execution_result.get("errors", [])
    if errors_list and not error_msg:
        # Combine all error messages
        error_messages = []
        for err in errors_list:
            endpoint = err.get("endpoint", "unknown")
            err_type = err.get("error_type", "error")
            err_msg = err.get("error_message", "Unknown error")
            error_messages.append(f"[{endpoint}] {err_type}: {err_msg}")
            # Check if any error is a timeout
            if err_type == "query_timeout" or "timeout" in err_msg.lower():
                error_type = "query_timeout"
        error_msg = "\n".join(error_messages)

    is_timeout = "timeout" in error_msg.lower() or error_type == "query_timeout"

    if is_timeout and sql_result and sql_result.get("success"):
        feedback = f"""## Execution Feedback: TIMEOUT

Your query timed out. Here is the generated SQL from OnTop:

```sql
{sql_result.get('generated_sql', 'Could not retrieve SQL')}
```

Analyze the SQL above and delegate to the appropriate agent. Look for:
- Self-joins (same table with different aliases)
- Complex CONCAT/string operations for IRI matching
- JOINs that only serve IRI construction

Then call `delegate_to_mapping_optimizer` or `delegate_to_multi_step_agent` based on your analysis."""

        return feedback

    if is_timeout:
        return """## Execution Feedback: TIMEOUT

Your query timed out. Could not retrieve the generated SQL for analysis.

You should delegate to `delegate_to_multi_step_agent` to decompose the query into smaller parts."""

    # Syntax error or other error
    return f"""## Execution Feedback: ERROR

{error_msg}

Fix the error and generate a corrected query with status="PENDING".
"""


class OrchestratorState(TypedDict):
    """State for the orchestrator graph.

    Architektur: Isolierte Agent-Memory
    ===================================
    - Jeder Agent hat seine eigene Konversation mit dem Orchestrator
    - Der Orchestrator sieht beide Konversationen, aber nicht die internen Tool-Calls
    - Agenten sehen nur ihre eigene Konversation + eigene Aktionen
    """

    # Input
    user_query: str
    approach: Literal["agentic_grep", "agentic_semantic"]
    llm_model: str
    semantic_model_paths: list[str]

    # Phase tracking
    current_phase: Literal["retrieval", "generation", "complete", "unanswerable"]
    iteration_count: int
    max_iterations: int

    # ==========================================================================
    # AGENT MEMORY - Persistent LLM Conversation History
    # ==========================================================================

    # Each agent's complete LLM conversation (serialized messages)
    retrieval_agent_memory: AgentMemory    # Retrieval Agent's conversation
    generation_agent_memory: AgentMemory   # SPARQL Generation Agent's conversation
    orchestrator_agent_memory: AgentMemory # Orchestrator's conversation

    # Approved triples: Vom Orchestrator geprüft und an Generation weitergegeben
    approved_triples: list[str]

    # ==========================================================================
    # SHARED CONTEXT (von Agenten geschrieben, von anderen gelesen)
    # ==========================================================================

    # Raw retrieved triples (Retrieval Agent schreibt, Orchestrator liest)
    retrieved_triples: list[str]
    mapping_files_used: list[str]

    # Detected endpoints from schema retrieval (Dataspace mode)
    detected_endpoints: list[str]

    # Generated queries (Generation Agent schreibt)
    generated_queries: list[str]
    query_reasonings: list[str]
    target_endpoints: list[str]

    # Final output (chosen by Orchestrator)
    final_query: str | None
    final_results: dict | None
    selection_reasoning: str

    # Unanswerable status
    is_unanswerable: bool
    unanswerable_reason: str
    # Why the run stopped without a query: "" | "missing_schema" | "underspecified" |
    # "generation_exhausted" (UNDER instructed ablation needs the distinction)
    stop_kind: str

    # Retrieval strategy (fixed for fair comparison)
    current_retrieval_strategy: Literal["grep", "semantic"]

    # Messages for LangGraph (kept for compatibility)
    messages: Annotated[list[BaseMessage], add_messages]

    # Error handling
    error_messages: list[str]

    # Token usage tracking
    total_input_tokens: int
    total_output_tokens: int

    # NEED_TRIPLE Support for targeted retrieval
    needs_targeted_retrieval: bool
    needed_triple_description: str

    # Current query being evaluated
    current_query: str | None
    sparql_iteration_count: int
    max_sparql_iterations: int

    # Mapping Optimizer call tracking
    mapping_optimizer_call_count: int
    max_mapping_optimizer_calls: int

    # Last SPARQL Agent decision (APPROVED/ADJUST/NEED_TRIPLE)
    last_sparql_agent_decision: str | None


def create_initial_state(
    user_query: str,
    approach: Literal["agentic_grep", "agentic_semantic"],
    llm_model: str = "deepseek-chat",
    semantic_model_paths: list[str] | None = None,
) -> OrchestratorState:
    """Create initial state for the orchestrator.

    In Dataspace mode, the agent automatically detects which endpoints
    to query based on the schema elements found during retrieval.
    """
    settings = get_settings()

    # Default to mapping files in MAPPINGS_DIR
    if semantic_model_paths is None:
        semantic_model_paths = [
            str(p) for p in MAPPINGS_DIR.glob("**/*.ttl")
        ]

    # Determine initial strategy from approach
    initial_strategy = "grep" if approach == "agentic_grep" else "semantic"

    return OrchestratorState(
        user_query=user_query,
        approach=approach,
        llm_model=llm_model,
        semantic_model_paths=semantic_model_paths,
        current_phase="retrieval",
        iteration_count=0,
        max_iterations=settings.max_iterations,
        # Agent Memory - empty at start, filled during execution
        retrieval_agent_memory=[],
        generation_agent_memory=[],
        orchestrator_agent_memory=[],
        approved_triples=[],
        # Shared context
        retrieved_triples=[],
        mapping_files_used=[],
        detected_endpoints=[],
        generated_queries=[],
        query_reasonings=[],
        target_endpoints=[],
        final_query=None,
        final_results=None,
        selection_reasoning="",
        is_unanswerable=False,
        unanswerable_reason="",
        stop_kind="",
        current_retrieval_strategy=initial_strategy,
        messages=[],
        error_messages=[],
        total_input_tokens=0,
        total_output_tokens=0,
        # NEED_TRIPLE Support
        needs_targeted_retrieval=False,
        needed_triple_description="",
        # SPARQL iteration tracking
        current_query=None,
        sparql_iteration_count=0,
        max_sparql_iterations=3,
        # Mapping Optimizer limited to 2 calls max - after that, query is unanswerable
        mapping_optimizer_call_count=0,
        max_mapping_optimizer_calls=2,
        last_sparql_agent_decision=None,
    )


def should_continue(state: OrchestratorState) -> Literal["retrieval", "generation", "complete", "unanswerable"]:
    """
    Determine the next phase based on Orchestrator decision.

    This is now much simpler - the OrchestratorLLM makes the actual decision
    and writes it to current_phase. We just read it here.
    """
    # Check for unanswerable first
    if state.get("is_unanswerable") or state.get("current_phase") == "unanswerable":
        logger.info("Query marked as unanswerable, ending")
        return "unanswerable"

    # Check iteration limit
    if state["iteration_count"] >= state["max_iterations"]:
        logger.info(f"Max iterations ({state['max_iterations']}) reached, completing")
        return "complete"

    # Use the phase set by OrchestratorLLM
    phase = state.get("current_phase", "complete")
    logger.debug(f"Routing to phase: {phase}")
    return phase


async def retrieval_node(state: OrchestratorState) -> dict:
    """
    Retrieval phase: Find relevant mapping triples.

    After retrieval, the OrchestratorLLM evaluates the results
    and decides whether to continue to generation or retry.

    MEMORY: Each agent stores its complete LLM conversation history
    as serialized messages (list[dict]) - no separate conversation/actions.
    """
    tracer = get_tracer()
    user_query = state["user_query"]
    llm_model = state["llm_model"]
    current_strategy = state["current_retrieval_strategy"]

    # Get persistent LLM message history (serialized messages)
    retrieval_messages = state["retrieval_agent_memory"]

    logger.info(f"Retrieval phase: strategy={current_strategy}, query='{user_query[:50]}...'")

    # Emit phase start event
    tracer.emit(
        TraceEvent(
            event_type=TraceEventType.PHASE_START,
            phase="retrieval",
            data={
                "strategy": current_strategy,
                "query": user_query[:100],
            },
        )
    )

    try:
        # Select agent based on current strategy (Dataspace mode - searches all datasets)
        if current_strategy == "grep":
            agent = GrepRetrievalAgent(llm_model=llm_model)
        else:
            agent = SemanticRetrievalAgent(llm_model=llm_model)

        # Check for targeted retrieval (NEED_TRIPLE from SPARQL agent)
        needs_targeted = state["needs_targeted_retrieval"]
        needed_triple_description = state["needed_triple_description"]

        # Pass conversation history to retrieval agent
        result, updated_retrieval_messages = await agent.retrieve(
            user_query=user_query,
            orchestrator_feedback="",  # Feedback is in the message history
            needed_triple_description=needed_triple_description if needs_targeted else None,
            conversation_history=retrieval_messages,
        )

        retrieved_triples = result.get("retrieved_triples", [])
        mapping_files = result.get("mapping_files_used", [])
        detected_endpoints = result.get("detected_endpoints", [])
        retrieval_status = result.get("status", "FOUND")

        # Check if Retrieval Agent signaled UNANSWERABLE (missing schema) or UNDERSPECIFIED
        # (UNDER instructed ablation: question does not fix a required criterion)
        if retrieval_status in ("UNANSWERABLE", "UNDERSPECIFIED"):
            stop_kind = result.get("stop_kind") or ("underspecified" if retrieval_status == "UNDERSPECIFIED" else "missing_schema")
            agent_reason = result.get("unanswerable_reason", "Schema does not support this query")
            logger.info(f"Retrieval Agent: {retrieval_status} ({stop_kind}) - {agent_reason}")
            tracer.emit(
                TraceEvent(
                    event_type=TraceEventType.ORCHESTRATOR_DECISION,
                    phase="retrieval",
                    data={
                        "next_phase": "unanswerable",
                        "reasoning": "Retrieval Agent determined query is unanswerable",
                        "agent_reason": agent_reason,
                        "stop_kind": stop_kind,
                    },
                )
            )
            return {
                "current_phase": "unanswerable",
                "retrieved_triples": [],
                "approved_triples": [],
                "mapping_files_used": mapping_files,
                "detected_endpoints": detected_endpoints,
                "retrieval_agent_memory": updated_retrieval_messages,
                "is_unanswerable": True,
                "unanswerable_reason": agent_reason,
                "stop_kind": stop_kind,
                "needs_targeted_retrieval": False,
                "needed_triple_description": "",
            }

        # Fallback: Derive endpoints from triple prefixes if none were detected
        if not detected_endpoints and retrieved_triples:
            detected_endpoints = _derive_endpoints_from_triples(retrieved_triples)
            if detected_endpoints:
                logger.info(f"Derived {len(detected_endpoints)} endpoints from triple prefixes: {detected_endpoints}")

        # Count actual RDF triples
        actual_triple_count = count_actual_triples(retrieved_triples)

        # Rule-based decision
        retrieval_attempts = len([m for m in updated_retrieval_messages if m.get("type") == "ai"])

        stop_kind = ""
        if actual_triple_count > 0:
            # Schema found - proceed to generation
            next_phase = "generation"
            is_unanswerable = False
            unanswerable_reason = ""
            logger.info(f"Retrieval found {actual_triple_count} triples - proceeding to generation")
        elif retrieval_attempts >= 2:
            # No schema found after multiple attempts - unanswerable
            next_phase = "unanswerable"
            is_unanswerable = True
            stop_kind = "missing_schema"
            unanswerable_reason = f"After {retrieval_attempts} retrieval attempts, no relevant schema was found."
            logger.info(f"Query unanswerable: {retrieval_attempts} attempts, no schema found")
        else:
            # First attempt with no results - retry
            next_phase = "retrieval"
            is_unanswerable = False
            unanswerable_reason = ""
            logger.info("No triples found, will retry retrieval")

        # Emit decision event
        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.ORCHESTRATOR_DECISION,
                phase="retrieval",
                data={
                    "next_phase": next_phase,
                    "reasoning": "Rule-based decision",
                    "triples_found": actual_triple_count,
                    "retrieved_triples": retrieved_triples,
                },
            )
        )

        # Emit delegation event for visualization
        if next_phase == "generation":
            tracer.emit(
                TraceEvent(
                    event_type=TraceEventType.DELEGATION_START,
                    phase="retrieval",
                    data={
                        "from_agent": "retrieval",
                        "to_agent": "sparql_generation",
                        "reason": f"Schema found ({actual_triple_count} triples)",
                    },
                )
            )

        # Set approved triples for generation phase
        approved_triples = retrieved_triples if next_phase == "generation" else []

        # Emit phase end event
        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.PHASE_END,
                phase="retrieval",
                data={
                    "strategy": current_strategy,
                    "triples_found": actual_triple_count,
                    "next_phase": next_phase,
                },
            )
        )

        return {
            "current_phase": next_phase,
            "retrieved_triples": retrieved_triples,
            "approved_triples": approved_triples,
            "mapping_files_used": mapping_files,
            "detected_endpoints": detected_endpoints,
            "retrieval_agent_memory": updated_retrieval_messages,
            "is_unanswerable": is_unanswerable,
            "unanswerable_reason": unanswerable_reason,
            "stop_kind": stop_kind,
            # Reset targeted retrieval flags
            "needs_targeted_retrieval": False,
            "needed_triple_description": "",
        }

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return {
            "current_phase": "generation",  # Try generation anyway
            "retrieved_triples": [],
            "approved_triples": [],
            "mapping_files_used": [],
            "error_messages": state["error_messages"] + [f"Retrieval error: {e}"],
            "retrieval_agent_memory": retrieval_messages,  # Keep existing messages
        }


async def generation_node(state: OrchestratorState) -> dict:
    """
    Generation phase: SPARQL Agent generates query, system executes and provides feedback.

    New workflow:
    1. Agent generates query with status=PENDING
    2. System executes query and provides feedback (results/timeout+SQL/error)
    3. Agent analyzes feedback and decides:
       - status=DONE: Results verified correct
       - status=PENDING: Try new query
       - Delegate to mapping optimizer or multi-step agent
    4. Loop continues until DONE, NEED_TRIPLE, or max iterations
    """
    from src.tools.sparql_tools import execute_sparql, get_endpoints_for_query

    tracer = get_tracer()
    user_query = state["user_query"]
    llm_model = state["llm_model"]

    # Get persistent LLM message history
    generation_messages = state["generation_agent_memory"]
    schema_context = state["approved_triples"]

    sparql_iteration = state["sparql_iteration_count"]
    max_sparql_iterations = state["max_sparql_iterations"]

    # Check if we've exceeded SPARQL iteration limit
    if sparql_iteration >= max_sparql_iterations:
        logger.warning(f"Max SPARQL iterations ({max_sparql_iterations}) reached")
        return {
            "current_phase": "complete",
            "is_unanswerable": True,
            "unanswerable_reason": f"Could not generate valid query after {sparql_iteration} attempts",
            "stop_kind": "generation_exhausted",
            "iteration_count": state["iteration_count"] + 1,
        }

    logger.info(f"Generation phase: iteration={sparql_iteration+1}/{max_sparql_iterations}, "
                f"triples={len(schema_context)}")

    # Emit phase start event
    tracer.emit(
        TraceEvent(
            event_type=TraceEventType.PHASE_START,
            phase="generation",
            data={
                "sparql_iteration": sparql_iteration + 1,
                "triples_count": len(schema_context),
                "schema_triples": schema_context,
            },
        )
    )

    try:
        agent = SPARQLGenerationAgent(llm_model=llm_model)
        current_messages = generation_messages
        execution_feedback = ""  # Feedback from query execution

        # Inner loop: generate query -> execute -> feedback -> repeat until DONE/NEED_TRIPLE/delegate
        max_inner_iterations = 3  # Prevent infinite loops within one generation phase
        for inner_iter in range(max_inner_iterations):
            logger.info(f"Generation inner loop: iteration {inner_iter + 1}/{max_inner_iterations}")

            # Call SPARQL Agent
            results, updated_messages = await agent.generate(
                user_query=user_query,
                schema_context=schema_context,
                k=1,
                orchestrator_feedback=execution_feedback,  # Pass execution feedback
                previous_attempts=[],
                conversation_history=current_messages,
            )
            current_messages = updated_messages

            if not results:
                logger.warning("SPARQL Agent returned no results")
                continue

            result = results[0]
            status = result.get("status", "PENDING")
            query = result.get("query", "")
            reasoning = result.get("reasoning", "")
            result_verification = result.get("result_verification", "")
            needed_triple = result.get("needed_triple", "")

            logger.info(f"SPARQL Agent status={status}, query_length={len(query)}")

            # Handle UNDERSPECIFIED (UNDER instructed ablation): stop without a query
            if status == "UNDERSPECIFIED":
                missing_criterion = result.get("missing_criterion", "") or reasoning or "Question does not fix a required criterion"
                logger.info(f"SPARQL Agent: UNDERSPECIFIED - {missing_criterion}")
                tracer.emit(
                    TraceEvent(
                        event_type=TraceEventType.ORCHESTRATOR_DECISION,
                        phase="generation",
                        data={
                            "next_phase": "unanswerable",
                            "sparql_agent_decision": "UNDERSPECIFIED",
                            "missing_criterion": missing_criterion,
                            "stop_kind": "underspecified",
                        },
                    )
                )
                return {
                    "current_phase": "unanswerable",
                    "is_unanswerable": True,
                    "unanswerable_reason": missing_criterion,
                    "stop_kind": "underspecified",
                    "sparql_iteration_count": sparql_iteration + 1,
                    "iteration_count": state["iteration_count"] + 1,
                    "generation_agent_memory": current_messages,
                }

            # Handle NEED_TRIPLE - need more schema
            if status == "NEED_TRIPLE":
                logger.info(f"SPARQL Agent needs more schema: {needed_triple}")
                tracer.emit(
                    TraceEvent(
                        event_type=TraceEventType.ORCHESTRATOR_DECISION,
                        phase="generation",
                        data={
                            "next_phase": "retrieval",
                            "sparql_agent_decision": "NEED_TRIPLE",
                            "needed_triple": needed_triple,
                        },
                    )
                )
                # Emit delegation event for visualization
                tracer.emit(
                    TraceEvent(
                        event_type=TraceEventType.DELEGATION_START,
                        phase="generation",
                        data={
                            "from_agent": "sparql_generation",
                            "to_agent": "retrieval",
                            "reason": f"Need more schema: {needed_triple[:100]}",
                        },
                    )
                )
                return {
                    "current_phase": "retrieval",
                    "needs_targeted_retrieval": True,
                    "needed_triple_description": needed_triple,
                    "sparql_iteration_count": sparql_iteration + 1,
                    "iteration_count": state["iteration_count"] + 1,
                    "generation_agent_memory": current_messages,
                }

            # Handle MULTI_STEP_RESULT - delegated to Multi-Step Agent
            if status == "MULTI_STEP_RESULT":
                multi_step_result = result.get("multi_step_result", {})
                logger.info(f"SPARQL Agent delegated to Multi-Step Agent: success={multi_step_result.get('success', False)}")

                tracer.emit(
                    TraceEvent(
                        event_type=TraceEventType.ORCHESTRATOR_DECISION,
                        phase="generation",
                        data={
                            "next_phase": "complete",
                            "sparql_agent_decision": "MULTI_STEP_RESULT",
                            "multi_step_success": multi_step_result.get("success", False),
                        },
                    )
                )
                # Emit delegation end event for visualization (return from multi-step)
                tracer.emit(
                    TraceEvent(
                        event_type=TraceEventType.DELEGATION_END,
                        phase="generation",
                        data={
                            "from_agent": "multi_step",
                            "to_agent": "sparql_generation",
                            "reason": f"Multi-step completed: success={multi_step_result.get('success', False)}",
                        },
                    )
                )

                # Build execution result from multi-step
                multi_step_final_results = multi_step_result.get("final_results", [])
                multi_step_steps = multi_step_result.get("steps", [])
                multi_step_transform = multi_step_result.get("transform_script", "")

                execution_result = {
                    "success": multi_step_result.get("success", False),
                    "results": {
                        "bindings": multi_step_final_results,
                        "total_count": len(multi_step_final_results),
                    },
                    "multi_step_execution": True,
                    "steps": multi_step_steps,
                    "transform_script": multi_step_transform,
                    "final_results": multi_step_final_results,
                }
                if not multi_step_result.get("success"):
                    execution_result["error"] = multi_step_result.get("error", "Multi-Step Agent failed")

                return {
                    "current_phase": "complete",
                    "generated_queries": [],
                    "query_reasonings": [reasoning],
                    "target_endpoints": result.get("target_datasets", []),
                    "current_query": None,
                    "sparql_iteration_count": sparql_iteration + 1,
                    "last_sparql_agent_decision": "MULTI_STEP_RESULT",
                    "final_query": None,
                    "final_results": execution_result,
                    "selection_reasoning": f"Delegated to Multi-Step Agent: {reasoning}",
                    "iteration_count": state["iteration_count"] + 1,
                    "generation_agent_memory": current_messages,
                    "is_unanswerable": not multi_step_result.get("success", False),
                    "unanswerable_reason": multi_step_result.get("error", "") if not multi_step_result.get("success") else "",
                }

            # Handle MAPPING_OPTIMIZER_RESULT - delegation to Mapping Optimizer completed
            if status == "MAPPING_OPTIMIZER_RESULT":
                mapping_result = result.get("mapping_optimizer_result", {})
                can_help = mapping_result.get("can_help", False)
                success = mapping_result.get("success", False)
                optimized_tables = mapping_result.get("optimized_tables", [])

                # Track MO calls - increment counter
                mo_call_count = state["mapping_optimizer_call_count"] + 1
                max_mo_calls = state["max_mapping_optimizer_calls"]
                logger.info(f"SPARQL Agent delegated to Mapping Optimizer: success={success}, can_help={can_help}, "
                            f"optimized_tables={optimized_tables}, mo_calls={mo_call_count}/{max_mo_calls}")

                tracer.emit(
                    TraceEvent(
                        event_type=TraceEventType.ORCHESTRATOR_DECISION,
                        phase="generation",
                        data={
                            "next_phase": "generation_retry" if (success and can_help) else "generation_continue",
                            "sparql_agent_decision": "MAPPING_OPTIMIZER_RESULT",
                            "can_help": can_help,
                            "success": success,
                            "optimized_tables": optimized_tables,
                            "mo_call_count": mo_call_count,
                        },
                    )
                )
                # Emit delegation end event for visualization
                tracer.emit(
                    TraceEvent(
                        event_type=TraceEventType.DELEGATION_END,
                        phase="generation",
                        data={
                            "from_agent": "mapping_optimizer",
                            "to_agent": "sparql_generation",
                            "reason": f"Mapping Optimizer: success={success}, can_help={can_help}, call {mo_call_count}/{max_mo_calls}",
                        },
                    )
                )

                # Check if MO call limit reached - mark as unanswerable
                if mo_call_count >= max_mo_calls and not (success and can_help and optimized_tables):
                    logger.warning(f"Mapping Optimizer call limit reached ({mo_call_count}/{max_mo_calls}) "
                                   f"without success - marking query as unanswerable")
                    return {
                        "current_phase": "complete",
                        "is_unanswerable": True,
                        "unanswerable_reason": f"Query times out and Mapping Optimizer could not resolve after {mo_call_count} attempts",
                        "mapping_optimizer_call_count": mo_call_count,
                        "iteration_count": state["iteration_count"] + 1,
                        "generation_agent_memory": current_messages,
                    }

                # Handle different cases
                if success and can_help and optimized_tables:
                    # Case 1: Mappings were optimized successfully - retry query
                    execution_feedback = f"""Mapping Optimizer has successfully optimized the R2RML mappings.

Optimized tables: {optimized_tables}
Changes: {mapping_result.get('changes_made', 'IRI templates optimized to use primary keys')}

The OnTop container has been restarted with the new mappings.
**Please retry your SPARQL query now** - it should execute faster with the optimized mappings.

If the query still times out after {mo_call_count} Mapping Optimizer attempt(s), the query may be inherently too expensive."""
                    logger.info("Mapping Optimizer succeeded - continuing loop for SPARQL Agent to retry")

                    # If MO succeeded but we've hit the limit, give one last chance
                    if mo_call_count >= max_mo_calls:
                        execution_feedback += f"\n\n**WARNING: This was the last Mapping Optimizer attempt ({mo_call_count}/{max_mo_calls}). If the query still times out, it will be marked as unanswerable.**"

                elif not can_help:
                    # Case 2: Mapping optimizer determined it cannot help
                    analysis = mapping_result.get("analysis", "")
                    mo_reasoning = mapping_result.get("reasoning", "")
                    execution_feedback = f"""Mapping Optimizer analyzed the query but CANNOT help with this timeout (attempt {mo_call_count}/{max_mo_calls}).

**Analysis:** {analysis}
**Reasoning:** {mo_reasoning}

The mapping structure is not the cause of the timeout. Try a simpler query structure with fewer joins."""
                    logger.info("Mapping Optimizer can't help - SPARQL Agent should simplify")

                else:
                    # Case 3: Error or partial success
                    error = mapping_result.get("error", "Unknown error")
                    mo_reasoning = mapping_result.get("reasoning", "")
                    execution_feedback = f"""Mapping Optimizer encountered an issue (attempt {mo_call_count}/{max_mo_calls}): {error}

{f'Analysis: {mo_reasoning}' if mo_reasoning else ''}

Try a simpler query structure."""
                    logger.info(f"Mapping Optimizer error: {error}")

                # Update state with incremented MO counter before continuing
                state["mapping_optimizer_call_count"] = mo_call_count
                continue  # Continue the while loop to let SPARQL agent respond

            # Handle DONE - results verified by agent
            if status == "DONE":
                if not query:
                    logger.warning("Agent said DONE but no query provided")
                    execution_feedback = "Error: You said DONE but did not provide a query. Please provide the verified query."
                    continue

                logger.info(f"SPARQL Agent verified results: {result_verification[:100] if result_verification else 'no verification'}")

                # Execute final query
                query_endpoints = get_endpoints_for_query(query)
                detected_endpoints = state["detected_endpoints"]
                target_endpoints = query_endpoints or detected_endpoints or []

                logger.info(f"Executing verified query on endpoints: {target_endpoints}")

                # Emit TOOL_CALL event for execute_sparql
                sparql_start = _time.time()

                if target_endpoints:
                    execution_result = execute_sparql.invoke({
                        "query": query,
                        "endpoints": list(set(target_endpoints)),
                    })
                else:
                    execution_result = execute_sparql.invoke({"query": query})

                sparql_duration = (_time.time() - sparql_start) * 1000
                result_count = len(execution_result.get("results", {}).get("bindings", [])) if execution_result.get("success") else 0
                tracer.emit(
                    TraceEvent(
                        event_type=TraceEventType.TOOL_CALL,
                        phase="generation",
                        data={
                            "tool_name": "execute_sparql",
                            "agent": "orchestrator",
                            "success": execution_result.get("success", False),
                            "duration_ms": sparql_duration,
                            "result_count": result_count,
                            "input_summary": f"query={len(query)} chars, endpoints={target_endpoints}",
                        },
                    )
                )

                tracer.emit(
                    TraceEvent(
                        event_type=TraceEventType.ORCHESTRATOR_DECISION,
                        phase="generation",
                        data={
                            "next_phase": "complete",
                            "sparql_agent_decision": "DONE",
                            "reasoning": reasoning,
                            "result_verification": result_verification,
                            "query_success": execution_result.get("success", False),
                        },
                    )
                )

                # Emit PHASE_END for generation
                tracer.emit(
                    TraceEvent(
                        event_type=TraceEventType.PHASE_END,
                        phase="generation",
                        data={
                            "status": "DONE",
                            "query_success": execution_result.get("success", False),
                            "result_count": result_count,
                        },
                    )
                )

                return {
                    "current_phase": "complete",
                    "generated_queries": [query],
                    "query_reasonings": [reasoning],
                    "target_endpoints": target_endpoints,
                    "current_query": query,
                    "sparql_iteration_count": sparql_iteration + 1,
                    "last_sparql_agent_decision": "DONE",
                    "final_query": query,
                    "final_results": execution_result,
                    "selection_reasoning": reasoning,
                    "iteration_count": state["iteration_count"] + 1,
                    "generation_agent_memory": current_messages,
                    "is_unanswerable": False,
                    "unanswerable_reason": "",
                }

            # Handle PENDING - execute query and provide feedback
            if status == "PENDING" and query:
                logger.info(f"Executing PENDING query for feedback")

                # Determine endpoints
                query_endpoints = get_endpoints_for_query(query)
                detected_endpoints = state["detected_endpoints"]
                target_endpoints = query_endpoints or detected_endpoints or []
                target_datasets = result.get("target_datasets", [])

                # CRITICAL: Derive target_datasets from detected_endpoints if empty
                # Without this, SQL won't be fetched on timeout and Mapping Optimizer can't be called
                if not target_datasets and detected_endpoints:
                    registry = load_dataset_registry()
                    for ep in detected_endpoints:
                        for ds in registry.get("datasets", []):
                            if ds.get("sparql_endpoint") == ep:
                                target_datasets.append(ds["id"])
                                break
                    if target_datasets:
                        logger.info(f"Derived target_datasets from detected_endpoints: {target_datasets}")

                # Execute query with TOOL_CALL tracing
                pending_sparql_start = _time.time()
                if target_endpoints:
                    execution_result = execute_sparql.invoke({
                        "query": query,
                        "endpoints": list(set(target_endpoints)),
                    })
                else:
                    execution_result = execute_sparql.invoke({"query": query})

                pending_sparql_duration = (_time.time() - pending_sparql_start) * 1000
                pending_result_count = len(execution_result.get("results", {}).get("bindings", [])) if execution_result.get("success") else 0
                tracer.emit(
                    TraceEvent(
                        event_type=TraceEventType.TOOL_CALL,
                        phase="generation",
                        data={
                            "tool_name": "execute_sparql",
                            "agent": "orchestrator",
                            "success": execution_result.get("success", False),
                            "duration_ms": pending_sparql_duration,
                            "result_count": pending_result_count,
                            "input_summary": f"PENDING query={len(query)} chars, endpoints={target_endpoints}",
                        },
                    )
                )

                # Check for timeout and get SQL analysis
                # Extract error from errors list (same logic as _format_execution_feedback)
                error_msg = execution_result.get("error_message", "") or execution_result.get("error", "")
                error_type = execution_result.get("error_type", "")
                errors_list = execution_result.get("errors", [])
                if errors_list and not error_msg:
                    for err in errors_list:
                        err_type = err.get("error_type", "")
                        err_msg = err.get("error_message", "")
                        if err_type == "query_timeout" or "timeout" in err_msg.lower():
                            error_type = "query_timeout"
                            error_msg = err_msg
                            break
                is_timeout = "timeout" in error_msg.lower() or error_type == "query_timeout"

                sql_result = None
                logger.info(f"Timeout check: is_timeout={is_timeout}, target_datasets={target_datasets}, error_type={error_type}")
                if is_timeout and target_datasets:
                    # Get SQL via reformulate for agent to analyze
                    dataset = target_datasets[0] if target_datasets else ""
                    if dataset:
                        logger.info(f"Calling reformulate API for dataset: {dataset}")
                        sql_result = _get_sql_for_timeout(query, dataset)
                        logger.info(f"Retrieved SQL for timeout analysis: success={sql_result.get('success')}")
                        if not sql_result.get('success'):
                            logger.warning(f"Reformulate API failed: {sql_result.get('error')}")

                # Format feedback for agent
                execution_feedback = _format_execution_feedback(execution_result, sql_result)
                logger.info(f"Providing execution feedback to agent ({len(execution_feedback)} chars)")

                # Continue loop - agent will see feedback and decide
                continue

            # No query provided with PENDING status
            if status == "PENDING" and not query:
                execution_feedback = "Error: You output status=PENDING but did not provide a query. Please generate a SPARQL query."
                continue

            # Unknown status
            logger.warning(f"Unknown status from agent: {status}")
            from src.config import UNDER_INSTRUCTED
            execution_feedback = (
                f"Error: Unknown status '{status}'. Valid statuses are: PENDING, DONE, NEED_TRIPLE"
                + (", UNDERSPECIFIED" if UNDER_INSTRUCTED else "")
            )

        # Max inner iterations reached
        logger.warning(f"Max inner iterations ({max_inner_iterations}) reached in generation phase")

        # Emit PHASE_END for generation (max iterations)
        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.PHASE_END,
                phase="generation",
                data={
                    "status": "max_iterations",
                    "inner_iterations": max_inner_iterations,
                },
            )
        )

        return {
            "current_phase": "generation",  # Will re-enter generation node
            "sparql_iteration_count": sparql_iteration + 1,
            "iteration_count": state["iteration_count"] + 1,
            "generation_agent_memory": current_messages,
            "mapping_optimizer_call_count": state["mapping_optimizer_call_count"],
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()

        # Emit PHASE_END for generation (error)
        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.PHASE_END,
                phase="generation",
                data={
                    "status": "error",
                    "error": str(e),
                },
            )
        )

        return {
            "current_phase": "complete",
            "generated_queries": [],
            "query_reasonings": [],
            "final_query": None,
            "final_results": None,
            "selection_reasoning": f"Generation error: {e}",
            "iteration_count": state["iteration_count"] + 1,
            "error_messages": state["error_messages"] + [f"Generation error: {e}"],
            "generation_agent_memory": generation_messages,
        }


def build_orchestrator_graph() -> StateGraph:
    """
    Build the orchestrator state graph.

    Two nodes with rule-based routing:
    - retrieval: Retrieval Agent finds schema
    - generation: SPARQL Agent generates, tests, decides (DONE/NEED_TRIPLE)
    """
    graph = StateGraph(OrchestratorState)

    graph.add_node("retrieval", retrieval_node)
    graph.add_node("generation", generation_node)

    # Set entry point
    graph.set_entry_point("retrieval")

    # Add conditional edges from retrieval
    graph.add_conditional_edges(
        "retrieval",
        should_continue,
        {
            "retrieval": "retrieval",  # Retry with different strategy
            "generation": "generation",
            "complete": END,
            "unanswerable": END,  # Query cannot be answered
        },
    )

    # Add conditional edges from generation
    graph.add_conditional_edges(
        "generation",
        should_continue,
        {
            "retrieval": "retrieval",  # Schema was incomplete
            "generation": "generation",  # Query errors, try again
            "complete": END,
            "unanswerable": END,  # Query cannot be answered
        },
    )

    return graph


class OrchestratorAgent:
    """High-level interface for the orchestrator."""

    def __init__(self):
        """Initialize the orchestrator agent."""
        self.graph = build_orchestrator_graph()
        self.compiled = self.graph.compile()

    async def run(
        self,
        user_query: str,
        approach: Literal["agentic_grep", "agentic_semantic"] = "agentic_semantic",
        llm_model: str = "deepseek-chat",
        semantic_model_paths: list[str] | None = None,
    ) -> OrchestratorState:
        """
        Run the orchestrator for a given query.

        In Dataspace mode, the agent automatically searches all datasets
        and determines which endpoints to query based on the schema elements found.

        Mapping Optimization Cleanup:
        Any mapping modifications made during this run (e.g., for IRI template
        optimization) are automatically reverted when the run completes.

        Args:
            user_query: Natural language query
            approach: Which retrieval approach to use
            llm_model: LLM model to use for all agents
            semantic_model_paths: Paths to semantic model files

        Returns:
            Final orchestrator state with results
        """
        # Set LLM model so sub-agents (MappingOptimizer, MultiStep) use the same model
        set_llm_model(llm_model)

        # Import mapping context for cleanup
        from src.tools.mapping_optimizer_tools import (
            get_mapping_context,
            reset_mapping_context,
        )

        # Reset mapping context for this run
        reset_mapping_context()
        mapping_context = get_mapping_context()

        try:
            initial_state = create_initial_state(
                user_query=user_query,
                approach=approach,
                llm_model=llm_model,
                semantic_model_paths=semantic_model_paths,
            )
            final_state = await self.compiled.ainvoke(initial_state)
            return final_state
        finally:
            # ALWAYS restore mappings at end of run
            if mapping_context.modified:
                logger.info("Restoring modified mappings after agent run...")
                mapping_context.restore_all()  # Synchron, kein await
                logger.info("Mapping restoration complete.")


async def run_agentic_pipeline(
    user_query: str,
    approach: Literal["agentic_grep", "agentic_semantic"] = "agentic_semantic",
    llm_model: str = "deepseek-chat",
    semantic_model_paths: list[str] | None = None,
) -> dict:
    """
    Convenience function to run the agentic pipeline.

    In Dataspace mode, automatically searches all datasets.

    Mapping Optimization Cleanup:
    Any mapping modifications made during this run are automatically reverted
    when the run completes, ensuring each query starts with clean mappings.

    Args:
        user_query: Natural language query
        approach: Which retrieval approach to use
        llm_model: LLM model to use
        semantic_model_paths: Paths to semantic model files

    Returns:
        Dict with final_query, final_results, etc.
    """
    # Set LLM model globally so sub-agents (MappingOptimizer, MultiStep) use the same model
    set_llm_model(llm_model)

    # Note: Cleanup is handled inside OrchestratorAgent.run()
    orchestrator = OrchestratorAgent()
    state = await orchestrator.run(
        user_query=user_query,
        approach=approach,
        llm_model=llm_model,
        semantic_model_paths=semantic_model_paths,
    )

    return {
        "final_query": state.get("final_query"),
        "final_results": state.get("final_results"),
        "selection_reasoning": state.get("selection_reasoning", ""),
        "generated_queries": state.get("generated_queries", []),
        "retrieved_triples": state.get("retrieved_triples", []),
        "orchestrator_history": state.get("orchestrator_history", []),
        "iteration_count": state.get("iteration_count", 0),
        "failed_approaches": state.get("failed_approaches", []),
        "error_messages": state.get("error_messages", []),
        "detected_endpoints": state.get("detected_endpoints", []),
        "is_unanswerable": state.get("is_unanswerable", False),
        "unanswerable_reason": state.get("unanswerable_reason", ""),
        "stop_kind": state.get("stop_kind", ""),
        "mapping_optimization_applied": state.get("mapping_optimization_applied", False),
    }
