"""
LLM-based Orchestrator for intelligent workflow control.

The OrchestratorLLM replaces the rule-based routing with LLM-based decisions.
It also takes over the Validation role - executing queries and analyzing results.
"""

import asyncio
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama


def serialize_messages(messages: list[BaseMessage]) -> list[dict]:
    """Serialize LangChain messages to dicts for state storage.

    Preserves ALL message attributes including:
    - AIMessage.tool_calls (which tools were called)
    - ToolMessage (tool results with tool_call_id and name)
    """
    serialized = []
    for msg in messages:
        item = {
            "type": msg.__class__.__name__,
            "content": msg.content,
        }
        # AIMessage: preserve tool_calls
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            item["tool_calls"] = msg.tool_calls
        # ToolMessage: preserve tool_call_id and name
        if isinstance(msg, ToolMessage):
            item["tool_call_id"] = getattr(msg, 'tool_call_id', '')
            item["name"] = getattr(msg, 'name', '')
        serialized.append(item)
    return serialized


def deserialize_messages(data: list[dict]) -> list[BaseMessage]:
    """Deserialize dicts back to LangChain messages.

    Restores ALL message types with their full attributes.
    """
    messages = []
    for item in data:
        msg_type = item.get("type", "HumanMessage")
        content = item.get("content", "")

        if msg_type == "SystemMessage":
            messages.append(SystemMessage(content=content))
        elif msg_type == "AIMessage":
            tool_calls = item.get("tool_calls", [])
            messages.append(AIMessage(content=content, tool_calls=tool_calls))
        elif msg_type == "ToolMessage":
            messages.append(ToolMessage(
                content=content,
                tool_call_id=item.get("tool_call_id", ""),
                name=item.get("name", ""),
            ))
        else:
            messages.append(HumanMessage(content=content))
    return messages

from rdflib import Graph

from src.config import get_settings, normalize_content
from src.config import get_openrouter_kwargs
from src.tools.sparql_tools import execute_sparql, get_endpoints_for_query, QUERY_TIMEOUT
from src.tracing import TraceEvent, TraceEventType, get_tracer
from src.tracing.llm_callback import create_llm_callback
from src.validation import validate_sparql_semantics, SemanticValidationResult
from src.utils.ontop_hints import get_ontop_timeout_hints
from src.utils.query_metadata import extract_query_metadata

logger = logging.getLogger(__name__)


def count_actual_triples(schema_strings: list[str]) -> int:
    """Count actual RDF triples in schema strings using rdflib."""
    if not schema_strings:
        return 0

    total = 0
    for schema in schema_strings:
        if not schema or not schema.strip():
            continue
        try:
            g = Graph()
            g.parse(data=schema, format="turtle")
            total += len(g)
        except Exception:
            # Fallback: count lines ending with . or ;
            lines = [l.strip() for l in schema.split("\n") if l.strip()]
            for line in lines:
                if line.endswith(".") or line.endswith(";"):
                    if not line.upper().startswith("PREFIX") and not line.startswith("@prefix"):
                        total += 1
    return total

# Thread pool for parallel query execution
_query_executor = ThreadPoolExecutor(max_workers=5)


@dataclass
class SPARQLAgentDecision:
    """Decision from SPARQL agent after reviewing feedback."""
    action: Literal["APPROVED", "ADJUST", "NEED_TRIPLE"]
    reason: str = ""              # For ADJUST: what to change
    triple_description: str = ""  # For NEED_TRIPLE: semantic description of needed triple


@dataclass
class QueryFeedback:
    """Complete feedback package for SPARQL agent - NO classification by system."""
    query: str
    user_query: str
    schema_context: list[str]

    # Execution result
    execution_success: bool
    execution_error: str | None = None

    # Semantic validation (complete - agent sees all including suggestions)
    semantic_validation: SemanticValidationResult | None = None

    # Results (if successfully executed)
    sample_results: str | None = None  # Top 5 formatted
    total_count: int = 0
    variables: list[str] = field(default_factory=list)

    def format_for_agent(self) -> str:
        """Format all info for SPARQL agent prompt."""
        lines = [
            f"## Original User Query\n{self.user_query}",
            f"\n## Your SPARQL Query\n```sparql\n{self.query}\n```",
        ]

        if self.execution_error:
            lines.append(f"\n## Execution Error\n{self.execution_error}")

            # Add OBDA-specific hints for timeout errors
            if "timeout" in self.execution_error.lower() or "timed out" in self.execution_error.lower():
                lines.append(get_ontop_timeout_hints())

        if self.semantic_validation:
            # SemanticValidationResult has format_issues() method
            if hasattr(self.semantic_validation, 'format_issues'):
                issues = self.semantic_validation.format_issues()
                if issues:
                    lines.append(f"\n## Semantic Validation\n{issues}")

        if self.sample_results:
            lines.append(f"\n## Query Results ({self.total_count} total)\n{self.sample_results}")
        elif self.execution_success and self.total_count == 0:
            lines.append("\n## Query Results\nNo results returned (empty result set).")

        return "\n".join(lines)


@dataclass
class OrchestratorDecision:
    """Decision made by the OrchestratorLLM after analyzing a phase result."""

    next_phase: Literal["retrieval", "generation", "complete", "unanswerable"]
    feedback_for_next_phase: str
    reasoning: str
    selected_query: str | None = None
    selected_query_index: int = -1
    selected_results: dict | None = None
    should_change_strategy: bool = False
    new_strategy: Literal["grep", "semantic"] | None = None
    is_unanswerable: bool = False
    unanswerable_reason: str = ""
    # NEW: SPARQL Agent Decision support
    sparql_agent_decision: SPARQLAgentDecision | None = None
    needed_triple_description: str = ""
    needs_targeted_retrieval: bool = False


ORCHESTRATOR_SYSTEM_PROMPT = """You are the orchestrator of a Text-to-SPARQL system.

You coordinate two agents (Retrieval and Generation) and handle validation yourself.

## Your Role
- Analyze results from agents
- Make informed decisions about next steps
- Provide specific, actionable feedback

## DERIVING DOMAIN CONTEXT FROM SCHEMA (when needed)
Derive the domain context from surrounding classes and properties when needed.

Example: If the schema contains classes like Hospital, Doctor, Patient, Diagnosis,
it's a Medical Domain. "medical patients" = simply Patient (all are medical by definition).

RULE: If an adjective is IMPLICITLY given by the domain context,
you do NOT need an explicit filter property for it.

## IMPORTANT RULES:
1. Trust your analysis, NOT the majority of results
2. Do NOT repeat the same failed approaches
3. Give SPECIFIC feedback (not "search better" but "search for 'advisor' property")
"""


RETRIEVAL_ANALYSIS_TEMPLATE = """## Current Task
You have just received RETRIEVAL results. Analyze them and decide the next step.

## User Query
{user_query}

## Found Schema Elements ({num_triples} triples)
{schema_triples}

## Previous History
{history}

---

## Your Decision Options

You have exactly THREE options:

### Option 1: "generation"
Choose this when schema is sufficient to generate a SPARQL query.
- All relevant classes/properties found
- Terms matched directly, via property, or implicitly via domain context

### Option 2: "retrieval"
Choose this when important schema elements are still missing.
- Provide specific hint what to search for
- Example feedback: "Search for 'advisor' property linking students to professors"

### Option 3: "unanswerable"
Choose this ONLY when ALL of these conditions are met:
1. Multiple retrieval attempts have been made (check history)
2. The Retrieval Agent has explicitly stated that core concepts were not found
3. The missing element is a CORE CLASS, not just an adjective/filter
4. The query refers to concepts fundamentally not modeled in the schema, i.e. a missing class or data type property

**NEVER choose this:**
- On the first retrieval attempt
- Just because an adjective or filter property is missing
- When the Retrieval Agent hasn't confirmed that concepts don't exist

**Rule:** After 2+ retrieval attempts with some schema found → go to "generation"

---

## Analysis Checklist

1. **All terms matched?** → "generation"
2. **Term marked as "NOT MATCHED"?**
   - Adjective implicitly satisfied by domain? → "generation"
   - Core class doesn't exist? → possibly "unanswerable"
3. **"implicit domain match" used?** Check if plausible

---

## Response Format (JSON)
{{
    "decision": "continue" | "go_back" | "unanswerable",
    "next_phase": "retrieval" | "generation" | null,
    "feedback": "Specific hint for the next phase",
    "reasoning": "Why this decision",
    "unanswerable_reason": "Only for unanswerable: Why impossible?"
}}"""


GENERATION_ANALYSIS_TEMPLATE = """## Current Task
You have just executed SPARQL queries. Analyze the results and decide the next step.

## User Query
{user_query}

## Used Schema
{schema_context}

## Generated Queries and Results
{queries_and_results}

## Previous Failed Approaches (DO NOT repeat!)
{failed_approaches}

## Previous History
{history}

---

## Your Decision Options

You have exactly FOUR options:

### Option 1: "complete" (accept_query)
Choose this when a query result is acceptable.
- Query executed successfully
- Results answer the user's question
- MUST specify selected_query_index

### Option 2: "generation"
Choose this when queries have fixable errors.
- Syntax errors that can be corrected
- Logic errors with clear fix
- Provide specific feedback for improvement

### Option 3: "retrieval"
Choose this when schema was incomplete.
- Missing properties/classes discovered during generation
- Need additional schema elements
- Provide specific hint what to search for

### Option 4: "unanswerable"
Choose this ONLY when ALL of these conditions are met:
1. Multiple generation attempts have failed (check history)
2. The errors are not fixable (not just syntax issues)
3. Schema is confirmed complete but query is still impossible
4. Going back to retrieval won't help (retrieval already exhausted)

**NEVER choose this:**
- On the first generation attempt
- When syntax errors can be fixed with "generation"
- When additional schema might help (use "retrieval" instead)

---

## Analysis Steps
1. Analyze each query and its result
2. Identify errors (syntax, logic, empty results)
3. Select the BEST result (even if minority!)
4. Decide: accept, retry generation, go back to retrieval, or give up

---

## Response Format (JSON)
{{
    "decision": "continue" | "go_back" | "complete" | "unanswerable",
    "next_phase": "retrieval" | "generation" | null,
    "feedback": "Specific hint for the next phase",
    "reasoning": "Why this decision",
    "selected_query_index": 0,
    "unanswerable_reason": "Only for unanswerable"
}}"""


EVALUATION_SYSTEM_PROMPT = """Du bist ein SPARQL Query Evaluator und Entscheider.

Du bekommst ALLE Informationen zu einer SPARQL Query:
- Die urspruengliche Benutzerfrage
- Die generierte Query
- Execution Result (Erfolg/Fehler)
- Semantic Validation (Fehler + Suggestions falls vorhanden)
- Die Top 5 Ergebnisse (falls Query erfolgreich)

## Deine Aufgabe

Analysiere ALLES und entscheide selbst was der beste naechste Schritt ist:

### APPROVED
Die Query funktioniert korrekt UND die Ergebnisse beantworten die Frage.
WICHTIG: Leere Ergebnisse (0 Resultate) sind NIEMALS APPROVED!

Beispiel: User fragt "Finde Studenten mit Advisors"
→ Ergebnisse zeigen Student-Advisor Paare → APPROVED
→ 0 Ergebnisse → NIEMALS APPROVED (versuche ADJUST oder NEED_TRIPLE)

### ADJUST: [was aendern]
Du kannst die Query SELBST verbessern mit dem vorhandenen Schema.

Beispiele:
- "ADJUST: Semantic validation suggests :hasAdvisor statt :advisor - korrigiere Property"
- "ADJUST: Ergebnisse zeigen alle Mitarbeiter, aber Frage will nur Deutsche - fuege FILTER hinzu"
- "ADJUST: Syntax-Fehler in Zeile 3 - fehlende Klammer"

### NEED_TRIPLE: [spezifische Beschreibung]
Dir fehlt ein SPEZIFISCHES Tripel/Schema-Element das du nicht selbst erfinden kannst.
Beschreibe GENAU welche Art von Beziehung/Property du brauchst.

Beispiele:
- "NEED_TRIPLE: Property um Mitarbeiter nach Beschaeftigungsland zu filtern (z.B. worksInCountry)"
- "NEED_TRIPLE: Beziehung zwischen Student und deren Betreuer/Advisor"
- "NEED_TRIPLE: Property um Produkte nach Kategorie zu klassifizieren"

## Wichtige Regeln

1. Du siehst IMMER die Semantic Validation mit Suggestions - entscheide SELBST ob du sie nutzen kannst
2. Bei Ergebnissen: Pruefe ob sie ZUR FRAGE passen, nicht nur ob Daten da sind
3. LEERE ERGEBNISSE (0 Resultate) sind NIEMALS APPROVED - versuche immer ADJUST oder NEED_TRIPLE!
4. NEED_TRIPLE nur wenn du wirklich Schema-Informationen brauchst die nicht da sind
5. Beschreibe bei NEED_TRIPLE GENAU was fuer ein Tripel fehlt (semantisch, nicht nur Syntax)

Antworte NUR mit einem der drei Formate. Keine zusaetzlichen Erklaerungen."""


class OrchestratorLLM:
    """
    LLM-based Orchestrator that coordinates agents and validates results.

    This replaces the rule-based should_continue() function and the
    separate ValidationAgent with intelligent, context-aware decisions.
    """

    def __init__(self, llm_model: str = "deepseek-chat"):
        """Initialize the OrchestratorLLM."""
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

        self.llm_callback = create_llm_callback("orchestrator", "orchestrator_decision")

    async def evaluate_retrieval(
        self,
        user_query: str,
        retrieved_triples: list[str],
        conversation_history: list[dict],
        current_strategy: str = "semantic",
    ) -> tuple[OrchestratorDecision, list[dict]]:
        """
        Evaluate the retrieval results and decide next step.

        Args:
            user_query: The original user question
            retrieved_triples: Schema elements found by retrieval
            conversation_history: Persistent LLM conversation history (serialized messages)
            current_strategy: Current retrieval strategy (grep/semantic)

        Returns:
            Tuple of (OrchestratorDecision, updated conversation history)
        """
        import time as _time
        tracer = get_tracer()
        start_time = _time.time()

        # Emit AGENT_START
        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.AGENT_START,
                agent="orchestrator_llm",
                phase="retrieval",
                data={"method": "evaluate_retrieval", "strategy": current_strategy},
            )
        )

        # Count actual RDF triples, not just list length
        actual_triple_count = count_actual_triples(retrieved_triples)
        logger.info(f"Orchestrator evaluating retrieval: {actual_triple_count} triples found")

        # Format schema triples
        schema_str = "\n".join(retrieved_triples[:50]) if retrieved_triples else "No schema elements found."

        # Build prompt for this evaluation
        prompt = f"""## Retrieval Evaluation

User Query: {user_query}
Strategy: {current_strategy}
Found Schema Elements ({actual_triple_count} triples):

{schema_str}

---

Analyze these results and decide:
- "generation": Schema is sufficient, proceed to SPARQL generation
- "retrieval": Need more schema elements (provide specific search hint)
- "unanswerable": Query cannot be answered (only after multiple failed attempts)

Respond in JSON:
{{"decision": "generation|retrieval|unanswerable", "next_phase": "generation|retrieval|null", "feedback": "...", "reasoning": "..."}}"""

        # Restore previous conversation or start fresh
        if conversation_history:
            messages = deserialize_messages(conversation_history)
        else:
            messages = [SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT)]

        # Add current prompt
        messages.append(HumanMessage(content=prompt))

        try:
            response = await self.llm.ainvoke(
                messages,
                config={"callbacks": [self.llm_callback]},
            )

            # Add AI response to conversation
            content_str = normalize_content(response.content)
            messages.append(AIMessage(content=content_str))

            decision = self._parse_decision(content_str)

            # NO automatic strategy switching - for fair comparison between approaches
            decision.should_change_strategy = False
            decision.new_strategy = None

            # Emit AGENT_END
            duration_ms = (_time.time() - start_time) * 1000
            tracer.emit(
                TraceEvent(
                    event_type=TraceEventType.AGENT_END,
                    agent="orchestrator_llm",
                    phase="retrieval",
                    duration_ms=duration_ms,
                    data={"method": "evaluate_retrieval", "decision": decision.next_phase},
                )
            )

            return decision, serialize_messages(messages)

        except Exception as e:
            logger.error(f"Orchestrator retrieval evaluation failed: {e}")

            # Emit AGENT_END (error case)
            duration_ms = (_time.time() - start_time) * 1000
            tracer.emit(
                TraceEvent(
                    event_type=TraceEventType.AGENT_END,
                    agent="orchestrator_llm",
                    phase="retrieval",
                    duration_ms=duration_ms,
                    data={"method": "evaluate_retrieval", "error": str(e)},
                )
            )

            # Fallback: continue to generation if we have any triples
            return OrchestratorDecision(
                next_phase="generation" if retrieved_triples else "retrieval",
                feedback_for_next_phase=f"Evaluation failed: {e}",
                reasoning="Fallback decision after error",
            ), serialize_messages(messages)

    async def execute_and_evaluate_single(
        self,
        user_query: str,
        query: str,
        schema_context: list[str],
        target_endpoints: list[str] | None = None,
        conversation_history: list[dict] | None = None,
    ) -> tuple[OrchestratorDecision, list[dict]]:
        """
        Execute single query, validate semantically, and let SPARQL agent decide.

        This is the NEW iterative evaluation method where SPARQL agent is the decision maker.

        Args:
            user_query: The original user question
            query: Single SPARQL query to evaluate
            schema_context: Schema triples used for generation
            target_endpoints: Optional list of endpoints to query
            conversation_history: Persistent LLM conversation history (serialized messages)

        Returns:
            Tuple of (OrchestratorDecision, updated conversation history)
        """
        logger.info(f"Executing and evaluating single query with SPARQL agent decision")

        tracer = get_tracer()
        method_start_time = time.time()

        # Emit AGENT_START
        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.AGENT_START,
                agent="orchestrator_llm",
                phase="generation",
                data={"method": "execute_and_evaluate_single"},
            )
        )

        # Initialize conversation history
        if conversation_history is None:
            conversation_history = []

        if not query:
            # Emit AGENT_END (no query case)
            duration_ms = (time.time() - method_start_time) * 1000
            tracer.emit(
                TraceEvent(
                    event_type=TraceEventType.AGENT_END,
                    agent="orchestrator_llm",
                    phase="generation",
                    duration_ms=duration_ms,
                    data={"method": "execute_and_evaluate_single", "status": "no_query"},
                )
            )
            return OrchestratorDecision(
                next_phase="generation",
                feedback_for_next_phase="No query generated. Try again.",
                reasoning="No query to execute",
            ), conversation_history

        # 1. Execute query
        loop = asyncio.get_event_loop()
        start_time = time.time()

        try:
            query_endpoints = get_endpoints_for_query(query)
            if query_endpoints:
                result = await loop.run_in_executor(
                    _query_executor,
                    lambda q=query, eps=query_endpoints: execute_sparql.invoke({
                        "query": q,
                        "endpoints": eps,
                    })
                )
            else:
                result = await loop.run_in_executor(
                    _query_executor,
                    lambda q=query: execute_sparql.invoke({"query": q})
                )
            result["duration_ms"] = (time.time() - start_time) * 1000
        except Exception as e:
            result = {
                "success": False,
                "error_type": "execution_error",
                "error_message": str(e),
                "duration_ms": (time.time() - start_time) * 1000,
            }

        # Emit TOOL_CALL event for execute_sparql
        sparql_result_count = len(result.get("results", {}).get("bindings", [])) if result.get("success") else 0
        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.TOOL_CALL,
                phase="generation",
                data={
                    "tool_name": "execute_sparql",
                    "agent": "orchestrator_llm",
                    "success": result.get("success", False),
                    "duration_ms": result.get("duration_ms", 0),
                    "result_count": sparql_result_count,
                    "input_summary": f"query={len(query)} chars",
                },
            )
        )

        # Emit SPARQL execution event
        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.SPARQL_EXECUTION,
                phase="generation",
                data={
                    "query_count": 1,
                    "success": result.get("success", False),
                },
            )
        )

        # 2. Semantic Validation (even if execution failed - may provide useful info)
        semantic_result = None
        try:
            dataset_ids = self._extract_dataset_ids(query)
            semantic_result = validate_sparql_semantics(
                query=query,
                dataset_ids=dataset_ids,
                include_suggestions=True,
            )
        except Exception as e:
            logger.warning(f"Semantic validation failed: {e}")
            semantic_result = None

        # 3. Build feedback for SPARQL agent
        bindings = result.get("results", {}).get("bindings", []) if result.get("success") else []

        # Extract error message from result (can be in error_message or errors list)
        execution_error = None
        if not result.get("success"):
            if result.get("error_message"):
                execution_error = result.get("error_message")
            elif result.get("errors"):
                # Combine errors from errors list
                error_msgs = [e.get("error_message", str(e)) for e in result.get("errors", [])]
                execution_error = "; ".join(error_msgs)

        feedback = QueryFeedback(
            query=query,
            user_query=user_query,
            schema_context=schema_context,
            execution_success=result.get("success", False),
            execution_error=execution_error,
            semantic_validation=semantic_result,
            sample_results=self._format_sample_results(bindings[:5]) if bindings else None,
            total_count=len(bindings),
            variables=result.get("results", {}).get("vars", []) if result.get("success") else [],
        )

        # 4. Bei Query-Fehler: Metadata extrahieren und als Feedback bereitstellen
        if not result.get("success") and execution_error:
            # Extract raw metadata for the failed query (LLM decides interpretation)
            try:
                settings = get_settings()
                metadata = extract_query_metadata(
                    query=query,
                    error_message=execution_error,
                    endpoint_url=settings.ontop_sparql_url,
                    fetch_counts=True,  # Get class sizes to help with optimization
                )
                metadata_feedback = metadata.to_feedback_string()
            except Exception as e:
                logger.warning(f"Failed to extract query metadata: {e}")
                metadata_feedback = ""

            # Timeout: Direct ADJUST without LLM call
            if "timeout" in execution_error.lower():
                logger.info("Query timeout detected - returning directly to SPARQL agent with optimization hints")
                timeout_feedback = f"Query timed out.\n\n{get_ontop_timeout_hints()}"
                if metadata_feedback:
                    timeout_feedback += f"\n\n{metadata_feedback}"

                # Emit AGENT_END (timeout case)
                duration_ms = (time.time() - method_start_time) * 1000
                tracer.emit(
                    TraceEvent(
                        event_type=TraceEventType.AGENT_END,
                        agent="orchestrator_llm",
                        phase="generation",
                        duration_ms=duration_ms,
                        data={"method": "execute_and_evaluate_single", "status": "timeout"},
                    )
                )

                return OrchestratorDecision(
                    next_phase="generation",
                    feedback_for_next_phase=timeout_feedback,
                    reasoning="Timeout - automatic retry with optimization hints",
                    sparql_agent_decision=SPARQLAgentDecision(action="ADJUST", reason="Query timeout"),
                ), conversation_history

            # Other OnTop errors: Add metadata to feedback for LLM to interpret
            if metadata_feedback:
                feedback.execution_error = f"{execution_error}\n\n{metadata_feedback}"

        # 5. SPARQL Agent makes the decision (with persistent conversation)
        decision, updated_history = await self._get_sparql_agent_decision(feedback, conversation_history)

        # Attach query and results to decision
        decision.selected_query = query
        if result.get("success"):
            decision.selected_results = result

        # Emit AGENT_END
        duration_ms = (time.time() - method_start_time) * 1000
        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.AGENT_END,
                agent="orchestrator_llm",
                phase="generation",
                duration_ms=duration_ms,
                data={"method": "execute_and_evaluate_single", "decision": decision.next_phase},
            )
        )

        return decision, updated_history

    async def execute_and_evaluate(
        self,
        user_query: str,
        queries: list[str],
        schema_context: list[str],
        history: list[dict],
        failed_approaches: list[str],
        target_endpoints: list[str] | None = None,
    ) -> OrchestratorDecision:
        """
        Execute all queries and evaluate the results.

        This is the core method that replaces the ValidationAgent.

        Args:
            user_query: The original user question
            queries: Generated SPARQL queries
            schema_context: Schema triples used for generation
            history: Previous orchestrator decisions
            failed_approaches: Previously failed query approaches
            target_endpoints: Optional list of endpoints to query

        Returns:
            OrchestratorDecision with selected query and results
        """
        logger.info(f"Orchestrator executing and evaluating {len(queries)} queries")

        tracer = get_tracer()
        method_start_time = time.time()

        # Emit AGENT_START
        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.AGENT_START,
                agent="orchestrator_llm",
                phase="generation",
                data={"method": "execute_and_evaluate", "query_count": len(queries)},
            )
        )

        if not queries:
            # Emit AGENT_END (no queries case)
            duration_ms = (time.time() - method_start_time) * 1000
            tracer.emit(
                TraceEvent(
                    event_type=TraceEventType.AGENT_END,
                    agent="orchestrator_llm",
                    phase="generation",
                    duration_ms=duration_ms,
                    data={"method": "execute_and_evaluate", "status": "no_queries"},
                )
            )
            return OrchestratorDecision(
                next_phase="generation",
                feedback_for_next_phase="No queries generated. Try again.",
                reasoning="No queries to execute",
            )

        # Execute all queries in parallel
        async def execute_query_async(query: str, index: int) -> dict:
            """Execute a single query asynchronously using thread pool."""
            logger.debug(f"Executing query {index + 1}/{len(queries)}")
            loop = asyncio.get_event_loop()
            start_time = time.time()
            try:
                # Determine endpoints based on the prefixes used in THIS query
                # This ensures we only query endpoints relevant to the query's schema
                query_endpoints = get_endpoints_for_query(query)

                if query_endpoints:
                    logger.debug(f"Query {index + 1} targets endpoints: {query_endpoints}")
                    result = await loop.run_in_executor(
                        _query_executor,
                        lambda q=query, eps=query_endpoints: execute_sparql.invoke({
                            "query": q,
                            "endpoints": eps,
                        })
                    )
                else:
                    # No matching prefixes found - use default endpoint
                    logger.debug(f"Query {index + 1} using default endpoint")
                    result = await loop.run_in_executor(
                        _query_executor,
                        lambda q=query: execute_sparql.invoke({"query": q})
                    )
                # Add duration to result
                result["duration_ms"] = (time.time() - start_time) * 1000
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                return {
                    "success": False,
                    "error_type": "execution_error",
                    "error_message": str(e),
                    "duration_ms": duration_ms,
                }

        tasks = [execute_query_async(query, i) for i, query in enumerate(queries)]
        execution_results = await asyncio.gather(*tasks)

        # Emit SPARQL execution event
        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.SPARQL_EXECUTION,
                phase="generation",
                data={"query_count": len(queries)},
            )
        )

        # Emit TOOL_CALL and SPARQL result events for each query
        for i, (query, result) in enumerate(zip(queries, execution_results)):
            result_count = len(result.get("results", {}).get("bindings", [])) if result.get("success") else 0

            # Emit TOOL_CALL event
            tracer.emit(
                TraceEvent(
                    event_type=TraceEventType.TOOL_CALL,
                    phase="generation",
                    data={
                        "tool_name": "execute_sparql",
                        "agent": "orchestrator_llm",
                        "success": result.get("success", False),
                        "duration_ms": result.get("duration_ms", 0),
                        "result_count": result_count,
                        "input_summary": f"query {i+1}/{len(queries)}, {len(query)} chars",
                    },
                )
            )

            error_msg = None
            if not result.get("success") and result.get("errors"):
                errors = result.get("errors", [])
                if errors and len(errors) > 0:
                    error_msg = errors[0].get("error_message", "Unknown error")
            elif not result.get("success"):
                error_msg = result.get("error_message", "Unknown error")

            tracer.emit(
                TraceEvent(
                    event_type=TraceEventType.SPARQL_RESULT,
                    phase="generation",
                    data={
                        "index": i,
                        "query": query,
                        "success": result.get("success", False),
                        "results": result.get("results", {}),
                        "error_message": error_msg,
                        "duration_ms": result.get("duration_ms", 0.0),
                        "timeout_seconds": result.get("timeout_seconds", QUERY_TIMEOUT),
                    },
                )
            )

        # Format queries and results for analysis
        queries_results_str = self._format_queries_and_results(queries, execution_results)

        # Format other context
        history_str = self._format_history(history) if history else "No previous history."
        schema_str = "\n".join(schema_context[:30]) if schema_context else "No schema context."
        failed_str = "\n".join(f"- {a}" for a in failed_approaches[-5:]) if failed_approaches else "None."

        prompt = GENERATION_ANALYSIS_TEMPLATE.format(
            user_query=user_query,
            schema_context=schema_str,
            queries_and_results=queries_results_str,
            failed_approaches=failed_str,
            history=history_str,
        )

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ],
                config={"callbacks": [self.llm_callback]},
            )

            decision = self._parse_decision(normalize_content(response.content))

            # Ensure selected_query_index is valid (defensive check)
            idx = decision.selected_query_index if decision.selected_query_index is not None else -1

            # Always try to attach the best query and results, even if we're retrying
            # This ensures we have a fallback when max_iterations is reached
            if 0 <= idx < len(queries):
                decision.selected_query = queries[idx]
                decision.selected_results = execution_results[idx]
            else:
                # No valid selection - pick the first successful query as fallback
                for i, result in enumerate(execution_results):
                    if result.get("success"):
                        decision.selected_query = queries[i]
                        decision.selected_results = result
                        decision.selected_query_index = i
                        break

            # Emit AGENT_END
            duration_ms = (time.time() - method_start_time) * 1000
            tracer.emit(
                TraceEvent(
                    event_type=TraceEventType.AGENT_END,
                    agent="orchestrator_llm",
                    phase="generation",
                    duration_ms=duration_ms,
                    data={"method": "execute_and_evaluate", "decision": decision.next_phase},
                )
            )

            return decision

        except Exception as e:
            logger.error(f"Orchestrator generation evaluation failed: {e}")

            # Fallback: select the first successful query
            for i, result in enumerate(execution_results):
                if result.get("success"):
                    # Emit AGENT_END (fallback case)
                    duration_ms = (time.time() - method_start_time) * 1000
                    tracer.emit(
                        TraceEvent(
                            event_type=TraceEventType.AGENT_END,
                            agent="orchestrator_llm",
                            phase="generation",
                            duration_ms=duration_ms,
                            data={"method": "execute_and_evaluate", "status": "fallback_success"},
                        )
                    )
                    return OrchestratorDecision(
                        next_phase="complete",
                        feedback_for_next_phase="",
                        reasoning=f"Fallback: First successful query selected after error: {e}",
                        selected_query=queries[i],
                        selected_query_index=i,
                        selected_results=result,
                    )

            # Emit AGENT_END (no successful queries)
            duration_ms = (time.time() - method_start_time) * 1000
            tracer.emit(
                TraceEvent(
                    event_type=TraceEventType.AGENT_END,
                    agent="orchestrator_llm",
                    phase="generation",
                    duration_ms=duration_ms,
                    data={"method": "execute_and_evaluate", "status": "no_successful_queries"},
                )
            )

            # No successful queries - return to generation
            return OrchestratorDecision(
                next_phase="generation",
                feedback_for_next_phase="All queries failed. Check syntax and schema usage.",
                reasoning=f"No successful query, evaluation failed: {e}",
            )

    def _format_history(self, history: list[dict]) -> str:
        """Format orchestrator history for prompt."""
        if not history:
            return "No previous history."

        lines = []
        for i, entry in enumerate(history[-5:], 1):  # Last 5 entries
            phase = entry.get("phase", "?")
            summary = entry.get("summary", "?")
            decision = entry.get("decision", "?")
            feedback = entry.get("feedback", "")
            lines.append(f"{i}. {phase}: {summary} → {decision}")
            if feedback:
                lines.append(f"   Feedback: {feedback}")

        return "\n".join(lines)

    def _format_queries_and_results(
        self,
        queries: list[str],
        results: list[dict],
    ) -> str:
        """Format queries and their execution results for analysis."""
        lines = []

        for i, (query, result) in enumerate(zip(queries, results), 1):
            lines.append(f"### Query {i}")
            lines.append(f"```sparql\n{query}\n```")
            lines.append("")

            if result.get("success"):
                bindings = result.get("results", {}).get("bindings", [])
                lines.append(f"**Status**: Successful")
                lines.append(f"**Result count**: {len(bindings)}")

                if bindings:
                    # Show first 3 results
                    lines.append("**Sample results**:")
                    for j, binding in enumerate(bindings[:3], 1):
                        # Format binding
                        formatted = {k: v.get("value", v) for k, v in binding.items() if not k.startswith("_")}
                        lines.append(f"  {j}. {formatted}")
                    if len(bindings) > 3:
                        lines.append(f"  ... and {len(bindings) - 3} more")
                else:
                    lines.append("**Note**: Query successful but no results")
            else:
                error_type = result.get("error_type", "unknown")
                error_msg = result.get("error_message", "Unknown error")
                lines.append(f"**Status**: Failed ({error_type})")
                lines.append(f"**Error**: {error_msg[:500]}")

            lines.append("")

        return "\n".join(lines)

    def _parse_decision(self, response_content: str) -> OrchestratorDecision:
        """Parse the LLM response into an OrchestratorDecision."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_content)
            if json_match:
                data = json.loads(json_match.group(0))

                # Map decision to next_phase
                decision_type = data.get("decision", "continue")
                next_phase = data.get("next_phase")

                if decision_type == "complete":
                    next_phase = "complete"
                elif decision_type == "unanswerable":
                    next_phase = "unanswerable"
                elif not next_phase:
                    next_phase = "generation" if decision_type == "continue" else "retrieval"

                # Ensure selected_query_index is always an int (not None)
                query_idx = data.get("selected_query_index")
                if query_idx is None:
                    query_idx = -1

                # Check for unanswerable
                is_unanswerable = decision_type == "unanswerable" or next_phase == "unanswerable"
                unanswerable_reason = data.get("unanswerable_reason", "") if is_unanswerable else ""

                return OrchestratorDecision(
                    next_phase=next_phase,
                    feedback_for_next_phase=data.get("feedback", ""),
                    reasoning=data.get("reasoning", ""),
                    selected_query_index=query_idx,
                    should_change_strategy=data.get("change_strategy", False),
                    new_strategy=data.get("new_strategy"),
                    is_unanswerable=is_unanswerable,
                    unanswerable_reason=unanswerable_reason,
                )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from response: {e}")

        # Fallback parsing based on keywords
        content_lower = response_content.lower()

        if "unanswerable" in content_lower:
            return OrchestratorDecision(
                next_phase="unanswerable",
                feedback_for_next_phase="",
                reasoning="Parsed from response (fallback)",
                is_unanswerable=True,
                unanswerable_reason="Query cannot be answered with available data",
            )
        elif "complete" in content_lower or "accept" in content_lower:
            return OrchestratorDecision(
                next_phase="complete",
                feedback_for_next_phase="",
                reasoning="Parsed from response (fallback)",
                selected_query_index=0,
            )
        elif "retrieval" in content_lower or "back to retrieval" in content_lower:
            return OrchestratorDecision(
                next_phase="retrieval",
                feedback_for_next_phase="Schema incomplete",
                reasoning="Parsed from response (fallback)",
            )
        else:
            return OrchestratorDecision(
                next_phase="generation",
                feedback_for_next_phase="Improve queries",
                reasoning="Default fallback",
            )

    def _format_sample_results(self, bindings: list[dict]) -> str:
        """Format top 5 result rows for SPARQL agent review."""
        if not bindings:
            return "No results returned."

        lines = ["Top 5 Results:"]
        for i, binding in enumerate(bindings[:5], 1):
            row_parts = []
            for var, value in binding.items():
                if var.startswith("_"):
                    continue
                val = value.get("value", "N/A") if isinstance(value, dict) else str(value)
                # Truncate long values
                if len(val) > 100:
                    val = val[:100] + "..."
                row_parts.append(f"{var}={val}")
            lines.append(f"  {i}. {' | '.join(row_parts)}")

        return "\n".join(lines)

    def _extract_dataset_ids(self, query: str) -> list[str]:
        """Extract dataset IDs from query prefixes."""
        dataset_ids = []
        # Common prefix to dataset mappings
        prefix_to_dataset = {
            "edu": "edu",
            "trn": "trn",
            "bsbm": "bsbm",
            "nrg": "nrg",
        }
        query_lower = query.lower()
        for prefix, dataset in prefix_to_dataset.items():
            if prefix in query_lower:
                dataset_ids.append(dataset)
        return dataset_ids if dataset_ids else None

    async def _get_sparql_agent_decision(
        self,
        feedback: QueryFeedback,
        conversation_history: list[dict],
    ) -> tuple[OrchestratorDecision, list[dict]]:
        """
        Send feedback to SPARQL Agent and get decision back.

        SPARQL Agent responds with one of:
        - APPROVED: Query is finished
        - ADJUST: reason - Query needs adjustment
        - NEED_TRIPLE: description - Specific triple is missing

        Returns:
            Tuple of (OrchestratorDecision, updated conversation history)
        """
        tracer = get_tracer()
        method_start_time = time.time()

        # Emit AGENT_START
        tracer.emit(
            TraceEvent(
                event_type=TraceEventType.AGENT_START,
                agent="sparql_agent_decision",
                phase="generation",
                data={"method": "_get_sparql_agent_decision"},
            )
        )

        prompt = feedback.format_for_agent()
        prompt += """

## Your Decision

Analyze the feedback and decide:

1. **APPROVED** - The query works correctly and returns relevant data that answers the question
2. **ADJUST: [reason]** - The query needs adjustment (explain what needs to change)
3. **NEED_TRIPLE: [description]** - You need additional schema elements (describe semantically what's missing)

Respond with ONLY one of these formats. No additional explanation."""

        # Restore previous conversation or start fresh
        if conversation_history:
            messages = deserialize_messages(conversation_history)
        else:
            messages = [SystemMessage(content=EVALUATION_SYSTEM_PROMPT)]

        # Add current prompt
        messages.append(HumanMessage(content=prompt))

        try:
            response = await self.llm.ainvoke(
                messages,
                config={"callbacks": [self.llm_callback]},
            )

            # Add AI response to conversation
            content_str = normalize_content(response.content)
            messages.append(AIMessage(content=content_str))
            updated_history = serialize_messages(messages)

            decision = self._parse_agent_decision(content_str, feedback.query)

            # Emit trace event for SPARQL agent decision
            tracer.emit(
                TraceEvent(
                    event_type=TraceEventType.ORCHESTRATOR_DECISION,
                    phase="generation",
                    data={
                        "sparql_agent_decision": decision.sparql_agent_decision.action if decision.sparql_agent_decision else "unknown",
                        "next_phase": decision.next_phase,
                        "reasoning": decision.reasoning,
                    },
                )
            )

            # Emit AGENT_END
            duration_ms = (time.time() - method_start_time) * 1000
            tracer.emit(
                TraceEvent(
                    event_type=TraceEventType.AGENT_END,
                    agent="sparql_agent_decision",
                    phase="generation",
                    duration_ms=duration_ms,
                    data={"method": "_get_sparql_agent_decision", "decision": decision.next_phase},
                )
            )

            return decision, updated_history

        except Exception as e:
            logger.error(f"SPARQL agent decision failed: {e}")
            updated_history = serialize_messages(messages)

            # Emit AGENT_END (error case)
            duration_ms = (time.time() - method_start_time) * 1000
            tracer.emit(
                TraceEvent(
                    event_type=TraceEventType.AGENT_END,
                    agent="sparql_agent_decision",
                    phase="generation",
                    duration_ms=duration_ms,
                    data={"method": "_get_sparql_agent_decision", "error": str(e)},
                )
            )

            # Fallback: if we have successful results, approve them
            if feedback.execution_success and feedback.total_count > 0:
                return OrchestratorDecision(
                    next_phase="complete",
                    feedback_for_next_phase="",
                    reasoning=f"Fallback approval after error: {e}",
                    selected_query=feedback.query,
                    selected_results={"success": True, "results": {"bindings": []}},
                    sparql_agent_decision=SPARQLAgentDecision(action="APPROVED"),
                ), updated_history
            return OrchestratorDecision(
                next_phase="generation",
                feedback_for_next_phase=f"Error during evaluation: {e}",
                reasoning=f"Fallback to generation after error: {e}",
            ), updated_history

    def _parse_agent_decision(self, response: str, query: str) -> OrchestratorDecision:
        """Parse SPARQL agent's decision from response."""
        response = response.strip()

        # Handle multi-line responses - take first non-empty line
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        if lines:
            response = lines[0]

        if response.upper() == "APPROVED" or response.upper().startswith("APPROVED"):
            return OrchestratorDecision(
                next_phase="complete",
                feedback_for_next_phase="",
                reasoning="SPARQL agent approved the query results",
                selected_query=query,
                sparql_agent_decision=SPARQLAgentDecision(action="APPROVED"),
            )

        if response.upper().startswith("ADJUST:"):
            reason = response[7:].strip()
            return OrchestratorDecision(
                next_phase="generation",
                feedback_for_next_phase=reason,
                reasoning=f"SPARQL agent requested adjustment: {reason}",
                sparql_agent_decision=SPARQLAgentDecision(action="ADJUST", reason=reason),
            )

        if response.upper().startswith("NEED_TRIPLE:"):
            description = response[12:].strip()
            return OrchestratorDecision(
                next_phase="retrieval",
                feedback_for_next_phase=f"SPARQL Agent needs: {description}",
                reasoning="SPARQL agent needs specific schema element",
                needs_targeted_retrieval=True,
                needed_triple_description=description,
                sparql_agent_decision=SPARQLAgentDecision(
                    action="NEED_TRIPLE",
                    triple_description=description
                ),
            )

        # Fallback: try to detect intent from response
        response_lower = response.lower()
        if "approved" in response_lower or "correct" in response_lower or "valid" in response_lower:
            return OrchestratorDecision(
                next_phase="complete",
                feedback_for_next_phase="",
                reasoning="SPARQL agent approved (detected from response)",
                selected_query=query,
                sparql_agent_decision=SPARQLAgentDecision(action="APPROVED"),
            )

        # Default: treat as ADJUST request
        return OrchestratorDecision(
            next_phase="generation",
            feedback_for_next_phase=response,
            reasoning="Could not parse agent decision, treating as adjustment request",
            sparql_agent_decision=SPARQLAgentDecision(action="ADJUST", reason=response),
        )
