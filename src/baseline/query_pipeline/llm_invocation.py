"""
LLM invocation wrappers and token usage tracking.

Simplified version for experiment - no workspace DB dependencies.
Always assumes OnTop endpoint.
"""

import asyncio
import logging
import re
from typing import Union, Optional, List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate

from ..llm_services import OllamaLLM, OpenAILLM, DeepSeekLLM, LLMResponse
from ..config import get_settings
from ..prompts import FEW_SHOT_EXAMPLES, FEW_SHOT_EXAMPLES_SPARQL_ONLY, OUTPUT_SEPARATOR
from .sparql_analysis import robust_json_parse

logger = logging.getLogger(__name__)


class PipelineTokenUsage:
    """Track token usage throughout the pipeline."""

    def __init__(self):
        self.step_tokens = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0

    def add_step_usage(self, step_name: str, llm_response: LLMResponse):
        """Add token usage for a pipeline step."""
        self.step_tokens[step_name] = {
            "input_tokens": llm_response.input_tokens,
            "output_tokens": llm_response.output_tokens,
            "total_tokens": llm_response.total_tokens
        }
        self.total_input_tokens += llm_response.input_tokens
        self.total_output_tokens += llm_response.output_tokens
        self.total_tokens += llm_response.total_tokens

    def get_summary(self) -> dict:
        """Get a summary of all token usage."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "step_breakdown": self.step_tokens
        }


def _parse_llm_response_with_separator(
    raw_response: str,
    request_id: str,
    step_name: str
) -> tuple[str, str]:
    """Parse LLM response into reasoning and structured output parts."""
    reasoning_text, structured_output_text = "", raw_response.strip()
    if OUTPUT_SEPARATOR in raw_response:
        reasoning_text, structured_output_text = (s.strip() for s in raw_response.split(OUTPUT_SEPARATOR, 1))
    return reasoning_text, structured_output_text


async def _invoke_llm_and_parse(
    llm_instance: Union[OllamaLLM, OpenAILLM, DeepSeekLLM],
    prompt_template: ChatPromptTemplate,
    format_kwargs: dict,
    output_instructions_str: str,
    few_shot_prompting_enabled: bool,
    request_id: str,
    step_name: str
) -> tuple[str, str, str]:
    """Invoke LLM and parse response into reasoning and structured output."""
    steps_with_hardcoded_examples = ["IdentifyKeywords", "SelectModelsWithInstanceFilter", "FinalModelValidation"]

    if step_name in steps_with_hardcoded_examples:
        few_shot_examples_to_inject = ""
    else:
        few_shot_examples_to_inject = FEW_SHOT_EXAMPLES.get(step_name, "") if few_shot_prompting_enabled else ""

    all_format_kwargs = {
        **format_kwargs,
        "optional_few_shot_examples": few_shot_examples_to_inject,
        "final_output_instructions": output_instructions_str
    }
    prompt_messages = prompt_template.format_messages(**all_format_kwargs)
    prompt_str = "".join([m.content for m in prompt_messages])

    raw_response = await asyncio.to_thread(llm_instance.invoke, prompt_str)

    reasoning, structured_output = _parse_llm_response_with_separator(raw_response, request_id, step_name)
    return reasoning, structured_output, raw_response


async def _invoke_llm_and_parse_with_usage(
    llm_instance: Union[OllamaLLM, OpenAILLM, DeepSeekLLM],
    prompt_template: ChatPromptTemplate,
    format_kwargs: dict,
    output_instructions_str: str,
    few_shot_prompting_enabled: bool,
    request_id: str,
    step_name: str
) -> tuple[str, str, str, LLMResponse]:
    """Invoke LLM with token usage tracking."""
    steps_with_hardcoded_examples = ["IdentifyKeywords", "SelectModelsWithInstanceFilter", "FinalModelValidation"]

    if step_name in steps_with_hardcoded_examples:
        few_shot_examples_to_inject = ""
    else:
        few_shot_examples_to_inject = FEW_SHOT_EXAMPLES.get(step_name, "") if few_shot_prompting_enabled else ""

    all_format_kwargs = {
        **format_kwargs,
        "optional_few_shot_examples": few_shot_examples_to_inject,
        "final_output_instructions": output_instructions_str
    }

    prompt_messages = prompt_template.format_messages(**all_format_kwargs)
    prompt_str = "".join([m.content for m in prompt_messages])

    llm_response = await asyncio.to_thread(llm_instance.invoke_with_usage, prompt_str)

    logger.info(f"[{request_id}] Token usage ({step_name}): Input={llm_response.input_tokens}, Output={llm_response.output_tokens}")

    reasoning, structured_output = _parse_llm_response_with_separator(llm_response.content, request_id, step_name)

    return reasoning, structured_output, llm_response.content, llm_response


async def _invoke_llm_and_parse_with_adaptive_few_shot(
    llm_instance: Union[OllamaLLM, OpenAILLM, DeepSeekLLM],
    prompt_template: ChatPromptTemplate,
    format_kwargs: dict,
    output_instructions_str: str,
    few_shot_prompting_enabled: bool,
    adaptive_few_shot_enabled: bool,
    request_id: str,
    step_name: str,
    workspace_id: str,
    is_sparql_only: bool = False,
    exclude_template_ids: Optional[List[str]] = None
) -> tuple[str, str, str, list, Optional[dict]]:
    """
    Invoke LLM with adaptive few-shot example selection.

    Simplified version - always uses static few-shot examples (no adaptive selection).
    """
    steps_with_hardcoded_examples = ["IdentifyKeywords", "SelectModelsWithInstanceFilter", "FinalModelValidation"]

    # Always use SPARQL-only examples for OnTop
    few_shot_dict = FEW_SHOT_EXAMPLES_SPARQL_ONLY

    few_shot_examples_to_inject = ""
    adaptive_examples_metadata = []
    operator_token_usage = None

    if step_name in steps_with_hardcoded_examples:
        few_shot_examples_to_inject = ""
    elif few_shot_prompting_enabled:
        few_shot_examples_to_inject = few_shot_dict.get(step_name, "")

    all_format_kwargs = {
        **format_kwargs,
        "optional_few_shot_examples": few_shot_examples_to_inject,
        "final_output_instructions": output_instructions_str
    }

    prompt_messages = prompt_template.format_messages(**all_format_kwargs)
    prompt_str = "".join([m.content for m in prompt_messages])

    settings = get_settings()
    timeout_seconds = float(settings.llm_timeout_seconds)

    try:
        raw_response = await asyncio.wait_for(
            asyncio.to_thread(llm_instance.invoke, prompt_str),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        raise Exception(f"LLM call timed out after {timeout_seconds}s for {step_name}")

    reasoning, structured_output = _parse_llm_response_with_separator(raw_response, request_id, step_name)

    return reasoning, structured_output, raw_response, adaptive_examples_metadata, operator_token_usage


async def _invoke_llm_and_parse_with_adaptive_few_shot_with_usage(
    llm_instance: Union[OllamaLLM, OpenAILLM, DeepSeekLLM],
    prompt_template: ChatPromptTemplate,
    format_kwargs: dict,
    output_instructions_str: str,
    few_shot_prompting_enabled: bool,
    adaptive_few_shot_enabled: bool,
    request_id: str,
    step_name: str,
    workspace_id: str,
    is_sparql_only: bool = False,
    exclude_template_ids: Optional[List[str]] = None
) -> tuple[str, str, str, LLMResponse, list, Optional[dict]]:
    """
    Invoke LLM with adaptive few-shot selection and token usage tracking.

    Simplified version - always uses static few-shot examples (no adaptive selection).
    """
    steps_with_hardcoded_examples = ["IdentifyKeywords", "SelectModelsWithInstanceFilter", "FinalModelValidation"]

    # Always use SPARQL-only examples for OnTop
    few_shot_dict = FEW_SHOT_EXAMPLES_SPARQL_ONLY

    few_shot_examples_to_inject = ""
    adaptive_examples_metadata = []
    operator_token_usage = None

    if step_name in steps_with_hardcoded_examples:
        few_shot_examples_to_inject = ""
    elif few_shot_prompting_enabled:
        few_shot_examples_to_inject = few_shot_dict.get(step_name, "")

    all_format_kwargs = {
        **format_kwargs,
        "optional_few_shot_examples": few_shot_examples_to_inject,
        "final_output_instructions": output_instructions_str
    }

    prompt_messages = prompt_template.format_messages(**all_format_kwargs)
    prompt_str = "".join([m.content for m in prompt_messages])

    settings = get_settings()
    timeout_seconds = float(settings.llm_timeout_seconds)

    try:
        llm_response = await asyncio.wait_for(
            asyncio.to_thread(llm_instance.invoke_with_usage, prompt_str),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        raise Exception(f"LLM call timed out after {timeout_seconds}s for {step_name}")

    logger.info(f"[{request_id}] Token usage ({step_name}): Input={llm_response.input_tokens}, Output={llm_response.output_tokens}")

    reasoning, structured_output = _parse_llm_response_with_separator(llm_response.content, request_id, step_name)

    return reasoning, structured_output, llm_response.content, llm_response, adaptive_examples_metadata, operator_token_usage


async def identify_keywords_from_query_with_usage(
    user_query: str,
    llm_instance: Union[OllamaLLM, OpenAILLM, DeepSeekLLM],
    few_shot_prompting_enabled: bool = False,
    agentic_reasoning_enabled: bool = False,
    request_id: str = "N/A"
) -> tuple[List[str], LLMResponse]:
    """Extract keywords from user query with token usage tracking."""
    from ..prompts import IDENTIFY_KEYWORDS_PROMPT_TEMPLATE, OUTPUT_INSTRUCTIONS_IDENTIFY_KEYWORDS

    logger.info(f"[{request_id}] Identifying keywords for: '{user_query[:100]}...'")
    try:
        _, structured, raw_response, llm_response = await _invoke_llm_and_parse_with_usage(
            llm_instance,
            IDENTIFY_KEYWORDS_PROMPT_TEMPLATE,
            {"query": user_query},
            OUTPUT_INSTRUCTIONS_IDENTIFY_KEYWORDS,
            few_shot_prompting_enabled,
            request_id,
            "IdentifyKeywords"
        )
        response_data = robust_json_parse(structured, request_id)
        if response_data:
            keywords = response_data.get("suchbegriffe", [])
            if keywords:
                logger.info(f"[{request_id}] Extracted keywords: {keywords}")
                return keywords, llm_response
            return [], llm_response
        return [], llm_response
    except Exception as e:
        logger.error(f"[{request_id}] Keyword identification error: {e}")
        return [], LLMResponse(content="", input_tokens=0, output_tokens=0, total_tokens=0)
