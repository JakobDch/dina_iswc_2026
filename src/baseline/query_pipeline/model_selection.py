"""
Semantic model selection and validation.

Provides functions for selecting appropriate semantic models for a user query,
validating model compatibility, and searching for instance data.
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Union, Optional, List, Dict, Any

from rdflib import Graph
from sqlmodel import select

from ..db import get_session
from ..llm_services import OllamaLLM, OpenAILLM, DeepSeekLLM, LLMResponse
from ..models import SemanticModel, ModelDatasetMapping, Dataset
from ..prompts import IDENTIFY_KEYWORDS_PROMPT_TEMPLATE, OUTPUT_INSTRUCTIONS_IDENTIFY_KEYWORDS
from ..utils import parse_ttl_to_clean_triples
from .llm_invocation import _invoke_llm_and_parse, _invoke_llm_and_parse_with_usage
from .sparql_analysis import robust_json_parse
from .formatting import transform_graph_with_literals

logger = logging.getLogger(__name__)


async def identify_keywords_from_query(
    user_query: str,
    llm_instance: Union[OllamaLLM, OpenAILLM, DeepSeekLLM],
    few_shot_prompting_enabled: bool = False,
    agentic_reasoning_enabled: bool = False,
    request_id: str = "N/A"
) -> List[str]:
    """Extract relevant keywords from user query for embedding search."""
    logger.info(f"[{request_id}] Identifying keywords for: '{user_query[:100]}...'")
    try:
        _, structured, raw_response = await _invoke_llm_and_parse(
            llm_instance,
            IDENTIFY_KEYWORDS_PROMPT_TEMPLATE,
            {"query": user_query},
            OUTPUT_INSTRUCTIONS_IDENTIFY_KEYWORDS,
            few_shot_prompting_enabled,
            request_id,
            "IdentifyKeywords"
        )

        json_match = re.search(r'(\{[\s\S]*\})', structured.strip())
        if not json_match:
            logger.error(f"[{request_id}] No JSON found in keyword response")
            return []
        parsed_response = json.loads(json_match.group(1))

        keywords = []
        if "search_terms" in parsed_response:
            search_terms = parsed_response["search_terms"]
            if isinstance(search_terms, list):
                keywords = [str(term) for term in search_terms if term]
        elif "suchbegriffe" in parsed_response:  # Legacy German key support
            suchbegriffe = parsed_response["suchbegriffe"]
            if isinstance(suchbegriffe, list):
                keywords = [str(term) for term in suchbegriffe if term]
        elif "keywords" in parsed_response:
            legacy_keywords = parsed_response["keywords"]
            if isinstance(legacy_keywords, list):
                keywords = [str(term) for term in legacy_keywords if term]
        else:
            for key, value in parsed_response.items():
                if key.startswith("keyword-kombination"):
                    if isinstance(value, list):
                        keywords.append(" ".join(value) if len(value) > 1 else value[0])
                    elif isinstance(value, str):
                        keywords.append(value)
                else:
                    if isinstance(value, list) and value:
                        combined_terms = [key] + [str(term) for term in value if term]
                        keywords.extend(combined_terms)
                    elif isinstance(value, str) and value:
                        keywords.extend([key, value])

        if isinstance(keywords, list) and all(isinstance(kw, str) for kw in keywords):
            logger.info(f"[{request_id}] Extracted keywords: {keywords}")
            return keywords
        return []
    except Exception as e:
        logger.error(f"[{request_id}] Keyword extraction error: {e}")
        return []


async def select_models_with_instance_filter_check(
    user_query: str,
    candidate_filenames: List[str],
    workspace_id: str,
    llm_instance: Union[OllamaLLM, OpenAILLM, DeepSeekLLM],
    request_id: str,
    few_shot_prompting_enabled: bool = False
) -> Dict[str, Any]:
    """
    Select semantic models and check if instance filtering is needed.

    Returns dict with selected_models, evaluation, reasoning, searched_terms.
    """
    logger.info(f"[{request_id}] Model selection: {len(candidate_filenames)} candidates")

    content_blocks = []
    db_session = next(get_session())
    try:
        for filename in candidate_filenames:
            db_model = db_session.exec(
                select(SemanticModel).where(
                    SemanticModel.workspace_id == workspace_id,
                    SemanticModel.filename == filename
                )
            ).first()

            if db_model and db_model.stored_path:
                clean_triples = await asyncio.to_thread(
                    parse_ttl_to_clean_triples,
                    Path(db_model.stored_path),
                    request_id
                )
                if clean_triples:
                    content_blocks.append(f"Model: {filename}\n{clean_triples}")
    finally:
        db_session.close()

    if not content_blocks:
        return {
            "selected_models": [],
            "evaluation": "not_possible",
            "reasoning": "No model content could be loaded.",
            "searched_terms": [],
            "instance_filter_hints": None
        }

    combined_content = "\n\n".join(content_blocks)

    from ..prompts import (
        SELECT_MODELS_WITH_INSTANCE_FILTER_PROMPT_TEMPLATE,
        OUTPUT_INSTRUCTIONS_SELECT_MODELS_WITH_INSTANCE_FILTER
    )

    def extract_literals(content_str: str) -> set:
        literal_pattern = r'"([^"]+)"\^\^'
        literals = set()
        for match in re.finditer(literal_pattern, content_str):
            literals.add(match.group(1).lower())
        return literals

    model_literals = extract_literals(combined_content)

    try:
        _, structured, _ = await _invoke_llm_and_parse(
            llm_instance,
            SELECT_MODELS_WITH_INSTANCE_FILTER_PROMPT_TEMPLATE,
            {"query": user_query, "candidate_models_content": combined_content},
            OUTPUT_INSTRUCTIONS_SELECT_MODELS_WITH_INSTANCE_FILTER,
            few_shot_prompting_enabled,
            request_id,
            "SelectModelsWithInstanceFilter"
        )

        parsed = robust_json_parse(structured, request_id)
        if not parsed:
            return {
                "selected_models": [],
                "evaluation": "not_possible",
                "reasoning": "LLM response parsing failed.",
                "searched_terms": [],
                "instance_filter_hints": None
            }

        selected_models = parsed.get("selected_models", parsed.get("auswahl", []))
        # Support both English and legacy German keys
        evaluation = parsed.get("evaluation", parsed.get("bewertung", "not_possible"))
        reasoning = parsed.get("reasoning", parsed.get("begruendung", ""))
        searched_terms = parsed.get("searched_terms", parsed.get("gesuchte_begriffe", []))
        instance_filter_hints = parsed.get("instance_filter_hints", parsed.get("instanzen_filter_hinweise"))

        if isinstance(selected_models, list):
            validated = []
            for model in selected_models:
                if model in candidate_filenames:
                    validated.append(model)
                else:
                    for candidate in candidate_filenames:
                        llm_base = model.replace("_semantic_model", "").replace(".ttl", "")
                        cand_base = candidate.replace("_semantic_model_transformed", "").replace("_semantic_model", "").replace(".ttl", "")
                        if llm_base in cand_base or cand_base in llm_base:
                            validated.append(candidate)
                            break
            selected_models = validated

        # Map legacy German values to English
        value_mapping = {
            "vollständig_möglich": "fully_possible",
            "filter_instanzen_nötig": "instance_filter_required",
            "nicht_möglich": "not_possible",
            "teilweise_möglich": "partially_possible"
        }
        evaluation = value_mapping.get(evaluation, evaluation)

        if evaluation not in ["fully_possible", "instance_filter_required", "not_possible", "partially_possible"]:
            evaluation = "not_possible"

        if evaluation == "instance_filter_required" and isinstance(searched_terms, list):
            searched_terms = [b for b in searched_terms if b.lower() not in model_literals]
            if len(searched_terms) == 0:
                evaluation = "fully_possible"

        return {
            "selected_models": selected_models,
            "evaluation": evaluation,
            "reasoning": reasoning,
            "searched_terms": searched_terms,
            "instance_filter_hints": instance_filter_hints
        }

    except Exception as e:
        logger.error(f"[{request_id}] Model selection error: {e}")
        return {
            "selected_models": [],
            "evaluation": "not_possible",
            "reasoning": f"System error: {e}",
            "searched_terms": [],
            "instance_filter_hints": None
        }


async def select_models_with_instance_filter_check_with_usage(
    user_query: str,
    candidate_filenames: List[str],
    workspace_id: str,
    llm_instance: Union[OllamaLLM, OpenAILLM, DeepSeekLLM],
    request_id: str,
    few_shot_prompting_enabled: bool = False
) -> tuple[Dict[str, Any], LLMResponse]:
    """Model selection with token usage tracking."""
    logger.info(f"[{request_id}] Model selection (with usage): {len(candidate_filenames)} candidates")

    content_blocks = []
    db_session = next(get_session())
    try:
        for filename in candidate_filenames:
            db_model = db_session.exec(
                select(SemanticModel).where(
                    SemanticModel.workspace_id == workspace_id,
                    SemanticModel.filename == filename
                )
            ).first()

            if db_model and db_model.stored_path:
                clean_triples = await asyncio.to_thread(
                    parse_ttl_to_clean_triples,
                    Path(db_model.stored_path),
                    request_id
                )
                if clean_triples:
                    content_blocks.append(f"Model: {filename}\n{clean_triples}")
    finally:
        db_session.close()

    if not content_blocks:
        dummy = LLMResponse(content="", input_tokens=0, output_tokens=0, total_tokens=0)
        return {
            "selected_models": [],
            "evaluation": "not_possible",
            "reasoning": "No model content.",
            "searched_terms": [],
            "instance_filter_hints": None
        }, dummy

    combined_content = "\n\n".join(content_blocks)

    from ..prompts import (
        SELECT_MODELS_WITH_INSTANCE_FILTER_PROMPT_TEMPLATE,
        OUTPUT_INSTRUCTIONS_SELECT_MODELS_WITH_INSTANCE_FILTER
    )

    def extract_literals(content_str: str) -> set:
        literals = set()
        for match in re.finditer(r'"([^"]+)"\^\^', content_str):
            literals.add(match.group(1).lower())
        return literals

    model_literals = extract_literals(combined_content)

    try:
        _, structured, _, llm_response = await _invoke_llm_and_parse_with_usage(
            llm_instance,
            SELECT_MODELS_WITH_INSTANCE_FILTER_PROMPT_TEMPLATE,
            {"query": user_query, "candidate_models_content": combined_content},
            OUTPUT_INSTRUCTIONS_SELECT_MODELS_WITH_INSTANCE_FILTER,
            few_shot_prompting_enabled,
            request_id,
            "SelectModelsWithInstanceFilter"
        )

        parsed = robust_json_parse(structured, request_id)
        if not parsed:
            return {
                "selected_models": [],
                "evaluation": "not_possible",
                "reasoning": "Parsing failed.",
                "searched_terms": [],
                "instance_filter_hints": None
            }, llm_response

        selected_models = parsed.get("selected_models", parsed.get("auswahl", []))
        # Support both English and legacy German keys
        evaluation = parsed.get("evaluation", parsed.get("bewertung", "not_possible"))
        reasoning = parsed.get("reasoning", parsed.get("begruendung", ""))
        searched_terms = parsed.get("searched_terms", parsed.get("gesuchte_begriffe", []))
        instance_filter_hints = parsed.get("instance_filter_hints", parsed.get("instanzen_filter_hinweise"))

        if isinstance(selected_models, list):
            validated = [m for m in selected_models if m in candidate_filenames]
            selected_models = validated

        # Map legacy German values to English
        value_mapping = {
            "vollständig_möglich": "fully_possible",
            "filter_instanzen_nötig": "instance_filter_required",
            "nicht_möglich": "not_possible",
            "teilweise_möglich": "partially_possible"
        }
        evaluation = value_mapping.get(evaluation, evaluation)

        if evaluation not in ["fully_possible", "instance_filter_required", "not_possible", "partially_possible"]:
            evaluation = "not_possible"

        if evaluation == "instance_filter_required" and isinstance(searched_terms, list):
            searched_terms = [b for b in searched_terms if b.lower() not in model_literals]
            if not searched_terms:
                evaluation = "fully_possible"

        return {
            "selected_models": selected_models,
            "evaluation": evaluation,
            "reasoning": reasoning,
            "searched_terms": searched_terms,
            "instance_filter_hints": instance_filter_hints
        }, llm_response

    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}")
        dummy = LLMResponse(content="", input_tokens=0, output_tokens=0, total_tokens=0)
        return {
            "selected_models": [],
            "evaluation": "not_possible",
            "reasoning": f"Error: {e}",
            "searched_terms": [],
            "instance_filter_hints": None
        }, dummy


async def final_model_validation(
    user_query: str,
    selected_models: List[str],
    workspace_id: str,
    instance_search_results: Optional[Dict[str, Any]],
    llm_instance: Union[OllamaLLM, OpenAILLM, DeepSeekLLM],
    request_id: str,
    few_shot_prompting_enabled: bool = False
) -> Dict[str, Any]:
    """Final validation of selected models for query answerability."""
    logger.info(f"[{request_id}] Final validation: {len(selected_models)} models")

    if not selected_models:
        return {
            "final_evaluation": "not_possible",
            "final_reasoning": "No models available.",
            "corrected_models": []
        }

    content_blocks = []
    db_session = next(get_session())
    try:
        for filename in selected_models:
            db_model = db_session.exec(
                select(SemanticModel).where(
                    SemanticModel.workspace_id == workspace_id,
                    SemanticModel.filename == filename
                )
            ).first()

            if db_model and db_model.stored_path:
                clean_triples = await asyncio.to_thread(
                    parse_ttl_to_clean_triples,
                    Path(db_model.stored_path),
                    request_id
                )
                if clean_triples:
                    content_blocks.append(f"Model: {filename}\n{clean_triples}")
    finally:
        db_session.close()

    if not content_blocks:
        return {
            "final_evaluation": "not_possible",
            "final_reasoning": "Model content not loadable.",
            "corrected_models": []
        }

    combined_content = "\n\n".join(content_blocks)

    instance_summary = ""
    if instance_search_results and not instance_search_results.get("error"):
        found_terms = instance_search_results.get("found_terms", [])
        if found_terms:
            instance_summary = f"Instance search: {', '.join(found_terms)}\n"

    from ..prompts import (
        FINAL_MODEL_VALIDATION_PROMPT_TEMPLATE,
        OUTPUT_INSTRUCTIONS_FINAL_MODEL_VALIDATION
    )

    try:
        _, structured, _ = await _invoke_llm_and_parse(
            llm_instance,
            FINAL_MODEL_VALIDATION_PROMPT_TEMPLATE,
            {
                "query": user_query,
                "selected_models_content": combined_content,
                "instance_search_summary": instance_summary
            },
            OUTPUT_INSTRUCTIONS_FINAL_MODEL_VALIDATION,
            few_shot_prompting_enabled,
            request_id,
            "FinalModelValidation"
        )

        parsed = robust_json_parse(structured, request_id)
        if not parsed:
            return {
                "final_evaluation": "not_possible",
                "final_reasoning": "Parsing failed.",
                "corrected_models": selected_models
            }

        # Support both English and legacy German keys
        final_evaluation = parsed.get("final_evaluation", parsed.get("final_bewertung", "not_possible"))
        final_reasoning = parsed.get("final_reasoning", parsed.get("final_begruendung", ""))

        # Map legacy German values to English
        value_mapping = {
            "vollständig_möglich": "fully_possible",
            "nicht_möglich": "not_possible"
        }
        final_evaluation = value_mapping.get(final_evaluation, final_evaluation)

        if final_evaluation not in ["fully_possible", "not_possible"]:
            final_evaluation = "not_possible"

        corrected = parsed.get("corrected_models", [])
        if isinstance(corrected, list) and corrected:
            valid_corrected = [m for m in corrected if m in selected_models]
            corrected_models = valid_corrected if valid_corrected else selected_models
        else:
            corrected_models = selected_models

        return {
            "final_evaluation": final_evaluation,
            "final_reasoning": final_reasoning,
            "corrected_models": corrected_models
        }

    except Exception as e:
        logger.error(f"[{request_id}] Validation error: {e}")
        return {
            "final_evaluation": "not_possible",
            "final_reasoning": f"Error: {e}",
            "corrected_models": selected_models
        }


async def final_model_validation_with_usage(
    user_query: str,
    selected_models: List[str],
    workspace_id: str,
    instance_search_results: Optional[Dict[str, Any]],
    llm_instance: Union[OllamaLLM, OpenAILLM, DeepSeekLLM],
    request_id: str,
    few_shot_prompting_enabled: bool = False
) -> tuple[Dict[str, Any], LLMResponse]:
    """Final validation with token usage tracking."""
    logger.info(f"[{request_id}] Final validation (with usage): {len(selected_models)} models")

    if not selected_models:
        dummy = LLMResponse(content="", input_tokens=0, output_tokens=0, total_tokens=0)
        return {
            "final_evaluation": "not_possible",
            "final_reasoning": "No models.",
            "corrected_models": []
        }, dummy

    content_blocks = []
    db_session = next(get_session())
    try:
        for filename in selected_models:
            db_model = db_session.exec(
                select(SemanticModel).where(
                    SemanticModel.workspace_id == workspace_id,
                    SemanticModel.filename == filename
                )
            ).first()

            if db_model and db_model.stored_path:
                clean_triples = await asyncio.to_thread(
                    parse_ttl_to_clean_triples,
                    Path(db_model.stored_path),
                    request_id
                )
                if clean_triples:
                    content_blocks.append(f"Model: {filename}\n{clean_triples}")
    finally:
        db_session.close()

    if not content_blocks:
        dummy = LLMResponse(content="", input_tokens=0, output_tokens=0, total_tokens=0)
        return {
            "final_evaluation": "not_possible",
            "final_reasoning": "Content not loadable.",
            "corrected_models": []
        }, dummy

    combined_content = "\n\n".join(content_blocks)

    instance_summary = ""
    if instance_search_results and not instance_search_results.get("error"):
        found_terms = instance_search_results.get("found_terms", [])
        if found_terms:
            instance_summary = f"Instance search: {', '.join(found_terms)}\n"

    from ..prompts import (
        FINAL_MODEL_VALIDATION_PROMPT_TEMPLATE,
        OUTPUT_INSTRUCTIONS_FINAL_MODEL_VALIDATION
    )

    try:
        _, structured, _, llm_response = await _invoke_llm_and_parse_with_usage(
            llm_instance,
            FINAL_MODEL_VALIDATION_PROMPT_TEMPLATE,
            {
                "query": user_query,
                "selected_models_content": combined_content,
                "instance_search_summary": instance_summary
            },
            OUTPUT_INSTRUCTIONS_FINAL_MODEL_VALIDATION,
            few_shot_prompting_enabled,
            request_id,
            "FinalModelValidation"
        )

        parsed = robust_json_parse(structured, request_id)
        if not parsed:
            return {
                "final_evaluation": "not_possible",
                "final_reasoning": "Parsing failed.",
                "corrected_models": selected_models
            }, llm_response

        # Support both English and legacy German keys
        final_evaluation = parsed.get("final_evaluation", parsed.get("final_bewertung", "not_possible"))
        final_reasoning = parsed.get("final_reasoning", parsed.get("final_begruendung", ""))

        # Map legacy German values to English
        value_mapping = {
            "vollständig_möglich": "fully_possible",
            "nicht_möglich": "not_possible"
        }
        final_evaluation = value_mapping.get(final_evaluation, final_evaluation)

        if final_evaluation not in ["fully_possible", "not_possible"]:
            final_evaluation = "not_possible"

        corrected = parsed.get("corrected_models", [])
        corrected_models = [m for m in corrected if m in selected_models] if corrected else selected_models

        return {
            "final_evaluation": final_evaluation,
            "final_reasoning": final_reasoning,
            "corrected_models": corrected_models
        }, llm_response

    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}")
        dummy = LLMResponse(content="", input_tokens=0, output_tokens=0, total_tokens=0)
        return {
            "final_evaluation": "not_possible",
            "final_reasoning": f"Error: {e}",
            "corrected_models": selected_models
        }, dummy


async def search_rdf_instances(
    workspace_id: str,
    search_terms: List[str],
    matched_dataset_paths: List[str],
    request_id: str = "N/A"
) -> Dict[str, Any]:
    """Search for instance data in RDF datasets."""
    logger.info(f"[{request_id}] Searching instances: {search_terms}")

    search_results = {
        "found_terms": [],
        "results_by_term": {},
        "total_matches": 0,
        "searched_datasets": len(matched_dataset_paths),
        "skipped_terms": []
    }

    try:
        for search_term in search_terms:
            if len(search_term.strip()) < 3:
                search_results["skipped_terms"].append({
                    "term": search_term,
                    "reason": "Too short (< 3 chars)"
                })
                continue

            term_results = {
                "term": search_term,
                "matches_count": 0,
                "matches_by_dataset": {},
                "example_matches": []
            }

            for dataset_path in matched_dataset_paths:
                path_obj = Path(dataset_path)
                if not path_obj.exists():
                    continue

                try:
                    graph = Graph()
                    graph.parse(path_obj, format="ttl")
                    transformed = transform_graph_with_literals(graph)
                    content = transformed.serialize(format="ttl")

                    matches = []
                    for line_num, line in enumerate(content.split('\n'), 1):
                        line = line.strip()
                        if search_term.lower() in line.lower() and line.endswith(' .'):
                            matches.append({
                                "line_number": line_num,
                                "semantic_structure": line
                            })

                    if matches:
                        dataset_name = path_obj.name
                        term_results["matches_by_dataset"][dataset_name] = len(matches)
                        term_results["matches_count"] += len(matches)
                        for match in matches[:3]:
                            term_results["example_matches"].append({
                                "dataset": dataset_name,
                                "semantic_structure": match["semantic_structure"]
                            })

                except Exception as e:
                    logger.error(f"[{request_id}] Search error in {dataset_path}: {e}")

            if term_results["matches_count"] > 0:
                search_results["found_terms"].append(search_term)
                search_results["results_by_term"][search_term] = term_results
                search_results["total_matches"] += term_results["matches_count"]

        return search_results

    except Exception as e:
        logger.error(f"[{request_id}] Instance search error: {e}")
        return {
            "error": str(e),
            "found_terms": [],
            "results_by_term": {},
            "total_matches": 0,
            "searched_datasets": 0,
            "skipped_terms": []
        }


async def search_specific_instances_for_terms(
    workspace_id: str,
    search_terms: List[str],
    selected_models: List[str] = None,
    max_results_per_term: int = 10,
    request_id: str = "search_instances"
) -> Dict[str, List[Dict[str, str]]]:
    """
    Search for specific terms in instance data.

    Returns dictionary mapping terms to found triple information.
    """
    logger.info(f"[{request_id}] Searching specific terms: {search_terms}")

    db_session = next(get_session())
    found_instances = {term: [] for term in search_terms}

    try:
        if not selected_models:
            datasets = db_session.exec(
                select(Dataset).where(
                    Dataset.workspace_id == workspace_id,
                    Dataset.filename.like('%.ttl')
                )
            ).all()
            dataset_to_model = {}
        else:
            semantic_models = db_session.exec(
                select(SemanticModel).where(
                    SemanticModel.workspace_id == workspace_id,
                    SemanticModel.filename.in_(selected_models)
                )
            ).all()

            if not semantic_models:
                return found_instances

            model_ids = [m.id for m in semantic_models]

            mappings = db_session.exec(
                select(ModelDatasetMapping).where(
                    ModelDatasetMapping.workspace_id == workspace_id,
                    ModelDatasetMapping.semantic_model_id.in_(model_ids)
                )
            ).all()

            if not mappings:
                datasets = db_session.exec(
                    select(Dataset).where(
                        Dataset.workspace_id == workspace_id,
                        Dataset.filename.like('%.ttl')
                    )
                ).all()
                dataset_to_model = {}
            else:
                dataset_ids = [m.dataset_id for m in mappings]
                datasets = db_session.exec(
                    select(Dataset).where(Dataset.id.in_(dataset_ids))
                ).all()

                model_map = {m.id: m.filename for m in semantic_models}
                dataset_to_model = {}
                for mapping in mappings:
                    dataset_to_model[mapping.dataset_id] = model_map.get(mapping.semantic_model_id, "unknown")

        for dataset in datasets:
            if not dataset.stored_path or not Path(dataset.stored_path).exists():
                continue

            try:
                graph = Graph()
                graph.parse(dataset.stored_path, format="ttl")
                transformed = transform_graph_with_literals(graph)
                content = transformed.serialize(format="ttl")

                model_name = dataset_to_model.get(dataset.id, dataset.filename)

                for term in search_terms:
                    if len(term.strip()) < 3:
                        continue

                    if len(found_instances[term]) >= max_results_per_term:
                        continue

                    for line in content.split('\n'):
                        line = line.strip()
                        if term.lower() in line.lower() and line.endswith(' .'):
                            found_instances[term].append({
                                "semantic_structure": line,
                                "dataset": model_name,
                                "content": line
                            })
                            if len(found_instances[term]) >= max_results_per_term:
                                break

            except Exception as e:
                logger.error(f"[{request_id}] Error reading {dataset.stored_path}: {e}")

        return found_instances

    except Exception as e:
        logger.error(f"[{request_id}] Search error: {e}")
        return found_instances
    finally:
        db_session.close()
