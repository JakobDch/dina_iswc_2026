

import os
from pathlib import Path
from collections import defaultdict
import logging
from typing import Dict, Optional
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from app.config import DATA_DIR, LLM_PROFILES, DEFAULT_RETRIEVAL_TOKEN_BUDGET

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large" # Matched to embedding_processor.py
FAISS_INDEX_DIRECTORY_NAME = "vector_store_semantic_models"

TOP_K                = 80
MIN_HITS             = 4
TOP_N_FOR_AVG        = 4
CANDIDATE_MARGIN     = 0.2
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FAISS_INDEX_CACHE: Dict[str, Optional[FAISS]] = {}


def clear_faiss_cache_for_workspace(workspace_id: str):
    """Removes a specific workspace's FAISS index from the memory cache."""
    if workspace_id in FAISS_INDEX_CACHE:
        logger.info(f"Invalidating and removing FAISS index for workspace '{workspace_id}' from cache.")
        FAISS_INDEX_CACHE.pop(workspace_id, None)
    else:
        logger.info(f"Cache invalidation for workspace '{workspace_id}' requested, but it was not in cache.")


import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

embedding_model_hf_global = None
embedding_model_loading = False
embedding_model_loaded = False
embedding_model_load_error = None
_embedding_load_lock = threading.Lock()

def _load_embedding_model():
    """Loads the embedding model synchronously (runs in separate thread)."""
    global embedding_model_hf_global, embedding_model_loading, embedding_model_loaded, embedding_model_load_error
    
    with _embedding_load_lock:
        if embedding_model_loaded or embedding_model_loading:
            return
        embedding_model_loading = True
    
    logger.info(f"BACKGROUND: Starting to load embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        
        with _embedding_load_lock:
            embedding_model_hf_global = model
            embedding_model_loaded = True
            embedding_model_loading = False
            embedding_model_load_error = None
        
        logger.info("BACKGROUND: Global embedding model loaded successfully.")
    except Exception as e:
        with _embedding_load_lock:
            embedding_model_load_error = str(e)
            embedding_model_loading = False
            embedding_model_loaded = False
        logger.error(f"BACKGROUND: Failed to load global embedding model: {e}", exc_info=True)

def start_embedding_model_loading():
    """Starts loading the embedding model in a separate thread."""
    global embedding_model_loading

    with _embedding_load_lock:
        if embedding_model_loading or embedding_model_loaded:
            return

    # Start loading in a separate thread
    thread = threading.Thread(target=_load_embedding_model, daemon=True)
    thread.start()
    logger.info("BACKGROUND: Started embedding model loading in background thread.")

def wait_for_embedding_model(timeout_seconds: int = 30) -> bool:
    """Waits until the embedding model is loaded or timeout is reached.

    Returns:
        bool: True if model loaded successfully, False on timeout or error
    """
    import time
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        with _embedding_load_lock:
            if embedding_model_loaded:
                return True
            if embedding_model_load_error:
                logger.error(f"Embedding model loading failed: {embedding_model_load_error}")
                return False

        time.sleep(0.1)  # Brief wait
    
    logger.warning(f"Timeout waiting for embedding model to load ({timeout_seconds}s)")
    return False

def get_embedding_model_status():
    """Returns the current status of the embedding model."""
    with _embedding_load_lock:
        return {
            "loaded": embedding_model_loaded,
            "loading": embedding_model_loading,
            "error": embedding_model_load_error,
            "available": embedding_model_hf_global is not None
        }

def get_rag_parameters_for_workspace(workspace_id: str) -> dict:
    """
    Returns workspace-specific RAG parameters based on the workspace's SPARQL endpoint type.

    OnTop workspaces need more generous parameters for better recall because:
    - R2RML mappings create more complex semantic structures
    - Virtual RDF graphs have different similarity patterns than native RDF

    Args:
        workspace_id: The workspace ID to get parameters for

    Returns:
        Dictionary with k, min_model_hits, candidate_margin, top_n_for_avg parameters
    """
    from sqlmodel import Session, select
    from .db import engine
    from .models import Workspace

    try:
        with Session(engine) as session:
            workspace = session.exec(select(Workspace).where(Workspace.id == workspace_id)).first()

            if workspace and workspace.sparql_endpoint_type in ["ontop", "both"]:
                # OnTop workspaces: Slightly more generous parameters for better recall
                return {
                    "k": 60,               
                    "min_model_hits": 3,   
                    "candidate_margin": 0.1,  # 18% tolerance (closer to GraphDB)
                    "top_n_for_avg": 3     # Same as GraphDB
                }
            else:
                # GraphDB workspaces: Current parameters work well
                return {
                    "k": 30,
                    "min_model_hits": 3,
                    "candidate_margin": 0.1,
                    "top_n_for_avg": 3
                }
    except Exception as e:
        logger.warning(f"Could not load workspace {workspace_id} for RAG parameters, using defaults: {e}")
        # Default to GraphDB parameters if workspace lookup fails
        return {
            "k": 30,
            "min_model_hits": 3,
            "candidate_margin": 0.1,
            "top_n_for_avg": 3
        }

def apply_token_budget_filter(
    qualified_models: Dict[str, float],
    workspace_id: str,
    token_budget: int,
    request_id: str = "N/A"
) -> Dict[str, float]:
    """
    Filters qualified models by token budget using precomputed token counts from DB.

    Models are sorted by their average score (best first) and added until the
    token budget is exhausted. This ensures the most relevant models are prioritized.

    Args:
        qualified_models: Dict of model_name -> avg_score (already passed score threshold)
        workspace_id: Workspace ID for DB lookup
        token_budget: Maximum total tokens allowed for all selected models
        request_id: For logging purposes

    Returns:
        Filtered dict of model_name -> avg_score within token budget
    """
    from sqlmodel import Session, select
    from .db import engine
    from .models import SemanticModel

    # Sort by score (lowest = best first)
    sorted_models = sorted(qualified_models.items(), key=lambda x: x[1])

    result = {}
    accumulated_tokens = 0

    with Session(engine) as session:
        for model_name, score in sorted_models:
            # Look up precomputed token count from DB
            model = session.exec(
                select(SemanticModel).where(
                    SemanticModel.workspace_id == workspace_id,
                    SemanticModel.filename == model_name
                )
            ).first()

            if not model:
                logger.warning(f"[{request_id}] Model {model_name} not found in DB, skipping token budget check")
                # Include model anyway if not found (backwards compatibility)
                result[model_name] = score
                continue

            if model.token_count is None:
                logger.warning(f"[{request_id}] No token count for {model_name}, skipping token budget check")
                # Include model anyway if no token count (backwards compatibility for non-re-embedded models)
                result[model_name] = score
                continue

            if accumulated_tokens + model.token_count <= token_budget:
                result[model_name] = score
                accumulated_tokens += model.token_count
                logger.debug(f"[{request_id}] Added {model_name} ({model.token_count} tokens), "
                           f"accumulated: {accumulated_tokens}/{token_budget}")
            else:
                logger.info(f"[{request_id}] Token budget {token_budget} reached at {accumulated_tokens} tokens. "
                          f"Excluding {model_name} ({model.token_count} tokens) and remaining models.")
                break

    logger.info(f"[{request_id}] Token budget filter: {len(qualified_models)} -> {len(result)} models, "
               f"{accumulated_tokens}/{token_budget} tokens used")

    return result


def search_workspace_embeddings(
    workspace_id: str,
    query_text: str,
    k: int = 30,
    min_model_hits: int = 3,
    candidate_margin: float = 0.1,
    top_n_for_avg: int = 3,
    token_budget: Optional[int] = None,
    llm_profile: Optional[str] = None
):
    """Searches workspace embeddings and returns relevant triples filtered by defined criteria.

    Args:
        workspace_id: The workspace to search in
        query_text: The search query
        k: Number of top triples to retrieve from FAISS
        min_model_hits: Minimum hits per model to be considered
        candidate_margin: Score tolerance margin for model threshold
        top_n_for_avg: Number of top scores to use for model average calculation
        token_budget: Optional maximum total tokens for selected models (overrides llm_profile)
        llm_profile: Optional LLM profile name to automatically determine token budget
    """
    
    vectorstore = None
    if workspace_id in FAISS_INDEX_CACHE:
        logger.info(f"FAISS index for workspace {workspace_id} found in cache. Using cached index.")
        vectorstore = FAISS_INDEX_CACHE[workspace_id]
    else:
        logger.info(f"FAISS index for workspace {workspace_id} not in cache. Attempting to load from disk.")
        index_folder_path = Path(DATA_DIR) / workspace_id / FAISS_INDEX_DIRECTORY_NAME
        logger.info(f"Attempting to load FAISS index from: {index_folder_path}")

        if not index_folder_path.exists() or not (index_folder_path / "index.faiss").exists():
            logger.warning(f"FAISS index not found for workspace {workspace_id} at {index_folder_path}")
            return {"error": "Embeddings index not found for this workspace. Please embed models first.", "results": []}

        try:
            if not embedding_model_loaded:
                if embedding_model_loading:
                    logger.info("Embedding model is still loading. Waiting...")
                    if not wait_for_embedding_model(timeout_seconds=30):
                        return {"error": "Embedding model is still loading or failed to load. Please try again later.", "results": []}
                else:
                    logger.error("Embedding model was not started. Starting background loading now.")
                    start_embedding_model_loading()
                    if not wait_for_embedding_model(timeout_seconds=30):
                        return {"error": "Embedding model could not be loaded. Check server logs.", "results": []}
            
            if not embedding_model_hf_global:
                logger.error("Global embedding model is not available after loading attempt.")
                return {"error": "Embedding model could not be loaded. Check server logs.", "results": []}

            logger.info(f"Loading FAISS index from {index_folder_path}...")
            vectorstore = FAISS.load_local(
                folder_path=str(index_folder_path),
                embeddings=embedding_model_hf_global,
                allow_dangerous_deserialization=True
            )
            logger.info("FAISS index loaded successfully and will be cached.")
            # Store the freshly loaded index in the cache
            FAISS_INDEX_CACHE[workspace_id] = vectorstore

        except Exception as e:
            logger.error(f"Error loading FAISS index or embedding model: {e}", exc_info=True)
            return {"error": f"Failed to load embeddings: {str(e)}", "results": []}

    # The rest of the function uses the `vectorstore` object

    logger.info(f"Performing similarity search for query: \"{query_text}\" with k={k}, min_model_hits={min_model_hits}, candidate_margin={candidate_margin}, top_n_for_avg={top_n_for_avg}")
    try:
        raw_hits_with_scores = vectorstore.similarity_search_with_score(query_text, k=k)
    except Exception as e:
        logger.error(f"Error during similarity search: {e}", exc_info=True)
        return {"error": f"Error during search: {str(e)}", "results": []}

    logger.info(f"Retrieved {len(raw_hits_with_scores)} raw hits from FAISS.")

    if len(raw_hits_with_scores) < min_model_hits:
        logger.info(f"Number of hits ({len(raw_hits_with_scores)}) is less than min_model_hits ({min_model_hits}). Returning few results.")
        return {"message": f"Too few results ({len(raw_hits_with_scores)} found, minimum is {min_model_hits}).", "results": []}

    hits_by_model = defaultdict(list)
    for doc, score in raw_hits_with_scores:
        source_file = doc.metadata.get("source_file", "unknown_model.ttl")
        model_name = Path(source_file).name
        hits_by_model[model_name].append({
            "text": doc.page_content,
            "score": float(score),
            "source_model": model_name
        })

    logger.info(f"Grouped hits by {len(hits_by_model)} models.")

    model_avg_scores = {}
    for model_name, model_hits_list in hits_by_model.items():
        if len(model_hits_list) >= min_model_hits:
            sorted_model_hits = sorted(model_hits_list, key=lambda x: x["score"])
            top_n_for_model = sorted_model_hits[:top_n_for_avg]

            if top_n_for_model:
                avg_score = sum(h["score"] for h in top_n_for_model) / len(top_n_for_model)
                model_avg_scores[model_name] = avg_score
                logger.debug(f"Model {model_name}: avg_score={avg_score} (from {len(top_n_for_model)} hits)")

    if not model_avg_scores:
        logger.info("No models had enough hits to calculate average scores.")
        all_raw_triples = [{"text": doc.page_content, "score": float(score), "source_model": Path(doc.metadata.get("source_file", "unknown_model.ttl")).name} for doc, score in raw_hits_with_scores]
        return {"message": "Could not determine model relevance, returning all hits.", "results": sorted(all_raw_triples, key=lambda x: x['score'])[:k]}


    best_overall_avg_score = min(model_avg_scores.values())
    score_threshold = best_overall_avg_score * (1 + candidate_margin)

    # Collect all models that pass the threshold
    qualified_models = {
        model_name: avg_s
        for model_name, avg_s in model_avg_scores.items()
        if avg_s <= score_threshold
    }

    logger.info(f"Best avg model score: {best_overall_avg_score}. Threshold: {score_threshold}. "
                f"Models passing threshold: {len(qualified_models)}.")

    # Apply token budget filter if specified
    effective_token_budget = token_budget
    if effective_token_budget is None and llm_profile:
        profile = LLM_PROFILES.get(llm_profile, {})
        effective_token_budget = profile.get("retrieval_token_budget", DEFAULT_RETRIEVAL_TOKEN_BUDGET)
        logger.info(f"Using token budget {effective_token_budget} from LLM profile '{llm_profile}'")

    if effective_token_budget and effective_token_budget > 0 and qualified_models:
        models_before_budget = len(qualified_models)
        qualified_models = apply_token_budget_filter(
            qualified_models,
            workspace_id,
            effective_token_budget,
            request_id=f"search_{workspace_id}"
        )
        logger.info(f"Token budget filter reduced models from {models_before_budget} to {len(qualified_models)}")

    final_triples = []
    for doc, score in raw_hits_with_scores:
        source_file = doc.metadata.get("source_file", "unknown_model.ttl")
        model_name = Path(source_file).name
        if model_name in qualified_models:
            final_triples.append({
                "text": doc.page_content,
                "score": float(score),
                "source_model": model_name,
                "model_average_score": qualified_models[model_name]
            })

    final_triples_sorted = sorted(final_triples, key=lambda x: x["score"])

    logger.info(f"Returning {len(final_triples_sorted)} sorted triples from qualified models.")

    return {
        "message": f"Search complete. Found {len(final_triples_sorted)} relevant triples from {len(qualified_models)} models.",
        "results": final_triples_sorted,
        "debug_info": {
            "total_raw_hits": len(raw_hits_with_scores),
            "min_hits_threshold": min_model_hits,
            "models_considered_for_avg": len(model_avg_scores),
            "best_model_avg_score": best_overall_avg_score if model_avg_scores else None,
            "score_threshold_for_models": score_threshold if model_avg_scores else None,
            "models_passing_threshold": len(threshold_qualified_models) if 'threshold_qualified_models' in locals() else len(qualified_models),
            "num_selected_models": len(qualified_models)
        }
    }


