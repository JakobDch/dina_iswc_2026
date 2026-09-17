"""Central configuration for the DINA agent experiment."""

from contextvars import ContextVar
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# Dataset Size Context (small vs large)
# ============================================================
# This allows automatic endpoint selection based on query set:
# - BASE, SYN, TYPO, UNDER, CROSS → small datasets
# - LARGE → large datasets
#
# Usage:
#   with dataset_size_context("large"):
#       # All get_ontop_endpoint() calls will return large endpoints
#       result = await agent.run(...)
#
#   # Or set globally:
#   set_dataset_size("large")

DatasetSize = Literal["small", "large"]
_dataset_size: ContextVar[DatasetSize] = ContextVar("dataset_size", default="small")
_dataset_size_global: DatasetSize = "small"  # Thread-safe fallback


def get_dataset_size() -> DatasetSize:
    """Get the current dataset size setting.

    Checks ContextVar first (works in same async context), then falls
    back to module-level global (works in LangGraph/LangChain threads).
    """
    cv_value = _dataset_size.get()
    # ContextVar default is "small", so we can't distinguish "not set" from "set to small"
    # Use the global as authoritative fallback
    return _dataset_size_global if _dataset_size_global != "small" else cv_value


def set_dataset_size(size: DatasetSize) -> None:
    """Set the dataset size both in ContextVar and global fallback."""
    global _dataset_size_global
    _dataset_size.set(size)
    _dataset_size_global = size


class dataset_size_context:
    """Context manager for temporarily setting dataset size.

    Usage:
        with dataset_size_context("large"):
            # Queries will use large endpoints
            ...
    """

    def __init__(self, size: DatasetSize):
        self.size = size
        self.token = None
        self._prev_global = None

    def __enter__(self):
        global _dataset_size_global
        self.token = _dataset_size.set(self.size)
        self._prev_global = _dataset_size_global
        _dataset_size_global = self.size
        return self

    def __exit__(self, *args):
        global _dataset_size_global
        _dataset_size.reset(self.token)
        _dataset_size_global = self._prev_global


# ============================================================
# Slot ID Context (for concurrent container isolation)
# ============================================================
# When USE_SLOT_CONTAINERS=True, each concurrent run gets its own
# exclusive set of OnTop containers. This prevents the Mapping Optimizer
# from affecting other concurrent queries when it modifies mappings
# and restarts containers.
#
# Usage:
#   with slot_context(0):
#       # All get_ontop_endpoint() calls will use slot 0 containers
#       result = await agent.run(...)

_slot_id: ContextVar[int | None] = ContextVar("slot_id", default=None)
_slot_id_global: int | None = None  # Thread-safe fallback


def get_slot_id() -> int | None:
    """Get the current slot ID from context.

    Checks ContextVar first, falls back to module-level global.
    """
    cv_value = _slot_id.get()
    if cv_value is not None:
        return cv_value
    return _slot_id_global


def set_slot_id(slot_id: int | None) -> None:
    """Set the slot ID both in ContextVar and global fallback."""
    global _slot_id_global
    _slot_id.set(slot_id)
    _slot_id_global = slot_id


class slot_context:
    """Context manager for temporarily setting slot ID.

    Usage:
        with slot_context(0):
            # Queries will use slot 0 containers
            ...
    """

    def __init__(self, slot_id: int):
        self.slot_id = slot_id
        self.token = None
        self._prev_global = None

    def __enter__(self):
        global _slot_id_global
        self.token = _slot_id.set(self.slot_id)
        self._prev_global = _slot_id_global
        _slot_id_global = self.slot_id
        return self

    def __exit__(self, *args):
        global _slot_id_global
        _slot_id.reset(self.token)
        _slot_id_global = self._prev_global


# ============================================================
# LLM Model Context
# ============================================================
# This allows sub-agents (like MappingOptimizerAgent) to use
# the same LLM model as the parent orchestrator.
#
# Uses both a ContextVar (for async task isolation) AND a module-level
# fallback (for threads). LangChain runs @tool functions in thread pools
# where ContextVars don't propagate, so the global fallback ensures
# sub-agents still get the correct model.
#
# Usage:
#   set_llm_model("claude-3-5-sonnet-20241022")
#   # All agents created will use this model

_llm_model: ContextVar[str] = ContextVar("llm_model", default="")
_llm_model_global: str = "deepseek-chat"  # Thread-safe fallback


def get_llm_model() -> str:
    """Get the current LLM model setting.

    Checks ContextVar first (works in same async context), then falls
    back to module-level global (works in LangChain tool threads).
    """
    cv_value = _llm_model.get()
    if cv_value:
        return cv_value
    return _llm_model_global


def set_llm_model(model: str) -> None:
    """Set the LLM model both in ContextVar and global fallback."""
    global _llm_model_global
    _llm_model.set(model)
    _llm_model_global = model


def get_openai_reasoning_kwargs(llm_model: str) -> dict:
    """Return reasoning kwargs for OpenAI models that support it (GPT-5+).

    Returns an empty dict for all other models, so it can be safely
    spread into any ChatOpenAI constructor via **get_openai_reasoning_kwargs(model).
    """
    if llm_model.startswith("gpt-5"):
        return {"reasoning": {"effort": "medium", "summary": "auto"}}
    return {}


def normalize_content(content) -> str:
    """Extract text from LLM response content.

    Reasoning models (GPT-5+) return content as a list of blocks
    (e.g. [{"type": "thinking", ...}, {"type": "text", ...}]) instead
    of a plain string.  This helper normalises both formats to a string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else ""
    return str(content) if content else ""


def get_size_for_query_set(query_set: str) -> DatasetSize:
    """Determine dataset size based on query set.

    Args:
        query_set: Query set name (BASE, SYN, TYPO, LARGE, UNDER, CROSS)

    Returns:
        "large" for LARGE queries, "small" for everything else
    """
    return "large" if query_set.upper() == "LARGE" else "small"


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = DATA_DIR / "cache"
EMBEDDINGS_CACHE_DIR = CACHE_DIR / "embeddings"

# Available datasets with semantic models
# NOTE: BGEE removed - not used in experimental corpus, and no slot container
# exists for it. Keeping it in would cause retrieval to return stale 8084 endpoints
# under slot-based container isolation (see _slot_aware_endpoint in retrieval_tools).
AVAILABLE_DATASETS = ["EDU", "TRN", "NRG", "BSBM", "LCA"]

# Default mapping directory (EDU class semantic models)
MAPPINGS_DIR = DATASETS_DIR / "EDU" / "class_semantic_models"


def get_dataset_path(dataset_name: str) -> Path:
    """Get the root path for a specific dataset."""
    return DATASETS_DIR / dataset_name


def get_semantic_models_path(dataset_name: str) -> Path:
    """Get the semantic models directory for a specific dataset."""
    return DATASETS_DIR / dataset_name / "class_semantic_models"


# Mappings directory (contains ontology.owl files per dataset)
MAPPINGS_DIR_ROOT = PROJECT_ROOT / "mappings"


def get_ontology_path(dataset_name: str) -> Path:
    """Get the ontology.owl path for a specific dataset.

    Returns the path to mappings/{dataset}/ontology.owl which contains
    rdfs:subClassOf class hierarchy information.
    """
    return MAPPINGS_DIR_ROOT / dataset_name.lower() / "ontology.owl"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM API Keys
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")

    # OnTop SPARQL Endpoint
    ontop_sparql_url: str = Field(
        default="http://127.0.0.1:8080/sparql",
        alias="ONTOP_SPARQL_URL",
    )

    # MySQL Database
    mysql_host: str = Field(default="localhost", alias="MYSQL_HOST")
    mysql_port: int = Field(default=3306, alias="MYSQL_PORT")
    mysql_user: str = Field(default="root", alias="MYSQL_USER")
    mysql_password: str = Field(default="root", alias="MYSQL_PASSWORD")
    mysql_database: str = Field(default="experiment", alias="MYSQL_DATABASE")

    # Embedding Settings
    openai_embedding_model: str = Field(
        default="text-embedding-3-large",
        alias="OPENAI_EMBEDDING_MODEL",
    )

    # Experiment Settings
    default_k_queries: int = Field(default=3, alias="DEFAULT_K_QUERIES")
    max_iterations: int = Field(default=3, alias="MAX_ITERATIONS")

    # Tracing Settings
    tracing_enabled: bool = Field(default=True, alias="TRACING_ENABLED")
    tracing_verbosity: int = Field(default=2, alias="TRACING_VERBOSITY")


# LLM Provider configurations
LLM_PROVIDERS = {
    "claude": {
        "models": [
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-5-20251101",
        ],
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "models": [
            "gpt-4o",
            "gpt-4-turbo",
        ],
        "api_key_env": "OPENAI_API_KEY",
    },
    "deepseek": {
        "models": [
            "deepseek-chat",
            "deepseek-reasoner",
        ],
        "api_key_env": "DEEPSEEK_API_KEY",
    },
}

# Approach types for the experiment
ApproachType = Literal["baseline", "agentic_grep", "agentic_semantic", "text2sql_chess"]

APPROACHES: list[ApproachType] = ["baseline", "agentic_grep", "agentic_semantic"]

# Text2SQL approach (runs separately via scripts/run_chess_experiment.py)
TEXT2SQL_APPROACHES: list[ApproachType] = ["text2sql_chess"]


# ============================================================
# OnTop Replica Configuration for Load Balancing
# ============================================================
# When using docker-compose.scale.yml, each dataset has 3 replicas.
# Set USE_ONTOP_REPLICAS=True to enable load balancing.

USE_ONTOP_REPLICAS: bool = True  # Enable load balancing with replicas

# Replica endpoints per dataset (used when USE_ONTOP_REPLICAS=True)
# Each dataset has 2 instances (original + 1 replica) for load balancing
# Note: BGEE not included as it's not used in experimental corpus
ONTOP_REPLICAS: dict[str, list[str]] = {
    "edu": [
        "http://127.0.0.1:8080/sparql",
        "http://127.0.0.1:8180/sparql",
    ],
    "trn": [
        "http://127.0.0.1:8081/sparql",
        "http://127.0.0.1:8181/sparql",
    ],
    "nrg": [
        "http://127.0.0.1:8082/sparql",
        "http://127.0.0.1:8182/sparql",
    ],
    "bsbm": [
        "http://127.0.0.1:8083/sparql",
        "http://127.0.0.1:8183/sparql",
    ],
    "lca": [
        "http://127.0.0.1:8085/sparql",
        "http://127.0.0.1:8185/sparql",
    ],
}

# Single instance endpoints (used when USE_ONTOP_REPLICAS=False)
ONTOP_SINGLE_ENDPOINTS: dict[str, str] = {
    "edu": "http://127.0.0.1:8080/sparql",
    "trn": "http://127.0.0.1:8081/sparql",
    "nrg": "http://127.0.0.1:8082/sparql",
    "bsbm": "http://127.0.0.1:8083/sparql",
    "bgee": "http://127.0.0.1:8084/sparql",
    "lca": "http://127.0.0.1:8085/sparql",
    # Large dataset endpoints
    "edu-large": "http://127.0.0.1:8090/sparql",
    "trn-large": "http://127.0.0.1:8091/sparql",
    "nrg-large": "http://127.0.0.1:8092/sparql",
}


# ============================================================
# Slot-Based Container Endpoints (for concurrent isolation)
# ============================================================
# When USE_SLOT_CONTAINERS=True and a slot_id is set in context,
# queries are routed to slot-specific containers that have their own
# mapping files. This allows the Mapping Optimizer to modify mappings
# without affecting other concurrent queries.
#
# Port allocation per slot:
#   Slot 0: 8100-8109 (small), 8190-8192 (large)
#   Slot 1: 8200-8209 (small), 8290-8292 (large)
#   Slot 2: 8300-8309 (small), 8390-8392 (large)
#   Slot 3: 8400-8409 (small)
#   Slot 4: 8500-8509 (small)
USE_SLOT_CONTAINERS: bool = False  # Set to True when using docker-compose.slots.yml
MAX_SLOTS: int = 5

# Slot-based endpoint configuration: SLOT_ENDPOINTS[slot_id][dataset] = endpoint_url
SLOT_ENDPOINTS: dict[int, dict[str, str]] = {
    0: {
        "edu-small": "http://127.0.0.1:8100/sparql",
        "trn-small": "http://127.0.0.1:8101/sparql",
        "nrg-small": "http://127.0.0.1:8102/sparql",
        "bsbm": "http://127.0.0.1:8103/sparql",
        "bgee": "http://127.0.0.1:8104/sparql",
        "lca": "http://127.0.0.1:8105/sparql",
        "edu-large": "http://127.0.0.1:8190/sparql",
        "trn-large": "http://127.0.0.1:8191/sparql",
        "nrg-large": "http://127.0.0.1:8192/sparql",
    },
    1: {
        "edu-small": "http://127.0.0.1:8200/sparql",
        "trn-small": "http://127.0.0.1:8201/sparql",
        "nrg-small": "http://127.0.0.1:8202/sparql",
        "bsbm": "http://127.0.0.1:8203/sparql",
        "bgee": "http://127.0.0.1:8204/sparql",
        "lca": "http://127.0.0.1:8205/sparql",
        "edu-large": "http://127.0.0.1:8290/sparql",
        "trn-large": "http://127.0.0.1:8291/sparql",
        "nrg-large": "http://127.0.0.1:8292/sparql",
    },
    2: {
        "edu-small": "http://127.0.0.1:8300/sparql",
        "trn-small": "http://127.0.0.1:8301/sparql",
        "nrg-small": "http://127.0.0.1:8302/sparql",
        "bsbm": "http://127.0.0.1:8303/sparql",
        "bgee": "http://127.0.0.1:8304/sparql",
        "lca": "http://127.0.0.1:8305/sparql",
        "edu-large": "http://127.0.0.1:8390/sparql",
        "trn-large": "http://127.0.0.1:8391/sparql",
        "nrg-large": "http://127.0.0.1:8392/sparql",
    },
    3: {
        "edu-small": "http://127.0.0.1:8400/sparql",
        "trn-small": "http://127.0.0.1:8401/sparql",
        "nrg-small": "http://127.0.0.1:8402/sparql",
        "bsbm": "http://127.0.0.1:8403/sparql",
        "bgee": "http://127.0.0.1:8404/sparql",
        "lca": "http://127.0.0.1:8405/sparql",
    },
    4: {
        "edu-small": "http://127.0.0.1:8500/sparql",
        "trn-small": "http://127.0.0.1:8501/sparql",
        "nrg-small": "http://127.0.0.1:8502/sparql",
        "bsbm": "http://127.0.0.1:8503/sparql",
        "bgee": "http://127.0.0.1:8504/sparql",
        "lca": "http://127.0.0.1:8505/sparql",
    },
}


# ============================================================
# Experiment Endpoints (Small vs Large Datasets)
# ============================================================
# Used with docker-compose.experiment.yml for Multi-Step Query testing.
#
# Small datasets: For correctness testing (fast queries)
# Large datasets: For Multi-Step Query optimization testing (slow single queries)

USE_EXPERIMENT_ENDPOINTS: bool = False  # Set to True when using docker-compose.experiment.yml

# Experiment endpoint configuration
EXPERIMENT_ENDPOINTS: dict[str, dict[str, str]] = {
    # Small datasets (for correctness testing)
    "edu_small": "http://127.0.0.1:8080/sparql",   # ~100k triples, 1 university
    "trn_small": "http://127.0.0.1:8081/sparql",   # ~2k StopTimes, Metro only
    "nrg_small": "http://127.0.0.1:8082/sparql",    # Original NRG data

    # Large datasets (for Multi-Step Query testing)
    "edu_large": "http://127.0.0.1:8090/sparql",   # ~1M+ triples, 10+ universities
    "trn_large": "http://127.0.0.1:8091/sparql",   # ~3.5M StopTimes, all data
    "nrg_large": "http://127.0.0.1:8092/sparql",    # VIG-scaled NRG

    # Unchanged datasets (same for small and large)
    "bsbm": "http://127.0.0.1:8083/sparql",
    "lca": "http://127.0.0.1:8085/sparql",
}

# Mapping from base dataset to small/large variants
EXPERIMENT_DATASET_MAPPING: dict[str, dict[str, str]] = {
    "edu": {"small": "edu_small", "large": "edu_large"},
    "trn": {"small": "trn_small", "large": "trn_large"},
    "nrg": {"small": "nrg_small", "large": "nrg_large"},
    "bsbm": {"small": "bsbm", "large": "bsbm"},  # No large variant
    "lca": {"small": "lca", "large": "lca"},      # No large variant
}


def get_experiment_endpoint(dataset_id: str, size: Literal["small", "large"] = "small") -> str:
    """
    Get the experiment endpoint URL for a dataset with specific size.

    Args:
        dataset_id: Base dataset identifier (e.g., "edu", "trn")
        size: Either "small" or "large"

    Returns:
        Endpoint URL for the specified dataset size.
    """
    dataset_key = dataset_id.lower()

    if dataset_key in EXPERIMENT_DATASET_MAPPING:
        variant = EXPERIMENT_DATASET_MAPPING[dataset_key].get(size, "small")
        return EXPERIMENT_ENDPOINTS.get(variant, EXPERIMENT_ENDPOINTS.get(f"{dataset_key}_small"))

    # Fallback to unchanged datasets
    return EXPERIMENT_ENDPOINTS.get(dataset_key, "http://127.0.0.1:8080/sparql")


def get_ontop_endpoint(dataset_id: str, size: DatasetSize | None = None) -> str:
    """
    Get the OnTop endpoint URL for a dataset.

    Automatically selects endpoint based on:
    1. Slot context (if USE_SLOT_CONTAINERS=True and slot_id is set)
    2. Explicit size parameter (if provided)
    3. Dataset ID containing "-large" or "_large" suffix
    4. Current dataset_size_context (set via context manager or set_dataset_size())
    5. Falls back to "small" if nothing is set

    Args:
        dataset_id: Dataset identifier (e.g., "edu", "trn", "edu-large")
        size: Override size selection ("small" or "large"). If None, uses context.

    Returns:
        Endpoint URL to use for queries.
    """
    dataset_lower = dataset_id.lower()

    # Infer size from dataset_id if it contains "-large" or "_large"
    has_large_suffix = "-large" in dataset_lower or "_large" in dataset_lower

    dataset_key = dataset_lower.replace("_large", "").replace("-large", "").replace("-small", "").replace("_small", "")

    # Determine size: explicit > inferred from name > context > default
    if size is not None:
        effective_size = size
    elif has_large_suffix:
        effective_size = "large"
    else:
        effective_size = get_dataset_size()

    # Check for slot-based routing first (highest priority when enabled)
    slot_id = get_slot_id()
    if USE_SLOT_CONTAINERS and slot_id is not None and slot_id in SLOT_ENDPOINTS:
        slot_endpoints = SLOT_ENDPOINTS[slot_id]

        # Build full dataset key with size suffix
        if effective_size == "large" and dataset_key in ("edu", "trn", "nrg"):
            full_key = f"{dataset_key}-large"
        elif dataset_key in ("edu", "trn", "nrg"):
            full_key = f"{dataset_key}-small"
        else:
            full_key = dataset_key

        if full_key in slot_endpoints:
            return slot_endpoints[full_key]

    # For large datasets, use the dedicated large endpoint (no replicas for large)
    if effective_size == "large" and dataset_key in ("edu", "trn", "nrg"):
        large_key = f"{dataset_key}-large"
        return ONTOP_SINGLE_ENDPOINTS.get(large_key, ONTOP_SINGLE_ENDPOINTS.get(dataset_key))

    # For small datasets, use replica pool if available
    if USE_ONTOP_REPLICAS:
        from src.endpoints.ontop import get_endpoint_pool

        pool = get_endpoint_pool(dataset_key)
        if pool:
            return pool.get_next()

    return ONTOP_SINGLE_ENDPOINTS.get(dataset_key, "http://127.0.0.1:8080/sparql")


def get_dataset_from_endpoint(endpoint_url: str) -> str | None:
    """
    Inverse lookup: get dataset name from endpoint URL.

    This is useful when you have an endpoint URL and need to determine
    which dataset (including size variant) it belongs to.

    Args:
        endpoint_url: The SPARQL endpoint URL (e.g., "http://127.0.0.1:8091/sparql")

    Returns:
        Dataset name (e.g., "trn-large") or None if not found.
    """
    # Normalize URL for comparison (remove trailing slashes)
    normalized = endpoint_url.rstrip("/")

    # Check ONTOP_SINGLE_ENDPOINTS first (most common case)
    for dataset, endpoint in ONTOP_SINGLE_ENDPOINTS.items():
        if endpoint.rstrip("/") == normalized:
            return dataset

    # Also check EXPERIMENT_ENDPOINTS
    for dataset, endpoint in EXPERIMENT_ENDPOINTS.items():
        if endpoint.rstrip("/") == normalized:
            # Convert underscore to hyphen for consistency (e.g., trn_large -> trn-large)
            return dataset.replace("_", "-")

    return None


def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()

# ---------------------------------------------------------------------------
# OpenRouter provider pinning.
# OpenRouter routes each request to a different host with different
# quantization unless told otherwise (observed: Venice/AtlasCloud/DeepInfra fp4
# for DeepSeek-V3.2). DeepSeek-V3.2 is released in FP8, so fp8 hosts are
# faithful; Qwen3.5-27B is pinned to the official Alibaba endpoint (bf16 Novita
# as fallback). Override with OPENROUTER_PROVIDER_JSON='{"<model>": {...}}'.
# ---------------------------------------------------------------------------
import json as _json
import os as _os

_OPENROUTER_PROVIDER_DEFAULTS: dict[str, dict] = {
    "deepseek/deepseek-v3.2": {"order": ["Novita", "AtlasCloud", "GMICloud"],
                               "allow_fallbacks": False, "quantizations": ["fp8"]},
    "qwen/qwen3.5-27b": {"order": ["Alibaba", "Novita"], "allow_fallbacks": False},
}


def get_openrouter_kwargs(llm_model: str) -> dict:
    """Extra ChatOpenAI kwargs for an ``openrouter/<model>`` id (provider pinning)."""
    model = llm_model[len("openrouter/"):] if llm_model.startswith("openrouter/") else llm_model
    prefs = dict(_OPENROUTER_PROVIDER_DEFAULTS)
    override = _os.getenv("OPENROUTER_PROVIDER_JSON", "")
    if override:
        try:
            prefs.update(_json.loads(override))
        except Exception:
            pass
    if model in prefs and prefs[model]:
        return {"extra_body": {"provider": prefs[model]}}
    return {}


# ---------------------------------------------------------------------------
# UNDER "instructed" configuration (RQ3).
# When UNDER_INSTRUCTED=1 the retrieval agents and the SPARQL agent get an explicit
# UNDERSPECIFIED output option appended to their system prompts. Default: off, so
# every other experiment keeps byte-identical prompts.
# ---------------------------------------------------------------------------
UNDER_INSTRUCTED: bool = _os.getenv("UNDER_INSTRUCTED", "0") == "1"

UNDERSPECIFIED_INSTRUCTION_RETRIEVAL = """

## Underspecified questions
If the question cannot be answered with any query without first asking the user for missing
information, do not invent that information. Output exactly:
`UNDERSPECIFIED: <what you would have to ask the user, in one sentence>`"""

UNDERSPECIFIED_FORCE_OPTION_RETRIEVAL = """

OPTION 3 - If the question cannot be answered with any query without first asking the user for missing information:
Do not invent that information. Output exactly: UNDERSPECIFIED: [what you would have to ask the user, in one sentence]"""

UNDERSPECIFIED_INSTRUCTION_SPARQL = """

## Underspecified questions
If the question cannot be answered with any query without first asking the user for missing
information, do not invent that information. Output exactly:
```json
{
  "status": "UNDERSPECIFIED",
  "missing_criterion": "<what you would have to ask the user, in one sentence>",
  "reasoning": "Why no query is possible without this information"
}
```"""
