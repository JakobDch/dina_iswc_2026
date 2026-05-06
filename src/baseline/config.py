"""
Configuration for baseline pipeline.

OnTop-only version - no GraphDB support needed.
"""

from pathlib import Path
import logging
import os
from dotenv import load_dotenv
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache

logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

# Paths and directories
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent.parent / "data" / "storage"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEMANTIC_MODELS_DIR_NAME = "class_semantic_models"
EMBEDDINGS_DIR_NAME = "embeddings"
FAISS_INDEX_DIRECTORY_NAME = "vector_store_semantic_models"

# Central settings and environment variables
ONTOP_SPARQL_URL = os.getenv("ONTOP_SPARQL_URL", "http://127.0.0.1:8080/sparql")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")

# MySQL Configuration for OnTop
MYSQL_HOST = os.getenv("MYSQL_HOST", "mysql")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "root")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "experiment")

logger.info(f"Storage directory set to: {DATA_DIR.resolve()}")
logger.info(f"Semantic models directory name set to: '{SEMANTIC_MODELS_DIR_NAME}'")

LLM_PROFILES = {
    "deepseek_chat": {
        "provider": "openai",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "requires_api_key": True,
        "api_key_env": "DEEPSEEK_API_KEY",
        "context_window": 64000,
        "retrieval_token_budget": 25000,
    },
    "deepseek_reasoner": {
        "provider": "openai",
        "model": "deepseek-reasoner",
        "base_url": "https://api.deepseek.com/v1",
        "requires_api_key": True,
        "api_key_env": "DEEPSEEK_API_KEY",
        "context_window": 64000,
        "retrieval_token_budget": 20000,
    },
    "openai_gpt4o": {
        "provider": "openai",
        "model": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "requires_api_key": True,
        "api_key_env": "OPENAI_API_KEY",
        "context_window": 128000,
        "retrieval_token_budget": 50000,
    },
    "ollama_remote_llama33_70b": {
        "provider": "ollama",
        "model": "llama3.3:70b",
        "base_url": OLLAMA_BASE_URL,
        "requires_api_key": False,
        "context_window": 128000,
        "retrieval_token_budget": 50000,
    },
    "ollama_remote_llama31_70b": {
        "provider": "ollama",
        "model": "llama3.1:70b",
        "base_url": OLLAMA_BASE_URL,
        "requires_api_key": False,
        "context_window": 128000,
        "retrieval_token_budget": 50000,
    },
    "ollama_remote_qwen35_27b": {
        "provider": "ollama",
        "model": "qwen3.5:27b",
        "base_url": OLLAMA_BASE_URL,
        "requires_api_key": False,
        "context_window": 32000,
        "retrieval_token_budget": 15000,
    },
    "ollama_remote_deepseek_r1_32b": {
        "provider": "ollama",
        "model": "deepseek-r1:32b",
        "base_url": OLLAMA_BASE_URL,
        "requires_api_key": False,
        "context_window": 64000,
        "retrieval_token_budget": 25000,
    },
    "deepinfra_gemma3": {
        "provider": "openai",
        "model": "google/gemma-3-12b-it",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "requires_api_key": True,
        "api_key_env": "DEEPINFRA_API_KEY",
        "context_window": 128000,
        "retrieval_token_budget": 50000,
    },
    # Claude models for this experiment
    "anthropic_claude_sonnet": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5-20250929",
        "requires_api_key": True,
        "api_key_env": "ANTHROPIC_API_KEY",
        "context_window": 200000,
        "retrieval_token_budget": 80000,
    },
    "anthropic_claude_opus": {
        "provider": "anthropic",
        "model": "claude-opus-4-5-20251101",
        "requires_api_key": True,
        "api_key_env": "ANTHROPIC_API_KEY",
        "context_window": 200000,
        "retrieval_token_budget": 80000,
    },
}

# Default retrieval token budget when LLM profile doesn't specify one
DEFAULT_RETRIEVAL_TOKEN_BUDGET = 30000


class AppSettings(BaseSettings):
    llm_timeout_seconds: int = 600  # Default timeout for LLM API calls (10 minutes)
    ontop_sparql_url: str = ONTOP_SPARQL_URL
    ontop_timeout_seconds: int = 15  # Default timeout for OnTop operations
    # LLM API keys
    ollama_base_url: Optional[str] = OLLAMA_BASE_URL
    deepseek_api_key: Optional[str] = DEEPSEEK_API_KEY
    openai_api_key: Optional[str] = OPENAI_API_KEY
    anthropic_api_key: Optional[str] = ANTHROPIC_API_KEY
    ollama_base_url: str = OLLAMA_BASE_URL
    fireworks_api_key: Optional[str] = FIREWORKS_API_KEY
    deepinfra_api_key: Optional[str] = DEEPINFRA_API_KEY
    # Paths
    data_dir: Path = DATA_DIR
    app_dir: Path = APP_DIR
    semantic_models_dir_name: str = SEMANTIC_MODELS_DIR_NAME
    embeddings_dir_name: str = EMBEDDINGS_DIR_NAME
    faiss_index_directory_name: str = FAISS_INDEX_DIRECTORY_NAME
    # SPARQL Edit Evaluation LLM Profile
    sparql_edit_evaluation_llm_profile: str = "deepseek_chat"
    # Iterative Query Correction Settings
    correction_max_iterations: int = 5  # Max iterations for iterative query correction
    correction_timeout_per_iteration: int = 15  # Timeout in seconds per iteration
    # MySQL Configuration for OnTop
    mysql_host: str = MYSQL_HOST
    mysql_port: int = MYSQL_PORT
    mysql_user: str = MYSQL_USER
    mysql_password: str = MYSQL_PASSWORD
    mysql_database: str = MYSQL_DATABASE

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'ignore'


@lru_cache()
def get_settings() -> AppSettings:
    return AppSettings()
