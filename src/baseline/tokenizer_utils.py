"""
Precise token counting utilities for scientific evaluation.
Uses model-specific tokenizers for accurate token measurements.
"""

import logging
from typing import Dict, Optional, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)

# Global tokenizer cache to avoid reloading
_TOKENIZER_CACHE: Dict[str, object] = {}

def get_model_tokenizer_name(model_name: str) -> str:
    """Map LLM model names to their corresponding HuggingFace tokenizer names."""
    model_mapping = {
        # Llama models - use Llama-3.1-8B tokenizer (requires HF token and acceptance)
        "llama3.3:70b": "meta-llama/Llama-3.1-8B",
        "llama3.1:70b": "meta-llama/Llama-3.1-8B",
        "llama3.1:8b": "meta-llama/Llama-3.1-8B",
        "llama3:70b": "meta-llama/Llama-3.1-8B",
        "llama3:8b": "meta-llama/Llama-3.1-8B",
        "llama4:16x17b": "meta-llama/Llama-3.1-8B",  # Llama 4 uses same tokenizer as Llama 3

        # OpenAI models - use tiktoken equivalent
        "gpt-4o": "gpt-4o",
        "gpt-4": "gpt-4",
        "gpt-3.5-turbo": "gpt-3.5-turbo",

        # DeepSeek models
        "deepseek-chat": "deepseek-ai/deepseek-llm-7b-chat",
        "deepseek-coder": "deepseek-ai/deepseek-coder-7b-instruct",

        # Gemma models
        "gemma2:27b": "google/gemma-2-27b-it",
        "gemma2:9b": "google/gemma-2-9b-it",
    }

    return model_mapping.get(model_name, model_name)

def load_tokenizer(model_name: str):
    """Load and cache tokenizer for a given model."""
    # Check cache but only return non-None values
    if model_name in _TOKENIZER_CACHE and _TOKENIZER_CACHE[model_name] is not None:
        return _TOKENIZER_CACHE[model_name]

    try:
        tokenizer_name = get_model_tokenizer_name(model_name)

        # Handle OpenAI models with tiktoken
        if model_name.startswith("gpt-"):
            try:
                import tiktoken
                tokenizer = tiktoken.encoding_for_model(model_name)
                _TOKENIZER_CACHE[model_name] = ("tiktoken", tokenizer)
                logger.info(f"Loaded tiktoken tokenizer for {model_name}")
                return ("tiktoken", tokenizer)
            except ImportError:
                logger.warning("tiktoken not available, falling back to transformers")
            except Exception as e:
                logger.warning(f"Failed to load tiktoken for {model_name}: {e}")

        # Use transformers for all other models
        try:
            from transformers import AutoTokenizer
            import os

            # Get HuggingFace token from environment
            hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                cache_dir="./tokenizer_cache",
                local_files_only=False,
                trust_remote_code=True,
                token=hf_token
            )
            _TOKENIZER_CACHE[model_name] = ("transformers", tokenizer)
            logger.info(f"Loaded transformers tokenizer for {model_name} -> {tokenizer_name}")
            return ("transformers", tokenizer)

        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model_name} -> {tokenizer_name}: {e}")
            return None

    except Exception as e:
        logger.error(f"Error loading tokenizer for {model_name}: {e}")
        return None

def count_tokens_precise(text: str, model_name: str) -> int:
    """Count tokens precisely using the model's tokenizer."""
    if not text:
        return 0

    tokenizer_info = load_tokenizer(model_name)
    if not tokenizer_info:
        # Fallback to character-based estimation
        logger.warning(f"No tokenizer available for {model_name}, using character estimation")
        return max(1, len(text) // 4)

    tokenizer_type, tokenizer = tokenizer_info

    try:
        if tokenizer_type == "tiktoken":
            return len(tokenizer.encode(text))
        elif tokenizer_type == "transformers":
            # Use encode method directly for transformers
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        else:
            logger.warning(f"Unknown tokenizer type: {tokenizer_type}")
            return max(1, len(text) // 4)

    except Exception as e:
        logger.error(f"Error counting tokens with {tokenizer_type} for {model_name}: {e}")
        return max(1, len(text) // 4)

def count_tokens_for_messages(messages: list, model_name: str) -> int:
    """Count tokens for a list of messages (system + user)."""
    total_tokens = 0

    for message in messages:
        content = message.get("content", "")
        role = message.get("role", "")

        # Count content tokens
        content_tokens = count_tokens_precise(content, model_name)

        # Add overhead for message formatting (approximate)
        # Most chat models add tokens for role markers, etc.
        overhead_tokens = 4  # Rough estimate for role formatting

        total_tokens += content_tokens + overhead_tokens

    # Add overhead for the overall chat format
    total_tokens += 3  # Approximate overhead for chat completion

    return total_tokens

def get_token_usage_precise(
    prompt_text: str,
    system_message: str,
    response_content: str,
    model_name: str
) -> Dict[str, int]:
    """
    Get precise token usage for a complete LLM interaction.

    Args:
        prompt_text: User's prompt
        system_message: System message
        response_content: LLM's response
        model_name: Name of the model used

    Returns:
        Dictionary with input_tokens, output_tokens, total_tokens
    """
    # Count input tokens (system + user messages)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt_text}
    ]
    input_tokens = count_tokens_for_messages(messages, model_name)

    # Count output tokens
    output_tokens = count_tokens_precise(response_content, model_name)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "method": "precise_tokenizer",
        "model_used": model_name
    }

def preload_tokenizers_for_models(model_names: list):
    """Preload tokenizers for multiple models to avoid delays during benchmarking."""
    logger.info(f"Preloading tokenizers for models: {model_names}")

    for model_name in model_names:
        try:
            load_tokenizer(model_name)
            logger.info(f"Successfully preloaded tokenizer for {model_name}")
        except Exception as e:
            logger.error(f"Failed to preload tokenizer for {model_name}: {e}")
