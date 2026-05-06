import os
import requests
import asyncio
from openai import OpenAI as OpenAIClient
import logging
from pydantic import BaseModel, Field
from fastapi import HTTPException, status
import json
from typing import Optional, Dict, Any
from .tokenizer_utils import get_token_usage_precise

logger = logging.getLogger(__name__)

class LLMResponse(BaseModel):
    """Structured response from LLM including content and token usage."""
    content: str
    input_tokens: int
    output_tokens: int
    total_tokens: int

class OpenAILLM:
    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1", timeout: float = 600.0):
        if not api_key:
            raise ValueError("OpenAI API Key is required.")
        self.client = OpenAIClient(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        if str(self.model).startswith("gpt-5"):
            self._temperature_floor = 1.0
            self._temperature_fixed = 1.0
        else:
            self._temperature_floor = 0.0
            self._temperature_fixed = None

    def _normalize_temperature(self, temperature: Optional[float]) -> float:
        """Apply model-specific constraints before sending to OpenAI."""
        if self._temperature_fixed is not None:
            if temperature is None or temperature != self._temperature_fixed:
                logger.info(
                    "Overriding requested temperature %s with fixed value %.2f for model %s",
                    temperature,
                    self._temperature_fixed,
                    self.model
                )
            return self._temperature_fixed

        normalized = self._temperature_floor if temperature is None else temperature
        if normalized < self._temperature_floor:
            logger.debug(f"Adjusted temperature from {normalized} to {self._temperature_floor} for model {self.model}")
            normalized = self._temperature_floor
        return normalized

    def invoke(self, prompt_text: str, system_message: str = "You are a helpful assistant.", temperature: float = 0.0):
        effective_temperature = self._normalize_temperature(temperature)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt_text}
                ],
                stream=False,
                temperature=effective_temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {type(e).__name__} - {str(e)[:100]}")
            return f"Error: OpenAI request failed ({self.model}): {type(e).__name__}"

    def invoke_with_usage(self, prompt_text: str, system_message: str = "You are a helpful assistant.", temperature: float = 0.0) -> LLMResponse:
        """Invoke LLM and return response with token usage information."""
        effective_temperature = self._normalize_temperature(temperature)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt_text}
                ],
                stream=False,
                temperature=effective_temperature
            )

            content = response.choices[0].message.content.strip()
            usage = response.usage

            return LLMResponse(
                content=content,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens
            )
        except Exception as e:
            print(f"OpenAI API error: {type(e).__name__} - {str(e)[:100]}")
            error_content = f"Error: OpenAI request failed ({self.model}): {type(e).__name__}"
            return LLMResponse(
                content=error_content,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0
            )



class OllamaLLM:
    def __init__(self, model: str, base_url: str, timeout: float = 600.0):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def invoke(self, prompt_text: str, system_message: str = "You are a helpful assistant.", temperature: float = 0.0):
        timeout_seconds = self.timeout
        try:
            api_endpoint = self.base_url
            if not api_endpoint.endswith('/v1/chat/completions'):
                if api_endpoint.endswith('/'):
                    api_endpoint += 'v1/chat/completions'
                else:
                    api_endpoint += '/v1/chat/completions'

            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt_text}
                ],
                "stream": False,
                "temperature": temperature,
                "options": {
                    "num_ctx": 4096
                }
            }

            def sync_post():
                return requests.post(api_endpoint, headers=headers, json=data, timeout=timeout_seconds)

            try:
                loop = asyncio.get_running_loop()
                response = loop.run_until_complete(loop.run_in_executor(None, sync_post)) if loop.is_running() else sync_post()
            except RuntimeError:
                response = sync_post()

            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.Timeout:
            return f"Error: Timeout during Ollama request ({self.model}) after {timeout_seconds} seconds."
        except requests.exceptions.RequestException as e:
            return f"Error: Ollama request ({self.model}) failed: {e}"
        except KeyError:
            return f"Error: Unexpected response structure from Ollama ({self.model}). Response: {response.text}"

    def invoke_with_usage(self, prompt_text: str, system_message: str = "You are a helpful assistant.", temperature: float = 0.0) -> LLMResponse:
        """Invoke LLM and return response with token usage information."""
        timeout_seconds = self.timeout
        try:
            api_endpoint = self.base_url
            if not api_endpoint.endswith('/v1/chat/completions'):
                if api_endpoint.endswith('/'):
                    api_endpoint += 'v1/chat/completions'
                else:
                    api_endpoint += '/v1/chat/completions'

            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt_text}
                ],
                "stream": False,
                "temperature": temperature,
                "options": {
                    "num_ctx": 4096
                }
            }

            def sync_post():
                return requests.post(api_endpoint, headers=headers, json=data, timeout=timeout_seconds)

            try:
                loop = asyncio.get_running_loop()
                response = loop.run_until_complete(loop.run_in_executor(None, sync_post)) if loop.is_running() else sync_post()
            except RuntimeError:
                response = sync_post()

            response.raise_for_status()
            response_data = response.json()

            content = response_data["choices"][0]["message"]["content"].strip()

            try:
                precise_usage = get_token_usage_precise(
                    prompt_text=prompt_text,
                    system_message=system_message,
                    response_content=content,
                    model_name=self.model
                )
                input_tokens = precise_usage["input_tokens"]
                output_tokens = precise_usage["output_tokens"]
                total_tokens = precise_usage["total_tokens"]
                logger.debug(f"Using precise tokenization for {self.model}: {input_tokens}+{output_tokens}={total_tokens}")
            except Exception as e:
                logger.warning(f"Precise tokenization failed for {self.model}, falling back to API/estimation: {e}")

                # Fallback: Try Ollama's usage data
                usage = response_data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

                # If still no tokens, use character estimation
                if input_tokens == 0 and output_tokens == 0:
                    input_tokens = max(1, len(prompt_text) // 4)
                    output_tokens = max(1, len(content) // 4)

                total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens
            )
        except requests.exceptions.Timeout:
            error_content = f"Error: Timeout during Ollama request ({self.model}) after {timeout_seconds} seconds."
            return LLMResponse(content=error_content, input_tokens=0, output_tokens=0, total_tokens=0)
        except requests.exceptions.RequestException as e:
            error_content = f"Error: Ollama request ({self.model}) failed: {e}"
            return LLMResponse(content=error_content, input_tokens=0, output_tokens=0, total_tokens=0)
        except KeyError:
            error_content = f"Error: Unexpected response structure from Ollama ({self.model}). Response: {response.text}"
            return LLMResponse(content=error_content, input_tokens=0, output_tokens=0, total_tokens=0)


class DeepSeekLLM:
    def __init__(self, api_key: str, model: str, base_url: str = "https://api.deepseek.com/v1", timeout: float = 600.0):
        if not api_key:
            raise ValueError("DeepSeek API Key is required.")
        self.client = OpenAIClient(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model

    def invoke(self, prompt_text: str, system_message: str = "You are a helpful assistant.", temperature: float = 0.0):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt_text}
                ],
                stream=False,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"DeepSeek API error: {type(e).__name__} - {str(e)[:100]}")
            return f"Error: DeepSeek request failed ({self.model}): {type(e).__name__}"

    def invoke_with_usage(self, prompt_text: str, system_message: str = "You are a helpful assistant.", temperature: float = 0.0) -> LLMResponse:
        """Invoke LLM and return response with token usage information."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt_text}
                ],
                stream=False,
                temperature=temperature
            )

            content = response.choices[0].message.content.strip()
            usage = response.usage

            return LLMResponse(
                content=content,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens
            )
        except Exception as e:
            print(f"DeepSeek API error: {type(e).__name__} - {str(e)[:100]}")
            error_content = f"Error: DeepSeek request failed ({self.model}): {type(e).__name__}"
            return LLMResponse(
                content=error_content,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0
            )





class ChatMessageRequest(BaseModel):
    message: str
    llm_profile: str = Field(default="ollama_local_gemma3", description="Identifier for the LLM configuration to use.")



class ChatMessageResponse(BaseModel):
    reply: str


# Type alias for LLM instances
LLMInstance = OpenAILLM | OllamaLLM | DeepSeekLLM


def create_llm_instance(
    profile_key: str,
    llm_profiles: Dict[str, Any],
    settings: Any,  # AppSettings from config.py
) -> LLMInstance:
    """
    Factory function to create an LLM instance based on a profile key.

    Args:
        profile_key: The key identifying the LLM profile (e.g., "deepseek_chat", "ollama_local_gemma3")
        llm_profiles: Dictionary of LLM profile configurations (LLM_PROFILES from config.py)
        settings: AppSettings instance containing API keys and configuration

    Returns:
        An LLM instance (OpenAILLM, OllamaLLM, or DeepSeekLLM)

    Raises:
        HTTPException: If profile is not found, API key is missing, or provider is unsupported
    """
    if profile_key not in llm_profiles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown LLM profile: {profile_key}"
        )

    profile = llm_profiles[profile_key]
    provider = profile.get("provider")
    model = profile.get("model")
    base_url = profile.get("base_url")
    timeout = float(settings.llm_timeout_seconds)

    if provider == "ollama":
        return OllamaLLM(
            model=model,
            base_url=base_url or settings.ollama_base_url,
            timeout=timeout
        )

    elif provider in ["openai", "deepseek", "deepinfra"]:
        api_key_to_use = None
        api_key_env_var = profile.get("api_key_env")

        if api_key_env_var:
            # Map environment variable names to settings attributes
            api_key_mapping = {
                "DEEPSEEK_API_KEY": settings.deepseek_api_key,
                "OPENAI_API_KEY": settings.openai_api_key,
                "FIREWORKS_API_KEY": settings.fireworks_api_key,
                "DEEPINFRA_API_KEY": settings.deepinfra_api_key
            }
            api_key_to_use = api_key_mapping.get(api_key_env_var)

            if not api_key_to_use or api_key_to_use.strip() == "":
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"API key for '{api_key_env_var}' is not configured."
                )
        else:
            # Fallback to standard OpenAI key if no specific key is defined
            api_key_to_use = settings.openai_api_key

        if not api_key_to_use:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"API key for profile '{profile_key}' is required but could not be found."
            )

        return OpenAILLM(
            api_key=api_key_to_use,
            model=model,
            base_url=base_url,
            timeout=timeout
        )

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported LLM provider: {provider}"
        )
