"""
Centralized Configuration for The Memory Twin
==============================================

Supports multiple LLM providers:
- google: Google Gemini
- openrouter: OpenRouter (access to multiple models)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables (single call for whole project)
load_dotenv()


class Settings(BaseSettings):
    """System configuration loaded from environment variables.

    pydantic-settings auto-loads from .env, no need for os.getenv().
    """

    # API Keys
    google_api_key: str = ""
    openrouter_api_key: str = ""

    # Database
    chroma_persist_dir: str = Field(default="./data/chroma")
    sqlite_db_path: str = Field(default="./data/memory.db")

    # MCP Server
    mcp_server_host: str = Field(default="localhost")
    mcp_server_port: int = Field(default=8765)

    # Gradio
    gradio_server_port: int = Field(default=7860)
    gradio_share: bool = Field(default=False)

    # provider: "google" or "openrouter"
    llm_provider: str = Field(default="openrouter")
    # model: "gemini-2.0-flash", "amazon/nova-2-lite-v1:free", etc.
    llm_model: str = Field(default="openrouter/auto")
    llm_temperature: float = Field(default=0.3)

    # Embedding Config
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    # Project
    project_root: Path = Path(__file__).parent.parent.parent.parent

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }


@lru_cache()
def get_settings() -> Settings:
    """Get singleton settings instance."""
    return Settings()


# Important paths
def get_data_dir() -> Path:
    """Get data directory."""
    data_dir = get_settings().project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_chroma_dir() -> Path:
    """Get ChromaDB directory."""
    chroma_dir = Path(get_settings().chroma_persist_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chroma_dir


def get_sqlite_path() -> Path:
    """Get SQLite database path."""
    sqlite_path = Path(get_settings().sqlite_db_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite_path


# =============================================================================
# UNIFIED LLM ABSTRACTION
# =============================================================================

@dataclass
class LLMResponse:
    """Unified response from any LLM."""
    text: str


class BaseLLMClient(ABC):
    """Base interface for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        """Generate synchronous response."""
        pass

    @abstractmethod
    async def generate_async(self, messages: list[dict]) -> LLMResponse:
        """Generate asynchronous response with structured messages."""
        pass


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini."""

    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_output_tokens: int,
        response_mime_type: Optional[str] = None
    ):
        import google.generativeai as genai

        settings = get_settings()
        if not settings.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required to use Gemini. "
                "Set it in .env or as an environment variable."
            )

        genai.configure(api_key=settings.google_api_key)

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if response_mime_type:
            generation_config["response_mime_type"] = response_mime_type

        self._model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config
        )

    def generate(self, prompt: str) -> LLMResponse:
        """Generate synchronous response."""
        response = self._model.generate_content(prompt)
        return LLMResponse(text=response.text)

    async def generate_async(self, messages: list[dict]) -> LLMResponse:
        """Generate asynchronous response with Gemini-style messages."""
        response = await self._model.generate_content_async(messages)
        return LLMResponse(text=response.text)


class OpenRouterClient(BaseLLMClient):
    """Client for OpenRouter (OpenAI-compatible API)."""

    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_output_tokens: int,
        response_mime_type: Optional[str] = None
    ):
        from openai import AsyncOpenAI, OpenAI

        settings = get_settings()
        if not settings.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required to use OpenRouter. "
                "Set it in .env or as an environment variable."
            )

        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_output_tokens
        self._json_mode = response_mime_type == "application/json"

        base_url = "https://openrouter.ai/api/v1"
        self._client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=base_url,
        )
        self._async_client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=base_url,
        )

    @staticmethod
    def _is_json_mode_unsupported_error(exc: Exception) -> bool:
        """Return True when provider rejects JSON mode/response_format."""
        message = str(exc).lower()
        return (
            "json mode is not enabled" in message
            or "response_format" in message
            or "json_object" in message
        )

    def generate(self, prompt: str) -> LLMResponse:
        """Generate synchronous response."""
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"} if self._json_mode else None,
            )
        except Exception as exc:
            if self._json_mode and self._is_json_mode_unsupported_error(exc):
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            else:
                raise
        return LLMResponse(text=response.choices[0].message.content or "")

    async def generate_async(self, messages: list[dict]) -> LLMResponse:
        """Generate asynchronous response.

        Converts messages from Gemini format to OpenAI format.
        Gemini: [{"role": "user", "parts": ["text"]}, {"role": "model", "parts": ["text"]}]
        OpenAI: [{"role": "user", "content": "text"}, {"role": "assistant", "content": "text"}]
        """
        openai_messages = []
        for msg in messages:
            role = "assistant" if msg["role"] == "model" else msg["role"]
            content = msg["parts"][0] if "parts" in msg else msg.get("content", "")
            openai_messages.append({"role": role, "content": content})

        try:
            response = await self._async_client.chat.completions.create(
                model=self._model_name,
                messages=openai_messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"} if self._json_mode else None,
            )
        except Exception as exc:
            if self._json_mode and self._is_json_mode_unsupported_error(exc):
                response = await self._async_client.chat.completions.create(
                    model=self._model_name,
                    messages=openai_messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            else:
                raise
        return LLMResponse(text=response.choices[0].message.content or "")


# =============================================================================
# LLM CLIENT FACTORY
# =============================================================================

_llm_client_cache: dict = {}


def get_llm_model(
    response_mime_type: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: int = 4096
) -> BaseLLMClient:
    """
    Factory to get a configured LLM client.

    Supports multiple providers based on LLM_PROVIDER in config:
    - "google": Google Gemini
    - "openrouter": OpenRouter (Llama, Mistral, etc.)

    Args:
        response_mime_type: Response MIME type (e.g., "application/json")
        temperature: Temperature override (uses config if None)
        max_output_tokens: Maximum output tokens

    Returns:
        Configured BaseLLMClient (GeminiClient or OpenRouterClient)

    Raises:
        ValueError: If the API key for the selected provider is missing
    """
    settings = get_settings()

    effective_temp = temperature if temperature is not None else settings.llm_temperature
    cache_key = (settings.llm_provider, settings.llm_model, response_mime_type, effective_temp, max_output_tokens)

    if cache_key not in _llm_client_cache:
        match settings.llm_provider.lower():
            case "google" | "gemini":
                _llm_client_cache[cache_key] = GeminiClient(
                    model_name=settings.llm_model,
                    temperature=effective_temp,
                    max_output_tokens=max_output_tokens,
                    response_mime_type=response_mime_type,
                )
            case "openrouter":
                _llm_client_cache[cache_key] = OpenRouterClient(
                    model_name=settings.llm_model,
                    temperature=effective_temp,
                    max_output_tokens=max_output_tokens,
                    response_mime_type=response_mime_type,
                )
            case _:
                raise ValueError(
                    f"Unsupported LLM provider: {settings.llm_provider}. "
                    "Use 'google' or 'openrouter'."
                )

    return _llm_client_cache[cache_key]
