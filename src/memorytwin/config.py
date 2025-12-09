"""
Configuración centralizada para The Memory Twin
================================================

Soporta múltiples proveedores LLM:
- google: Google Gemini (default)
- openrouter: OpenRouter (acceso a múltiples modelos)
"""

from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Cargar variables de entorno (llamada única para todo el proyecto)
load_dotenv()


class Settings(BaseSettings):
    """Configuración del sistema cargada desde variables de entorno.
    
    pydantic-settings carga automáticamente desde .env, no usar os.getenv().
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
    
    # LLM Config
    # provider: "google" o "openrouter"
    llm_provider: str = Field(default="google")
    # modelo: "gemini-2.0-flash", "meta-llama/llama-3.1-8b-instruct:free", etc.
    llm_model: str = Field(default="gemini-2.0-flash")
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
    """Obtener instancia singleton de configuración."""
    return Settings()


# Paths importantes
def get_data_dir() -> Path:
    """Obtener directorio de datos."""
    data_dir = get_settings().project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_chroma_dir() -> Path:
    """Obtener directorio de ChromaDB."""
    chroma_dir = Path(get_settings().chroma_persist_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chroma_dir


def get_sqlite_path() -> Path:
    """Obtener path de SQLite."""
    sqlite_path = Path(get_settings().sqlite_db_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite_path


# =============================================================================
# ABSTRACCIÓN LLM UNIFICADA
# =============================================================================

@dataclass
class LLMResponse:
    """Respuesta unificada de cualquier LLM."""
    text: str


class BaseLLMClient(ABC):
    """Interfaz base para clientes LLM."""
    
    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        """Generar respuesta sincrónica."""
        pass
    
    @abstractmethod
    async def generate_async(self, messages: list[dict]) -> LLMResponse:
        """Generar respuesta asincrónica con mensajes estructurados."""
        pass


class GeminiClient(BaseLLMClient):
    """Cliente para Google Gemini."""
    
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
                "Se requiere GOOGLE_API_KEY para usar Gemini. "
                "Configúrala en .env o como variable de entorno."
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
        """Generar respuesta sincrónica."""
        response = self._model.generate_content(prompt)
        return LLMResponse(text=response.text)
    
    async def generate_async(self, messages: list[dict]) -> LLMResponse:
        """Generar respuesta asincrónica con mensajes estilo Gemini."""
        response = await self._model.generate_content_async(messages)
        return LLMResponse(text=response.text)


class OpenRouterClient(BaseLLMClient):
    """Cliente para OpenRouter (compatible con API de OpenAI)."""
    
    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_output_tokens: int,
        response_mime_type: Optional[str] = None
    ):
        from openai import OpenAI, AsyncOpenAI
        
        settings = get_settings()
        if not settings.openrouter_api_key:
            raise ValueError(
                "Se requiere OPENROUTER_API_KEY para usar OpenRouter. "
                "Configúrala en .env o como variable de entorno."
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
    
    def generate(self, prompt: str) -> LLMResponse:
        """Generar respuesta sincrónica."""
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"} if self._json_mode else None,
        )
        return LLMResponse(text=response.choices[0].message.content or "")
    
    async def generate_async(self, messages: list[dict]) -> LLMResponse:
        """Generar respuesta asincrónica.
        
        Convierte mensajes de formato Gemini a formato OpenAI.
        Gemini: [{"role": "user", "parts": ["text"]}, {"role": "model", "parts": ["text"]}]
        OpenAI: [{"role": "user", "content": "text"}, {"role": "assistant", "content": "text"}]
        """
        openai_messages = []
        for msg in messages:
            role = "assistant" if msg["role"] == "model" else msg["role"]
            content = msg["parts"][0] if "parts" in msg else msg.get("content", "")
            openai_messages.append({"role": role, "content": content})
        
        response = await self._async_client.chat.completions.create(
            model=self._model_name,
            messages=openai_messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"} if self._json_mode else None,
        )
        return LLMResponse(text=response.choices[0].message.content or "")


# =============================================================================
# FACTORY PARA CLIENTE LLM
# =============================================================================

_llm_client_cache: dict = {}


def get_llm_model(
    response_mime_type: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: int = 4096
) -> BaseLLMClient:
    """
    Factory para obtener cliente LLM configurado.
    
    Soporta múltiples proveedores según LLM_PROVIDER en config:
    - "google": Google Gemini
    - "openrouter": OpenRouter (Llama, Mistral, etc.)
    
    Args:
        response_mime_type: MIME type de respuesta (ej: "application/json")
        temperature: Override de temperatura (usa config si None)
        max_output_tokens: Máximo de tokens de salida
        
    Returns:
        BaseLLMClient configurado (GeminiClient u OpenRouterClient)
        
    Raises:
        ValueError: Si falta la API key del proveedor seleccionado
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
                    f"Proveedor LLM no soportado: {settings.llm_provider}. "
                    "Usa 'google' o 'openrouter'."
                )
    
    return _llm_client_cache[cache_key]
