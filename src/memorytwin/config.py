"""
Configuración centralizada para The Memory Twin
================================================
"""

import os
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Cargar variables de entorno
load_dotenv()


class Settings(BaseSettings):
    """Configuración del sistema cargada desde variables de entorno."""
    
    # API Keys
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Langfuse
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    langfuse_host: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    langfuse_enabled: bool = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
    
    # Database
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    sqlite_db_path: str = os.getenv("SQLITE_DB_PATH", "./data/memory.db")
    
    # MCP Server
    mcp_server_host: str = os.getenv("MCP_SERVER_HOST", "localhost")
    mcp_server_port: int = int(os.getenv("MCP_SERVER_PORT", "8765"))
    
    # Gradio
    gradio_server_port: int = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    gradio_share: bool = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    
    # LLM Config
    llm_provider: str = os.getenv("LLM_PROVIDER", "google")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    
    # Embedding Config
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Project
    project_root: Path = Path(__file__).parent.parent.parent.parent
    
    class Config:
        env_file = ".env"
        extra = "ignore"


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
