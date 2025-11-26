"""
Modelos de datos para The Memory Twin
=====================================

Define los esquemas Pydantic para episodios de memoria, 
metadatos y configuración del sistema.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EpisodeType(str, Enum):
    """Tipos de episodios de memoria capturados."""
    
    DECISION = "decision"           # Decisión técnica tomada
    BUG_FIX = "bug_fix"             # Corrección de error
    REFACTOR = "refactor"           # Refactorización de código
    FEATURE = "feature"             # Nueva funcionalidad
    OPTIMIZATION = "optimization"   # Mejora de rendimiento
    LEARNING = "learning"           # Aprendizaje o descubrimiento
    EXPERIMENT = "experiment"       # Prueba o experimento


class ReasoningTrace(BaseModel):
    """
    Traza de razonamiento capturada del asistente de IA.
    Representa el "thinking" visible del modelo.
    """
    
    raw_thinking: str = Field(
        ..., 
        description="Texto crudo del razonamiento del modelo"
    )
    alternatives_considered: list[str] = Field(
        default_factory=list,
        description="Alternativas consideradas y descartadas"
    )
    decision_factors: list[str] = Field(
        default_factory=list,
        description="Factores que influyeron en la decisión"
    )
    confidence_level: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nivel de confianza en la decisión (0-1)"
    )


class Episode(BaseModel):
    """
    Episodio de memoria - Unidad fundamental de conocimiento.
    
    Captura el contexto completo de una decisión técnica:
    qué se hizo, cómo se razonó y por qué.
    """
    
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Contexto del episodio
    task: str = Field(
        ..., 
        description="Descripción de la tarea o problema abordado"
    )
    context: str = Field(
        ..., 
        description="Contexto técnico: archivos, módulos, stack involucrado"
    )
    
    # Razonamiento
    reasoning_trace: ReasoningTrace = Field(
        ...,
        description="Traza del pensamiento del modelo"
    )
    
    # Solución
    solution: str = Field(
        ..., 
        description="Código o solución implementada"
    )
    solution_summary: str = Field(
        ..., 
        description="Resumen ejecutivo de la solución"
    )
    
    # Resultado
    outcome: Optional[str] = Field(
        default=None,
        description="Resultado observado tras aplicar la solución"
    )
    success: bool = Field(
        default=True,
        description="Si la solución fue exitosa"
    )
    
    # Metadatos
    episode_type: EpisodeType = Field(
        default=EpisodeType.DECISION,
        description="Tipo de episodio"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Etiquetas para categorización"
    )
    files_affected: list[str] = Field(
        default_factory=list,
        description="Archivos modificados"
    )
    
    # Lecciones
    lessons_learned: list[str] = Field(
        default_factory=list,
        description="Lecciones extraídas de este episodio"
    )
    
    # Fuente
    source_assistant: str = Field(
        default="unknown",
        description="Asistente de código fuente (copilot, claude, cursor)"
    )
    project_name: str = Field(
        default="default",
        description="Nombre del proyecto asociado"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "task": "Implementar autenticación JWT",
                "context": "Módulo auth/ en proyecto FastAPI, usuarios en PostgreSQL",
                "reasoning_trace": {
                    "raw_thinking": "Consideré usar sessions pero JWT es mejor para APIs...",
                    "alternatives_considered": ["Sessions con Redis", "OAuth2 puro"],
                    "decision_factors": ["Stateless", "Escalabilidad", "Mobile-friendly"],
                    "confidence_level": 0.85
                },
                "solution": "from jose import jwt...",
                "solution_summary": "JWT con PyJWT, tokens de 24h, refresh tokens",
                "episode_type": "feature",
                "tags": ["auth", "security", "jwt"],
                "lessons_learned": ["Siempre validar el algoritmo del JWT"]
            }
        }


class MemoryQuery(BaseModel):
    """Consulta al sistema de memoria del Oráculo."""
    
    query: str = Field(
        ...,
        description="Pregunta o búsqueda del usuario"
    )
    project_filter: Optional[str] = Field(
        default=None,
        description="Filtrar por proyecto específico"
    )
    type_filter: Optional[EpisodeType] = Field(
        default=None,
        description="Filtrar por tipo de episodio"
    )
    date_from: Optional[datetime] = Field(
        default=None,
        description="Fecha inicio del rango de búsqueda"
    )
    date_to: Optional[datetime] = Field(
        default=None,
        description="Fecha fin del rango de búsqueda"
    )
    tags_filter: list[str] = Field(
        default_factory=list,
        description="Filtrar por etiquetas"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Número de resultados a retornar"
    )


class MemorySearchResult(BaseModel):
    """Resultado de búsqueda en la memoria."""
    
    episode: Episode
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Puntuación de relevancia semántica"
    )
    match_reason: str = Field(
        default="",
        description="Explicación de por qué este resultado es relevante"
    )


class ProcessedInput(BaseModel):
    """
    Input procesado del Escriba.
    Representa el texto capturado antes de ser convertido en Episode.
    """
    
    raw_text: str = Field(
        ...,
        description="Texto crudo capturado (thinking del modelo)"
    )
    user_prompt: Optional[str] = Field(
        default=None,
        description="Prompt original del usuario"
    )
    code_changes: Optional[str] = Field(
        default=None,
        description="Cambios de código asociados (diff o código nuevo)"
    )
    source: str = Field(
        default="manual",
        description="Fuente de captura: manual, clipboard, mcp"
    )
    captured_at: datetime = Field(default_factory=datetime.utcnow)


class SystemConfig(BaseModel):
    """Configuración del sistema Memory Twin."""
    
    # LLM Settings
    llm_provider: str = Field(default="google")
    llm_model: str = Field(default="gemini-2.0-flash")
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    
    # Embedding Settings  
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    
    # Database Settings
    chroma_persist_dir: str = Field(default="./data/chroma")
    sqlite_db_path: str = Field(default="./data/memory.db")
    
    # MCP Settings
    mcp_host: str = Field(default="localhost")
    mcp_port: int = Field(default=8765)
    
    # Langfuse Settings
    langfuse_enabled: bool = Field(default=True)
