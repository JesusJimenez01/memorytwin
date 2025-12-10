"""
Observabilidad con Langfuse v3 - Versión Simplificada
=====================================================

Solo 3 trazas principales:
1. 📥 Almacenar Recuerdo - Input/Output del LLM al estructurar pensamiento
2. 🔍 Acceder Recuerdos - Input/Output del LLM en consultas RAG  
3. 🔄 Consolidar Memorias - Episodios consolidados → MetaMemoria creada

Configuración via .env:
  - LANGFUSE_PUBLIC_KEY
  - LANGFUSE_SECRET_KEY
  - LANGFUSE_HOST (opcional)
"""

import logging
import os
import sys
from functools import wraps

# Silenciar warnings molestos de Langfuse ("Calling end() on an ended span")
logging.getLogger("langfuse").setLevel(logging.ERROR)

# Importar config primero para cargar .env
from memorytwin.config import get_settings  # noqa: F401 - asegura que .env esté cargado

try:
    from langfuse import Langfuse  # type: ignore
except ImportError:
    Langfuse = None

__all__ = ["trace_store_memory", "trace_access_memory", "trace_consolidation", "flush_traces"]

# Cliente Langfuse singleton
_langfuse_client = None


def _is_disabled() -> bool:
    """Verificar si Langfuse está deshabilitado (tests o credenciales vacías)."""
    if Langfuse is None:
        return True
    if "pytest" in sys.modules:
        return True
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return True
    return False


def _get_langfuse():
    """Obtener cliente Langfuse singleton."""
    global _langfuse_client
    if _langfuse_client is None and not _is_disabled():
        try:
            _langfuse_client = Langfuse()
        except Exception:
            pass
    return _langfuse_client


def flush_traces():
    """Forzar envío de trazas pendientes."""
    client = _get_langfuse()
    if client:
        try:
            client.flush()
        except Exception:
            pass


def trace_store_memory(func):
    """
    Decorador para trazar almacenamiento de recuerdos.
    Captura: input al LLM (thinking text) → output del LLM (episodio estructurado)
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if _is_disabled():
            return await func(*args, **kwargs)
        
        # Extraer input (raw_input está en args[1] o kwargs)
        raw_input = kwargs.get('raw_input') or (args[1] if len(args) > 1 else None)
        project_name = kwargs.get('project_name', 'default')
        input_text = raw_input.raw_text[:500] if raw_input else "N/A"
        
        client = _get_langfuse()
        if not client:
            return await func(*args, **kwargs)
        
        try:
            with client.start_as_current_span(
                name="📥 Almacenar Recuerdo",
                input={"thinking_text": input_text, "project": project_name},
                metadata={"project": project_name, "operation": "store"}
            ) as span:
                result = await func(*args, **kwargs)
                span.update(output={
                    "episode_id": str(result.id),
                    "task": result.task,
                    "type": result.episode_type.value,
                    "tags": result.tags[:5],
                    "lessons": result.lessons_learned[:3]
                })
                return result
        except Exception as e:
            # En caso de error, igual crear span para registrarlo
            with client.start_as_current_span(
                name="📥 Almacenar Recuerdo - ERROR",
                input={"thinking_text": input_text},
                level="ERROR"
            ) as span:
                span.update(output={"error": str(e)}, status_message=str(e))
            raise
        finally:
            client.flush()
    
    return wrapper


def trace_access_memory(func):
    """
    Decorador para trazar acceso a recuerdos (RAG).
    Captura: pregunta del usuario → respuesta generada con contexto
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if _is_disabled():
            return await func(*args, **kwargs)
        
        # Extraer input (question está en args[1] o kwargs)
        question = kwargs.get('question') or (args[1] if len(args) > 1 else "N/A")
        project_name = kwargs.get('project_name', 'all')
        
        client = _get_langfuse()
        if not client:
            return await func(*args, **kwargs)
        
        try:
            with client.start_as_current_span(
                name="🔍 Acceder Recuerdos",
                input={"question": question, "project": project_name},
                metadata={"project": project_name or "all", "operation": "access"}
            ) as span:
                result = await func(*args, **kwargs)
                span.update(output={
                    "answer": result.get("answer", "")[:500],
                    "episodes_count": len(result.get("episodes_used", [])),
                    "meta_memories_count": len(result.get("meta_memories_used", [])),
                    "context_provided": result.get("context_provided", False)
                })
                return result
        except Exception as e:
            with client.start_as_current_span(
                name="🔍 Acceder Recuerdos - ERROR",
                input={"question": question},
                level="ERROR"
            ) as span:
                span.update(output={"error": str(e)}, status_message=str(e))
            raise
        finally:
            client.flush()
    
    return wrapper


def trace_consolidation(func):
    """
    Decorador para trazar consolidación de memorias.
    Captura: episodios a consolidar → meta-memoria creada
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if _is_disabled():
            return func(*args, **kwargs)
        
        # Extraer episodios (están en args[1] para métodos de clase)
        episodes = kwargs.get('episodes') or (args[1] if len(args) > 1 else [])
        project_name = kwargs.get('project_name') or (args[2] if len(args) > 2 else 'unknown')
        
        episode_summaries = [
            {"id": str(ep.id), "task": ep.task[:100], "type": ep.episode_type.value}
            for ep in episodes[:10]
        ]
        
        client = _get_langfuse()
        if not client:
            return func(*args, **kwargs)
        
        try:
            with client.start_as_current_span(
                name="🔄 Consolidar Memorias",
                input={
                    "episodes_count": len(episodes),
                    "episodes": episode_summaries,
                    "project": project_name
                },
                metadata={"project": project_name, "operation": "consolidate"}
            ) as span:
                result = func(*args, **kwargs)
                if result:
                    span.update(output={
                        "meta_memory_id": str(result.id),
                        "pattern": result.pattern[:200],
                        "pattern_summary": result.pattern_summary[:200],
                        "lessons": result.lessons[:3],
                        "best_practices": result.best_practices[:3],
                        "confidence": result.confidence
                    })
                return result
        except Exception as e:
            with client.start_as_current_span(
                name="🔄 Consolidar Memorias - ERROR",
                input={"episodes_count": len(episodes)},
                level="ERROR"
            ) as span:
                span.update(output={"error": str(e)}, status_message=str(e))
            raise
        finally:
            client.flush()
    
    return wrapper
