"""
Integración con Langfuse para Observabilidad
============================================

Proporciona trazabilidad completa de las operaciones
del Escriba y Oráculo para evaluar la calidad del sistema.
"""

from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Optional
import logging

from memorytwin.config import get_settings

logger = logging.getLogger("memorytwin.observability")

# Cliente Langfuse (inicializado lazy)
_langfuse = None


def get_langfuse():
    """Obtener cliente Langfuse singleton."""
    global _langfuse
    
    settings = get_settings()
    
    if not settings.langfuse_enabled:
        return None
        
    if _langfuse is None:
        try:
            from langfuse import Langfuse
            
            _langfuse = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host
            )
            logger.info("Langfuse inicializado correctamente")
        except Exception as e:
            logger.warning(f"No se pudo inicializar Langfuse: {e}")
            return None
            
    return _langfuse


class MemoryTwinObservability:
    """
    Wrapper de observabilidad para Memory Twin.
    
    Proporciona decoradores y context managers para trazar:
    - Capturas del Escriba (procesamiento de pensamientos)
    - Consultas del Oráculo (RAG queries)
    - Búsquedas en la memoria
    - Rendimiento del sistema
    """
    
    def __init__(self):
        """Inicializar observabilidad."""
        self.langfuse = get_langfuse()
        self.enabled = self.langfuse is not None
        
    @contextmanager
    def trace_capture(
        self,
        thinking_text: str,
        project_name: str,
        source_assistant: str,
        metadata: Optional[dict] = None
    ):
        """
        Context manager para trazar captura de pensamiento.
        
        Usage:
            with obs.trace_capture(text, project, source) as trace:
                episode = processor.process(...)
                trace.update(episode_id=str(episode.id))
        """
        if not self.enabled:
            yield DummyTrace()
            return
            
        trace = self.langfuse.trace(
            name="escriba_capture",
            input={"thinking_text": thinking_text[:500]},  # Truncar para Langfuse
            metadata={
                "project_name": project_name,
                "source_assistant": source_assistant,
                "text_length": len(thinking_text),
                **(metadata or {})
            }
        )
        
        try:
            yield trace
            trace.update(status="success")
        except Exception as e:
            trace.update(
                status="error",
                output={"error": str(e)}
            )
            raise
        finally:
            self.langfuse.flush()
    
    @contextmanager
    def trace_rag_query(
        self,
        question: str,
        project_filter: Optional[str] = None,
        num_episodes: int = 5,
        metadata: Optional[dict] = None
    ):
        """
        Context manager para trazar consulta RAG.
        
        Usage:
            with obs.trace_rag_query(question, project) as trace:
                result = rag_engine.query(...)
                trace.update(output=result)
        """
        if not self.enabled:
            yield DummyTrace()
            return
            
        trace = self.langfuse.trace(
            name="oraculo_query",
            input={"question": question},
            metadata={
                "project_filter": project_filter,
                "num_episodes": num_episodes,
                **(metadata or {})
            }
        )
        
        try:
            yield trace
            trace.update(status="success")
        except Exception as e:
            trace.update(
                status="error",
                output={"error": str(e)}
            )
            raise
        finally:
            self.langfuse.flush()
    
    def trace_generation(
        self,
        parent_trace,
        name: str,
        model: str,
        input_text: str,
        output_text: str,
        usage: Optional[dict] = None
    ):
        """
        Registrar una generación de LLM dentro de una traza.
        
        Args:
            parent_trace: Traza padre
            name: Nombre de la generación
            model: Modelo usado
            input_text: Texto de entrada
            output_text: Texto de salida
            usage: Métricas de uso (tokens, etc.)
        """
        if not self.enabled or parent_trace is None:
            return
            
        parent_trace.generation(
            name=name,
            model=model,
            input=input_text,
            output=output_text,
            usage=usage
        )
    
    def trace_retrieval(
        self,
        parent_trace,
        query: str,
        num_results: int,
        results_summary: list[dict]
    ):
        """
        Registrar una operación de retrieval (búsqueda vectorial).
        
        Args:
            parent_trace: Traza padre
            query: Consulta de búsqueda
            num_results: Número de resultados encontrados
            results_summary: Resumen de resultados (ids, scores)
        """
        if not self.enabled or parent_trace is None:
            return
            
        parent_trace.span(
            name="vector_search",
            input={"query": query},
            output={
                "num_results": num_results,
                "results": results_summary
            }
        )
    
    def log_feedback(
        self,
        trace_id: str,
        score: float,
        comment: Optional[str] = None,
        feedback_type: str = "quality"
    ):
        """
        Registrar feedback sobre una respuesta.
        
        Args:
            trace_id: ID de la traza a evaluar
            score: Puntuación (0-1)
            comment: Comentario opcional
            feedback_type: Tipo de feedback (quality, relevance, etc.)
        """
        if not self.enabled:
            return
            
        self.langfuse.score(
            trace_id=trace_id,
            name=feedback_type,
            value=score,
            comment=comment
        )
        self.langfuse.flush()
    
    def create_dataset(
        self,
        name: str,
        description: str,
        items: list[dict]
    ) -> Optional[str]:
        """
        Crear un dataset en Langfuse para evaluación.
        
        Args:
            name: Nombre del dataset
            description: Descripción
            items: Lista de items {input, expected_output}
            
        Returns:
            ID del dataset creado o None
        """
        if not self.enabled:
            return None
            
        dataset = self.langfuse.create_dataset(
            name=name,
            description=description
        )
        
        for item in items:
            self.langfuse.create_dataset_item(
                dataset_name=name,
                input=item.get("input"),
                expected_output=item.get("expected_output"),
                metadata=item.get("metadata")
            )
        
        self.langfuse.flush()
        return dataset.id


class DummyTrace:
    """Traza dummy cuando Langfuse está deshabilitado."""
    
    def update(self, **kwargs):
        pass
    
    def generation(self, **kwargs):
        pass
    
    def span(self, **kwargs):
        pass
    
    @property
    def id(self):
        return "dummy-trace-id"


# Decoradores de conveniencia
def observe_capture(func: Callable) -> Callable:
    """
    Decorador para observar funciones de captura.
    
    La función debe tener parámetros: thinking_text, project_name, source_assistant
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        obs = MemoryTwinObservability()
        
        thinking_text = kwargs.get("thinking_text") or (args[0] if args else "")
        project_name = kwargs.get("project_name", "default")
        source_assistant = kwargs.get("source_assistant", "unknown")
        
        with obs.trace_capture(thinking_text, project_name, source_assistant) as trace:
            result = await func(*args, **kwargs)
            
            if hasattr(result, "id"):
                trace.update(output={"episode_id": str(result.id)})
                
            return result
            
    return wrapper


def observe_query(func: Callable) -> Callable:
    """
    Decorador para observar funciones de consulta RAG.
    
    La función debe tener parámetro: question
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        obs = MemoryTwinObservability()
        
        question = kwargs.get("question") or (args[0] if args else "")
        project_name = kwargs.get("project_name")
        
        with obs.trace_rag_query(question, project_name) as trace:
            result = await func(*args, **kwargs)
            
            if isinstance(result, dict) and "answer" in result:
                trace.update(output={
                    "answer": result["answer"][:500],
                    "episodes_count": len(result.get("episodes_used", []))
                })
                
            return result
            
    return wrapper


# Instancia global para uso simple
observability = MemoryTwinObservability()
