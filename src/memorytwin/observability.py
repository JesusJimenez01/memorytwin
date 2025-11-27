"""
Observabilidad con Langfuse v3
==============================

Decoradores y utilidades para trazabilidad de operaciones.
Usa @observe de Langfuse directamente - configuración via .env:
  - LANGFUSE_PUBLIC_KEY
  - LANGFUSE_SECRET_KEY
  - LANGFUSE_HOST (opcional)
"""

from langfuse import observe, Langfuse

__all__ = ["observe", "get_langfuse", "flush"]


def get_langfuse() -> Langfuse:
    """Obtener cliente Langfuse (usa variables de entorno automáticamente)."""
    return Langfuse()


def flush():
    """Forzar envío de trazas pendientes."""
    get_langfuse().flush()
