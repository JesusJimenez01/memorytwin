"""
Agente Oráculo - Frontend de Consulta de Memoria
================================================

Asistente de recuperación de conocimiento y onboarding
con interfaz Gradio y capacidades RAG.
"""

from memorytwin.oraculo.rag_engine import RAGEngine
from memorytwin.oraculo.oraculo import Oraculo

__all__ = ["RAGEngine", "Oraculo"]
