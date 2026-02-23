"""
Oráculo Agent - Memory Query Frontend
======================================

Knowledge retrieval and onboarding assistant
with Gradio interface and RAG capabilities.
"""

from memorytwin.oraculo.oraculo import Oraculo
from memorytwin.oraculo.rag_engine import RAGEngine

__all__ = ["RAGEngine", "Oraculo"]


def launch_app():
    """Launch the Gradio web interface."""
    from memorytwin.oraculo.app import main
    main()
