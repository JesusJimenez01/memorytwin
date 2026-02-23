"""
Escriba Agent - Memory Ingestion Backend
=========================================

Passive observer and active documenter that captures,
processes, and stores AI assistant reasoning.
"""

from memorytwin.escriba.escriba import Escriba
from memorytwin.escriba.processor import ThoughtProcessor
from memorytwin.escriba.storage import MemoryStorage

__all__ = ["ThoughtProcessor", "MemoryStorage", "Escriba"]
