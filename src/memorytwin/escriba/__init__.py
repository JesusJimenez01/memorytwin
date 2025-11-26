"""
Agente Escriba - Backend de Ingesta de Memoria
==============================================

Observador pasivo y documentador activo que captura,
procesa y almacena el razonamiento de los asistentes de IA.
"""

from memorytwin.escriba.processor import ThoughtProcessor
from memorytwin.escriba.storage import MemoryStorage
from memorytwin.escriba.escriba import Escriba

__all__ = ["ThoughtProcessor", "MemoryStorage", "Escriba"]
