"""
Interfaces de Storage - Abstracción para múltiples backends
===========================================================

Define la interfaz base que deben implementar todos los backends de storage.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from memorytwin.models import Episode, MemoryQuery, MemorySearchResult


class StorageBackend(ABC):
    """
    Interfaz abstracta para backends de almacenamiento.
    
    Todos los backends (Local, ChromaDB Server, PostgreSQL, etc.)
    deben implementar estos métodos.
    """
    
    @abstractmethod
    def store_episode(self, episode: Episode) -> str:
        """
        Almacenar un episodio.
        
        Args:
            episode: Episodio a almacenar
            
        Returns:
            ID del episodio almacenado
        """
        pass
    
    @abstractmethod
    def search_episodes(self, query: MemoryQuery) -> list[MemorySearchResult]:
        """
        Buscar episodios por similitud semántica.
        
        Args:
            query: Consulta de búsqueda
            
        Returns:
            Lista de resultados ordenados por relevancia
        """
        pass
    
    @abstractmethod
    def get_episode_by_id(self, episode_id: str) -> Optional[Episode]:
        """Recuperar un episodio por su ID."""
        pass
    
    @abstractmethod
    def get_timeline(
        self,
        project_name: Optional[str] = None,
        limit: int = 100
    ) -> list[Episode]:
        """Obtener timeline cronológico de episodios."""
        pass
    
    @abstractmethod
    def get_lessons_learned(
        self,
        project_name: Optional[str] = None,
        tags: Optional[list[str]] = None
    ) -> list[dict]:
        """Obtener lecciones aprendidas agregadas."""
        pass
    
    @abstractmethod
    def get_statistics(self, project_name: Optional[str] = None) -> dict:
        """Obtener estadísticas del almacenamiento."""
        pass


def get_storage_backend(backend_type: Optional[str] = None) -> StorageBackend:
    """
    Factory para obtener el backend de storage configurado.
    
    Args:
        backend_type: Tipo de backend. Si es None, se lee de configuración.
                     Opciones: "local", "chromadb_server"
    
    Returns:
        Instancia del backend configurado
    """
    import os
    
    backend = backend_type or os.getenv("STORAGE_BACKEND", "local")
    
    if backend == "local":
        from memorytwin.escriba.storage import MemoryStorage
        return MemoryStorage()
    elif backend == "chromadb_server":
        from memorytwin.escriba.storage_chromadb_server import ChromaDBServerStorage
        return ChromaDBServerStorage()
    else:
        raise ValueError(
            f"Backend de storage desconocido: '{backend}'. "
            f"Opciones válidas: 'local', 'chromadb_server'"
        )
