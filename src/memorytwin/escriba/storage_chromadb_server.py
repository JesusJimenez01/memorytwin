"""
Storage Backend - ChromaDB Server (Compartido)
==============================================

Backend para ChromaDB corriendo como servidor.
Permite que múltiples usuarios/agentes compartan la misma base de datos.

Configuración:
    CHROMADB_SERVER_HOST: Host del servidor (default: localhost)
    CHROMADB_SERVER_PORT: Puerto del servidor (default: 8000)
"""

import json
import os
from datetime import datetime
from typing import Optional
from uuid import UUID

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from memorytwin.escriba.storage_interface import StorageBackend
from memorytwin.models import Episode, EpisodeType, MemoryQuery, MemorySearchResult, ReasoningTrace


class ChromaDBServerStorage(StorageBackend):
    """
    Backend de almacenamiento usando ChromaDB Server.
    
    Ideal para equipos pequeños/medianos que quieren compartir memorias
    sin configurar una base de datos tradicional.
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Inicializar conexión a ChromaDB Server.
        
        Args:
            host: Host del servidor ChromaDB
            port: Puerto del servidor
            embedding_model: Modelo de embeddings a usar
        """
        self.host = host or os.getenv("CHROMADB_SERVER_HOST", "localhost")
        self.port = port or int(os.getenv("CHROMADB_SERVER_PORT", "8000"))
        
        # Cliente HTTP a ChromaDB Server
        self.client = chromadb.HttpClient(
            host=self.host,
            port=self.port,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Colección principal
        self.collection = self.client.get_or_create_collection(
            name="memory_episodes",
            metadata={"description": "Episodios de memoria compartidos"}
        )
        
        # Colección de metadatos completos (ChromaDB tiene límite en metadata)
        self.metadata_collection = self.client.get_or_create_collection(
            name="episode_metadata",
            metadata={"description": "Metadatos completos de episodios"}
        )
        
        # Modelo de embeddings
        self.embedder = SentenceTransformer(embedding_model, device="cpu")
        
    def _generate_embedding(self, episode: Episode) -> list[float]:
        """Generar embedding del episodio."""
        text_parts = [
            f"Tarea: {episode.task}",
            f"Contexto: {episode.context}",
            f"Razonamiento: {episode.reasoning_trace.raw_thinking}",
            f"Solución: {episode.solution_summary}",
        ]
        
        if episode.lessons_learned:
            text_parts.append(f"Lecciones: {' '.join(episode.lessons_learned)}")
            
        combined_text = "\n".join(text_parts)
        return self.embedder.encode(combined_text).tolist()
    
    def store_episode(self, episode: Episode) -> str:
        """Almacenar episodio en ChromaDB Server."""
        episode_id = str(episode.id)
        
        # Generar embedding
        embedding = self._generate_embedding(episode)
        
        # Almacenar en colección principal (para búsqueda vectorial)
        self.collection.add(
            ids=[episode_id],
            embeddings=[embedding],
            metadatas=[{
                "task": episode.task[:500],
                "episode_type": episode.episode_type.value,
                "project_name": episode.project_name,
                "source_assistant": episode.source_assistant,
                "timestamp": episode.timestamp.isoformat(),
                "tags": ",".join(episode.tags),
                "success": str(episode.success),
            }],
            documents=[episode.reasoning_trace.raw_thinking]
        )
        
        # Almacenar metadatos completos en colección separada
        full_metadata = {
            "id": episode_id,
            "timestamp": episode.timestamp.isoformat(),
            "task": episode.task,
            "context": episode.context,
            "reasoning_trace": episode.reasoning_trace.model_dump_json(),
            "solution": episode.solution,
            "solution_summary": episode.solution_summary,
            "outcome": episode.outcome or "",
            "success": episode.success,
            "episode_type": episode.episode_type.value,
            "tags": json.dumps(episode.tags),
            "files_affected": json.dumps(episode.files_affected),
            "lessons_learned": json.dumps(episode.lessons_learned),
            "source_assistant": episode.source_assistant,
            "project_name": episode.project_name,
        }
        
        self.metadata_collection.add(
            ids=[episode_id],
            documents=[json.dumps(full_metadata)],
            metadatas=[{"episode_id": episode_id}]
        )
        
        return episode_id
    
    def search_episodes(self, query: MemoryQuery) -> list[MemorySearchResult]:
        """Buscar episodios por similitud semántica."""
        query_embedding = self.embedder.encode(query.query).tolist()
        
        # Filtros
        where_filters = {}
        if query.project_filter:
            where_filters["project_name"] = query.project_filter
        if query.type_filter:
            where_filters["episode_type"] = query.type_filter.value
            
        # Búsqueda vectorial
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=query.top_k,
            where=where_filters if where_filters else None,
            include=["metadatas", "distances"]
        )
        
        search_results = []
        
        if results["ids"] and results["ids"][0]:
            for i, episode_id in enumerate(results["ids"][0]):
                episode = self.get_episode_by_id(episode_id)
                if episode:
                    distance = results["distances"][0][i] if results["distances"] else 0
                    relevance_score = max(0, 1 - distance / 2)
                    
                    search_results.append(MemorySearchResult(
                        episode=episode,
                        relevance_score=relevance_score,
                        match_reason="Coincidencia semántica"
                    ))
                    
        return search_results
    
    def get_episode_by_id(self, episode_id: str) -> Optional[Episode]:
        """Recuperar episodio por ID."""
        try:
            result = self.metadata_collection.get(
                ids=[episode_id],
                include=["documents"]
            )
            
            if not result["documents"] or not result["documents"][0]:
                return None
                
            metadata = json.loads(result["documents"][0])
            return self._metadata_to_episode(metadata)
            
        except Exception:
            return None
    
    def get_timeline(
        self,
        project_name: Optional[str] = None,
        limit: int = 100
    ) -> list[Episode]:
        """Obtener timeline de episodios."""
        # Obtener todos los metadatos
        where_filter = {"project_name": project_name} if project_name else None
        
        results = self.collection.get(
            where=where_filter,
            include=["metadatas"]
        )
        
        episodes = []
        if results["ids"]:
            for episode_id in results["ids"][:limit]:
                episode = self.get_episode_by_id(episode_id)
                if episode:
                    episodes.append(episode)
        
        # Ordenar por timestamp descendente
        episodes.sort(key=lambda e: e.timestamp, reverse=True)
        return episodes[:limit]
    
    def get_lessons_learned(
        self,
        project_name: Optional[str] = None,
        tags: Optional[list[str]] = None
    ) -> list[dict]:
        """Obtener lecciones aprendidas."""
        episodes = self.get_timeline(project_name=project_name, limit=500)
        
        lessons = []
        for episode in episodes:
            if not episode.lessons_learned:
                continue
                
            # Filtrar por tags si se especifican
            if tags and not any(t in episode.tags for t in tags):
                continue
                
            for lesson in episode.lessons_learned:
                lessons.append({
                    "lesson": lesson,
                    "from_task": episode.task,
                    "timestamp": episode.timestamp,
                    "tags": episode.tags,
                    "episode_id": str(episode.id)
                })
                
        return lessons
    
    def get_statistics(self, project_name: Optional[str] = None) -> dict:
        """Obtener estadísticas."""
        where_filter = {"project_name": project_name} if project_name else None
        
        results = self.collection.get(
            where=where_filter,
            include=["metadatas"]
        )
        
        total = len(results["ids"]) if results["ids"] else 0
        
        # Contar por tipo y asistente
        type_counts = {t.value: 0 for t in EpisodeType}
        assistant_counts = {}
        
        if results["metadatas"]:
            for metadata in results["metadatas"]:
                ep_type = metadata.get("episode_type", "feature")
                if ep_type in type_counts:
                    type_counts[ep_type] += 1
                    
                assistant = metadata.get("source_assistant", "unknown")
                assistant_counts[assistant] = assistant_counts.get(assistant, 0) + 1
        
        return {
            "total_episodes": total,
            "by_type": type_counts,
            "by_assistant": assistant_counts,
            "storage_type": "chromadb_server",
            "server": f"{self.host}:{self.port}"
        }
    
    def _metadata_to_episode(self, metadata: dict) -> Episode:
        """Convertir metadatos a Episode."""
        reasoning_data = json.loads(metadata["reasoning_trace"])
        
        return Episode(
            id=UUID(metadata["id"]),
            timestamp=datetime.fromisoformat(metadata["timestamp"]),
            task=metadata["task"],
            context=metadata["context"],
            reasoning_trace=ReasoningTrace(**reasoning_data),
            solution=metadata.get("solution", ""),
            solution_summary=metadata.get("solution_summary", ""),
            outcome=metadata.get("outcome"),
            success=metadata.get("success", True),
            episode_type=EpisodeType(metadata["episode_type"]),
            tags=json.loads(metadata.get("tags", "[]")),
            files_affected=json.loads(metadata.get("files_affected", "[]")),
            lessons_learned=json.loads(metadata.get("lessons_learned", "[]")),
            source_assistant=metadata.get("source_assistant", "unknown"),
            project_name=metadata.get("project_name", "default")
        )
