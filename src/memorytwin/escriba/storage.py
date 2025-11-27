"""
Almacenamiento de Memoria - ChromaDB + SQLite
=============================================

Gestiona el almacenamiento dual:
- ChromaDB para búsqueda vectorial (embeddings)
- SQLite para metadatos y consultas estructuradas
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from sqlalchemy import (
    Column,
    DateTime,
    String,
    Text,
    Boolean,
    Float,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from memorytwin.config import get_chroma_dir, get_settings, get_sqlite_path
from memorytwin.models import Episode, EpisodeType, MemoryQuery, MemorySearchResult, ReasoningTrace


Base = declarative_base()


class EpisodeRecord(Base):
    """Modelo SQLAlchemy para episodios de memoria."""
    
    __tablename__ = "episodes"
    
    id = Column(String(36), primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    task = Column(Text, nullable=False)
    context = Column(Text, nullable=False)
    reasoning_trace_json = Column(Text, nullable=False)
    
    solution = Column(Text)
    solution_summary = Column(Text)
    
    outcome = Column(Text)
    success = Column(Boolean, default=True)
    
    episode_type = Column(String(50), index=True)
    tags_json = Column(Text)  # JSON array
    files_affected_json = Column(Text)  # JSON array
    lessons_learned_json = Column(Text)  # JSON array
    
    source_assistant = Column(String(100), index=True)
    project_name = Column(String(200), index=True)
    
    # Embedding ID en ChromaDB
    chroma_id = Column(String(100))


class MemoryStorage:
    """
    Almacenamiento dual de memorias episódicas.
    Combina ChromaDB (vectores) y SQLite (metadatos).
    """
    
    def __init__(
        self,
        chroma_path: Optional[Path] = None,
        sqlite_path: Optional[Path] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Inicializar almacenamiento.
        
        Args:
            chroma_path: Directorio de persistencia de ChromaDB
            sqlite_path: Path al archivo SQLite
            embedding_model: Nombre del modelo de embeddings
        """
        settings = get_settings()
        
        # Configurar paths
        self.chroma_path = chroma_path or get_chroma_dir()
        self.sqlite_path = sqlite_path or get_sqlite_path()
        
        # Inicializar modelo de embeddings con device explícito
        model_name = embedding_model or settings.embedding_model
        self.embedder = SentenceTransformer(model_name, device="cpu")
        
        # Inicializar ChromaDB
        self._init_chroma()
        
        # Inicializar SQLite
        self._init_sqlite()
        
    def _init_chroma(self):
        """Inicializar cliente y colección de ChromaDB."""
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Colección principal de memorias
        self.collection = self.chroma_client.get_or_create_collection(
            name="memory_episodes",
            metadata={"description": "Episodios de memoria del Memory Twin"}
        )
        
    def _init_sqlite(self):
        """Inicializar base de datos SQLite."""
        engine = create_engine(f"sqlite:///{self.sqlite_path}")
        Base.metadata.create_all(engine)
        self.SessionLocal = sessionmaker(bind=engine)
        
    def _get_session(self) -> Session:
        """Obtener sesión de base de datos."""
        return self.SessionLocal()
    
    def _generate_embedding(self, episode: Episode) -> list[float]:
        """
        Generar embedding combinando tarea, contexto y razonamiento.
        """
        # Texto a embeber: combinación de elementos clave
        text_parts = [
            f"Tarea: {episode.task}",
            f"Contexto: {episode.context}",
            f"Razonamiento: {episode.reasoning_trace.raw_thinking}",
            f"Solución: {episode.solution_summary}",
        ]
        
        if episode.lessons_learned:
            text_parts.append(f"Lecciones: {' '.join(episode.lessons_learned)}")
            
        combined_text = "\n".join(text_parts)
        
        # Generar embedding
        embedding = self.embedder.encode(combined_text).tolist()
        return embedding
    
    def store_episode(self, episode: Episode) -> str:
        """
        Almacenar un episodio en ambas bases de datos.
        
        Args:
            episode: Episodio a almacenar
            
        Returns:
            ID del episodio almacenado
        """
        episode_id = str(episode.id)
        
        # Generar embedding
        embedding = self._generate_embedding(episode)
        
        # Almacenar en ChromaDB
        self.collection.add(
            ids=[episode_id],
            embeddings=[embedding],
            metadatas=[{
                "task": episode.task[:500],  # Limitar para metadatos
                "episode_type": episode.episode_type.value,
                "project_name": episode.project_name,
                "source_assistant": episode.source_assistant,
                "timestamp": episode.timestamp.isoformat(),
                "tags": ",".join(episode.tags),
            }],
            documents=[episode.reasoning_trace.raw_thinking]
        )
        
        # Almacenar en SQLite
        with self._get_session() as session:
            record = EpisodeRecord(
                id=episode_id,
                timestamp=episode.timestamp,
                task=episode.task,
                context=episode.context,
                reasoning_trace_json=episode.reasoning_trace.model_dump_json(),
                solution=episode.solution,
                solution_summary=episode.solution_summary,
                outcome=episode.outcome,
                success=episode.success,
                episode_type=episode.episode_type.value,
                tags_json=json.dumps(episode.tags),
                files_affected_json=json.dumps(episode.files_affected),
                lessons_learned_json=json.dumps(episode.lessons_learned),
                source_assistant=episode.source_assistant,
                project_name=episode.project_name,
                chroma_id=episode_id
            )
            session.add(record)
            session.commit()
            
        return episode_id
    
    def search_episodes(
        self,
        query: MemoryQuery
    ) -> list[MemorySearchResult]:
        """
        Buscar episodios relevantes usando búsqueda vectorial.
        
        Args:
            query: Consulta de búsqueda
            
        Returns:
            Lista de resultados ordenados por relevancia
        """
        # Generar embedding de la consulta
        query_embedding = self.embedder.encode(query.query).tolist()
        
        # Construir filtros para ChromaDB
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
            include=["metadatas", "distances", "documents"]
        )
        
        # Convertir resultados
        search_results = []
        
        if results["ids"] and results["ids"][0]:
            for i, episode_id in enumerate(results["ids"][0]):
                # Recuperar episodio completo de SQLite
                episode = self.get_episode_by_id(episode_id)
                if episode:
                    # Calcular score de relevancia (ChromaDB usa distancia, convertir a similaridad)
                    distance = results["distances"][0][i] if results["distances"] else 0
                    relevance_score = max(0, 1 - distance / 2)  # Normalizar
                    
                    search_results.append(MemorySearchResult(
                        episode=episode,
                        relevance_score=relevance_score,
                        match_reason=f"Coincidencia semántica con la consulta"
                    ))
                    
        return search_results
    
    def get_episode_by_id(self, episode_id: str) -> Optional[Episode]:
        """Recuperar un episodio por su ID."""
        with self._get_session() as session:
            record = session.query(EpisodeRecord).filter(
                EpisodeRecord.id == episode_id
            ).first()
            
            if not record:
                return None
                
            return self._record_to_episode(record)
    
    def get_episodes_by_project(
        self,
        project_name: str,
        limit: int = 50
    ) -> list[Episode]:
        """Obtener episodios de un proyecto específico."""
        with self._get_session() as session:
            records = session.query(EpisodeRecord).filter(
                EpisodeRecord.project_name == project_name
            ).order_by(EpisodeRecord.timestamp.desc()).limit(limit).all()
            
            return [self._record_to_episode(r) for r in records]
    
    def get_timeline(
        self,
        project_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> list[Episode]:
        """
        Obtener timeline de episodios para visualización.
        """
        with self._get_session() as session:
            query = session.query(EpisodeRecord)
            
            if project_name:
                query = query.filter(EpisodeRecord.project_name == project_name)
            if start_date:
                query = query.filter(EpisodeRecord.timestamp >= start_date)
            if end_date:
                query = query.filter(EpisodeRecord.timestamp <= end_date)
                
            records = query.order_by(
                EpisodeRecord.timestamp.desc()
            ).limit(limit).all()
            
            return [self._record_to_episode(r) for r in records]
    
    def get_lessons_learned(
        self,
        project_name: Optional[str] = None,
        tags: Optional[list[str]] = None
    ) -> list[dict]:
        """
        Agregar lecciones aprendidas de múltiples episodios.
        """
        with self._get_session() as session:
            query = session.query(EpisodeRecord).filter(
                EpisodeRecord.lessons_learned_json != "[]"
            )
            
            if project_name:
                query = query.filter(EpisodeRecord.project_name == project_name)
                
            records = query.order_by(EpisodeRecord.timestamp.desc()).all()
            
            lessons = []
            for record in records:
                record_lessons = json.loads(record.lessons_learned_json)
                record_tags = json.loads(record.tags_json)
                
                # Filtrar por tags si se especifican
                if tags and not any(t in record_tags for t in tags):
                    continue
                    
                for lesson in record_lessons:
                    lessons.append({
                        "lesson": lesson,
                        "from_task": record.task,
                        "timestamp": record.timestamp,
                        "tags": record_tags,
                        "episode_id": record.id
                    })
                    
            return lessons
    
    def get_statistics(self, project_name: Optional[str] = None) -> dict:
        """Obtener estadísticas del almacenamiento."""
        with self._get_session() as session:
            query = session.query(EpisodeRecord)
            
            if project_name:
                query = query.filter(EpisodeRecord.project_name == project_name)
                
            total = query.count()
            
            # Contar por tipo
            type_counts = {}
            for episode_type in EpisodeType:
                count = query.filter(
                    EpisodeRecord.episode_type == episode_type.value
                ).count()
                type_counts[episode_type.value] = count
                
            # Contar por asistente (obtener valores únicos primero)
            assistant_counts = {}
            assistants = session.query(EpisodeRecord.source_assistant).distinct().all()
            for (assistant,) in assistants:
                if project_name:
                    count = session.query(EpisodeRecord).filter(
                        EpisodeRecord.project_name == project_name,
                        EpisodeRecord.source_assistant == assistant
                    ).count()
                else:
                    count = session.query(EpisodeRecord).filter(
                        EpisodeRecord.source_assistant == assistant
                    ).count()
                assistant_counts[assistant] = count
                
            return {
                "total_episodes": total,
                "by_type": type_counts,
                "by_assistant": assistant_counts,
                "chroma_count": self.collection.count()
            }
    
    def _record_to_episode(self, record: EpisodeRecord) -> Episode:
        """Convertir registro SQLite a Episode."""
        reasoning_data = json.loads(record.reasoning_trace_json)
        
        return Episode(
            id=UUID(record.id),
            timestamp=record.timestamp,
            task=record.task,
            context=record.context,
            reasoning_trace=ReasoningTrace(**reasoning_data),
            solution=record.solution or "",
            solution_summary=record.solution_summary or "",
            outcome=record.outcome,
            success=record.success,
            episode_type=EpisodeType(record.episode_type),
            tags=json.loads(record.tags_json),
            files_affected=json.loads(record.files_affected_json),
            lessons_learned=json.loads(record.lessons_learned_json),
            source_assistant=record.source_assistant,
            project_name=record.project_name
        )
