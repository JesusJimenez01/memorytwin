"""
Almacenamiento de Memoria - ChromaDB + SQLite
=============================================

Gestiona el almacenamiento dual:
- ChromaDB para búsqueda vectorial (embeddings)
- SQLite para metadatos y consultas estructuradas
"""

import json
from datetime import datetime, timezone
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
    Integer,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from memorytwin.config import get_chroma_dir, get_settings, get_sqlite_path
from memorytwin.models import Episode, EpisodeType, MemoryQuery, MemorySearchResult, ReasoningTrace, MetaMemory, MetaMemorySearchResult
from memorytwin.scoring import compute_hybrid_score


Base = declarative_base()


class EpisodeRecord(Base):
    """Modelo SQLAlchemy para episodios de memoria."""
    
    __tablename__ = "episodes"
    
    id = Column(String(36), primary_key=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    
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
    
    # Campos para Forgetting Curve
    importance_score = Column(Float, default=1.0)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, nullable=True)


class MetaMemoryRecord(Base):
    """Modelo SQLAlchemy para meta-memorias consolidadas."""
    
    __tablename__ = "meta_memories"
    
    id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Patrón identificado
    pattern = Column(Text, nullable=False)
    pattern_summary = Column(Text, nullable=False)
    
    # Conocimiento consolidado (JSON arrays)
    lessons_json = Column(Text, default="[]")
    best_practices_json = Column(Text, default="[]")
    antipatterns_json = Column(Text, default="[]")
    
    # Excepciones y matices
    exceptions_json = Column(Text, default="[]")
    edge_cases_json = Column(Text, default="[]")
    
    # Contextos aplicables
    contexts_json = Column(Text, default="[]")
    technologies_json = Column(Text, default="[]")
    
    # Trazabilidad
    source_episode_ids_json = Column(Text, default="[]")  # JSON array of UUIDs
    episode_count = Column(Integer, default=0)
    
    # Calidad y confianza
    confidence = Column(Float, default=0.5)
    coherence_score = Column(Float, default=0.5)
    
    # Metadatos
    project_name = Column(String(200), index=True)
    tags_json = Column(Text, default="[]")
    
    # Uso y relevancia
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, nullable=True)
    
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
        """Inicializar cliente y colecciones de ChromaDB."""
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Colección principal de memorias episódicas
        self.collection = self.chroma_client.get_or_create_collection(
            name="memory_episodes",
            metadata={"description": "Episodios de memoria del Memory Twin"}
        )
        
        # Colección de meta-memorias consolidadas
        self.meta_collection = self.chroma_client.get_or_create_collection(
            name="meta_memories",
            metadata={"description": "Meta-memorias consolidadas del Memory Twin"}
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
                chroma_id=episode_id,
                # Campos para Forgetting Curve
                importance_score=episode.importance_score,
                access_count=episode.access_count,
                last_accessed=episode.last_accessed
            )
            session.add(record)
            session.commit()
            
        return episode_id
    
    def search_episodes(
        self,
        query: MemoryQuery,
        use_hybrid_scoring: bool = True
    ) -> list[MemorySearchResult]:
        """
        Buscar episodios relevantes usando búsqueda vectorial.
        
        Implementa scoring híbrido que combina:
        - Similitud semántica (embeddings)
        - Decaimiento temporal (forgetting curve)
        - Boost por uso frecuente
        - Importancia base del episodio
        
        Args:
            query: Consulta de búsqueda
            use_hybrid_scoring: Si True, aplica scoring híbrido (default: True)
            
        Returns:
            Lista de resultados ordenados por relevancia híbrida
        """
        from datetime import datetime, timezone
        
        # Generar embedding de la consulta
        query_embedding = self.embedder.encode(query.query).tolist()
        
        # Construir filtros para ChromaDB
        where_filters = {}
        if query.project_filter:
            where_filters["project_name"] = query.project_filter
        if query.type_filter:
            where_filters["episode_type"] = query.type_filter.value
        
        # Pedir más resultados si usamos hybrid scoring (para re-rankear)
        n_results = query.top_k * 3 if use_hybrid_scoring else query.top_k
        
        # Búsqueda vectorial
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filters if where_filters else None,
            include=["metadatas", "distances", "documents"]
        )
        
        # Convertir resultados
        search_results = []
        now = datetime.now(timezone.utc)
        
        if results["ids"] and results["ids"][0]:
            for i, episode_id in enumerate(results["ids"][0]):
                # Recuperar episodio completo de SQLite
                episode = self.get_episode_by_id(episode_id)
                if episode:
                    # Calcular score semántico base (ChromaDB usa distancia L2)
                    distance = results["distances"][0][i] if results["distances"] else 0
                    semantic_score = max(0, 1 - distance / 2)  # Normalizar
                    
                    # Aplicar scoring híbrido si está habilitado
                    if use_hybrid_scoring:
                        final_score = compute_hybrid_score(
                            episode=episode,
                            semantic_score=semantic_score,
                            now=now
                        )
                    else:
                        final_score = semantic_score
                    
                    search_results.append(MemorySearchResult(
                        episode=episode,
                        relevance_score=min(1.0, final_score),  # Normalizar a max 1.0
                        match_reason=f"Coincidencia semántica con scoring híbrido" if use_hybrid_scoring else "Coincidencia semántica"
                    ))
        
        # Ordenar por score híbrido (descendente) y limitar resultados
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        final_results = search_results[:query.top_k]
        
        # Actualizar estadísticas de acceso para los episodios devueltos
        for result in final_results:
            self.update_episode_access(str(result.episode.id))
        
        return final_results
    
    def update_episode_access(self, episode_id: str) -> bool:
        """
        Actualizar estadísticas de acceso de un episodio.
        
        Incrementa access_count y actualiza last_accessed.
        Esto es parte de la implementación de la forgetting curve.
        
        Args:
            episode_id: ID del episodio a actualizar
            
        Returns:
            True si se actualizó correctamente, False si no existe
        """
        from datetime import datetime, timezone
        
        with self._get_session() as session:
            record = session.query(EpisodeRecord).filter(
                EpisodeRecord.id == episode_id
            ).first()
            
            if not record:
                return False
            
            # Incrementar contador de accesos
            record.access_count = (record.access_count or 0) + 1
            record.last_accessed = datetime.now(timezone.utc)
            
            session.commit()
            return True
    
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
            project_name=record.project_name,
            # Campos para Forgetting Curve (con defaults para compatibilidad)
            importance_score=record.importance_score if record.importance_score is not None else 1.0,
            access_count=record.access_count if record.access_count is not None else 0,
            last_accessed=record.last_accessed
        )

    # =========================================================================
    # MÉTODOS PARA META-MEMORIAS
    # =========================================================================
    
    def _generate_meta_embedding(self, meta_memory: MetaMemory) -> list[float]:
        """
        Generar embedding para una meta-memoria.
        Combina patrón, lecciones y contextos.
        """
        text_parts = [
            f"Patrón: {meta_memory.pattern}",
            f"Resumen: {meta_memory.pattern_summary}",
        ]
        
        if meta_memory.lessons:
            text_parts.append(f"Lecciones: {' '.join(meta_memory.lessons)}")
        if meta_memory.best_practices:
            text_parts.append(f"Mejores prácticas: {' '.join(meta_memory.best_practices)}")
        if meta_memory.contexts:
            text_parts.append(f"Contextos: {' '.join(meta_memory.contexts)}")
        if meta_memory.technologies:
            text_parts.append(f"Tecnologías: {' '.join(meta_memory.technologies)}")
            
        combined_text = "\n".join(text_parts)
        embedding = self.embedder.encode(combined_text).tolist()
        return embedding
    
    def store_meta_memory(self, meta_memory: MetaMemory) -> str:
        """
        Almacenar una meta-memoria en ambas bases de datos.
        
        Args:
            meta_memory: Meta-memoria a almacenar
            
        Returns:
            ID de la meta-memoria almacenada
        """
        meta_id = str(meta_memory.id)
        
        # Generar embedding
        embedding = self._generate_meta_embedding(meta_memory)
        
        # Almacenar en ChromaDB
        self.meta_collection.add(
            ids=[meta_id],
            embeddings=[embedding],
            metadatas=[{
                "pattern_summary": meta_memory.pattern_summary[:500],
                "project_name": meta_memory.project_name,
                "episode_count": meta_memory.episode_count,
                "confidence": meta_memory.confidence,
                "created_at": meta_memory.created_at.isoformat(),
                "tags": ",".join(meta_memory.tags),
            }],
            documents=[meta_memory.pattern]
        )
        
        # Convertir source_episode_ids a JSON
        source_ids_json = json.dumps([str(uid) for uid in meta_memory.source_episode_ids])
        
        # Almacenar en SQLite
        with self._get_session() as session:
            record = MetaMemoryRecord(
                id=meta_id,
                created_at=meta_memory.created_at,
                updated_at=meta_memory.updated_at,
                pattern=meta_memory.pattern,
                pattern_summary=meta_memory.pattern_summary,
                lessons_json=json.dumps(meta_memory.lessons),
                best_practices_json=json.dumps(meta_memory.best_practices),
                antipatterns_json=json.dumps(meta_memory.antipatterns),
                exceptions_json=json.dumps(meta_memory.exceptions),
                edge_cases_json=json.dumps(meta_memory.edge_cases),
                contexts_json=json.dumps(meta_memory.contexts),
                technologies_json=json.dumps(meta_memory.technologies),
                source_episode_ids_json=source_ids_json,
                episode_count=meta_memory.episode_count,
                confidence=meta_memory.confidence,
                coherence_score=meta_memory.coherence_score,
                project_name=meta_memory.project_name,
                tags_json=json.dumps(meta_memory.tags),
                access_count=meta_memory.access_count,
                last_accessed=meta_memory.last_accessed,
                chroma_id=meta_id
            )
            session.add(record)
            session.commit()
            
        return meta_id
    
    def search_meta_memories(
        self,
        query: str,
        project_name: Optional[str] = None,
        top_k: int = 5
    ) -> list[MetaMemorySearchResult]:
        """
        Buscar meta-memorias relevantes usando búsqueda vectorial.
        
        Args:
            query: Texto de búsqueda
            project_name: Filtrar por proyecto
            top_k: Número de resultados
            
        Returns:
            Lista de resultados ordenados por relevancia
        """
        # Generar embedding de la consulta
        query_embedding = self.embedder.encode(query).tolist()
        
        # Construir filtros
        where_filters = {}
        if project_name:
            where_filters["project_name"] = project_name
        
        # Búsqueda vectorial
        results = self.meta_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filters if where_filters else None,
            include=["metadatas", "distances", "documents"]
        )
        
        # Convertir resultados
        search_results = []
        
        if results["ids"] and results["ids"][0]:
            for i, meta_id in enumerate(results["ids"][0]):
                # Recuperar meta-memoria completa de SQLite
                meta_memory = self.get_meta_memory_by_id(meta_id)
                if meta_memory:
                    # Calcular score
                    distance = results["distances"][0][i] if results["distances"] else 0
                    relevance_score = max(0, 1 - distance / 2)
                    
                    search_results.append(MetaMemorySearchResult(
                        meta_memory=meta_memory,
                        relevance_score=relevance_score,
                        match_reason=f"Patrón consolidado de {meta_memory.episode_count} episodios"
                    ))
        
        # Actualizar estadísticas de acceso
        for result in search_results:
            self.update_meta_memory_access(str(result.meta_memory.id))
        
        return search_results
    
    def get_meta_memory_by_id(self, meta_id: str) -> Optional[MetaMemory]:
        """Recuperar una meta-memoria por su ID."""
        with self._get_session() as session:
            record = session.query(MetaMemoryRecord).filter(
                MetaMemoryRecord.id == meta_id
            ).first()
            
            if not record:
                return None
                
            return self._record_to_meta_memory(record)
    
    def get_meta_memories_by_project(
        self,
        project_name: str,
        limit: int = 50
    ) -> list[MetaMemory]:
        """Obtener meta-memorias de un proyecto específico."""
        with self._get_session() as session:
            records = session.query(MetaMemoryRecord).filter(
                MetaMemoryRecord.project_name == project_name
            ).order_by(MetaMemoryRecord.created_at.desc()).limit(limit).all()
            
            return [self._record_to_meta_memory(r) for r in records]
    
    def update_meta_memory_access(self, meta_id: str) -> bool:
        """Actualizar estadísticas de acceso de una meta-memoria."""
        with self._get_session() as session:
            record = session.query(MetaMemoryRecord).filter(
                MetaMemoryRecord.id == meta_id
            ).first()
            
            if not record:
                return False
            
            record.access_count = (record.access_count or 0) + 1
            record.last_accessed = datetime.now(timezone.utc)
            
            session.commit()
            return True
    
    def get_meta_memory_statistics(self, project_name: Optional[str] = None) -> dict:
        """Obtener estadísticas de meta-memorias."""
        with self._get_session() as session:
            query = session.query(MetaMemoryRecord)
            
            if project_name:
                query = query.filter(MetaMemoryRecord.project_name == project_name)
                
            total = query.count()
            
            # Episodios totales consolidados
            total_episodes = 0
            avg_confidence = 0.0
            
            if total > 0:
                records = query.all()
                total_episodes = sum(r.episode_count for r in records)
                avg_confidence = sum(r.confidence for r in records) / total
                
            return {
                "total_meta_memories": total,
                "total_episodes_consolidated": total_episodes,
                "average_confidence": round(avg_confidence, 3),
                "chroma_count": self.meta_collection.count()
            }
    
    def _record_to_meta_memory(self, record: MetaMemoryRecord) -> MetaMemory:
        """Convertir registro SQLite a MetaMemory."""
        # Parsear source_episode_ids de JSON a lista de UUIDs
        source_ids = [UUID(uid) for uid in json.loads(record.source_episode_ids_json)]
        
        return MetaMemory(
            id=UUID(record.id),
            created_at=record.created_at,
            updated_at=record.updated_at,
            pattern=record.pattern,
            pattern_summary=record.pattern_summary,
            lessons=json.loads(record.lessons_json),
            best_practices=json.loads(record.best_practices_json),
            antipatterns=json.loads(record.antipatterns_json),
            exceptions=json.loads(record.exceptions_json),
            edge_cases=json.loads(record.edge_cases_json),
            contexts=json.loads(record.contexts_json),
            technologies=json.loads(record.technologies_json),
            source_episode_ids=source_ids,
            episode_count=record.episode_count,
            confidence=record.confidence,
            coherence_score=record.coherence_score,
            project_name=record.project_name,
            tags=json.loads(record.tags_json),
            access_count=record.access_count if record.access_count is not None else 0,
            last_accessed=record.last_accessed
        )
