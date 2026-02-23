"""
Memory Storage - ChromaDB + SQLite
===================================

Manages dual storage:
- ChromaDB for vector search (embeddings)
- SQLite for metadata and structured queries
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
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from memorytwin.config import get_chroma_dir, get_settings, get_sqlite_path
from memorytwin.models import (
    Episode,
    EpisodeType,
    MemoryQuery,
    MemorySearchResult,
    MetaMemory,
    MetaMemorySearchResult,
    ReasoningTrace,
)
from memorytwin.scoring import compute_hybrid_score

Base = declarative_base()


class EpisodeRecord(Base):
    """SQLAlchemy model for memory episodes."""

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

    # ChromaDB embedding ID
    chroma_id = Column(String(100))

    # Forgetting Curve fields
    importance_score = Column(Float, default=1.0)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, nullable=True)

    # Active and useful memory fields
    is_antipattern = Column(Boolean, default=False, index=True)
    is_critical = Column(Boolean, default=False, index=True)
    superseded_by = Column(String(36), nullable=True)
    deprecation_reason = Column(Text, nullable=True)


class MetaMemoryRecord(Base):
    """SQLAlchemy model for consolidated meta-memories."""

    __tablename__ = "meta_memories"

    id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Identified pattern
    pattern = Column(Text, nullable=False)
    pattern_summary = Column(Text, nullable=False)

    # Consolidated knowledge (JSON arrays)
    lessons_json = Column(Text, default="[]")
    best_practices_json = Column(Text, default="[]")
    antipatterns_json = Column(Text, default="[]")

    # Exceptions and nuances
    exceptions_json = Column(Text, default="[]")
    edge_cases_json = Column(Text, default="[]")

    # Applicable contexts
    contexts_json = Column(Text, default="[]")
    technologies_json = Column(Text, default="[]")

    # Traceability
    source_episode_ids_json = Column(Text, default="[]")  # JSON array of UUIDs
    episode_count = Column(Integer, default=0)

    # Quality and confidence
    confidence = Column(Float, default=0.5)
    coherence_score = Column(Float, default=0.5)

    # Metadata
    project_name = Column(String(200), index=True)
    tags_json = Column(Text, default="[]")

    # Usage and relevance
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, nullable=True)

    # ChromaDB embedding ID
    chroma_id = Column(String(100))


class MemoryStorage:
    """
    Dual storage for episodic memories.
    Combines ChromaDB (vectors) and SQLite (metadata).
    """

    _embedder = None  # Singleton for lazy model loading
    _embedding_model_name = None

    def __init__(
        self,
        chroma_path: Optional[Path] = None,
        sqlite_path: Optional[Path] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize storage.

        Args:
            chroma_path: ChromaDB persistence directory
            sqlite_path: Path to the SQLite file
            embedding_model: Embedding model name
        """
        settings = get_settings()

        # Configure paths
        self.chroma_path = chroma_path or get_chroma_dir()
        self.sqlite_path = sqlite_path or get_sqlite_path()

        # Store model name for lazy loading
        MemoryStorage._embedding_model_name = embedding_model or settings.embedding_model

        # Initialize ChromaDB
        self._init_chroma()

        # Initialize SQLite
        self._init_sqlite()

    @property
    def embedder(self):
        """Lazy loading of the embedding model (loaded only when needed)."""
        if MemoryStorage._embedder is None:
            MemoryStorage._embedder = SentenceTransformer(
                MemoryStorage._embedding_model_name,
                device="cpu"
            )
        return MemoryStorage._embedder

    def _init_chroma(self):
        """Initialize ChromaDB client and collections."""
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Main collection for episodic memories
        self.collection = self.chroma_client.get_or_create_collection(
            name="memory_episodes",
            metadata={"description": "Memory Twin episodic memory episodes"}
        )

        # Collection for consolidated meta-memories
        self.meta_collection = self.chroma_client.get_or_create_collection(
            name="meta_memories",
            metadata={"description": "Memory Twin consolidated meta-memories"}
        )

    def _init_sqlite(self):
        """Initialize SQLite database."""
        engine = create_engine(f"sqlite:///{self.sqlite_path}")
        Base.metadata.create_all(engine)
        self.SessionLocal = sessionmaker(bind=engine)

    def _get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()

    def _generate_embedding(self, episode: Episode) -> list[float]:
        """
        Generate embedding combining task, context, and reasoning.
        """
        # Text to embed: combination of key elements
        text_parts = [
            f"Task: {episode.task}",
            f"Context: {episode.context}",
            f"Reasoning: {episode.reasoning_trace.raw_thinking}",
            f"Solution: {episode.solution_summary}",
        ]

        if episode.lessons_learned:
            text_parts.append(f"Lessons: {' '.join(episode.lessons_learned)}")

        combined_text = "\n".join(text_parts)

        # Generate embedding
        embedding = self.embedder.encode(combined_text).tolist()
        return embedding

    def store_episode(self, episode: Episode) -> str:
        """
        Store an episode in both databases.

        Args:
            episode: Episode to store

        Returns:
            ID of the stored episode
        """
        episode_id = str(episode.id)

        # Generate embedding
        embedding = self._generate_embedding(episode)

        # Store in ChromaDB
        self.collection.add(
            ids=[episode_id],
            embeddings=[embedding],
            metadatas=[{
                "task": episode.task[:500],  # Limit for metadata
                "episode_type": episode.episode_type.value,
                "project_name": episode.project_name,
                "source_assistant": episode.source_assistant,
                "timestamp": episode.timestamp.isoformat(),
                "tags": ",".join(episode.tags),
            }],
            documents=[episode.reasoning_trace.raw_thinking]
        )

        # Store in SQLite
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
                # Forgetting Curve fields
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
        Search for relevant episodes using vector search.

        Implements hybrid scoring that combines:
        - Semantic similarity (embeddings)
        - Temporal decay (forgetting curve)
        - Boost from frequent access
        - Base importance of the episode

        Args:
            query: Search query
            use_hybrid_scoring: If True, applies hybrid scoring (default: True)

        Returns:
            List of results ordered by hybrid relevance
        """

        # Generate query embedding
        query_embedding = self.embedder.encode(query.query).tolist()

        # Build ChromaDB filters
        where_filters = {}
        if query.project_filter:
            where_filters["project_name"] = query.project_filter
        if query.type_filter:
            where_filters["episode_type"] = query.type_filter.value

        # Request more results if using hybrid scoring (for re-ranking)
        n_results = query.top_k * 3 if use_hybrid_scoring else query.top_k

        # Vector search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filters if where_filters else None,
            include=["metadatas", "distances", "documents"]
        )

        # Convert results
        search_results = []

        if results["ids"] and results["ids"][0]:
            for i, episode_id in enumerate(results["ids"][0]):
                # Retrieve full episode from SQLite
                episode = self.get_episode_by_id(episode_id)
                if episode:
                    # Calculate base semantic score (ChromaDB uses L2 distance)
                    distance = results["distances"][0][i] if results["distances"] else 0
                    semantic_score = max(0, 1 - distance / 2)  # Normalize

                    # Apply hybrid scoring if enabled
                    if use_hybrid_scoring:
                        final_score = compute_hybrid_score(
                            episode=episode,
                            semantic_score=semantic_score,
                        )
                    else:
                        final_score = semantic_score

                    match_reason = (
                        "Semantic match with hybrid scoring" if use_hybrid_scoring
                        else "Semantic match"
                    )
                    search_results.append(MemorySearchResult(
                        episode=episode,
                        relevance_score=min(1.0, final_score),  # Normalize to max 1.0
                        match_reason=match_reason,
                    ))

        # Sort by hybrid score (descending) and limit results
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        final_results = search_results[:query.top_k]

        # Update access statistics for returned episodes
        for result in final_results:
            self.update_episode_access(str(result.episode.id))

        return final_results

    def update_episode_access(self, episode_id: str) -> tuple[bool, bool]:
        """
        Update access statistics of an episode.

        Increments access_count and updates last_accessed.
        Also checks if automatic consolidation should be triggered.

        Args:
            episode_id: ID of the episode to update

        Returns:
            Tuple (updated, needs_consolidation):
            - updated: True if successfully updated
            - needs_consolidation: True if consolidation is recommended
        """
        from memorytwin.scoring import CONSOLIDATION_ACCESS_THRESHOLD

        with self._get_session() as session:
            record = session.query(EpisodeRecord).filter(
                EpisodeRecord.id == episode_id
            ).first()

            if not record:
                return False, False

            # Increment access counter
            new_access_count = (record.access_count or 0) + 1
            record.access_count = new_access_count
            record.last_accessed = datetime.now(timezone.utc)

            session.commit()

            # Check if this episode indicates a need for consolidation
            needs_consolidation = new_access_count >= CONSOLIDATION_ACCESS_THRESHOLD

            return True, needs_consolidation

    def check_consolidation_needed(self, project_name: Optional[str] = None) -> dict:
        """
        Check if automatic consolidation is recommended.

        Analyzes:
        1. Episodes with high access_count ("hot" patterns)
        2. Total unconsolidated episodes

        Args:
            project_name: Filter by project (optional)

        Returns:
            Dict with recommendation and statistics
        """
        from memorytwin.scoring import (
            CONSOLIDATION_ACCESS_THRESHOLD,
            CONSOLIDATION_EPISODE_THRESHOLD,
            should_trigger_consolidation,
        )

        with self._get_session() as session:
            query = session.query(EpisodeRecord)

            if project_name:
                query = query.filter(EpisodeRecord.project_name == project_name)

            # Count total episodes
            total_episodes = query.count()

            # Find "hot" episodes (high access_count)
            hot_episodes = query.filter(
                EpisodeRecord.access_count >= CONSOLIDATION_ACCESS_THRESHOLD
            ).all()

            # Count existing meta-memories
            meta_query = session.query(MetaMemoryRecord)
            if project_name:
                meta_query = meta_query.filter(MetaMemoryRecord.project_name == project_name)
            total_meta_memories = meta_query.count()

            # Estimate consolidated episodes
            total_consolidated = 0
            if total_meta_memories > 0:
                metas = meta_query.all()
                for meta in metas:
                    total_consolidated += meta.episode_count

            unconsolidated = max(0, total_episodes - total_consolidated)

            # Determine whether to consolidate
            max_access = max([ep.access_count for ep in hot_episodes], default=0) if hot_episodes else 0
            should_consolidate = should_trigger_consolidation(max_access, unconsolidated)

            return {
                "should_consolidate": should_consolidate,
                "total_episodes": total_episodes,
                "hot_episodes_count": len(hot_episodes),
                "max_access_count": max_access,
                "total_meta_memories": total_meta_memories,
                "estimated_unconsolidated": unconsolidated,
                "thresholds": {
                    "access_threshold": CONSOLIDATION_ACCESS_THRESHOLD,
                    "episode_threshold": CONSOLIDATION_EPISODE_THRESHOLD
                },
                "hot_episode_ids": [ep.id for ep in hot_episodes[:5]]  # Top 5
            }

    def get_episode_by_id(self, episode_id: str) -> Optional[Episode]:
        """Retrieve an episode by its ID."""
        with self._get_session() as session:
            record = session.query(EpisodeRecord).filter(
                EpisodeRecord.id == episode_id
            ).first()

            if not record:
                return None

            return self._record_to_episode(record)

    def update_episode_flags(
        self,
        episode_id: str,
        updates: dict
    ) -> bool:
        """
        Update an episode's flags (is_antipattern, is_critical, etc).

        Args:
            episode_id: Episode UUID
            updates: Dict with fields to update

        Returns:
            True if successfully updated
        """
        with self._get_session() as session:
            record = session.query(EpisodeRecord).filter(
                EpisodeRecord.id == episode_id
            ).first()

            if not record:
                return False

            # Update allowed fields
            allowed_fields = {
                'is_antipattern', 'is_critical', 'superseded_by',
                'deprecation_reason', 'importance_score',
            }
            for field, value in updates.items():
                if field in allowed_fields and hasattr(record, field):
                    setattr(record, field, value)

            session.commit()

            # Also update ChromaDB metadata if antipattern
            if updates.get('is_antipattern') or updates.get('is_critical'):
                try:
                    self.collection.update(
                        ids=[episode_id],
                        metadatas=[{
                            "is_antipattern": str(updates.get('is_antipattern', False)),
                            "is_critical": str(updates.get('is_critical', False))
                        }]
                    )
                except Exception:
                    pass  # Not critical if ChromaDB fails

            return True

    def delete_episode(self, episode_id: str) -> bool:
        """
        Delete an episode from both databases.

        Args:
            episode_id: ID of the episode to delete

        Returns:
            True if successfully deleted, False if it didn't exist
        """
        try:
            # Delete from ChromaDB
            try:
                self.collection.delete(ids=[episode_id])
            except Exception:
                pass  # May not exist in ChromaDB

            # Delete from SQLite
            with self._get_session() as session:
                record = session.query(EpisodeRecord).filter(EpisodeRecord.id == episode_id).first()
                if record:
                    session.delete(record)
                    session.commit()
                    return True
                return False
        except Exception as e:
            print(f"Error deleting episode {episode_id}: {e}")
            return False

    def get_episodes_by_project(
        self,
        project_name: str,
        limit: int = 50
    ) -> list[Episode]:
        """Get episodes from a specific project."""
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
        Get episode timeline for visualization.
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
        Aggregate lessons learned from multiple episodes.
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

                # Filter by tags if specified
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

    def get_all_projects(self) -> list[str]:
        """Get list of all unique projects."""
        with self._get_session() as session:
            projects = session.query(EpisodeRecord.project_name).distinct().all()
            return sorted([p[0] for p in projects if p[0]])

    def get_statistics(self, project_name: Optional[str] = None) -> dict:
        """Get storage statistics."""
        with self._get_session() as session:
            query = session.query(EpisodeRecord)

            if project_name:
                query = query.filter(EpisodeRecord.project_name == project_name)

            total = query.count()

            # Count by type
            type_counts = {}
            for episode_type in EpisodeType:
                count = query.filter(
                    EpisodeRecord.episode_type == episode_type.value
                ).count()
                type_counts[episode_type.value] = count

            # Count by assistant (get unique values first)
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
        """Convert SQLite record to Episode."""
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
            # Forgetting Curve fields (with defaults for compatibility)
            importance_score=record.importance_score if record.importance_score is not None else 1.0,
            access_count=record.access_count if record.access_count is not None else 0,
            last_accessed=record.last_accessed,
            # Active memory fields (with defaults for compatibility)
            is_antipattern=getattr(record, 'is_antipattern', False) or False,
            is_critical=getattr(record, 'is_critical', False) or False,
            superseded_by=UUID(record.superseded_by) if getattr(record, 'superseded_by', None) else None,
            deprecation_reason=getattr(record, 'deprecation_reason', None)
        )

    # =========================================================================
    # META-MEMORY METHODS
    # =========================================================================

    def _generate_meta_embedding(self, meta_memory: MetaMemory) -> list[float]:
        """
        Generate embedding for a meta-memory.
        Combines pattern, lessons, and contexts.
        """
        text_parts = [
            f"Pattern: {meta_memory.pattern}",
            f"Summary: {meta_memory.pattern_summary}",
        ]

        if meta_memory.lessons:
            text_parts.append(f"Lessons: {' '.join(meta_memory.lessons)}")
        if meta_memory.best_practices:
            text_parts.append(f"Best practices: {' '.join(meta_memory.best_practices)}")
        if meta_memory.contexts:
            text_parts.append(f"Contexts: {' '.join(meta_memory.contexts)}")
        if meta_memory.technologies:
            text_parts.append(f"Technologies: {' '.join(meta_memory.technologies)}")

        combined_text = "\n".join(text_parts)
        embedding = self.embedder.encode(combined_text).tolist()
        return embedding

    def store_meta_memory(self, meta_memory: MetaMemory) -> str:
        """
        Store a meta-memory in both databases.

        Args:
            meta_memory: Meta-memory to store

        Returns:
            ID of the stored meta-memory
        """
        meta_id = str(meta_memory.id)

        # Generate embedding
        embedding = self._generate_meta_embedding(meta_memory)

        # Store in ChromaDB
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

        # Convert source_episode_ids to JSON
        source_ids_json = json.dumps([str(uid) for uid in meta_memory.source_episode_ids])

        # Store in SQLite
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
        Search for relevant meta-memories using vector search.

        Args:
            query: Search text
            project_name: Filter by project
            top_k: Number of results

        Returns:
            List of results ordered by relevance
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()

        # Build filters
        where_filters = {}
        if project_name:
            where_filters["project_name"] = project_name

        # Vector search
        results = self.meta_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filters if where_filters else None,
            include=["metadatas", "distances", "documents"]
        )

        # Convert results
        search_results = []

        if results["ids"] and results["ids"][0]:
            for i, meta_id in enumerate(results["ids"][0]):
                # Retrieve full meta-memory from SQLite
                meta_memory = self.get_meta_memory_by_id(meta_id)
                if meta_memory:
                    # Calculate score
                    distance = results["distances"][0][i] if results["distances"] else 0
                    relevance_score = max(0, 1 - distance / 2)

                    search_results.append(MetaMemorySearchResult(
                        meta_memory=meta_memory,
                        relevance_score=relevance_score,
                        match_reason=f"Consolidated pattern from {meta_memory.episode_count} episodes"
                    ))

        # Update access statistics
        for result in search_results:
            self.update_meta_memory_access(str(result.meta_memory.id))

        return search_results

    def get_meta_memory_by_id(self, meta_id: str) -> Optional[MetaMemory]:
        """Retrieve a meta-memory by its ID."""
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
        """Get meta-memories from a specific project."""
        with self._get_session() as session:
            records = session.query(MetaMemoryRecord).filter(
                MetaMemoryRecord.project_name == project_name
            ).order_by(MetaMemoryRecord.created_at.desc()).limit(limit).all()

            return [self._record_to_meta_memory(r) for r in records]

    def update_meta_memory_access(self, meta_id: str) -> bool:
        """Update access statistics of a meta-memory."""
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
        """Get meta-memory statistics."""
        with self._get_session() as session:
            query = session.query(MetaMemoryRecord)

            if project_name:
                query = query.filter(MetaMemoryRecord.project_name == project_name)

            total = query.count()

            # Total consolidated episodes
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
        """Convert SQLite record to MetaMemory."""
        # Parse source_episode_ids from JSON to UUID list
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
