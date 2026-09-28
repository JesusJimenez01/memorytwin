"""
Memory Consolidation
====================

Implements the consolidation process that groups related episodes
into meta-memories, following an approach inspired by human memory
consolidation during sleep.

Process:
1. Groups similar episodes using embedding-based clustering
2. For each cluster, uses an LLM to synthesize the knowledge
3. Generates a MetaMemory with patterns, lessons, and exceptions
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import numpy as np
from sklearn.cluster import DBSCAN

from memorytwin.config import get_llm_model, parse_json_response
from memorytwin.escriba.storage import MemoryStorage
from memorytwin.models import Episode, MetaMemory
from memorytwin.observability import trace_consolidation

logger = logging.getLogger(__name__)


# Prompt for synthesizing episodes into a meta-memory (optimized for speed)
CONSOLIDATION_PROMPT = """Synthesize these technical memory episodes into a consolidated meta-memory.

EPISODES:
{episodes_text}

Respond ONLY in JSON:
{{
    "pattern": "Common pattern identified (1-2 sentences)",
    "pattern_summary": "Summary in 1 short sentence",
    "lessons": ["lesson 1", "lesson 2"],
    "best_practices": ["practice 1"],
    "antipatterns": ["antipattern 1"],
    "technologies": ["tech1", "tech2"],
    "coherence_score": 0.8
}}
"""


def format_episode_for_consolidation(episode: Episode) -> str:
    """Format an episode for the consolidation prompt (compact version)."""
    reasoning = episode.reasoning_trace.raw_thinking[:200] if episode.reasoning_trace.raw_thinking else ""
    lessons = ', '.join(episode.lessons_learned[:2]) if episode.lessons_learned else 'N/A'

    return f"""[{episode.timestamp.strftime('%Y-%m-%d')}] {episode.task}
Reasoning: {reasoning}
Solution: {episode.solution_summary[:100]}
Lessons: {lessons}"""


class MemoryConsolidator:
    """
    Consolidates related episodes into meta-memories.

    Uses embedding-based clustering to group similar episodes
    and an LLM to synthesize the consolidated knowledge.
    """

    def __init__(
        self,
        storage: Optional[MemoryStorage] = None,
        min_cluster_size: int = 3,
        cluster_eps: float = 0.4,
        max_episodes_per_cluster: int = 8,
    ):
        """
        Initialize the consolidator.

        Args:
            storage: Memory storage instance
            min_cluster_size: Minimum episodes to form a cluster
            cluster_eps: Maximum radius for DBSCAN clustering (lower = stricter)
            max_episodes_per_cluster: Max episodes per cluster to limit prompt size
        """
        self.storage = storage or MemoryStorage()

        # Use centralized factory (low temperature, reduced tokens for speed)
        self.model = get_llm_model(temperature=0.2, max_output_tokens=1024)

        self.min_cluster_size = min_cluster_size
        self.cluster_eps = cluster_eps
        self.max_episodes_per_cluster = max_episodes_per_cluster

    def consolidate_project(
        self,
        project_name: str,
        force: bool = False
    ) -> list[MetaMemory]:
        """
        Consolidate a project's episodes into meta-memories.

        Args:
            project_name: Project name
            force: If True, reconsolidate even already-consolidated episodes

        Returns:
            List of generated meta-memories
        """
        logger.info("Starting consolidation for project: %s", project_name)

        # Get project episodes
        episodes = self.storage.get_episodes_by_project(project_name, limit=200)
        logger.info("Found %d episodes", len(episodes))

        # Skip episodes that already feed a meta-memory, so repeated runs
        # don't produce duplicate meta-memories for the same cluster
        if not force:
            consolidated_ids = self._get_consolidated_episode_ids(project_name)
            episodes = [ep for ep in episodes if ep.id not in consolidated_ids]
            logger.info("%d episodes pending consolidation", len(episodes))

        if len(episodes) < self.min_cluster_size:
            logger.info("Insufficient episodes (%d < %d)", len(episodes), self.min_cluster_size)
            return []

        # Get embeddings from ChromaDB
        embeddings, episode_ids = self._get_episode_embeddings(episodes)
        logger.info("Retrieved %d embeddings", len(embeddings))

        if len(embeddings) < self.min_cluster_size:
            logger.info("Insufficient embeddings")
            return []

        # Clustering
        clusters = self._cluster_episodes(embeddings, episode_ids)
        logger.info("Generated %d clusters", len(clusters))

        # Generate meta-memories for each cluster
        meta_memories = []
        for i, cluster_episode_ids in enumerate(clusters, 1):
            logger.info("Processing cluster %d/%d (%d episodes)", i, len(clusters), len(cluster_episode_ids))

            # Get cluster episodes
            cluster_ids = set(cluster_episode_ids)
            cluster_episodes = [ep for ep in episodes if str(ep.id) in cluster_ids]

            # Limit episodes per cluster to avoid huge prompts
            if len(cluster_episodes) > self.max_episodes_per_cluster:
                # Select the most recent ones
                cluster_episodes = sorted(
                    cluster_episodes,
                    key=lambda e: e.timestamp,
                    reverse=True
                )[:self.max_episodes_per_cluster]
                logger.info("Cluster limited to %d most recent episodes", self.max_episodes_per_cluster)

            if len(cluster_episodes) >= self.min_cluster_size:
                logger.info("Synthesizing cluster %d with LLM...", i)
                meta_memory = self._synthesize_cluster(
                    cluster_episodes,
                    project_name
                )
                if meta_memory:
                    self.storage.store_meta_memory(meta_memory)
                    meta_memories.append(meta_memory)
                    logger.info("Meta-memory %d created: %s", i, meta_memory.pattern_summary[:50])

        logger.info("Consolidation completed: %d meta-memories generated", len(meta_memories))
        return meta_memories

    def _get_consolidated_episode_ids(self, project_name: str) -> set:
        """IDs of the episodes already summarized by an existing meta-memory."""
        existing = self.storage.get_meta_memories_by_project(project_name, limit=1000)
        return {episode_id for meta in existing for episode_id in meta.source_episode_ids}

    def _get_episode_embeddings(
        self,
        episodes: list[Episode]
    ) -> tuple[np.ndarray, list[str]]:
        """Get episode embeddings from ChromaDB."""
        episode_ids = [str(ep.id) for ep in episodes]

        # Get embeddings from ChromaDB
        result = self.storage.collection.get(
            ids=episode_ids,
            include=["embeddings"]
        )

        if result["embeddings"] is None or len(result["embeddings"]) == 0:
            return np.array([]), []

        embeddings = np.array(result["embeddings"])
        valid_ids = result["ids"]

        return embeddings, valid_ids

    def _cluster_episodes(
        self,
        embeddings: np.ndarray,
        episode_ids: list[str]
    ) -> list[list[str]]:
        """
        Group episodes by similarity using DBSCAN.

        DBSCAN is ideal because:
        - It does not require specifying the number of clusters
        - It can detect arbitrarily shaped clusters
        - It identifies outliers (unique episodes)
        """
        # Normalize embeddings for cosine distance
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)

        # DBSCAN with cosine distance
        clustering = DBSCAN(
            eps=self.cluster_eps,
            min_samples=self.min_cluster_size,
            metric='cosine'
        ).fit(normalized)

        # Group IDs by cluster label
        clusters: dict[int, list[str]] = {}
        for idx, label in enumerate(clustering.labels_):
            if label == -1:  # Outlier
                continue
            clusters.setdefault(label, []).append(episode_ids[idx])

        return list(clusters.values())

    @trace_consolidation
    def _synthesize_cluster(
        self,
        episodes: list[Episode],
        project_name: str
    ) -> Optional[MetaMemory]:
        """
        Use LLM to synthesize a cluster of episodes.

        Args:
            episodes: Cluster episodes
            project_name: Project name

        Returns:
            Generated MetaMemory or None on failure
        """
        # Format episodes for the prompt
        episodes_text = "\n---\n".join(
            format_episode_for_consolidation(ep) for ep in episodes
        )

        prompt = CONSOLIDATION_PROMPT.format(episodes_text=episodes_text)

        try:
            # Call the LLM (unified interface)
            response = self.model.generate(prompt)
            data = parse_json_response(response.text)

            # Calculate confidence based on number of episodes
            # More episodes = higher confidence (up to a point)
            confidence = min(0.95, 0.5 + (len(episodes) * 0.1))

            # Create MetaMemory
            now = datetime.now(timezone.utc)
            meta_memory = MetaMemory(
                id=uuid4(),
                created_at=now,
                updated_at=now,
                pattern=data.get("pattern", "Pattern not identified"),
                pattern_summary=data.get("pattern_summary", ""),
                lessons=data.get("lessons", []),
                best_practices=data.get("best_practices", []),
                antipatterns=data.get("antipatterns", []),
                exceptions=data.get("exceptions", []),
                edge_cases=data.get("edge_cases", []),
                contexts=data.get("contexts", []),
                technologies=data.get("technologies", []),
                source_episode_ids=[ep.id for ep in episodes],
                episode_count=len(episodes),
                confidence=confidence,
                coherence_score=self._clamp_score(data.get("coherence_score", 0.5)),
                project_name=project_name,
                tags=self._extract_common_tags(episodes)
            )

            return meta_memory

        except ValueError as e:
            logger.warning("Could not parse LLM consolidation response: %s", e)
            return None
        except Exception as e:
            logger.error("Error synthesizing cluster: %s", e)
            return None

    @staticmethod
    def _clamp_score(value, default: float = 0.5) -> float:
        """Coerce an LLM-provided score into the [0, 1] range."""
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return default

    def _extract_common_tags(self, episodes: list[Episode]) -> list[str]:
        """Extract common tags across episodes."""
        if not episodes:
            return []

        # Count tag frequency
        tag_counts: dict[str, int] = {}
        for ep in episodes:
            for tag in ep.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Return tags that appear in at least 50% of episodes
        threshold = len(episodes) / 2
        common_tags = [
            tag for tag, count in tag_counts.items()
            if count >= threshold
        ]

        return common_tags


def consolidate_memories(
    project_name: str,
    min_cluster_size: int = 3,
    storage: Optional[MemoryStorage] = None,
    force: bool = False,
) -> list[MetaMemory]:
    """
    Convenience function to consolidate a project's memories.

    Args:
        project_name: Project name to consolidate
        min_cluster_size: Minimum episodes per cluster
        storage: Storage instance (optional)
        force: Re-consolidate episodes already covered by a meta-memory

    Returns:
        List of generated meta-memories
    """
    consolidator = MemoryConsolidator(
        storage=storage,
        min_cluster_size=min_cluster_size
    )

    return consolidator.consolidate_project(project_name, force=force)
