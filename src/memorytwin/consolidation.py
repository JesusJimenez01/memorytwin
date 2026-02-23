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

import json
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import numpy as np
from sklearn.cluster import DBSCAN

from memorytwin.config import get_llm_model
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
        logger.info(f"Starting consolidation for project: {project_name}")

        # Get project episodes
        episodes = self.storage.get_episodes_by_project(project_name, limit=200)
        logger.info(f"Found {len(episodes)} episodes")

        if len(episodes) < self.min_cluster_size:
            logger.info(f"Insufficient episodes ({len(episodes)} < {self.min_cluster_size})")
            return []

        # Get embeddings from ChromaDB
        embeddings, episode_ids = self._get_episode_embeddings(episodes)
        logger.info(f"Retrieved {len(embeddings)} embeddings")

        if len(embeddings) < self.min_cluster_size:
            logger.info("Insufficient embeddings")
            return []

        # Clustering
        clusters = self._cluster_episodes(embeddings, episode_ids)
        logger.info(f"Generated {len(clusters)} clusters")

        # Generate meta-memories for each cluster
        meta_memories = []
        for i, cluster_episode_ids in enumerate(clusters):
            logger.info(f"Processing cluster {i+1}/{len(clusters)} ({len(cluster_episode_ids)} episodes)")

            # Get cluster episodes
            cluster_episodes = [
                ep for ep in episodes
                if str(ep.id) in cluster_episode_ids
            ]

            # Limit episodes per cluster to avoid huge prompts
            if len(cluster_episodes) > self.max_episodes_per_cluster:
                # Select the most recent ones
                cluster_episodes = sorted(
                    cluster_episodes,
                    key=lambda e: e.timestamp,
                    reverse=True
                )[:self.max_episodes_per_cluster]
                logger.info(f"Cluster limited to {self.max_episodes_per_cluster} most recent episodes")

            if len(cluster_episodes) >= self.min_cluster_size:
                logger.info(f"Synthesizing cluster {i+1} with LLM...")
                meta_memory = self._synthesize_cluster(
                    cluster_episodes,
                    project_name
                )
                if meta_memory:
                    # Store
                    self.storage.store_meta_memory(meta_memory)
                    meta_memories.append(meta_memory)
                    logger.info(f"Meta-memory {i+1} created: {meta_memory.pattern_summary[:50]}...")

        logger.info(f"Consolidation completed: {len(meta_memories)} meta-memories generated")
        return meta_memories

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
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label == -1:  # Outlier
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(episode_ids[idx])

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

            # Parse JSON response
            response_text = response.text.strip()

            # Clean possible code markers
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            data = json.loads(response_text.strip())

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
                coherence_score=data.get("coherence_score", 0.5),
                project_name=project_name,
                tags=self._extract_common_tags(episodes)
            )

            return meta_memory

        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {e}")
            return None
        except Exception as e:
            print(f"Error in synthesis: {e}")
            return None

    def _extract_common_tags(self, episodes: list[Episode]) -> list[str]:
        """Extract common tags across episodes."""
        if not episodes:
            return []

        # Count tag frequency
        tag_counts = {}
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
    storage: Optional[MemoryStorage] = None
) -> list[MetaMemory]:
    """
    Convenience function to consolidate a project's memories.

    Args:
        project_name: Project name to consolidate
        min_cluster_size: Minimum episodes per cluster
        storage: Storage instance (optional)

    Returns:
        List of generated meta-memories
    """
    consolidator = MemoryConsolidator(
        storage=storage,
        min_cluster_size=min_cluster_size
    )

    return consolidator.consolidate_project(project_name)
