"""
Tests for memory consolidation
==============================

Unit tests for MemoryConsolidator with mocked storage and LLM.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from memorytwin.consolidation import MemoryConsolidator
from memorytwin.models import Episode, MetaMemory, ReasoningTrace


def _episode(task: str, tags: list[str] | None = None) -> Episode:
    return Episode(
        task=task,
        context="API service",
        reasoning_trace=ReasoningTrace(raw_thinking=f"Thinking about {task}"),
        solution="...",
        solution_summary=f"Solved {task}",
        tags=tags or [],
        lessons_learned=[f"Lesson from {task}"],
        project_name="demo",
    )


@pytest.fixture
def mock_llm():
    with patch("memorytwin.consolidation.get_llm_model") as factory:
        model = MagicMock()
        factory.return_value = model
        yield model


@pytest.fixture
def episodes():
    return [_episode(f"Retry policy {i}", tags=["http", "retry"]) for i in range(4)]


@pytest.fixture
def storage(episodes):
    storage = MagicMock()
    storage.get_episodes_by_project.return_value = episodes
    storage.get_meta_memories_by_project.return_value = []
    # Nearly identical embeddings -> a single dense cluster
    storage.collection.get.side_effect = lambda ids, include: {
        "ids": ids,
        "embeddings": [[1.0, 0.0, 0.01 * i] for i in range(len(ids))],
    }
    return storage


LLM_JSON = """```json
{"pattern": "Retries with exponential backoff", "pattern_summary": "Use backoff",
 "lessons": ["Cap the number of retries"], "best_practices": ["Add jitter"],
 "antipatterns": ["Retrying non-idempotent calls"], "technologies": ["httpx"],
 "coherence_score": 1.7}
```"""


class TestMemoryConsolidator:
    """Tests for MemoryConsolidator."""

    def test_consolidates_cluster_into_meta_memory(self, mock_llm, storage, episodes):
        mock_llm.generate.return_value = MagicMock(text=LLM_JSON)

        consolidator = MemoryConsolidator(storage=storage, min_cluster_size=3)
        metas = consolidator.consolidate_project("demo")

        assert len(metas) == 1
        meta = metas[0]
        assert meta.pattern_summary == "Use backoff"
        assert meta.episode_count == len(episodes)
        assert set(meta.source_episode_ids) == {ep.id for ep in episodes}
        assert meta.tags == ["http", "retry"]
        # Out-of-range LLM score is clamped instead of failing validation
        assert meta.coherence_score == 1.0
        storage.store_meta_memory.assert_called_once_with(meta)

    def test_skips_already_consolidated_episodes(self, mock_llm, storage, episodes):
        """Re-running consolidation must not duplicate meta-memories."""
        storage.get_meta_memories_by_project.return_value = [
            MetaMemory(
                pattern="p",
                pattern_summary="p",
                source_episode_ids=[ep.id for ep in episodes[:2]],
                episode_count=2,
                project_name="demo",
            )
        ]

        consolidator = MemoryConsolidator(storage=storage, min_cluster_size=3)
        metas = consolidator.consolidate_project("demo")

        # Only 2 pending episodes remain, below the minimum cluster size
        assert metas == []
        mock_llm.generate.assert_not_called()

    def test_force_reconsolidates_everything(self, mock_llm, storage, episodes):
        storage.get_meta_memories_by_project.return_value = [
            MetaMemory(
                pattern="p",
                pattern_summary="p",
                source_episode_ids=[ep.id for ep in episodes],
                episode_count=len(episodes),
                project_name="demo",
            )
        ]
        mock_llm.generate.return_value = MagicMock(text=LLM_JSON)

        consolidator = MemoryConsolidator(storage=storage, min_cluster_size=3)
        metas = consolidator.consolidate_project("demo", force=True)

        assert len(metas) == 1

    def test_invalid_llm_response_yields_no_meta_memory(self, mock_llm, storage):
        mock_llm.generate.return_value = MagicMock(text="I cannot help with that")

        consolidator = MemoryConsolidator(storage=storage, min_cluster_size=3)

        assert consolidator.consolidate_project("demo") == []
        storage.store_meta_memory.assert_not_called()

    def test_outliers_are_not_clustered(self, mock_llm):
        consolidator = MemoryConsolidator(storage=MagicMock(), min_cluster_size=2)
        embeddings = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])

        clusters = consolidator._cluster_episodes(embeddings, ["a", "b", "c"])

        assert clusters == [["a", "b"]]

    @pytest.mark.parametrize("raw, expected", [(0.8, 0.8), (-1, 0.0), (3, 1.0), ("0.4", 0.4), ("high", 0.5)])
    def test_clamp_score(self, raw, expected):
        assert MemoryConsolidator._clamp_score(raw) == expected

    def test_common_tags_require_half_of_episodes(self, mock_llm):
        consolidator = MemoryConsolidator(storage=MagicMock())
        eps = [_episode("a", ["x", "y"]), _episode("b", ["x"]), _episode("c", ["z"])]

        assert consolidator._extract_common_tags(eps) == ["x"]
        assert consolidator._extract_common_tags([]) == []
