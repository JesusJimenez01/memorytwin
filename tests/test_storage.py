"""
Tests for the storage module
======================================
"""

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from memorytwin.models import (
    Episode,
    EpisodeType,
    MemoryQuery,
    ReasoningTrace,
)


@pytest.fixture
def temp_storage():
    """Create temporary storage for tests."""
    from memorytwin.escriba.storage import MemoryStorage

    # Use ignore_cleanup_errors=True to avoid Windows errors with ChromaDB
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        chroma_path = Path(tmpdir) / "chroma"
        sqlite_path = Path(tmpdir) / "test.db"

        storage = MemoryStorage(
            chroma_path=chroma_path,
            sqlite_path=sqlite_path
        )

        yield storage


@pytest.fixture
def sample_episode():
    """Create sample episode."""
    return Episode(
        task="Implement JWT authentication",
        context="REST API with FastAPI, PostgreSQL as DB",
        reasoning_trace=ReasoningTrace(
            raw_thinking="The user needs authentication. I considered various options: "
                        "sessions with Redis (discarded due to infrastructure), "
                        "OAuth2 (too complex), JWT (chosen for simplicity).",
            alternatives_considered=["Sessions with Redis", "Full OAuth2"],
            decision_factors=["Stateless", "Scalability", "Simplicity"],
            confidence_level=0.9
        ),
        solution="from jose import jwt\n\ndef create_token(user_id): ...",
        solution_summary="JWT with PyJWT, 24h tokens, 7-day refresh tokens",
        episode_type=EpisodeType.FEATURE,
        tags=["auth", "jwt", "security", "fastapi"],
        files_affected=["auth/jwt.py", "auth/routes.py"],
        lessons_learned=[
            "Always validate the JWT algorithm",
            "Use refresh tokens for better UX"
        ],
        source_assistant="copilot",
        project_name="test-api"
    )


class TestMemoryStorage:
    """Tests for MemoryStorage."""

    def test_store_and_retrieve_episode(self, temp_storage, sample_episode):
        """Test for storage and retrieval."""
        # Store
        episode_id = temp_storage.store_episode(sample_episode)

        assert episode_id is not None
        assert episode_id == str(sample_episode.id)

        # Retrieve
        retrieved = temp_storage.get_episode_by_id(episode_id)

        assert retrieved is not None
        assert retrieved.task == sample_episode.task
        assert retrieved.solution_summary == sample_episode.solution_summary
        assert retrieved.episode_type == sample_episode.episode_type

    def test_search_episodes(self, temp_storage, sample_episode):
        """Test for semantic search."""
        # Store
        temp_storage.store_episode(sample_episode)

        # Search
        query = MemoryQuery(
            query="JWT authentication",
            top_k=5
        )

        results = temp_storage.search_episodes(query)

        assert len(results) > 0
        assert results[0].episode.task == sample_episode.task
        assert results[0].relevance_score > 0

    def test_get_episodes_by_project(self, temp_storage, sample_episode):
        """Test for project filtering."""
        temp_storage.store_episode(sample_episode)

        # Search by correct project
        episodes = temp_storage.get_episodes_by_project("test-api")
        assert len(episodes) == 1

        # Search by incorrect project
        episodes = temp_storage.get_episodes_by_project("other-project")
        assert len(episodes) == 0

    def test_get_timeline(self, temp_storage, sample_episode):
        """Test for timeline retrieval."""
        temp_storage.store_episode(sample_episode)

        timeline = temp_storage.get_timeline(project_name="test-api")

        assert len(timeline) == 1
        assert timeline[0].task == sample_episode.task

    def test_get_lessons_learned(self, temp_storage, sample_episode):
        """Test for lessons retrieval."""
        temp_storage.store_episode(sample_episode)

        lessons = temp_storage.get_lessons_learned(project_name="test-api")

        assert len(lessons) == 2
        assert any("JWT" in item["lesson"] for item in lessons)

    def test_get_statistics(self, temp_storage, sample_episode):
        """Test for statistics."""
        temp_storage.store_episode(sample_episode)

        stats = temp_storage.get_statistics()

        assert stats["total_episodes"] == 1
        assert stats["by_type"]["feature"] == 1
        assert stats["by_assistant"]["copilot"] == 1

    def test_multiple_episodes(self, temp_storage):
        """Test with multiple episodes."""
        episodes = [
            Episode(
                task=f"Task {i}",
                context=f"Context {i}",
                reasoning_trace=ReasoningTrace(raw_thinking=f"Thinking {i}"),
                solution=f"Code {i}",
                solution_summary=f"Summary {i}",
                project_name="multi-test"
            )
            for i in range(5)
        ]

        for ep in episodes:
            temp_storage.store_episode(ep)

        stats = temp_storage.get_statistics("multi-test")
        assert stats["total_episodes"] == 5

    def test_delete_episode(self, temp_storage, sample_episode):
        """Test for episode deletion."""
        # Store
        episode_id = temp_storage.store_episode(sample_episode)

        # Verify it exists
        assert temp_storage.get_episode_by_id(episode_id) is not None

        # Delete
        success = temp_storage.delete_episode(episode_id)
        assert success is True

        # Verify it no longer exists
        assert temp_storage.get_episode_by_id(episode_id) is None

        # Try deleting again (should fail)
        success = temp_storage.delete_episode(episode_id)
        assert success is False

    def test_flags_are_persisted_on_store(self, temp_storage, sample_episode):
        """is_critical / is_antipattern / superseded_by survive a round trip."""
        replacement_id = uuid4()
        flagged = sample_episode.model_copy(update={
            "is_critical": True,
            "is_antipattern": True,
            "superseded_by": replacement_id,
            "deprecation_reason": "Replaced by OAuth2 flow",
        })

        episode_id = temp_storage.store_episode(flagged)
        retrieved = temp_storage.get_episode_by_id(episode_id)

        assert retrieved.is_critical is True
        assert retrieved.is_antipattern is True
        assert retrieved.superseded_by == replacement_id
        assert retrieved.deprecation_reason == "Replaced by OAuth2 flow"

    def test_update_episode_flags(self, temp_storage, sample_episode):
        """Flags can be set and cleared after storing."""
        episode_id = temp_storage.store_episode(sample_episode)

        assert temp_storage.update_episode_flags(episode_id, {"is_antipattern": True})
        assert temp_storage.get_episode_by_id(episode_id).is_antipattern is True

        assert temp_storage.update_episode_flags(episode_id, {"is_antipattern": False})
        assert temp_storage.get_episode_by_id(episode_id).is_antipattern is False

        assert temp_storage.update_episode_flags(str(uuid4()), {"is_critical": True}) is False

    def test_update_episode_flags_ignores_unknown_fields(self, temp_storage, sample_episode):
        """Only whitelisted fields can be modified."""
        episode_id = temp_storage.store_episode(sample_episode)

        temp_storage.update_episode_flags(episode_id, {"task": "hijacked", "is_critical": True})
        retrieved = temp_storage.get_episode_by_id(episode_id)

        assert retrieved.task == sample_episode.task
        assert retrieved.is_critical is True

    def test_statistics_filtered_by_project(self, temp_storage, sample_episode):
        """Per-project statistics only count that project's episodes."""
        temp_storage.store_episode(sample_episode)
        temp_storage.store_episode(Episode(
            task="Other task",
            context="Other context",
            reasoning_trace=ReasoningTrace(raw_thinking="Other thinking"),
            solution="",
            solution_summary="",
            episode_type=EpisodeType.BUG_FIX,
            source_assistant="claude",
            project_name="other-project"
        ))

        stats = temp_storage.get_statistics("test-api")

        assert stats["total_episodes"] == 1
        assert stats["by_type"]["feature"] == 1
        assert stats["by_type"]["bug_fix"] == 0
        assert stats["by_assistant"] == {"copilot": 1}

        global_stats = temp_storage.get_statistics()
        assert global_stats["total_episodes"] == 2
        assert global_stats["by_assistant"] == {"copilot": 1, "claude": 1}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
