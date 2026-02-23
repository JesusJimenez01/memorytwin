"""
Tests for MetaMemory and consolidation
=====================================
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from memorytwin.models import (
    Episode,
    MetaMemory,
    MetaMemorySearchResult,
    ReasoningTrace,
)


class TestMetaMemoryModel:
    """Tests for the MetaMemory model."""

    def test_create_minimal_meta_memory(self):
        """Create MetaMemory with minimal fields."""
        mm = MetaMemory(
            pattern="Test pattern",
            pattern_summary="Pattern summary"
        )

        assert mm.pattern == "Test pattern"
        assert mm.pattern_summary == "Pattern summary"
        assert mm.id is not None
        assert mm.created_at is not None
        assert mm.confidence == 0.5  # default
        assert mm.episode_count == 0

    def test_create_full_meta_memory(self):
        """Create MetaMemory with all fields."""
        source_ids = [uuid4(), uuid4(), uuid4()]

        mm = MetaMemory(
            pattern="Error handling pattern in REST APIs",
            pattern_summary="Use try-except with structured logging",
            lessons=["Always log the context", "Use appropriate HTTP codes"],
            best_practices=["Centralize error handling", "Use middleware"],
            antipatterns=["Catching generic Exception", "Silencing errors"],
            exceptions=["Validation errors are 400, not 500"],
            edge_cases=["Cascading timeouts"],
            contexts=["REST APIs", "Microservices"],
            technologies=["Python", "FastAPI", "SQLAlchemy"],
            source_episode_ids=source_ids,
            episode_count=3,
            confidence=0.85,
            coherence_score=0.9,
            project_name="test-project",
            tags=["api", "errors", "python"]
        )

        assert len(mm.lessons) == 2
        assert len(mm.best_practices) == 2
        assert len(mm.antipatterns) == 2
        assert len(mm.source_episode_ids) == 3
        assert mm.confidence == 0.85
        assert "FastAPI" in mm.technologies

    def test_meta_memory_defaults(self):
        """Verify default values."""
        mm = MetaMemory(
            pattern="Test",
            pattern_summary="Test"
        )

        assert mm.lessons == []
        assert mm.best_practices == []
        assert mm.antipatterns == []
        assert mm.exceptions == []
        assert mm.edge_cases == []
        assert mm.contexts == []
        assert mm.technologies == []
        assert mm.source_episode_ids == []
        assert mm.episode_count == 0
        assert mm.confidence == 0.5
        assert mm.coherence_score == 0.5
        assert mm.project_name == "default"
        assert mm.tags == []
        assert mm.access_count == 0
        assert mm.last_accessed is None

    def test_confidence_bounds(self):
        """Confidence should be between 0 and 1."""
        # Valid value
        mm = MetaMemory(
            pattern="Test",
            pattern_summary="Test",
            confidence=0.95
        )
        assert mm.confidence == 0.95

        # Out-of-range values should fail
        with pytest.raises(ValueError):
            MetaMemory(
                pattern="Test",
                pattern_summary="Test",
                confidence=1.5
            )

        with pytest.raises(ValueError):
            MetaMemory(
                pattern="Test",
                pattern_summary="Test",
                confidence=-0.1
            )

    def test_coherence_score_bounds(self):
        """Coherence score should be between 0 and 1."""
        mm = MetaMemory(
            pattern="Test",
            pattern_summary="Test",
            coherence_score=0.0
        )
        assert mm.coherence_score == 0.0

        mm2 = MetaMemory(
            pattern="Test",
            pattern_summary="Test",
            coherence_score=1.0
        )
        assert mm2.coherence_score == 1.0

    def test_access_count_non_negative(self):
        """Access count cannot be negative."""
        with pytest.raises(ValueError):
            MetaMemory(
                pattern="Test",
                pattern_summary="Test",
                access_count=-1
            )


class TestMetaMemorySearchResult:
    """Tests for MetaMemorySearchResult."""

    def test_create_search_result(self):
        """Create meta-memory search result."""
        mm = MetaMemory(
            pattern="Test pattern",
            pattern_summary="Test summary"
        )

        result = MetaMemorySearchResult(
            meta_memory=mm,
            relevance_score=0.85,
            match_reason="Consolidated pattern from 5 episodes"
        )

        assert result.meta_memory == mm
        assert result.relevance_score == 0.85
        assert "5 episodes" in result.match_reason

    def test_relevance_score_bounds(self):
        """Relevance score should be between 0 and 1."""
        mm = MetaMemory(pattern="Test", pattern_summary="Test")

        with pytest.raises(ValueError):
            MetaMemorySearchResult(
                meta_memory=mm,
                relevance_score=1.5
            )

        with pytest.raises(ValueError):
            MetaMemorySearchResult(
                meta_memory=mm,
                relevance_score=-0.1
            )


class TestConsolidationHelpers:
    """Tests for consolidation helper functions."""

    def test_format_episode_for_consolidation(self):
        """Verify episode formatting for consolidation."""
        from memorytwin.consolidation import format_episode_for_consolidation

        episode = Episode(
            id=uuid4(),
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            task="Implement JWT authentication",
            context="REST API with FastAPI",
            reasoning_trace=ReasoningTrace(
                raw_thinking="I considered OAuth2 but JWT is simpler for this case..."
            ),
            solution="Implement with pyjwt",
            solution_summary="Use pyjwt to generate tokens",
            lessons_learned=["Rotate keys regularly", "Use short expiration times"],
            tags=["auth", "jwt", "security"]
        )

        formatted = format_episode_for_consolidation(episode)

        # The compact format includes date, task, solution and lessons
        assert "Implement JWT authentication" in formatted
        assert "2024-01-15" in formatted
        assert "pyjwt" in formatted  # From the solution
        assert "Rotate keys regularly" in formatted  # From the lessons


class TestMetaMemoryTimestamps:
    """Tests for MetaMemory timestamp handling."""

    def test_created_at_auto_generated(self):
        """created_at is auto-generated."""
        before = datetime.now(timezone.utc)
        mm = MetaMemory(pattern="Test", pattern_summary="Test")
        after = datetime.now(timezone.utc)

        assert before <= mm.created_at <= after

    def test_updated_at_auto_generated(self):
        """updated_at is auto-generated."""
        mm = MetaMemory(pattern="Test", pattern_summary="Test")

        assert mm.updated_at is not None
        assert mm.created_at == mm.updated_at  # Initially equal

    def test_last_accessed_initially_none(self):
        """last_accessed is initially None."""
        mm = MetaMemory(pattern="Test", pattern_summary="Test")

        assert mm.last_accessed is None
