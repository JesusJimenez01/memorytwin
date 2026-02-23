"""
Tests for the models module
============================
"""

from datetime import datetime
from uuid import UUID

import pytest

from memorytwin.models import (
    Episode,
    EpisodeType,
    MemoryQuery,
    ProcessedInput,
    ReasoningTrace,
)


class TestModels:
    """Tests for Pydantic models."""

    def test_reasoning_trace_creation(self):
        """Test for ReasoningTrace creation."""
        rt = ReasoningTrace(
            raw_thinking="I considered using JWT for its stateless nature",
            alternatives_considered=["Sessions", "OAuth2"],
            decision_factors=["Scalability", "Simplicity"],
            confidence_level=0.85
        )

        assert rt.raw_thinking == "I considered using JWT for its stateless nature"
        assert len(rt.alternatives_considered) == 2
        assert rt.confidence_level == 0.85

    def test_episode_creation(self):
        """Test for Episode creation."""
        episode = Episode(
            task="Implement authentication",
            context="REST API in FastAPI",
            reasoning_trace=ReasoningTrace(
                raw_thinking="I chose JWT for scalability"
            ),
            solution="from jose import jwt...",
            solution_summary="JWT con tokens de 24h",
            episode_type=EpisodeType.FEATURE,
            tags=["auth", "jwt"],
            project_name="test-project"
        )

        assert isinstance(episode.id, UUID)
        assert isinstance(episode.timestamp, datetime)
        assert episode.task == "Implement authentication"
        assert episode.episode_type == EpisodeType.FEATURE
        assert episode.success is True  # Default

    def test_episode_types(self):
        """Test for episode types."""
        assert EpisodeType.DECISION.value == "decision"
        assert EpisodeType.BUG_FIX.value == "bug_fix"
        assert EpisodeType.REFACTOR.value == "refactor"
        assert EpisodeType.FEATURE.value == "feature"

    def test_processed_input(self):
        """Test for ProcessedInput."""
        pi = ProcessedInput(
            raw_text="Thinking text here",
            user_prompt="Implement authentication",
            code_changes="def auth(): pass",
            source="clipboard"
        )

        assert pi.raw_text == "Thinking text here"
        assert pi.source == "clipboard"
        assert isinstance(pi.captured_at, datetime)

    def test_memory_query(self):
        """Test for MemoryQuery."""
        query = MemoryQuery(
            query="Why JWT?",
            project_filter="test-project",
            type_filter=EpisodeType.DECISION,
            top_k=3
        )

        assert query.query == "Why JWT?"
        assert query.top_k == 3

    def test_episode_json_serialization(self):
        """Test for Episode JSON serialization."""
        episode = Episode(
            task="Test task",
            context="Test context",
            reasoning_trace=ReasoningTrace(raw_thinking="Test thinking"),
            solution="code",
            solution_summary="summary"
        )

        json_str = episode.model_dump_json()
        assert "Test task" in json_str
        assert "Test thinking" in json_str


class TestProcessedInput:
    """Specific tests for ProcessedInput."""

    def test_minimal_input(self):
        """Test with minimal input."""
        pi = ProcessedInput(raw_text="Just text")

        assert pi.raw_text == "Just text"
        assert pi.user_prompt is None
        assert pi.code_changes is None
        assert pi.source == "manual"

    def test_full_input(self):
        """Test with complete input."""
        pi = ProcessedInput(
            raw_text="Complete model reasoning...",
            user_prompt="User prompt",
            code_changes="```python\ndef hello(): pass\n```",
            source="mcp"
        )

        assert pi.source == "mcp"
        assert "Complete" in pi.raw_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
