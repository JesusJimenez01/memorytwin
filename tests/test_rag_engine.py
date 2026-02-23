"""
Tests for the Oracle RAG engine
===================================

Unit tests for RAGEngine with mocks.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from memorytwin.models import (
    Episode,
    EpisodeType,
    MemorySearchResult,
    ReasoningTrace,
)
from memorytwin.oraculo.rag_engine import ORACLE_SYSTEM_PROMPT, RAGEngine


class TestRAGEngine:
    """Tests for RAGEngine."""

    @pytest.fixture
    def mock_llm_model(self):
        """Mock for get_llm_model factory."""
        with patch("memorytwin.oraculo.rag_engine.get_llm_model") as mock:
            mock_model = MagicMock()
            mock.return_value = mock_model
            yield mock, mock_model

    @pytest.fixture
    def mock_storage(self):
        """Mock for storage."""
        storage = MagicMock()
        return storage

    @pytest.fixture
    def sample_episode(self):
        """Sample episode."""
        return Episode(
            id=uuid4(),
            task="Implement JWT authentication",
            context="REST API with FastAPI",
            reasoning_trace=ReasoningTrace(
                raw_thinking="I chose JWT for scalability",
                alternatives_considered=["Sessions", "OAuth2"],
                decision_factors=["Stateless", "Escalabilidad"]
            ),
            solution="from jose import jwt",
            solution_summary="JWT with 24h tokens",
            episode_type=EpisodeType.FEATURE,
            tags=["auth", "jwt"],
            files_affected=["auth.py"],
            lessons_learned=["Validate JWT algorithm"],
            project_name="test-project",
            source_assistant="copilot"
        )

    @pytest.fixture
    def sample_search_result(self, sample_episode):
        """Sample search result."""
        return MemorySearchResult(
            episode=sample_episode,
            relevance_score=0.92
        )

    def test_rag_engine_init(self, mock_llm_model, mock_storage):
        """Test for RAGEngine initialization."""
        mock_factory, mock_model = mock_llm_model

        engine = RAGEngine(storage=mock_storage)

        # Verify that the factory was called
        mock_factory.assert_called_once()
        assert engine.storage == mock_storage
        assert engine.model == mock_model

    def test_rag_engine_init_no_api_key_raises(self, mock_storage):
        """Test that initialization fails without API key in config."""
        with patch("memorytwin.oraculo.rag_engine.get_llm_model") as mock:
            mock.side_effect = ValueError("GOOGLE_API_KEY is required")

            with pytest.raises(ValueError, match="GOOGLE_API_KEY is required"):
                RAGEngine(storage=mock_storage)

    def test_build_context(self, mock_llm_model, mock_storage, sample_search_result):
        """Test for context construction."""
        engine = RAGEngine(storage=mock_storage)

        context = engine._build_context([sample_search_result])

        assert "## RELEVANT MEMORY EPISODES" in context
        assert "Episode 1" in context
        assert "Relevance: 92%" in context
        assert "Implement JWT authentication" in context
        assert "JWT for scalability" in context
        assert "Sessions" in context
        assert "Stateless" in context
        assert "Validate JWT algorithm" in context
        assert "auth" in context

    def test_build_context_multiple_episodes(
        self, mock_llm_model, mock_storage, sample_episode
    ):
        """Test for context with multiple episodes."""
        episode2 = Episode(
            id=uuid4(),
            task="Add rate limiting",
            context="API protection",
            reasoning_trace=ReasoningTrace(raw_thinking="Rate limit for security"),
            solution="rate_limit()",
            solution_summary="Rate limiting implemented",
            episode_type=EpisodeType.FEATURE,
            project_name="test"
        )

        results = [
            MemorySearchResult(episode=sample_episode, relevance_score=0.95),
            MemorySearchResult(episode=episode2, relevance_score=0.78)
        ]

        engine = RAGEngine(storage=mock_storage)
        context = engine._build_context(results)

        assert "Episode 1" in context
        assert "Episode 2" in context
        assert "95%" in context
        assert "78%" in context
        assert "JWT authentication" in context
        assert "rate limiting" in context

    @pytest.mark.asyncio
    async def test_query_no_results(self, mock_llm_model, mock_storage):
        """Test for query with no results."""
        mock_storage.search_episodes.return_value = []
        mock_storage.search_meta_memories.return_value = []

        engine = RAGEngine(storage=mock_storage)

        result = await engine.query("Why did we use GraphQL?")

        assert "I found no" in result["answer"]
        assert result["episodes_used"] == []
        assert result["meta_memories_used"] == []
        assert result["context_provided"] is False

    @pytest.mark.asyncio
    async def test_query_with_results(
        self, mock_llm_model, mock_storage, sample_search_result
    ):
        """Test for query with results."""
        mock_factory, mock_model = mock_llm_model

        mock_storage.search_episodes.return_value = [sample_search_result]

        # Model mock
        mock_response = MagicMock()
        mock_response.text = "JWT fue elegido por su naturaleza stateless y escalabilidad."
        mock_model.generate_async = AsyncMock(return_value=mock_response)

        engine = RAGEngine(storage=mock_storage)

        result = await engine.query(
            question="Why did we use JWT?",
            project_name="test-project",
            top_k=3
        )

        assert result["context_provided"] is True
        assert len(result["episodes_used"]) == 1
        assert result["relevance_scores"][0] == 0.92
        assert "JWT" in result["answer"]

        # Verify storage call
        mock_storage.search_episodes.assert_called_once()
        call_args = mock_storage.search_episodes.call_args
        query = call_args.args[0]
        assert query.query == "Why did we use JWT?"
        assert query.project_filter == "test-project"
        assert query.top_k == 3

    def test_query_sync(
        self, mock_llm_model, mock_storage, sample_search_result
    ):
        """Test for synchronous query version."""
        mock_factory, mock_model = mock_llm_model

        mock_storage.search_episodes.return_value = [sample_search_result]

        mock_response = MagicMock()
        mock_response.text = "Response about JWT"
        mock_model.generate_async = AsyncMock(return_value=mock_response)

        engine = RAGEngine(storage=mock_storage)

        result = engine.query_sync("Why JWT?")

        assert result["context_provided"] is True

    def test_get_timeline(self, mock_llm_model, mock_storage, sample_episode):
        """Test for timeline retrieval."""
        mock_storage.get_timeline.return_value = [sample_episode]

        engine = RAGEngine(storage=mock_storage)

        timeline = engine.get_timeline(project_name="test", limit=10)

        assert len(timeline) == 1
        assert timeline[0]["task"] == "Implement JWT authentication"
        assert timeline[0]["type"] == "feature"
        assert "date" in timeline[0]
        assert "time" in timeline[0]
        assert timeline[0]["assistant"] == "copilot"

        mock_storage.get_timeline.assert_called_once_with(
            project_name="test",
            limit=10
        )

    def test_get_lessons(self, mock_llm_model, mock_storage):
        """Test for lessons retrieval."""
        mock_lessons = [
            {"lesson": "Validar algoritmos JWT", "from_task": "Auth"},
            {"lesson": "Usar rate limiting", "from_task": "Security"}
        ]
        mock_storage.get_lessons_learned.return_value = mock_lessons

        engine = RAGEngine(storage=mock_storage)

        lessons = engine.get_lessons(project_name="test", tags=["security"])

        assert len(lessons) == 2
        mock_storage.get_lessons_learned.assert_called_once_with(
            project_name="test",
            tags=["security"]
        )

    def test_get_statistics(self, mock_llm_model, mock_storage):
        """Test for statistics retrieval."""
        mock_stats = {
            "total_episodes": 25,
            "by_type": {"feature": 15, "bug_fix": 10}
        }
        mock_storage.get_statistics.return_value = mock_stats

        engine = RAGEngine(storage=mock_storage)

        stats = engine.get_statistics(project_name="test")

        assert stats["total_episodes"] == 25
        mock_storage.get_statistics.assert_called_once_with("test")


class TestOracleSystemPrompt:
    """Tests for the Oracle system prompt."""

    def test_prompt_contains_role_description(self):
        """Test that the prompt describes the role."""
        assert "Oracle" in ORACLE_SYSTEM_PROMPT
        assert "Memory Twin" in ORACLE_SYSTEM_PROMPT

    def test_prompt_mentions_episodic_memory(self):
        """Test that it mentions episodes and meta-memories."""
        assert "EPISODES" in ORACLE_SYSTEM_PROMPT
        assert "META-MEMORIES" in ORACLE_SYSTEM_PROMPT

    def test_prompt_has_instructions(self):
        """Test that it has instructions."""
        assert "INSTRUCTIONS" in ORACLE_SYSTEM_PROMPT
        # Updated: no longer mentions "ONLY" as it now includes meta-memories
        assert "Prioritize META-MEMORIES" in ORACLE_SYSTEM_PROMPT

    def test_prompt_mentions_format(self):
        """Test that it mentions format."""
        assert "Markdown" in ORACLE_SYSTEM_PROMPT


class TestRAGEngineEdgeCases:
    """Tests for RAGEngine edge cases."""

    @pytest.fixture
    def mock_llm_model(self):
        """Mock for get_llm_model factory."""
        with patch("memorytwin.oraculo.rag_engine.get_llm_model") as mock:
            mock_model = MagicMock()
            mock.return_value = mock_model
            yield mock, mock_model

    def test_build_context_empty_fields(self, mock_llm_model):
        """Test for context with empty fields."""
        episode = Episode(
            id=uuid4(),
            task="Simple task",
            context="Basic context",
            reasoning_trace=ReasoningTrace(
                raw_thinking="Thinking",
                alternatives_considered=[],
                decision_factors=[]
            ),
            solution="",
            solution_summary="",
            episode_type=EpisodeType.DECISION,
            tags=[],
            lessons_learned=[],
            project_name="test"
        )

        result = MemorySearchResult(episode=episode, relevance_score=0.5)

        mock_storage = MagicMock()
        engine = RAGEngine(storage=mock_storage)

        context = engine._build_context([result])

        assert "Not documented" in context or "None documented" in context

    def test_timeline_formatting(self, mock_llm_model):
        """Test for timeline formatting."""
        episode = Episode(
            id=uuid4(),
            task="Test task",
            context="Test context",
            reasoning_trace=ReasoningTrace(raw_thinking="test"),
            solution="fix",
            solution_summary="Bug fixed",
            episode_type=EpisodeType.BUG_FIX,
            project_name="test",
            success=False
        )

        mock_storage = MagicMock()
        mock_storage.get_timeline.return_value = [episode]

        engine = RAGEngine(storage=mock_storage)
        timeline = engine.get_timeline()

        assert timeline[0]["success"] is False
        assert timeline[0]["type"] == "bug_fix"
        assert "id" in timeline[0]
        assert "timestamp" in timeline[0]
