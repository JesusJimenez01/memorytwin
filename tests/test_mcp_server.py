"""
Tests for the MCP server
==========================

Unit tests for MemoryTwinMCPServer.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from memorytwin.models import (
    Episode,
    EpisodeType,
    MemorySearchResult,
    ReasoningTrace,
)


class TestMCPServerHelpers:
    """Tests for MCP server helper functions."""

    def test_format_lessons_with_datetime(self):
        """Test for lessons formatting with datetime."""
        from memorytwin.mcp_server.server import _format_lessons

        lessons = [
            {
                "lesson": "Test lesson",
                "timestamp": datetime(2025, 1, 15, 10, 30),
                "from_task": "Test task"
            },
            {
                "lesson": "Another lesson",
                "timestamp": datetime(2025, 1, 16, 14, 0),
                "tags": ["tag1", "tag2"]
            }
        ]

        result = _format_lessons(lessons)

        assert len(result) == 2
        assert result[0]["timestamp"] == "2025-01-15T10:30:00"
        assert result[1]["timestamp"] == "2025-01-16T14:00:00"
        assert result[0]["lesson"] == "Test lesson"
        assert result[1]["tags"] == ["tag1", "tag2"]

    def test_format_lessons_without_datetime(self):
        """Test for lessons formatting without datetime."""
        from memorytwin.mcp_server.server import _format_lessons

        lessons = [
            {
                "lesson": "Simple lesson",
                "count": 5
            }
        ]

        result = _format_lessons(lessons)

        assert result[0]["lesson"] == "Simple lesson"
        assert result[0]["count"] == 5

    def test_format_lessons_empty(self):
        """Test for empty list formatting."""
        from memorytwin.mcp_server.server import _format_lessons

        result = _format_lessons([])

        assert result == []


class TestMCPServerInit:
    """Tests for MCP server initialization."""

    @patch("memorytwin.mcp_server.server.Server")
    def test_server_initialization(self, mock_server_class):
        """Test for server initialization."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mcp_server = MemoryTwinMCPServer()

        mock_server_class.assert_called_once_with("memorytwin")
        assert mcp_server.processor is None  # Lazy init
        assert mcp_server.storage is None
        assert mcp_server.rag_engine is None

    @patch("memorytwin.mcp_server.server.Server")
    @patch("memorytwin.mcp_server.server.ThoughtProcessor")
    @patch("memorytwin.mcp_server.server.MemoryStorage")
    @patch("memorytwin.mcp_server.server.RAGEngine")
    def test_lazy_init(
        self,
        mock_rag,
        mock_storage,
        mock_processor,
        mock_server_class
    ):
        """Test for lazy initialization of components."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        storage_instance = MagicMock()
        mock_storage.return_value = storage_instance

        mcp_server = MemoryTwinMCPServer()

        # Before lazy_init
        assert mcp_server.processor is None

        # Call lazy_init
        mcp_server._lazy_init()

        # After lazy_init
        mock_processor.assert_called_once()
        mock_storage.assert_called_once()
        mock_rag.assert_called_once_with(storage=storage_instance)

        assert mcp_server.processor is not None
        assert mcp_server.storage is not None
        assert mcp_server.rag_engine is not None

    @patch("memorytwin.mcp_server.server.Server")
    @patch("memorytwin.mcp_server.server.ThoughtProcessor")
    @patch("memorytwin.mcp_server.server.MemoryStorage")
    @patch("memorytwin.mcp_server.server.RAGEngine")
    def test_lazy_init_only_once(
        self,
        mock_rag,
        mock_storage,
        mock_processor,
        mock_server_class
    ):
        """Test that lazy_init only executes once."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mcp_server = MemoryTwinMCPServer()

        # Call multiple times
        mcp_server._lazy_init()
        mcp_server._lazy_init()
        mcp_server._lazy_init()

        # Should only initialize once
        assert mock_processor.call_count == 1
        assert mock_storage.call_count == 1


class TestMCPServerTools:
    """Tests for MCP server tools."""

    @pytest.fixture
    def sample_episode(self):
        """Sample episode."""
        return Episode(
            id=uuid4(),
            task="Implement JWT authentication",
            context="REST API with FastAPI",
            reasoning_trace=ReasoningTrace(
                raw_thinking="I chose JWT for scalability",
                alternatives_considered=["Sessions"],
                decision_factors=["Stateless"]
            ),
            solution="from jose import jwt",
            solution_summary="JWT with 24h tokens",
            episode_type=EpisodeType.FEATURE,
            tags=["auth", "jwt"],
            lessons_learned=["Validate algorithm"],
            project_name="test-project",
            source_assistant="copilot"
        )

    @patch("memorytwin.mcp_server.server.Server")
    @patch("memorytwin.mcp_server.server.ThoughtProcessor")
    @patch("memorytwin.mcp_server.server.MemoryStorage")
    @patch("memorytwin.mcp_server.server.RAGEngine")
    @pytest.mark.asyncio
    async def test_get_statistics_tool(
        self,
        mock_rag,
        mock_storage_class,
        mock_processor,
        mock_server_class
    ):
        """Test for get_statistics tool."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_storage = MagicMock()
        mock_storage.get_statistics.return_value = {
            "total_episodes": 15,
            "by_type": {"feature": 10, "bug_fix": 5}
        }
        mock_storage_class.return_value = mock_storage

        mcp_server = MemoryTwinMCPServer()
        mcp_server._lazy_init()

        result = await mcp_server._get_statistics({"project_name": "test"})

        assert result.isError is False
        mock_storage.get_statistics.assert_called_once_with("test")

    @patch("memorytwin.mcp_server.server.Server")
    @patch("memorytwin.mcp_server.server.ThoughtProcessor")
    @patch("memorytwin.mcp_server.server.MemoryStorage")
    @patch("memorytwin.mcp_server.server.RAGEngine")
    @pytest.mark.asyncio
    async def test_get_timeline_tool(
        self,
        mock_rag_class,
        mock_storage_class,
        mock_processor,
        mock_server_class,
        sample_episode
    ):
        """Test for get_timeline tool."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_storage = MagicMock()
        mock_storage_class.return_value = mock_storage

        mock_rag = MagicMock()
        mock_rag.get_timeline.return_value = [
            {
                "id": str(sample_episode.id),
                "task": sample_episode.task,
                "type": "feature"
            }
        ]
        mock_rag_class.return_value = mock_rag

        mcp_server = MemoryTwinMCPServer()
        mcp_server._lazy_init()

        result = await mcp_server._get_timeline({"limit": 10})

        assert result.isError is False
        mock_rag.get_timeline.assert_called_once()

    @patch("memorytwin.mcp_server.server.Server")
    @patch("memorytwin.mcp_server.server.ThoughtProcessor")
    @patch("memorytwin.mcp_server.server.MemoryStorage")
    @patch("memorytwin.mcp_server.server.RAGEngine")
    @pytest.mark.asyncio
    async def test_get_lessons_tool(
        self,
        mock_rag_class,
        mock_storage_class,
        mock_processor,
        mock_server_class
    ):
        """Test for get_lessons tool."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_storage = MagicMock()
        mock_storage.get_lessons_learned.return_value = [
            {
                "lesson": "Test lesson",
                "from_task": "Test task",
                "timestamp": datetime.now()
            }
        ]
        mock_storage_class.return_value = mock_storage

        mcp_server = MemoryTwinMCPServer()
        mcp_server._lazy_init()

        result = await mcp_server._get_lessons({"project_name": "test"})

        assert result.isError is False

    @patch("memorytwin.mcp_server.server.Server")
    @patch("memorytwin.mcp_server.server.ThoughtProcessor")
    @patch("memorytwin.mcp_server.server.MemoryStorage")
    @patch("memorytwin.mcp_server.server.RAGEngine")
    @pytest.mark.asyncio
    async def test_get_episode_tool(
        self,
        mock_rag_class,
        mock_storage_class,
        mock_processor,
        mock_server_class,
        sample_episode
    ):
        """Test for get_episode tool."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_storage = MagicMock()
        mock_storage.get_episode_by_id.return_value = sample_episode
        mock_storage_class.return_value = mock_storage

        mcp_server = MemoryTwinMCPServer()
        mcp_server._lazy_init()

        result = await mcp_server._get_episode({"episode_id": str(sample_episode.id)})

        assert result.isError is False
        mock_storage.get_episode_by_id.assert_called_once_with(str(sample_episode.id))

    @patch("memorytwin.mcp_server.server.Server")
    @patch("memorytwin.mcp_server.server.ThoughtProcessor")
    @patch("memorytwin.mcp_server.server.MemoryStorage")
    @patch("memorytwin.mcp_server.server.RAGEngine")
    @pytest.mark.asyncio
    async def test_get_episode_not_found(
        self,
        mock_rag_class,
        mock_storage_class,
        mock_processor,
        mock_server_class
    ):
        """Test for get_episode tool when it doesn't exist."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_storage = MagicMock()
        mock_storage.get_episode_by_id.return_value = None
        mock_storage_class.return_value = mock_storage

        mcp_server = MemoryTwinMCPServer()
        mcp_server._lazy_init()

        result = await mcp_server._get_episode({"episode_id": "non-existent-id"})

        # Should indicate it was not found
        content = result.content[0].text
        assert "not found" in content.lower()

    @patch("memorytwin.mcp_server.server.Server")
    @patch("memorytwin.mcp_server.server.ThoughtProcessor")
    @patch("memorytwin.mcp_server.server.MemoryStorage")
    @patch("memorytwin.mcp_server.server.RAGEngine")
    @pytest.mark.asyncio
    async def test_search_episodes_tool(
        self,
        mock_rag_class,
        mock_storage_class,
        mock_processor,
        mock_server_class,
        sample_episode
    ):
        """Test for search_episodes tool."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_storage = MagicMock()
        mock_storage.search_episodes.return_value = [
            MemorySearchResult(episode=sample_episode, relevance_score=0.9)
        ]
        mock_storage_class.return_value = mock_storage

        mcp_server = MemoryTwinMCPServer()
        mcp_server._lazy_init()

        result = await mcp_server._search_episodes({
            "query": "authentication",
            "top_k": 5
        })

        assert result.isError is False
        mock_storage.search_episodes.assert_called_once()


class TestMCPServerCaptureThinking:
    """Tests for MCP server capture_thinking."""

    @patch("memorytwin.mcp_server.server.Server")
    @patch("memorytwin.mcp_server.server.ThoughtProcessor")
    @patch("memorytwin.mcp_server.server.MemoryStorage")
    @patch("memorytwin.mcp_server.server.RAGEngine")
    @pytest.mark.asyncio
    async def test_capture_thinking_success(
        self,
        mock_rag_class,
        mock_storage_class,
        mock_processor_class,
        mock_server_class
    ):
        """Test for successful thought capture."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_storage = MagicMock()
        mock_storage.store_episode.return_value = "test-episode-id"
        mock_storage_class.return_value = mock_storage

        sample_episode = Episode(
            id=uuid4(),
            task="Test task",
            context="Test context",
            reasoning_trace=ReasoningTrace(raw_thinking="test"),
            solution="test solution",
            solution_summary="test summary",
            episode_type=EpisodeType.FEATURE,
            tags=["test"],
            lessons_learned=["lesson1"],
            project_name="test"
        )

        mock_processor = MagicMock()
        mock_processor.process_thought = AsyncMock(return_value=sample_episode)
        mock_processor_class.return_value = mock_processor

        mcp_server = MemoryTwinMCPServer()
        mcp_server._lazy_init()

        result = await mcp_server._capture_thinking({
            "thinking_text": "My reasoning here",
            "user_prompt": "User prompt",
            "project_name": "test-project",
            "source_assistant": "copilot"
        })

        assert result.isError is False
        content = result.content[0].text
        assert "test-episode-id" in content
        assert "success" in content


class TestMCPServerCaptureDecision:
    """Tests for MCP server capture_decision."""

    @patch("memorytwin.mcp_server.server.Server")
    @patch("memorytwin.mcp_server.server.ThoughtProcessor")
    @patch("memorytwin.mcp_server.server.MemoryStorage")
    @patch("memorytwin.mcp_server.server.RAGEngine")
    @pytest.mark.asyncio
    async def test_capture_decision_success(
        self,
        mock_rag_class,
        mock_storage_class,
        mock_processor_class,
        mock_server_class
    ):
        """Test for successful structured decision capture."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_storage = MagicMock()
        mock_storage.store_episode.return_value = "decision-episode-id"
        mock_storage_class.return_value = mock_storage

        sample_episode = Episode(
            id=uuid4(),
            task="Choose database",
            context="Test context",
            reasoning_trace=ReasoningTrace(raw_thinking="test"),
            solution="PostgreSQL",
            solution_summary="PostgreSQL was chosen",
            episode_type=EpisodeType.DECISION,
            tags=["database", "postgresql"],
            lessons_learned=["For relational data, SQL is better"],
            project_name="test"
        )

        mock_processor = MagicMock()
        mock_processor.process_thought = AsyncMock(return_value=sample_episode)
        mock_processor_class.return_value = mock_processor

        mcp_server = MemoryTwinMCPServer()
        mcp_server._lazy_init()

        result = await mcp_server._capture_decision({
            "task": "Choose database for the project",
            "decision": "PostgreSQL",
            "alternatives": ["MongoDB", "MySQL", "SQLite"],
            "reasoning": "We need ACID transactions and complex queries",
            "lesson": "For relational data with transactions, SQL > NoSQL",
            "project_name": "test-project",
            "source_assistant": "copilot"
        })

        assert result.isError is False
        content = result.content[0].text
        assert "decision-episode-id" in content
        assert "success" in content
        assert "PostgreSQL" in content

    @patch("memorytwin.mcp_server.server.Server")
    @patch("memorytwin.mcp_server.server.ThoughtProcessor")
    @patch("memorytwin.mcp_server.server.MemoryStorage")
    @patch("memorytwin.mcp_server.server.RAGEngine")
    @pytest.mark.asyncio
    async def test_capture_decision_minimal(
        self,
        mock_rag_class,
        mock_storage_class,
        mock_processor_class,
        mock_server_class
    ):
        """Test for decision capture with minimum required fields."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_storage = MagicMock()
        mock_storage.store_episode.return_value = "minimal-decision-id"
        mock_storage_class.return_value = mock_storage

        sample_episode = Episode(
            id=uuid4(),
            task="Test task",
            context="",
            reasoning_trace=ReasoningTrace(raw_thinking="test"),
            solution="test",
            solution_summary="test",
            episode_type=EpisodeType.DECISION,
            tags=[],
            lessons_learned=[],
            project_name="default"
        )

        mock_processor = MagicMock()
        mock_processor.process_thought = AsyncMock(return_value=sample_episode)
        mock_processor_class.return_value = mock_processor

        mcp_server = MemoryTwinMCPServer()
        mcp_server._lazy_init()

        # Only required fields: task, decision, reasoning
        result = await mcp_server._capture_decision({
            "task": "Choose config format",
            "decision": "YAML",
            "reasoning": "More readable than JSON"
        })

        assert result.isError is False
        content = result.content[0].text
        assert "minimal-decision-id" in content

    @patch("memorytwin.mcp_server.server.Server")
    @patch("memorytwin.mcp_server.server.ThoughtProcessor")
    @patch("memorytwin.mcp_server.server.MemoryStorage")
    @patch("memorytwin.mcp_server.server.RAGEngine")
    @pytest.mark.asyncio
    async def test_capture_decision_fallback_when_llm_fails(
        self,
        mock_rag_class,
        mock_storage_class,
        mock_processor_class,
        mock_server_class
    ):
        """Capture decision should still store an episode if LLM processing fails."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_storage = MagicMock()
        mock_storage.store_episode.return_value = "fallback-decision-id"
        mock_storage_class.return_value = mock_storage

        mock_processor = MagicMock()
        mock_processor.process_thought = AsyncMock(side_effect=Exception("Model not found"))
        mock_processor_class.return_value = mock_processor

        mcp_server = MemoryTwinMCPServer()
        mcp_server._lazy_init()

        result = await mcp_server._capture_decision({
            "task": "Choose queue system",
            "decision": "Redis Streams",
            "reasoning": "Simple operations and low latency",
            "project_name": "test-project"
        })

        assert result.isError is False
        content = result.content[0].text
        assert "fallback-decision-id" in content
        mock_storage.store_episode.assert_called_once()


class TestMCPServerCaptureQuick:
    """Tests for MCP server capture_quick."""

    @patch("memorytwin.mcp_server.server.Server")
    @patch("memorytwin.mcp_server.server.ThoughtProcessor")
    @patch("memorytwin.mcp_server.server.MemoryStorage")
    @patch("memorytwin.mcp_server.server.RAGEngine")
    @pytest.mark.asyncio
    async def test_capture_quick_success(
        self,
        mock_rag_class,
        mock_storage_class,
        mock_processor_class,
        mock_server_class
    ):
        """Test for successful quick capture."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_storage = MagicMock()
        mock_storage.store_episode.return_value = "quick-episode-id"
        mock_storage_class.return_value = mock_storage

        sample_episode = Episode(
            id=uuid4(),
            task="Added retry logic",
            context="",
            reasoning_trace=ReasoningTrace(raw_thinking="test"),
            solution="Retry con exponential backoff",
            solution_summary="Retry was added",
            episode_type=EpisodeType.BUG_FIX,
            tags=["http", "retry"],
            lessons_learned=["Always use retry for external calls"],
            project_name="test"
        )

        mock_processor = MagicMock()
        mock_processor.process_thought = AsyncMock(return_value=sample_episode)
        mock_processor_class.return_value = mock_processor

        mcp_server = MemoryTwinMCPServer()
        mcp_server._lazy_init()

        result = await mcp_server._capture_quick({
            "what": "Added retry logic to HTTP client",
            "why": "API calls were failing intermittently",
            "lesson": "Always use retry for external calls",
            "project_name": "test-project"
        })

        assert result.isError is False
        content = result.content[0].text
        assert "quick-episode-id" in content
        assert "success" in content

    @patch("memorytwin.mcp_server.server.Server")
    @patch("memorytwin.mcp_server.server.ThoughtProcessor")
    @patch("memorytwin.mcp_server.server.MemoryStorage")
    @patch("memorytwin.mcp_server.server.RAGEngine")
    @pytest.mark.asyncio
    async def test_capture_quick_minimal(
        self,
        mock_rag_class,
        mock_storage_class,
        mock_processor_class,
        mock_server_class
    ):
        """Test for quick capture with minimum fields."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_storage = MagicMock()
        mock_storage.store_episode.return_value = "minimal-quick-id"
        mock_storage_class.return_value = mock_storage

        sample_episode = Episode(
            id=uuid4(),
            task="Test",
            context="",
            reasoning_trace=ReasoningTrace(raw_thinking="test"),
            solution="test",
            solution_summary="test",
            episode_type=EpisodeType.DECISION,
            tags=[],
            lessons_learned=[],
            project_name="default"
        )

        mock_processor = MagicMock()
        mock_processor.process_thought = AsyncMock(return_value=sample_episode)
        mock_processor_class.return_value = mock_processor

        mcp_server = MemoryTwinMCPServer()
        mcp_server._lazy_init()

        # Only required fields: what, why
        result = await mcp_server._capture_quick({
            "what": "Switched from axios to fetch",
            "why": "Reduce dependencies"
        })

        assert result.isError is False
        content = result.content[0].text
        assert "minimal-quick-id" in content

    @patch("memorytwin.mcp_server.server.Server")
    @patch("memorytwin.mcp_server.server.ThoughtProcessor")
    @patch("memorytwin.mcp_server.server.MemoryStorage")
    @patch("memorytwin.mcp_server.server.RAGEngine")
    @pytest.mark.asyncio
    async def test_capture_quick_fallback_when_llm_fails(
        self,
        mock_rag_class,
        mock_storage_class,
        mock_processor_class,
        mock_server_class
    ):
        """Quick capture should still store an episode if LLM processing fails."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_storage = MagicMock()
        mock_storage.store_episode.return_value = "fallback-quick-id"
        mock_storage_class.return_value = mock_storage

        mock_processor = MagicMock()
        mock_processor.process_thought = AsyncMock(side_effect=Exception("Model not found"))
        mock_processor_class.return_value = mock_processor

        mcp_server = MemoryTwinMCPServer()
        mcp_server._lazy_init()

        result = await mcp_server._capture_quick({
            "what": "Added timeout handling",
            "why": "Prevent hanging external calls",
            "project_name": "test-project"
        })

        assert result.isError is False
        content = result.content[0].text
        assert "fallback-quick-id" in content
        mock_storage.store_episode.assert_called_once()


class TestMCPServerQueryMemory:
    """Tests for MCP server query_memory."""

    @patch("memorytwin.mcp_server.server.Server")
    @patch("memorytwin.mcp_server.server.ThoughtProcessor")
    @patch("memorytwin.mcp_server.server.MemoryStorage")
    @patch("memorytwin.mcp_server.server.RAGEngine")
    @pytest.mark.asyncio
    async def test_query_memory_success(
        self,
        mock_rag_class,
        mock_storage_class,
        mock_processor,
        mock_server_class
    ):
        """Test for successful RAG query."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_storage = MagicMock()
        mock_storage_class.return_value = mock_storage

        mock_rag = MagicMock()
        mock_rag.query = AsyncMock(return_value={
            "answer": "JWT fue elegido por escalabilidad",
            "episodes_used": [],
            "context_provided": True
        })
        mock_rag_class.return_value = mock_rag

        mcp_server = MemoryTwinMCPServer()
        mcp_server._lazy_init()

        result = await mcp_server._query_memory({
            "question": "Why did we use JWT?",
            "project_name": "test",
            "num_episodes": 3
        })

        assert result.isError is False
        assert "JWT" in result.content[0].text
        mock_rag.query.assert_called_once_with(
            question="Why did we use JWT?",
            project_name="test",
            top_k=3
        )


class TestMCPServerProjectContext:
    """Tests for MCP server get_project_context."""

    @patch("memorytwin.mcp_server.server.Server")
    @patch("memorytwin.mcp_server.server.ThoughtProcessor")
    @patch("memorytwin.mcp_server.server.MemoryStorage")
    @patch("memorytwin.mcp_server.server.RAGEngine")
    @pytest.mark.asyncio
    async def test_project_context_empty(
        self,
        mock_rag_class,
        mock_storage_class,
        mock_processor,
        mock_server_class
    ):
        """Test for empty context."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_storage = MagicMock()
        mock_storage.get_meta_memory_statistics.return_value = {"total_meta_memories": 0}
        mock_storage_class.return_value = mock_storage

        mock_rag = MagicMock()
        mock_rag.get_statistics.return_value = {"total_episodes": 0}
        mock_rag_class.return_value = mock_rag

        mcp_server = MemoryTwinMCPServer()
        mcp_server._lazy_init()

        result = await mcp_server._get_project_context({})

        assert result.isError is False
        content = result.content[0].text
        assert "empty" in content or "no hay" in content.lower()

    @patch("memorytwin.mcp_server.server.Server")
    @patch("memorytwin.mcp_server.server.ThoughtProcessor")
    @patch("memorytwin.mcp_server.server.MemoryStorage")
    @patch("memorytwin.mcp_server.server.RAGEngine")
    @pytest.mark.asyncio
    async def test_project_context_full_mode(
        self,
        mock_rag_class,
        mock_storage_class,
        mock_processor,
        mock_server_class
    ):
        """Test for full context (few memories)."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_storage = MagicMock()
        mock_storage.get_meta_memory_statistics.return_value = {"total_meta_memories": 0}
        mock_storage_class.return_value = mock_storage

        mock_rag = MagicMock()
        mock_rag.get_statistics.return_value = {"total_episodes": 10}
        mock_rag.get_timeline.return_value = [
            {"id": "1", "type": "feature", "task": "Test", "summary": "sum", "date": "2025-01-01", "tags": []}
        ]
        mock_rag.get_lessons.return_value = []
        mock_rag_class.return_value = mock_rag

        mcp_server = MemoryTwinMCPServer()
        mcp_server._lazy_init()

        result = await mcp_server._get_project_context({"topic": "test"})

        assert result.isError is False
        content = result.content[0].text
        assert "full_context" in content

    @patch("memorytwin.mcp_server.server.Server")
    @patch("memorytwin.mcp_server.server.ThoughtProcessor")
    @patch("memorytwin.mcp_server.server.MemoryStorage")
    @patch("memorytwin.mcp_server.server.RAGEngine")
    @pytest.mark.asyncio
    async def test_project_context_smart_mode(
        self,
        mock_rag_class,
        mock_storage_class,
        mock_processor,
        mock_server_class
    ):
        """Test for smart context (many memories)."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_storage = MagicMock()
        mock_storage.search_episodes.return_value = []
        mock_storage.get_meta_memory_statistics.return_value = {"total_meta_memories": 0}
        mock_storage_class.return_value = mock_storage

        mock_rag = MagicMock()
        mock_rag.get_statistics.return_value = {"total_episodes": 50}
        mock_rag.get_timeline.return_value = []
        mock_rag.get_lessons.return_value = []
        mock_rag_class.return_value = mock_rag

        mcp_server = MemoryTwinMCPServer()
        mcp_server._lazy_init()

        result = await mcp_server._get_project_context({"topic": "auth"})

        assert result.isError is False
        content = result.content[0].text
        assert "smart_context" in content


class TestMCPServerErrorHandling:
    """Tests for MCP server error handling."""

    @patch("memorytwin.mcp_server.server.Server")
    @patch("memorytwin.mcp_server.server.ThoughtProcessor")
    @patch("memorytwin.mcp_server.server.MemoryStorage")
    @patch("memorytwin.mcp_server.server.RAGEngine")
    @pytest.mark.asyncio
    async def test_tool_error_handling(
        self,
        mock_rag_class,
        mock_storage_class,
        mock_processor,
        mock_server_class
    ):
        """Test for error handling in tools."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_storage = MagicMock()
        mock_storage.get_statistics.side_effect = Exception("Database error")
        mock_storage_class.return_value = mock_storage

        mcp_server = MemoryTwinMCPServer()
        mcp_server._lazy_init()

        # The _get_statistics method doesn't catch exceptions, so it should raise
        with pytest.raises(Exception, match="Database error"):
            await mcp_server._get_statistics({})

    @patch("memorytwin.mcp_server.server.Server")
    @patch("memorytwin.mcp_server.server.ThoughtProcessor")
    @patch("memorytwin.mcp_server.server.MemoryStorage")
    @patch("memorytwin.mcp_server.server.RAGEngine")
    @pytest.mark.asyncio
    async def test_get_episode_missing_id(
        self,
        mock_rag_class,
        mock_storage_class,
        mock_processor,
        mock_server_class
    ):
        """Test for get_episode without ID."""
        from memorytwin.mcp_server.server import MemoryTwinMCPServer

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_storage = MagicMock()
        mock_storage_class.return_value = mock_storage

        mcp_server = MemoryTwinMCPServer()
        mcp_server._lazy_init()

        result = await mcp_server._get_episode({})

        assert result.isError is True
        assert "episode_id" in result.content[0].text.lower()
