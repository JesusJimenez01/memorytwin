"""
Tests for the Escriba module
============================

Unit tests for the Escriba agent with mocks.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from memorytwin.escriba.escriba import Escriba
from memorytwin.models import (
    Episode,
    EpisodeType,
    ReasoningTrace,
)


class TestEscriba:
    """Tests for Escriba."""

    @pytest.fixture
    def mock_processor(self):
        """Mock for the thought processor."""
        processor = MagicMock()
        processor.process_thought = AsyncMock()
        return processor

    @pytest.fixture
    def mock_storage(self):
        """Mock for storage."""
        storage = MagicMock()
        storage.store_episode = MagicMock(return_value=str(uuid4()))
        storage.get_statistics = MagicMock(return_value={"total": 10})
        storage.search_episodes = MagicMock(return_value=[])
        return storage

    @pytest.fixture
    def sample_episode(self):
        """Sample episode."""
        return Episode(
            task="Implement JWT authentication",
            context="REST API with FastAPI",
            reasoning_trace=ReasoningTrace(
                raw_thinking="I chose JWT for scalability"
            ),
            solution="from jose import jwt",
            solution_summary="JWT with 24h tokens",
            episode_type=EpisodeType.FEATURE,
            tags=["auth", "jwt"],
            project_name="test-project"
        )

    @patch("memorytwin.escriba.escriba.console")
    def test_escriba_initialization(self, mock_console, mock_processor, mock_storage):
        """Test for Escriba initialization."""
        escriba = Escriba(
            processor=mock_processor,
            storage=mock_storage,
            project_name="test-project"
        )

        assert escriba.processor == mock_processor
        assert escriba.storage == mock_storage
        assert escriba.project_name == "test-project"
        mock_console.print.assert_called()  # Verify that it prints startup message

    @patch("memorytwin.escriba.escriba.console")
    @pytest.mark.asyncio
    async def test_capture_thinking_success(
        self, mock_console, mock_processor, mock_storage, sample_episode
    ):
        """Test for successful thought capture."""
        mock_processor.process_thought.return_value = sample_episode

        escriba = Escriba(
            processor=mock_processor,
            storage=mock_storage,
            project_name="default"
        )

        result = await escriba.capture_thinking(
            thinking_text="I considered JWT for scalability",
            user_prompt="Implement auth",
            code_changes="def auth(): pass",
            source_assistant="copilot",
            project_name="my-project"
        )

        assert result == sample_episode
        mock_processor.process_thought.assert_called_once()
        mock_storage.store_episode.assert_called_once_with(sample_episode)

    @patch("memorytwin.escriba.escriba.console")
    @pytest.mark.asyncio
    async def test_capture_thinking_uses_default_project(
        self, mock_console, mock_processor, mock_storage, sample_episode
    ):
        """Test that uses default project if not specified."""
        mock_processor.process_thought.return_value = sample_episode

        escriba = Escriba(
            processor=mock_processor,
            storage=mock_storage,
            project_name="default-project"
        )

        await escriba.capture_thinking(
            thinking_text="Test thinking",
            source_assistant="test"
        )

        # Verify it was called with the default project
        call_args = mock_processor.process_thought.call_args
        assert call_args.kwargs["project_name"] == "default-project"

    @patch("memorytwin.escriba.escriba.console")
    def test_capture_thinking_sync(
        self, mock_console, mock_processor, mock_storage, sample_episode
    ):
        """Test for synchronous capture version."""
        mock_processor.process_thought.return_value = sample_episode

        escriba = Escriba(
            processor=mock_processor,
            storage=mock_storage
        )

        result = escriba.capture_thinking_sync(
            thinking_text="Test sync",
            source_assistant="test"
        )

        assert result == sample_episode

    @patch("memorytwin.escriba.escriba.console")
    def test_capture_from_file(
        self, mock_console, mock_processor, mock_storage, sample_episode, tmp_path
    ):
        """Test for capture from file."""
        mock_processor.process_thought.return_value = sample_episode

        # Create temporary file with content
        test_file = tmp_path / "thinking.txt"
        test_file.write_text("Thinking content from file", encoding="utf-8")

        escriba = Escriba(
            processor=mock_processor,
            storage=mock_storage
        )

        result = escriba.capture_from_file(
            file_path=str(test_file),
            source_assistant="file",
            project_name="file-project"
        )

        assert result == sample_episode

        # Verify that the file content was processed
        call_args = mock_processor.process_thought.call_args
        raw_input = call_args.args[0]
        assert "Thinking content from file" in raw_input.raw_text

    @patch("memorytwin.escriba.escriba.console")
    def test_get_statistics(self, mock_console, mock_processor, mock_storage):
        """Test for retrieving statistics."""
        mock_storage.get_statistics.return_value = {
            "total": 15,
            "by_type": {"feature": 10, "bug_fix": 5}
        }

        escriba = Escriba(
            processor=mock_processor,
            storage=mock_storage,
            project_name="stats-project"
        )

        stats = escriba.get_statistics()

        assert stats["total"] == 15
        mock_storage.get_statistics.assert_called_once_with("stats-project")

    @patch("memorytwin.escriba.escriba.console")
    def test_search(self, mock_console, mock_processor, mock_storage):
        """Test for search."""
        mock_result = MagicMock()
        mock_result.episode = MagicMock()
        mock_result.relevance_score = 0.9
        mock_storage.search_episodes.return_value = [mock_result]

        escriba = Escriba(
            processor=mock_processor,
            storage=mock_storage,
            project_name="search-project"
        )

        results = escriba.search("authentication", top_k=3)

        assert len(results) == 1
        mock_storage.search_episodes.assert_called_once()

        # Verify the query
        call_args = mock_storage.search_episodes.call_args
        query = call_args.args[0]
        assert query.query == "authentication"
        assert query.project_filter == "search-project"
        assert query.top_k == 3


class TestEscribaInputValidation:
    """Tests for input validation in capture."""

    @patch("memorytwin.escriba.escriba.console")
    @pytest.mark.asyncio
    async def test_capture_creates_processed_input(
        self, mock_console
    ):
        """Test that ProcessedInput is created correctly."""
        mock_processor = MagicMock()
        mock_storage = MagicMock()
        mock_storage.store_episode.return_value = "test-id"

        captured_input = None

        async def capture_input(raw_input, **kwargs):
            nonlocal captured_input
            captured_input = raw_input
            return Episode(
                task="Test",
                context="Test",
                reasoning_trace=ReasoningTrace(raw_thinking="test"),
                solution="test solution",
                solution_summary="test summary",
                episode_type=EpisodeType.FEATURE,
                project_name="test"
            )

        mock_processor.process_thought = capture_input

        escriba = Escriba(
            processor=mock_processor,
            storage=mock_storage
        )

        await escriba.capture_thinking(
            thinking_text="My thinking",
            user_prompt="Original prompt",
            code_changes="code here"
        )

        assert captured_input is not None
        assert captured_input.raw_text == "My thinking"
        assert captured_input.user_prompt == "Original prompt"
        assert captured_input.code_changes == "code here"
        assert captured_input.source == "api"
        assert isinstance(captured_input.captured_at, datetime)
