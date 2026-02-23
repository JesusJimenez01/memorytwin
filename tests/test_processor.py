"""
Tests for the processor module (ThoughtProcessor)
==================================================

Unit tests for the thought processor with mocks
to avoid real LLM calls.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memorytwin.escriba.processor import STRUCTURING_PROMPT, ThoughtProcessor
from memorytwin.models import (
    EpisodeType,
    ProcessedInput,
)


class TestThoughtProcessor:
    """Tests for ThoughtProcessor."""

    @pytest.fixture
    def mock_llm_model(self):
        """Mock for get_llm_model factory."""
        with patch("memorytwin.escriba.processor.get_llm_model") as mock:
            mock_model = MagicMock()
            mock.return_value = mock_model
            yield mock, mock_model

    @pytest.fixture
    def sample_input(self):
        """Sample input for tests."""
        return ProcessedInput(
            raw_text="I considered using JWT because it's stateless and scales well. "
                     "Discarded sessions for requiring Redis.",
            user_prompt="Implement authentication",
            code_changes="def create_token(user): pass",
            source="test"
        )

    @pytest.fixture
    def sample_llm_response(self):
        """Sample LLM response."""
        return {
            "task": "Implement JWT authentication",
            "context": "REST API with FastAPI",
            "reasoning_trace": {
                "raw_thinking": "I chose JWT for scalability",
                "alternatives_considered": ["Sessions con Redis"],
                "decision_factors": ["Stateless", "Escalabilidad"],
                "confidence_level": 0.85
            },
            "solution": "from jose import jwt",
            "solution_summary": "JWT with 24h tokens",
            "episode_type": "feature",
            "tags": ["auth", "jwt"],
            "files_affected": ["auth.py"],
            "lessons_learned": ["Validate JWT algorithm"]
        }

    def test_processor_init_with_factory(self, mock_llm_model):
        """Test for initialization using factory."""
        mock_factory, mock_model = mock_llm_model

        processor = ThoughtProcessor()

        # Verify that the factory was called with JSON mime type
        mock_factory.assert_called_once_with(response_mime_type="application/json")
        assert processor.model == mock_model

    def test_processor_init_no_api_key_raises(self):
        """Test that initialization fails without API key in config."""
        with patch("memorytwin.escriba.processor.get_llm_model") as mock:
            mock.side_effect = ValueError("GOOGLE_API_KEY is required")

            with pytest.raises(ValueError, match="GOOGLE_API_KEY is required"):
                ThoughtProcessor()

    def test_build_user_prompt_minimal(self, mock_llm_model):
        """Test for minimal prompt construction."""
        processor = ThoughtProcessor()

        raw_input = ProcessedInput(
            raw_text="Simple thinking",
            source="test"
        )

        prompt = processor._build_user_prompt(raw_input)

        assert "## REASONING TEXT (THINKING):" in prompt
        assert "Simple thinking" in prompt
        assert "ORIGINAL USER PROMPT" not in prompt
        assert "CODE CHANGES" not in prompt

    def test_build_user_prompt_full(self, mock_llm_model, sample_input):
        """Test for complete prompt construction."""
        processor = ThoughtProcessor()

        prompt = processor._build_user_prompt(sample_input)

        assert "## REASONING TEXT (THINKING):" in prompt
        assert "I considered using JWT" in prompt
        assert "## ORIGINAL USER PROMPT:" in prompt
        assert "Implement authentication" in prompt
        assert "## CODE CHANGES:" in prompt
        assert "def create_token" in prompt

    def test_build_episode_from_data(self, mock_llm_model, sample_llm_response):
        """Test for building Episode from structured data."""
        processor = ThoughtProcessor()

        episode = processor._build_episode(
            sample_llm_response,
            project_name="test-project",
            source_assistant="copilot"
        )

        assert episode.task == "Implement JWT authentication"
        assert episode.context == "REST API with FastAPI"
        assert episode.episode_type == EpisodeType.FEATURE
        assert episode.project_name == "test-project"
        assert episode.source_assistant == "copilot"
        assert "auth" in episode.tags
        assert "jwt" in episode.tags
        assert len(episode.reasoning_trace.alternatives_considered) == 1
        assert episode.reasoning_trace.confidence_level == 0.85

    def test_build_episode_invalid_type_defaults(self, mock_llm_model):
        """Test that invalid type uses default."""
        processor = ThoughtProcessor()

        data = {
            "task": "Test task",
            "episode_type": "invalid_type"
        }

        episode = processor._build_episode(data, "project", "assistant")

        assert episode.episode_type == EpisodeType.DECISION

    def test_build_episode_missing_fields(self, mock_llm_model):
        """Test with missing fields uses defaults."""
        processor = ThoughtProcessor()

        data = {}

        episode = processor._build_episode(data, "project", "assistant")

        assert episode.task == "Unspecified task"
        assert episode.context == "Unspecified context"
        assert episode.tags == []
        assert episode.lessons_learned == []

    @pytest.mark.asyncio
    async def test_process_thought_success(
        self, mock_llm_model, sample_input, sample_llm_response
    ):
        """Test for successful processing."""
        import json

        mock_factory, mock_model = mock_llm_model

        # Configure model mock
        mock_response = MagicMock()
        mock_response.text = json.dumps(sample_llm_response)
        mock_model.generate_async = AsyncMock(return_value=mock_response)

        processor = ThoughtProcessor()

        episode = await processor.process_thought(
            sample_input,
            project_name="test-project",
            source_assistant="copilot"
        )

        assert episode.task == "Implement JWT authentication"
        assert episode.project_name == "test-project"
        mock_model.generate_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_thought_extracts_json_from_text(
        self, mock_llm_model, sample_input, sample_llm_response
    ):
        """Test that extracts JSON from text with additional content."""
        import json

        mock_factory, mock_model = mock_llm_model

        # Response with additional text
        mock_response = MagicMock()
        mock_response.text = f"Here is the result:\n{json.dumps(sample_llm_response)}\nEnd."
        mock_model.generate_async = AsyncMock(return_value=mock_response)

        processor = ThoughtProcessor()

        episode = await processor.process_thought(
            sample_input,
            project_name="test-project",
            source_assistant="copilot"
        )

        assert episode.task == "Implement JWT authentication"

    @pytest.mark.asyncio
    async def test_process_thought_invalid_json_raises(
        self, mock_llm_model, sample_input
    ):
        """Test that invalid JSON raises error."""
        mock_factory, mock_model = mock_llm_model

        mock_response = MagicMock()
        mock_response.text = "this is not valid json without braces"
        mock_model.generate_async = AsyncMock(return_value=mock_response)

        processor = ThoughtProcessor()

        # Can raise ValueError or json.JSONDecodeError
        with pytest.raises((ValueError, Exception)):
            await processor.process_thought(sample_input)

    def test_process_thought_sync(self, mock_llm_model, sample_input, sample_llm_response):
        """Test for synchronous version."""
        import json

        mock_factory, mock_model = mock_llm_model

        mock_response = MagicMock()
        mock_response.text = json.dumps(sample_llm_response)
        mock_model.generate_async = AsyncMock(return_value=mock_response)

        processor = ThoughtProcessor()

        episode = processor.process_thought_sync(
            sample_input,
            project_name="test-project",
            source_assistant="copilot"
        )

        assert episode.task == "Implement JWT authentication"


class TestStructuringPrompt:
    """Tests for the structuring prompt."""

    def test_prompt_contains_required_fields(self):
        """Test that the prompt mentions required fields."""
        assert "task" in STRUCTURING_PROMPT
        assert "context" in STRUCTURING_PROMPT
        assert "reasoning_trace" in STRUCTURING_PROMPT
        assert "alternatives_considered" in STRUCTURING_PROMPT
        assert "decision_factors" in STRUCTURING_PROMPT
        assert "confidence_level" in STRUCTURING_PROMPT
        assert "episode_type" in STRUCTURING_PROMPT
        assert "tags" in STRUCTURING_PROMPT
        assert "lessons_learned" in STRUCTURING_PROMPT

    def test_prompt_mentions_json_format(self):
        """Test that the prompt requests JSON."""
        assert "JSON" in STRUCTURING_PROMPT
        assert "ALWAYS respond with valid JSON" in STRUCTURING_PROMPT
