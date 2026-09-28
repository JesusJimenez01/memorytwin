"""
Tests for configuration
========================

Unit tests for config.py.
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSettings:
    """Tests for the Settings class."""

    def test_settings_defaults(self):
        """Test for default values when there is no .env."""
        from memorytwin.config import Settings

        # Save and clear environment variables that affect the test
        env_vars_to_clear = [
            'LLM_PROVIDER', 'LLM_MODEL', 'LLM_TEMPERATURE',
            'GOOGLE_API_KEY', 'OPENROUTER_API_KEY',
            'GRADIO_SERVER_NAME', 'GRADIO_SERVER_PORT', 'GRADIO_SHARE', 'EMBEDDING_MODEL'
        ]
        saved_env = {k: os.environ.pop(k, None) for k in env_vars_to_clear}

        try:
            # Create settings without environment variables
            settings = Settings(
                _env_file=None  # Don't load any .env
            )

            assert settings.gradio_server_name == "127.0.0.1"
            assert settings.gradio_server_port == 7860
            assert settings.gradio_share is False
            assert settings.llm_provider == "openrouter"
            assert isinstance(settings.llm_model, str)
            assert settings.llm_model.strip() != ""
            assert settings.llm_temperature == 0.3
            assert settings.embedding_model == "all-MiniLM-L6-v2"
        finally:
            # Restore environment variables
            for k, v in saved_env.items():
                if v is not None:
                    os.environ[k] = v

    def test_settings_chroma_default(self):
        """Test for ChromaDB default path."""
        from memorytwin.config import Settings

        settings = Settings()

        assert "./data/chroma" in settings.chroma_persist_dir or "data" in settings.chroma_persist_dir

    def test_settings_sqlite_default(self):
        """Test for SQLite default path."""
        from memorytwin.config import Settings

        settings = Settings()

        assert "memory.db" in settings.sqlite_db_path or "data" in settings.sqlite_db_path


class TestGetSettings:
    """Tests for the get_settings function."""

    def test_get_settings_returns_settings(self):
        """Test that get_settings returns Settings."""
        from memorytwin.config import Settings, get_settings

        settings = get_settings()

        assert isinstance(settings, Settings)

    def test_get_settings_cached(self):
        """Test that get_settings is cached."""
        from memorytwin.config import get_settings

        settings1 = get_settings()
        settings2 = get_settings()

        # Same object (cached)
        assert settings1 is settings2


class TestPathFunctions:
    """Tests for path functions."""

    def test_get_chroma_dir_exists(self, tmp_path):
        """Test that get_chroma_dir creates the directory."""
        from memorytwin.config import get_chroma_dir

        chroma_dir = get_chroma_dir()

        assert isinstance(chroma_dir, Path)
        assert chroma_dir.exists()

    def test_get_sqlite_path_parent_exists(self, tmp_path):
        """Test that get_sqlite_path creates parent directory."""
        from memorytwin.config import get_sqlite_path

        sqlite_path = get_sqlite_path()

        assert isinstance(sqlite_path, Path)
        assert sqlite_path.parent.exists()


class TestSettingsWithEnvVars:
    """Tests for configuration with environment variables."""

    def test_settings_gradio_port_from_env(self):
        """Test that GRADIO_SERVER_PORT is read from the environment."""
        with patch.dict(os.environ, {"GRADIO_SERVER_PORT": "9999"}):
            from memorytwin.config import Settings

            settings = Settings()

            assert settings.gradio_server_port == 9999

    def test_settings_gradio_share_true(self):
        """Test for GRADIO_SHARE=true."""
        with patch.dict(os.environ, {"GRADIO_SHARE": "true"}):
            from memorytwin.config import Settings

            settings = Settings()

            assert settings.gradio_share is True

    def test_settings_gradio_share_false(self):
        """Test for GRADIO_SHARE=false."""
        with patch.dict(os.environ, {"GRADIO_SHARE": "false"}):
            from memorytwin.config import Settings

            settings = Settings()

            assert settings.gradio_share is False

    def test_settings_llm_temperature_custom(self):
        """Test for custom LLM_TEMPERATURE."""
        with patch.dict(os.environ, {"LLM_TEMPERATURE": "0.7"}):
            from memorytwin.config import Settings

            settings = Settings()

            assert settings.llm_temperature == 0.7


class TestOpenRouterJsonFallback:
    """Tests for JSON mode fallback behavior in OpenRouterClient."""

    def test_generate_falls_back_when_json_mode_unsupported(self):
        """If provider rejects JSON mode, client should retry without response_format."""
        from memorytwin.config import OpenRouterClient

        client = OpenRouterClient.__new__(OpenRouterClient)
        client._model_name = "google/gemma-3-4b-it:free"
        client._temperature = 0.3
        client._max_tokens = 128
        client._json_mode = True

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"ok": true}'))]

        create_mock = MagicMock(
            side_effect=[
                Exception("JSON mode is not enabled for models/gemma-3-4b-it"),
                mock_response,
            ]
        )
        client._client = MagicMock(chat=MagicMock(completions=MagicMock(create=create_mock)))

        result = client.generate("Return JSON")

        assert result.text == '{"ok": true}'
        assert create_mock.call_count == 2
        first_call_kwargs = create_mock.call_args_list[0].kwargs
        second_call_kwargs = create_mock.call_args_list[1].kwargs
        assert first_call_kwargs["response_format"] == {"type": "json_object"}
        assert "response_format" not in second_call_kwargs

    def test_generate_async_falls_back_when_json_mode_unsupported(self):
        """Async generation should retry without response_format when JSON mode is unsupported."""
        import asyncio

        from memorytwin.config import LLMResponse, OpenRouterClient

        client = OpenRouterClient.__new__(OpenRouterClient)
        client._model_name = "google/gemma-3-4b-it:free"
        client._temperature = 0.3
        client._max_tokens = 128
        client._json_mode = True

        async_create = AsyncMock(
            side_effect=[
                Exception("JSON mode is not enabled"),
                MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))]),
            ]
        )

        client._async_client = MagicMock(chat=MagicMock(completions=MagicMock(create=async_create)))

        result = asyncio.run(client.generate_async([{"role": "user", "parts": ["Return JSON"]}]))

        assert isinstance(result, LLMResponse)
        assert result.text == "ok"
        assert async_create.call_count == 2
        first_call_kwargs = async_create.call_args_list[0].kwargs
        second_call_kwargs = async_create.call_args_list[1].kwargs
        assert first_call_kwargs["response_format"] == {"type": "json_object"}
        assert "response_format" not in second_call_kwargs


class TestParseJsonResponse:
    """Tests for the LLM JSON response parser."""

    def test_plain_json(self):
        from memorytwin.config import parse_json_response

        assert parse_json_response('{"a": 1}') == {"a": 1}

    def test_json_in_code_fence(self):
        from memorytwin.config import parse_json_response

        text = '```json\n{"pattern": "retry", "lessons": ["x"]}\n```'
        assert parse_json_response(text) == {"pattern": "retry", "lessons": ["x"]}

    def test_json_surrounded_by_prose(self):
        from memorytwin.config import parse_json_response

        text = 'Sure! Here is the result: {"task": "t"} Hope it helps.'
        assert parse_json_response(text) == {"task": "t"}

    def test_invalid_json_raises(self):
        from memorytwin.config import parse_json_response

        with pytest.raises(ValueError):
            parse_json_response("no json here")

    def test_non_object_json_raises(self):
        from memorytwin.config import parse_json_response

        with pytest.raises(ValueError):
            parse_json_response("[1, 2, 3]")
