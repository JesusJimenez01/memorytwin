"""
Tests para el módulo Escriba
============================

Tests unitarios para el agente Escriba con mocks.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from uuid import uuid4

from memorytwin.models import (
    Episode,
    EpisodeType,
    ProcessedInput,
    ReasoningTrace,
    MemoryQuery,
)
from memorytwin.escriba.escriba import Escriba


class TestEscriba:
    """Tests para Escriba."""

    @pytest.fixture
    def mock_processor(self):
        """Mock del procesador de pensamientos."""
        processor = MagicMock()
        processor.process_thought = AsyncMock()
        return processor

    @pytest.fixture
    def mock_storage(self):
        """Mock del almacenamiento."""
        storage = MagicMock()
        storage.store_episode = MagicMock(return_value=str(uuid4()))
        storage.get_statistics = MagicMock(return_value={"total": 10})
        storage.search_episodes = MagicMock(return_value=[])
        return storage

    @pytest.fixture
    def sample_episode(self):
        """Episodio de ejemplo."""
        return Episode(
            task="Implementar autenticación JWT",
            context="API REST con FastAPI",
            reasoning_trace=ReasoningTrace(
                raw_thinking="Elegí JWT por escalabilidad"
            ),
            solution="from jose import jwt",
            solution_summary="JWT con tokens de 24h",
            episode_type=EpisodeType.FEATURE,
            tags=["auth", "jwt"],
            project_name="test-project"
        )

    @patch("memorytwin.escriba.escriba.console")
    def test_escriba_initialization(self, mock_console, mock_processor, mock_storage):
        """Test de inicialización de Escriba."""
        escriba = Escriba(
            processor=mock_processor,
            storage=mock_storage,
            project_name="test-project"
        )
        
        assert escriba.processor == mock_processor
        assert escriba.storage == mock_storage
        assert escriba.project_name == "test-project"
        mock_console.print.assert_called()  # Verifica que imprime mensaje de inicio

    @patch("memorytwin.escriba.escriba.console")
    @pytest.mark.asyncio
    async def test_capture_thinking_success(
        self, mock_console, mock_processor, mock_storage, sample_episode
    ):
        """Test de captura exitosa de pensamiento."""
        mock_processor.process_thought.return_value = sample_episode
        
        escriba = Escriba(
            processor=mock_processor,
            storage=mock_storage,
            project_name="default"
        )
        
        result = await escriba.capture_thinking(
            thinking_text="Consideré JWT por escalabilidad",
            user_prompt="Implementa auth",
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
        """Test que usa proyecto por defecto si no se especifica."""
        mock_processor.process_thought.return_value = sample_episode
        
        escriba = Escriba(
            processor=mock_processor,
            storage=mock_storage,
            project_name="default-project"
        )
        
        await escriba.capture_thinking(
            thinking_text="Pensamiento test",
            source_assistant="test"
        )
        
        # Verificar que se llamó con el proyecto default
        call_args = mock_processor.process_thought.call_args
        assert call_args.kwargs["project_name"] == "default-project"

    @patch("memorytwin.escriba.escriba.console")
    def test_capture_thinking_sync(
        self, mock_console, mock_processor, mock_storage, sample_episode
    ):
        """Test de versión síncrona de captura."""
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
        """Test de captura desde archivo."""
        mock_processor.process_thought.return_value = sample_episode
        
        # Crear archivo temporal con contenido
        test_file = tmp_path / "thinking.txt"
        test_file.write_text("Contenido de thinking desde archivo", encoding="utf-8")
        
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
        
        # Verificar que se procesó el contenido del archivo
        call_args = mock_processor.process_thought.call_args
        raw_input = call_args.args[0]
        assert "Contenido de thinking desde archivo" in raw_input.raw_text

    @patch("memorytwin.escriba.escriba.console")
    def test_get_statistics(self, mock_console, mock_processor, mock_storage):
        """Test de obtención de estadísticas."""
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
        """Test de búsqueda."""
        mock_result = MagicMock()
        mock_result.episode = MagicMock()
        mock_result.relevance_score = 0.9
        mock_storage.search_episodes.return_value = [mock_result]
        
        escriba = Escriba(
            processor=mock_processor,
            storage=mock_storage,
            project_name="search-project"
        )
        
        results = escriba.search("autenticación", top_k=3)
        
        assert len(results) == 1
        mock_storage.search_episodes.assert_called_once()
        
        # Verificar el query
        call_args = mock_storage.search_episodes.call_args
        query = call_args.args[0]
        assert query.query == "autenticación"
        assert query.project_filter == "search-project"
        assert query.top_k == 3


class TestEscribaClipboard:
    """Tests para captura desde clipboard."""

    @patch("memorytwin.escriba.escriba.console")
    def test_capture_from_clipboard_missing_pyperclip(
        self, mock_console
    ):
        """Test que falla sin pyperclip instalado."""
        with patch.dict("sys.modules", {"pyperclip": None}):
            # Forzar que el import falle
            import builtins
            original_import = builtins.__import__
            
            def mock_import(name, *args):
                if name == "pyperclip":
                    raise ImportError("No module named 'pyperclip'")
                return original_import(name, *args)
            
            with patch.object(builtins, "__import__", mock_import):
                mock_processor = MagicMock()
                mock_storage = MagicMock()
                
                escriba = Escriba(
                    processor=mock_processor,
                    storage=mock_storage
                )
                
                with pytest.raises(ImportError, match="pyperclip"):
                    escriba.capture_from_clipboard()

    @patch("memorytwin.escriba.escriba.console")
    @patch("memorytwin.escriba.escriba.pyperclip", create=True)
    def test_capture_from_clipboard_empty(self, mock_pyperclip, mock_console):
        """Test que falla con clipboard vacío."""
        mock_pyperclip.paste.return_value = ""
        
        # Necesitamos importar pyperclip en el módulo
        import sys
        sys.modules["pyperclip"] = mock_pyperclip
        
        mock_processor = MagicMock()
        mock_storage = MagicMock()
        
        escriba = Escriba(
            processor=mock_processor,
            storage=mock_storage
        )
        
        with pytest.raises(ValueError, match="clipboard está vacío"):
            escriba.capture_from_clipboard()


class TestEscribaInputValidation:
    """Tests para validación de entrada en captura."""

    @patch("memorytwin.escriba.escriba.console")
    @pytest.mark.asyncio
    async def test_capture_creates_processed_input(
        self, mock_console
    ):
        """Test que crea ProcessedInput correctamente."""
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
            thinking_text="Mi pensamiento",
            user_prompt="Prompt original",
            code_changes="código aquí"
        )
        
        assert captured_input is not None
        assert captured_input.raw_text == "Mi pensamiento"
        assert captured_input.user_prompt == "Prompt original"
        assert captured_input.code_changes == "código aquí"
        assert captured_input.source == "api"
        assert isinstance(captured_input.captured_at, datetime)
