"""
Tests para el módulo processor (ThoughtProcessor)
==================================================

Tests unitarios para el procesador de pensamientos con mocks
para evitar llamadas reales al LLM.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from memorytwin.models import (
    Episode,
    EpisodeType,
    ProcessedInput,
    ReasoningTrace,
)
from memorytwin.escriba.processor import ThoughtProcessor, STRUCTURING_PROMPT


class TestThoughtProcessor:
    """Tests para ThoughtProcessor."""

    @pytest.fixture
    def mock_settings(self):
        """Mock de configuración."""
        with patch("memorytwin.escriba.processor.get_settings") as mock:
            settings = MagicMock()
            settings.google_api_key = "test-api-key"
            settings.llm_model = "gemini-2.0-flash"
            settings.llm_temperature = 0.3
            mock.return_value = settings
            yield mock

    @pytest.fixture
    def mock_genai(self):
        """Mock de Google Generative AI."""
        with patch("memorytwin.escriba.processor.genai") as mock:
            yield mock

    @pytest.fixture
    def sample_input(self):
        """Input de ejemplo para tests."""
        return ProcessedInput(
            raw_text="Consideré usar JWT porque es stateless y escala bien. "
                     "Descartado sessions por requerir Redis.",
            user_prompt="Implementa autenticación",
            code_changes="def create_token(user): pass",
            source="test"
        )

    @pytest.fixture
    def sample_llm_response(self):
        """Respuesta de ejemplo del LLM."""
        return {
            "task": "Implementar autenticación JWT",
            "context": "API REST con FastAPI",
            "reasoning_trace": {
                "raw_thinking": "Elegí JWT por escalabilidad",
                "alternatives_considered": ["Sessions con Redis"],
                "decision_factors": ["Stateless", "Escalabilidad"],
                "confidence_level": 0.85
            },
            "solution": "from jose import jwt",
            "solution_summary": "JWT con tokens de 24h",
            "episode_type": "feature",
            "tags": ["auth", "jwt"],
            "files_affected": ["auth.py"],
            "lessons_learned": ["Validar algoritmo JWT"]
        }

    def test_processor_init_with_api_key(self, mock_settings, mock_genai):
        """Test de inicialización con API key."""
        processor = ThoughtProcessor(api_key="custom-key")
        
        mock_genai.configure.assert_called_once_with(api_key="custom-key")
        assert processor.api_key == "custom-key"

    def test_processor_init_from_settings(self, mock_settings, mock_genai):
        """Test de inicialización desde configuración."""
        processor = ThoughtProcessor()
        
        mock_genai.configure.assert_called_once_with(api_key="test-api-key")
        assert processor.api_key == "test-api-key"

    def test_processor_init_no_api_key_raises(self, mock_genai):
        """Test que falla sin API key."""
        with patch("memorytwin.escriba.processor.get_settings") as mock:
            settings = MagicMock()
            settings.google_api_key = None
            mock.return_value = settings
            
            with pytest.raises(ValueError, match="Se requiere GOOGLE_API_KEY"):
                ThoughtProcessor()

    def test_build_user_prompt_minimal(self, mock_settings, mock_genai):
        """Test de construcción de prompt mínimo."""
        processor = ThoughtProcessor()
        
        raw_input = ProcessedInput(
            raw_text="Pensamiento simple",
            source="test"
        )
        
        prompt = processor._build_user_prompt(raw_input)
        
        assert "## TEXTO DE RAZONAMIENTO (THINKING):" in prompt
        assert "Pensamiento simple" in prompt
        assert "PROMPT ORIGINAL" not in prompt
        assert "CAMBIOS DE CÓDIGO" not in prompt

    def test_build_user_prompt_full(self, mock_settings, mock_genai, sample_input):
        """Test de construcción de prompt completo."""
        processor = ThoughtProcessor()
        
        prompt = processor._build_user_prompt(sample_input)
        
        assert "## TEXTO DE RAZONAMIENTO (THINKING):" in prompt
        assert "Consideré usar JWT" in prompt
        assert "## PROMPT ORIGINAL DEL USUARIO:" in prompt
        assert "Implementa autenticación" in prompt
        assert "## CAMBIOS DE CÓDIGO:" in prompt
        assert "def create_token" in prompt

    def test_build_episode_from_data(self, mock_settings, mock_genai, sample_llm_response):
        """Test de construcción de Episode desde datos estructurados."""
        processor = ThoughtProcessor()
        
        episode = processor._build_episode(
            sample_llm_response,
            project_name="test-project",
            source_assistant="copilot"
        )
        
        assert episode.task == "Implementar autenticación JWT"
        assert episode.context == "API REST con FastAPI"
        assert episode.episode_type == EpisodeType.FEATURE
        assert episode.project_name == "test-project"
        assert episode.source_assistant == "copilot"
        assert "auth" in episode.tags
        assert "jwt" in episode.tags
        assert len(episode.reasoning_trace.alternatives_considered) == 1
        assert episode.reasoning_trace.confidence_level == 0.85

    def test_build_episode_invalid_type_defaults(self, mock_settings, mock_genai):
        """Test que tipo inválido usa default."""
        processor = ThoughtProcessor()
        
        data = {
            "task": "Test task",
            "episode_type": "invalid_type"
        }
        
        episode = processor._build_episode(data, "project", "assistant")
        
        assert episode.episode_type == EpisodeType.DECISION

    def test_build_episode_missing_fields(self, mock_settings, mock_genai):
        """Test con campos faltantes usa defaults."""
        processor = ThoughtProcessor()
        
        data = {}
        
        episode = processor._build_episode(data, "project", "assistant")
        
        assert episode.task == "Tarea no especificada"
        assert episode.context == "Contexto no especificado"
        assert episode.tags == []
        assert episode.lessons_learned == []

    @pytest.mark.asyncio
    async def test_process_thought_success(
        self, mock_settings, mock_genai, sample_input, sample_llm_response
    ):
        """Test de procesamiento exitoso."""
        import json
        
        # Configurar mock del modelo
        mock_response = MagicMock()
        mock_response.text = json.dumps(sample_llm_response)
        
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = ThoughtProcessor()
        
        episode = await processor.process_thought(
            sample_input,
            project_name="test-project",
            source_assistant="copilot"
        )
        
        assert episode.task == "Implementar autenticación JWT"
        assert episode.project_name == "test-project"
        mock_model.generate_content_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_thought_extracts_json_from_text(
        self, mock_settings, mock_genai, sample_input, sample_llm_response
    ):
        """Test que extrae JSON de texto con contenido adicional."""
        import json
        
        # Respuesta con texto adicional
        mock_response = MagicMock()
        mock_response.text = f"Aquí está el resultado:\n{json.dumps(sample_llm_response)}\nFin."
        
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = ThoughtProcessor()
        
        episode = await processor.process_thought(
            sample_input,
            project_name="test-project",
            source_assistant="copilot"
        )
        
        assert episode.task == "Implementar autenticación JWT"

    @pytest.mark.asyncio
    async def test_process_thought_invalid_json_raises(
        self, mock_settings, mock_genai, sample_input
    ):
        """Test que JSON inválido lanza error."""
        mock_response = MagicMock()
        mock_response.text = "esto no es json válido sin llaves"
        
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = ThoughtProcessor()
        
        # Puede lanzar ValueError o json.JSONDecodeError
        with pytest.raises((ValueError, Exception)):
            await processor.process_thought(sample_input)

    def test_process_thought_sync(self, mock_settings, mock_genai, sample_input, sample_llm_response):
        """Test de versión síncrona."""
        import json
        
        mock_response = MagicMock()
        mock_response.text = json.dumps(sample_llm_response)
        
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = ThoughtProcessor()
        
        episode = processor.process_thought_sync(
            sample_input,
            project_name="test-project",
            source_assistant="copilot"
        )
        
        assert episode.task == "Implementar autenticación JWT"


class TestStructuringPrompt:
    """Tests para el prompt de estructuración."""

    def test_prompt_contains_required_fields(self):
        """Test que el prompt menciona campos requeridos."""
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
        """Test que el prompt pide JSON."""
        assert "JSON" in STRUCTURING_PROMPT
        assert "SIEMPRE responde con JSON válido" in STRUCTURING_PROMPT
