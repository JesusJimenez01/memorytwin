"""
Tests para el motor RAG del Oráculo
===================================

Tests unitarios para RAGEngine con mocks.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from uuid import uuid4

from memorytwin.models import (
    Episode,
    EpisodeType,
    ReasoningTrace,
    MemoryQuery,
    MemorySearchResult,
)
from memorytwin.oraculo.rag_engine import RAGEngine, ORACLE_SYSTEM_PROMPT


class TestRAGEngine:
    """Tests para RAGEngine."""

    @pytest.fixture
    def mock_settings(self):
        """Mock de configuración."""
        with patch("memorytwin.oraculo.rag_engine.get_settings") as mock:
            settings = MagicMock()
            settings.google_api_key = "test-api-key"
            settings.llm_model = "gemini-2.0-flash"
            mock.return_value = settings
            yield mock

    @pytest.fixture
    def mock_genai(self):
        """Mock de Google Generative AI."""
        with patch("memorytwin.oraculo.rag_engine.genai") as mock:
            yield mock

    @pytest.fixture
    def mock_storage(self):
        """Mock del almacenamiento."""
        storage = MagicMock()
        return storage

    @pytest.fixture
    def sample_episode(self):
        """Episodio de ejemplo."""
        return Episode(
            id=uuid4(),
            task="Implementar autenticación JWT",
            context="API REST con FastAPI",
            reasoning_trace=ReasoningTrace(
                raw_thinking="Elegí JWT por escalabilidad",
                alternatives_considered=["Sessions", "OAuth2"],
                decision_factors=["Stateless", "Escalabilidad"]
            ),
            solution="from jose import jwt",
            solution_summary="JWT con tokens de 24h",
            episode_type=EpisodeType.FEATURE,
            tags=["auth", "jwt"],
            files_affected=["auth.py"],
            lessons_learned=["Validar algoritmo JWT"],
            project_name="test-project",
            source_assistant="copilot"
        )

    @pytest.fixture
    def sample_search_result(self, sample_episode):
        """Resultado de búsqueda de ejemplo."""
        return MemorySearchResult(
            episode=sample_episode,
            relevance_score=0.92
        )

    def test_rag_engine_init(self, mock_settings, mock_genai, mock_storage):
        """Test de inicialización de RAGEngine."""
        engine = RAGEngine(storage=mock_storage)
        
        mock_genai.configure.assert_called_once_with(api_key="test-api-key")
        assert engine.storage == mock_storage
        assert engine.api_key == "test-api-key"

    def test_rag_engine_init_custom_api_key(self, mock_settings, mock_genai, mock_storage):
        """Test de inicialización con API key personalizada."""
        engine = RAGEngine(storage=mock_storage, api_key="custom-key")
        
        mock_genai.configure.assert_called_once_with(api_key="custom-key")
        assert engine.api_key == "custom-key"

    def test_rag_engine_init_no_api_key_raises(self, mock_genai, mock_storage):
        """Test que falla sin API key."""
        with patch("memorytwin.oraculo.rag_engine.get_settings") as mock:
            settings = MagicMock()
            settings.google_api_key = None
            mock.return_value = settings
            
            with pytest.raises(ValueError, match="Se requiere GOOGLE_API_KEY"):
                RAGEngine(storage=mock_storage)

    def test_build_context(self, mock_settings, mock_genai, mock_storage, sample_search_result):
        """Test de construcción de contexto."""
        engine = RAGEngine(storage=mock_storage)
        
        context = engine._build_context([sample_search_result])
        
        assert "## EPISODIOS DE MEMORIA RELEVANTES" in context
        assert "Episodio 1" in context
        assert "Relevancia: 92%" in context
        assert "Implementar autenticación JWT" in context
        assert "JWT por escalabilidad" in context
        assert "Sessions" in context
        assert "Stateless" in context
        assert "Validar algoritmo JWT" in context
        assert "auth" in context

    def test_build_context_multiple_episodes(
        self, mock_settings, mock_genai, mock_storage, sample_episode
    ):
        """Test de contexto con múltiples episodios."""
        episode2 = Episode(
            id=uuid4(),
            task="Añadir rate limiting",
            context="Protección de API",
            reasoning_trace=ReasoningTrace(raw_thinking="Rate limit para seguridad"),
            solution="rate_limit()",
            solution_summary="Rate limiting implementado",
            episode_type=EpisodeType.FEATURE,
            project_name="test"
        )
        
        results = [
            MemorySearchResult(episode=sample_episode, relevance_score=0.95),
            MemorySearchResult(episode=episode2, relevance_score=0.78)
        ]
        
        engine = RAGEngine(storage=mock_storage)
        context = engine._build_context(results)
        
        assert "Episodio 1" in context
        assert "Episodio 2" in context
        assert "95%" in context
        assert "78%" in context
        assert "autenticación JWT" in context
        assert "rate limiting" in context

    @pytest.mark.asyncio
    async def test_query_no_results(self, mock_settings, mock_genai, mock_storage):
        """Test de query sin resultados."""
        mock_storage.search_episodes.return_value = []
        
        engine = RAGEngine(storage=mock_storage)
        
        result = await engine.query("¿Por qué usamos GraphQL?")
        
        assert "No encontré episodios" in result["answer"]
        assert result["episodes_used"] == []
        assert result["context_provided"] is False

    @pytest.mark.asyncio
    async def test_query_with_results(
        self, mock_settings, mock_genai, mock_storage, sample_search_result
    ):
        """Test de query con resultados."""
        mock_storage.search_episodes.return_value = [sample_search_result]
        
        # Mock del modelo
        mock_response = MagicMock()
        mock_response.text = "JWT fue elegido por su naturaleza stateless y escalabilidad."
        
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model
        
        engine = RAGEngine(storage=mock_storage)
        
        result = await engine.query(
            question="¿Por qué usamos JWT?",
            project_name="test-project",
            top_k=3
        )
        
        assert result["context_provided"] is True
        assert len(result["episodes_used"]) == 1
        assert result["relevance_scores"][0] == 0.92
        assert "JWT" in result["answer"]
        
        # Verificar llamada a storage
        mock_storage.search_episodes.assert_called_once()
        call_args = mock_storage.search_episodes.call_args
        query = call_args.args[0]
        assert query.query == "¿Por qué usamos JWT?"
        assert query.project_filter == "test-project"
        assert query.top_k == 3

    def test_query_sync(
        self, mock_settings, mock_genai, mock_storage, sample_search_result
    ):
        """Test de versión síncrona de query."""
        mock_storage.search_episodes.return_value = [sample_search_result]
        
        mock_response = MagicMock()
        mock_response.text = "Respuesta sobre JWT"
        
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model
        
        engine = RAGEngine(storage=mock_storage)
        
        result = engine.query_sync("¿Por qué JWT?")
        
        assert result["context_provided"] is True

    def test_get_timeline(self, mock_settings, mock_genai, mock_storage, sample_episode):
        """Test de obtención de timeline."""
        mock_storage.get_timeline.return_value = [sample_episode]
        
        engine = RAGEngine(storage=mock_storage)
        
        timeline = engine.get_timeline(project_name="test", limit=10)
        
        assert len(timeline) == 1
        assert timeline[0]["task"] == "Implementar autenticación JWT"
        assert timeline[0]["type"] == "feature"
        assert "date" in timeline[0]
        assert "time" in timeline[0]
        assert timeline[0]["assistant"] == "copilot"
        
        mock_storage.get_timeline.assert_called_once_with(
            project_name="test",
            limit=10
        )

    def test_get_lessons(self, mock_settings, mock_genai, mock_storage):
        """Test de obtención de lecciones."""
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

    def test_get_statistics(self, mock_settings, mock_genai, mock_storage):
        """Test de obtención de estadísticas."""
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
    """Tests para el prompt del sistema del Oráculo."""

    def test_prompt_contains_role_description(self):
        """Test que el prompt describe el rol."""
        assert "Oráculo" in ORACLE_SYSTEM_PROMPT
        assert "Memory Twin" in ORACLE_SYSTEM_PROMPT

    def test_prompt_mentions_episodic_memory(self):
        """Test que menciona memoria episódica."""
        assert "episodios de memoria" in ORACLE_SYSTEM_PROMPT

    def test_prompt_has_instructions(self):
        """Test que tiene instrucciones."""
        assert "INSTRUCCIONES" in ORACLE_SYSTEM_PROMPT
        assert "basándote ÚNICAMENTE" in ORACLE_SYSTEM_PROMPT

    def test_prompt_mentions_format(self):
        """Test que menciona formato."""
        assert "Markdown" in ORACLE_SYSTEM_PROMPT


class TestRAGEngineEdgeCases:
    """Tests para casos especiales del RAGEngine."""

    @pytest.fixture
    def mock_settings(self):
        """Mock de configuración."""
        with patch("memorytwin.oraculo.rag_engine.get_settings") as mock:
            settings = MagicMock()
            settings.google_api_key = "test-api-key"
            settings.llm_model = "gemini-2.0-flash"
            mock.return_value = settings
            yield mock

    @pytest.fixture
    def mock_genai(self):
        """Mock de Google Generative AI."""
        with patch("memorytwin.oraculo.rag_engine.genai") as mock:
            yield mock

    def test_build_context_empty_fields(self, mock_settings, mock_genai):
        """Test de contexto con campos vacíos."""
        episode = Episode(
            id=uuid4(),
            task="Task simple",
            context="Contexto básico",
            reasoning_trace=ReasoningTrace(
                raw_thinking="Pensamiento",
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
        
        assert "No documentadas" in context or "Ninguna documentada" in context

    def test_timeline_formatting(self, mock_settings, mock_genai):
        """Test de formato de timeline."""
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
