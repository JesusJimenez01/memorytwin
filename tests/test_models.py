"""
Tests para el módulo Escriba
============================
"""

import pytest
from datetime import datetime
from uuid import UUID

from memorytwin.models import (
    Episode,
    EpisodeType,
    ProcessedInput,
    ReasoningTrace,
    MemoryQuery,
)


class TestModels:
    """Tests para los modelos Pydantic."""
    
    def test_reasoning_trace_creation(self):
        """Test de creación de ReasoningTrace."""
        rt = ReasoningTrace(
            raw_thinking="Consideré usar JWT por su naturaleza stateless",
            alternatives_considered=["Sessions", "OAuth2"],
            decision_factors=["Escalabilidad", "Simplicidad"],
            confidence_level=0.85
        )
        
        assert rt.raw_thinking == "Consideré usar JWT por su naturaleza stateless"
        assert len(rt.alternatives_considered) == 2
        assert rt.confidence_level == 0.85
    
    def test_episode_creation(self):
        """Test de creación de Episode."""
        episode = Episode(
            task="Implementar autenticación",
            context="API REST en FastAPI",
            reasoning_trace=ReasoningTrace(
                raw_thinking="Elegí JWT por escalabilidad"
            ),
            solution="from jose import jwt...",
            solution_summary="JWT con tokens de 24h",
            episode_type=EpisodeType.FEATURE,
            tags=["auth", "jwt"],
            project_name="test-project"
        )
        
        assert isinstance(episode.id, UUID)
        assert isinstance(episode.timestamp, datetime)
        assert episode.task == "Implementar autenticación"
        assert episode.episode_type == EpisodeType.FEATURE
        assert episode.success is True  # Default
    
    def test_episode_types(self):
        """Test de tipos de episodio."""
        assert EpisodeType.DECISION.value == "decision"
        assert EpisodeType.BUG_FIX.value == "bug_fix"
        assert EpisodeType.REFACTOR.value == "refactor"
        assert EpisodeType.FEATURE.value == "feature"
    
    def test_processed_input(self):
        """Test de ProcessedInput."""
        pi = ProcessedInput(
            raw_text="Thinking text here",
            user_prompt="Implementa autenticación",
            code_changes="def auth(): pass",
            source="clipboard"
        )
        
        assert pi.raw_text == "Thinking text here"
        assert pi.source == "clipboard"
        assert isinstance(pi.captured_at, datetime)
    
    def test_memory_query(self):
        """Test de MemoryQuery."""
        query = MemoryQuery(
            query="¿Por qué JWT?",
            project_filter="test-project",
            type_filter=EpisodeType.DECISION,
            top_k=3
        )
        
        assert query.query == "¿Por qué JWT?"
        assert query.top_k == 3
    
    def test_episode_json_serialization(self):
        """Test de serialización JSON del Episode."""
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
    """Tests específicos para ProcessedInput."""
    
    def test_minimal_input(self):
        """Test con input mínimo."""
        pi = ProcessedInput(raw_text="Solo texto")
        
        assert pi.raw_text == "Solo texto"
        assert pi.user_prompt is None
        assert pi.code_changes is None
        assert pi.source == "manual"
    
    def test_full_input(self):
        """Test con input completo."""
        pi = ProcessedInput(
            raw_text="Razonamiento completo del modelo...",
            user_prompt="Prompt del usuario",
            code_changes="```python\ndef hello(): pass\n```",
            source="mcp"
        )
        
        assert pi.source == "mcp"
        assert "Razonamiento" in pi.raw_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
