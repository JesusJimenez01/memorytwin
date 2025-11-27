"""
Tests para MetaMemory y consolidación
=====================================
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from memorytwin.models import (
    MetaMemory,
    MetaMemorySearchResult,
    Episode,
    EpisodeType,
    ReasoningTrace
)


class TestMetaMemoryModel:
    """Tests para el modelo MetaMemory."""
    
    def test_create_minimal_meta_memory(self):
        """Crear MetaMemory con campos mínimos."""
        mm = MetaMemory(
            pattern="Patrón de prueba",
            pattern_summary="Resumen del patrón"
        )
        
        assert mm.pattern == "Patrón de prueba"
        assert mm.pattern_summary == "Resumen del patrón"
        assert mm.id is not None
        assert mm.created_at is not None
        assert mm.confidence == 0.5  # default
        assert mm.episode_count == 0
    
    def test_create_full_meta_memory(self):
        """Crear MetaMemory con todos los campos."""
        source_ids = [uuid4(), uuid4(), uuid4()]
        
        mm = MetaMemory(
            pattern="Patrón de manejo de errores en APIs REST",
            pattern_summary="Usar try-except con logging estructurado",
            lessons=["Siempre loguear el contexto", "Usar códigos HTTP apropiados"],
            best_practices=["Centralizar manejo de errores", "Usar middleware"],
            antipatterns=["Capturar Exception genérica", "Silenciar errores"],
            exceptions=["Errores de validación son 400, no 500"],
            edge_cases=["Timeouts en cascada"],
            contexts=["APIs REST", "Microservicios"],
            technologies=["Python", "FastAPI", "SQLAlchemy"],
            source_episode_ids=source_ids,
            episode_count=3,
            confidence=0.85,
            coherence_score=0.9,
            project_name="test-project",
            tags=["api", "errores", "python"]
        )
        
        assert len(mm.lessons) == 2
        assert len(mm.best_practices) == 2
        assert len(mm.antipatterns) == 2
        assert len(mm.source_episode_ids) == 3
        assert mm.confidence == 0.85
        assert "FastAPI" in mm.technologies
    
    def test_meta_memory_defaults(self):
        """Verificar valores por defecto."""
        mm = MetaMemory(
            pattern="Test",
            pattern_summary="Test"
        )
        
        assert mm.lessons == []
        assert mm.best_practices == []
        assert mm.antipatterns == []
        assert mm.exceptions == []
        assert mm.edge_cases == []
        assert mm.contexts == []
        assert mm.technologies == []
        assert mm.source_episode_ids == []
        assert mm.episode_count == 0
        assert mm.confidence == 0.5
        assert mm.coherence_score == 0.5
        assert mm.project_name == "default"
        assert mm.tags == []
        assert mm.access_count == 0
        assert mm.last_accessed is None
    
    def test_confidence_bounds(self):
        """Confidence debe estar entre 0 y 1."""
        # Valor válido
        mm = MetaMemory(
            pattern="Test",
            pattern_summary="Test",
            confidence=0.95
        )
        assert mm.confidence == 0.95
        
        # Valores fuera de rango deben fallar
        with pytest.raises(ValueError):
            MetaMemory(
                pattern="Test",
                pattern_summary="Test",
                confidence=1.5
            )
        
        with pytest.raises(ValueError):
            MetaMemory(
                pattern="Test",
                pattern_summary="Test",
                confidence=-0.1
            )
    
    def test_coherence_score_bounds(self):
        """Coherence score debe estar entre 0 y 1."""
        mm = MetaMemory(
            pattern="Test",
            pattern_summary="Test",
            coherence_score=0.0
        )
        assert mm.coherence_score == 0.0
        
        mm2 = MetaMemory(
            pattern="Test",
            pattern_summary="Test",
            coherence_score=1.0
        )
        assert mm2.coherence_score == 1.0
    
    def test_access_count_non_negative(self):
        """Access count no puede ser negativo."""
        with pytest.raises(ValueError):
            MetaMemory(
                pattern="Test",
                pattern_summary="Test",
                access_count=-1
            )


class TestMetaMemorySearchResult:
    """Tests para MetaMemorySearchResult."""
    
    def test_create_search_result(self):
        """Crear resultado de búsqueda de meta-memoria."""
        mm = MetaMemory(
            pattern="Test pattern",
            pattern_summary="Test summary"
        )
        
        result = MetaMemorySearchResult(
            meta_memory=mm,
            relevance_score=0.85,
            match_reason="Patrón consolidado de 5 episodios"
        )
        
        assert result.meta_memory == mm
        assert result.relevance_score == 0.85
        assert "5 episodios" in result.match_reason
    
    def test_relevance_score_bounds(self):
        """Relevance score debe estar entre 0 y 1."""
        mm = MetaMemory(pattern="Test", pattern_summary="Test")
        
        with pytest.raises(ValueError):
            MetaMemorySearchResult(
                meta_memory=mm,
                relevance_score=1.5
            )
        
        with pytest.raises(ValueError):
            MetaMemorySearchResult(
                meta_memory=mm,
                relevance_score=-0.1
            )


class TestConsolidationHelpers:
    """Tests para funciones auxiliares de consolidación."""
    
    def test_format_episode_for_consolidation(self):
        """Verificar formateo de episodio para consolidación."""
        from memorytwin.consolidation import format_episode_for_consolidation
        
        episode = Episode(
            id=uuid4(),
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            task="Implementar autenticación JWT",
            context="API REST con FastAPI",
            reasoning_trace=ReasoningTrace(
                raw_thinking="Consideré OAuth2 pero JWT es más simple para este caso..."
            ),
            solution="Implementar con pyjwt",
            solution_summary="Usar pyjwt para generar tokens",
            lessons_learned=["Rotar claves regularmente", "Usar tiempos cortos de expiración"],
            tags=["auth", "jwt", "security"]
        )
        
        formatted = format_episode_for_consolidation(episode)
        
        assert "Implementar autenticación JWT" in formatted
        assert "2024-01-15" in formatted
        assert "FastAPI" in formatted
        assert "OAuth2" in formatted  # Del thinking truncado
        assert "auth" in formatted


class TestMetaMemoryTimestamps:
    """Tests para manejo de timestamps en MetaMemory."""
    
    def test_created_at_auto_generated(self):
        """created_at se genera automáticamente."""
        before = datetime.now(timezone.utc)
        mm = MetaMemory(pattern="Test", pattern_summary="Test")
        after = datetime.now(timezone.utc)
        
        assert before <= mm.created_at <= after
    
    def test_updated_at_auto_generated(self):
        """updated_at se genera automáticamente."""
        mm = MetaMemory(pattern="Test", pattern_summary="Test")
        
        assert mm.updated_at is not None
        assert mm.created_at == mm.updated_at  # Inicialmente iguales
    
    def test_last_accessed_initially_none(self):
        """last_accessed es None inicialmente."""
        mm = MetaMemory(pattern="Test", pattern_summary="Test")
        
        assert mm.last_accessed is None
