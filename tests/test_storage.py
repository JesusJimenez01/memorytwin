"""
Tests para el módulo de almacenamiento
======================================
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from memorytwin.models import (
    Episode,
    EpisodeType,
    ReasoningTrace,
    MemoryQuery,
)


@pytest.fixture
def temp_storage():
    """Crear almacenamiento temporal para tests."""
    from memorytwin.escriba.storage import MemoryStorage
    
    with tempfile.TemporaryDirectory() as tmpdir:
        chroma_path = Path(tmpdir) / "chroma"
        sqlite_path = Path(tmpdir) / "test.db"
        
        storage = MemoryStorage(
            chroma_path=chroma_path,
            sqlite_path=sqlite_path
        )
        
        yield storage


@pytest.fixture
def sample_episode():
    """Crear episodio de ejemplo."""
    return Episode(
        task="Implementar autenticación JWT",
        context="API REST con FastAPI, PostgreSQL como BD",
        reasoning_trace=ReasoningTrace(
            raw_thinking="El usuario necesita autenticación. Consideré varias opciones: "
                        "sessions con Redis (descartado por infraestructura), "
                        "OAuth2 (muy complejo), JWT (elegido por simplicidad).",
            alternatives_considered=["Sessions con Redis", "OAuth2 completo"],
            decision_factors=["Stateless", "Escalabilidad", "Simplicidad"],
            confidence_level=0.9
        ),
        solution="from jose import jwt\n\ndef create_token(user_id): ...",
        solution_summary="JWT con PyJWT, tokens de 24h, refresh tokens de 7 días",
        episode_type=EpisodeType.FEATURE,
        tags=["auth", "jwt", "security", "fastapi"],
        files_affected=["auth/jwt.py", "auth/routes.py"],
        lessons_learned=[
            "Siempre validar el algoritmo del JWT",
            "Usar refresh tokens para mejor UX"
        ],
        source_assistant="copilot",
        project_name="test-api"
    )


class TestMemoryStorage:
    """Tests para MemoryStorage."""
    
    def test_store_and_retrieve_episode(self, temp_storage, sample_episode):
        """Test de almacenamiento y recuperación."""
        # Almacenar
        episode_id = temp_storage.store_episode(sample_episode)
        
        assert episode_id is not None
        assert episode_id == str(sample_episode.id)
        
        # Recuperar
        retrieved = temp_storage.get_episode_by_id(episode_id)
        
        assert retrieved is not None
        assert retrieved.task == sample_episode.task
        assert retrieved.solution_summary == sample_episode.solution_summary
        assert retrieved.episode_type == sample_episode.episode_type
    
    def test_search_episodes(self, temp_storage, sample_episode):
        """Test de búsqueda semántica."""
        # Almacenar
        temp_storage.store_episode(sample_episode)
        
        # Buscar
        query = MemoryQuery(
            query="autenticación JWT",
            top_k=5
        )
        
        results = temp_storage.search_episodes(query)
        
        assert len(results) > 0
        assert results[0].episode.task == sample_episode.task
        assert results[0].relevance_score > 0
    
    def test_get_episodes_by_project(self, temp_storage, sample_episode):
        """Test de filtrado por proyecto."""
        temp_storage.store_episode(sample_episode)
        
        # Buscar por proyecto correcto
        episodes = temp_storage.get_episodes_by_project("test-api")
        assert len(episodes) == 1
        
        # Buscar por proyecto incorrecto
        episodes = temp_storage.get_episodes_by_project("otro-proyecto")
        assert len(episodes) == 0
    
    def test_get_timeline(self, temp_storage, sample_episode):
        """Test de obtención de timeline."""
        temp_storage.store_episode(sample_episode)
        
        timeline = temp_storage.get_timeline(project_name="test-api")
        
        assert len(timeline) == 1
        assert timeline[0].task == sample_episode.task
    
    def test_get_lessons_learned(self, temp_storage, sample_episode):
        """Test de obtención de lecciones."""
        temp_storage.store_episode(sample_episode)
        
        lessons = temp_storage.get_lessons_learned(project_name="test-api")
        
        assert len(lessons) == 2
        assert any("JWT" in l["lesson"] for l in lessons)
    
    def test_get_statistics(self, temp_storage, sample_episode):
        """Test de estadísticas."""
        temp_storage.store_episode(sample_episode)
        
        stats = temp_storage.get_statistics()
        
        assert stats["total_episodes"] == 1
        assert stats["by_type"]["feature"] == 1
        assert stats["by_assistant"]["copilot"] == 1
    
    def test_multiple_episodes(self, temp_storage):
        """Test con múltiples episodios."""
        episodes = [
            Episode(
                task=f"Tarea {i}",
                context=f"Contexto {i}",
                reasoning_trace=ReasoningTrace(raw_thinking=f"Pensamiento {i}"),
                solution=f"Código {i}",
                solution_summary=f"Resumen {i}",
                project_name="multi-test"
            )
            for i in range(5)
        ]
        
        for ep in episodes:
            temp_storage.store_episode(ep)
        
        stats = temp_storage.get_statistics("multi-test")
        assert stats["total_episodes"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
