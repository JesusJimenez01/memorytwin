"""
Tests para el módulo de scoring (forgetting curve)
==================================================
"""

import math
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from memorytwin.models import Episode, EpisodeType, ReasoningTrace
from memorytwin.scoring import (
    compute_hybrid_score,
    DEFAULT_DECAY_RATE as DECAY_RATE,
    DEFAULT_ACCESS_BOOST as ACCESS_BOOST
)


def create_test_episode(
    days_ago: int = 0,
    access_count: int = 0,
    importance_score: float = 1.0
) -> Episode:
    """Crear episodio de prueba con parámetros configurables."""
    timestamp = datetime.now(timezone.utc) - timedelta(days=days_ago)
    
    return Episode(
        id=uuid4(),
        timestamp=timestamp,
        task="Tarea de prueba",
        context="Contexto de prueba",
        reasoning_trace=ReasoningTrace(raw_thinking="Pensamiento de prueba"),
        solution="Solución de prueba",
        solution_summary="Resumen de prueba",
        importance_score=importance_score,
        access_count=access_count,
        last_accessed=datetime.now(timezone.utc) if access_count > 0 else None
    )


class TestComputeHybridScore:
    """Tests para compute_hybrid_score."""
    
    def test_same_day_episode_no_decay(self):
        """Un episodio del mismo día no tiene decay significativo."""
        episode = create_test_episode(days_ago=0)
        now = datetime.now(timezone.utc)
        
        score = compute_hybrid_score(episode, semantic_score=1.0, now=now)
        
        # Sin decay, sin boost adicional, score ≈ 1.0
        assert score >= 0.99  # Pequeño margen por precisión float
    
    def test_old_episode_has_decay(self):
        """Un episodio antiguo tiene decay significativo."""
        episode = create_test_episode(days_ago=30)
        now = datetime.now(timezone.utc)
        
        score = compute_hybrid_score(episode, semantic_score=1.0, now=now)
        
        # Con 30 días, decay = exp(-0.05 * 30) ≈ 0.22
        expected_decay = math.exp(-DECAY_RATE * 30)
        assert 0.2 < score < 0.3  # Aproximado
        assert abs(score - expected_decay) < 0.05
    
    def test_access_boost_increases_score(self):
        """Episodios accedidos frecuentemente tienen mayor score."""
        episode_no_access = create_test_episode(days_ago=10, access_count=0)
        episode_many_access = create_test_episode(days_ago=10, access_count=10)
        now = datetime.now(timezone.utc)
        
        score_no = compute_hybrid_score(episode_no_access, 1.0, now)
        score_many = compute_hybrid_score(episode_many_access, 1.0, now)
        
        # El episodio con más accesos debe tener mayor score
        assert score_many > score_no
        
        # El boost debe ser proporcional: 1 + 0.1 * 10 = 2.0
        expected_ratio = 1 + ACCESS_BOOST * 10
        actual_ratio = score_many / score_no
        assert abs(actual_ratio - expected_ratio) < 0.1
    
    def test_importance_affects_score(self):
        """Episodios con mayor importance_score tienen mayor score."""
        episode_low = create_test_episode(days_ago=5, importance_score=0.5)
        episode_high = create_test_episode(days_ago=5, importance_score=1.0)
        now = datetime.now(timezone.utc)
        
        score_low = compute_hybrid_score(episode_low, 1.0, now)
        score_high = compute_hybrid_score(episode_high, 1.0, now)
        
        assert score_high > score_low
        # El ratio debe ser 2:1
        assert abs(score_high / score_low - 2.0) < 0.1
    
    def test_semantic_score_multiplied(self):
        """El score semántico se multiplica correctamente."""
        episode = create_test_episode(days_ago=0)
        now = datetime.now(timezone.utc)
        
        score_full = compute_hybrid_score(episode, semantic_score=1.0, now=now)
        score_half = compute_hybrid_score(episode, semantic_score=0.5, now=now)
        
        # Con episodio reciente sin boost, la relación debe ser 2:1
        assert abs(score_full / score_half - 2.0) < 0.1
    
    def test_combined_factors(self):
        """Verificar fórmula completa: semantic * decay * boost * importance."""
        episode = create_test_episode(
            days_ago=20,
            access_count=5,
            importance_score=0.8
        )
        now = datetime.now(timezone.utc)
        semantic_score = 0.9
        
        score = compute_hybrid_score(episode, semantic_score, now)
        
        # Calcular esperado manualmente
        decay = math.exp(-DECAY_RATE * 20)  # ≈ 0.368
        boost = 1 + ACCESS_BOOST * 5  # = 1.5
        expected = semantic_score * decay * boost * 0.8
        
        assert abs(score - expected) < 0.01
    
    def test_zero_semantic_gives_zero(self):
        """Score semántico cero da score final cero."""
        episode = create_test_episode(days_ago=0, access_count=100)
        now = datetime.now(timezone.utc)
        
        score = compute_hybrid_score(episode, semantic_score=0.0, now=now)
        
        assert score == 0.0
    
    def test_very_old_episode_near_zero(self):
        """Un episodio muy antiguo tiene score cercano a cero."""
        episode = create_test_episode(days_ago=365)  # 1 año
        now = datetime.now(timezone.utc)
        
        score = compute_hybrid_score(episode, semantic_score=1.0, now=now)
        
        # decay = exp(-0.05 * 365) ≈ 0.000000001
        assert score < 0.001


class TestDecayConstants:
    """Tests para las constantes de decay."""
    
    def test_decay_rate_reasonable(self):
        """DECAY_RATE está en rango razonable."""
        # Un decay muy rápido (>0.2) olvidaría todo en días
        # Un decay muy lento (<0.01) no olvidaría nada
        assert 0.01 < DECAY_RATE < 0.2
    
    def test_access_boost_reasonable(self):
        """ACCESS_BOOST está en rango razonable."""
        # Un boost muy alto (>0.5) haría que accesos dominen todo
        # Un boost muy bajo (<0.01) no tendría efecto
        assert 0.01 < ACCESS_BOOST < 0.5
    
    def test_half_life_calculation(self):
        """Verificar que la vida media es razonable."""
        # Vida media: cuando decay = 0.5
        # 0.5 = exp(-DECAY_RATE * t) => t = ln(2) / DECAY_RATE
        half_life_days = math.log(2) / DECAY_RATE
        
        # Debería ser entre 1 semana y 3 meses
        assert 7 < half_life_days < 90
