"""
Tests para el módulo de scoring (Sistema de Refuerzo sin Olvido)
================================================================

El sistema ahora usa "refuerzo sin olvido" donde:
- No hay decay temporal por defecto
- Los episodios mantienen relevancia completa
- El access_count actúa como mecanismo de priorización
"""

import math
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from memorytwin.models import Episode, EpisodeType, ReasoningTrace
from memorytwin.scoring import (
    compute_hybrid_score,
    compute_boost,
    compute_decay,
    should_trigger_consolidation,
    get_hot_episodes_for_reclustering,
    DEFAULT_DECAY_RATE as DECAY_RATE,
    DEFAULT_ACCESS_BOOST as ACCESS_BOOST,
    CONSOLIDATION_ACCESS_THRESHOLD,
    CONSOLIDATION_EPISODE_THRESHOLD
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
    """Tests para compute_hybrid_score con sistema de refuerzo sin olvido."""
    
    def test_same_day_episode_no_decay(self):
        """Un episodio del mismo día mantiene score completo."""
        episode = create_test_episode(days_ago=0)
        now = datetime.now(timezone.utc)
        
        score = compute_hybrid_score(episode, semantic_score=1.0, now=now)
        
        # Sin decay por defecto, score = 1.0
        assert score >= 0.99
    
    def test_old_episode_no_decay_by_default(self):
        """Un episodio antiguo NO tiene decay por defecto (refuerzo sin olvido)."""
        episode = create_test_episode(days_ago=30)
        now = datetime.now(timezone.utc)
        
        # Sin use_decay=True, no hay decaimiento
        score = compute_hybrid_score(episode, semantic_score=1.0, now=now)
        
        # El episodio antiguo mantiene relevancia completa
        assert score >= 0.99
    
    def test_old_episode_with_decay_enabled(self):
        """Un episodio antiguo tiene decay SI se habilita explícitamente."""
        episode = create_test_episode(days_ago=30)
        now = datetime.now(timezone.utc)
        
        # Con use_decay=True, sí hay decaimiento
        score = compute_hybrid_score(episode, semantic_score=1.0, now=now, use_decay=True)
        
        # Con 30 días, decay = exp(-0.05 * 30) ≈ 0.22
        expected_decay = math.exp(-DECAY_RATE * 30)
        assert 0.2 < score < 0.3
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
        
        # Sin decay, la relación debe ser 2:1
        assert abs(score_full / score_half - 2.0) < 0.1
    
    def test_combined_factors_no_decay(self):
        """Verificar fórmula sin decay: semantic * boost * importance."""
        episode = create_test_episode(
            days_ago=20,
            access_count=5,
            importance_score=0.8
        )
        now = datetime.now(timezone.utc)
        semantic_score = 0.9
        
        score = compute_hybrid_score(episode, semantic_score, now)
        
        # Sin decay: score = semantic * boost * importance
        boost = 1 + ACCESS_BOOST * 5  # = 1.5
        expected = semantic_score * boost * 0.8
        
        assert abs(score - expected) < 0.01
    
    def test_combined_factors_with_decay(self):
        """Verificar fórmula completa CON decay habilitado."""
        episode = create_test_episode(
            days_ago=20,
            access_count=5,
            importance_score=0.8
        )
        now = datetime.now(timezone.utc)
        semantic_score = 0.9
        
        score = compute_hybrid_score(episode, semantic_score, now, use_decay=True)
        
        # Con decay: score = semantic * decay * boost * importance
        decay = math.exp(-DECAY_RATE * 20)
        boost = 1 + ACCESS_BOOST * 5
        expected = semantic_score * decay * boost * 0.8
        
        assert abs(score - expected) < 0.01
    
    def test_zero_semantic_gives_zero(self):
        """Score semántico cero da score final cero."""
        episode = create_test_episode(days_ago=0, access_count=100)
        now = datetime.now(timezone.utc)
        
        score = compute_hybrid_score(episode, semantic_score=0.0, now=now)
        
        assert score == 0.0
    
    def test_very_old_episode_keeps_relevance(self):
        """Un episodio muy antiguo mantiene relevancia (sin decay por defecto)."""
        episode = create_test_episode(days_ago=365)  # 1 año
        now = datetime.now(timezone.utc)
        
        score = compute_hybrid_score(episode, semantic_score=1.0, now=now)
        
        # Sin decay, mantiene relevancia completa
        assert score >= 0.99


class TestConsolidationTriggers:
    """Tests para los triggers de consolidación automática."""
    
    def test_should_trigger_on_high_access(self):
        """Debe disparar consolidación con alto access_count."""
        result = should_trigger_consolidation(
            episode_access_count=CONSOLIDATION_ACCESS_THRESHOLD,
            total_unconsolidated=0
        )
        assert result is True
    
    def test_should_trigger_on_many_episodes(self):
        """Debe disparar consolidación con muchos episodios sin consolidar."""
        result = should_trigger_consolidation(
            episode_access_count=0,
            total_unconsolidated=CONSOLIDATION_EPISODE_THRESHOLD
        )
        assert result is True
    
    def test_should_not_trigger_low_activity(self):
        """No debe disparar con poca actividad."""
        result = should_trigger_consolidation(
            episode_access_count=1,
            total_unconsolidated=5
        )
        assert result is False
    
    def test_get_hot_episodes(self):
        """Debe identificar episodios calientes correctamente."""
        episodes = [
            create_test_episode(access_count=1),
            create_test_episode(access_count=CONSOLIDATION_ACCESS_THRESHOLD),
            create_test_episode(access_count=CONSOLIDATION_ACCESS_THRESHOLD + 5),
            create_test_episode(access_count=3),
        ]
        
        hot = get_hot_episodes_for_reclustering(episodes)
        
        assert len(hot) == 2  # Solo los que superan el umbral


class TestDecayConstants:
    """Tests para las constantes de decay (mantenidas por compatibilidad)."""
    
    def test_decay_rate_reasonable(self):
        """DECAY_RATE está en rango razonable."""
        assert 0.01 < DECAY_RATE < 0.2
    
    def test_access_boost_reasonable(self):
        """ACCESS_BOOST está en rango razonable."""
        assert 0.01 < ACCESS_BOOST < 0.5
    
    def test_half_life_calculation(self):
        """Verificar que la vida media es razonable (si se usa decay)."""
        half_life_days = math.log(2) / DECAY_RATE
        assert 7 < half_life_days < 90
    
    def test_consolidation_thresholds_reasonable(self):
        """Los umbrales de consolidación son razonables."""
        assert 5 <= CONSOLIDATION_ACCESS_THRESHOLD <= 20
        assert 10 <= CONSOLIDATION_EPISODE_THRESHOLD <= 50


class TestCriticalAndAntipatternModifiers:
    """Tests para los modificadores is_critical e is_antipattern."""
    
    def test_critical_episode_gets_boost(self):
        """Episodios críticos reciben un boost de 1.5x."""
        episode_normal = create_test_episode(days_ago=0)
        episode_critical = create_test_episode(days_ago=0)
        episode_critical.is_critical = True
        
        now = datetime.now(timezone.utc)
        
        score_normal = compute_hybrid_score(episode_normal, 1.0, now)
        score_critical = compute_hybrid_score(episode_critical, 1.0, now)
        
        # El episodio crítico debe tener 1.5x más score
        assert score_critical == score_normal * 1.5
    
    def test_antipattern_episode_gets_penalty(self):
        """Episodios antipattern reciben penalización a 0.3x."""
        episode_normal = create_test_episode(days_ago=0)
        episode_antipattern = create_test_episode(days_ago=0)
        episode_antipattern.is_antipattern = True
        
        now = datetime.now(timezone.utc)
        
        score_normal = compute_hybrid_score(episode_normal, 1.0, now)
        score_antipattern = compute_hybrid_score(episode_antipattern, 1.0, now)
        
        # El antipattern debe tener 0.3x del score normal
        assert score_antipattern == score_normal * 0.3
    
    def test_critical_antipattern_combination(self):
        """Un episodio marcado como crítico Y antipattern aplica ambos modificadores."""
        episode_both = create_test_episode(days_ago=0)
        episode_both.is_critical = True
        episode_both.is_antipattern = True
        
        now = datetime.now(timezone.utc)
        
        score = compute_hybrid_score(episode_both, 1.0, now)
        
        # 1.0 * 1.5 (critical) * 0.3 (antipattern) = 0.45
        assert abs(score - 0.45) < 0.01
    
    def test_antipattern_still_appears_in_results(self):
        """Antipatterns no se excluyen completamente, solo bajan en relevancia."""
        episode_antipattern = create_test_episode(days_ago=0)
        episode_antipattern.is_antipattern = True
        
        now = datetime.now(timezone.utc)
        
        score = compute_hybrid_score(episode_antipattern, 1.0, now)
        
        # Score > 0 significa que sigue apareciendo
        assert score > 0
        assert score == 0.3  # 1.0 * 0.3


class TestComputeBoost:
    """Tests para compute_boost."""
    
    def test_zero_access_returns_one(self):
        """Sin accesos, el boost es 1.0."""
        episode = create_test_episode(access_count=0)
        assert compute_boost(episode) == 1.0
    
    def test_boost_increases_with_access(self):
        """El boost aumenta con cada acceso."""
        ep1 = create_test_episode(access_count=5)
        ep2 = create_test_episode(access_count=10)
        
        assert compute_boost(ep1) < compute_boost(ep2)
        assert compute_boost(ep1) == 1 + ACCESS_BOOST * 5
        assert compute_boost(ep2) == 1 + ACCESS_BOOST * 10


class TestComputeDecay:
    """Tests para compute_decay (función de utilidad)."""
    
    def test_recent_episode_no_decay(self):
        """Episodio reciente tiene decay ~1.0."""
        episode = create_test_episode(days_ago=0)
        decay = compute_decay(episode)
        assert decay >= 0.99
    
    def test_old_episode_has_decay(self):
        """Episodio antiguo tiene decay < 1.0."""
        episode = create_test_episode(days_ago=30)
        decay = compute_decay(episode)
        assert decay < 0.5
