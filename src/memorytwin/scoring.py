"""
Sistema de Scoring Híbrido para Memory Twin
============================================

Implementa un sistema de "refuerzo sin olvido" inspirado en neurociencia:
- Similitud semántica (embeddings) como base
- Boost por uso frecuente (refuerzo positivo)
- Importancia base del episodio

NOTA: Se eliminó el decay temporal porque:
1. Evita el decaimiento innecesario de conocimiento valioso
2. Todos los episodios mantienen relevancia completa
3. El access_count actúa como mecanismo natural de priorización
4. Las meta-memorias manejan la consolidación de patrones frecuentes

La fórmula es:
    final_score = semantic_score * boost * importance_score

Donde:
- semantic_score: similitud coseno entre query y episodio
- boost: 1 + access_boost * access_count, premia episodios consultados frecuentemente
- importance_score: relevancia base asignada al episodio
"""

import math
from datetime import datetime, timezone
from typing import Optional

from memorytwin.models import Episode


# Constantes configurables
DEFAULT_DECAY_RATE = 0.05  # DEPRECATED: Ya no se usa, mantenido por compatibilidad
DEFAULT_ACCESS_BOOST = 0.1  # Boost adicional por cada acceso


# Constantes para is_critical e is_antipattern
CRITICAL_BOOST = 1.5  # Episodios críticos reciben 50% más relevancia
ANTIPATTERN_PENALTY = 0.3  # Antipatterns reducidos al 30% de relevancia (pero no excluidos)


def compute_hybrid_score(
    episode: Episode,
    semantic_score: float,
    now: Optional[datetime] = None,
    decay_rate: float = DEFAULT_DECAY_RATE,
    access_boost: float = DEFAULT_ACCESS_BOOST,
    use_decay: bool = False  # NUEVO: desactivado por defecto
) -> float:
    """
    Calcular score híbrido combinando semántica, uso e importancia.
    
    Implementa un sistema de "refuerzo sin olvido" donde:
    - La similitud semántica es la base
    - El uso frecuente (access_count) refuerza la relevancia
    - No hay penalización temporal (decay desactivado por defecto)
    - is_critical: BOOST de 1.5x para episodios críticos
    - is_antipattern: PENALIZACIÓN a 0.3x (siguen apareciendo pero al final)
    
    Args:
        episode: Episodio a evaluar
        semantic_score: Similitud semántica (0-1) entre query y episodio
        now: Momento actual (para cálculo de decay si está habilitado)
        decay_rate: Tasa de decaimiento temporal (DEPRECATED, ignorado por defecto)
        access_boost: Factor de boost por acceso (default: 0.1)
        use_decay: Si True, aplica decay temporal (default: False)
        
    Returns:
        Score híbrido final (puede ser > 1 debido al boost)
        
    Example:
        >>> episode = Episode(...)
        >>> semantic_score = 0.85
        >>> score = compute_hybrid_score(episode, semantic_score)
        >>> # Score basado en semántica + boost por uso, sin penalización temporal
    """
    # 1. Calcular boost por uso (refuerzo positivo)
    # Cada acceso añade access_boost al multiplicador
    # Esto simula "memoria consolidada" por uso frecuente
    boost = 1.0 + access_boost * episode.access_count
    
    # 2. Obtener importance_score (default 1.0)
    importance = episode.importance_score
    
    # 3. Calcular decay solo si está explícitamente habilitado
    decay = 1.0  # Sin decaimiento por defecto
    if use_decay:
        if now is None:
            now = datetime.now(timezone.utc)
        
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        
        episode_time = episode.timestamp
        if episode_time.tzinfo is None:
            episode_time = episode_time.replace(tzinfo=timezone.utc)
        
        age_delta = now - episode_time
        age_days = max(0, age_delta.total_seconds() / 86400)
        decay = math.exp(-decay_rate * age_days)
    
    # 4. Aplicar modificadores por is_critical e is_antipattern
    critical_modifier = CRITICAL_BOOST if getattr(episode, 'is_critical', False) else 1.0
    antipattern_modifier = ANTIPATTERN_PENALTY if getattr(episode, 'is_antipattern', False) else 1.0
    
    # 5. Calcular score final
    final_score = semantic_score * decay * boost * importance * critical_modifier * antipattern_modifier
    
    return final_score


def compute_decay(
    episode: Episode,
    now: Optional[datetime] = None,
    decay_rate: float = DEFAULT_DECAY_RATE
) -> float:
    """
    Calcular solo el factor de decaimiento temporal.
    
    DEPRECATED: Esta función se mantiene por compatibilidad pero 
    el decay ya no se usa por defecto en compute_hybrid_score.
    
    Útil para debugging o visualización de la curva de olvido
    si se decide reactivar en el futuro.
    
    Args:
        episode: Episodio a evaluar
        now: Momento actual
        decay_rate: Tasa de decaimiento
        
    Returns:
        Factor de decaimiento (0-1)
    """
    if now is None:
        now = datetime.now(timezone.utc)
    
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    
    episode_time = episode.timestamp
    if episode_time.tzinfo is None:
        episode_time = episode_time.replace(tzinfo=timezone.utc)
    
    age_delta = now - episode_time
    age_days = max(0, age_delta.total_seconds() / 86400)
    
    return math.exp(-decay_rate * age_days)


def compute_boost(
    episode: Episode,
    access_boost: float = DEFAULT_ACCESS_BOOST
) -> float:
    """
    Calcular solo el factor de boost por uso.
    
    Este es ahora el mecanismo principal de priorización:
    episodios consultados frecuentemente obtienen mayor relevancia.
    
    Args:
        episode: Episodio a evaluar
        access_boost: Factor de boost por acceso
        
    Returns:
        Factor de boost (>= 1.0)
    """
    return 1.0 + access_boost * episode.access_count


def estimate_half_life_days(decay_rate: float = DEFAULT_DECAY_RATE) -> float:
    """
    Estimar la vida media en días para la tasa de decaimiento dada.
    
    DEPRECATED: El decay ya no se usa por defecto.
    Se mantiene para análisis o si se decide reactivar.
    
    La vida media es el tiempo en que el factor de decaimiento llega a 0.5.
    
    Args:
        decay_rate: Tasa de decaimiento
        
    Returns:
        Días hasta que el decaimiento sea 0.5
        
    Example:
        >>> estimate_half_life_days(0.05)
        13.86...  # Aproximadamente 14 días
    """
    # decay = e^(-rate * days) = 0.5
    # -rate * days = ln(0.5)
    # days = -ln(0.5) / rate = ln(2) / rate
    return math.log(2) / decay_rate


# =============================================================================
# NUEVAS FUNCIONES PARA SISTEMA DE CONSOLIDACIÓN AUTOMÁTICA
# =============================================================================

# Umbrales para trigger de consolidación automática
CONSOLIDATION_ACCESS_THRESHOLD = 10  # Consolidar si un episodio supera este access_count
CONSOLIDATION_EPISODE_THRESHOLD = 20  # Consolidar si hay más de N episodios sin consolidar


def should_trigger_consolidation(
    episode_access_count: int,
    total_unconsolidated: int = 0
) -> bool:
    """
    Determinar si se debe disparar consolidación automática.
    
    Criterios:
    1. Un episodio individual ha sido consultado muchas veces (cluster "caliente")
    2. Hay muchos episodios sin consolidar
    
    Args:
        episode_access_count: Número de accesos del episodio más consultado
        total_unconsolidated: Episodios sin consolidar en meta-memorias
        
    Returns:
        True si se recomienda consolidar
    """
    # Criterio 1: Episodio muy consultado indica patrón frecuente
    if episode_access_count >= CONSOLIDATION_ACCESS_THRESHOLD:
        return True
    
    # Criterio 2: Muchos episodios sin consolidar
    if total_unconsolidated >= CONSOLIDATION_EPISODE_THRESHOLD:
        return True
    
    return False


def get_hot_episodes_for_reclustering(
    episodes: list[Episode],
    access_threshold: int = CONSOLIDATION_ACCESS_THRESHOLD
) -> list[Episode]:
    """
    Identificar episodios "calientes" que deberían priorizarse en re-clustering.
    
    Episodios con alto access_count indican patrones frecuentes que
    deberían consolidarse en meta-memorias para acceso rápido.
    
    Args:
        episodes: Lista de episodios a evaluar
        access_threshold: Umbral de accesos para considerar "caliente"
        
    Returns:
        Lista de episodios con alto uso
    """
    return [ep for ep in episodes if ep.access_count >= access_threshold]
