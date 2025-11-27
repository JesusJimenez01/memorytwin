"""
Sistema de Scoring Híbrido para Memory Twin
============================================

Implementa la curva de olvido (forgetting curve) combinando:
- Similitud semántica (embeddings)
- Decaimiento temporal (exponencial)
- Boost por uso frecuente
- Importancia base del episodio

La fórmula es:
    final_score = semantic_score * decay * boost * importance_score

Donde:
- semantic_score: similitud coseno entre query y episodio
- decay: exp(-decay_rate * age_days), controla el olvido temporal
- boost: 1 + access_boost * access_count, premia episodios consultados frecuentemente
- importance_score: relevancia base asignada al episodio
"""

import math
from datetime import datetime, timezone
from typing import Optional

from memorytwin.models import Episode


# Constantes configurables
DEFAULT_DECAY_RATE = 0.05  # Tasa de decaimiento por día (ajustable)
DEFAULT_ACCESS_BOOST = 0.1  # Boost adicional por cada acceso


def compute_hybrid_score(
    episode: Episode,
    semantic_score: float,
    now: Optional[datetime] = None,
    decay_rate: float = DEFAULT_DECAY_RATE,
    access_boost: float = DEFAULT_ACCESS_BOOST
) -> float:
    """
    Calcular score híbrido combinando semántica, tiempo y uso.
    
    Implementa la curva de olvido (forgetting curve) de Ebbinghaus
    adaptada para memoria episódica de proyectos de software.
    
    Args:
        episode: Episodio a evaluar
        semantic_score: Similitud semántica (0-1) entre query y episodio
        now: Momento actual (default: ahora)
        decay_rate: Tasa de decaimiento temporal (default: 0.05)
        access_boost: Factor de boost por acceso (default: 0.1)
        
    Returns:
        Score híbrido final (puede ser > 1 debido al boost)
        
    Example:
        >>> episode = Episode(...)
        >>> semantic_score = 0.85
        >>> score = compute_hybrid_score(episode, semantic_score)
        >>> # Para un episodio reciente y frecuentemente consultado,
        >>> # el score será cercano o superior a semantic_score
    """
    if now is None:
        now = datetime.now(timezone.utc)
    
    # Asegurar que now tiene timezone
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    
    # 1. Calcular edad en días
    episode_time = episode.timestamp
    if episode_time.tzinfo is None:
        episode_time = episode_time.replace(tzinfo=timezone.utc)
    
    age_delta = now - episode_time
    age_days = max(0, age_delta.total_seconds() / 86400)  # Convertir a días
    
    # 2. Calcular decaimiento temporal (forgetting curve)
    # decay = e^(-rate * days)
    # Con rate=0.05, después de 14 días el decay es ~0.5
    decay = math.exp(-decay_rate * age_days)
    
    # 3. Calcular boost por uso
    # Cada acceso añade access_boost al multiplicador
    boost = 1.0 + access_boost * episode.access_count
    
    # 4. Obtener importance_score (default 1.0)
    importance = episode.importance_score
    
    # 5. Calcular score final
    final_score = semantic_score * decay * boost * importance
    
    return final_score


def compute_decay(
    episode: Episode,
    now: Optional[datetime] = None,
    decay_rate: float = DEFAULT_DECAY_RATE
) -> float:
    """
    Calcular solo el factor de decaimiento temporal.
    
    Útil para debugging o visualización de la curva de olvido.
    
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
