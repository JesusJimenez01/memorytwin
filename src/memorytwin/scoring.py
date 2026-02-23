"""
Hybrid Scoring System for Memory Twin
======================================

Implements a "reinforcement without forgetting" system inspired by neuroscience:
- Semantic similarity (embeddings) as the base
- Boost from frequent usage (positive reinforcement)
- Base importance of the episode

The formula is:
    final_score = semantic_score * boost * importance_score

Where:
- semantic_score: cosine similarity between query and episode
- boost: 1 + access_boost * access_count, rewards frequently accessed episodes
- importance_score: base relevance assigned to the episode
"""

from memorytwin.models import Episode

# Configurable constants
DEFAULT_ACCESS_BOOST = 0.1  # Additional boost per access

# Modifiers for is_critical and is_antipattern
CRITICAL_BOOST = 1.5  # Critical episodes receive 50% more relevance
ANTIPATTERN_PENALTY = 0.3  # Antipatterns reduced to 30% relevance (not excluded)


def compute_hybrid_score(
    episode: Episode,
    semantic_score: float,
    access_boost: float = DEFAULT_ACCESS_BOOST,
) -> float:
    """
    Calculate hybrid score combining semantics, usage, and importance.

    Implements a "reinforcement without forgetting" system where:
    - Semantic similarity is the base
    - Frequent usage (access_count) reinforces relevance
    - is_critical: 1.5x BOOST for critical episodes
    - is_antipattern: 0.3x PENALTY (still appears but ranked lower)

    Args:
        episode: Episode to evaluate
        semantic_score: Semantic similarity (0-1) between query and episode
        access_boost: Boost factor per access (default: 0.1)

    Returns:
        Final hybrid score (can be > 1 due to boost)
    """
    # 1. Boost from usage (positive reinforcement)
    # Each access adds access_boost to the multiplier
    # Simulates "consolidated memory" through frequent use
    boost = 1.0 + access_boost * episode.access_count

    # 2. Get importance_score (default 1.0)
    importance = episode.importance_score

    # 3. Apply modifiers for is_critical and is_antipattern
    critical_modifier = CRITICAL_BOOST if getattr(episode, 'is_critical', False) else 1.0
    antipattern_modifier = ANTIPATTERN_PENALTY if getattr(episode, 'is_antipattern', False) else 1.0

    # 4. Calculate final score
    final_score = semantic_score * boost * importance * critical_modifier * antipattern_modifier

    return final_score


def compute_boost(
    episode: Episode,
    access_boost: float = DEFAULT_ACCESS_BOOST
) -> float:
    """
    Calculate the usage boost factor.

    This is the main prioritization mechanism:
    frequently accessed episodes get higher relevance.

    Args:
        episode: Episode to evaluate
        access_boost: Boost factor per access

    Returns:
        Boost factor (>= 1.0)
    """
    return 1.0 + access_boost * episode.access_count


# =============================================================================
# AUTOMATIC CONSOLIDATION SYSTEM
# =============================================================================

# Thresholds for automatic consolidation trigger
CONSOLIDATION_ACCESS_THRESHOLD = 10  # Consolidate if an episode exceeds this access_count
CONSOLIDATION_EPISODE_THRESHOLD = 20  # Consolidate if there are more than N unconsolidated episodes


def should_trigger_consolidation(
    episode_access_count: int,
    total_unconsolidated: int = 0
) -> bool:
    """
    Determine if automatic consolidation should be triggered.

    Criteria:
    1. An individual episode has been accessed many times ("hot" cluster)
    2. There are many unconsolidated episodes

    Args:
        episode_access_count: Access count of the most accessed episode
        total_unconsolidated: Episodes not yet consolidated into meta-memories

    Returns:
        True if consolidation is recommended
    """
    # Criterion 1: Highly accessed episode indicates a frequent pattern
    if episode_access_count >= CONSOLIDATION_ACCESS_THRESHOLD:
        return True

    # Criterion 2: Many unconsolidated episodes
    if total_unconsolidated >= CONSOLIDATION_EPISODE_THRESHOLD:
        return True

    return False


def get_hot_episodes_for_reclustering(
    episodes: list[Episode],
    access_threshold: int = CONSOLIDATION_ACCESS_THRESHOLD
) -> list[Episode]:
    """
    Identify "hot" episodes that should be prioritized for re-clustering.

    Episodes with high access_count indicate frequent patterns that
    should be consolidated into meta-memories for quick access.

    Args:
        episodes: List of episodes to evaluate
        access_threshold: Access threshold to consider "hot"

    Returns:
        List of high-usage episodes
    """
    return [ep for ep in episodes if ep.access_count >= access_threshold]
