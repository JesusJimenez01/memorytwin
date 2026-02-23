"""
Tests for the scoring module (Reinforcement Without Forgetting)
===============================================================

The system uses "reinforcement without forgetting" where:
- There is no temporal decay
- Episodes maintain full relevance
- The access_count acts as the prioritization mechanism
"""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from memorytwin.models import Episode, ReasoningTrace
from memorytwin.scoring import (
    CONSOLIDATION_ACCESS_THRESHOLD,
    CONSOLIDATION_EPISODE_THRESHOLD,
    compute_boost,
    compute_hybrid_score,
    get_hot_episodes_for_reclustering,
    should_trigger_consolidation,
)
from memorytwin.scoring import DEFAULT_ACCESS_BOOST as ACCESS_BOOST


def create_test_episode(
    days_ago: int = 0,
    access_count: int = 0,
    importance_score: float = 1.0
) -> Episode:
    """Create test episode with configurable parameters."""
    timestamp = datetime.now(timezone.utc) - timedelta(days=days_ago)

    return Episode(
        id=uuid4(),
        timestamp=timestamp,
        task="Test task",
        context="Test context",
        reasoning_trace=ReasoningTrace(raw_thinking="Test thinking"),
        solution="Test solution",
        solution_summary="Test summary",
        importance_score=importance_score,
        access_count=access_count,
        last_accessed=datetime.now(timezone.utc) if access_count > 0 else None
    )


class TestComputeHybridScore:
    """Tests for compute_hybrid_score with reinforcement without forgetting system."""

    def test_same_day_episode_no_decay(self):
        """A same-day episode maintains full score."""
        episode = create_test_episode(days_ago=0)

        score = compute_hybrid_score(episode, semantic_score=1.0)

        assert score >= 0.99

    def test_old_episode_no_decay_by_default(self):
        """An old episode has NO decay (reinforcement without forgetting)."""
        episode = create_test_episode(days_ago=30)

        score = compute_hybrid_score(episode, semantic_score=1.0)

        # Old episode maintains full relevance
        assert score >= 0.99

    def test_access_boost_increases_score(self):
        """Episodes accessed frequently have a higher score."""
        episode_no_access = create_test_episode(days_ago=10, access_count=0)
        episode_many_access = create_test_episode(days_ago=10, access_count=10)

        score_no = compute_hybrid_score(episode_no_access, 1.0)
        score_many = compute_hybrid_score(episode_many_access, 1.0)

        # Episode with more accesses should have higher score
        assert score_many > score_no

        # Boost should be proportional: 1 + 0.1 * 10 = 2.0
        expected_ratio = 1 + ACCESS_BOOST * 10
        actual_ratio = score_many / score_no
        assert abs(actual_ratio - expected_ratio) < 0.1

    def test_importance_affects_score(self):
        """Episodes with higher importance_score have higher score."""
        episode_low = create_test_episode(days_ago=5, importance_score=0.5)
        episode_high = create_test_episode(days_ago=5, importance_score=1.0)

        score_low = compute_hybrid_score(episode_low, 1.0)
        score_high = compute_hybrid_score(episode_high, 1.0)

        assert score_high > score_low
        # Ratio should be 2:1
        assert abs(score_high / score_low - 2.0) < 0.1

    def test_semantic_score_multiplied(self):
        """Semantic score is multiplied correctly."""
        episode = create_test_episode(days_ago=0)

        score_full = compute_hybrid_score(episode, semantic_score=1.0)
        score_half = compute_hybrid_score(episode, semantic_score=0.5)

        # Without decay, ratio should be 2:1
        assert abs(score_full / score_half - 2.0) < 0.1

    def test_combined_factors_no_decay(self):
        """Verify formula: semantic * boost * importance."""
        episode = create_test_episode(
            days_ago=20,
            access_count=5,
            importance_score=0.8
        )
        semantic_score = 0.9

        score = compute_hybrid_score(episode, semantic_score)

        # score = semantic * boost * importance
        boost = 1 + ACCESS_BOOST * 5  # = 1.5
        expected = semantic_score * boost * 0.8

        assert abs(score - expected) < 0.01

    def test_zero_semantic_gives_zero(self):
        """Zero semantic score gives zero final score."""
        episode = create_test_episode(days_ago=0, access_count=100)

        score = compute_hybrid_score(episode, semantic_score=0.0)

        assert score == 0.0

    def test_very_old_episode_keeps_relevance(self):
        """A very old episode maintains full relevance (no decay)."""
        episode = create_test_episode(days_ago=365)  # 1 year

        score = compute_hybrid_score(episode, semantic_score=1.0)

        # Maintains full relevance
        assert score >= 0.99


class TestConsolidationTriggers:
    """Tests for automatic consolidation triggers."""

    def test_should_trigger_on_high_access(self):
        """Should trigger consolidation with high access_count."""
        result = should_trigger_consolidation(
            episode_access_count=CONSOLIDATION_ACCESS_THRESHOLD,
            total_unconsolidated=0
        )
        assert result is True

    def test_should_trigger_on_many_episodes(self):
        """Should trigger consolidation with many unconsolidated episodes."""
        result = should_trigger_consolidation(
            episode_access_count=0,
            total_unconsolidated=CONSOLIDATION_EPISODE_THRESHOLD
        )
        assert result is True

    def test_should_not_trigger_low_activity(self):
        """Should not trigger with low activity."""
        result = should_trigger_consolidation(
            episode_access_count=1,
            total_unconsolidated=5
        )
        assert result is False

    def test_get_hot_episodes(self):
        """Should correctly identify hot episodes."""
        episodes = [
            create_test_episode(access_count=1),
            create_test_episode(access_count=CONSOLIDATION_ACCESS_THRESHOLD),
            create_test_episode(access_count=CONSOLIDATION_ACCESS_THRESHOLD + 5),
            create_test_episode(access_count=3),
        ]

        hot = get_hot_episodes_for_reclustering(episodes)

        assert len(hot) == 2  # Only those exceeding the threshold


class TestScoringConstants:
    """Tests for scoring constants."""

    def test_access_boost_reasonable(self):
        """ACCESS_BOOST is in a reasonable range."""
        assert 0.01 < ACCESS_BOOST < 0.5

    def test_consolidation_thresholds_reasonable(self):
        """Consolidation thresholds are reasonable."""
        assert 5 <= CONSOLIDATION_ACCESS_THRESHOLD <= 20
        assert 10 <= CONSOLIDATION_EPISODE_THRESHOLD <= 50


class TestCriticalAndAntipatternModifiers:
    """Tests for is_critical and is_antipattern modifiers."""

    def test_critical_episode_gets_boost(self):
        """Critical episodes receive a 1.5x boost."""
        episode_normal = create_test_episode(days_ago=0)
        episode_critical = create_test_episode(days_ago=0)
        episode_critical.is_critical = True

        score_normal = compute_hybrid_score(episode_normal, 1.0)
        score_critical = compute_hybrid_score(episode_critical, 1.0)

        assert score_critical == score_normal * 1.5

    def test_antipattern_episode_gets_penalty(self):
        """Antipattern episodes receive a 0.3x penalty."""
        episode_normal = create_test_episode(days_ago=0)
        episode_antipattern = create_test_episode(days_ago=0)
        episode_antipattern.is_antipattern = True

        score_normal = compute_hybrid_score(episode_normal, 1.0)
        score_antipattern = compute_hybrid_score(episode_antipattern, 1.0)

        assert score_antipattern == score_normal * 0.3

    def test_critical_antipattern_combination(self):
        """An episode marked as both critical AND antipattern applies both modifiers."""
        episode_both = create_test_episode(days_ago=0)
        episode_both.is_critical = True
        episode_both.is_antipattern = True

        score = compute_hybrid_score(episode_both, 1.0)

        # 1.0 * 1.5 (critical) * 0.3 (antipattern) = 0.45
        assert abs(score - 0.45) < 0.01

    def test_antipattern_still_appears_in_results(self):
        """Antipatterns are not fully excluded, just ranked lower."""
        episode_antipattern = create_test_episode(days_ago=0)
        episode_antipattern.is_antipattern = True

        score = compute_hybrid_score(episode_antipattern, 1.0)

        # Score > 0 means it still appears
        assert score > 0
        assert score == 0.3  # 1.0 * 0.3


class TestComputeBoost:
    """Tests for compute_boost."""

    def test_zero_access_returns_one(self):
        """With no accesses, the boost is 1.0."""
        episode = create_test_episode(access_count=0)
        assert compute_boost(episode) == 1.0

    def test_boost_increases_with_access(self):
        """The boost increases with each access."""
        ep1 = create_test_episode(access_count=5)
        ep2 = create_test_episode(access_count=10)

        assert compute_boost(ep1) < compute_boost(ep2)
        assert compute_boost(ep1) == 1 + ACCESS_BOOST * 5
        assert compute_boost(ep2) == 1 + ACCESS_BOOST * 10
