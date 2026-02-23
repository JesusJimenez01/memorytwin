"""
Data Models for Memory Twin
============================

Defines Pydantic schemas for memory episodes,
metadata, and system configuration.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    """Return the current datetime in UTC."""
    return datetime.now(timezone.utc)


class EpisodeType(str, Enum):
    """Types of captured memory episodes."""

    DECISION = "decision"           # Technical decision made
    BUG_FIX = "bug_fix"             # Bug fix
    REFACTOR = "refactor"           # Code refactoring
    FEATURE = "feature"             # New feature
    OPTIMIZATION = "optimization"   # Performance improvement
    LEARNING = "learning"           # Learning or discovery
    EXPERIMENT = "experiment"       # Test or experiment


class ReasoningTrace(BaseModel):
    """
    Reasoning trace captured from an AI assistant.
    Represents the model's visible "thinking" output.
    """

    raw_thinking: str = Field(
        ...,
        description="Raw reasoning text from the model"
    )
    alternatives_considered: list[str] = Field(
        default_factory=list,
        description="Alternatives considered and discarded"
    )
    decision_factors: list[str] = Field(
        default_factory=list,
        description="Factors that influenced the decision"
    )
    confidence_level: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence level in the decision (0-1)"
    )


class Episode(BaseModel):
    """
    Memory Episode - Fundamental unit of knowledge.

    Captures the full context of a technical decision:
    what was done, how it was reasoned, and why.

    Includes fields for the forgetting curve:
    - importance_score: base relevance of the episode
    - access_count: number of times it has been retrieved
    - last_accessed: last time it was queried
    """

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=_utc_now)

    # Episode context
    task: str = Field(
        ...,
        description="Description of the task or problem addressed"
    )
    context: str = Field(
        ...,
        description="Technical context: files, modules, stack involved"
    )

    # Reasoning
    reasoning_trace: ReasoningTrace = Field(
        ...,
        description="Model's thinking trace"
    )

    # Solution
    solution: str = Field(
        ...,
        description="Code or implemented solution"
    )
    solution_summary: str = Field(
        ...,
        description="Executive summary of the solution"
    )

    # Outcome
    outcome: Optional[str] = Field(
        default=None,
        description="Observed result after applying the solution"
    )
    success: bool = Field(
        default=True,
        description="Whether the solution was successful"
    )

    # Metadata
    episode_type: EpisodeType = Field(
        default=EpisodeType.DECISION,
        description="Episode type"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization"
    )
    files_affected: list[str] = Field(
        default_factory=list,
        description="Modified files"
    )

    # Lessons
    lessons_learned: list[str] = Field(
        default_factory=list,
        description="Lessons extracted from this episode"
    )

    # Source
    source_assistant: str = Field(
        default="unknown",
        description="Source code assistant (copilot, claude, cursor)"
    )
    project_name: str = Field(
        default="default",
        description="Associated project name"
    )

    # Forgetting Curve fields
    importance_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Base relevance of the episode (0-1). May be adjusted automatically."
    )
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this episode has been retrieved/queried"
    )
    last_accessed: Optional[datetime] = Field(
        default=None,
        description="Last time this episode was queried"
    )

    # Active and useful memory fields
    is_antipattern: bool = Field(
        default=False,
        description="If True, this episode represents something NOT to do. Shown as a WARNING."
    )
    is_critical: bool = Field(
        default=False,
        description="If True, this episode is critical and should be prioritized in searches."
    )
    superseded_by: Optional[UUID] = Field(
        default=None,
        description="If this episode was replaced by a newer one, reference to the new episode."
    )
    deprecation_reason: Optional[str] = Field(
        default=None,
        description="Reason why this episode no longer applies or was marked as an antipattern."
    )


class MemoryQuery(BaseModel):
    """Query to the Oráculo memory system."""

    query: str = Field(
        ...,
        description="User's question or search text"
    )
    project_filter: Optional[str] = Field(
        default=None,
        description="Filter by specific project"
    )
    type_filter: Optional[EpisodeType] = Field(
        default=None,
        description="Filter by episode type"
    )
    date_from: Optional[datetime] = Field(
        default=None,
        description="Start date for search range"
    )
    date_to: Optional[datetime] = Field(
        default=None,
        description="End date for search range"
    )
    tags_filter: list[str] = Field(
        default_factory=list,
        description="Filter by tags"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return"
    )


class MemorySearchResult(BaseModel):
    """Memory search result."""

    episode: Episode
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Semantic relevance score"
    )
    match_reason: str = Field(
        default="",
        description="Explanation of why this result is relevant"
    )


class ProcessedInput(BaseModel):
    """
    Escriba's processed input.
    Represents captured text before being converted into an Episode.
    """

    raw_text: str = Field(
        ...,
        description="Raw captured text (model's thinking)"
    )
    user_prompt: Optional[str] = Field(
        default=None,
        description="Original user prompt"
    )
    code_changes: Optional[str] = Field(
        default=None,
        description="Associated code changes (diff or new code)"
    )
    source: str = Field(
        default="manual",
        description="Capture source: manual, clipboard, mcp"
    )
    captured_at: datetime = Field(default_factory=_utc_now)


class MetaMemory(BaseModel):
    """
    Meta-Memory - Consolidated knowledge from multiple episodes.

    Represents patterns, lessons, and emergent knowledge that
    arises from analyzing related episodes. Generated through
    clustering and LLM synthesis.

    Example: If there are 5 episodes about "API error handling",
    they are consolidated into a MetaMemory with the common pattern,
    aggregated lessons, and important exceptions.
    """

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    # Identified pattern
    pattern: str = Field(
        ...,
        description="Common pattern or theme identified across source episodes"
    )
    pattern_summary: str = Field(
        ...,
        description="Executive summary of the pattern (1-2 sentences)"
    )

    # Consolidated knowledge
    lessons: list[str] = Field(
        default_factory=list,
        description="Consolidated lessons learned from all episodes"
    )
    best_practices: list[str] = Field(
        default_factory=list,
        description="Best practices derived from the pattern"
    )
    antipatterns: list[str] = Field(
        default_factory=list,
        description="Anti-patterns or common mistakes to avoid"
    )

    # Exceptions and nuances
    exceptions: list[str] = Field(
        default_factory=list,
        description="Special cases where the pattern does not apply"
    )
    edge_cases: list[str] = Field(
        default_factory=list,
        description="Discovered edge cases"
    )

    # Applicable contexts
    contexts: list[str] = Field(
        default_factory=list,
        description="Contexts where this knowledge is applicable"
    )
    technologies: list[str] = Field(
        default_factory=list,
        description="Related technologies (languages, frameworks, libs)"
    )

    # Traceability
    source_episode_ids: list[UUID] = Field(
        default_factory=list,
        description="IDs of source episodes that originated this meta-memory"
    )
    episode_count: int = Field(
        default=0,
        ge=0,
        description="Number of consolidated episodes"
    )

    # Quality and confidence
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Consolidation confidence (0-1). Higher with more episodes."
    )
    coherence_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How coherent the source episodes are with each other"
    )

    # Metadata
    project_name: str = Field(
        default="default",
        description="Associated project"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization"
    )

    # Usage and relevance
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this meta-memory has been queried"
    )
    last_accessed: Optional[datetime] = Field(
        default=None,
        description="Last query time"
    )


class MetaMemorySearchResult(BaseModel):
    """Meta-memory search result."""

    meta_memory: MetaMemory
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Semantic relevance score"
    )
    match_reason: str = Field(
        default="",
        description="Explanation of why this result is relevant"
    )



