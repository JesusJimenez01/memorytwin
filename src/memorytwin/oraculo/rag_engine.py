"""
RAG Engine for the Oráculo
===========================

Implements Retrieval-Augmented Generation over the episodic
memory database to answer contextual questions about
technical decisions.

Includes support for MetaMemories (consolidated knowledge).
"""

import logging
from typing import Optional

from memorytwin.config import BaseLLMClient, get_llm_model, get_settings
from memorytwin.escriba.storage import MemoryStorage
from memorytwin.models import MemoryQuery, MemorySearchResult, MetaMemorySearchResult
from memorytwin.observability import trace_access_memory, trace_observation

logger = logging.getLogger(__name__)

# Generation parameters for Oracle answers (slightly creative, bounded length)
ORACLE_TEMPERATURE = 0.4
ORACLE_MAX_OUTPUT_TOKENS = 2048

# System prompt for the Oráculo
ORACLE_SYSTEM_PROMPT = (
    "You are the Oracle of Memory Twin, a specialized assistant that answers questions "
    "about the technical history and development decisions of a software project.\n\n"
    "Your knowledge comes from two sources:\n\n"
    "1. **META-MEMORIES**: Consolidated knowledge from multiple related episodes. "
    "They represent automatically identified patterns, lessons, and best practices. "
    "They are more reliable and general.\n\n"
    "2. **EPISODES**: Individual memories documenting the reasoning ('thinking') of AI "
    "assistants during development. They contain:\n"
    "   - The task or problem addressed\n"
    "   - The technical context\n"
    "   - The reasoning trace (alternatives considered, decision factors)\n"
    "   - The implemented solution\n"
    "   - Lessons learned\n\n"
    "INSTRUCTIONS:\n"
    "1. Prioritize META-MEMORIES if available (they are consolidated knowledge)\n"
    "2. Complement with EPISODES for specific details\n"
    "3. If the information is not in the memory, indicate there are no records on that topic\n"
    "4. Cite sources (meta-memories or episodes) when useful\n"
    "5. Explain the 'why' behind decisions, not just the 'what'\n"
    "6. If there are relevant lessons learned or best practices, include them\n"
    "7. Be concise but complete\n"
    "8. Use Markdown format for better readability\n\n"
    "RESPONSE FORMAT:\n"
    "- Direct answer to the question\n"
    "- Relevant context from the sources\n"
    "- Applicable lessons learned / best practices\n"
    "- References to specific meta-memories or episodes"
)


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for querying
    episodic memories.
    """

    def __init__(
        self,
        storage: Optional[MemoryStorage] = None,
    ):
        """
        Initialize the RAG engine.

        Args:
            storage: Memory storage instance (creates one if not provided)
        """
        self.storage = storage or MemoryStorage()
        self._model: Optional[BaseLLMClient] = None

    @property
    def model(self) -> BaseLLMClient:
        """
        LLM client, created on first use.

        Retrieval-only operations (timeline, lessons, statistics) never touch
        the LLM, so they keep working without an API key configured.
        """
        if self._model is None:
            self._model = get_llm_model(
                temperature=ORACLE_TEMPERATURE,
                max_output_tokens=ORACLE_MAX_OUTPUT_TOKENS,
            )
        return self._model

    @trace_access_memory
    async def query(
        self,
        question: str,
        project_name: Optional[str] = None,
        top_k: int = 5,
        include_meta_memories: bool = True
    ) -> dict:
        """
        Perform a RAG query over the memories.

        Searches MetaMemories (consolidated knowledge) first,
        then complements with individual episodes.

        Args:
            question: User's question
            project_name: Filter by specific project
            top_k: Number of results to retrieve
            include_meta_memories: Whether to include meta-memories in the search

        Returns:
            Dict with answer, episodes used, meta-memories, and metadata
        """
        meta_results = []

        # Search meta-memories first (consolidated knowledge)
        if include_meta_memories:
            meta_results = self.storage.search_meta_memories(
                query=question,
                project_name=project_name,
                top_k=min(3, top_k)  # Max 3 meta-memories
            )

        # Search relevant episodes
        memory_query = MemoryQuery(
            query=question,
            project_filter=project_name,
            top_k=top_k
        )

        search_results = self.storage.search_episodes(memory_query)

        if not search_results and not meta_results:
            return {
                "answer": (
                    "I found no memory episodes or consolidated knowledge "
                    "related to your question. "
                    "This topic may not have been documented yet."
                ),
                "episodes_used": [],
                "meta_memories_used": [],
                "relevance_scores": [],
                "meta_relevance_scores": [],
                "context_provided": False,
                "llm_generated": False,
            }

        # Build context combining meta-memories and episodes
        context = self._build_combined_context(meta_results, search_results)

        # Generate answer. Retrieval already succeeded, so if the LLM is down
        # (no API key, rate limit, outage) we still return what was found.
        try:
            answer = await self._generate_answer(question, context)
            llm_generated = True
        except Exception as e:
            logger.warning("LLM unavailable, answering with retrieved memories only: %s", e)
            answer = self._build_fallback_answer(meta_results, search_results)
            llm_generated = False

        return {
            "answer": answer,
            "episodes_used": [r.episode for r in search_results],
            "meta_memories_used": [r.meta_memory for r in meta_results],
            "relevance_scores": [r.relevance_score for r in search_results],
            "meta_relevance_scores": [r.relevance_score for r in meta_results],
            "context_provided": True,
            "llm_generated": llm_generated,
        }

    def query_sync(
        self,
        question: str,
        project_name: Optional[str] = None,
        top_k: int = 5
    ) -> dict:
        """Synchronous version of query."""
        import asyncio
        return asyncio.run(self.query(question, project_name, top_k))

    def _build_context(self, results: list[MemorySearchResult]) -> str:
        """Build episode context for the LLM."""
        context_parts = ["## RELEVANT MEMORY EPISODES\n"]

        for i, result in enumerate(results, 1):
            ep = result.episode

            context_parts.append(f"""
### Episode {i} (Relevance: {result.relevance_score:.0%})
- **ID**: {ep.id}
- **Date**: {ep.timestamp.strftime('%Y-%m-%d %H:%M')}
- **Type**: {ep.episode_type.value}
- **Project**: {ep.project_name}
- **Assistant**: {ep.source_assistant}

**Task**: {ep.task}

**Context**: {ep.context}

**Reasoning**:
{ep.reasoning_trace.raw_thinking}

**Alternatives considered**: {', '.join(ep.reasoning_trace.alternatives_considered) or 'Not documented'}

**Decision factors**: {', '.join(ep.reasoning_trace.decision_factors) or 'Not documented'}

**Solution**: {ep.solution_summary}

**Lessons learned**: {', '.join(ep.lessons_learned) or 'None documented'}

**Tags**: {', '.join(ep.tags)}
---
""")

        return "\n".join(context_parts)

    def _build_combined_context(
        self,
        meta_results: list[MetaMemorySearchResult],
        episode_results: list[MemorySearchResult]
    ) -> str:
        """
        Build context combining meta-memories and episodes.

        Meta-memories go first (higher priority).
        """
        context_parts = []

        # Meta-memories first (consolidated knowledge)
        if meta_results:
            context_parts.append("## META-MEMORIES (Consolidated Knowledge)\n")
            context_parts.append("*These are consolidated lessons from multiple related episodes.*\n")

            for i, result in enumerate(meta_results, 1):
                mm = result.meta_memory

                context_parts.append(f"""
### Meta-Memory {i} (Relevance: {result.relevance_score:.0%} | Confidence: {mm.confidence:.0%})
- **Pattern**: {mm.pattern_summary}
- **Based on**: {mm.episode_count} episodes
- **Technologies**: {', '.join(mm.technologies) or 'Not specified'}

**Pattern Description**:
{mm.pattern}

**Lessons Learned**:
{chr(10).join(f'• {lesson}' for lesson in mm.lessons) or '• None documented'}

**Best Practices**:
{chr(10).join(f'• {p}' for p in mm.best_practices) or '• None documented'}

**Anti-patterns to Avoid**:
{chr(10).join(f'• {a}' for a in mm.antipatterns) or '• None documented'}

**Exceptions/Special Cases**:
{chr(10).join(f'• {e}' for e in mm.exceptions) or '• None documented'}

**Applicable Contexts**: {', '.join(mm.contexts) or 'General'}
---
""")

        # Then individual episodes
        if episode_results:
            context_parts.append("\n## INDIVIDUAL EPISODES\n")
            context_parts.append("*Specific details of particular decisions.*\n")
            context_parts.append(self._build_context(episode_results))

        return "\n".join(context_parts)

    @staticmethod
    def _build_fallback_answer(
        meta_results: list[MetaMemorySearchResult],
        episode_results: list[MemorySearchResult]
    ) -> str:
        """Markdown digest of the retrieved memories, used when no LLM answer is available."""
        lines = [
            "_The language model is unavailable, so here are the most relevant memories "
            "found for your question:_",
            "",
        ]

        for result in meta_results:
            mm = result.meta_memory
            lines.append(f"- **Pattern** ({result.relevance_score:.0%}): {mm.pattern_summary}")
            lines.extend(f"  - Lesson: {lesson}" for lesson in mm.lessons[:3])

        for result in episode_results:
            ep = result.episode
            prefix = "⚠️ **Antipattern** " if ep.is_antipattern else ""
            lines.append(
                f"- {prefix}**{ep.task}** ({result.relevance_score:.0%}, "
                f"{ep.timestamp.strftime('%Y-%m-%d')}): {ep.solution_summary}"
            )
            lines.extend(f"  - Lesson: {lesson}" for lesson in ep.lessons_learned[:2])

        return "\n".join(lines)

    async def _generate_answer(self, question: str, context: str) -> str:
        """Generate an answer using the LLM with observability."""
        prompt = f"""## MEMORY CONTEXT
{context}

## USER QUESTION
{question}

## YOUR ANSWER (using Markdown)
"""

        with trace_observation(
            "Oracle LLM Response",
            as_type="generation",
            model=get_settings().llm_model,
            model_parameters={
                "temperature": ORACLE_TEMPERATURE,
                "max_output_tokens": ORACLE_MAX_OUTPUT_TOKENS,
            },
            input={"question": question, "context_length": len(context)},
        ) as generation:
            response = await self.model.generate_async(
                [
                    {"role": "user", "parts": [ORACLE_SYSTEM_PROMPT]},
                    {"role": "model", "parts": [
                        "Understood. I'm ready to answer questions about the project's "
                        "technical memory based exclusively on the provided episodes."
                    ]},
                    {"role": "user", "parts": [prompt]}
                ]
            )
            if generation is not None:
                generation.update(output=response.text[:1000])  # Limit output to avoid saturation

        return response.text

    def get_timeline(
        self,
        project_name: Optional[str] = None,
        limit: int = 50
    ) -> list:
        """
        Get decision timeline for visualization.

        Args:
            project_name: Filter by project
            limit: Maximum number of episodes

        Returns:
            List of episodes ordered chronologically
        """
        episodes = self.storage.get_timeline(
            project_name=project_name,
            limit=limit
        )

        # Format for visualization
        timeline = []
        for ep in episodes:
            timeline.append({
                "id": str(ep.id),
                "timestamp": ep.timestamp.isoformat(),
                "date": ep.timestamp.strftime("%Y-%m-%d"),
                "time": ep.timestamp.strftime("%H:%M"),
                "task": ep.task,
                "type": ep.episode_type.value,
                "summary": ep.solution_summary,
                "tags": ep.tags,
                "assistant": ep.source_assistant,
                "success": ep.success
            })

        return timeline

    def get_lessons(
        self,
        project_name: Optional[str] = None,
        tags: Optional[list[str]] = None
    ) -> list:
        """
        Get aggregated lessons learned.

        Args:
            project_name: Filter by project
            tags: Filter by tags

        Returns:
            List of lessons with context
        """
        return self.storage.get_lessons_learned(
            project_name=project_name,
            tags=tags
        )

    def get_statistics(self, project_name: Optional[str] = None) -> dict:
        """Get memory statistics."""
        return self.storage.get_statistics(project_name)
