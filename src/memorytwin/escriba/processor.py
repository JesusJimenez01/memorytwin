"""
Thought Processor - LLM-based reasoning structuring
====================================================

Uses an LLM (e.g., Gemini Flash) to convert raw
"thinking" text into structured memory episodes.
"""

import json
import logging
import re
from typing import Optional

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from memorytwin.config import get_llm_model, get_settings
from memorytwin.models import Episode, EpisodeType, ProcessedInput, ReasoningTrace
from memorytwin.observability import _get_langfuse, _is_disabled, flush_traces, trace_store_memory

logger = logging.getLogger("memorytwin.processor")


# Exceptions that merit retry (broad catch - LLM APIs can raise various errors)
RETRYABLE_EXCEPTIONS = (Exception,)


# System prompt for structuring thoughts
STRUCTURING_PROMPT = """You are an assistant specialized in analyzing and structuring the technical reasoning \
of AI assistants during software development.

Your task is to convert raw "thinking" text (visible reasoning) from a code assistant into a structured memory episode.

INPUT:
- Model's reasoning text (visible thinking)
- Optionally: original user prompt and code changes

OUTPUT (strict JSON):
{
    "task": "Concise description of the task or problem addressed",
    "context": "Technical context: files, modules, technologies involved",
    "reasoning_trace": {
        "raw_thinking": "Summary of the main reasoning (max 500 words)",
        "alternatives_considered": ["discarded alternative 1", "discarded alternative 2"],
        "decision_factors": ["influencing factor 1", "influencing factor 2"],
        "confidence_level": 0.85
    },
    "solution": "Code or implemented solution (relevant excerpt)",
    "solution_summary": "Executive summary of the solution in 1-2 sentences",
    "episode_type": "decision|bug_fix|refactor|feature|optimization|learning|experiment",
    "tags": ["tag1", "tag2", "tag3"],
    "files_affected": ["file1.py", "file2.ts"],
    "lessons_learned": ["lesson 1", "lesson 2"]
}

RULES:
1. Be concise but complete
2. Extract ALL considered and discarded alternatives
3. Identify key decision factors
4. Assign a confidence level based on the reasoning tone
5. Generate relevant tags for future searching
6. Extract lessons learned if any (avoided errors, discovered patterns)
7. ALWAYS respond with valid JSON, no additional text
"""


class ThoughtProcessor:
    """
    Thought processor using LLM.
    Converts raw thinking text into structured episodes.
    """

    def __init__(self):
        """Initialize processor with LLM model."""
        # Use centralized factory (JSON response)
        self.model = get_llm_model(response_mime_type="application/json")

    @trace_store_memory
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def process_thought(
        self,
        raw_input: ProcessedInput,
        project_name: str = "default",
        source_assistant: str = "unknown"
    ) -> Episode:
        """
        Process raw thinking and convert it into a structured episode.

        Args:
            raw_input: Captured input (thinking + optional context)
            project_name: Project name
            source_assistant: Source assistant (copilot, claude, etc.)

        Returns:
            Structured Episode ready for storage
        """
        settings = get_settings()

        # Build prompt with input
        user_content = self._build_user_prompt(raw_input)

        # Trace LLM generation
        langfuse = _get_langfuse() if not _is_disabled() else None
        generation = None

        try:
            if langfuse:
                generation = langfuse.start_as_current_generation(
                    name="Escriba - Structure Thought",
                    model=settings.llm_model,
                    model_parameters={"temperature": settings.llm_temperature},
                    input={"thinking_text": raw_input.raw_text[:500], "project": project_name}
                ).__enter__()

            # Call the LLM (unified interface)
            response = await self.model.generate_async(
                [
                    {"role": "user", "parts": [STRUCTURING_PROMPT]},
                    {
                        "role": "model",
                        "parts": [
                            "Understood. I'm ready to structure"
                            " the technical reasoning in JSON format."
                        ],
                    },
                    {"role": "user", "parts": [user_content]}
                ]
            )

            if generation:
                generation.update(output=response.text[:1000])

        finally:
            if generation:
                try:
                    generation.end()
                except Exception:
                    pass
            if langfuse:
                flush_traces()

        # Parse JSON response
        try:
            structured_data = json.loads(response.text)
        except json.JSONDecodeError as e:
            # Try to extract JSON if there's extra text
            json_match = re.search(r'\{[\s\S]*\}', response.text)
            if json_match:
                structured_data = json.loads(json_match.group())
            else:
                raise ValueError(f"LLM did not return valid JSON: {e}")

        # Build Episode
        episode = self._build_episode(
            structured_data,
            project_name=project_name,
            source_assistant=source_assistant
        )

        return episode

    def process_thought_sync(
        self,
        raw_input: ProcessedInput,
        project_name: str = "default",
        source_assistant: str = "unknown"
    ) -> Episode:
        """Synchronous version of process_thought."""
        import asyncio
        return asyncio.run(
            self.process_thought(raw_input, project_name, source_assistant)
        )

    def _build_user_prompt(self, raw_input: ProcessedInput) -> str:
        """Build user prompt with captured input."""
        parts = ["## REASONING TEXT (THINKING):\n"]
        parts.append(raw_input.raw_text)

        if raw_input.user_prompt:
            parts.append("\n\n## ORIGINAL USER PROMPT:\n")
            parts.append(raw_input.user_prompt)

        if raw_input.code_changes:
            parts.append("\n\n## CODE CHANGES:\n```\n")
            parts.append(raw_input.code_changes)
            parts.append("\n```")

        parts.append("\n\n---\nStructure this reasoning into the specified JSON format.")

        return "".join(parts)

    def _build_episode(
        self,
        data: dict,
        project_name: str,
        source_assistant: str
    ) -> Episode:
        """Build an Episode from structured data."""
        # Parse reasoning_trace
        rt_data = data.get("reasoning_trace", {})
        reasoning_trace = ReasoningTrace(
            raw_thinking=rt_data.get("raw_thinking", data.get("task", "")),
            alternatives_considered=rt_data.get("alternatives_considered", []),
            decision_factors=rt_data.get("decision_factors", []),
            confidence_level=rt_data.get("confidence_level")
        )

        # Parse episode_type
        episode_type_str = data.get("episode_type", "decision")
        try:
            episode_type = EpisodeType(episode_type_str)
        except ValueError:
            episode_type = EpisodeType.DECISION

        return Episode(
            task=data.get("task", "Unspecified task"),
            context=data.get("context", "Unspecified context"),
            reasoning_trace=reasoning_trace,
            solution=data.get("solution", ""),
            solution_summary=data.get("solution_summary", ""),
            episode_type=episode_type,
            tags=data.get("tags", []),
            files_affected=data.get("files_affected", []),
            lessons_learned=data.get("lessons_learned", []),
            project_name=project_name,
            source_assistant=source_assistant
        )


# Convenience function for simple usage
async def process_thinking_text(
    thinking_text: str,
    user_prompt: Optional[str] = None,
    code_changes: Optional[str] = None,
    project_name: str = "default",
    source_assistant: str = "unknown"
) -> Episode:
    """
    Convenience function to process thinking text.

    Args:
        thinking_text: Model's reasoning text
        user_prompt: Original user prompt (optional)
        code_changes: Associated code changes (optional)
        project_name: Project name
        source_assistant: Source code assistant

    Returns:
        Structured Episode
    """
    processor = ThoughtProcessor()
    raw_input = ProcessedInput(
        raw_text=thinking_text,
        user_prompt=user_prompt,
        code_changes=code_changes,
        source="manual"
    )
    return await processor.process_thought(raw_input, project_name, source_assistant)
