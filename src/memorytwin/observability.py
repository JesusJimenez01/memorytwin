"""
Observability with Langfuse
===========================

Only 3 main traces:
1. Store Memory - LLM Input/Output when structuring thoughts
2. Access Memories - LLM Input/Output for RAG queries
3. Consolidate Memories - Consolidated episodes -> MetaMemory created

Uses the observation API of the Langfuse SDK (>= 3.4). Tracing is fully
optional: without the package or credentials every decorator is a no-op.

Configuration via .env:
  - LANGFUSE_PUBLIC_KEY
  - LANGFUSE_SECRET_KEY
  - LANGFUSE_HOST (optional)
"""

import logging
import os
import sys
from contextlib import contextmanager
from functools import wraps
from typing import Any, Iterator

# Importing config loads .env, so LANGFUSE_* variables are visible below
from memorytwin.config import get_settings  # noqa: F401

try:
    from langfuse import Langfuse  # type: ignore
except ImportError:
    Langfuse = None

# Silence noisy Langfuse warnings ("Calling end() on an ended span")
logging.getLogger("langfuse").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

__all__ = [
    "trace_store_memory",
    "trace_access_memory",
    "trace_consolidation",
    "trace_observation",
    "flush_traces",
]

# Singleton Langfuse client
_langfuse_client = None


def _is_disabled() -> bool:
    """Check if Langfuse is disabled (tests or missing credentials)."""
    if Langfuse is None:
        return True
    if "pytest" in sys.modules:
        return True
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return True
    return False


def _get_langfuse():
    """Get the singleton Langfuse client."""
    global _langfuse_client
    if _langfuse_client is None and not _is_disabled():
        try:
            _langfuse_client = Langfuse()
        except Exception as e:
            logger.warning("Langfuse disabled, client could not be created: %s", e)
    return _langfuse_client


def flush_traces():
    """Force sending pending traces."""
    client = _get_langfuse()
    if client:
        try:
            client.flush()
        except Exception as e:
            logger.debug("Could not flush Langfuse traces: %s", e)


@contextmanager
def trace_observation(name: str, as_type: str = "span", **attributes: Any) -> Iterator[Any]:
    """
    Trace a block of code as a Langfuse observation (span, generation...).

    Yields the observation so the caller can attach its output, or None when
    tracing is disabled, which keeps call sites free of Langfuse checks::

        with trace_observation("Oracle LLM Response", as_type="generation", model=m) as gen:
            answer = call_llm()
            if gen is not None:
                gen.update(output=answer)
    """
    client = None if _is_disabled() else _get_langfuse()
    if client is None:
        yield None
        return

    try:
        with client.start_as_current_observation(as_type=as_type, name=name, **attributes) as observation:
            yield observation
    finally:
        flush_traces()


def trace_store_memory(func):
    """
    Decorator for tracing memory storage.
    Captures: LLM input (thinking text) -> LLM output (structured episode)
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if _is_disabled():
            return await func(*args, **kwargs)

        # Extract input (raw_input is in args[1] or kwargs)
        raw_input = kwargs.get('raw_input') or (args[1] if len(args) > 1 else None)
        project_name = kwargs.get('project_name', 'default')
        input_text = raw_input.raw_text[:500] if raw_input else "N/A"

        client = _get_langfuse()
        if not client:
            return await func(*args, **kwargs)

        try:
            with client.start_as_current_observation(
                as_type="span",
                name="Store Memory",
                input={"thinking_text": input_text, "project": project_name},
                metadata={"project": project_name, "operation": "store"}
            ) as span:
                result = await func(*args, **kwargs)
                span.update(output={
                    "episode_id": str(result.id),
                    "task": result.task,
                    "type": result.episode_type.value,
                    "tags": result.tags[:5],
                    "lessons": result.lessons_learned[:3]
                })
                return result
        except Exception as e:
            # On error, still create a span to log it
            with client.start_as_current_observation(
                as_type="span",
                name="Store Memory - ERROR",
                input={"thinking_text": input_text},
                level="ERROR"
            ) as span:
                span.update(output={"error": str(e)}, status_message=str(e))
            raise
        finally:
            client.flush()

    return wrapper


def trace_access_memory(func):
    """
    Decorator for tracing memory access (RAG).
    Captures: user question -> generated response with context
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if _is_disabled():
            return await func(*args, **kwargs)

        # Extract input (question is in args[1] or kwargs)
        question = kwargs.get('question') or (args[1] if len(args) > 1 else "N/A")
        project_name = kwargs.get('project_name', 'all')

        client = _get_langfuse()
        if not client:
            return await func(*args, **kwargs)

        try:
            with client.start_as_current_observation(
                as_type="span",
                name="Access Memories",
                input={"question": question, "project": project_name},
                metadata={"project": project_name or "all", "operation": "access"}
            ) as span:
                result = await func(*args, **kwargs)
                span.update(output={
                    "answer": result.get("answer", "")[:500],
                    "episodes_count": len(result.get("episodes_used", [])),
                    "meta_memories_count": len(result.get("meta_memories_used", [])),
                    "context_provided": result.get("context_provided", False)
                })
                return result
        except Exception as e:
            with client.start_as_current_observation(
                as_type="span",
                name="Access Memories - ERROR",
                input={"question": question},
                level="ERROR"
            ) as span:
                span.update(output={"error": str(e)}, status_message=str(e))
            raise
        finally:
            client.flush()

    return wrapper


def trace_consolidation(func):
    """
    Decorator for tracing memory consolidation.
    Captures: episodes to consolidate -> created meta-memory
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if _is_disabled():
            return func(*args, **kwargs)

        # Extract episodes (in args[1] for class methods)
        episodes = kwargs.get('episodes') or (args[1] if len(args) > 1 else [])
        project_name = kwargs.get('project_name') or (args[2] if len(args) > 2 else 'unknown')

        episode_summaries = [
            {"id": str(ep.id), "task": ep.task[:100], "type": ep.episode_type.value}
            for ep in episodes[:10]
        ]

        client = _get_langfuse()
        if not client:
            return func(*args, **kwargs)

        try:
            with client.start_as_current_observation(
                as_type="span",
                name="Consolidate Memories",
                input={
                    "episodes_count": len(episodes),
                    "episodes": episode_summaries,
                    "project": project_name
                },
                metadata={"project": project_name, "operation": "consolidate"}
            ) as span:
                result = func(*args, **kwargs)
                if result:
                    span.update(output={
                        "meta_memory_id": str(result.id),
                        "pattern": result.pattern[:200],
                        "pattern_summary": result.pattern_summary[:200],
                        "lessons": result.lessons[:3],
                        "best_practices": result.best_practices[:3],
                        "confidence": result.confidence
                    })
                return result
        except Exception as e:
            with client.start_as_current_observation(
                as_type="span",
                name="Consolidate Memories - ERROR",
                input={"episodes_count": len(episodes)},
                level="ERROR"
            ) as span:
                span.update(output={"error": str(e)}, status_message=str(e))
            raise
        finally:
            client.flush()

    return wrapper
