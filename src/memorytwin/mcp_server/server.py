"""
MCP Server for Memory Twin
==========================

Implements the Model Context Protocol to expose the
capabilities of Escriba and Oráculo to compatible clients.

The server speaks JSON-RPC over stdio, so nothing in this process may
write to stdout: logs and progress output go to stderr.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
from uuid import UUID

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from memorytwin.consolidation import MemoryConsolidator
from memorytwin.escriba.processor import ThoughtProcessor
from memorytwin.escriba.storage import MemoryStorage
from memorytwin.mcp_server.tools import TOOLS
from memorytwin.models import Episode, EpisodeType, MemoryQuery, ProcessedInput, ReasoningTrace
from memorytwin.observability import trace_observation
from memorytwin.oraculo.rag_engine import RAGEngine

logger = logging.getLogger("memorytwin.mcp")

# Below this many episodes the whole memory fits in context; above it we switch
# to "smart" mode (recent + semantically relevant episodes only)
FULL_CONTEXT_THRESHOLD = 20

# Upper bound for a consolidation run triggered from an MCP client
CONSOLIDATION_TIMEOUT_SECONDS = 120.0


def _format_lessons(lessons: list) -> list:
    """Format lessons list ensuring datetime objects are serializable."""
    formatted = []
    for lesson in lessons:
        formatted_lesson = {}
        for key, value in lesson.items():
            if isinstance(value, datetime):
                formatted_lesson[key] = value.isoformat()
            else:
                formatted_lesson[key] = value
        formatted.append(formatted_lesson)
    return formatted


def _json_result(data: Any, is_error: bool = False) -> CallToolResult:
    """Wrap a JSON-serializable payload as an MCP tool result."""
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(data, indent=2, ensure_ascii=False, default=str))],
        isError=is_error,
    )


def _text_result(text: str, is_error: bool = False) -> CallToolResult:
    """Wrap plain text as an MCP tool result."""
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=is_error)


def _episode_brief(episode: dict) -> dict:
    """Compact view of a timeline entry for context responses."""
    return {key: episode[key] for key in ("id", "type", "task", "summary", "date", "tags")}


def _detect_project_name() -> str:
    """
    Auto-detect the project name based on the CWD.

    Strategy:
    1. Get the current working directory (CWD)
    2. Use the folder name as the project name
    3. Avoid generic names like 'home', 'Users', etc.

    Returns:
        Detected project name, or 'default' if it cannot be determined.
    """
    try:
        cwd = Path(os.getcwd())
        project_name = cwd.name

        # List of names to ignore (too generic)
        generic_names = {
            'home', 'users', 'user', 'desktop', 'documents',
            'downloads', 'tmp', 'temp', 'root', 'var', 'opt',
            'src', 'source', 'code', 'projects', 'repos', 'git',
            'c:', 'd:', 'e:'  # Windows drive roots
        }

        if project_name.lower() in generic_names:
            # Try going up one level if the name is generic
            parent_name = cwd.parent.name
            if parent_name.lower() not in generic_names:
                project_name = parent_name
            else:
                return "default"

        # Clean the name (remove problematic characters)
        project_name = project_name.strip().replace(' ', '_')

        if not project_name:
            return "default"

        logger.debug("Auto-detected project: %s", project_name)
        return project_name

    except Exception as e:
        logger.warning("Could not detect project: %s", e)
        return "default"


class MemoryTwinMCPServer:
    """
    MCP server that exposes Memory Twin tools.

    Tool groups (schemas live in ``memorytwin.mcp_server.tools``):
    - Capture: capture_thinking, capture_decision, capture_quick, onboard_project
    - Retrieval: get_project_context, query_memory, search_episodes, get_episode,
      get_timeline, get_lessons, get_statistics
    - Curation: mark_episode, consolidate_memories, check_consolidation_status
    """

    def __init__(self):
        """Initialize MCP server."""
        self.server = Server("memorytwin")
        self.processor: Optional[ThoughtProcessor] = None
        self.storage: Optional[MemoryStorage] = None
        self.rag_engine: Optional[RAGEngine] = None

        # Register tools
        self._register_tools()

        logger.info("Memory Twin MCP Server initialized")

    def _lazy_init(self):
        """
        Lazy initialization of components.

        Storage and retrieval are always available. The LLM processor is
        optional: without an API key, captures are stored unstructured and
        every read-only tool keeps working.
        """
        if self.storage is None:
            self.storage = MemoryStorage()
        if self.rag_engine is None:
            self.rag_engine = RAGEngine(storage=self.storage)
        if self.processor is None:
            try:
                self.processor = ThoughtProcessor()
            except Exception as e:
                logger.warning("LLM processor unavailable, captures will be stored raw: %s", e)

    def _register_tools(self):
        """Register all MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools."""
            return TOOLS

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
            """Execute a tool."""
            self._lazy_init()

            handler = self._handlers().get(name)
            if handler is None:
                return _text_result(f"Unknown tool: {name}", is_error=True)

            try:
                return await handler(arguments)
            except Exception as e:
                logger.exception("Error in tool %s", name)
                return _text_result(f"Error executing {name}: {e}", is_error=True)

    def _handlers(self) -> dict[str, Callable[[dict], Awaitable[CallToolResult]]]:
        """Map each MCP tool name to its handler."""
        return {
            "capture_thinking": self._capture_thinking,
            "capture_decision": self._capture_decision,
            "capture_quick": self._capture_quick,
            "query_memory": self._query_memory,
            "get_timeline": self._get_timeline,
            "get_lessons": self._get_lessons,
            "search_episodes": self._search_episodes,
            "get_statistics": self._get_statistics,
            "get_episode": self._get_episode,
            "onboard_project": self._onboard_project,
            "get_project_context": self._get_project_context,
            "consolidate_memories": self._consolidate_memories,
            "check_consolidation_status": self._check_consolidation_status,
            "mark_episode": self._mark_episode,
        }

    async def _structure_episode(
        self,
        raw_input: ProcessedInput,
        project_name: str,
        source_assistant: str,
        fallback: Callable[[], Episode],
        tool_name: str,
    ) -> tuple[Episode, bool]:
        """
        Structure a capture with the LLM, degrading to a raw episode on failure.

        Knowledge must never be lost because the LLM is unavailable (no API key,
        rate limit, invalid JSON...), so every capture tool provides a fallback
        that builds the episode directly from the user-supplied fields.

        Returns:
            Tuple (episode, structured_by_llm)
        """
        if self.processor is None:
            logger.warning("%s: LLM not configured, storing raw capture", tool_name)
            return fallback(), False

        try:
            episode = await self.processor.process_thought(
                raw_input,
                project_name=project_name,
                source_assistant=source_assistant
            )
            return episode, True
        except Exception as exc:
            logger.warning("%s fallback activated due to LLM processing error: %s", tool_name, exc)
            return fallback(), False

    async def _capture_thinking(self, args: dict) -> CallToolResult:
        """Capture thinking and store it."""
        project_name = args.get("project_name") or _detect_project_name()
        source_assistant = args.get("source_assistant", "unknown")

        raw_input = ProcessedInput(
            raw_text=args["thinking_text"],
            user_prompt=args.get("user_prompt"),
            code_changes=args.get("code_changes"),
            source="mcp"
        )

        def fallback() -> Episode:
            return Episode(
                task="Raw technical reasoning capture",
                context="Captured via MCP fallback without LLM structuring",
                reasoning_trace=ReasoningTrace(raw_thinking=args["thinking_text"]),
                solution=args.get("code_changes", ""),
                solution_summary="Raw capture stored without LLM structuring",
                episode_type=EpisodeType.LEARNING,
                tags=["mcp", "fallback", "raw_capture"],
                project_name=project_name,
                source_assistant=source_assistant,
            )

        episode, structured = await self._structure_episode(
            raw_input, project_name, source_assistant, fallback, "capture_thinking"
        )
        episode_id = self.storage.store_episode(episode)

        return _json_result({
            "success": True,
            "episode_id": episode_id,
            "project": project_name,
            "structured_by_llm": structured,
            "task": episode.task,
            "type": episode.episode_type.value,
            "tags": episode.tags,
            "lessons_learned": episode.lessons_learned
        })

    async def _capture_decision(self, args: dict) -> CallToolResult:
        """Capture a technical decision in structured format."""
        # Build structured text from separate fields
        parts = [f"## Task\n{args['task']}"]

        if args.get("context"):
            parts.append(f"## Context\n{args['context']}")

        if args.get("alternatives"):
            alts = "\n".join(f"- {alt}" for alt in args["alternatives"])
            parts.append(f"## Alternatives considered\n{alts}")

        parts.append(f"## Decision\n{args['decision']}")
        parts.append(f"## Reasoning\n{args['reasoning']}")

        if args.get("lesson"):
            parts.append(f"## Lesson learned\n{args['lesson']}")

        thinking_text = "\n\n".join(parts)
        project_name = args.get("project_name") or _detect_project_name()
        source_assistant = args.get("source_assistant", "unknown")

        raw_input = ProcessedInput(
            raw_text=thinking_text,
            code_changes=args.get("code_changes"),
            source="mcp"
        )

        def fallback() -> Episode:
            tags = ["mcp", "capture_decision", "fallback"]
            if args.get("alternatives"):
                tags.append("alternatives")
            return Episode(
                task=args["task"],
                context=args.get("context", "Captured via structured decision tool"),
                reasoning_trace=ReasoningTrace(
                    raw_thinking=thinking_text,
                    alternatives_considered=args.get("alternatives", []),
                    decision_factors=[args["reasoning"]],
                ),
                solution=args["decision"],
                solution_summary=args["decision"],
                episode_type=EpisodeType.DECISION,
                tags=tags,
                lessons_learned=[args["lesson"]] if args.get("lesson") else [],
                project_name=project_name,
                source_assistant=source_assistant,
            )

        episode, structured = await self._structure_episode(
            raw_input, project_name, source_assistant, fallback, "capture_decision"
        )
        episode_id = self.storage.store_episode(episode)

        return _json_result({
            "success": True,
            "episode_id": episode_id,
            "project": project_name,
            "structured_by_llm": structured,
            "task": episode.task,
            "decision": args["decision"],
            "type": episode.episode_type.value,
            "tags": episode.tags,
            "lessons_learned": episode.lessons_learned
        })

    async def _capture_quick(self, args: dict) -> CallToolResult:
        """Quick capture with minimum effort."""
        parts = [
            f"## What I did\n{args['what']}",
            f"## Why\n{args['why']}"
        ]
        if args.get("lesson"):
            parts.append(f"## Lesson\n{args['lesson']}")

        thinking_text = "\n\n".join(parts)
        project_name = args.get("project_name") or _detect_project_name()
        source_assistant = args.get("source_assistant", "unknown")

        raw_input = ProcessedInput(raw_text=thinking_text, source="mcp")

        def fallback() -> Episode:
            return Episode(
                task=args["what"],
                context=f"Reason: {args['why']}",
                reasoning_trace=ReasoningTrace(raw_thinking=thinking_text),
                solution=args["what"],
                solution_summary=args["what"],
                episode_type=EpisodeType.LEARNING,
                tags=["mcp", "capture_quick", "fallback"],
                lessons_learned=[args["lesson"]] if args.get("lesson") else [],
                project_name=project_name,
                source_assistant=source_assistant,
            )

        episode, structured = await self._structure_episode(
            raw_input, project_name, source_assistant, fallback, "capture_quick"
        )
        episode_id = self.storage.store_episode(episode)

        return _json_result({
            "success": True,
            "episode_id": episode_id,
            "project": project_name,
            "structured_by_llm": structured,
            "task": episode.task,
            "type": episode.episode_type.value,
            "lessons_learned": episode.lessons_learned
        })

    async def _query_memory(self, args: dict) -> CallToolResult:
        """Query memory with RAG."""
        result = await self.rag_engine.query(
            question=args["question"],
            project_name=args.get("project_name"),
            top_k=args.get("num_episodes", 5)
        )
        return _text_result(result["answer"])

    async def _get_timeline(self, args: dict) -> CallToolResult:
        """Get decision timeline."""
        timeline = self.rag_engine.get_timeline(
            project_name=args.get("project_name"),
            limit=args.get("limit", 20)
        )
        return _json_result(timeline)

    async def _get_lessons(self, args: dict) -> CallToolResult:
        """Get lessons learned."""
        lessons = self.rag_engine.get_lessons(
            project_name=args.get("project_name"),
            tags=args.get("tags")
        )

        formatted = [
            {
                "lesson": lesson["lesson"],
                "from_task": lesson["from_task"],
                "date": lesson["timestamp"].strftime("%Y-%m-%d"),
                "tags": lesson["tags"]
            }
            for lesson in lessons
        ]
        return _json_result(formatted)

    async def _search_episodes(self, args: dict) -> CallToolResult:
        """Semantic search of episodes."""
        query = MemoryQuery(
            query=args["query"],
            project_filter=args.get("project_name"),
            top_k=args.get("top_k", 5)
        )

        results = self.storage.search_episodes(query)

        formatted = [
            {
                "id": str(r.episode.id),
                "task": r.episode.task,
                "summary": r.episode.solution_summary,
                "type": r.episode.episode_type.value,
                "relevance": f"{r.relevance_score:.0%}",
                "date": r.episode.timestamp.strftime("%Y-%m-%d")
            }
            for r in results
        ]
        return _json_result(formatted)

    async def _get_statistics(self, args: dict) -> CallToolResult:
        """Get statistics."""
        return _json_result(self.storage.get_statistics(args.get("project_name")))

    async def _get_episode(self, args: dict) -> CallToolResult:
        """Get full episode by ID."""
        episode_id = args.get("episode_id")
        if not episode_id:
            return _text_result("Error: episode_id is required", is_error=True)

        episode = self.storage.get_episode_by_id(episode_id)
        if not episode:
            return _text_result(f"Episode not found with ID: {episode_id}", is_error=True)

        return _json_result({
            "id": str(episode.id),
            "timestamp": episode.timestamp.isoformat(),
            "task": episode.task,
            "context": episode.context,
            "reasoning_trace": {
                "raw_thinking": episode.reasoning_trace.raw_thinking,
                "alternatives_considered": episode.reasoning_trace.alternatives_considered,
                "decision_factors": episode.reasoning_trace.decision_factors,
                "confidence_level": episode.reasoning_trace.confidence_level
            },
            "solution": episode.solution,
            "solution_summary": episode.solution_summary,
            "outcome": episode.outcome,
            "success": episode.success,
            "episode_type": episode.episode_type.value,
            "tags": episode.tags,
            "files_affected": episode.files_affected,
            "lessons_learned": episode.lessons_learned,
            "source_assistant": episode.source_assistant,
            "project_name": episode.project_name,
            # Forgetting Curve fields
            "importance_score": episode.importance_score,
            "access_count": episode.access_count,
            "last_accessed": episode.last_accessed.isoformat() if episode.last_accessed else None,
            # Active memory flags
            "is_antipattern": episode.is_antipattern,
            "is_critical": episode.is_critical,
            "superseded_by": str(episode.superseded_by) if episode.superseded_by else None,
            "deprecation_reason": episode.deprecation_reason,
        })

    async def _onboard_project(self, args: dict) -> CallToolResult:
        """Analyze project and create an onboarding episode."""
        from memorytwin.escriba.project_analyzer import onboard_project

        project_path = args.get("project_path")
        if not project_path:
            return _text_result("Error: project_path is required", is_error=True)
        if not Path(project_path).is_dir():
            return _text_result(f"Error: directory does not exist: {project_path}", is_error=True)

        result = await onboard_project(
            project_path=project_path,
            project_name=args.get("project_name"),
            source_assistant="mcp-onboarding"
        )

        analysis = result['analysis']
        return _json_result({
            "success": True,
            "episode_id": result['episode_id'],
            "project_name": result['project_name'],
            "stack": [s['technology'] for s in analysis['stack']],
            "patterns": [p.get('pattern', p.get('directory', '')) for p in analysis['patterns']],
            "main_dependencies": analysis['dependencies']['main'][:10],
            "conventions": analysis['conventions'],
            "message": (
                f"Onboarding completed. The agent now knows the structure "
                f"of the {result['project_name']} project."
            )
        })

    async def _get_project_context(self, args: dict) -> CallToolResult:
        """
        Get intelligent project context, traced as a single Langfuse span.

        Args:
            project_name: Filter by project
            topic: Topic for semantic search
            include_reasoning: If True, includes full raw_thinking (more tokens)
        """
        project_name = args.get("project_name")
        topic = args.get("topic", "")
        include_reasoning = args.get("include_reasoning", False)

        with trace_observation(
            "Access Memories",
            input={"topic": topic or "no topic", "project": project_name or "all"},
            metadata={"project": project_name or "all", "operation": "get_project_context"},
        ) as span:
            result = self._build_project_context(project_name, topic, include_reasoning)
            if span is not None:
                span.update(output={
                    "mode": result["mode"],
                    "episodes_count": result["total_episodes"],
                    "meta_memories_count": result["total_meta_memories"],
                    "warnings_count": len(result.get("WARNINGS", [])),
                    "relevant_found": len(result.get("relevant_episodes", [])),
                    "meta_memories_found": len(result.get("meta_memories", [])),
                })

        return _json_result(result)

    def _build_project_context(self, project_name: Optional[str], topic: str, include_reasoning: bool) -> dict:
        """
        Assemble the project context with prioritization:

        0. ANTIPATTERNS relevant to the topic (warnings, shown first)
        1. META-MEMORIES (consolidated knowledge)
        2. Individual EPISODES (everything for small memories, recent + relevant otherwise)
        """
        stats = self.rag_engine.get_statistics(project_name)
        total_episodes = stats.get("total_episodes", 0)
        meta_stats = self.storage.get_meta_memory_statistics(project_name)

        result: dict[str, Any] = {
            "mode": "",
            "total_episodes": total_episodes,
            "total_meta_memories": meta_stats.get("total_meta_memories", 0),
            "statistics": stats,
            "meta_statistics": meta_stats
        }

        if total_episodes == 0:
            result["mode"] = "empty"
            result["message"] = (
                "No memories recorded yet. "
                "Consider running onboard_project to create initial context."
            )
            return result

        # A single semantic search feeds both the antipattern warnings and the
        # relevant episodes, so each retrieved episode is reinforced only once
        topic_results = []
        if topic:
            topic_results = self.storage.search_episodes(
                MemoryQuery(query=topic, project_filter=project_name, top_k=10)
            )

        # PRIORITY 0: antipatterns (critical warnings)
        warnings = []
        for r in topic_results:
            if not r.episode.is_antipattern:
                continue
            warning = {
                "type": "ANTIPATTERN",
                "severity": "HIGH",
                "task": r.episode.task,
                "lesson": r.episode.lessons_learned[0] if r.episode.lessons_learned else "Avoid this approach",
                "relevance": f"{r.relevance_score:.0%}"
            }
            if include_reasoning:
                warning["reasoning"] = r.episode.reasoning_trace.raw_thinking
            warnings.append(warning)

        if warnings:
            result["WARNINGS"] = warnings
            result["warning_note"] = (
                "ATTENTION: Relevant antipatterns found. "
                "Review these warnings BEFORE proceeding."
            )

        # PRIORITY 1: meta-memories (consolidated knowledge)
        meta_memories_included = []
        if meta_stats.get("total_meta_memories", 0) > 0:
            if topic:
                meta_memories_included = [
                    {
                        **self._meta_memory_brief(r.meta_memory),
                        "relevance": f"{r.relevance_score:.0%}"
                    }
                    for r in self.storage.search_meta_memories(query=topic, project_name=project_name, top_k=3)
                ]
            else:
                meta_memories_included = [
                    self._meta_memory_brief(mm)
                    for mm in self.storage.get_meta_memories_by_project(
                        project_name=project_name or "default",
                        limit=3
                    )
                ]

        if meta_memories_included:
            result["meta_memories"] = meta_memories_included
            result["meta_memory_note"] = "META-MEMORIES: Consolidated knowledge from multiple episodes."

        consolidation_check = self.storage.check_consolidation_needed(project_name)
        if consolidation_check.get("should_consolidate"):
            result["consolidation_recommendation"] = {
                "should_consolidate": True,
                "reason": (
                    f"There are {consolidation_check['hot_episodes_count']} high-usage episodes "
                    f"or {consolidation_check['estimated_unconsolidated']} unconsolidated"
                ),
                "suggestion": "Run the consolidate_memories tool or: mt consolidate --project <name>"
            }

        # PRIORITY 2: individual episodes
        if total_episodes < FULL_CONTEXT_THRESHOLD:
            result["mode"] = "full_context"
            result["message"] = f"Small memory ({total_episodes} episodes) - showing full context."
            timeline = self.rag_engine.get_timeline(limit=total_episodes, project_name=project_name)
            result["episodes"] = [_episode_brief(ep) for ep in timeline]
        else:
            result["mode"] = "smart_context"
            result["message"] = f"Mature memory ({total_episodes} episodes) - showing optimized context."
            recent = self.rag_engine.get_timeline(limit=5, project_name=project_name)
            result["recent_episodes"] = [_episode_brief(ep) for ep in recent]

            if topic:
                relevant_episodes = []
                for r in topic_results[:5]:
                    ep_data = {
                        "id": str(r.episode.id),
                        "type": r.episode.episode_type.value,
                        "task": r.episode.task,
                        "summary": r.episode.solution_summary,
                        "relevance": f"{r.relevance_score:.0%}",
                        "tags": r.episode.tags,
                        "lessons": r.episode.lessons_learned,
                        "is_critical": r.episode.is_critical
                    }
                    if include_reasoning:
                        ep_data["reasoning"] = r.episode.reasoning_trace.raw_thinking
                        ep_data["alternatives"] = r.episode.reasoning_trace.alternatives_considered
                        ep_data["decision_factors"] = r.episode.reasoning_trace.decision_factors
                    relevant_episodes.append(ep_data)
                result["relevant_episodes"] = relevant_episodes
            else:
                result["tip"] = "Provide a 'topic' to get semantically relevant episodes."

        if topic:
            lessons = self.rag_engine.get_lessons(project_name=project_name)
            result["aggregated_lessons"] = _format_lessons(lessons)

        return result

    @staticmethod
    def _meta_memory_brief(meta_memory) -> dict:
        """Compact view of a meta-memory for context responses."""
        return {
            "id": str(meta_memory.id),
            "pattern": meta_memory.pattern_summary,
            "lessons": meta_memory.lessons[:3],
            "best_practices": meta_memory.best_practices[:2],
            "technologies": meta_memory.technologies,
            "episode_count": meta_memory.episode_count,
            "confidence": f"{meta_memory.confidence:.0%}"
        }

    async def _consolidate_memories(self, args: dict) -> CallToolResult:
        """
        Consolidate similar episodes into meta-memories.

        Uses DBSCAN clustering + LLM to synthesize knowledge.
        """
        project_name = args.get("project_name")
        if not project_name:
            return _text_result("Error: project_name is required to consolidate memories", is_error=True)

        min_cluster_size = args.get("min_cluster_size", 3)
        force = bool(args.get("force", False))

        stats = self.storage.get_statistics(project_name)
        total_episodes = stats['total_episodes']

        if total_episodes < min_cluster_size:
            return _json_result({
                "success": False,
                "message": (
                    f"Project '{project_name}' only has {total_episodes} episodes. "
                    f"At least {min_cluster_size} are needed to consolidate."
                ),
                "total_episodes": total_episodes,
                "min_required": min_cluster_size
            })

        try:
            consolidator = MemoryConsolidator(storage=self.storage, min_cluster_size=min_cluster_size)
            # Clustering + LLM calls are blocking: run them off the event loop
            meta_memories = await asyncio.wait_for(
                asyncio.to_thread(consolidator.consolidate_project, project_name, force),
                timeout=CONSOLIDATION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            return _json_result({
                "success": False,
                "message": (
                    f"Consolidation exceeded the time limit ({CONSOLIDATION_TIMEOUT_SECONDS:.0f}s). "
                    "It keeps running in the background; results will appear in later queries."
                ),
                "suggestion": "Try a larger min_cluster_size to reduce the number of clusters"
            }, is_error=True)
        except Exception as e:
            logger.error("Error in consolidation: %s", e)
            return _text_result(f"Error during consolidation: {e}", is_error=True)

        if not meta_memories:
            return _json_result({
                "success": False,
                "message": (
                    "No new clusters large enough to consolidate were found. "
                    "Try a smaller min_cluster_size, or force=true to re-consolidate."
                ),
                "total_episodes": total_episodes,
                "min_cluster_size": min_cluster_size,
                "suggestion": "Episodes may be too semantically diverse or already consolidated"
            })

        return _json_result({
            "success": True,
            "message": f"Consolidation completed! {len(meta_memories)} meta-memories generated.",
            "meta_memories_generated": len(meta_memories),
            "episodes_consolidated": sum(mm.episode_count for mm in meta_memories),
            "meta_memories": [
                {
                    "id": str(mm.id),
                    "pattern": mm.pattern_summary,
                    "lessons_count": len(mm.lessons),
                    "best_practices_count": len(mm.best_practices),
                    "episode_count": mm.episode_count,
                    "confidence": f"{mm.confidence:.0%}",
                    "technologies": mm.technologies
                }
                for mm in meta_memories
            ]
        })

    async def _mark_episode(self, args: dict) -> CallToolResult:
        """
        Mark an episode with special flags (antipattern, critical, superseded, deprecated).
        """
        episode_id = args.get("episode_id")
        if not episode_id:
            return _text_result("Error: episode_id is required", is_error=True)

        episode = self.storage.get_episode_by_id(episode_id)
        if not episode:
            return _text_result(f"Error: Episode not found with ID {episode_id}", is_error=True)

        updates = {
            field: args[field]
            for field in ("is_antipattern", "is_critical", "superseded_by", "deprecation_reason")
            if args.get(field) is not None
        }

        if not updates:
            return _text_result(
                "No changes specified. Provide at least one of: "
                "is_antipattern, is_critical, superseded_by, deprecation_reason",
                is_error=True
            )

        # superseded_by is parsed back into a UUID on every read: reject bad values upfront
        if "superseded_by" in updates:
            try:
                updates["superseded_by"] = str(UUID(str(updates["superseded_by"])))
            except ValueError:
                return _text_result(
                    f"Error: superseded_by must be an episode UUID, got '{updates['superseded_by']}'",
                    is_error=True
                )

        if not self.storage.update_episode_flags(episode_id, updates):
            return _text_result(f"Error updating episode {episode_id}", is_error=True)

        messages = []
        if updates.get("is_antipattern"):
            messages.append(
                "Episode marked as ANTIPATTERN. It will be shown as a warning in future relevant queries."
            )
        if updates.get("is_critical"):
            messages.append("Episode marked as CRITICAL. It will be prioritized in searches.")
        if "superseded_by" in updates:
            messages.append(f"Episode superseded by {updates['superseded_by']}.")

        return _json_result({
            "success": True,
            "episode_id": episode_id,
            "task": episode.task,
            "updates_applied": updates,
            "message": " ".join(messages) or "Episode updated."
        })

    async def _check_consolidation_status(self, args: dict) -> CallToolResult:
        """
        Check if consolidation is needed.

        Analyzes episode access_count and number of unconsolidated episodes.
        """
        status = self.storage.check_consolidation_needed(args.get("project_name"))

        if status["should_consolidate"]:
            status["recommendation"] = (
                "CONSOLIDATION RECOMMENDED: "
                f"There are {status['hot_episodes_count']} high-usage episodes "
                f"and approximately {status['estimated_unconsolidated']} unconsolidated. "
                "Run consolidate_memories to generate meta-memories."
            )
        else:
            status["recommendation"] = (
                "CONSOLIDATION NOT NEEDED YET: "
                f"There are only {status['total_episodes']} episodes, "
                f"with {status['hot_episodes_count']} high-usage. "
                "The system will work well with individual episodes for now."
            )

        return _json_result(status)

    async def run(self):
        """Run the MCP server."""
        logger.info("Starting Memory Twin MCP Server...")
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


def _configure_logging() -> None:
    """Send logs to stderr: stdout carries the MCP JSON-RPC stream."""
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


async def _async_main():
    """Async entry point for the MCP server."""
    _configure_logging()
    server = MemoryTwinMCPServer()
    await server.run()


def main():
    """Synchronous entry point for console scripts."""
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
