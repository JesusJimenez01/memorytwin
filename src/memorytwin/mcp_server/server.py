"""
MCP Server for Memory Twin
============================

Implements the Model Context Protocol to expose the
capabilities of Escriba and Oráculo to compatible clients.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
)

# Importar observabilidad (config.py ya carga .env)
from memorytwin.observability import _get_langfuse, _is_disabled, flush_traces


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
from memorytwin.escriba.processor import ThoughtProcessor
from memorytwin.escriba.storage import MemoryStorage
from memorytwin.models import Episode, EpisodeType, MemoryQuery, ProcessedInput, ReasoningTrace
from memorytwin.oraculo.rag_engine import RAGEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memorytwin.mcp")


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

        logger.debug(f"Auto-detected project: {project_name}")
        return project_name

    except Exception as e:
        logger.warning(f"Could not detect project: {e}")
        return "default"


class MemoryTwinMCPServer:
    """
    MCP server that exposes Memory Twin tools.

    Available tools:
    - capture_thinking: Capture an assistant's reasoning
    - query_memory: Query memories with RAG
    - get_timeline: Get decision timeline
    - get_lessons: Get lessons learned
    - get_episode: Get episode by ID
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
        """Lazy initialization of components."""
        if self.processor is None:
            self.processor = ThoughtProcessor()
        if self.storage is None:
            self.storage = MemoryStorage()
        if self.rag_engine is None:
            self.rag_engine = RAGEngine(storage=self.storage)

    def _register_tools(self):
        """Register all MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="capture_thinking",
                    description=(
                        "Captures and stores the reasoning ('thinking') of an AI assistant. "
                        "Processes the text with an LLM to structure it into a memory episode "
                        "that includes task, context, alternatives considered, decision factors, "
                        "solution, and lessons learned."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "thinking_text": {
                                "type": "string",
                                "description": "Visible model reasoning text"
                            },
                            "user_prompt": {
                                "type": "string",
                                "description": "Original user prompt (optional)"
                            },
                            "code_changes": {
                                "type": "string",
                                "description": "Associated code changes (optional)"
                            },
                            "source_assistant": {
                                "type": "string",
                                "description": "Source assistant: copilot, claude, cursor, etc.",
                                "default": "unknown"
                            },
                            "project_name": {
                                "type": "string",
                                "description": "Project name (auto-detected from working directory if not specified)"
                            }
                        },
                        "required": ["thinking_text"]
                    }
                ),
                # Structured capture tool: capture_decision
                Tool(
                    name="capture_decision",
                    description=(
                        "PREFERRED TOOL FOR CAPTURING DECISIONS.\n\n"
                        "Captures a technical decision in a structured format with separate fields. "
                        "More convenient than capture_thinking when data is organized.\n\n"
                        "Usage example:\n"
                        "- task: 'Choose database for the project'\n"
                        "- decision: 'PostgreSQL'\n"
                        "- alternatives: ['MongoDB', 'MySQL', 'SQLite']\n"
                        "- reasoning: 'We need ACID transactions and complex queries'\n"
                        "- lesson: 'For relational data with transactions, SQL > NoSQL'"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "Brief description of the task or problem solved"
                            },
                            "decision": {
                                "type": "string",
                                "description": "The decision or solution taken"
                            },
                            "alternatives": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Alternatives considered (optional)"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Why this decision was made"
                            },
                            "lesson": {
                                "type": "string",
                                "description": "Lesson learned for the future (optional)"
                            },
                            "context": {
                                "type": "string",
                                "description": "Additional relevant context (optional)"
                            },
                            "code_changes": {
                                "type": "string",
                                "description": "Associated code changes (optional)"
                            },
                            "source_assistant": {
                                "type": "string",
                                "description": "Source assistant: copilot, claude, cursor, etc.",
                                "default": "unknown"
                            },
                            "project_name": {
                                "type": "string",
                                "description": "Project name (auto-detected from working directory if not specified)"
                            }
                        },
                        "required": ["task", "decision", "reasoning"]
                    }
                ),
                # Quick capture tool: capture_quick
                Tool(
                    name="capture_quick",
                    description=(
                        "QUICK CAPTURE - Minimum effort.\n\n"
                        "For when you need to save something quickly without much detail. "
                        "Only requires WHAT (what you did) and WHY (why you did it).\n\n"
                        "Examples:\n"
                        "- what: 'Added retry logic to HTTP client'\n"
                        "  why: 'API calls were failing intermittently'\n\n"
                        "- what: 'Switched from axios to fetch'\n"
                        "  why: 'Reduce dependencies, native fetch is sufficient'"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "what": {
                                "type": "string",
                                "description": "What did you do? (action or change performed)"
                            },
                            "why": {
                                "type": "string",
                                "description": "Why did you do it? (reason or problem solved)"
                            },
                            "lesson": {
                                "type": "string",
                                "description": "Lesson learned (optional but recommended)"
                            },
                            "project_name": {
                                "type": "string",
                                "description": "Project name (auto-detected from working directory if not specified)"
                            },
                            "source_assistant": {
                                "type": "string",
                                "description": "Source assistant",
                                "default": "unknown"
                            }
                        },
                        "required": ["what", "why"]
                    }
                ),
                Tool(
                    name="query_memory",
                    description=(
                        "Queries episodic memories using RAG (Retrieval-Augmented Generation). "
                        "Allows asking questions like 'Why did we choose X?' or "
                        "'What alternatives did we consider for Y?'. "
                        "Returns a generated answer based on relevant episodes."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Question to answer using stored memories"
                            },
                            "project_name": {
                                "type": "string",
                                "description": "Filter by specific project (optional)"
                            },
                            "num_episodes": {
                                "type": "integer",
                                "description": "Number of episodes to query (1-10)",
                                "default": 5
                            }
                        },
                        "required": ["question"]
                    }
                ),
                Tool(
                    name="get_timeline",
                    description=(
                        "Gets the chronological timeline of technical decisions. "
                        "Useful for viewing project evolution and understanding what was done and when."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": {
                                "type": "string",
                                "description": "Filter by project (optional)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum episodes to return",
                                "default": 20
                            }
                        },
                        "required": []
                    }
                ),
                Tool(
                    name="get_lessons",
                    description=(
                        "Gets aggregated lessons learned from multiple episodes. "
                        "Useful for onboarding and avoiding past mistakes."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": {
                                "type": "string",
                                "description": "Filter by project (optional)"
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Filter by specific tags (optional)"
                            }
                        },
                        "required": []
                    }
                ),
                Tool(
                    name="search_episodes",
                    description=(
                        "Simple semantic search of episodes. "
                        "Returns the most relevant episodes for a search term."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search term"
                            },
                            "project_name": {
                                "type": "string",
                                "description": "Filter by project (optional)"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_statistics",
                    description=(
                        "Gets memory database statistics: "
                        "total episodes, distribution by type and assistant."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": {
                                "type": "string",
                                "description": "Filter by project (optional)"
                            }
                        },
                        "required": []
                    }
                ),
                Tool(
                    name="get_episode",
                    description=(
                        "Gets the FULL content of a specific episode by its ID. "
                        "Includes the complete reasoning (thinking), all alternatives considered, "
                        "decision factors, detailed context, and lessons learned. "
                        "Use when you need to dive into the details of a specific decision."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "episode_id": {
                                "type": "string",
                                "description": "UUID of the episode to retrieve"
                            }
                        },
                        "required": ["episode_id"]
                    }
                ),
                Tool(
                    name="onboard_project",
                    description=(
                        "Analyzes an existing project and creates an 'onboarding' episode with information "
                        "about its structure, tech stack, architectural patterns, dependencies, "
                        "and conventions. Use when starting work on a new or unfamiliar project "
                        "to provide initial context to the agent."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_path": {
                                "type": "string",
                                "description": "Absolute path to the project to analyze"
                            },
                            "project_name": {
                                "type": "string",
                                "description": "Project name (auto-detected if not specified)"
                            }
                        },
                        "required": ["project_path"]
                    }
                ),
                Tool(
                    name="get_project_context",
                    description=(
                        "MAIN TOOL - USE ALWAYS AT THE START OF EACH TASK.\n\n"
                        "Gets intelligent context with PRIORITIZATION:\n"
                        "0. ANTIPATTERNS: Warnings about previous errors (if relevant)\n"
                        "1. META-MEMORIES: Consolidated knowledge and patterns\n"
                        "2. EPISODES: Relevant individual decisions\n\n"
                        "If there are antipattern WARNINGS, you MUST review them before proceeding.\n"
                        "Use include_reasoning=true to get the full reasoning."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Topic or keywords for the current task (for semantic search)"
                            },
                            "project_name": {
                                "type": "string",
                                "description": "Filter by specific project (optional)"
                            },
                            "include_reasoning": {
                                "type": "boolean",
                                "description": (
                                    "If true, includes full raw_thinking from relevant episodes "
                                    "(more tokens but more context)"
                                )
                            }
                        },
                        "required": []
                    }
                ),
                Tool(
                    name="consolidate_memories",
                    description=(
                        "Consolidates similar episodes into META-MEMORIES using clustering and LLM. "
                        "Meta-memories group recurring patterns, lessons learned, and best practices, "
                        "enabling quick access to consolidated knowledge without searching individual "
                        "episodes.\n\n"
                        "Use when:\n"
                        "- The system suggests consolidation (indicator in get_project_context)\n"
                        "- There are many unconsolidated episodes (>20)\n"
                        "- Episodes with high access_count indicate frequent patterns\n\n"
                        "Result: Meta-memories with patterns, lessons, best practices, and antipatterns."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": {
                                "type": "string",
                                "description": "Project name to consolidate (required)"
                            },
                            "min_cluster_size": {
                                "type": "integer",
                                "description": "Minimum episodes to form a cluster (default: 3)"
                            }
                        },
                        "required": ["project_name"]
                    }
                ),
                Tool(
                    name="check_consolidation_status",
                    description=(
                        "Checks if the project needs memory consolidation. "
                        "Analyzes episodes with high usage (access_count) and number of unconsolidated "
                        "episodes. Useful for deciding whether to run consolidate_memories."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": {
                                "type": "string",
                                "description": "Project name to check (optional)"
                            }
                        },
                        "required": []
                    }
                ),
                Tool(
                    name="mark_episode",
                    description=(
                        "Marks an episode with special flags:\n"
                        "- is_antipattern=true: This episode represents a MISTAKE or something NOT to do. "
                        "It will be shown as a WARNING in future queries.\n"
                        "- is_critical=true: This episode is critical and should be prioritized in searches.\n"
                        "- superseded_by: UUID of the episode that replaces this one.\n"
                        "- deprecation_reason: Reason why this episode no longer applies.\n\n"
                        "Use after discovering that a previous solution was incorrect or to highlight "
                        "important decisions."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "episode_id": {
                                "type": "string",
                                "description": "UUID of the episode to mark"
                            },
                            "is_antipattern": {
                                "type": "boolean",
                                "description": "Mark as antipattern (something NOT to do)"
                            },
                            "is_critical": {
                                "type": "boolean",
                                "description": "Mark as critical (high priority)"
                            },
                            "superseded_by": {
                                "type": "string",
                                "description": "UUID of the episode that replaces this one"
                            },
                            "deprecation_reason": {
                                "type": "string",
                                "description": "Reason why this episode no longer applies"
                            }
                        },
                        "required": ["episode_id"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
            """Execute a tool."""
            self._lazy_init()

            try:
                match name:
                    case "capture_thinking":
                        return await self._capture_thinking(arguments)
                    case "capture_decision":
                        return await self._capture_decision(arguments)
                    case "capture_quick":
                        return await self._capture_quick(arguments)
                    case "query_memory":
                        return await self._query_memory(arguments)
                    case "get_timeline":
                        return await self._get_timeline(arguments)
                    case "get_lessons":
                        return await self._get_lessons(arguments)
                    case "search_episodes":
                        return await self._search_episodes(arguments)
                    case "get_statistics":
                        return await self._get_statistics(arguments)
                    case "get_episode":
                        return await self._get_episode(arguments)
                    case "onboard_project":
                        return await self._onboard_project(arguments)
                    case "get_project_context":
                        return await self._get_project_context(arguments)
                    case "consolidate_memories":
                        return await self._consolidate_memories(arguments)
                    case "check_consolidation_status":
                        return await self._check_consolidation_status(arguments)
                    case "mark_episode":
                        return await self._mark_episode(arguments)
                    case _:
                        return CallToolResult(
                            content=[TextContent(
                                type="text",
                                text=f"Unknown tool: {name}"
                            )],
                            isError=True
                        )
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error executing {name}: {str(e)}"
                    )],
                    isError=True
                )

    async def _capture_thinking(self, args: dict) -> CallToolResult:
        """Capture thinking and store it."""
        # Auto-detect project if not provided
        project_name = args.get("project_name") or _detect_project_name()

        raw_input = ProcessedInput(
            raw_text=args["thinking_text"],
            user_prompt=args.get("user_prompt"),
            code_changes=args.get("code_changes"),
            source="mcp"
        )

        source_assistant = args.get("source_assistant", "unknown")

        try:
            episode = await self.processor.process_thought(
                raw_input,
                project_name=project_name,
                source_assistant=source_assistant
            )
        except Exception as exc:
            logger.warning(
                "capture_thinking fallback activated due to LLM processing error: %s",
                exc,
            )
            episode = Episode(
                task="Raw technical reasoning capture",
                context="Captured via MCP fallback without LLM structuring",
                reasoning_trace=ReasoningTrace(raw_thinking=args["thinking_text"]),
                solution=args.get("code_changes", ""),
                solution_summary="Raw capture stored without LLM structuring",
                episode_type=EpisodeType.LEARNING,
                tags=["mcp", "fallback", "raw_capture"],
                lessons_learned=[],
                project_name=project_name,
                source_assistant=source_assistant,
            )

        episode_id = self.storage.store_episode(episode)

        result = {
            "success": True,
            "episode_id": episode_id,
            "project": project_name,  # Project where it was saved
            "task": episode.task,
            "type": episode.episode_type.value,
            "tags": episode.tags,
            "lessons_learned": episode.lessons_learned
        }

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False)
            )]
        )

    async def _capture_decision(self, args: dict) -> CallToolResult:
        """Capture a technical decision in structured format."""
        # Build structured text from separate fields
        parts = []
        parts.append(f"## Task\n{args['task']}")

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

        # Auto-detect project if not provided
        project_name = args.get("project_name") or _detect_project_name()

        raw_input = ProcessedInput(
            raw_text=thinking_text,
            code_changes=args.get("code_changes"),
            source="mcp"
        )

        source_assistant = args.get("source_assistant", "unknown")

        try:
            episode = await self.processor.process_thought(
                raw_input,
                project_name=project_name,
                source_assistant=source_assistant
            )
        except Exception as exc:
            logger.warning(
                "capture_decision fallback activated due to LLM processing error: %s",
                exc,
            )
            lessons = [args["lesson"]] if args.get("lesson") else []
            fallback_tags = ["mcp", "capture_decision", "fallback"]
            if args.get("alternatives"):
                fallback_tags.append("alternatives")

            episode = Episode(
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
                tags=fallback_tags,
                lessons_learned=lessons,
                project_name=project_name,
                source_assistant=source_assistant,
            )

        episode_id = self.storage.store_episode(episode)

        result = {
            "success": True,
            "episode_id": episode_id,
            "project": project_name,  # Project where it was saved
            "task": episode.task,
            "decision": args["decision"],
            "type": episode.episode_type.value,
            "tags": episode.tags,
            "lessons_learned": episode.lessons_learned
        }

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False)
            )]
        )

    async def _capture_quick(self, args: dict) -> CallToolResult:
        """Quick capture with minimum effort."""
        # Build simple text from what/why
        parts = [
            f"## What I did\n{args['what']}",
            f"## Why\n{args['why']}"
        ]

        if args.get("lesson"):
            parts.append(f"## Lesson\n{args['lesson']}")

        thinking_text = "\n\n".join(parts)

        # Auto-detect project if not provided
        project_name = args.get("project_name") or _detect_project_name()

        raw_input = ProcessedInput(
            raw_text=thinking_text,
            source="mcp"
        )

        source_assistant = args.get("source_assistant", "unknown")

        try:
            episode = await self.processor.process_thought(
                raw_input,
                project_name=project_name,
                source_assistant=source_assistant
            )
        except Exception as exc:
            logger.warning(
                "capture_quick fallback activated due to LLM processing error: %s",
                exc,
            )
            lessons = [args["lesson"]] if args.get("lesson") else []
            episode = Episode(
                task=args["what"],
                context=f"Reason: {args['why']}",
                reasoning_trace=ReasoningTrace(raw_thinking=thinking_text),
                solution=args["what"],
                solution_summary=args["what"],
                episode_type=EpisodeType.LEARNING,
                tags=["mcp", "capture_quick", "fallback"],
                lessons_learned=lessons,
                project_name=project_name,
                source_assistant=source_assistant,
            )

        episode_id = self.storage.store_episode(episode)

        result = {
            "success": True,
            "episode_id": episode_id,
            "project": project_name,  # Include project in response
            "task": episode.task,
            "type": episode.episode_type.value,
            "lessons_learned": episode.lessons_learned
        }

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False)
            )]
        )

    async def _query_memory(self, args: dict) -> CallToolResult:
        """Query memory with RAG."""
        result = await self.rag_engine.query(
            question=args["question"],
            project_name=args.get("project_name"),
            top_k=args.get("num_episodes", 5)
        )

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=result["answer"]
            )]
        )

    async def _get_timeline(self, args: dict) -> CallToolResult:
        """Get decision timeline."""
        timeline = self.rag_engine.get_timeline(
            project_name=args.get("project_name"),
            limit=args.get("limit", 20)
        )

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(timeline, indent=2, ensure_ascii=False)
            )]
        )

    async def _get_lessons(self, args: dict) -> CallToolResult:
        """Get lessons learned."""
        lessons = self.rag_engine.get_lessons(
            project_name=args.get("project_name"),
            tags=args.get("tags")
        )

        # Format for readability
        formatted = []
        for lesson in lessons:
            formatted.append({
                "lesson": lesson["lesson"],
                "from_task": lesson["from_task"],
                "date": lesson["timestamp"].strftime("%Y-%m-%d"),
                "tags": lesson["tags"]
            })

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(formatted, indent=2, ensure_ascii=False)
            )]
        )

    async def _search_episodes(self, args: dict) -> CallToolResult:
        """Semantic search of episodes."""
        query = MemoryQuery(
            query=args["query"],
            project_filter=args.get("project_name"),
            top_k=args.get("top_k", 5)
        )

        results = self.storage.search_episodes(query)

        formatted = []
        for r in results:
            formatted.append({
                "id": str(r.episode.id),
                "task": r.episode.task,
                "summary": r.episode.solution_summary,
                "type": r.episode.episode_type.value,
                "relevance": f"{r.relevance_score:.0%}",
                "date": r.episode.timestamp.strftime("%Y-%m-%d")
            })

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(formatted, indent=2, ensure_ascii=False)
            )]
        )

    async def _get_statistics(self, args: dict) -> CallToolResult:
        """Get statistics."""
        stats = self.storage.get_statistics(args.get("project_name"))

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(stats, indent=2, ensure_ascii=False)
            )]
        )

    async def _get_episode(self, args: dict) -> CallToolResult:
        """Get full episode by ID."""
        episode_id = args.get("episode_id")

        if not episode_id:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="Error: episode_id is required"
                )],
                isError=True
            )

        episode = self.storage.get_episode_by_id(episode_id)

        if not episode:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Episode not found with ID: {episode_id}"
                )],
                isError=True
            )

        # Return the full episode with all information
        full_episode = {
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
            "last_accessed": episode.last_accessed.isoformat() if episode.last_accessed else None
        }

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(full_episode, indent=2, ensure_ascii=False)
            )]
        )

    async def _onboard_project(self, args: dict) -> CallToolResult:
        """Analyze project and create an onboarding episode."""
        from memorytwin.escriba.project_analyzer import onboard_project

        project_path = args.get("project_path")
        if not project_path:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="Error: project_path is required"
                )],
                isError=True
            )

        result = await onboard_project(
            project_path=project_path,
            project_name=args.get("project_name"),
            source_assistant="mcp-onboarding"
        )

        # Analysis summary
        analysis = result['analysis']
        summary = {
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
        }

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(summary, indent=2, ensure_ascii=False)
            )]
        )

    async def _get_project_context(self, args: dict) -> CallToolResult:
        """
        Get intelligent project context.

        Hybrid strategy with prioritization:
        0. FIRST: Search for relevant ANTIPATTERNS (warnings)
        1. Then META-MEMORIES (consolidated knowledge)
        2. Finally individual EPISODES

        Args:
            project_name: Filter by project
            topic: Topic for semantic search
            include_reasoning: If True, includes full raw_thinking (more tokens)
        """
        project_name = args.get("project_name")
        topic = args.get("topic", "")
        include_reasoning = args.get("include_reasoning", False)

        threshold = 20  # Threshold to switch strategy

        # Trace memory access
        langfuse = _get_langfuse() if not _is_disabled() else None
        span_ctx = None
        output_data = {}

        try:
            if langfuse:
                # Use context manager for the span
                span_ctx = langfuse.start_as_current_span(
                    name="Access Memories",
                    input={"topic": topic or "no topic", "project": project_name or "all"},
                    metadata={"project": project_name or "all", "operation": "get_project_context"}
                )
                span_ctx.__enter__()

            # Get base statistics
            stats = self.rag_engine.get_statistics()
            total_episodes = stats.get("total_episodes", 0)

            # Get meta-memory statistics
            meta_stats = self.storage.get_meta_memory_statistics(project_name)

            result = {
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
                # Save output for the span
                output_data = {"mode": "empty", "episodes": 0, "message": "No memories"}
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, ensure_ascii=False)
                    )]
                )

            # =================================================================
            # PRIORITY 0: ANTIPATTERNS (CRITICAL WARNINGS)
            # =================================================================
            warnings = []
            if topic:
                query = MemoryQuery(
                    query=topic,
                    project_filter=project_name,
                    top_k=10
                )
                all_results = self.storage.search_episodes(query)
                for r in all_results:
                    if getattr(r.episode, 'is_antipattern', False):
                        warning = {
                            "type": "ANTIPATTERN",
                            "severity": "HIGH",
                            "task": r.episode.task,
                            "lesson": (
                                r.episode.lessons_learned[0]
                                if r.episode.lessons_learned
                                else "Avoid this approach"
                            ),
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

            # =================================================================
            # PRIORITY 1: META-MEMORIES (Consolidated Knowledge)
            # =================================================================
            meta_memories_included = []
            if meta_stats.get("total_meta_memories", 0) > 0:
                if topic:
                    meta_results = self.storage.search_meta_memories(
                        query=topic,
                        project_name=project_name,
                        top_k=3
                    )
                    meta_memories_included = [
                        {
                            "id": str(r.meta_memory.id),
                            "pattern": r.meta_memory.pattern_summary,
                            "lessons": r.meta_memory.lessons[:3],
                            "best_practices": r.meta_memory.best_practices[:2],
                            "technologies": r.meta_memory.technologies,
                            "episode_count": r.meta_memory.episode_count,
                            "confidence": f"{r.meta_memory.confidence:.0%}",
                            "relevance": f"{r.relevance_score:.0%}"
                        }
                        for r in meta_results
                    ]
                else:
                    recent_metas = self.storage.get_meta_memories_by_project(
                        project_name=project_name or "default",
                        limit=3
                    )
                    meta_memories_included = [
                        {
                            "id": str(mm.id),
                            "pattern": mm.pattern_summary,
                            "lessons": mm.lessons[:3],
                            "best_practices": mm.best_practices[:2],
                            "technologies": mm.technologies,
                            "episode_count": mm.episode_count,
                            "confidence": f"{mm.confidence:.0%}"
                        }
                        for mm in recent_metas
                    ]

            if meta_memories_included:
                result["meta_memories"] = meta_memories_included
                result["meta_memory_note"] = (
                    "META-MEMORIES: Consolidated knowledge from multiple episodes."
                )

            # =================================================================
            # CHECK CONSOLIDATION NEED
            # =================================================================
            consolidation_check = self.storage.check_consolidation_needed(project_name)
            if consolidation_check.get("should_consolidate"):
                result["consolidation_recommendation"] = {
                    "should_consolidate": True,
                    "reason": f"There are {consolidation_check['hot_episodes_count']} high-usage episodes "
                             f"or {consolidation_check['estimated_unconsolidated']} unconsolidated",
                    "suggestion": "Consider running consolidation with: mt consolidate --project <name>"
                }

            # =================================================================
            # PRIORITY 2: INDIVIDUAL EPISODES
            # =================================================================
            if total_episodes < threshold:
                result["mode"] = "full_context"
                result["message"] = f"Small memory ({total_episodes} episodes) - showing full context."

                timeline = self.rag_engine.get_timeline(
                    limit=total_episodes,
                    project_name=project_name
                )

                episodes_summary = []
                for ep in timeline:
                    episode_brief = {
                        "id": ep["id"],
                        "type": ep["type"],
                        "task": ep["task"],
                        "summary": ep["summary"],
                        "date": ep["date"],
                        "tags": ep["tags"]
                    }
                    episodes_summary.append(episode_brief)

                result["episodes"] = episodes_summary

                if topic:
                    lessons = self.rag_engine.get_lessons(project_name=project_name)
                    lessons_list = lessons if isinstance(lessons, list) else lessons.get("lessons", [])
                    result["aggregated_lessons"] = _format_lessons(lessons_list)

            else:
                result["mode"] = "smart_context"
                result["message"] = f"Mature memory ({total_episodes} episodes) - showing optimized context."

                recent = self.rag_engine.get_timeline(limit=5, project_name=project_name)
                result["recent_episodes"] = [
                    {
                        "id": ep["id"],
                        "type": ep["type"],
                        "task": ep["task"],
                        "summary": ep["summary"],
                        "date": ep["date"],
                        "tags": ep["tags"]
                    }
                    for ep in recent
                ]

                if topic:
                    query = MemoryQuery(
                        query=topic,
                        project_filter=project_name,
                        top_k=5
                    )
                    relevant_results = self.storage.search_episodes(query)
                    relevant_episodes = []
                    for r in relevant_results:
                        ep_data = {
                            "id": str(r.episode.id),
                            "type": r.episode.episode_type.value,
                            "task": r.episode.task,
                            "summary": r.episode.solution_summary,
                            "relevance": f"{r.relevance_score:.0%}",
                            "tags": r.episode.tags,
                            "lessons": r.episode.lessons_learned,
                            "is_critical": getattr(r.episode, 'is_critical', False)
                        }
                        if include_reasoning:
                            ep_data["reasoning"] = r.episode.reasoning_trace.raw_thinking
                            ep_data["alternatives"] = r.episode.reasoning_trace.alternatives_considered
                            ep_data["decision_factors"] = r.episode.reasoning_trace.decision_factors
                        relevant_episodes.append(ep_data)
                    result["relevant_episodes"] = relevant_episodes

                    lessons = self.rag_engine.get_lessons(project_name=project_name)
                    lessons_list = lessons if isinstance(lessons, list) else lessons.get("lessons", [])
                    result["aggregated_lessons"] = _format_lessons(lessons_list)
                else:
                    result["tip"] = "Provide a 'topic' to get semantically relevant episodes."

            # Save output for the span
            output_data = {
                "mode": result.get("mode"),
                "episodes_count": total_episodes,
                "meta_memories_count": meta_stats.get("total_meta_memories", 0),
                "warnings_count": len(warnings),
                "relevant_found": len(result.get("relevant_episodes", [])),
                "meta_memories_found": len(meta_memories_included)
            }

            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, ensure_ascii=False)
                )]
            )

        finally:
            # Close span with output and flush
            if span_ctx:
                try:
                    from langfuse.decorators import langfuse_context
                    langfuse_context.update_current_observation(output=output_data)
                    span_ctx.__exit__(None, None, None)
                except Exception:
                    pass
            if langfuse:
                flush_traces()

    async def _consolidate_memories(self, args: dict) -> CallToolResult:
        """
        Consolidate similar episodes into meta-memories.

        Uses DBSCAN clustering + LLM to synthesize knowledge.
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        from memorytwin.consolidation import MemoryConsolidator

        project_name = args.get("project_name")
        if not project_name:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="Error: project_name is required to consolidate memories"
                )],
                isError=True
            )

        min_cluster_size = args.get("min_cluster_size", 3)

        try:
            # Check that there are enough episodes
            stats = self.storage.get_statistics(project_name)
            total_episodes = stats['total_episodes']

            if total_episodes < min_cluster_size:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=json.dumps({
                            "success": False,
                            "message": f"Project '{project_name}' only has {total_episodes} episodes. "
                                      f"At least {min_cluster_size} are needed to consolidate.",
                            "total_episodes": total_episodes,
                            "min_required": min_cluster_size
                        }, indent=2, ensure_ascii=False)
                    )]
                )

            # Run consolidation in thread pool to avoid blocking the event loop
            consolidator = MemoryConsolidator(
                storage=self.storage,
                min_cluster_size=min_cluster_size
            )

            # Use ThreadPoolExecutor for synchronous LLM operations
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                try:
                    # Timeout of 120 seconds (may take time with several clusters)
                    meta_memories = await asyncio.wait_for(
                        loop.run_in_executor(
                            executor,
                            consolidator.consolidate_project,
                            project_name
                        ),
                        timeout=120.0
                    )
                except asyncio.TimeoutError:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=json.dumps({
                                "success": False,
                                "message": "Consolidation exceeded the time limit (120s). "
                                          "This may happen with many episodes or a slow LLM connection.",
                                "suggestion": "Try a larger min_cluster_size to reduce clusters"
                            }, indent=2, ensure_ascii=False)
                        )],
                        isError=True
                    )

            if not meta_memories:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=json.dumps({
                            "success": False,
                            "message": "No clusters large enough to consolidate were found. "
                                      "Try a smaller min_cluster_size.",
                            "total_episodes": total_episodes,
                            "min_cluster_size": min_cluster_size,
                            "suggestion": "Episodes may be too semantically diverse"
                        }, indent=2, ensure_ascii=False)
                    )]
                )

            # Summary of generated meta-memories
            result = {
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
            }

            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, ensure_ascii=False)
                )]
            )

        except Exception as e:
            logger.error(f"Error in consolidation: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error during consolidation: {str(e)}"
                )],
                isError=True
            )

    async def _mark_episode(self, args: dict) -> CallToolResult:
        """
        Mark an episode with special flags (antipattern, critical, superseded, deprecated).
        """
        episode_id = args.get("episode_id")
        is_antipattern = args.get("is_antipattern")
        is_critical = args.get("is_critical")
        superseded_by = args.get("superseded_by")
        deprecation_reason = args.get("deprecation_reason")

        if not episode_id:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="Error: episode_id is required"
                )],
                isError=True
            )

        # Get current episode
        episode = self.storage.get_episode_by_id(episode_id)
        if not episode:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error: Episode not found with ID {episode_id}"
                )],
                isError=True
            )

        # Update flags
        updates = {}
        if is_antipattern is not None:
            updates["is_antipattern"] = is_antipattern
        if is_critical is not None:
            updates["is_critical"] = is_critical
        if superseded_by is not None:
            updates["superseded_by"] = superseded_by
        if deprecation_reason is not None:
            updates["deprecation_reason"] = deprecation_reason

        if not updates:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="No changes specified (is_antipattern or is_critical)"
                )],
                isError=True
            )

        # Apply updates
        success = self.storage.update_episode_flags(episode_id, updates)

        if success:
            result = {
                "success": True,
                "episode_id": episode_id,
                "task": episode.task,
                "updates_applied": updates,
                "message": ""
            }
            if updates.get("is_antipattern"):
                result["message"] = (
                    "Episode marked as ANTIPATTERN. "
                    "It will be shown as a warning in future relevant queries."
                )
            if updates.get("is_critical"):
                result["message"] += (
                    "Episode marked as CRITICAL. "
                    "It will be prioritized in searches."
                )
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, ensure_ascii=False)
                )]
            )
        else:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error updating episode {episode_id}"
                )],
                isError=True
            )

    async def _check_consolidation_status(self, args: dict) -> CallToolResult:
        """
        Check if consolidation is needed.

        Analyzes episode access_count and number of unconsolidated episodes.
        """
        project_name = args.get("project_name")

        status = self.storage.check_consolidation_needed(project_name)

        # Add readable recommendation
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

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(status, indent=2, ensure_ascii=False)
            )]
        )

    async def run(self):
        """Run the MCP server."""
        logger.info("Starting Memory Twin MCP Server...")
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def _async_main():
    """Async entry point for the MCP server."""
    server = MemoryTwinMCPServer()
    await server.run()


def main():
    """Synchronous entry point for console scripts."""
    import asyncio
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
