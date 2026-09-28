"""
MCP Tool Definitions
====================

JSON-schema definitions of every tool Memory Twin exposes over the
Model Context Protocol. Kept apart from the server so the contract
with MCP clients can be reviewed in one place.
"""

from mcp.types import Tool

TOOLS: list[Tool] = [
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
                },
                "force": {
                    "type": "boolean",
                    "description": (
                        "Re-consolidate episodes already covered by a meta-memory "
                        "(default: false, only new episodes are clustered)"
                    )
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
    ),
]
