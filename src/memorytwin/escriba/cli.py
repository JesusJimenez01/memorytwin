"""
Escriba CLI - Command Line Interface
=====================================
"""

import argparse
import logging
import sys
import warnings

# Silence noisy warnings BEFORE any imports
logging.getLogger("langfuse").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*ended span.*")

from rich.console import Console
from rich.panel import Panel

console = Console()


def handle_capture(args):
    """Handle capture command."""
    from memorytwin.escriba import Escriba

    escriba = Escriba(project_name=args.project)

    if args.file:
        episode = escriba.capture_from_file(
            args.file,
            source_assistant=args.assistant,
            project_name=args.project
        )
    elif args.clipboard:
        episode = escriba.capture_from_clipboard(
            source_assistant=args.assistant,
            project_name=args.project
        )
    else:
        console.print("[yellow]Paste thinking text (end with Ctrl+D or empty line):[/yellow]")
        lines = []
        try:
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
        except EOFError:
            pass

        if not lines:
            console.print("[red]No text provided.[/red]")
            return

        thinking_text = "\n".join(lines)
        episode = escriba.capture_thinking_sync(
            thinking_text,
            source_assistant=args.assistant,
            project_name=args.project
        )

    console.print(f"\n[green]✓ Episode saved: {episode.id}[/green]")


def handle_stats(args):
    """Handle statistics command."""
    from memorytwin.escriba import MemoryStorage

    storage = MemoryStorage()
    stats = storage.get_statistics(args.project)

    console.print(Panel(
        f"[bold]Total episodes:[/bold] {stats['total_episodes']}\n"
        f"[bold]In ChromaDB:[/bold] {stats['chroma_count']}\n\n"
        f"[bold]By type:[/bold]\n" +
        "\n".join(f"  • {k}: {v}" for k, v in stats['by_type'].items() if v > 0) +
        "\n\n[bold]By assistant:[/bold]\n" +
        "\n".join(f"  • {k}: {v}" for k, v in stats['by_assistant'].items()),
        title="📊 Memory Statistics",
        border_style="blue"
    ))


def handle_search(args):
    """Handle search command."""
    from memorytwin.escriba import MemoryStorage
    from memorytwin.models import MemoryQuery

    storage = MemoryStorage()

    query = MemoryQuery(
        query=args.query,
        project_filter=args.project,
        top_k=args.top
    )

    results = storage.search_episodes(query)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(f"\n[bold]🔍 {len(results)} results for:[/bold] {args.query}\n")

    for i, result in enumerate(results, 1):
        ep = result.episode
        console.print(Panel(
            f"[bold]Task:[/bold] {ep.task}\n"
            f"[bold]Summary:[/bold] {ep.solution_summary}\n"
            f"[bold]Type:[/bold] {ep.episode_type.value} | "
            f"[bold]Date:[/bold] {ep.timestamp.strftime('%Y-%m-%d %H:%M')}\n"
            f"[bold]Relevance:[/bold] {result.relevance_score:.2%}",
            title=f"Result {i}",
            border_style="cyan"
        ))


def handle_query(args):
    """Handle RAG query."""
    import asyncio

    from memorytwin.escriba import MemoryStorage
    from memorytwin.oraculo import RAGEngine

    storage = MemoryStorage()
    rag = RAGEngine(storage=storage)

    console.print(f"\n[bold cyan]🤔 Querying:[/bold cyan] {args.question}\n")

    result = asyncio.run(rag.query(
        question=args.question,
        project_name=args.project
    ))

    console.print(Panel(
        result["answer"],
        title="💡 Answer",
        border_style="green"
    ))

    if result.get("sources"):
        console.print("\n[dim]Sources consulted:[/dim]")
        for src in result["sources"][:3]:
            console.print(f"  • {src['task'][:60]}...")


def handle_lessons(args):
    """Handle lessons command."""
    from memorytwin.escriba import MemoryStorage
    from memorytwin.oraculo import RAGEngine

    storage = MemoryStorage()
    rag = RAGEngine(storage=storage)

    lessons = rag.get_lessons(project_name=args.project)

    if not lessons:
        console.print("[yellow]No lessons recorded yet.[/yellow]")
        return

    console.print(f"\n[bold]📚 {len(lessons)} lessons learned:[/bold]\n")

    for lesson in lessons:
        console.print(Panel(
            f"[bold]{lesson['lesson']}[/bold]\n\n"
            f"[dim]From: {lesson['from_task'][:60]}...[/dim]\n"
            f"[dim]Date: {lesson['timestamp'].strftime('%Y-%m-%d')} | Tags: {', '.join(lesson['tags'][:3])}[/dim]",
            border_style="yellow"
        ))


def handle_onboard(args):
    """Run onboarding for an existing project."""
    import asyncio
    from pathlib import Path

    from memorytwin.escriba.project_analyzer import onboard_project

    project_path = Path(args.path).resolve()

    if not project_path.exists():
        console.print(f"[red]Error: Directory does not exist: {project_path}[/red]")
        return

    console.print(Panel(
        f"[bold cyan]🔍 Analyzing project...[/bold cyan]\n"
        f"Path: {project_path}",
        title="Memory Twin - Onboarding",
        border_style="cyan"
    ))

    try:
        result = asyncio.run(onboard_project(
            project_path=str(project_path),
            project_name=args.project,
            source_assistant="onboarding-analyzer"
        ))

        analysis = result['analysis']

        # Show summary
        stack_list = ", ".join([s['technology'] for s in analysis['stack'][:5]]) or "Not detected"
        patterns_list = ", ".join([p.get('pattern', p.get('directory', '')) for p in analysis['patterns'][:3]]) or "Not detected"
        deps_list = ", ".join(analysis['dependencies']['main'][:8]) or "Not detected"

        console.print(Panel(
            f"[bold green]✓ Onboarding completed![/bold green]\n\n"
            f"[bold]Project:[/bold] {result['project_name']}\n"
            f"[bold]Episode:[/bold] {result['episode_id']}\n\n"
            f"[bold]Detected stack:[/bold]\n  {stack_list}\n\n"
            f"[bold]Patterns:[/bold]\n  {patterns_list}\n\n"
            f"[bold]Main dependencies:[/bold]\n  {deps_list}\n\n"
            f"[dim]Initial project memory has been created.\n"
            f"The agent now knows the structure and conventions.[/dim]",
            title="🧠 Analysis Completed",
            border_style="green"
        ))

        if args.verbose:
            console.print("\n[bold]Generated onboarding text:[/bold]")
            console.print(result['onboarding_text'])

    except Exception as e:
        console.print(f"[red]Error during onboarding: {e}[/red]")
        raise


def handle_health_check(args):
    """Verify Memory Twin system integrity."""
    from memorytwin.config import get_chroma_dir, get_sqlite_path
    from memorytwin.escriba import MemoryStorage

    console.print(Panel(
        "[bold cyan]🔍 Verifying system integrity...[/bold cyan]",
        title="Memory Twin - Health Check",
        border_style="cyan"
    ))

    issues = []
    warnings = []

    try:
        storage = MemoryStorage()
        stats = storage.get_statistics()

        sqlite_count = stats['total_episodes']
        chroma_count = stats['chroma_count']

        # Verify consistency between SQLite and ChromaDB
        if sqlite_count != chroma_count:
            issues.append(
                f"⚠️ Inconsistency: SQLite has {sqlite_count} episodes, "
                f"ChromaDB has {chroma_count}"
            )

        # Verify database files
        chroma_dir = get_chroma_dir()
        sqlite_path = get_sqlite_path()

        if not chroma_dir.exists():
            issues.append(f"❌ ChromaDB directory does not exist: {chroma_dir}")

        if not sqlite_path.exists():
            issues.append(f"❌ SQLite file does not exist: {sqlite_path}")

        # Verify database size
        if sqlite_path.exists():
            size_mb = sqlite_path.stat().st_size / (1024 * 1024)
            if size_mb > 100:
                warnings.append(f"📦 Large database: {size_mb:.1f} MB")

        # Results
        if issues:
            console.print("\n[bold red]❌ Problems found:[/bold red]")
            for issue in issues:
                console.print(f"  {issue}")

        if warnings:
            console.print("\n[bold yellow]⚠️ Warnings:[/bold yellow]")
            for warning in warnings:
                console.print(f"  {warning}")

        if not issues and not warnings:
            console.print(Panel(
                f"[bold green]✓ System healthy[/bold green]\n\n"
                f"[bold]Episodes in SQLite:[/bold] {sqlite_count}\n"
                f"[bold]Episodes in ChromaDB:[/bold] {chroma_count}\n"
                f"[bold]Synchronization:[/bold] ✓ OK",
                title="✅ Health Check Passed",
                border_style="green"
            ))
        else:
            console.print(Panel(
                f"[bold]Episodes in SQLite:[/bold] {sqlite_count}\n"
                f"[bold]Episodes in ChromaDB:[/bold] {chroma_count}\n\n"
                f"[dim]Use 'mt sync --repair' to fix inconsistencies[/dim]",
                title="📊 Current State",
                border_style="yellow"
            ))

    except Exception as e:
        console.print(f"[red]Error during health check: {e}[/red]")
        raise


def handle_consolidate(args):
    """Consolidate related episodes into meta-memories."""
    from memorytwin.consolidation import MemoryConsolidator
    from memorytwin.escriba import MemoryStorage

    console.print(Panel(
        f"[bold cyan]🧠 Consolidating project memories: {args.project}[/bold cyan]\n"
        f"Minimum episodes per cluster: {args.min_cluster}",
        title="Memory Twin - Consolidation",
        border_style="cyan"
    ))

    try:
        storage = MemoryStorage()

        # Verify there are enough episodes
        stats = storage.get_statistics(args.project)
        total_episodes = stats['total_episodes']

        if total_episodes < args.min_cluster:
            console.print(
                f"[yellow]⚠️ The project only has {total_episodes} episodes. "
                f"At least {args.min_cluster} are needed to consolidate.[/yellow]"
            )
            return

        console.print(f"[dim]Analyzing {total_episodes} episodes...[/dim]")

        # Run consolidation
        consolidator = MemoryConsolidator(
            storage=storage,
            min_cluster_size=args.min_cluster
        )

        meta_memories = consolidator.consolidate_project(args.project)

        if not meta_memories:
            console.print(
                "[yellow]No clusters large enough to consolidate were found. "
                "Try with a lower --min-cluster value.[/yellow]"
            )
            return

        # Show results
        console.print(Panel(
            f"[bold green]✓ Consolidation completed![/bold green]\n\n"
            f"[bold]Meta-memories generated:[/bold] {len(meta_memories)}\n"
            f"[bold]Episodes consolidated:[/bold] "
            f"{sum(m.episode_count for m in meta_memories)}",
            title="🧠 Result",
            border_style="green"
        ))

        if args.verbose:
            for i, meta in enumerate(meta_memories, 1):
                console.print(Panel(
                    f"[bold]Pattern:[/bold] {meta.pattern_summary}\n\n"
                    f"[bold]Lessons:[/bold]\n" +
                    "\n".join(f"  • {lesson}" for lesson in meta.lessons[:3]) + "\n\n"
                    "[bold]Best practices:[/bold]\n" +
                    "\n".join(f"  • {p}" for p in meta.best_practices[:2]) + "\n\n"
                    f"[dim]Episodes: {meta.episode_count} | "
                    f"Confidence: {meta.confidence:.0%} | "
                    f"Coherence: {meta.coherence_score:.0%}[/dim]",
                    title=f"Meta-Memory {i}",
                    border_style="magenta"
                ))

    except Exception as e:
        console.print(f"[red]Error during consolidation: {e}[/red]")
        raise


def handle_mcp(args):
    """Start the MCP server."""
    import asyncio

    from memorytwin.mcp_server.server import _async_main

    console.print("[cyan]Starting MCP server...[/cyan]")
    asyncio.run(_async_main())


def handle_setup(args):
    """Configure Memory Twin in a project."""
    import json
    import shutil
    from pathlib import Path

    project_path = Path(args.path).resolve()

    if not project_path.exists():
        console.print(f"[red]Error: Directory does not exist: {project_path}[/red]")
        return

    console.print(Panel(
        f"[bold cyan]🔧 Configuring Memory Twin...[/bold cyan]\n"
        f"Project: {project_path}",
        title="Memory Twin - Setup",
        border_style="cyan"
    ))

    # Detect the best way to invoke the MCP server
    # Priority order: mt in PATH (absolute) > uvx > python -m
    mt_path = shutil.which("mt")
    uv_available = shutil.which("uv") is not None

    if mt_path:
        # Option 1: mt mcp (using absolute path for robustness)
        mcp_command = mt_path
        mcp_args = ["mcp"]
        install_method = "mt"
    elif uv_available:
        # Option 2: uvx (auto-downloads if not installed)
        mcp_command = "uvx"
        mcp_args = ["--from", "memorytwin", "mt", "mcp"]
        install_method = "uvx"
    else:
        # Option 3: Use current environment's Python
        python_exe = sys.executable
        mcp_command = python_exe
        mcp_args = ["-m", "memorytwin.escriba.cli", "mcp"]
        install_method = "python"

    # Copilot instructions content
    instructions_content = '''# Memory Twin - Instructions for AI Agents

---
## 🚨 MANDATORY RULE - READ FIRST 🚨

**BEFORE answering ANY technical question, I MUST run:**

```
get_project_context(topic="<relevant terms from the user's question>")
```

**NO EXCEPTIONS.** If I don't check memory first, I may:
- Suggest solutions that already failed
- Ignore previous architecture decisions
- Repeat errors we already solved
- Contradict the project's style/conventions

**My first action is ALWAYS to get project context. ALWAYS.**

## 🛠️ Tool Availability
These instructions assume the MemoryTwin MCP server is active.
- If the `get_project_context` tool fails or is not available:
1. NOTIFY ME immediately: "⚠️ I cannot access my memory. Check the MCP server."
2. Ask me to paste the context manually if it's critical.

---

## What is Memory Twin?
Episodic memory system that captures technical reasoning to prevent "technical amnesia" in projects.

## ⚡ MANDATORY QUERY PROTOCOL

### STEP 1: Get context (MANDATORY)
Before writing ANY technical response:

```
get_project_context(topic="<keywords from the question>")
```

This tool is SMART:
- If there are **few memories (<20)**: returns ALL project context
- If there are **many memories (>=20)**: returns statistics + recent + relevant to the topic

**Important Note:**
This tool activates the **"Forgetting Curve"** mechanism. By querying relevant memories, you increase their `access_count`, preventing the system from "forgetting" them over time. Querying is reinforcing!

Query examples:
- User asks about authentication → `get_project_context(topic="authentication login JWT")`
- User has an error → `get_project_context(topic="error <error type>")`
- User wants to implement a feature → `get_project_context(topic="<feature name>")`
- User asks about architecture → `get_project_context(topic="architecture design")`

### STEP 2: Analyze results
- If there are relevant memories → Incorporate that knowledge
- If no results → Proceed normally, but consider capturing afterwards

### STEP 3: Respond mentioning context
- "According to previous project memories..."
- "I found no previous decisions about this, proceeding with..."
- "This was already solved before, the solution was..."

### When to query (ALWAYS when applicable):
| Situation | Mandatory query |
|-----------|---------------------|
| Technical question | `get_project_context(topic="<topic>")` |
| Error/Bug | `get_project_context(topic="error <description>")` |
| New feature | `get_project_context(topic="<feature>")` + `get_lessons()` |
| Architecture decision | `query_memory("<question>")` |
| First time in project | `onboard_project("<path>")` |
| Choose library/approach | `get_project_context(topic="<options>")` |

## Available MCP Tools

### `get_project_context` - ⭐ MAIN TOOL
**ALWAYS use before any technical response.**

Smart behavior:
- **Few memories (<20)**: Returns ALL complete context
- **Many memories (>=20)**: Returns statistics + 5 recent + 5 relevant to the topic

**Important Note:**
This tool activates the **"Forgetting Curve"** mechanism. By querying relevant memories, you increase their `access_count`, preventing the system from "forgetting" them over time. Querying is reinforcing!

Parameters:
- `topic` (optional): Topic for semantic search
- `project_name` (optional): Filter by project

### `capture_thinking` - 🔴 MANDATORY CAPTURE
**I MUST use this tool AUTOMATICALLY after any technical task.**

**IMPORTANT:** This is the **ONLY** way to persist knowledge. If I don't use it, the work is lost.

**💡 TIP:** There are 3 ways to capture, choose the most convenient:
1. `capture_quick` - ⚡ The fastest (only what + why)
2. `capture_decision` - 🎯 For decisions (task + decision + reasoning)
3. `capture_thinking` - 📝 For extensive free text

#### ✅ ALWAYS CAPTURE (no exceptions):
- Fixed a bug or error (any, no matter if it's "simple")
- Made a technical decision (library, pattern, approach)
- Compared alternatives before choosing
- Discovered something unexpected (gotcha, edge case, weird behavior)
- Modified existing code (refactor, improvement, fix)
- Implemented a new feature
- Configured something (environment, tools, dependencies)
- Researched documentation or code to understand something
- The user asked me to do something and I completed it
- Found a problem in documentation/code and corrected it

#### ❌ Do NOT capture ONLY when:
- Purely informational response without action (e.g., "what time is it?")
- Casual conversation without technical content
- The user explicitly says "don't save this"

#### 🎯 GOLDEN RULE: When in doubt, CAPTURE
It's better to have an "extra" memory than to lose valuable knowledge.

Parameters:
- `thinking_text` (required): Model reasoning text
- `user_prompt` (optional): Original user prompt
- `code_changes` (optional): Associated code changes
- `source_assistant` (optional): copilot, claude, cursor, etc.
- `project_name` (optional): Project name

### `capture_decision` - 🎯 STRUCTURED CAPTURE (PREFERRED)
**Most convenient way to capture technical decisions.**

Use when you have data organized in separate fields. More convenient than writing free text.

Parameters:
- `task` (required): Brief description of the task or problem
- `decision` (required): The decision or solution taken
- `reasoning` (required): Why this decision was made
- `alternatives` (optional): Array of alternatives considered
- `lesson` (optional): Lesson learned for the future
- `context` (optional): Additional context
- `project_name` (optional): Project name

**Ejemplo:**
```
capture_decision(
    task="Choose database",
    decision="PostgreSQL",
    alternatives=["MongoDB", "MySQL"],
    reasoning="We need ACID and complex queries",
    lesson="For relational data with transactions, SQL > NoSQL"
)
```

### `capture_quick` - ⚡ QUICK CAPTURE (MINIMUM EFFORT)
**The simplest way to capture. Only 2 required fields.**

Use for quick captures without much detail. Ideal when you're in a hurry.

Parameters:
- `what` (required): What did you do? (action performed)
- `why` (required): Why did you do it? (reason)
- `lesson` (optional but recommended): Lesson learned
- `project_name` (optional): Project name

**Examples:**
```
capture_quick(
    what="Added retry logic to HTTP client",
    why="API calls were failing intermittently"
)

capture_quick(
    what="Switched from axios to fetch",
    why="Reduce dependencies, native fetch is sufficient",
    lesson="Always evaluate if a dependency is really necessary"
)
```

### `query_memory` - Query memories with RAG
Use when:
- The user asks "why did we do X?"
- The user asks "how did we solve something similar?"
- Before making an important decision (check precedents)

Parameters:
- `question` (required): Question to answer
- `project_name` (optional): Filter by project
- `num_episodes` (optional): Number of episodes to query (1-10, default: 5)

### `search_episodes` - Semantic episode search
Use for specific searches on topics or technologies.
Returns the most relevant episodes for a search term.
*Note: Queried results receive a relevance boost for the future.*

Parameters:
- `query` (required): Search term
- `project_name` (optional): Filter by project
- `top_k` (optional): Number of results (default: 5)

### `get_episode` - Get complete episode
Use when you need to dive into the details of a specific decision.
Returns the COMPLETE content: thinking, alternatives, decision factors, context and lessons.

Parameters:
- `episode_id` (required): UUID of the episode to retrieve

### `get_lessons` - Lessons learned
Use for:
- Onboarding new members
- Review before starting a similar feature
- The user asks "what have we learned about X?"

Parameters:
- `project_name` (optional): Filter by project
- `tags` (optional): Array of tags to filter

### `get_timeline` - View chronological history
Use to see chronological evolution of the project and understand what was done when.

Parameters:
- `project_name` (optional): Filter by project
- `limit` (optional): Maximum episodes to return (default: 20)

### `get_statistics` - Memory statistics
Gets memory database statistics: total episodes, distribution by type and assistant.

Parameters:
- `project_name` (optional): Filter by project

### `onboard_project` - Onboard existing project
Use when:
- ✅ It's the first time I work on this project
- ✅ The user asks "analyze the project", "learn the code"
- ✅ I need to understand the structure before making big changes
- ✅ There are no previous memories and I want to create initial context

Automatically generates an episode with:
- Detected technology stack
- Architectural patterns
- Main dependencies
- Code conventions

Parameters:
- `project_path` (required): Absolute path to the project
- `project_name` (optional): Project name (auto-detected)

## MANDATORY Workflow

### 🔄 COMPLETE CYCLE (ALWAYS):
```
1. START: get_project_context(topic="...")
2. WORK: Perform the requested task
3. END: capture_thinking(thinking_text="...")
```

### Checklist before finishing response:
- [ ] Did I check memory at the start? If not → DO IT NOW
- [ ] Did I do something technical? If yes → CAPTURE MEMORY
- [ ] Did I solve a problem? If yes → CAPTURE MEMORY
- [ ] Did I make a decision? If yes → CAPTURE MEMORY
- [ ] Did I modify code? If yes → CAPTURE MEMORY
- [ ] Did I research something? If yes → CAPTURE MEMORY

### ⚠️ CRITICAL REMINDER:
**Do NOT finish a technical response without having run `capture_thinking`.**
Knowledge that is not captured is LOST FOREVER.

## Thinking Format to Capture

Always include when possible:
- **Task**: What was being solved
- **Context**: Initial state, constraints
- **Alternatives**: Options considered
- **Decision**: What was chosen and why
- **Solution**: How it was implemented
- **Lessons**: What we learned

## Automatic Capture Example

```
When I detect that I just:
1. Debugged an error for more than 2 exchanges
2. Compared 2+ options before choosing
3. Discovered unexpected behavior
4. Implemented something that required research

→ Call capture_thinking with the full context
```

## 🚨 CORRECT FLOW EXAMPLE

### User asks: "Why is my login function failing?"

```
# 1. FIRST: Query memory
get_project_context(topic="login authentication error")

# 2. THEN: Work on the solution
[Analyze code, debug, find the problem, propose fix]

# 3. FINALLY: Capture the knowledge
capture_thinking(
    thinking_text="## Task\\nResolve login function error...\\n## Problem\\nThe JWT token...\\n## Solution\\n...\\n## Lessons\\n...",
    project_name="my-project",
    source_assistant="copilot"
)
```

**IF I DON'T CAPTURE AT THE END, I'M FAILING MY PURPOSE.**

## Current Project
- **Project name**: Use the workspace folder name
- **Source assistant**: "copilot" for GitHub Copilot
'''

    # MCP configuration for VS Code - use detected command
    mcp_config = {
        "servers": {
            "memorytwin": {
                "command": mcp_command,
                "args": mcp_args,
                "type": "stdio"
            }
        }
    }

    files_created = []
    files_updated = []

    # Create .vscode directory if it doesn't exist
    vscode_dir = project_path / ".vscode"
    vscode_dir.mkdir(parents=True, exist_ok=True)

    # Write standard instructions file
    instructions_path = project_path / "AGENTS.md"
    instructions_path.write_text(instructions_content, encoding="utf-8")
    files_created.append(str(instructions_path.relative_to(project_path)))

    # Write mcp.json
    mcp_path = vscode_dir / "mcp.json"
    mcp_path.write_text(json.dumps(mcp_config, indent=2), encoding="utf-8")
    files_created.append(str(mcp_path.relative_to(project_path)))

    # Create .env if it doesn't exist
    env_path = project_path / ".env"
    if not env_path.exists():
        env_content = """# Memory Twin - Configuration
# ============================
# Copy this file to .env and fill in your API keys

# =============================================================================
# LLM PROVIDER - Choose ONE of the options below
# =============================================================================

# Option A: OpenRouter (recommended - access to many free models)
# Get your key at: https://openrouter.ai/keys
OPENROUTER_API_KEY=your_openrouter_key_here
LLM_PROVIDER=openrouter
LLM_MODEL=amazon/nova-2-lite-v1:free

# Free models on OpenRouter (updated Dec 2025):
# - amazon/nova-2-lite-v1:free (1M context, fast)
# - qwen/qwen3-coder:free (262K context, great for code)
# - tngtech/deepseek-r1t-chimera:free (164K context, reasoning)

# Option B: Google Gemini
# Get your key at: https://aistudio.google.com/apikey
# GOOGLE_API_KEY=your_google_api_key_here
# LLM_PROVIDER=google
# LLM_MODEL=gemini-2.0-flash

# =============================================================================
# OPTIONAL SETTINGS
# =============================================================================

# LLM Temperature (0.0 = deterministic, 1.0 = creative)
# LLM_TEMPERATURE=0.3

# Observability (Langfuse)
# LANGFUSE_PUBLIC_KEY=pk-lf-...
# LANGFUSE_SECRET_KEY=sk-lf-...
# LANGFUSE_HOST=https://cloud.langfuse.com

# Gradio UI
# GRADIO_SERVER_PORT=7860
# GRADIO_SHARE=false

# Storage Configuration (defaults are fine for local use)
# CHROMA_PERSIST_DIR=./data/chroma
# SQLITE_DB_PATH=./data/memory.db
"""
        env_path.write_text(env_content, encoding="utf-8")
        files_created.append(".env")

    # Ensure .env and data/ are in .gitignore
    gitignore_path = project_path / ".gitignore"
    if gitignore_path.exists():
        gitignore_content = gitignore_path.read_text(encoding="utf-8")
        new_entries = []

        if ".env" not in gitignore_content:
            new_entries.append(".env")

        if "data/" not in gitignore_content:
            new_entries.append("data/")

        if new_entries:
            with open(gitignore_path, "a", encoding="utf-8") as f:
                f.write("\n# Memory Twin\n")
                for entry in new_entries:
                    f.write(f"{entry}\n")
            files_updated.append(".gitignore")
    else:
        gitignore_path.write_text("# Memory Twin\n.env\ndata/\n", encoding="utf-8")
        files_created.append(".gitignore")

    # Show summary
    files_list = "\n".join(f"  • [cyan]{f}[/cyan]" for f in files_created)
    updated_list = "\n".join(f"  • [yellow]{f}[/yellow]" for f in files_updated) if files_updated else ""

    # Message based on installation method
    if install_method == "mt":
        mcp_info = "[green]mt mcp[/green] (simple and universal)"
        portability_note = ""
    elif install_method == "uvx":
        mcp_info = "[green]uvx[/green] (universal)"
        portability_note = ""
    else:
        mcp_info = "[yellow]Environment-specific Python[/yellow]"
        portability_note = """
[yellow]⚠️ Note:[/yellow] The configuration uses your current Python path.
   For a more portable setup, make sure 'mt' is in your PATH.
"""

    next_steps = """
[bold]Next steps:[/bold]
  1. Edit [cyan].env[/cyan] and configure your LLM provider
  2. Restart VS Code (F1 → "Developer: Reload Window")
  3. Check MCP: List Servers to verify memorytwin is connected
  4. Done! Copilot will use Memory Twin automatically
"""

    result_text = "[bold green]✓ Memory Twin configured![/bold green]\n\n"
    result_text += f"[bold]Files created:[/bold]\n{files_list}\n"
    if updated_list:
        result_text += f"\n[bold]Files updated:[/bold]\n{updated_list}\n"
    result_text += f"\n[bold]MCP Server:[/bold] {mcp_info}\n"
    if portability_note:
        result_text += portability_note
    result_text += next_steps

    console.print(Panel(
        result_text,
        title="🧠 Setup Complete",
        border_style="green"
    ))


def handle_oraculo(args):
    """Launch web interface (Oráculo)."""
    from memorytwin.oraculo.app import main as launch_oraculo
    launch_oraculo()


def main():
    """Escriba CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Memory Twin - Escriba: Technical memory capture"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: capture
    capture_parser = subparsers.add_parser(
        "capture",
        help="Capture thinking from file or clipboard"
    )
    capture_parser.add_argument(
        "--file", "-f",
        help="File containing the thinking text"
    )
    capture_parser.add_argument(
        "--clipboard", "-c",
        action="store_true",
        help="Capture from clipboard"
    )
    capture_parser.add_argument(
        "--assistant", "-a",
        default="unknown",
        help="Source assistant (copilot, claude, cursor)"
    )
    capture_parser.add_argument(
        "--project", "-p",
        default="default",
        help="Project name"
    )

    # Command: stats
    stats_parser = subparsers.add_parser(
        "stats",
        help="View memory statistics"
    )
    stats_parser.add_argument(
        "--project", "-p",
        help="Filter by project"
    )

    # Command: search
    search_parser = subparsers.add_parser(
        "search",
        help="Search the memory"
    )
    search_parser.add_argument(
        "query",
        help="Search text"
    )
    search_parser.add_argument(
        "--top", "-k",
        type=int,
        default=5,
        help="Number of results"
    )
    search_parser.add_argument(
        "--project", "-p",
        help="Filter by project"
    )

    # Command: query (RAG)
    query_parser = subparsers.add_parser(
        "query",
        help="Query with RAG (generated answer)"
    )
    query_parser.add_argument(
        "question",
        help="Question to answer"
    )
    query_parser.add_argument(
        "--project", "-p",
        help="Filter by project"
    )

    # Command: lessons
    lessons_parser = subparsers.add_parser(
        "lessons",
        help="View lessons learned"
    )
    lessons_parser.add_argument(
        "--project", "-p",
        help="Filter by project"
    )

    # Command: setup
    setup_parser = subparsers.add_parser(
        "setup",
        help="Configure Memory Twin in a project (creates AGENTS.md)"
    )
    setup_parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to the project (default: current directory)"
    )

    # Command: onboard
    onboard_parser = subparsers.add_parser(
        "onboard",
        help="Analyze existing project and create initial onboarding memory"
    )
    onboard_parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to the project to analyze (default: current directory)"
    )
    onboard_parser.add_argument(
        "--project", "-p",
        help="Project name (auto-detected if not specified)"
    )
    onboard_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full analysis text"
    )

    # Command: health-check
    subparsers.add_parser(
        "health-check",
        help="Verify system integrity (SQLite + ChromaDB)"
    )

    # Command: consolidate
    consolidate_parser = subparsers.add_parser(
        "consolidate",
        help="Consolidate related episodes into meta-memories"
    )

    # Command: mcp (start MCP server)
    subparsers.add_parser(
        "mcp",
        help="Start the MCP server for VS Code/Copilot"
    )

    # Command: oraculo (web interface)
    subparsers.add_parser(
        "oraculo",
        help="Open web interface to manage memories"
    )

    consolidate_parser.add_argument(
        "--project", "-p",
        required=True,
        help="Project name to consolidate"
    )
    consolidate_parser.add_argument(
        "--min-cluster", "-m",
        type=int,
        default=3,
        help="Minimum episodes to form a cluster (default: 3)"
    )
    consolidate_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show details of each generated meta-memory"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        match args.command:
            case "capture":
                handle_capture(args)
            case "stats":
                handle_stats(args)
            case "search":
                handle_search(args)
            case "query":
                handle_query(args)
            case "lessons":
                handle_lessons(args)
            case "setup":
                handle_setup(args)
            case "onboard":
                handle_onboard(args)
            case "health-check":
                handle_health_check(args)
            case "consolidate":
                handle_consolidate(args)
            case "mcp":
                handle_mcp(args)
            case "oraculo":
                handle_oraculo(args)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
