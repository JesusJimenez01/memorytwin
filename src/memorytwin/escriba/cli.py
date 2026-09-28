"""
Memory Twin CLI (`mt`)
======================

Command-line entry point for capturing, querying and curating memories,
setting up projects and launching the MCP server or the web UI.
"""

import argparse
import asyncio
import json
import logging
import shutil
import sys
import warnings
from importlib import resources
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Heavy modules (torch, chromadb, gradio...) are imported inside each handler
# so that `mt --help` and simple commands start instantly.

console = Console()


def handle_capture(args):
    """Handle capture command (text argument, --file or stdin)."""
    from memorytwin.escriba import Escriba

    if args.text:
        thinking_text = args.text
    elif args.file:
        thinking_text = Path(args.file).read_text(encoding="utf-8")
    else:
        if sys.stdin.isatty():
            console.print("[yellow]Paste the thinking text and finish with Ctrl+D (Ctrl+Z, Enter on Windows):[/yellow]")
        thinking_text = sys.stdin.read()

    if not thinking_text.strip():
        console.print("[red]No text provided.[/red]")
        return

    escriba = Escriba(project_name=args.project)
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
        Markdown(result["answer"]),
        title="💡 Answer",
        border_style="green"
    ))

    if result.get("episodes_used"):
        console.print("\n[dim]Sources consulted:[/dim]")
        for episode in result["episodes_used"][:3]:
            console.print(f"  • [dim]{episode.id}[/dim] {episode.task[:70]}")


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

    result = asyncio.run(onboard_project(
        project_path=str(project_path),
        project_name=args.project,
        source_assistant="onboarding-analyzer"
    ))

    analysis = result['analysis']

    # Show summary
    stack_list = ", ".join(s['technology'] for s in analysis['stack'][:5]) or "Not detected"
    patterns_list = ", ".join(
        p.get('pattern', p.get('directory', '')) for p in analysis['patterns'][:3]
    ) or "Not detected"
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
    notes = []

    storage = MemoryStorage()
    stats = storage.get_statistics()

    sqlite_count = stats['total_episodes']
    chroma_count = stats['chroma_count']

    # SQLite (metadata) and ChromaDB (vectors) must describe the same episodes
    if sqlite_count != chroma_count:
        issues.append(
            f"⚠️ Inconsistency: SQLite has {sqlite_count} episodes, "
            f"ChromaDB has {chroma_count}"
        )

    chroma_dir = get_chroma_dir()
    sqlite_path = get_sqlite_path()

    if not chroma_dir.exists():
        issues.append(f"❌ ChromaDB directory does not exist: {chroma_dir}")

    if not sqlite_path.exists():
        issues.append(f"❌ SQLite file does not exist: {sqlite_path}")
    else:
        size_mb = sqlite_path.stat().st_size / (1024 * 1024)
        if size_mb > 100:
            notes.append(f"📦 Large database: {size_mb:.1f} MB")

    if issues:
        console.print("\n[bold red]❌ Problems found:[/bold red]")
        for issue in issues:
            console.print(f"  {issue}")

    if notes:
        console.print("\n[bold yellow]⚠️ Warnings:[/bold yellow]")
        for note in notes:
            console.print(f"  {note}")

    if not issues and not notes:
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
            f"[dim]Inconsistent episodes can be removed from the web UI (mt oraculo → Management)[/dim]",
            title="📊 Current State",
            border_style="yellow"
        ))

    if issues:
        sys.exit(1)


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

    storage = MemoryStorage()

    # Verify there are enough episodes
    total_episodes = storage.get_statistics(args.project)['total_episodes']

    if total_episodes < args.min_cluster:
        console.print(
            f"[yellow]⚠️ The project only has {total_episodes} episodes. "
            f"At least {args.min_cluster} are needed to consolidate.[/yellow]"
        )
        return

    console.print(f"[dim]Analyzing {total_episodes} episodes...[/dim]")

    consolidator = MemoryConsolidator(
        storage=storage,
        min_cluster_size=args.min_cluster
    )
    meta_memories = consolidator.consolidate_project(args.project, force=args.force)

    if not meta_memories:
        console.print(
            "[yellow]No new clusters large enough to consolidate were found. "
            "Try a lower --min-cluster value, or --force to re-consolidate.[/yellow]"
        )
        return

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


def handle_mcp(args):
    """Start the MCP server (stdio transport)."""
    from memorytwin.mcp_server.server import main as run_mcp_server

    # stdout carries the MCP JSON-RPC stream: anything human-facing goes to stderr
    Console(stderr=True).print("[cyan]Starting Memory Twin MCP server (stdio)...[/cyan]")
    run_mcp_server()


def handle_chat(args):
    """Interactive Q&A session with the Oráculo in the terminal."""
    from memorytwin.oraculo import Oraculo

    Oraculo(project_name=args.project).interactive_mode()


REPO_URL = "https://github.com/JesusJimenez01/memorytwin"


def _read_template(name: str) -> str:
    """Read a text template shipped inside the package."""
    return resources.files("memorytwin").joinpath("templates", name).read_text(encoding="utf-8")


def _detect_mcp_command() -> tuple[str, list[str], str]:
    """
    Pick the most portable way to launch the MCP server.

    Priority: `mt` on PATH (absolute path) > uvx (installs on demand) > current Python.

    Returns:
        Tuple (command, args, install_method)
    """
    mt_path = shutil.which("mt")
    if mt_path:
        return mt_path, ["mcp"], "mt"
    if shutil.which("uv"):
        return "uvx", ["--from", f"git+{REPO_URL}", "mt", "mcp"], "uvx"
    return sys.executable, ["-m", "memorytwin.escriba.cli", "mcp"], "python"


def _merge_mcp_config(mcp_path: Path, command: str, args: list[str]) -> bool:
    """
    Register the memorytwin server in .vscode/mcp.json, keeping any other servers.

    Returns:
        True if the file already existed (and was updated), False if it was created.
    """
    existed = mcp_path.exists()
    config: dict = {}
    if existed:
        try:
            config = json.loads(mcp_path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError as e:
            raise ValueError(f"{mcp_path} is not valid JSON, fix it before running setup: {e}") from e

    config.setdefault("servers", {})["memorytwin"] = {
        "command": command,
        "args": args,
        "type": "stdio",
    }
    mcp_path.parent.mkdir(parents=True, exist_ok=True)
    mcp_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return existed


def _ensure_gitignore(project_path: Path) -> Optional[str]:
    """
    Make sure secrets and local memories are never committed.

    Returns:
        "created", "updated" or None if nothing had to change.
    """
    gitignore_path = project_path / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text("# Memory Twin\n.env\ndata/\n", encoding="utf-8")
        return "created"

    existing = {line.strip() for line in gitignore_path.read_text(encoding="utf-8").splitlines()}
    missing = [entry for entry in (".env", "data/") if entry not in existing]
    if not missing:
        return None

    with open(gitignore_path, "a", encoding="utf-8") as f:
        f.write("\n# Memory Twin\n" + "".join(f"{entry}\n" for entry in missing))
    return "updated"


def handle_setup(args):
    """Configure Memory Twin in a project (non-destructive unless --force)."""
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

    created, updated, skipped = [], [], []

    # 1. Agent instructions (never overwrite the user's own AGENTS.md by default)
    agents_path = project_path / "AGENTS.md"
    if agents_path.exists() and not args.force:
        skipped.append("AGENTS.md (already exists, use --force to overwrite)")
    else:
        (updated if agents_path.exists() else created).append("AGENTS.md")
        agents_path.write_text(_read_template("AGENTS.md"), encoding="utf-8")

    # 2. MCP server registration for VS Code (merged into any existing config)
    mcp_command, mcp_args, install_method = _detect_mcp_command()
    mcp_path = project_path / ".vscode" / "mcp.json"
    if _merge_mcp_config(mcp_path, mcp_command, mcp_args):
        updated.append(".vscode/mcp.json")
    else:
        created.append(".vscode/mcp.json")

    # 3. Local configuration (an existing .env is never touched)
    env_path = project_path / ".env"
    if env_path.exists():
        skipped.append(".env (already exists, add your LLM keys manually)")
    else:
        env_path.write_text(_read_template("env.example"), encoding="utf-8")
        created.append(".env")

    # 4. Keep secrets and memories out of git
    gitignore_status = _ensure_gitignore(project_path)
    if gitignore_status == "created":
        created.append(".gitignore")
    elif gitignore_status == "updated":
        updated.append(".gitignore")

    mcp_info = {
        "mt": "[green]mt mcp[/green] (simple and universal)",
        "uvx": "[green]uvx[/green] (installs Memory Twin on demand)",
        "python": "[yellow]Environment-specific Python[/yellow]",
    }[install_method]

    result_text = "[bold green]✓ Memory Twin configured![/bold green]\n"
    for label, color, items in (("Created", "cyan", created), ("Updated", "yellow", updated),
                                ("Skipped", "dim", skipped)):
        if items:
            result_text += f"\n[bold]{label}:[/bold]\n" + "\n".join(f"  • [{color}]{f}[/{color}]" for f in items) + "\n"
    result_text += f"\n[bold]MCP Server:[/bold] {mcp_info}\n"
    if install_method == "python":
        result_text += (
            "\n[yellow]⚠️ Note:[/yellow] The configuration uses your current Python path.\n"
            "   For a more portable setup, make sure 'mt' is in your PATH.\n"
        )
    result_text += """
[bold]Next steps:[/bold]
  1. Edit [cyan].env[/cyan] and configure your LLM provider
  2. Restart VS Code (F1 → "Developer: Reload Window")
  3. Run "MCP: List Servers" to verify memorytwin is connected
  4. Done! Your assistant will use Memory Twin automatically
"""

    console.print(Panel(result_text, title="🧠 Setup Complete", border_style="green"))


def handle_oraculo(args):
    """Launch web interface (Oráculo)."""
    try:
        from memorytwin.oraculo.app import main as launch_oraculo
    except ImportError:
        console.print('[red]The web UI needs extra dependencies: pip install "memorytwin[ui]"[/red]')
        sys.exit(1)
    launch_oraculo()


def build_parser() -> argparse.ArgumentParser:
    """Build the `mt` argument parser."""
    parser = argparse.ArgumentParser(
        prog="mt",
        description="Memory Twin - episodic memory for AI coding assistants",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # capture
    capture_parser = subparsers.add_parser(
        "capture",
        help="Capture reasoning from an argument, a file or stdin"
    )
    capture_parser.add_argument(
        "text",
        nargs="?",
        help="Reasoning text to capture (reads --file or stdin when omitted)"
    )
    capture_parser.add_argument("--file", "-f", help="File containing the thinking text")
    capture_parser.add_argument(
        "--assistant", "-a",
        default="unknown",
        help="Source assistant (copilot, claude, cursor)"
    )
    capture_parser.add_argument("--project", "-p", default="default", help="Project name")

    # stats
    stats_parser = subparsers.add_parser("stats", help="View memory statistics")
    stats_parser.add_argument("--project", "-p", help="Filter by project")

    # search
    search_parser = subparsers.add_parser("search", help="Semantic search over episodes")
    search_parser.add_argument("query", help="Search text")
    search_parser.add_argument("--top", "-k", type=int, default=5, help="Number of results")
    search_parser.add_argument("--project", "-p", help="Filter by project")

    # query (RAG)
    query_parser = subparsers.add_parser("query", help="Ask a question (RAG-generated answer)")
    query_parser.add_argument("question", help="Question to answer")
    query_parser.add_argument("--project", "-p", help="Filter by project")

    # chat
    chat_parser = subparsers.add_parser("chat", help="Interactive Q&A session with the Oráculo")
    chat_parser.add_argument("--project", "-p", help="Filter by project")

    # lessons
    lessons_parser = subparsers.add_parser("lessons", help="View lessons learned")
    lessons_parser.add_argument("--project", "-p", help="Filter by project")

    # setup
    setup_parser = subparsers.add_parser(
        "setup",
        help="Configure Memory Twin in a project (AGENTS.md, MCP config, .env)"
    )
    setup_parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to the project (default: current directory)"
    )
    setup_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing AGENTS.md"
    )

    # onboard
    onboard_parser = subparsers.add_parser(
        "onboard",
        help="Analyze an existing project and create an onboarding memory"
    )
    onboard_parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to the project to analyze (default: current directory)"
    )
    onboard_parser.add_argument("--project", "-p", help="Project name (auto-detected if not specified)")
    onboard_parser.add_argument("--verbose", "-v", action="store_true", help="Show full analysis text")

    # consolidate
    consolidate_parser = subparsers.add_parser(
        "consolidate",
        help="Consolidate related episodes into meta-memories"
    )
    consolidate_parser.add_argument("--project", "-p", required=True, help="Project name to consolidate")
    consolidate_parser.add_argument(
        "--min-cluster", "-m",
        type=int,
        default=3,
        help="Minimum episodes to form a cluster (default: 3)"
    )
    consolidate_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-consolidate episodes already covered by a meta-memory"
    )
    consolidate_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show details of each generated meta-memory"
    )

    # health-check
    subparsers.add_parser("health-check", help="Verify system integrity (SQLite + ChromaDB)")

    # mcp
    subparsers.add_parser("mcp", help="Start the MCP server (stdio) for VS Code, Cursor, Claude...")

    # oraculo
    subparsers.add_parser("oraculo", help="Open the web interface to explore and manage memories")

    return parser


HANDLERS = {
    "capture": handle_capture,
    "stats": handle_stats,
    "search": handle_search,
    "query": handle_query,
    "chat": handle_chat,
    "lessons": handle_lessons,
    "setup": handle_setup,
    "onboard": handle_onboard,
    "health-check": handle_health_check,
    "consolidate": handle_consolidate,
    "mcp": handle_mcp,
    "oraculo": handle_oraculo,
}


def main(argv: Optional[list[str]] = None):
    """`mt` entry point."""
    # Rich output uses emoji: make sure Windows consoles don't choke on them
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

    # Keep third-party noise out of the terminal
    logging.getLogger("langfuse").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message=".*ended span.*")

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return

    try:
        HANDLERS[args.command](args)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
