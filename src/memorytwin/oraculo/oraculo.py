"""
Oráculo - Main Query Agent
===========================

Coordinates knowledge retrieval and provides a conversational
interface over the episodic memories.
"""

from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from memorytwin.oraculo.rag_engine import RAGEngine

console = Console()


class Oraculo:
    """
    Oráculo Agent - Knowledge retrieval assistant.

    Answers questions about technical decisions, provides
    evolution timelines, and aggregates lessons learned.
    """

    def __init__(
        self,
        rag_engine: Optional[RAGEngine] = None,
        project_name: Optional[str] = None
    ):
        """
        Initialize the Oráculo.

        Args:
            rag_engine: RAG engine (creates one if not provided)
            project_name: Default project to filter by
        """
        self.rag_engine = rag_engine or RAGEngine()
        self.project_name = project_name

        console.print(Panel(
            f"[bold blue]✓ Oráculo initialized[/bold blue]\n"
            f"Project: {project_name or 'All'}",
            title="Memory Twin - Oráculo",
            border_style="blue"
        ))

    async def ask(self, question: str) -> str:
        """
        Ask the Oráculo a question.

        Args:
            question: Question about the project

        Returns:
            Answer based on memories
        """
        console.print("\n[cyan]🔮 Querying memories...[/cyan]")

        result = await self.rag_engine.query(
            question,
            project_name=self.project_name
        )

        answer = result["answer"]
        episodes_count = len(result["episodes_used"])

        if result["context_provided"]:
            console.print(
                f"[dim]Based on {episodes_count} memory episode(s)[/dim]"
            )

        return answer

    def ask_sync(self, question: str) -> str:
        """Synchronous version of ask."""
        import asyncio
        return asyncio.run(self.ask(question))

    def show_timeline(self, limit: int = 20):
        """
        Display decision timeline in console.

        Args:
            limit: Number of episodes to show
        """
        timeline = self.rag_engine.get_timeline(
            project_name=self.project_name,
            limit=limit
        )

        if not timeline:
            console.print("[yellow]No episodes recorded.[/yellow]")
            return

        console.print(f"\n[bold]📅 Decision Timeline ({len(timeline)} episodes)[/bold]\n")

        for item in timeline:
            icon = "✓" if item["success"] else "✗"
            color = "green" if item["success"] else "red"

            console.print(Panel(
                f"[bold]{item['task']}[/bold]\n"
                f"{item['summary']}\n"
                f"[dim]Tags: {', '.join(item['tags'][:5])}[/dim]",
                title=f"[{color}]{icon}[/{color}] {item['date']} {item['time']} - {item['type']}",
                border_style="dim"
            ))

    def show_lessons(self, tags: Optional[list[str]] = None):
        """
        Display lessons learned.

        Args:
            tags: Filter by specific tags
        """
        lessons = self.rag_engine.get_lessons(
            project_name=self.project_name,
            tags=tags
        )

        if not lessons:
            console.print("[yellow]No documented lessons.[/yellow]")
            return

        console.print(f"\n[bold]📚 Lessons Learned ({len(lessons)})[/bold]\n")

        for i, lesson in enumerate(lessons, 1):
            console.print(Panel(
                f"[bold yellow]{lesson['lesson']}[/bold yellow]\n\n"
                f"[dim]From: {lesson['from_task'][:80]}...[/dim]\n"
                f"[dim]Date: {lesson['timestamp'].strftime('%Y-%m-%d')}[/dim]\n"
                f"[dim]Tags: {', '.join(lesson['tags'][:5])}[/dim]",
                title=f"Lesson {i}",
                border_style="yellow"
            ))

    def show_statistics(self):
        """Display memory statistics."""
        stats = self.rag_engine.get_statistics(self.project_name)

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

    def interactive_mode(self):
        """
        Interactive query mode.
        Allows asking questions continuously.
        """
        console.print(Panel(
            "[bold]Oráculo Interactive Mode[/bold]\n\n"
            "Special commands:\n"
            "  [cyan]/timeline[/cyan] - View decision timeline\n"
            "  [cyan]/lessons[/cyan] - View lessons learned\n"
            "  [cyan]/stats[/cyan] - View statistics\n"
            "  [cyan]/exit[/cyan] - Exit\n\n"
            "Type your question:",
            border_style="blue"
        ))

        while True:
            try:
                question = console.input("\n[bold cyan]🔮 > [/bold cyan]").strip()

                if not question:
                    continue

                if question.lower() == "/exit":
                    console.print("[dim]Goodbye![/dim]")
                    break
                elif question.lower() == "/timeline":
                    self.show_timeline()
                elif question.lower() == "/lessons":
                    self.show_lessons()
                elif question.lower() == "/stats":
                    self.show_statistics()
                else:
                    answer = self.ask_sync(question)
                    console.print("\n")
                    console.print(Markdown(answer))

            except KeyboardInterrupt:
                console.print("\n[dim]Goodbye![/dim]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
