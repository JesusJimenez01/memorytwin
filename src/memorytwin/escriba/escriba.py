"""
Escriba - Main Memory Ingestion Agent
======================================

Coordinates the processing and storage of
technical memory episodes.
"""

from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from memorytwin.escriba.processor import ThoughtProcessor
from memorytwin.escriba.storage import MemoryStorage
from memorytwin.models import Episode, ProcessedInput

console = Console()


class Escriba:
    """
    Escriba Agent - Passive observer and active documenter.

    Captures AI assistant reasoning, structures it
    using a lightweight LLM, and stores it for future queries.
    """

    def __init__(
        self,
        processor: Optional[ThoughtProcessor] = None,
        storage: Optional[MemoryStorage] = None,
        project_name: str = "default"
    ):
        """
        Initialize Escriba.

        Args:
            processor: Thought processor (creates one if not provided)
            storage: Memory storage (creates one if not provided)
            project_name: Default project name
        """
        self.processor = processor or ThoughtProcessor()
        self.storage = storage or MemoryStorage()
        self.project_name = project_name

        console.print(Panel(
            f"[bold green]✓ Escriba initialized[/bold green]\n"
            f"Project: {project_name}",
            title="Memory Twin - Escriba",
            border_style="green"
        ))

    async def capture_thinking(
        self,
        thinking_text: str,
        user_prompt: Optional[str] = None,
        code_changes: Optional[str] = None,
        source_assistant: str = "unknown",
        project_name: Optional[str] = None
    ) -> Episode:
        """
        Capture and process "thinking" text from an assistant.

        Args:
            thinking_text: Visible reasoning text from the model
            user_prompt: Original user prompt
            code_changes: Associated code changes
            source_assistant: Source assistant (copilot, claude, cursor)
            project_name: Project name (uses default if not specified)

        Returns:
            Structured and stored Episode
        """
        project = project_name or self.project_name

        console.print("[yellow]📝 Capturing thought...[/yellow]")
        console.print(f"   Source: {source_assistant}")
        console.print(f"   Project: {project}")

        # Create processed input
        raw_input = ProcessedInput(
            raw_text=thinking_text,
            user_prompt=user_prompt,
            code_changes=code_changes,
            source="api",
            captured_at=datetime.now(timezone.utc)
        )

        # Process with LLM
        console.print("[yellow]🔄 Structuring with LLM...[/yellow]")
        episode = await self.processor.process_thought(
            raw_input,
            project_name=project,
            source_assistant=source_assistant
        )

        # Store
        console.print("[yellow]💾 Storing episode...[/yellow]")
        episode_id = self.storage.store_episode(episode)

        console.print(Panel(
            f"[bold green]✓ Episode captured[/bold green]\n"
            f"ID: {episode_id}\n"
            f"Task: {episode.task[:100]}...\n"
            f"Type: {episode.episode_type.value}\n"
            f"Tags: {', '.join(episode.tags[:5])}",
            title="Memory Registered",
            border_style="green"
        ))

        return episode

    def capture_thinking_sync(
        self,
        thinking_text: str,
        user_prompt: Optional[str] = None,
        code_changes: Optional[str] = None,
        source_assistant: str = "unknown",
        project_name: Optional[str] = None
    ) -> Episode:
        """Synchronous version of capture_thinking."""
        import asyncio
        return asyncio.run(
            self.capture_thinking(
                thinking_text,
                user_prompt,
                code_changes,
                source_assistant,
                project_name
            )
        )

    def capture_from_file(
        self,
        file_path: str,
        source_assistant: str = "unknown",
        project_name: Optional[str] = None
    ) -> Episode:
        """
        Capture thinking from a text file.

        Args:
            file_path: Path to the file containing thinking text
            source_assistant: Source assistant
            project_name: Project name

        Returns:
            Structured and stored Episode
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            thinking_text = f.read()

        return self.capture_thinking_sync(
            thinking_text,
            source_assistant=source_assistant,
            project_name=project_name
        )

    def get_statistics(self) -> dict:
        """Get storage statistics."""
        return self.storage.get_statistics(self.project_name)

    def search(self, query: str, top_k: int = 5):
        """
        Search memory (simple wrapper).

        Args:
            query: Search text
            top_k: Number of results

        Returns:
            List of search results
        """
        from memorytwin.models import MemoryQuery

        memory_query = MemoryQuery(
            query=query,
            project_filter=self.project_name,
            top_k=top_k
        )

        return self.storage.search_episodes(memory_query)
