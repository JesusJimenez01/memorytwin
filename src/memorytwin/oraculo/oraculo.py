"""
Oráculo - Agente Principal de Consulta
======================================

Coordina la recuperación de conocimiento y proporciona
una interfaz conversacional sobre las memorias.
"""

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from memorytwin.config import get_settings
from memorytwin.oraculo.rag_engine import RAGEngine

console = Console()


class Oraculo:
    """
    Agente Oráculo - Asistente de recuperación de conocimiento.
    
    Responde preguntas sobre decisiones técnicas, proporciona
    timeline de evolución y agrega lecciones aprendidas.
    """
    
    def __init__(
        self,
        rag_engine: Optional[RAGEngine] = None,
        project_name: Optional[str] = None
    ):
        """
        Inicializar el Oráculo.
        
        Args:
            rag_engine: Motor RAG (se crea uno si no se provee)
            project_name: Proyecto por defecto para filtrar
        """
        self.rag_engine = rag_engine or RAGEngine()
        self.project_name = project_name
        
        console.print(Panel(
            f"[bold blue]✓ Oráculo inicializado[/bold blue]\n"
            f"Proyecto: {project_name or 'Todos'}",
            title="Memory Twin - Oráculo",
            border_style="blue"
        ))
    
    async def ask(self, question: str) -> str:
        """
        Hacer una pregunta al Oráculo.
        
        Args:
            question: Pregunta sobre el proyecto
            
        Returns:
            Respuesta basada en las memorias
        """
        console.print(f"\n[cyan]🔮 Consultando memorias...[/cyan]")
        
        result = await self.rag_engine.query(
            question,
            project_name=self.project_name
        )
        
        answer = result["answer"]
        episodes_count = len(result["episodes_used"])
        
        if result["context_provided"]:
            console.print(
                f"[dim]Basado en {episodes_count} episodio(s) de memoria[/dim]"
            )
        
        return answer
    
    def ask_sync(self, question: str) -> str:
        """Versión síncrona de ask."""
        import asyncio
        return asyncio.run(self.ask(question))
    
    def show_timeline(self, limit: int = 20):
        """
        Mostrar timeline de decisiones en consola.
        
        Args:
            limit: Número de episodios a mostrar
        """
        timeline = self.rag_engine.get_timeline(
            project_name=self.project_name,
            limit=limit
        )
        
        if not timeline:
            console.print("[yellow]No hay episodios registrados.[/yellow]")
            return
            
        console.print(f"\n[bold]📅 Timeline de Decisiones ({len(timeline)} episodios)[/bold]\n")
        
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
        Mostrar lecciones aprendidas.
        
        Args:
            tags: Filtrar por tags específicos
        """
        lessons = self.rag_engine.get_lessons(
            project_name=self.project_name,
            tags=tags
        )
        
        if not lessons:
            console.print("[yellow]No hay lecciones documentadas.[/yellow]")
            return
            
        console.print(f"\n[bold]📚 Lecciones Aprendidas ({len(lessons)})[/bold]\n")
        
        for i, lesson in enumerate(lessons, 1):
            console.print(Panel(
                f"[bold yellow]{lesson['lesson']}[/bold yellow]\n\n"
                f"[dim]De: {lesson['from_task'][:80]}...[/dim]\n"
                f"[dim]Fecha: {lesson['timestamp'].strftime('%Y-%m-%d')}[/dim]\n"
                f"[dim]Tags: {', '.join(lesson['tags'][:5])}[/dim]",
                title=f"Lección {i}",
                border_style="yellow"
            ))
    
    def show_statistics(self):
        """Mostrar estadísticas de la memoria."""
        stats = self.rag_engine.get_statistics(self.project_name)
        
        console.print(Panel(
            f"[bold]Total de episodios:[/bold] {stats['total_episodes']}\n"
            f"[bold]En ChromaDB:[/bold] {stats['chroma_count']}\n\n"
            f"[bold]Por tipo:[/bold]\n" +
            "\n".join(f"  • {k}: {v}" for k, v in stats['by_type'].items() if v > 0) +
            "\n\n[bold]Por asistente:[/bold]\n" +
            "\n".join(f"  • {k}: {v}" for k, v in stats['by_assistant'].items()),
            title="📊 Estadísticas de Memoria",
            border_style="blue"
        ))
    
    def interactive_mode(self):
        """
        Modo interactivo de consulta.
        Permite hacer preguntas continuamente.
        """
        console.print(Panel(
            "[bold]Modo Interactivo del Oráculo[/bold]\n\n"
            "Comandos especiales:\n"
            "  [cyan]/timeline[/cyan] - Ver timeline de decisiones\n"
            "  [cyan]/lessons[/cyan] - Ver lecciones aprendidas\n"
            "  [cyan]/stats[/cyan] - Ver estadísticas\n"
            "  [cyan]/exit[/cyan] - Salir\n\n"
            "Escribe tu pregunta:",
            border_style="blue"
        ))
        
        while True:
            try:
                question = console.input("\n[bold cyan]🔮 > [/bold cyan]").strip()
                
                if not question:
                    continue
                    
                if question.lower() == "/exit":
                    console.print("[dim]¡Hasta luego![/dim]")
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
                console.print("\n[dim]¡Hasta luego![/dim]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
