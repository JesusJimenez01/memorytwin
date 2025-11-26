"""
CLI del Escriba - Interfaz de línea de comandos
===============================================
"""

import argparse
import sys

from rich.console import Console
from rich.panel import Panel

console = Console()


def handle_capture(args):
    """Manejar comando de captura."""
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
        console.print("[yellow]Pega el texto de thinking (termina con Ctrl+D o línea vacía):[/yellow]")
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
            console.print("[red]No se proporcionó texto.[/red]")
            return
            
        thinking_text = "\n".join(lines)
        episode = escriba.capture_thinking_sync(
            thinking_text,
            source_assistant=args.assistant,
            project_name=args.project
        )
    
    console.print(f"\n[green]✓ Episodio guardado: {episode.id}[/green]")


def handle_stats(args):
    """Manejar comando de estadísticas."""
    from memorytwin.escriba import MemoryStorage
    
    storage = MemoryStorage()
    stats = storage.get_statistics(args.project)
    
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


def handle_search(args):
    """Manejar comando de búsqueda."""
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
        console.print("[yellow]No se encontraron resultados.[/yellow]")
        return
        
    console.print(f"\n[bold]🔍 {len(results)} resultados para:[/bold] {args.query}\n")
    
    for i, result in enumerate(results, 1):
        ep = result.episode
        console.print(Panel(
            f"[bold]Tarea:[/bold] {ep.task}\n"
            f"[bold]Resumen:[/bold] {ep.solution_summary}\n"
            f"[bold]Tipo:[/bold] {ep.episode_type.value} | "
            f"[bold]Fecha:[/bold] {ep.timestamp.strftime('%Y-%m-%d %H:%M')}\n"
            f"[bold]Relevancia:[/bold] {result.relevance_score:.2%}",
            title=f"Resultado {i}",
            border_style="cyan"
        ))


def handle_query(args):
    """Manejar consulta RAG."""
    import asyncio
    from memorytwin.oraculo import RAGEngine
    from memorytwin.escriba import MemoryStorage
    
    storage = MemoryStorage()
    rag = RAGEngine(storage=storage)
    
    console.print(f"\n[bold cyan]🤔 Consultando:[/bold cyan] {args.question}\n")
    
    result = asyncio.run(rag.query(
        question=args.question,
        project_name=args.project
    ))
    
    console.print(Panel(
        result["answer"],
        title="💡 Respuesta",
        border_style="green"
    ))
    
    if result.get("sources"):
        console.print("\n[dim]Fuentes consultadas:[/dim]")
        for src in result["sources"][:3]:
            console.print(f"  • {src['task'][:60]}...")


def handle_lessons(args):
    """Manejar comando de lecciones."""
    from memorytwin.oraculo import RAGEngine
    from memorytwin.escriba import MemoryStorage
    
    storage = MemoryStorage()
    rag = RAGEngine(storage=storage)
    
    lessons = rag.get_lessons(project_name=args.project)
    
    if not lessons:
        console.print("[yellow]No hay lecciones registradas aún.[/yellow]")
        return
    
    console.print(f"\n[bold]📚 {len(lessons)} lecciones aprendidas:[/bold]\n")
    
    for lesson in lessons:
        console.print(Panel(
            f"[bold]{lesson['lesson']}[/bold]\n\n"
            f"[dim]De: {lesson['from_task'][:60]}...[/dim]\n"
            f"[dim]Fecha: {lesson['timestamp'].strftime('%Y-%m-%d')} | Tags: {', '.join(lesson['tags'][:3])}[/dim]",
            border_style="yellow"
        ))


def main():
    """Punto de entrada del CLI del Escriba."""
    parser = argparse.ArgumentParser(
        description="Memory Twin - Escriba: Captura de memoria técnica"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")
    
    # Comando: capture
    capture_parser = subparsers.add_parser(
        "capture", 
        help="Capturar pensamiento desde archivo o clipboard"
    )
    capture_parser.add_argument(
        "--file", "-f",
        help="Archivo con el texto de thinking"
    )
    capture_parser.add_argument(
        "--clipboard", "-c",
        action="store_true",
        help="Capturar desde clipboard"
    )
    capture_parser.add_argument(
        "--assistant", "-a",
        default="unknown",
        help="Asistente fuente (copilot, claude, cursor)"
    )
    capture_parser.add_argument(
        "--project", "-p",
        default="default",
        help="Nombre del proyecto"
    )
    
    # Comando: stats
    stats_parser = subparsers.add_parser(
        "stats",
        help="Ver estadísticas de la memoria"
    )
    stats_parser.add_argument(
        "--project", "-p",
        help="Filtrar por proyecto"
    )
    
    # Comando: search
    search_parser = subparsers.add_parser(
        "search",
        help="Buscar en la memoria"
    )
    search_parser.add_argument(
        "query",
        help="Texto de búsqueda"
    )
    search_parser.add_argument(
        "--top", "-k",
        type=int,
        default=5,
        help="Número de resultados"
    )
    search_parser.add_argument(
        "--project", "-p",
        help="Filtrar por proyecto"
    )
    
    # Comando: query (RAG)
    query_parser = subparsers.add_parser(
        "query",
        help="Consultar con RAG (respuesta generada)"
    )
    query_parser.add_argument(
        "question",
        help="Pregunta a responder"
    )
    query_parser.add_argument(
        "--project", "-p",
        help="Filtrar por proyecto"
    )
    
    # Comando: lessons
    lessons_parser = subparsers.add_parser(
        "lessons",
        help="Ver lecciones aprendidas"
    )
    lessons_parser.add_argument(
        "--project", "-p",
        help="Filtrar por proyecto"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    try:
        if args.command == "capture":
            handle_capture(args)
        elif args.command == "stats":
            handle_stats(args)
        elif args.command == "search":
            handle_search(args)
        elif args.command == "query":
            handle_query(args)
        elif args.command == "lessons":
            handle_lessons(args)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
