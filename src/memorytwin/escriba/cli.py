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


def handle_setup(args):
    """Configurar Memory Twin en un proyecto."""
    import json
    import os
    from pathlib import Path
    
    project_path = Path(args.path).resolve()
    
    if not project_path.exists():
        console.print(f"[red]Error: El directorio no existe: {project_path}[/red]")
        return
    
    # Detectar ruta de Python del entorno actual
    python_exe = sys.executable
    
    # Contenido de las instrucciones para Copilot
    instructions_content = '''# Memory Twin - Instrucciones para Agentes IA

## ¿Qué es Memory Twin?
Sistema de memoria episódica que captura el razonamiento técnico para evitar "amnesia técnica" en proyectos.

## ⚡ COMPORTAMIENTO AUTOMÁTICO (IMPORTANTE)

### Antes de responder cualquier pregunta técnica:
1. **SIEMPRE** usar `query_memory` o `search_episodes` para buscar contexto relevante
2. Si hay memorias relacionadas, incorporar ese conocimiento en la respuesta
3. Mencionar brevemente si encontré información útil de sesiones anteriores

### Consultas automáticas recomendadas:
- Antes de sugerir una solución → buscar si ya resolvimos algo similar
- Antes de elegir una librería/enfoque → buscar decisiones previas
- Cuando el usuario menciona un error → buscar si lo vimos antes
- Al empezar una nueva feature → consultar lecciones aprendidas relevantes

## Herramientas MCP Disponibles

### `capture_thinking` - Capturar razonamiento
Usar cuando:
- ✅ Se resuelve un bug no trivial
- ✅ Se toma una decisión de arquitectura
- ✅ Se comparan alternativas y se elige una
- ✅ Se descubre algo inesperado (gotcha, edge case)
- ✅ El usuario dice "guarda esto", "recuerda esto", "captura esto"

NO usar cuando:
- ❌ Cambios triviales (typos, formateo)
- ❌ Preguntas simples sin razonamiento complejo
- ❌ Código boilerplate sin decisiones

### `query_memory` - Consultar memorias
Usar cuando:
- El usuario pregunta "¿por qué hicimos X?"
- El usuario pregunta "¿cómo resolvimos algo similar?"
- Antes de tomar una decisión importante (consultar precedentes)

### `get_lessons` - Lecciones aprendidas
Usar para:
- Onboarding de nuevos miembros
- Revisión antes de empezar feature similar
- El usuario pide "¿qué hemos aprendido sobre X?"

### `search_episodes` - Buscar episodios
Usar para búsquedas específicas de temas o tecnologías.

### `get_timeline` - Ver historial
Usar para ver evolución cronológica del proyecto.

## Flujo de Trabajo Recomendado

### Durante desarrollo:
1. Cuando resuelvas algo complejo → `capture_thinking` automáticamente
2. Incluir: contexto, alternativas consideradas, decisión final, lecciones

### Antes de empezar tarea:
1. `query_memory` para ver si hay contexto relevante
2. `get_lessons` para evitar errores pasados

## Proyecto Actual
- **Nombre del proyecto**: Usar el nombre de la carpeta del workspace
- **Source assistant**: "copilot" para GitHub Copilot
'''
    
    # Configuración MCP para VS Code
    mcp_config = {
        "mcpServers": {
            "memorytwin": {
                "command": python_exe,
                "args": ["-m", "memorytwin.mcp_server.server"]
            }
        }
    }
    
    # Crear directorio .github si no existe
    github_dir = project_path / ".github"
    github_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear directorio .vscode si no existe
    vscode_dir = project_path / ".vscode"
    vscode_dir.mkdir(parents=True, exist_ok=True)
    
    # Escribir archivo de instrucciones
    instructions_path = github_dir / "copilot-instructions.md"
    instructions_path.write_text(instructions_content, encoding="utf-8")
    
    # Escribir mcp.json
    mcp_path = vscode_dir / "mcp.json"
    mcp_path.write_text(json.dumps(mcp_config, indent=2), encoding="utf-8")
    
    console.print(Panel(
        f"[bold green]✓ Memory Twin configurado![/bold green]\n\n"
        f"Archivos creados:\n"
        f"  • [cyan]{instructions_path}[/cyan]\n"
        f"  • [cyan]{mcp_path}[/cyan]\n\n"
        f"El agente ahora capturará razonamiento automáticamente\n"
        f"y consultará memorias previas en este proyecto.",
        title="🧠 Setup Completado",
        border_style="green"
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
    
    # Comando: setup
    setup_parser = subparsers.add_parser(
        "setup",
        help="Configurar Memory Twin en un proyecto (crea .github/copilot-instructions.md)"
    )
    setup_parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Ruta al proyecto (por defecto: directorio actual)"
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
        elif args.command == "setup":
            handle_setup(args)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
