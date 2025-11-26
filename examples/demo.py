"""
Ejemplo de uso de Memory Twin
=============================

Este script demuestra cómo usar el sistema completo:
1. Capturar pensamiento con el Escriba
2. Consultar memoria con el Oráculo
"""

import asyncio
from rich.console import Console
from rich.panel import Panel

console = Console()


# Ejemplo de texto de "thinking" que capturaríamos de un asistente de IA
EXAMPLE_THINKING = """
El usuario quiere implementar un sistema de autenticación para su API REST.

Analicé varias opciones disponibles:

1. **Sessions con Redis**
   - Pros: Familiar, fácil de invalidar
   - Cons: Requiere infraestructura adicional (Redis), no es stateless
   - Descartado porque el usuario quiere una arquitectura simple sin dependencias adicionales

2. **OAuth2 completo**
   - Pros: Estándar de la industria, muy seguro
   - Cons: Complejo de implementar, overkill para las necesidades actuales
   - Descartado por la complejidad de implementación para un MVP

3. **JWT (JSON Web Tokens)** ✓ ELEGIDO
   - Pros: Stateless, escalable, estándar, fácil de implementar
   - Cons: No se pueden invalidar tokens individualmente (sin blacklist)
   - Elegido porque cumple con los requisitos de simplicidad y escalabilidad

Factores de decisión principales:
- El API necesita ser stateless para escalar horizontalmente
- El equipo tiene experiencia previa con JWT
- No hay requisitos de revocación inmediata de tokens
- Se necesita soporte para clientes móviles

Implementación decidida:
- Usar la librería PyJWT (más ligera que python-jose para este caso)
- Tokens de acceso con expiración de 1 hora
- Refresh tokens con expiración de 7 días
- Almacenar refresh tokens en la base de datos para poder revocarlos
- Usar HS256 para simplicidad (RS256 si se necesita verificación distribuida en el futuro)

Lecciones del equipo previas que apliqué:
- Siempre validar el algoritmo en el decode para evitar ataques de "none algorithm"
- No almacenar datos sensibles en el payload del JWT
- Usar HTTPS obligatorio en producción
"""


async def demo_escriba():
    """Demostración del Escriba."""
    console.print(Panel(
        "[bold cyan]Demo del Escriba - Captura de Pensamiento[/bold cyan]",
        border_style="cyan"
    ))
    
    try:
        from memorytwin.escriba import Escriba
        
        # Inicializar Escriba
        escriba = Escriba(project_name="demo-project")
        
        # Capturar el pensamiento de ejemplo
        episode = await escriba.capture_thinking(
            thinking_text=EXAMPLE_THINKING,
            user_prompt="Implementa autenticación para mi API REST en FastAPI",
            code_changes="# Código generado por el asistente...",
            source_assistant="demo-assistant"
        )
        
        console.print("\n[bold green]✓ Episodio capturado exitosamente![/bold green]\n")
        
        console.print(f"[bold]ID:[/bold] {episode.id}")
        console.print(f"[bold]Tarea:[/bold] {episode.task}")
        console.print(f"[bold]Tipo:[/bold] {episode.episode_type.value}")
        console.print(f"[bold]Tags:[/bold] {', '.join(episode.tags)}")
        console.print(f"\n[bold]Alternativas consideradas:[/bold]")
        for alt in episode.reasoning_trace.alternatives_considered:
            console.print(f"  • {alt}")
        console.print(f"\n[bold]Factores de decisión:[/bold]")
        for factor in episode.reasoning_trace.decision_factors:
            console.print(f"  • {factor}")
        console.print(f"\n[bold]Lecciones aprendidas:[/bold]")
        for lesson in episode.lessons_learned:
            console.print(f"  • {lesson}")
            
        return episode
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Asegúrate de tener GOOGLE_API_KEY configurada en .env[/yellow]")
        return None


async def demo_oraculo():
    """Demostración del Oráculo."""
    console.print("\n" + "="*60 + "\n")
    console.print(Panel(
        "[bold blue]Demo del Oráculo - Consulta de Memoria[/bold blue]",
        border_style="blue"
    ))
    
    try:
        from memorytwin.oraculo import Oraculo
        
        # Inicializar Oráculo
        oraculo = Oraculo(project_name="demo-project")
        
        # Hacer una pregunta
        question = "¿Por qué elegimos JWT para la autenticación y qué alternativas consideramos?"
        console.print(f"\n[bold]Pregunta:[/bold] {question}\n")
        
        answer = await oraculo.ask(question)
        
        console.print("[bold]Respuesta del Oráculo:[/bold]\n")
        console.print(answer)
        
        # Mostrar estadísticas
        console.print("\n" + "-"*40 + "\n")
        oraculo.show_statistics()
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


async def main():
    """Ejecutar demo completa."""
    console.print(Panel(
        "[bold magenta]🧠 The Memory Twin - Demo[/bold magenta]\n\n"
        "Este script demuestra el flujo completo del sistema:\n"
        "1. Escriba captura y procesa un 'thinking' de ejemplo\n"
        "2. Oráculo responde preguntas sobre las memorias",
        title="Memory Twin",
        border_style="magenta"
    ))
    
    # Demo Escriba
    episode = await demo_escriba()
    
    if episode:
        # Demo Oráculo
        await demo_oraculo()
    
    console.print("\n" + "="*60)
    console.print("[bold green]Demo completada![/bold green]")
    console.print("\nPróximos pasos:")
    console.print("  • Inicia la interfaz web: [cyan]python -m memorytwin.oraculo.app[/cyan]")
    console.print("  • Usa el CLI: [cyan]memorytwin-escriba --help[/cyan]")
    console.print("  • Configura MCP en VS Code para integración automática")


if __name__ == "__main__":
    asyncio.run(main())
