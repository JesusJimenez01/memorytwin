"""
CLI del Escriba - Interfaz de línea de comandos
===============================================
"""

import argparse
import logging
import sys
import warnings

# Silenciar warnings molestos ANTES de cualquier import
logging.getLogger("langfuse").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*ended span.*")

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


def handle_onboard(args):
    """Ejecutar onboarding de un proyecto existente."""
    import asyncio
    from pathlib import Path
    from memorytwin.escriba.project_analyzer import ProjectAnalyzer, onboard_project
    
    project_path = Path(args.path).resolve()
    
    if not project_path.exists():
        console.print(f"[red]Error: El directorio no existe: {project_path}[/red]")
        return
    
    console.print(Panel(
        f"[bold cyan]🔍 Analizando proyecto...[/bold cyan]\n"
        f"Ruta: {project_path}",
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
        
        # Mostrar resumen
        stack_list = ", ".join([s['technology'] for s in analysis['stack'][:5]]) or "No detectado"
        patterns_list = ", ".join([p.get('pattern', p.get('directory', '')) for p in analysis['patterns'][:3]]) or "No detectados"
        deps_list = ", ".join(analysis['dependencies']['main'][:8]) or "No detectadas"
        
        console.print(Panel(
            f"[bold green]✓ Onboarding completado![/bold green]\n\n"
            f"[bold]Proyecto:[/bold] {result['project_name']}\n"
            f"[bold]Episodio:[/bold] {result['episode_id']}\n\n"
            f"[bold]Stack detectado:[/bold]\n  {stack_list}\n\n"
            f"[bold]Patrones:[/bold]\n  {patterns_list}\n\n"
            f"[bold]Dependencias principales:[/bold]\n  {deps_list}\n\n"
            f"[dim]La memoria inicial del proyecto ha sido creada.\n"
            f"El agente ahora conoce la estructura y convenciones.[/dim]",
            title="🧠 Análisis Completado",
            border_style="green"
        ))
        
        if args.verbose:
            console.print("\n[bold]Texto de onboarding generado:[/bold]")
            console.print(result['onboarding_text'])
            
    except Exception as e:
        console.print(f"[red]Error durante el onboarding: {e}[/red]")
        raise


def handle_health_check(args):
    """Verificar integridad del sistema Memory Twin."""
    from memorytwin.escriba import MemoryStorage
    from memorytwin.config import get_chroma_dir, get_sqlite_path
    
    console.print(Panel(
        "[bold cyan]🔍 Verificando integridad del sistema...[/bold cyan]",
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
        
        # Verificar consistencia entre SQLite y ChromaDB
        if sqlite_count != chroma_count:
            issues.append(
                f"⚠️ Inconsistencia: SQLite tiene {sqlite_count} episodios, "
                f"ChromaDB tiene {chroma_count}"
            )
        
        # Verificar archivos de base de datos
        chroma_dir = get_chroma_dir()
        sqlite_path = get_sqlite_path()
        
        if not chroma_dir.exists():
            issues.append(f"❌ Directorio ChromaDB no existe: {chroma_dir}")
        
        if not sqlite_path.exists():
            issues.append(f"❌ Archivo SQLite no existe: {sqlite_path}")
        
        # Verificar tamaño de base de datos
        if sqlite_path.exists():
            size_mb = sqlite_path.stat().st_size / (1024 * 1024)
            if size_mb > 100:
                warnings.append(f"📦 Base de datos grande: {size_mb:.1f} MB")
        
        # Resultados
        if issues:
            console.print("\n[bold red]❌ Problemas encontrados:[/bold red]")
            for issue in issues:
                console.print(f"  {issue}")
        
        if warnings:
            console.print("\n[bold yellow]⚠️ Advertencias:[/bold yellow]")
            for warning in warnings:
                console.print(f"  {warning}")
        
        if not issues and not warnings:
            console.print(Panel(
                f"[bold green]✓ Sistema saludable[/bold green]\n\n"
                f"[bold]Episodios en SQLite:[/bold] {sqlite_count}\n"
                f"[bold]Episodios en ChromaDB:[/bold] {chroma_count}\n"
                f"[bold]Sincronización:[/bold] ✓ OK",
                title="✅ Health Check Passed",
                border_style="green"
            ))
        else:
            console.print(Panel(
                f"[bold]Episodios en SQLite:[/bold] {sqlite_count}\n"
                f"[bold]Episodios en ChromaDB:[/bold] {chroma_count}\n\n"
                f"[dim]Usa 'mt sync --repair' para corregir inconsistencias[/dim]",
                title="📊 Estado Actual",
                border_style="yellow"
            ))
        
    except Exception as e:
        console.print(f"[red]Error durante health check: {e}[/red]")
        raise


def handle_consolidate(args):
    """Consolidar episodios relacionados en meta-memorias."""
    from memorytwin.consolidation import MemoryConsolidator
    from memorytwin.escriba import MemoryStorage
    
    console.print(Panel(
        f"[bold cyan]🧠 Consolidando memorias del proyecto: {args.project}[/bold cyan]\n"
        f"Mínimo episodios por cluster: {args.min_cluster}",
        title="Memory Twin - Consolidación",
        border_style="cyan"
    ))
    
    try:
        storage = MemoryStorage()
        
        # Verificar que hay suficientes episodios
        stats = storage.get_statistics(args.project)
        total_episodes = stats['total_episodes']
        
        if total_episodes < args.min_cluster:
            console.print(
                f"[yellow]⚠️ El proyecto solo tiene {total_episodes} episodios. "
                f"Se necesitan al menos {args.min_cluster} para consolidar.[/yellow]"
            )
            return
        
        console.print(f"[dim]Analizando {total_episodes} episodios...[/dim]")
        
        # Ejecutar consolidación
        consolidator = MemoryConsolidator(
            storage=storage,
            min_cluster_size=args.min_cluster
        )
        
        meta_memories = consolidator.consolidate_project(args.project)
        
        if not meta_memories:
            console.print(
                "[yellow]No se encontraron clusters suficientemente grandes "
                "para consolidar. Intenta con --min-cluster menor.[/yellow]"
            )
            return
        
        # Mostrar resultados
        console.print(Panel(
            f"[bold green]✓ Consolidación completada![/bold green]\n\n"
            f"[bold]Meta-memorias generadas:[/bold] {len(meta_memories)}\n"
            f"[bold]Episodios consolidados:[/bold] "
            f"{sum(m.episode_count for m in meta_memories)}",
            title="🧠 Resultado",
            border_style="green"
        ))
        
        if args.verbose:
            for i, meta in enumerate(meta_memories, 1):
                console.print(Panel(
                    f"[bold]Patrón:[/bold] {meta.pattern_summary}\n\n"
                    f"[bold]Lecciones:[/bold]\n" +
                    "\n".join(f"  • {l}" for l in meta.lessons[:3]) + "\n\n"
                    f"[bold]Mejores prácticas:[/bold]\n" +
                    "\n".join(f"  • {p}" for p in meta.best_practices[:2]) + "\n\n"
                    f"[dim]Episodios: {meta.episode_count} | "
                    f"Confianza: {meta.confidence:.0%} | "
                    f"Coherencia: {meta.coherence_score:.0%}[/dim]",
                    title=f"Meta-Memoria {i}",
                    border_style="magenta"
                ))
        
    except Exception as e:
        console.print(f"[red]Error durante consolidación: {e}[/red]")
        raise


def handle_mcp(args):
    """Iniciar el servidor MCP."""
    import asyncio
    from memorytwin.mcp_server.server import _async_main
    
    console.print("[cyan]Iniciando servidor MCP...[/cyan]")
    asyncio.run(_async_main())


def handle_setup(args):
    """Configurar Memory Twin en un proyecto."""
    import json
    import os
    import shutil
    from pathlib import Path
    
    project_path = Path(args.path).resolve()
    
    if not project_path.exists():
        console.print(f"[red]Error: El directorio no existe: {project_path}[/red]")
        return
    
    console.print(Panel(
        f"[bold cyan]🔧 Configurando Memory Twin...[/bold cyan]\n"
        f"Proyecto: {project_path}",
        title="Memory Twin - Setup",
        border_style="cyan"
    ))
    
    # Detectar la mejor forma de invocar el MCP server
    # Orden de prioridad: mt en PATH (absoluto) > uvx > python -m
    mt_path = shutil.which("mt")
    uv_available = shutil.which("uv") is not None
    
    if mt_path:
        # Opción 1: mt mcp (usando ruta absoluta para robustez)
        mcp_command = mt_path
        mcp_args = ["mcp"]
        install_method = "mt"
    elif uv_available:
        # Opción 2: uvx (descarga automáticamente si no está instalado)
        mcp_command = "uvx"
        mcp_args = ["--from", "memorytwin", "mt", "mcp"]
        install_method = "uvx"
    else:
        # Opción 3: Usar Python del entorno actual
        python_exe = sys.executable
        mcp_command = python_exe
        mcp_args = ["-m", "memorytwin.escriba.cli", "mcp"]
        install_method = "python"
    
    # Contenido de las instrucciones para Copilot
    instructions_content = '''# Memory Twin - Instrucciones para Agentes IA

---
## 🚨 REGLA OBLIGATORIA - LEER PRIMERO 🚨

**ANTES de responder CUALQUIER pregunta técnica, DEBO ejecutar:**

```
get_project_context(topic="<términos relevantes de la pregunta del usuario>")
```

**NO HAY EXCEPCIONES.** Si no consulto la memoria primero, puedo:
- Sugerir soluciones que ya fallaron
- Ignorar decisiones de arquitectura previas
- Repetir errores que ya resolvimos
- Contradecir el estilo/convenciones del proyecto

**Mi primera acción SIEMPRE es obtener contexto del proyecto. SIEMPRE.**

---

## ¿Qué es Memory Twin?
Sistema de memoria episódica que captura el razonamiento técnico para evitar "amnesia técnica" en proyectos.

## ⚡ PROTOCOLO DE CONSULTA OBLIGATORIO

### PASO 1: Obtener contexto (OBLIGATORIO)
Antes de escribir CUALQUIER respuesta técnica:

```
get_project_context(topic="<palabras clave de la pregunta>")
```

Esta herramienta es INTELIGENTE:
- Si hay **pocas memorias (<20)**: devuelve TODO el contexto del proyecto
- Si hay **muchas memorias (>=20)**: devuelve estadísticas + recientes + relevantes al topic

**Nota Importante:**
Esta herramienta activa el mecanismo de **"Forgetting Curve"**. Al consultar memorias relevantes, incrementas su `access_count`, evitando que el sistema las "olvide" con el tiempo. ¡Consultar es reforzar!

Ejemplos de consultas:
- Usuario pregunta sobre autenticación → `get_project_context(topic="autenticación login JWT")`
- Usuario tiene un error → `get_project_context(topic="error <tipo de error>")`
- Usuario quiere implementar feature → `get_project_context(topic="<nombre feature>")`
- Usuario pregunta arquitectura → `get_project_context(topic="arquitectura diseño")`

### PASO 2: Analizar resultados
- Si hay memorias relevantes → Incorporar ese conocimiento
- Si no hay resultados → Proceder normalmente, pero considerar capturar después

### PASO 3: Responder mencionando contexto
- "Según memorias previas del proyecto..."
- "No encontré decisiones previas sobre esto, procedo con..."
- "Esto ya se resolvió anteriormente, la solución fue..."

### Cuándo consultar (SIEMPRE que aplique):
| Situación | Consulta obligatoria |
|-----------|---------------------|
| Pregunta técnica | `get_project_context(topic="<tema>")` |
| Error/Bug | `get_project_context(topic="error <descripción>")` |
| Nueva feature | `get_project_context(topic="<feature>")` + `get_lessons()` |
| Decisión de arquitectura | `query_memory("<pregunta>")` |
| Primera vez en proyecto | `onboard_project("<ruta>")` |
| Elegir librería/enfoque | `get_project_context(topic="<opciones>")` |

## Herramientas MCP Disponibles

### `get_project_context` - ⭐ HERRAMIENTA PRINCIPAL
**Usar SIEMPRE antes de cualquier respuesta técnica.**

Comportamiento inteligente:
- **Pocas memorias (<20)**: Devuelve TODO el contexto completo
- **Muchas memorias (>=20)**: Devuelve estadísticas + 5 recientes + 5 relevantes al topic

**Nota Importante:**
Esta herramienta activa el mecanismo de **"Forgetting Curve"**. Al consultar memorias relevantes, incrementas su `access_count`, evitando que el sistema las "olvide" con el tiempo. ¡Consultar es reforzar!

Parámetros:
- `topic` (opcional): Tema para búsqueda semántica
- `project_name` (opcional): Filtrar por proyecto

### `capture_thinking` - 🔴 CAPTURA OBLIGATORIA
**DEBO usar esta herramienta AUTOMÁTICAMENTE después de cualquier tarea técnica.**

**IMPORTANTE:** Esta es la **ÚNICA** forma de persistir conocimiento. Si no la uso, el trabajo se pierde.

**💡 TIP:** Hay 3 formas de capturar, elige la más conveniente:
1. `capture_quick` - ⚡ La más rápida (solo what + why)
2. `capture_decision` - 🎯 Para decisiones (task + decision + reasoning)
3. `capture_thinking` - 📝 Para texto libre extenso

#### ✅ CAPTURAR SIEMPRE (sin excepción):
- Resolví un bug o error (cualquiera, no importa si es "simple")
- Tomé una decisión técnica (librería, patrón, enfoque)
- Comparé alternativas antes de elegir
- Descubrí algo inesperado (gotcha, edge case, comportamiento raro)
- Modifiqué código existente (refactor, mejora, fix)
- Implementé una feature nueva
- Configuré algo (entorno, herramientas, dependencias)
- Investigué documentación o código para entender algo
- El usuario me pidió hacer algo y lo completé
- Encontré un problema en documentación/código y lo corregí

#### ❌ NO capturar SOLO cuando:
- Respuesta puramente informativa sin acción (ej: "¿qué hora es?")
- Conversación casual sin contenido técnico
- El usuario explícitamente dice "no guardes esto"

#### 🎯 REGLA DE ORO: Ante la duda, CAPTURAR
Es mejor tener una memoria "de más" que perder conocimiento valioso.

Parámetros:
- `thinking_text` (requerido): Texto de razonamiento del modelo
- `user_prompt` (opcional): Prompt original del usuario
- `code_changes` (opcional): Cambios de código asociados
- `source_assistant` (opcional): copilot, claude, cursor, etc.
- `project_name` (opcional): Nombre del proyecto

### `capture_decision` - 🎯 CAPTURA ESTRUCTURADA (PREFERIDA)
**Forma más conveniente de capturar decisiones técnicas.**

Usar cuando tengas los datos organizados en campos separados. Más cómodo que escribir texto libre.

Parámetros:
- `task` (requerido): Descripción breve de la tarea o problema
- `decision` (requerido): La decisión o solución tomada
- `reasoning` (requerido): Por qué se tomó esta decisión
- `alternatives` (opcional): Array de alternativas consideradas
- `lesson` (opcional): Lección aprendida para el futuro
- `context` (opcional): Contexto adicional
- `project_name` (opcional): Nombre del proyecto

**Ejemplo:**
```
capture_decision(
    task="Elegir base de datos",
    decision="PostgreSQL",
    alternatives=["MongoDB", "MySQL"],
    reasoning="Necesitamos ACID y queries complejas",
    lesson="Para datos relacionales con transacciones, SQL > NoSQL"
)
```

### `capture_quick` - ⚡ CAPTURA RÁPIDA (MÍNIMO ESFUERZO)
**La forma más simple de capturar. Solo 2 campos requeridos.**

Usar para capturas rápidas sin mucho detalle. Ideal cuando tienes prisa.

Parámetros:
- `what` (requerido): ¿Qué hiciste? (acción realizada)
- `why` (requerido): ¿Por qué lo hiciste? (razón)
- `lesson` (opcional pero recomendado): Lección aprendida
- `project_name` (opcional): Nombre del proyecto

**Ejemplos:**
```
capture_quick(
    what="Añadí retry logic al cliente HTTP",
    why="Las llamadas a la API fallaban intermitentemente"
)

capture_quick(
    what="Cambié de axios a fetch",
    why="Reducir dependencias, fetch nativo es suficiente",
    lesson="Evaluar siempre si una dependencia es realmente necesaria"
)
```

### `query_memory` - Consultar memorias con RAG
Usar cuando:
- El usuario pregunta "¿por qué hicimos X?"
- El usuario pregunta "¿cómo resolvimos algo similar?"
- Antes de tomar una decisión importante (consultar precedentes)

Parámetros:
- `question` (requerido): Pregunta a responder
- `project_name` (opcional): Filtrar por proyecto
- `num_episodes` (opcional): Número de episodios a consultar (1-10, default: 5)

### `search_episodes` - Búsqueda semántica de episodios
Usar para búsquedas específicas de temas o tecnologías.
Devuelve los episodios más relevantes para un término de búsqueda.
*Nota: Los resultados consultados reciben un boost de relevancia para el futuro.*

Parámetros:
- `query` (requerido): Término de búsqueda
- `project_name` (opcional): Filtrar por proyecto
- `top_k` (opcional): Número de resultados (default: 5)

### `get_episode` - Obtener episodio completo
Usar cuando necesitas profundizar en los detalles de una decisión específica.
Devuelve el contenido COMPLETO: thinking, alternativas, factores de decisión, contexto y lecciones.

Parámetros:
- `episode_id` (requerido): UUID del episodio a recuperar

### `get_lessons` - Lecciones aprendidas
Usar para:
- Onboarding de nuevos miembros
- Revisión antes de empezar feature similar
- El usuario pide "¿qué hemos aprendido sobre X?"

Parámetros:
- `project_name` (opcional): Filtrar por proyecto
- `tags` (opcional): Array de tags para filtrar

### `get_timeline` - Ver historial cronológico
Usar para ver evolución cronológica del proyecto y entender qué se hizo cuándo.

Parámetros:
- `project_name` (opcional): Filtrar por proyecto
- `limit` (opcional): Máximo de episodios a retornar (default: 20)

### `get_statistics` - Estadísticas de la memoria
Obtiene estadísticas de la base de memoria: total de episodios, distribución por tipo y asistente.

Parámetros:
- `project_name` (opcional): Filtrar por proyecto

### `onboard_project` - Onboarding de proyecto existente
Usar cuando:
- ✅ Es la primera vez que trabajo en este proyecto
- ✅ El usuario pide "analiza el proyecto", "conoce el código"
- ✅ Necesito entender la estructura antes de hacer cambios grandes
- ✅ No hay memorias previas y quiero crear contexto inicial

Genera automáticamente un episodio con:
- Stack tecnológico detectado
- Patrones arquitectónicos
- Dependencias principales
- Convenciones de código

Parámetros:
- `project_path` (requerido): Ruta absoluta al proyecto
- `project_name` (opcional): Nombre del proyecto (se detecta automáticamente)

## Flujo de Trabajo OBLIGATORIO

### 🔄 CICLO COMPLETO (SIEMPRE):
```
1. INICIO: get_project_context(topic="...") 
2. TRABAJO: Realizar la tarea solicitada
3. FIN: capture_thinking(thinking_text="...") 
```

### Checklist antes de terminar respuesta:
- [ ] ¿Consulté la memoria al inicio? Si no → HACERLO AHORA
- [ ] ¿Hice algo técnico? Si sí → CAPTURAR MEMORIA
- [ ] ¿Resolví un problema? Si sí → CAPTURAR MEMORIA  
- [ ] ¿Tomé una decisión? Si sí → CAPTURAR MEMORIA
- [ ] ¿Modifiqué código? Si sí → CAPTURAR MEMORIA
- [ ] ¿Investigué algo? Si sí → CAPTURAR MEMORIA

### ⚠️ RECORDATORIO CRÍTICO:
**NO terminar una respuesta técnica sin haber ejecutado `capture_thinking`.**
El conocimiento que no se captura, SE PIERDE PARA SIEMPRE.

## Formato del Thinking a Capturar

Incluir siempre que sea posible:
- **Tarea**: Qué se intentaba resolver
- **Contexto**: Estado inicial, restricciones
- **Alternativas**: Opciones consideradas
- **Decisión**: Qué se eligió y por qué
- **Solución**: Cómo se implementó
- **Lecciones**: Qué aprendimos

## Ejemplo de Captura Automática

```
Cuando detecto que acabo de:
1. Debuggear un error por más de 2 intercambios
2. Comparar 2+ opciones antes de elegir
3. Descubrir un comportamiento inesperado
4. Implementar algo que requirió investigación

→ Llamar a capture_thinking con el contexto completo
```

## 🚨 EJEMPLO DE FLUJO CORRECTO

### Usuario pregunta: "¿Por qué falla mi función de login?"

```
# 1. PRIMERO: Consultar memoria
get_project_context(topic="login autenticación error")

# 2. DESPUÉS: Trabajar en la solución
[Analizar código, debuggear, encontrar el problema, proponer fix]

# 3. FINALMENTE: Capturar el conocimiento
capture_thinking(
    thinking_text="## Tarea\\nResolver error en función login...\\n## Problema\\nEl token JWT...\\n## Solución\\n...\\n## Lecciones\\n...",
    project_name="mi-proyecto",
    source_assistant="copilot"
)
```

**SI NO CAPTURO AL FINAL, ESTOY FALLANDO MI FUNCIÓN.**

## Proyecto Actual
- **Nombre del proyecto**: Usar el nombre de la carpeta del workspace
- **Source assistant**: "copilot" para GitHub Copilot
'''
    
    # Configuración MCP para VS Code - usar comando detectado
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
    
    # Crear directorio .github si no existe
    github_dir = project_path / ".github"
    github_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear directorio .vscode si no existe
    vscode_dir = project_path / ".vscode"
    vscode_dir.mkdir(parents=True, exist_ok=True)
    
    # Escribir archivo de instrucciones
    instructions_path = github_dir / "copilot-instructions.md"
    instructions_path.write_text(instructions_content, encoding="utf-8")
    files_created.append(str(instructions_path.relative_to(project_path)))
    
    # Escribir mcp.json
    mcp_path = vscode_dir / "mcp.json"
    mcp_path.write_text(json.dumps(mcp_config, indent=2), encoding="utf-8")
    files_created.append(str(mcp_path.relative_to(project_path)))
    
    # Crear .env si no existe
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
    
    # Asegurar que .env y data/ están en .gitignore
    gitignore_path = project_path / ".gitignore"
    gitignore_updated = False
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
            gitignore_updated = True
            files_updated.append(".gitignore")
    else:
        gitignore_path.write_text("# Memory Twin\n.env\ndata/\n", encoding="utf-8")
        files_created.append(".gitignore")
    
    # Mostrar resumen
    files_list = "\n".join(f"  • [cyan]{f}[/cyan]" for f in files_created)
    updated_list = "\n".join(f"  • [yellow]{f}[/yellow]" for f in files_updated) if files_updated else ""
    
    # Mensaje según método de instalación
    if install_method == "mt":
        mcp_info = "[green]mt mcp[/green] (simple y universal)"
        portability_note = ""
    elif install_method == "uvx":
        mcp_info = "[green]uvx[/green] (universal)"
        portability_note = ""
    else:
        mcp_info = f"[yellow]Python específico[/yellow]"
        portability_note = """
[yellow]⚠️ Nota:[/yellow] La configuración usa la ruta de tu Python actual.
   Para una configuración más portable, asegúrate de que 'mt' esté en PATH.
"""
    
    next_steps = """
[bold]Próximos pasos:[/bold]
  1. Edita [cyan].env[/cyan] y configura tu LLM (Gemini u OpenRouter)
  2. Reinicia VS Code (F1 → "Developer: Reload Window")
  3. Verifica en MCP: List Servers que memorytwin está conectado
  4. ¡Listo! Copilot usará Memory Twin automáticamente
"""
    
    result_text = f"[bold green]✓ Memory Twin configurado![/bold green]\n\n"
    result_text += f"[bold]Archivos creados:[/bold]\n{files_list}\n"
    if updated_list:
        result_text += f"\n[bold]Archivos actualizados:[/bold]\n{updated_list}\n"
    result_text += f"\n[bold]MCP Server:[/bold] {mcp_info}\n"
    if portability_note:
        result_text += portability_note
    result_text += next_steps
    
    console.print(Panel(
        result_text,
        title="🧠 Setup Completado",
        border_style="green"
    ))


def handle_oraculo(args):
    """Lanzar interfaz web (Oráculo)."""
    from memorytwin.oraculo.app import main as launch_oraculo
    launch_oraculo()


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
    
    # Comando: onboard
    onboard_parser = subparsers.add_parser(
        "onboard",
        help="Analizar proyecto existente y crear memoria inicial de onboarding"
    )
    onboard_parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Ruta al proyecto a analizar (por defecto: directorio actual)"
    )
    onboard_parser.add_argument(
        "--project", "-p",
        help="Nombre del proyecto (se detecta automáticamente si no se especifica)"
    )
    onboard_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostrar texto completo del análisis"
    )
    
    # Comando: health-check
    health_parser = subparsers.add_parser(
        "health-check",
        help="Verificar integridad del sistema (SQLite + ChromaDB)"
    )
    
    # Comando: consolidate
    consolidate_parser = subparsers.add_parser(
        "consolidate",
        help="Consolidar episodios relacionados en meta-memorias"
    )
    
    # Comando: mcp (lanzar servidor MCP)
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Iniciar el servidor MCP para VS Code/Copilot"
    )

    # Comando: oraculo (interfaz web)
    oraculo_parser = subparsers.add_parser(
        "oraculo",
        help="Abrir interfaz web para gestionar memorias"
    )

    consolidate_parser.add_argument(
        "--project", "-p",
        required=True,
        help="Nombre del proyecto a consolidar"
    )
    consolidate_parser.add_argument(
        "--min-cluster", "-m",
        type=int,
        default=3,
        help="Mínimo de episodios para formar un cluster (default: 3)"
    )
    consolidate_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostrar detalles de cada meta-memoria generada"
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
