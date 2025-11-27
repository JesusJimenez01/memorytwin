"""
Servidor MCP para Memory Twin
=============================

Implementa el Model Context Protocol para exponer las
capacidades del Escriba y Oráculo a clientes compatibles.
"""

import json
import logging
from datetime import datetime
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)

from memorytwin.config import get_settings


def _format_lessons(lessons: list) -> list:
    """Format lessons list ensuring datetime objects are serializable."""
    formatted = []
    for lesson in lessons:
        formatted_lesson = {}
        for key, value in lesson.items():
            if isinstance(value, datetime):
                formatted_lesson[key] = value.isoformat()
            else:
                formatted_lesson[key] = value
        formatted.append(formatted_lesson)
    return formatted
from memorytwin.models import ProcessedInput, MemoryQuery
from memorytwin.escriba.processor import ThoughtProcessor
from memorytwin.escriba.storage import MemoryStorage
from memorytwin.oraculo.rag_engine import RAGEngine

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memorytwin.mcp")


class MemoryTwinMCPServer:
    """
    Servidor MCP que expone las herramientas de Memory Twin.
    
    Herramientas disponibles:
    - capture_thinking: Capturar razonamiento de un asistente
    - query_memory: Consultar memorias con RAG
    - get_timeline: Obtener timeline de decisiones
    - get_lessons: Obtener lecciones aprendidas
    - get_episode: Obtener episodio por ID
    """
    
    def __init__(self):
        """Inicializar servidor MCP."""
        self.server = Server("memorytwin")
        self.processor: Optional[ThoughtProcessor] = None
        self.storage: Optional[MemoryStorage] = None
        self.rag_engine: Optional[RAGEngine] = None
        
        # Registrar herramientas
        self._register_tools()
        
        logger.info("Memory Twin MCP Server inicializado")
    
    def _lazy_init(self):
        """Inicialización lazy de componentes."""
        if self.processor is None:
            self.processor = ThoughtProcessor()
        if self.storage is None:
            self.storage = MemoryStorage()
        if self.rag_engine is None:
            self.rag_engine = RAGEngine(storage=self.storage)
    
    def _register_tools(self):
        """Registrar todas las herramientas MCP."""
        
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """Listar herramientas disponibles."""
            return [
                Tool(
                    name="capture_thinking",
                    description=(
                        "Captura y almacena el razonamiento ('thinking') de un asistente de IA. "
                        "Procesa el texto con un LLM para estructurarlo en un episodio de memoria "
                        "que incluye tarea, contexto, alternativas consideradas, factores de decisión, "
                        "solución y lecciones aprendidas."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "thinking_text": {
                                "type": "string",
                                "description": "Texto de razonamiento visible del modelo"
                            },
                            "user_prompt": {
                                "type": "string",
                                "description": "Prompt original del usuario (opcional)"
                            },
                            "code_changes": {
                                "type": "string",
                                "description": "Cambios de código asociados (opcional)"
                            },
                            "source_assistant": {
                                "type": "string",
                                "description": "Asistente fuente: copilot, claude, cursor, etc.",
                                "default": "unknown"
                            },
                            "project_name": {
                                "type": "string",
                                "description": "Nombre del proyecto",
                                "default": "default"
                            }
                        },
                        "required": ["thinking_text"]
                    }
                ),
                Tool(
                    name="query_memory",
                    description=(
                        "Consulta las memorias episódicas usando RAG (Retrieval-Augmented Generation). "
                        "Permite hacer preguntas como '¿Por qué elegimos X?' o "
                        "'¿Qué alternativas consideramos para Y?'. "
                        "Devuelve una respuesta generada basada en los episodios relevantes."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Pregunta a responder usando las memorias"
                            },
                            "project_name": {
                                "type": "string",
                                "description": "Filtrar por proyecto específico (opcional)"
                            },
                            "num_episodes": {
                                "type": "integer",
                                "description": "Número de episodios a consultar (1-10)",
                                "default": 5
                            }
                        },
                        "required": ["question"]
                    }
                ),
                Tool(
                    name="get_timeline",
                    description=(
                        "Obtiene el timeline cronológico de decisiones técnicas. "
                        "Útil para ver la evolución del proyecto y entender qué se hizo cuándo."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": {
                                "type": "string",
                                "description": "Filtrar por proyecto (opcional)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Máximo de episodios a retornar",
                                "default": 20
                            }
                        },
                        "required": []
                    }
                ),
                Tool(
                    name="get_lessons",
                    description=(
                        "Obtiene las lecciones aprendidas agregadas de múltiples episodios. "
                        "Útil para onboarding y evitar repetir errores del pasado."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": {
                                "type": "string",
                                "description": "Filtrar por proyecto (opcional)"
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Filtrar por tags específicos (opcional)"
                            }
                        },
                        "required": []
                    }
                ),
                Tool(
                    name="search_episodes",
                    description=(
                        "Búsqueda semántica simple de episodios. "
                        "Devuelve los episodios más relevantes para un término de búsqueda."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Término de búsqueda"
                            },
                            "project_name": {
                                "type": "string",
                                "description": "Filtrar por proyecto (opcional)"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Número de resultados",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_statistics",
                    description=(
                        "Obtiene estadísticas de la base de memoria: "
                        "total de episodios, distribución por tipo y asistente."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": {
                                "type": "string",
                                "description": "Filtrar por proyecto (opcional)"
                            }
                        },
                        "required": []
                    }
                ),
                Tool(
                    name="get_episode",
                    description=(
                        "Obtiene el contenido COMPLETO de un episodio específico por su ID. "
                        "Incluye el razonamiento completo (thinking), todas las alternativas consideradas, "
                        "factores de decisión, contexto detallado y lecciones aprendidas. "
                        "Usar cuando se necesite profundizar en los detalles de una decisión específica."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "episode_id": {
                                "type": "string",
                                "description": "UUID del episodio a recuperar"
                            }
                        },
                        "required": ["episode_id"]
                    }
                ),
                Tool(
                    name="onboard_project",
                    description=(
                        "Analiza un proyecto existente y crea un episodio de 'onboarding' con información "
                        "sobre su estructura, stack tecnológico, patrones arquitectónicos, dependencias "
                        "y convenciones. Usar al empezar a trabajar en un proyecto nuevo o desconocido "
                        "para dar contexto inicial al agente."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_path": {
                                "type": "string",
                                "description": "Ruta absoluta al proyecto a analizar"
                            },
                            "project_name": {
                                "type": "string",
                                "description": "Nombre del proyecto (se detecta automáticamente si no se especifica)"
                            }
                        },
                        "required": ["project_path"]
                    }
                ),
                Tool(
                    name="get_project_context",
                    description=(
                        "Obtiene un resumen contextual INTELIGENTE de las memorias del proyecto. "
                        "USAR ESTA HERRAMIENTA AL INICIO de cada tarea técnica. "
                        "Comportamiento adaptativo:\n"
                        "- Si hay POCAS memorias (<20): devuelve resumen de TODAS las memorias\n"
                        "- Si hay MUCHAS memorias (>=20): devuelve estadísticas + memorias recientes + "
                        "memorias relevantes al tema consultado\n"
                        "Esto garantiza contexto completo en proyectos nuevos y eficiencia en proyectos maduros."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Tema o palabras clave de la tarea actual (para búsqueda semántica si hay muchas memorias)"
                            },
                            "project_name": {
                                "type": "string",
                                "description": "Filtrar por proyecto específico (opcional)"
                            }
                        },
                        "required": []
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
            """Ejecutar una herramienta."""
            self._lazy_init()
            
            try:
                if name == "capture_thinking":
                    return await self._capture_thinking(arguments)
                elif name == "query_memory":
                    return await self._query_memory(arguments)
                elif name == "get_timeline":
                    return await self._get_timeline(arguments)
                elif name == "get_lessons":
                    return await self._get_lessons(arguments)
                elif name == "search_episodes":
                    return await self._search_episodes(arguments)
                elif name == "get_statistics":
                    return await self._get_statistics(arguments)
                elif name == "get_episode":
                    return await self._get_episode(arguments)
                elif name == "onboard_project":
                    return await self._onboard_project(arguments)
                elif name == "get_project_context":
                    return await self._get_project_context(arguments)
                else:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=f"Herramienta desconocida: {name}"
                        )],
                        isError=True
                    )
            except Exception as e:
                logger.error(f"Error en herramienta {name}: {e}")
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error ejecutando {name}: {str(e)}"
                    )],
                    isError=True
                )
    
    async def _capture_thinking(self, args: dict) -> CallToolResult:
        """Capturar pensamiento y almacenarlo."""
        raw_input = ProcessedInput(
            raw_text=args["thinking_text"],
            user_prompt=args.get("user_prompt"),
            code_changes=args.get("code_changes"),
            source="mcp"
        )
        
        episode = await self.processor.process_thought(
            raw_input,
            project_name=args.get("project_name", "default"),
            source_assistant=args.get("source_assistant", "unknown")
        )
        
        episode_id = self.storage.store_episode(episode)
        
        result = {
            "success": True,
            "episode_id": episode_id,
            "task": episode.task,
            "type": episode.episode_type.value,
            "tags": episode.tags,
            "lessons_learned": episode.lessons_learned
        }
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False)
            )]
        )
    
    async def _query_memory(self, args: dict) -> CallToolResult:
        """Consultar memoria con RAG."""
        result = await self.rag_engine.query(
            question=args["question"],
            project_name=args.get("project_name"),
            top_k=args.get("num_episodes", 5)
        )
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=result["answer"]
            )]
        )
    
    async def _get_timeline(self, args: dict) -> CallToolResult:
        """Obtener timeline de decisiones."""
        timeline = self.rag_engine.get_timeline(
            project_name=args.get("project_name"),
            limit=args.get("limit", 20)
        )
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(timeline, indent=2, ensure_ascii=False)
            )]
        )
    
    async def _get_lessons(self, args: dict) -> CallToolResult:
        """Obtener lecciones aprendidas."""
        lessons = self.rag_engine.get_lessons(
            project_name=args.get("project_name"),
            tags=args.get("tags")
        )
        
        # Formatear para mejor legibilidad
        formatted = []
        for lesson in lessons:
            formatted.append({
                "lesson": lesson["lesson"],
                "from_task": lesson["from_task"],
                "date": lesson["timestamp"].strftime("%Y-%m-%d"),
                "tags": lesson["tags"]
            })
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(formatted, indent=2, ensure_ascii=False)
            )]
        )
    
    async def _search_episodes(self, args: dict) -> CallToolResult:
        """Búsqueda semántica de episodios."""
        query = MemoryQuery(
            query=args["query"],
            project_filter=args.get("project_name"),
            top_k=args.get("top_k", 5)
        )
        
        results = self.storage.search_episodes(query)
        
        formatted = []
        for r in results:
            formatted.append({
                "id": str(r.episode.id),
                "task": r.episode.task,
                "summary": r.episode.solution_summary,
                "type": r.episode.episode_type.value,
                "relevance": f"{r.relevance_score:.0%}",
                "date": r.episode.timestamp.strftime("%Y-%m-%d")
            })
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(formatted, indent=2, ensure_ascii=False)
            )]
        )
    
    async def _get_statistics(self, args: dict) -> CallToolResult:
        """Obtener estadísticas."""
        stats = self.storage.get_statistics(args.get("project_name"))
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(stats, indent=2, ensure_ascii=False)
            )]
        )
    
    async def _get_episode(self, args: dict) -> CallToolResult:
        """Obtener episodio completo por ID."""
        episode_id = args.get("episode_id")
        
        if not episode_id:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="Error: Se requiere episode_id"
                )],
                isError=True
            )
        
        episode = self.storage.get_episode_by_id(episode_id)
        
        if not episode:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"No se encontró episodio con ID: {episode_id}"
                )],
                isError=True
            )
        
        # Devolver el episodio completo con toda la información
        full_episode = {
            "id": str(episode.id),
            "timestamp": episode.timestamp.isoformat(),
            "task": episode.task,
            "context": episode.context,
            "reasoning_trace": {
                "raw_thinking": episode.reasoning_trace.raw_thinking,
                "alternatives_considered": episode.reasoning_trace.alternatives_considered,
                "decision_factors": episode.reasoning_trace.decision_factors,
                "confidence_level": episode.reasoning_trace.confidence_level
            },
            "solution": episode.solution,
            "solution_summary": episode.solution_summary,
            "outcome": episode.outcome,
            "success": episode.success,
            "episode_type": episode.episode_type.value,
            "tags": episode.tags,
            "files_affected": episode.files_affected,
            "lessons_learned": episode.lessons_learned,
            "source_assistant": episode.source_assistant,
            "project_name": episode.project_name,
            # Campos de Forgetting Curve
            "importance_score": episode.importance_score,
            "access_count": episode.access_count,
            "last_accessed": episode.last_accessed.isoformat() if episode.last_accessed else None
        }
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(full_episode, indent=2, ensure_ascii=False)
            )]
        )
    
    async def _onboard_project(self, args: dict) -> CallToolResult:
        """Analizar proyecto y crear episodio de onboarding."""
        from memorytwin.escriba.project_analyzer import onboard_project
        
        project_path = args.get("project_path")
        if not project_path:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="Error: Se requiere project_path"
                )],
                isError=True
            )
        
        result = await onboard_project(
            project_path=project_path,
            project_name=args.get("project_name"),
            source_assistant="mcp-onboarding"
        )
        
        # Resumen del análisis
        analysis = result['analysis']
        summary = {
            "success": True,
            "episode_id": result['episode_id'],
            "project_name": result['project_name'],
            "stack": [s['technology'] for s in analysis['stack']],
            "patterns": [p.get('pattern', p.get('directory', '')) for p in analysis['patterns']],
            "main_dependencies": analysis['dependencies']['main'][:10],
            "conventions": analysis['conventions'],
            "message": f"Onboarding completado. El agente ahora conoce la estructura del proyecto {result['project_name']}."
        }
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(summary, indent=2, ensure_ascii=False)
            )]
        )
    
    async def _get_project_context(self, args: dict) -> CallToolResult:
        """
        Obtiene contexto inteligente del proyecto.
        
        Estrategia híbrida:
        - Si hay pocas memorias (<20): devuelve TODAS
        - Si hay muchas (>=20): devuelve estadísticas + recientes + relevantes
        """
        project_name = args.get("project_name")
        topic = args.get("topic", "")
        
        THRESHOLD = 20  # Umbral para cambiar de estrategia
        
        # Obtener estadísticas base
        stats = self.rag_engine.get_statistics()
        total_episodes = stats.get("total_episodes", 0)
        
        result = {
            "mode": "",
            "total_episodes": total_episodes,
            "statistics": stats
        }
        
        if total_episodes == 0:
            result["mode"] = "empty"
            result["message"] = (
                "No hay memorias registradas aún. "
                "Considera ejecutar onboard_project para crear contexto inicial."
            )
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, ensure_ascii=False)
                )]
            )
        
        if total_episodes < THRESHOLD:
            # Modo FULL: devolver resumen de TODAS las memorias
            result["mode"] = "full_context"
            result["message"] = f"Memoria pequeña ({total_episodes} episodios) - mostrando contexto completo."
            
            # Obtener timeline completo (más recientes primero)
            timeline = self.rag_engine.get_timeline(
                limit=total_episodes,
                project_name=project_name
            )
            
            # Crear resumen compacto de cada episodio
            episodes_summary = []
            for ep in timeline:
                episode_brief = {
                    "id": ep["id"],
                    "type": ep["type"],
                    "task": ep["task"],
                    "summary": ep["summary"],
                    "date": ep["date"],
                    "tags": ep["tags"]
                }
                episodes_summary.append(episode_brief)
            
            result["episodes"] = episodes_summary
            
            # Extraer lecciones agregadas si hay topic
            if topic:
                lessons = self.rag_engine.get_lessons(project_name=project_name)
                lessons_list = lessons if isinstance(lessons, list) else lessons.get("lessons", [])
                result["aggregated_lessons"] = _format_lessons(lessons_list)
        
        else:
            # Modo SMART: estadísticas + recientes + relevantes
            result["mode"] = "smart_context"
            result["message"] = f"Memoria madura ({total_episodes} episodios) - mostrando contexto optimizado."
            
            # 5 más recientes
            recent = self.rag_engine.get_timeline(limit=5, project_name=project_name)
            result["recent_episodes"] = [
                {
                    "id": ep["id"],
                    "type": ep["type"],
                    "task": ep["task"],
                    "summary": ep["summary"],
                    "date": ep["date"],
                    "tags": ep["tags"]
                }
                for ep in recent
            ]
            
            # 5 más relevantes al topic (si se proporciona)
            if topic:
                query = MemoryQuery(
                    query=topic,
                    project_filter=project_name,
                    top_k=5
                )
                relevant_results = self.storage.search_episodes(query)
                result["relevant_episodes"] = [
                    {
                        "id": str(r.episode.id),
                        "type": r.episode.episode_type.value,
                        "task": r.episode.task,
                        "summary": r.episode.solution_summary,
                        "relevance": f"{r.relevance_score:.0%}",
                        "tags": r.episode.tags
                    }
                    for r in relevant_results
                ]
                
                # También incluir lecciones filtradas por topic
                lessons = self.rag_engine.get_lessons(project_name=project_name)
                lessons_list = lessons if isinstance(lessons, list) else lessons.get("lessons", [])
                result["aggregated_lessons"] = _format_lessons(lessons_list)
            else:
                result["tip"] = "Proporciona un 'topic' para obtener episodios semánticamente relevantes."
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False)
            )]
        )
    
    async def run(self):
        """Ejecutar el servidor MCP."""
        logger.info("Iniciando Memory Twin MCP Server...")
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Punto de entrada del servidor MCP."""
    server = MemoryTwinMCPServer()
    await server.run()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
