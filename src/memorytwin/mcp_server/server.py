"""
Servidor MCP para Memory Twin
=============================

Implementa el Model Context Protocol para exponer las
capacidades del Escriba y Oráculo a clientes compatibles.
"""

import json
import logging
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)

from memorytwin.config import get_settings
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
