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
                        "⭐ HERRAMIENTA PRINCIPAL - USAR SIEMPRE AL INICIO DE CADA TAREA.\n\n"
                        "Obtiene contexto inteligente con PRIORIZACIÓN:\n"
                        "0. ⚠️ ANTIPATTERNS: Advertencias de errores previos (si hay relevantes)\n"
                        "1. META-MEMORIAS: Conocimiento consolidado y patrones\n"
                        "2. EPISODIOS: Decisiones individuales relevantes\n\n"
                        "Si hay WARNINGS de antipatterns, DEBES revisarlos antes de proceder.\n"
                        "Usa include_reasoning=true para obtener el razonamiento completo."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Tema o palabras clave de la tarea actual (para búsqueda semántica)"
                            },
                            "project_name": {
                                "type": "string",
                                "description": "Filtrar por proyecto específico (opcional)"
                            },
                            "include_reasoning": {
                                "type": "boolean",
                                "description": "Si true, incluye raw_thinking completo de episodios relevantes (más tokens pero más contexto)"
                            }
                        },
                        "required": []
                    }
                ),
                Tool(
                    name="consolidate_memories",
                    description=(
                        "Consolida episodios similares en META-MEMORIAS usando clustering y LLM. "
                        "Las meta-memorias agrupan patrones recurrentes, lecciones aprendidas y mejores prácticas, "
                        "permitiendo acceso rápido a conocimiento consolidado sin buscar en episodios individuales.\n\n"
                        "Usar cuando:\n"
                        "- El sistema sugiere consolidación (indicador en get_project_context)\n"
                        "- Hay muchos episodios (>20) sin consolidar\n"
                        "- Episodios con alto access_count indican patrones frecuentes\n\n"
                        "Resultado: Meta-memorias con patrones, lecciones, mejores prácticas y antipatrones."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": {
                                "type": "string",
                                "description": "Nombre del proyecto a consolidar (requerido)"
                            },
                            "min_cluster_size": {
                                "type": "integer",
                                "description": "Mínimo de episodios para formar un cluster (default: 3)"
                            }
                        },
                        "required": ["project_name"]
                    }
                ),
                Tool(
                    name="check_consolidation_status",
                    description=(
                        "Verifica si el proyecto necesita consolidación de memorias. "
                        "Analiza episodios con alto uso (access_count) y cantidad de episodios sin consolidar. "
                        "Útil para decidir si ejecutar consolidate_memories."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_name": {
                                "type": "string",
                                "description": "Nombre del proyecto a verificar (opcional)"
                            }
                        },
                        "required": []
                    }
                ),
                Tool(
                    name="mark_episode",
                    description=(
                        "Marca un episodio con flags especiales:\n"
                        "- is_antipattern=true: Este episodio representa un ERROR o algo que NO se debe hacer. "
                        "Se mostrará como ADVERTENCIA en futuras consultas.\n"
                        "- is_critical=true: Este episodio es crítico y debe priorizarse en búsquedas.\n"
                        "- superseded_by: UUID del episodio que reemplaza a este.\n"
                        "- deprecation_reason: Razón por la que este episodio ya no aplica.\n\n"
                        "Usar después de descubrir que una solución previa fue incorrecta o para destacar "
                        "decisiones importantes."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "episode_id": {
                                "type": "string",
                                "description": "UUID del episodio a marcar"
                            },
                            "is_antipattern": {
                                "type": "boolean",
                                "description": "Marcar como antipatrón (algo que NO se debe hacer)"
                            },
                            "is_critical": {
                                "type": "boolean",
                                "description": "Marcar como crítico (alta prioridad)"
                            },
                            "superseded_by": {
                                "type": "string",
                                "description": "UUID del episodio que reemplaza a este"
                            },
                            "deprecation_reason": {
                                "type": "string",
                                "description": "Razón por la que este episodio ya no aplica"
                            }
                        },
                        "required": ["episode_id"]
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
                elif name == "consolidate_memories":
                    return await self._consolidate_memories(arguments)
                elif name == "check_consolidation_status":
                    return await self._check_consolidation_status(arguments)
                elif name == "mark_episode":
                    return await self._mark_episode(arguments)
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
        
        Estrategia híbrida con priorización:
        0. PRIMERO: Buscar ANTIPATTERNS relevantes (advertencias)
        1. Luego META-MEMORIAS (conocimiento consolidado)
        2. Finalmente EPISODIOS individuales
        
        Args:
            project_name: Filtrar por proyecto
            topic: Tema para búsqueda semántica
            include_reasoning: Si True, incluye raw_thinking completo (más tokens)
        """
        project_name = args.get("project_name")
        topic = args.get("topic", "")
        include_reasoning = args.get("include_reasoning", False)
        
        THRESHOLD = 20  # Umbral para cambiar de estrategia
        
        # Obtener estadísticas base
        stats = self.rag_engine.get_statistics()
        total_episodes = stats.get("total_episodes", 0)
        
        # Obtener estadísticas de meta-memorias
        meta_stats = self.storage.get_meta_memory_statistics(project_name)
        
        result = {
            "mode": "",
            "total_episodes": total_episodes,
            "total_meta_memories": meta_stats.get("total_meta_memories", 0),
            "statistics": stats,
            "meta_statistics": meta_stats
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
        
        # =====================================================================
        # PRIORIDAD 0: ANTIPATTERNS (ADVERTENCIAS CRÍTICAS)
        # Buscar episodios marcados como antipattern relevantes al topic
        # Estos DEBEN mostrarse primero como advertencias
        # =====================================================================
        warnings = []
        if topic:
            query = MemoryQuery(
                query=topic,
                project_filter=project_name,
                top_k=10
            )
            all_results = self.storage.search_episodes(query)
            for r in all_results:
                if getattr(r.episode, 'is_antipattern', False):
                    warning = {
                        "type": "ANTIPATTERN",
                        "severity": "HIGH",
                        "task": r.episode.task,
                        "lesson": r.episode.lessons_learned[0] if r.episode.lessons_learned else "Evitar este enfoque",
                        "relevance": f"{r.relevance_score:.0%}"
                    }
                    if include_reasoning:
                        warning["reasoning"] = r.episode.reasoning_trace.raw_thinking
                    warnings.append(warning)
        
        if warnings:
            result["⚠️ WARNINGS"] = warnings
            result["warning_note"] = (
                "⛔ ATENCIÓN: Se encontraron antipatterns relevantes. "
                "Revisa estas advertencias ANTES de proceder. "
                "Si decides ignorarlas, JUSTIFICA tu decisión."
            )
        
        # =====================================================================
        # PRIORIDAD 1: META-MEMORIAS (Conocimiento Consolidado)
        # Siempre incluir meta-memorias si existen (acceso rápido a patrones)
        # =====================================================================
        meta_memories_included = []
        if meta_stats.get("total_meta_memories", 0) > 0:
            if topic:
                # Búsqueda semántica en meta-memorias
                meta_results = self.storage.search_meta_memories(
                    query=topic,
                    project_name=project_name,
                    top_k=3
                )
                meta_memories_included = [
                    {
                        "id": str(r.meta_memory.id),
                        "pattern": r.meta_memory.pattern_summary,
                        "lessons": r.meta_memory.lessons[:3],
                        "best_practices": r.meta_memory.best_practices[:2],
                        "technologies": r.meta_memory.technologies,
                        "episode_count": r.meta_memory.episode_count,
                        "confidence": f"{r.meta_memory.confidence:.0%}",
                        "relevance": f"{r.relevance_score:.0%}"
                    }
                    for r in meta_results
                ]
            else:
                # Sin topic, obtener meta-memorias más recientes
                recent_metas = self.storage.get_meta_memories_by_project(
                    project_name=project_name or "default",
                    limit=3
                )
                meta_memories_included = [
                    {
                        "id": str(mm.id),
                        "pattern": mm.pattern_summary,
                        "lessons": mm.lessons[:3],
                        "best_practices": mm.best_practices[:2],
                        "technologies": mm.technologies,
                        "episode_count": mm.episode_count,
                        "confidence": f"{mm.confidence:.0%}"
                    }
                    for mm in recent_metas
                ]
        
        if meta_memories_included:
            result["meta_memories"] = meta_memories_included
            result["meta_memory_note"] = (
                "⭐ META-MEMORIAS: Conocimiento consolidado de múltiples episodios. "
                "Priorizar esta información para patrones generales y mejores prácticas."
            )
        
        # =====================================================================
        # VERIFICAR NECESIDAD DE CONSOLIDACIÓN
        # =====================================================================
        consolidation_check = self.storage.check_consolidation_needed(project_name)
        if consolidation_check.get("should_consolidate"):
            result["consolidation_recommendation"] = {
                "should_consolidate": True,
                "reason": f"Hay {consolidation_check['hot_episodes_count']} episodios con alto uso "
                         f"o {consolidation_check['estimated_unconsolidated']} sin consolidar",
                "suggestion": "Considera ejecutar consolidación con: mt consolidate --project <nombre>"
            }
        
        # =====================================================================
        # PRIORIDAD 2: EPISODIOS INDIVIDUALES (según modo)
        # =====================================================================
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
                relevant_episodes = []
                for r in relevant_results:
                    ep_data = {
                        "id": str(r.episode.id),
                        "type": r.episode.episode_type.value,
                        "task": r.episode.task,
                        "summary": r.episode.solution_summary,
                        "relevance": f"{r.relevance_score:.0%}",
                        "tags": r.episode.tags,
                        "lessons": r.episode.lessons_learned,
                        "is_critical": getattr(r.episode, 'is_critical', False)
                    }
                    # Incluir razonamiento completo si se solicita
                    if include_reasoning:
                        ep_data["reasoning"] = r.episode.reasoning_trace.raw_thinking
                        ep_data["alternatives"] = r.episode.reasoning_trace.alternatives_considered
                        ep_data["decision_factors"] = r.episode.reasoning_trace.decision_factors
                    relevant_episodes.append(ep_data)
                result["relevant_episodes"] = relevant_episodes
                
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
    
    async def _consolidate_memories(self, args: dict) -> CallToolResult:
        """
        Consolidar episodios similares en meta-memorias.
        
        Usa clustering DBSCAN + LLM para sintetizar conocimiento.
        """
        from memorytwin.consolidation import MemoryConsolidator
        
        project_name = args.get("project_name")
        if not project_name:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="Error: Se requiere project_name para consolidar memorias"
                )],
                isError=True
            )
        
        min_cluster_size = args.get("min_cluster_size", 3)
        
        try:
            # Verificar que hay suficientes episodios
            stats = self.storage.get_statistics(project_name)
            total_episodes = stats['total_episodes']
            
            if total_episodes < min_cluster_size:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=json.dumps({
                            "success": False,
                            "message": f"El proyecto '{project_name}' solo tiene {total_episodes} episodios. "
                                      f"Se necesitan al menos {min_cluster_size} para consolidar.",
                            "total_episodes": total_episodes,
                            "min_required": min_cluster_size
                        }, indent=2, ensure_ascii=False)
                    )]
                )
            
            # Ejecutar consolidación
            consolidator = MemoryConsolidator(
                storage=self.storage,
                min_cluster_size=min_cluster_size
            )
            
            meta_memories = consolidator.consolidate_project(project_name)
            
            if not meta_memories:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=json.dumps({
                            "success": False,
                            "message": "No se encontraron clusters suficientemente grandes para consolidar. "
                                      "Intenta con un min_cluster_size menor.",
                            "total_episodes": total_episodes,
                            "min_cluster_size": min_cluster_size,
                            "suggestion": "Los episodios pueden ser muy diversos semánticamente"
                        }, indent=2, ensure_ascii=False)
                    )]
                )
            
            # Resumen de meta-memorias generadas
            result = {
                "success": True,
                "message": f"¡Consolidación completada! Se generaron {len(meta_memories)} meta-memorias.",
                "meta_memories_generated": len(meta_memories),
                "episodes_consolidated": sum(mm.episode_count for mm in meta_memories),
                "meta_memories": [
                    {
                        "id": str(mm.id),
                        "pattern": mm.pattern_summary,
                        "lessons_count": len(mm.lessons),
                        "best_practices_count": len(mm.best_practices),
                        "episode_count": mm.episode_count,
                        "confidence": f"{mm.confidence:.0%}",
                        "technologies": mm.technologies
                    }
                    for mm in meta_memories
                ]
            }
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, ensure_ascii=False)
                )]
            )
            
        except Exception as e:
            logger.error(f"Error en consolidación: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error durante consolidación: {str(e)}"
                )],
                isError=True
            )
    
    async def _mark_episode(self, args: dict) -> CallToolResult:
        """
        Marcar un episodio con flags especiales (antipattern, critical, superseded, deprecated).
        """
        episode_id = args.get("episode_id")
        is_antipattern = args.get("is_antipattern")
        is_critical = args.get("is_critical")
        superseded_by = args.get("superseded_by")
        deprecation_reason = args.get("deprecation_reason")
        
        if not episode_id:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="Error: episode_id es requerido"
                )],
                isError=True
            )
        
        # Obtener episodio actual
        episode = self.storage.get_episode_by_id(episode_id)
        if not episode:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error: No se encontró episodio con ID {episode_id}"
                )],
                isError=True
            )
        
        # Actualizar flags
        updates = {}
        if is_antipattern is not None:
            updates["is_antipattern"] = is_antipattern
        if is_critical is not None:
            updates["is_critical"] = is_critical
        if superseded_by is not None:
            updates["superseded_by"] = superseded_by
        if deprecation_reason is not None:
            updates["deprecation_reason"] = deprecation_reason
        
        if not updates:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="No se especificaron cambios (is_antipattern o is_critical)"
                )],
                isError=True
            )
        
        # Aplicar actualizaciones
        success = self.storage.update_episode_flags(episode_id, updates)
        
        if success:
            result = {
                "success": True,
                "episode_id": episode_id,
                "task": episode.task,
                "updates_applied": updates,
                "message": ""
            }
            if updates.get("is_antipattern"):
                result["message"] = (
                    "⚠️ Episodio marcado como ANTIPATTERN. "
                    "Se mostrará como advertencia en futuras consultas relevantes."
                )
            if updates.get("is_critical"):
                result["message"] += (
                    "⭐ Episodio marcado como CRÍTICO. "
                    "Tendrá prioridad en búsquedas."
                )
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, ensure_ascii=False)
                )]
            )
        else:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error al actualizar episodio {episode_id}"
                )],
                isError=True
            )

    async def _check_consolidation_status(self, args: dict) -> CallToolResult:
        """
        Verificar si se necesita consolidación.
        
        Analiza access_count de episodios y cantidad sin consolidar.
        """
        project_name = args.get("project_name")
        
        status = self.storage.check_consolidation_needed(project_name)
        
        # Añadir recomendación legible
        if status["should_consolidate"]:
            status["recommendation"] = (
                "✅ SE RECOMIENDA CONSOLIDAR: "
                f"Hay {status['hot_episodes_count']} episodios con alto uso "
                f"y aproximadamente {status['estimated_unconsolidated']} sin consolidar. "
                "Ejecuta consolidate_memories para generar meta-memorias."
            )
        else:
            status["recommendation"] = (
                "⏸️ NO ES NECESARIO CONSOLIDAR AÚN: "
                f"Solo hay {status['total_episodes']} episodios, "
                f"con {status['hot_episodes_count']} de alto uso. "
                "El sistema funcionará bien con episodios individuales por ahora."
            )
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(status, indent=2, ensure_ascii=False)
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
