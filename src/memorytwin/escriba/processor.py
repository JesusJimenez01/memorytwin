"""
Procesador de Pensamientos - LLM para estructurar razonamiento
==============================================================

Utiliza Gemini Flash (u otro LLM) para convertir texto crudo
de "thinking" en episodios estructurados de memoria.
"""

import json
import logging
from typing import Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from memorytwin.config import get_llm_model, get_settings
from memorytwin.models import Episode, EpisodeType, ProcessedInput, ReasoningTrace
from memorytwin.observability import trace_store_memory, _get_langfuse, _is_disabled, flush_traces


# Configurar logging
logger = logging.getLogger("memorytwin.processor")


# Excepciones que merecen retry
RETRYABLE_EXCEPTIONS = (
    Exception,  # TODO: Refinar con excepciones específicas de Gemini
)


# Prompt del sistema para estructurar pensamientos
STRUCTURING_PROMPT = """Eres un asistente especializado en analizar y estructurar el razonamiento técnico de asistentes de IA durante el desarrollo de software.

Tu tarea es convertir texto crudo de "thinking" (razonamiento visible) de un asistente de código en un episodio de memoria estructurado.

ENTRADA:
- Texto de razonamiento del modelo (thinking visible)
- Opcionalmente: prompt original del usuario y cambios de código

SALIDA (JSON estricto):
{
    "task": "Descripción concisa de la tarea o problema abordado",
    "context": "Contexto técnico: archivos, módulos, tecnologías involucradas",
    "reasoning_trace": {
        "raw_thinking": "Resumen del razonamiento principal (máx 500 palabras)",
        "alternatives_considered": ["alternativa 1 descartada", "alternativa 2 descartada"],
        "decision_factors": ["factor 1 que influyó", "factor 2 que influyó"],
        "confidence_level": 0.85
    },
    "solution": "Código o solución implementada (extracto relevante)",
    "solution_summary": "Resumen ejecutivo de la solución en 1-2 oraciones",
    "episode_type": "decision|bug_fix|refactor|feature|optimization|learning|experiment",
    "tags": ["tag1", "tag2", "tag3"],
    "files_affected": ["archivo1.py", "archivo2.ts"],
    "lessons_learned": ["lección 1", "lección 2"]
}

REGLAS:
1. Sé conciso pero completo
2. Extrae TODAS las alternativas consideradas y descartadas
3. Identifica los factores de decisión clave
4. Asigna un nivel de confianza basado en el tono del razonamiento
5. Genera tags relevantes para búsqueda futura
6. Extrae lecciones aprendidas si las hay (errores evitados, patrones descubiertos)
7. SIEMPRE responde con JSON válido, sin texto adicional
"""


class ThoughtProcessor:
    """
    Procesador de pensamientos usando LLM.
    Convierte texto crudo en episodios estructurados.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Inicializar procesador con modelo LLM.
        
        Args:
            api_key: DEPRECATED - ya no se usa, la API key se lee de config.
        """
        # Usar factory centralizada (respuesta JSON)
        self.model = get_llm_model(response_mime_type="application/json")
        
    @trace_store_memory
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def process_thought(
        self,
        raw_input: ProcessedInput,
        project_name: str = "default",
        source_assistant: str = "unknown"
    ) -> Episode:
        """
        Procesar pensamiento crudo y convertirlo en episodio estructurado.
        
        Args:
            raw_input: Input capturado (thinking + contexto opcional)
            project_name: Nombre del proyecto
            source_assistant: Asistente fuente (copilot, claude, etc.)
            
        Returns:
            Episode estructurado listo para almacenamiento
        """
        settings = get_settings()
        
        # Construir prompt con el input
        user_content = self._build_user_prompt(raw_input)
        
        # Trazar generación del LLM
        langfuse = _get_langfuse() if not _is_disabled() else None
        generation = None
        
        try:
            if langfuse:
                generation = langfuse.start_as_current_generation(
                    name="✍️ Escriba - Estructurar Pensamiento",
                    model=settings.llm_model,
                    model_parameters={"temperature": settings.llm_temperature},
                    input={"thinking_text": raw_input.raw_text[:500], "project": project_name}
                ).__enter__()
            
            # Llamar al LLM (interfaz unificada)
            response = await self.model.generate_async(
                [
                    {"role": "user", "parts": [STRUCTURING_PROMPT]},
                    {"role": "model", "parts": ["Entendido. Estoy listo para estructurar el razonamiento técnico en formato JSON."]},
                    {"role": "user", "parts": [user_content]}
                ]
            )
            
            if generation:
                generation.update(output=response.text[:1000])
                
        finally:
            if generation:
                try:
                    generation.end()
                except Exception:
                    pass
            if langfuse:
                flush_traces()
        
        # Parsear respuesta JSON
        try:
            structured_data = json.loads(response.text)
        except json.JSONDecodeError as e:
            # Intentar extraer JSON si hay texto adicional
            import re
            json_match = re.search(r'\{[\s\S]*\}', response.text)
            if json_match:
                structured_data = json.loads(json_match.group())
            else:
                raise ValueError(f"El LLM no devolvió JSON válido: {e}")
        
        # Construir Episode
        episode = self._build_episode(
            structured_data,
            project_name=project_name,
            source_assistant=source_assistant
        )
        
        return episode
    
    def process_thought_sync(
        self,
        raw_input: ProcessedInput,
        project_name: str = "default",
        source_assistant: str = "unknown"
    ) -> Episode:
        """Versión síncrona del procesamiento."""
        import asyncio
        return asyncio.run(
            self.process_thought(raw_input, project_name, source_assistant)
        )
    
    def _build_user_prompt(self, raw_input: ProcessedInput) -> str:
        """Construir prompt de usuario con el input capturado."""
        parts = ["## TEXTO DE RAZONAMIENTO (THINKING):\n"]
        parts.append(raw_input.raw_text)
        
        if raw_input.user_prompt:
            parts.append("\n\n## PROMPT ORIGINAL DEL USUARIO:\n")
            parts.append(raw_input.user_prompt)
            
        if raw_input.code_changes:
            parts.append("\n\n## CAMBIOS DE CÓDIGO:\n```\n")
            parts.append(raw_input.code_changes)
            parts.append("\n```")
            
        parts.append("\n\n---\nEstructura este razonamiento en el formato JSON especificado.")
        
        return "".join(parts)
    
    def _build_episode(
        self,
        data: dict,
        project_name: str,
        source_assistant: str
    ) -> Episode:
        """Construir Episode a partir de datos estructurados."""
        # Parsear reasoning_trace
        rt_data = data.get("reasoning_trace", {})
        reasoning_trace = ReasoningTrace(
            raw_thinking=rt_data.get("raw_thinking", data.get("task", "")),
            alternatives_considered=rt_data.get("alternatives_considered", []),
            decision_factors=rt_data.get("decision_factors", []),
            confidence_level=rt_data.get("confidence_level")
        )
        
        # Parsear episode_type
        episode_type_str = data.get("episode_type", "decision")
        try:
            episode_type = EpisodeType(episode_type_str)
        except ValueError:
            episode_type = EpisodeType.DECISION
            
        return Episode(
            task=data.get("task", "Tarea no especificada"),
            context=data.get("context", "Contexto no especificado"),
            reasoning_trace=reasoning_trace,
            solution=data.get("solution", ""),
            solution_summary=data.get("solution_summary", ""),
            episode_type=episode_type,
            tags=data.get("tags", []),
            files_affected=data.get("files_affected", []),
            lessons_learned=data.get("lessons_learned", []),
            project_name=project_name,
            source_assistant=source_assistant
        )


# Función de conveniencia para uso simple
async def process_thinking_text(
    thinking_text: str,
    user_prompt: Optional[str] = None,
    code_changes: Optional[str] = None,
    project_name: str = "default",
    source_assistant: str = "unknown"
) -> Episode:
    """
    Función de conveniencia para procesar texto de thinking.
    
    Args:
        thinking_text: Texto de razonamiento del modelo
        user_prompt: Prompt original del usuario (opcional)
        code_changes: Cambios de código asociados (opcional)
        project_name: Nombre del proyecto
        source_assistant: Asistente de código fuente
        
    Returns:
        Episode estructurado
    """
    processor = ThoughtProcessor()
    raw_input = ProcessedInput(
        raw_text=thinking_text,
        user_prompt=user_prompt,
        code_changes=code_changes,
        source="manual"
    )
    return await processor.process_thought(raw_input, project_name, source_assistant)
