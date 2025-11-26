"""
Motor RAG para el Oráculo
=========================

Implementa Retrieval-Augmented Generation sobre la base
de datos de memorias episódicas para responder preguntas
contextuales sobre decisiones técnicas.
"""

from typing import Optional

import google.generativeai as genai

from memorytwin.config import get_settings
from memorytwin.models import MemoryQuery, MemorySearchResult
from memorytwin.escriba.storage import MemoryStorage


# Prompt del sistema para el Oráculo
ORACLE_SYSTEM_PROMPT = """Eres el Oráculo del Memory Twin, un asistente especializado en responder preguntas sobre la historia técnica y las decisiones de desarrollo de un proyecto de software.

Tu conocimiento proviene de episodios de memoria que documentan el razonamiento ("thinking") de asistentes de IA durante el desarrollo. Cada episodio contiene:
- La tarea o problema abordado
- El contexto técnico
- La traza de razonamiento (alternativas consideradas, factores de decisión)
- La solución implementada
- Lecciones aprendidas

INSTRUCCIONES:
1. Responde basándote ÚNICAMENTE en los episodios de memoria proporcionados
2. Si la información no está en la memoria, indica que no hay registros sobre ese tema
3. Cita los episodios relevantes mencionando la fecha y el ID cuando sea útil
4. Explica el "porqué" detrás de las decisiones, no solo el "qué"
5. Si hay lecciones aprendidas relevantes, inclúyelas
6. Sé conciso pero completo
7. Usa formato Markdown para mejor legibilidad

FORMATO DE RESPUESTA:
- Respuesta directa a la pregunta
- Contexto relevante de los episodios
- Lecciones aprendidas aplicables (si las hay)
- Referencias a episodios específicos
"""


class RAGEngine:
    """
    Motor de Retrieval-Augmented Generation para consultas
    sobre memorias episódicas.
    """
    
    def __init__(
        self,
        storage: Optional[MemoryStorage] = None,
        api_key: Optional[str] = None
    ):
        """
        Inicializar el motor RAG.
        
        Args:
            storage: Almacenamiento de memoria (se crea uno si no se provee)
            api_key: API key para el LLM (usa la de configuración si no se provee)
        """
        self.storage = storage or MemoryStorage()
        
        settings = get_settings()
        self.api_key = api_key or settings.google_api_key
        
        if not self.api_key:
            raise ValueError(
                "Se requiere GOOGLE_API_KEY para el motor RAG. "
                "Configúrala en .env o pásala como parámetro."
            )
        
        # Configurar Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name=settings.llm_model,
            generation_config={
                "temperature": 0.4,  # Un poco más creativo para respuestas
                "max_output_tokens": 2048,
            }
        )
        
    async def query(
        self,
        question: str,
        project_name: Optional[str] = None,
        top_k: int = 5
    ) -> dict:
        """
        Realizar una consulta RAG sobre las memorias.
        
        Args:
            question: Pregunta del usuario
            project_name: Filtrar por proyecto específico
            top_k: Número de episodios a recuperar
            
        Returns:
            Dict con respuesta, episodios usados y metadatos
        """
        # Buscar episodios relevantes
        memory_query = MemoryQuery(
            query=question,
            project_filter=project_name,
            top_k=top_k
        )
        
        search_results = self.storage.search_episodes(memory_query)
        
        if not search_results:
            return {
                "answer": "No encontré episodios de memoria relacionados con tu pregunta. "
                         "Es posible que este tema no haya sido documentado aún.",
                "episodes_used": [],
                "relevance_scores": [],
                "context_provided": False
            }
        
        # Construir contexto para el LLM
        context = self._build_context(search_results)
        
        # Generar respuesta
        answer = await self._generate_answer(question, context)
        
        return {
            "answer": answer,
            "episodes_used": [r.episode for r in search_results],
            "relevance_scores": [r.relevance_score for r in search_results],
            "context_provided": True
        }
    
    def query_sync(
        self,
        question: str,
        project_name: Optional[str] = None,
        top_k: int = 5
    ) -> dict:
        """Versión síncrona de query."""
        import asyncio
        return asyncio.run(self.query(question, project_name, top_k))
    
    def _build_context(self, results: list[MemorySearchResult]) -> str:
        """Construir contexto de episodios para el LLM."""
        context_parts = ["## EPISODIOS DE MEMORIA RELEVANTES\n"]
        
        for i, result in enumerate(results, 1):
            ep = result.episode
            
            context_parts.append(f"""
### Episodio {i} (Relevancia: {result.relevance_score:.0%})
- **ID**: {ep.id}
- **Fecha**: {ep.timestamp.strftime('%Y-%m-%d %H:%M')}
- **Tipo**: {ep.episode_type.value}
- **Proyecto**: {ep.project_name}
- **Asistente**: {ep.source_assistant}

**Tarea**: {ep.task}

**Contexto**: {ep.context}

**Razonamiento**:
{ep.reasoning_trace.raw_thinking}

**Alternativas consideradas**: {', '.join(ep.reasoning_trace.alternatives_considered) or 'No documentadas'}

**Factores de decisión**: {', '.join(ep.reasoning_trace.decision_factors) or 'No documentados'}

**Solución**: {ep.solution_summary}

**Lecciones aprendidas**: {', '.join(ep.lessons_learned) or 'Ninguna documentada'}

**Tags**: {', '.join(ep.tags)}
---
""")
        
        return "\n".join(context_parts)
    
    async def _generate_answer(self, question: str, context: str) -> str:
        """Generar respuesta usando el LLM."""
        prompt = f"""## CONTEXTO DE MEMORIA
{context}

## PREGUNTA DEL USUARIO
{question}

## TU RESPUESTA (en español, usando Markdown)
"""
        
        response = await self.model.generate_content_async(
            [
                {"role": "user", "parts": [ORACLE_SYSTEM_PROMPT]},
                {"role": "model", "parts": ["Entendido. Estoy listo para responder preguntas sobre la memoria técnica del proyecto basándome únicamente en los episodios proporcionados."]},
                {"role": "user", "parts": [prompt]}
            ]
        )
        
        return response.text
    
    def get_timeline(
        self,
        project_name: Optional[str] = None,
        limit: int = 50
    ) -> list:
        """
        Obtener timeline de decisiones para visualización.
        
        Args:
            project_name: Filtrar por proyecto
            limit: Número máximo de episodios
            
        Returns:
            Lista de episodios ordenados cronológicamente
        """
        episodes = self.storage.get_timeline(
            project_name=project_name,
            limit=limit
        )
        
        # Formatear para visualización
        timeline = []
        for ep in episodes:
            timeline.append({
                "id": str(ep.id),
                "timestamp": ep.timestamp.isoformat(),
                "date": ep.timestamp.strftime("%Y-%m-%d"),
                "time": ep.timestamp.strftime("%H:%M"),
                "task": ep.task,
                "type": ep.episode_type.value,
                "summary": ep.solution_summary,
                "tags": ep.tags,
                "assistant": ep.source_assistant,
                "success": ep.success
            })
            
        return timeline
    
    def get_lessons(
        self,
        project_name: Optional[str] = None,
        tags: Optional[list[str]] = None
    ) -> list:
        """
        Obtener lecciones aprendidas agregadas.
        
        Args:
            project_name: Filtrar por proyecto
            tags: Filtrar por tags
            
        Returns:
            Lista de lecciones con contexto
        """
        return self.storage.get_lessons_learned(
            project_name=project_name,
            tags=tags
        )
    
    def get_statistics(self, project_name: Optional[str] = None) -> dict:
        """Obtener estadísticas de la memoria."""
        return self.storage.get_statistics(project_name)
