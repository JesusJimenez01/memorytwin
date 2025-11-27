"""
Motor RAG para el Oráculo
=========================

Implementa Retrieval-Augmented Generation sobre la base
de datos de memorias episódicas para responder preguntas
contextuales sobre decisiones técnicas.

Ahora incluye soporte para MetaMemories (conocimiento consolidado).
"""

from typing import Optional

import google.generativeai as genai

from memorytwin.config import get_settings
from memorytwin.models import MemoryQuery, MemorySearchResult, MetaMemory, MetaMemorySearchResult
from memorytwin.escriba.storage import MemoryStorage


# Prompt del sistema para el Oráculo
ORACLE_SYSTEM_PROMPT = """Eres el Oráculo del Memory Twin, un asistente especializado en responder preguntas sobre la historia técnica y las decisiones de desarrollo de un proyecto de software.

Tu conocimiento proviene de dos fuentes:

1. **META-MEMORIAS**: Conocimiento consolidado de múltiples episodios relacionados. Representan patrones, lecciones y mejores prácticas identificadas automáticamente. Son más confiables y generales.

2. **EPISODIOS**: Memorias individuales que documentan el razonamiento ("thinking") de asistentes de IA durante el desarrollo. Contienen:
   - La tarea o problema abordado
   - El contexto técnico
   - La traza de razonamiento (alternativas consideradas, factores de decisión)
   - La solución implementada
   - Lecciones aprendidas

INSTRUCCIONES:
1. Prioriza las META-MEMORIAS si están disponibles (son conocimiento consolidado)
2. Complementa con EPISODIOS para detalles específicos
3. Si la información no está en la memoria, indica que no hay registros sobre ese tema
4. Cita las fuentes (meta-memorias o episodios) cuando sea útil
5. Explica el "porqué" detrás de las decisiones, no solo el "qué"
6. Si hay lecciones aprendidas o mejores prácticas relevantes, inclúyelas
7. Sé conciso pero completo
8. Usa formato Markdown para mejor legibilidad

FORMATO DE RESPUESTA:
- Respuesta directa a la pregunta
- Contexto relevante de las fuentes
- Lecciones aprendidas / mejores prácticas aplicables
- Referencias a meta-memorias o episodios específicos
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
        top_k: int = 5,
        include_meta_memories: bool = True
    ) -> dict:
        """
        Realizar una consulta RAG sobre las memorias.
        
        Busca primero en MetaMemories (conocimiento consolidado)
        y luego complementa con episodios individuales.
        
        Args:
            question: Pregunta del usuario
            project_name: Filtrar por proyecto específico
            top_k: Número de resultados a recuperar
            include_meta_memories: Si incluir meta-memorias en la búsqueda
            
        Returns:
            Dict con respuesta, episodios usados, meta-memorias y metadatos
        """
        meta_results = []
        
        # Buscar meta-memorias primero (conocimiento consolidado)
        if include_meta_memories:
            meta_results = self.storage.search_meta_memories(
                query=question,
                project_name=project_name,
                top_k=min(3, top_k)  # Máximo 3 meta-memorias
            )
        
        # Buscar episodios relevantes
        memory_query = MemoryQuery(
            query=question,
            project_filter=project_name,
            top_k=top_k
        )
        
        search_results = self.storage.search_episodes(memory_query)
        
        if not search_results and not meta_results:
            return {
                "answer": "No encontré episodios de memoria ni conocimiento consolidado "
                         "relacionados con tu pregunta. "
                         "Es posible que este tema no haya sido documentado aún.",
                "episodes_used": [],
                "meta_memories_used": [],
                "relevance_scores": [],
                "context_provided": False
            }
        
        # Construir contexto combinando meta-memorias y episodios
        context = self._build_combined_context(meta_results, search_results)
        
        # Generar respuesta
        answer = await self._generate_answer(question, context)
        
        return {
            "answer": answer,
            "episodes_used": [r.episode for r in search_results],
            "meta_memories_used": [r.meta_memory for r in meta_results],
            "relevance_scores": [r.relevance_score for r in search_results],
            "meta_relevance_scores": [r.relevance_score for r in meta_results],
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
    
    def _build_combined_context(
        self,
        meta_results: list[MetaMemorySearchResult],
        episode_results: list[MemorySearchResult]
    ) -> str:
        """
        Construir contexto combinando meta-memorias y episodios.
        
        Las meta-memorias van primero (mayor prioridad).
        """
        context_parts = []
        
        # Primero las meta-memorias (conocimiento consolidado)
        if meta_results:
            context_parts.append("## META-MEMORIAS (Conocimiento Consolidado)\n")
            context_parts.append("*Estas son lecciones consolidadas de múltiples episodios relacionados.*\n")
            
            for i, result in enumerate(meta_results, 1):
                mm = result.meta_memory
                
                context_parts.append(f"""
### Meta-Memoria {i} (Relevancia: {result.relevance_score:.0%} | Confianza: {mm.confidence:.0%})
- **Patrón**: {mm.pattern_summary}
- **Basado en**: {mm.episode_count} episodios
- **Tecnologías**: {', '.join(mm.technologies) or 'No especificadas'}

**Descripción del Patrón**:
{mm.pattern}

**Lecciones Aprendidas**:
{chr(10).join(f'• {l}' for l in mm.lessons) or '• Ninguna documentada'}

**Mejores Prácticas**:
{chr(10).join(f'• {p}' for p in mm.best_practices) or '• Ninguna documentada'}

**Anti-patrones a Evitar**:
{chr(10).join(f'• {a}' for a in mm.antipatterns) or '• Ninguno documentado'}

**Excepciones/Casos Especiales**:
{chr(10).join(f'• {e}' for e in mm.exceptions) or '• Ninguna documentada'}

**Contextos Aplicables**: {', '.join(mm.contexts) or 'General'}
---
""")
        
        # Luego los episodios individuales
        if episode_results:
            context_parts.append("\n## EPISODIOS INDIVIDUALES\n")
            context_parts.append("*Detalles específicos de decisiones particulares.*\n")
            context_parts.append(self._build_context(episode_results))
        
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
