"""
Consolidación de Memorias
=========================

Implementa el proceso de consolidación de episodios relacionados
en meta-memorias, siguiendo un enfoque inspirado en la consolidación
de la memoria humana durante el sueño.

El proceso:
1. Agrupa episodios similares usando clustering por embeddings
2. Para cada cluster, usa un LLM para sintetizar el conocimiento
3. Genera una MetaMemory con patrones, lecciones y excepciones
"""

import json
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import google.generativeai as genai
import numpy as np
from sklearn.cluster import DBSCAN

from memorytwin.config import get_settings
from memorytwin.models import Episode, MetaMemory
from memorytwin.escriba.storage import MemoryStorage


# Prompt para síntesis de episodios en meta-memoria
CONSOLIDATION_PROMPT = """Eres un experto en sintetizar conocimiento técnico. 
Analiza los siguientes episodios de memoria relacionados y genera una meta-memoria consolidada.

EPISODIOS A CONSOLIDAR:
{episodes_text}

---

INSTRUCCIONES:
1. Identifica el PATRÓN COMÚN que conecta estos episodios
2. Extrae las LECCIONES APRENDIDAS más importantes (máximo 5)
3. Identifica MEJORES PRÁCTICAS derivadas (máximo 3)
4. Detecta ANTI-PATRONES o errores comunes a evitar (máximo 3)
5. Lista EXCEPCIONES donde el patrón no aplica (máximo 3)
6. Identifica CASOS LÍMITE descubiertos (máximo 2)
7. Define los CONTEXTOS donde este conocimiento aplica
8. Lista las TECNOLOGÍAS involucradas

RESPONDE EN JSON con esta estructura exacta:
{{
    "pattern": "Descripción detallada del patrón identificado (2-3 oraciones)",
    "pattern_summary": "Resumen ejecutivo en 1 oración",
    "lessons": ["lección 1", "lección 2", ...],
    "best_practices": ["práctica 1", ...],
    "antipatterns": ["antipatrón 1", ...],
    "exceptions": ["excepción 1", ...],
    "edge_cases": ["caso límite 1", ...],
    "contexts": ["contexto 1", ...],
    "technologies": ["tech1", "tech2", ...],
    "coherence_score": 0.8
}}

El coherence_score indica qué tan relacionados están los episodios (0.0-1.0).
Si los episodios son muy diversos, el score será bajo.

IMPORTANTE: Responde SOLO con el JSON, sin explicaciones adicionales.
"""


def format_episode_for_consolidation(episode: Episode) -> str:
    """Formatear un episodio para incluirlo en el prompt de consolidación."""
    return f"""
### Episodio {episode.id}
- **Fecha**: {episode.timestamp.strftime('%Y-%m-%d')}
- **Tarea**: {episode.task}
- **Contexto**: {episode.context}
- **Razonamiento**: {episode.reasoning_trace.raw_thinking[:500]}...
- **Solución**: {episode.solution_summary}
- **Lecciones**: {', '.join(episode.lessons_learned) if episode.lessons_learned else 'N/A'}
- **Tags**: {', '.join(episode.tags) if episode.tags else 'N/A'}
"""


class MemoryConsolidator:
    """
    Consolida episodios relacionados en meta-memorias.
    
    Usa clustering por embeddings para agrupar episodios similares
    y un LLM para sintetizar el conocimiento consolidado.
    """
    
    def __init__(
        self,
        storage: Optional[MemoryStorage] = None,
        api_key: Optional[str] = None,
        min_cluster_size: int = 3,
        cluster_eps: float = 0.5
    ):
        """
        Inicializar el consolidador.
        
        Args:
            storage: Almacenamiento de memoria
            api_key: API key para el LLM
            min_cluster_size: Mínimo de episodios para formar cluster
            cluster_eps: Radio máximo para clustering DBSCAN
        """
        self.storage = storage or MemoryStorage()
        
        settings = get_settings()
        self.api_key = api_key or settings.google_api_key
        
        if not self.api_key:
            raise ValueError(
                "Se requiere GOOGLE_API_KEY para consolidación. "
                "Configúrala en .env o pásala como parámetro."
            )
        
        # Configurar Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name=settings.llm_model,
            generation_config={
                "temperature": 0.3,  # Más determinístico para síntesis
                "max_output_tokens": 2048,
            }
        )
        
        self.min_cluster_size = min_cluster_size
        self.cluster_eps = cluster_eps
    
    def consolidate_project(
        self,
        project_name: str,
        force: bool = False
    ) -> list[MetaMemory]:
        """
        Consolidar episodios de un proyecto en meta-memorias.
        
        Args:
            project_name: Nombre del proyecto
            force: Si True, reconsolida incluso episodios ya consolidados
            
        Returns:
            Lista de meta-memorias generadas
        """
        # Obtener episodios del proyecto
        episodes = self.storage.get_episodes_by_project(project_name, limit=200)
        
        if len(episodes) < self.min_cluster_size:
            return []
        
        # Obtener embeddings de ChromaDB
        embeddings, episode_ids = self._get_episode_embeddings(episodes)
        
        if len(embeddings) < self.min_cluster_size:
            return []
        
        # Clustering
        clusters = self._cluster_episodes(embeddings, episode_ids)
        
        # Generar meta-memorias para cada cluster
        meta_memories = []
        for cluster_episode_ids in clusters:
            # Obtener episodios del cluster
            cluster_episodes = [
                ep for ep in episodes 
                if str(ep.id) in cluster_episode_ids
            ]
            
            if len(cluster_episodes) >= self.min_cluster_size:
                meta_memory = self._synthesize_cluster(
                    cluster_episodes, 
                    project_name
                )
                if meta_memory:
                    # Almacenar
                    self.storage.store_meta_memory(meta_memory)
                    meta_memories.append(meta_memory)
        
        return meta_memories
    
    def _get_episode_embeddings(
        self, 
        episodes: list[Episode]
    ) -> tuple[np.ndarray, list[str]]:
        """Obtener embeddings de episodios desde ChromaDB."""
        episode_ids = [str(ep.id) for ep in episodes]
        
        # Obtener embeddings de ChromaDB
        result = self.storage.collection.get(
            ids=episode_ids,
            include=["embeddings"]
        )
        
        if not result["embeddings"]:
            return np.array([]), []
        
        embeddings = np.array(result["embeddings"])
        valid_ids = result["ids"]
        
        return embeddings, valid_ids
    
    def _cluster_episodes(
        self, 
        embeddings: np.ndarray,
        episode_ids: list[str]
    ) -> list[list[str]]:
        """
        Agrupar episodios por similitud usando DBSCAN.
        
        DBSCAN es ideal porque:
        - No requiere especificar número de clusters
        - Puede detectar clusters de forma arbitraria
        - Identifica outliers (episodios únicos)
        """
        # Normalizar embeddings para usar distancia coseno
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)
        
        # DBSCAN con distancia coseno (convertida a distancia)
        clustering = DBSCAN(
            eps=self.cluster_eps,
            min_samples=self.min_cluster_size,
            metric='cosine'
        ).fit(normalized)
        
        # Agrupar IDs por label de cluster
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label == -1:  # Outlier
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(episode_ids[idx])
        
        return list(clusters.values())
    
    def _synthesize_cluster(
        self,
        episodes: list[Episode],
        project_name: str
    ) -> Optional[MetaMemory]:
        """
        Usar LLM para sintetizar un cluster de episodios.
        
        Args:
            episodes: Episodios del cluster
            project_name: Nombre del proyecto
            
        Returns:
            MetaMemory generada o None si falla
        """
        # Formatear episodios para el prompt
        episodes_text = "\n---\n".join(
            format_episode_for_consolidation(ep) for ep in episodes
        )
        
        prompt = CONSOLIDATION_PROMPT.format(episodes_text=episodes_text)
        
        try:
            # Llamar al LLM
            response = self.model.generate_content(prompt)
            
            # Parsear respuesta JSON
            response_text = response.text.strip()
            
            # Limpiar posibles marcadores de código
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            data = json.loads(response_text.strip())
            
            # Calcular confianza basada en número de episodios
            # Más episodios = mayor confianza (hasta cierto punto)
            confidence = min(0.95, 0.5 + (len(episodes) * 0.1))
            
            # Crear MetaMemory
            now = datetime.now(timezone.utc)
            meta_memory = MetaMemory(
                id=uuid4(),
                created_at=now,
                updated_at=now,
                pattern=data.get("pattern", "Patrón no identificado"),
                pattern_summary=data.get("pattern_summary", ""),
                lessons=data.get("lessons", []),
                best_practices=data.get("best_practices", []),
                antipatterns=data.get("antipatterns", []),
                exceptions=data.get("exceptions", []),
                edge_cases=data.get("edge_cases", []),
                contexts=data.get("contexts", []),
                technologies=data.get("technologies", []),
                source_episode_ids=[ep.id for ep in episodes],
                episode_count=len(episodes),
                confidence=confidence,
                coherence_score=data.get("coherence_score", 0.5),
                project_name=project_name,
                tags=self._extract_common_tags(episodes)
            )
            
            return meta_memory
            
        except json.JSONDecodeError as e:
            print(f"Error parseando respuesta del LLM: {e}")
            return None
        except Exception as e:
            print(f"Error en síntesis: {e}")
            return None
    
    def _extract_common_tags(self, episodes: list[Episode]) -> list[str]:
        """Extraer tags comunes entre episodios."""
        if not episodes:
            return []
        
        # Contar frecuencia de tags
        tag_counts = {}
        for ep in episodes:
            for tag in ep.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Retornar tags que aparecen en al menos 50% de episodios
        threshold = len(episodes) / 2
        common_tags = [
            tag for tag, count in tag_counts.items() 
            if count >= threshold
        ]
        
        return common_tags


def consolidate_memories(
    project_name: str,
    min_cluster_size: int = 3,
    storage: Optional[MemoryStorage] = None
) -> list[MetaMemory]:
    """
    Función de conveniencia para consolidar memorias de un proyecto.
    
    Args:
        project_name: Nombre del proyecto a consolidar
        min_cluster_size: Mínimo de episodios por cluster
        storage: Almacenamiento (opcional)
        
    Returns:
        Lista de meta-memorias generadas
    """
    consolidator = MemoryConsolidator(
        storage=storage,
        min_cluster_size=min_cluster_size
    )
    
    return consolidator.consolidate_project(project_name)
