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
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import numpy as np
from sklearn.cluster import DBSCAN

from memorytwin.config import get_llm_model
from memorytwin.observability import trace_consolidation
from memorytwin.models import Episode, MetaMemory
from memorytwin.escriba.storage import MemoryStorage

logger = logging.getLogger(__name__)


# Prompt para síntesis de episodios en meta-memoria (optimizado para rapidez)
CONSOLIDATION_PROMPT = """Sintetiza estos episodios de memoria técnica en una meta-memoria consolidada.

EPISODIOS:
{episodes_text}

Responde SOLO en JSON:
{{
    "pattern": "Patrón común identificado (1-2 oraciones)",
    "pattern_summary": "Resumen en 1 oración corta",
    "lessons": ["lección 1", "lección 2"],
    "best_practices": ["práctica 1"],
    "antipatterns": ["antipatrón 1"],
    "technologies": ["tech1", "tech2"],
    "coherence_score": 0.8
}}
"""


def format_episode_for_consolidation(episode: Episode) -> str:
    """Formatear un episodio para incluirlo en el prompt de consolidación (versión compacta)."""
    # Limitar razonamiento a 200 chars para mantener prompts pequeños
    reasoning = episode.reasoning_trace.raw_thinking[:200] if episode.reasoning_trace.raw_thinking else ""
    lessons = ', '.join(episode.lessons_learned[:2]) if episode.lessons_learned else 'N/A'
    
    return f"""[{episode.timestamp.strftime('%Y-%m-%d')}] {episode.task}
Solución: {episode.solution_summary[:100]}
Lecciones: {lessons}"""


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
        cluster_eps: float = 0.4,  # Balance entre cohesión y cobertura
        max_episodes_per_cluster: int = 8  # Limitar tamaño de prompts
    ):
        """
        Inicializar el consolidador.
        
        Args:
            storage: Almacenamiento de memoria
            api_key: DEPRECATED - ya no se usa, la API key se lee de config.
            min_cluster_size: Mínimo de episodios para formar cluster
            cluster_eps: Radio máximo para clustering DBSCAN (menor = más estricto)
            max_episodes_per_cluster: Máximo de episodios por cluster para limitar prompts
        """
        self.storage = storage or MemoryStorage()
        
        # Usar factory centralizada (temperatura baja, tokens reducidos para rapidez)
        self.model = get_llm_model(temperature=0.2, max_output_tokens=1024)
        
        self.min_cluster_size = min_cluster_size
        self.cluster_eps = cluster_eps
        self.max_episodes_per_cluster = max_episodes_per_cluster

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
        logger.info(f"Iniciando consolidación para proyecto: {project_name}")
        
        # Obtener episodios del proyecto
        episodes = self.storage.get_episodes_by_project(project_name, limit=200)
        logger.info(f"Encontrados {len(episodes)} episodios")
        
        if len(episodes) < self.min_cluster_size:
            logger.info(f"Insuficientes episodios ({len(episodes)} < {self.min_cluster_size})")
            return []
        
        # Obtener embeddings de ChromaDB
        embeddings, episode_ids = self._get_episode_embeddings(episodes)
        logger.info(f"Obtenidos {len(embeddings)} embeddings")
        
        if len(embeddings) < self.min_cluster_size:
            logger.info("Insuficientes embeddings")
            return []
        
        # Clustering
        clusters = self._cluster_episodes(embeddings, episode_ids)
        logger.info(f"Generados {len(clusters)} clusters")
        
        # Generar meta-memorias para cada cluster
        meta_memories = []
        for i, cluster_episode_ids in enumerate(clusters):
            logger.info(f"Procesando cluster {i+1}/{len(clusters)} ({len(cluster_episode_ids)} episodios)")
            
            # Obtener episodios del cluster
            cluster_episodes = [
                ep for ep in episodes 
                if str(ep.id) in cluster_episode_ids
            ]
            
            # Limitar episodios por cluster para evitar prompts enormes
            if len(cluster_episodes) > self.max_episodes_per_cluster:
                # Seleccionar los más recientes
                cluster_episodes = sorted(
                    cluster_episodes, 
                    key=lambda e: e.timestamp, 
                    reverse=True
                )[:self.max_episodes_per_cluster]
                logger.info(f"Cluster limitado a {self.max_episodes_per_cluster} episodios más recientes")
            
            if len(cluster_episodes) >= self.min_cluster_size:
                logger.info(f"Sintetizando cluster {i+1} con LLM...")
                meta_memory = self._synthesize_cluster(
                    cluster_episodes, 
                    project_name
                )
                if meta_memory:
                    # Almacenar
                    self.storage.store_meta_memory(meta_memory)
                    meta_memories.append(meta_memory)
                    logger.info(f"Meta-memoria {i+1} creada: {meta_memory.pattern_summary[:50]}...")
        
        logger.info(f"Consolidación completada: {len(meta_memories)} meta-memorias generadas")
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
        
        if result["embeddings"] is None or len(result["embeddings"]) == 0:
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

    @trace_consolidation
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
            # Llamar al LLM (interfaz unificada)
            response = self.model.generate(prompt)
            
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
