"""
Escriba - Agente Principal de Ingesta de Memoria
================================================

Coordina el procesamiento y almacenamiento de episodios
de memoria técnica.
"""

from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from memorytwin.config import get_settings
from memorytwin.models import Episode, ProcessedInput
from memorytwin.escriba.processor import ThoughtProcessor
from memorytwin.escriba.storage import MemoryStorage

console = Console()


class Escriba:
    """
    Agente Escriba - Observador pasivo y documentador activo.
    
    Captura el razonamiento de asistentes de IA, lo estructura
    usando un LLM ligero y lo almacena para consulta futura.
    """
    
    def __init__(
        self,
        processor: Optional[ThoughtProcessor] = None,
        storage: Optional[MemoryStorage] = None,
        project_name: str = "default"
    ):
        """
        Inicializar el Escriba.
        
        Args:
            processor: Procesador de pensamientos (se crea uno si no se provee)
            storage: Almacenamiento de memoria (se crea uno si no se provee)
            project_name: Nombre del proyecto por defecto
        """
        self.processor = processor or ThoughtProcessor()
        self.storage = storage or MemoryStorage()
        self.project_name = project_name
        
        console.print(Panel(
            f"[bold green]✓ Escriba inicializado[/bold green]\n"
            f"Proyecto: {project_name}",
            title="Memory Twin - Escriba",
            border_style="green"
        ))
        
    async def capture_thinking(
        self,
        thinking_text: str,
        user_prompt: Optional[str] = None,
        code_changes: Optional[str] = None,
        source_assistant: str = "unknown",
        project_name: Optional[str] = None
    ) -> Episode:
        """
        Capturar y procesar texto de "thinking" de un asistente.
        
        Args:
            thinking_text: Texto de razonamiento visible del modelo
            user_prompt: Prompt original del usuario
            code_changes: Cambios de código asociados
            source_assistant: Asistente fuente (copilot, claude, cursor)
            project_name: Nombre del proyecto (usa el default si no se especifica)
            
        Returns:
            Episode estructurado y almacenado
        """
        project = project_name or self.project_name
        
        console.print(f"[yellow]📝 Capturando pensamiento...[/yellow]")
        console.print(f"   Fuente: {source_assistant}")
        console.print(f"   Proyecto: {project}")
        
        # Crear input procesado
        raw_input = ProcessedInput(
            raw_text=thinking_text,
            user_prompt=user_prompt,
            code_changes=code_changes,
            source="api",
            captured_at=datetime.utcnow()
        )
        
        # Procesar con LLM
        console.print(f"[yellow]🔄 Estructurando con LLM...[/yellow]")
        episode = await self.processor.process_thought(
            raw_input,
            project_name=project,
            source_assistant=source_assistant
        )
        
        # Almacenar
        console.print(f"[yellow]💾 Almacenando episodio...[/yellow]")
        episode_id = self.storage.store_episode(episode)
        
        console.print(Panel(
            f"[bold green]✓ Episodio capturado[/bold green]\n"
            f"ID: {episode_id}\n"
            f"Tarea: {episode.task[:100]}...\n"
            f"Tipo: {episode.episode_type.value}\n"
            f"Tags: {', '.join(episode.tags[:5])}",
            title="Memoria Registrada",
            border_style="green"
        ))
        
        return episode
    
    def capture_thinking_sync(
        self,
        thinking_text: str,
        user_prompt: Optional[str] = None,
        code_changes: Optional[str] = None,
        source_assistant: str = "unknown",
        project_name: Optional[str] = None
    ) -> Episode:
        """Versión síncrona de capture_thinking."""
        import asyncio
        return asyncio.run(
            self.capture_thinking(
                thinking_text,
                user_prompt,
                code_changes,
                source_assistant,
                project_name
            )
        )
    
    def capture_from_file(
        self,
        file_path: str,
        source_assistant: str = "unknown",
        project_name: Optional[str] = None
    ) -> Episode:
        """
        Capturar pensamiento desde un archivo de texto.
        
        Args:
            file_path: Ruta al archivo con el texto de thinking
            source_assistant: Asistente fuente
            project_name: Nombre del proyecto
            
        Returns:
            Episode estructurado y almacenado
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            thinking_text = f.read()
            
        return self.capture_thinking_sync(
            thinking_text,
            source_assistant=source_assistant,
            project_name=project_name
        )
    
    def capture_from_clipboard(
        self,
        source_assistant: str = "unknown",
        project_name: Optional[str] = None
    ) -> Episode:
        """
        Capturar pensamiento desde el clipboard del sistema.
        
        Args:
            source_assistant: Asistente fuente
            project_name: Nombre del proyecto
            
        Returns:
            Episode estructurado y almacenado
        """
        try:
            import pyperclip
            thinking_text = pyperclip.paste()
            
            if not thinking_text or len(thinking_text.strip()) < 50:
                raise ValueError(
                    "El clipboard está vacío o tiene muy poco contenido. "
                    "Copia el texto de 'thinking' del asistente primero."
                )
                
            return self.capture_thinking_sync(
                thinking_text,
                source_assistant=source_assistant,
                project_name=project_name
            )
            
        except ImportError:
            raise ImportError(
                "Se requiere 'pyperclip' para captura desde clipboard. "
                "Instálalo con: pip install pyperclip"
            )
    
    def get_statistics(self) -> dict:
        """Obtener estadísticas del almacenamiento."""
        return self.storage.get_statistics(self.project_name)
    
    def search(self, query: str, top_k: int = 5):
        """
        Buscar en la memoria (wrapper simple).
        
        Args:
            query: Texto de búsqueda
            top_k: Número de resultados
            
        Returns:
            Lista de resultados de búsqueda
        """
        from memorytwin.models import MemoryQuery
        
        memory_query = MemoryQuery(
            query=query,
            project_filter=self.project_name,
            top_k=top_k
        )
        
        return self.storage.search_episodes(memory_query)
