"""
Aplicación Gradio para el Oráculo
=================================

Interfaz web interactiva para consultar memorias,
ver timeline y explorar lecciones aprendidas.
"""

import gradio as gr
from typing import Optional
from datetime import datetime

from memorytwin.config import get_settings
from memorytwin.oraculo.rag_engine import RAGEngine
from memorytwin.escriba.storage import MemoryStorage


# Inicialización global (se crea al importar)
_rag_engine: Optional[RAGEngine] = None
_storage: Optional[MemoryStorage] = None


def get_rag_engine() -> RAGEngine:
    """Obtener instancia singleton del motor RAG."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


def get_storage() -> MemoryStorage:
    """Obtener instancia singleton del storage."""
    global _storage
    if _storage is None:
        _storage = MemoryStorage()
    return _storage


# ============================================================
# Funciones para los componentes de Gradio
# ============================================================

def answer_question(question: str, project: str, num_results: int) -> tuple[str, str]:
    """
    Responder una pregunta usando RAG.
    
    Returns:
        (respuesta, episodios_usados_md)
    """
    if not question.strip():
        return "Por favor, escribe una pregunta.", ""
    
    try:
        rag = get_rag_engine()
        project_filter = project if project and project != "Todos" else None
        
        result = rag.query_sync(
            question=question,
            project_name=project_filter,
            top_k=num_results
        )
        
        answer = result["answer"]
        
        # Formatear episodios usados
        episodes_md = "### 📚 Episodios consultados\n\n"
        if result["episodes_used"]:
            for i, (ep, score) in enumerate(zip(
                result["episodes_used"], 
                result["relevance_scores"]
            ), 1):
                episodes_md += f"""
**{i}. {ep.task[:80]}...**
- Relevancia: {score:.0%}
- Tipo: {ep.episode_type.value}
- Fecha: {ep.timestamp.strftime('%Y-%m-%d %H:%M')}
- Tags: {', '.join(ep.tags[:5])}

"""
        else:
            episodes_md += "No se encontraron episodios relacionados."
            
        return answer, episodes_md
        
    except Exception as e:
        return f"❌ Error al procesar la consulta: {str(e)}", ""


def get_timeline_data(project: str, limit: int) -> str:
    """Obtener timeline como Markdown."""
    try:
        rag = get_rag_engine()
        project_filter = project if project and project != "Todos" else None
        
        timeline = rag.get_timeline(
            project_name=project_filter,
            limit=limit
        )
        
        if not timeline:
            return "📅 No hay episodios registrados en la memoria."
        
        md = f"# 📅 Timeline de Decisiones ({len(timeline)} episodios)\n\n"
        
        current_date = None
        for item in timeline:
            # Agrupar por fecha
            if item["date"] != current_date:
                current_date = item["date"]
                md += f"\n## {current_date}\n\n"
            
            icon = "✅" if item["success"] else "❌"
            type_emoji = {
                "decision": "🎯",
                "bug_fix": "🐛",
                "refactor": "♻️",
                "feature": "✨",
                "optimization": "⚡",
                "learning": "📖",
                "experiment": "🧪"
            }.get(item["type"], "📝")
            
            md += f"""### {icon} {type_emoji} {item["time"]} - {item["task"][:60]}...
> {item["summary"]}

Tags: `{"`  `".join(item["tags"][:4])}`

---

"""
        
        return md
        
    except Exception as e:
        return f"❌ Error al obtener timeline: {str(e)}"


def get_lessons_data(project: str, tag_filter: str) -> str:
    """Obtener lecciones aprendidas como Markdown."""
    try:
        rag = get_rag_engine()
        project_filter = project if project and project != "Todos" else None
        tags = [t.strip() for t in tag_filter.split(",")] if tag_filter else None
        
        lessons = rag.get_lessons(
            project_name=project_filter,
            tags=tags
        )
        
        if not lessons:
            return "📚 No hay lecciones documentadas (o ninguna coincide con los filtros)."
        
        md = f"# 📚 Lecciones Aprendidas ({len(lessons)})\n\n"
        
        for i, lesson in enumerate(lessons, 1):
            md += f"""## {i}. {lesson["lesson"]}

**Origen:** {lesson["from_task"][:100]}...

**Fecha:** {lesson["timestamp"].strftime('%Y-%m-%d')}

**Tags:** {', '.join(lesson["tags"][:5])}

---

"""
        
        return md
        
    except Exception as e:
        return f"❌ Error al obtener lecciones: {str(e)}"


def get_statistics() -> str:
    """Obtener estadísticas como Markdown."""
    try:
        rag = get_rag_engine()
        stats = rag.get_statistics()
        
        type_stats = "\n".join(
            f"| {k} | {v} |" 
            for k, v in stats["by_type"].items() if v > 0
        )
        
        assistant_stats = "\n".join(
            f"| {k} | {v} |" 
            for k, v in stats["by_assistant"].items()
        )
        
        md = f"""# 📊 Estadísticas de Memoria

## Resumen General

| Métrica | Valor |
|---------|-------|
| Total de Episodios | **{stats["total_episodes"]}** |
| Registros en ChromaDB | **{stats["chroma_count"]}** |

## Por Tipo de Episodio

| Tipo | Cantidad |
|------|----------|
{type_stats}

## Por Asistente de IA

| Asistente | Episodios |
|-----------|-----------|
{assistant_stats}

---
*Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""
        
        return md
        
    except Exception as e:
        return f"❌ Error al obtener estadísticas: {str(e)}"


def get_project_list() -> list[str]:
    """Obtener lista de proyectos disponibles."""
    try:
        storage = get_storage()
        # Consultar proyectos únicos desde SQLite
        with storage._get_session() as session:
            from memorytwin.escriba.storage import EpisodeRecord
            projects = session.query(
                EpisodeRecord.project_name
            ).distinct().all()
            
            project_list = ["Todos"] + [p[0] for p in projects if p[0]]
            return project_list
    except:
        return ["Todos", "default"]


# ============================================================
# Interfaz Gradio
# ============================================================

def create_gradio_interface() -> gr.Blocks:
    """Crear la interfaz Gradio completa."""
    
    with gr.Blocks(title="Memory Twin - Oráculo") as interface:
        
        gr.Markdown("""
        # 🔮 Memory Twin - Oráculo
        ### Asistente de Recuperación de Conocimiento Técnico
        
        Consulta el razonamiento detrás de las decisiones de código de tu equipo.
        """)
        
        with gr.Tabs():
            
            # ==================== TAB: Q&A ====================
            with gr.Tab("💬 Preguntas"):
                with gr.Row():
                    with gr.Column(scale=2):
                        question_input = gr.Textbox(
                            label="Tu pregunta",
                            placeholder="¿Por qué elegimos usar JWT para autenticación?",
                            lines=2
                        )
                        
                        with gr.Row():
                            project_filter = gr.Dropdown(
                                label="Proyecto",
                                choices=get_project_list(),
                                value="Todos",
                                interactive=True
                            )
                            num_results = gr.Slider(
                                label="Episodios a consultar",
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1
                            )
                        
                        ask_btn = gr.Button("🔍 Consultar Memoria", variant="primary")
                        
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### Ejemplos de preguntas:
                        - ¿Por qué elegimos la librería X?
                        - ¿Qué alternativas consideramos para...?
                        - ¿Qué errores tuvimos al implementar...?
                        - ¿Cuándo se tomó la decisión de...?
                        - ¿Qué lecciones aprendimos sobre...?
                        """)
                
                gr.Markdown("---")
                
                answer_output = gr.Markdown(
                    label="Respuesta",
                    value="*Las respuestas aparecerán aquí...*"
                )
                
                episodes_output = gr.Markdown(
                    label="Episodios consultados",
                    value=""
                )
                
                ask_btn.click(
                    fn=answer_question,
                    inputs=[question_input, project_filter, num_results],
                    outputs=[answer_output, episodes_output]
                )
            
            # ==================== TAB: Timeline ====================
            with gr.Tab("📅 Timeline"):
                with gr.Row():
                    timeline_project = gr.Dropdown(
                        label="Proyecto",
                        choices=get_project_list(),
                        value="Todos"
                    )
                    timeline_limit = gr.Slider(
                        label="Máximo de episodios",
                        minimum=10,
                        maximum=100,
                        value=30,
                        step=10
                    )
                    refresh_timeline_btn = gr.Button("🔄 Actualizar", variant="secondary")
                
                timeline_output = gr.Markdown(
                    value="*Haz clic en 'Actualizar' para ver el timeline...*"
                )
                
                refresh_timeline_btn.click(
                    fn=get_timeline_data,
                    inputs=[timeline_project, timeline_limit],
                    outputs=[timeline_output]
                )
            
            # ==================== TAB: Lecciones ====================
            with gr.Tab("📚 Lecciones"):
                with gr.Row():
                    lessons_project = gr.Dropdown(
                        label="Proyecto",
                        choices=get_project_list(),
                        value="Todos"
                    )
                    lessons_tags = gr.Textbox(
                        label="Filtrar por tags (separados por coma)",
                        placeholder="auth, jwt, security"
                    )
                    refresh_lessons_btn = gr.Button("🔄 Actualizar", variant="secondary")
                
                lessons_output = gr.Markdown(
                    value="*Haz clic en 'Actualizar' para ver las lecciones...*"
                )
                
                refresh_lessons_btn.click(
                    fn=get_lessons_data,
                    inputs=[lessons_project, lessons_tags],
                    outputs=[lessons_output]
                )
            
            # ==================== TAB: Estadísticas ====================
            with gr.Tab("📊 Estadísticas"):
                refresh_stats_btn = gr.Button("🔄 Actualizar Estadísticas", variant="secondary")
                
                stats_output = gr.Markdown(
                    value="*Haz clic en 'Actualizar' para ver las estadísticas...*"
                )
                
                refresh_stats_btn.click(
                    fn=get_statistics,
                    inputs=[],
                    outputs=[stats_output]
                )
        
        gr.Markdown("""
        ---
        **Memory Twin** - Agente de Memoria Episódica para Desarrollo de Software  
        Arquitectura dual: Escriba (Ingesta) + Oráculo (Consulta)
        """)
    
    return interface


def main():
    """Punto de entrada para ejecutar la interfaz Gradio."""
    settings = get_settings()
    
    interface = create_gradio_interface()
    
    interface.launch(
        server_port=settings.gradio_server_port,
        share=settings.gradio_share
    )


if __name__ == "__main__":
    main()
