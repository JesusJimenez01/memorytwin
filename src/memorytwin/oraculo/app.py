"""
Oráculo - Gradio Interface for Memory Twin
==========================================

Interfaz gráfica moderna para interactuar con el sistema de memoria.
Permite:
1. Consultar memorias (RAG)
2. Visualizar timeline y lecciones
3. Gestionar memorias (borrar episodios)
4. Ver estadísticas del sistema
"""

import gradio as gr
import pandas as pd
from datetime import datetime
from memorytwin.escriba import MemoryStorage
from memorytwin.oraculo import RAGEngine
from memorytwin.config import get_settings

# Singleton instances
_rag_engine = None
_storage = None

def get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine

def get_storage():
    global _storage
    if _storage is None:
        _storage = MemoryStorage()
    return _storage

# --- Logic Functions ---

def get_available_projects() -> list[str]:
    """Get list of available projects from storage."""
    try:
        storage = get_storage()
        projects = storage.get_all_projects()
        return ["(Todos)"] + projects
    except Exception:
        return ["(Todos)"]

def answer_question(question: str, project_name: str = "", num_episodes: int = 5) -> str:
    """Answer a question using RAG over memory episodes."""
    if not question.strip():
        return "⚠️ Por favor, ingresa una pregunta."
    
    try:
        rag = get_rag_engine()
        project = None if project_name in ["", "(Todos)"] else project_name.strip()
        response = rag.query_sync(question, project_name=project, top_k=num_episodes)
        return response["answer"]
    except Exception as e:
        return f"❌ Error al procesar la pregunta: {str(e)}"

def get_timeline_markdown(project_name: str = "", limit: int = 20) -> str:
    """Get timeline of episodes formatted as Markdown."""
    try:
        storage = get_storage()
        project = None if project_name in ["", "(Todos)"] else project_name.strip()
        episodes = storage.get_timeline(project_name=project, limit=limit)
        
        if not episodes:
            return "ℹ️ No hay episodios registrados."
        
        result = ""
        for ep in episodes:
            date = ep.timestamp.strftime("%Y-%m-%d %H:%M")
            
            # Badges
            badges = []
            if ep.is_antipattern:
                badges.append("🔴 ANTIPATTERN")
            if ep.is_critical:
                badges.append("⭐ CRÍTICO")
            badges_str = " ".join(badges)
            
            # Content
            task_title = ep.task.split('\n')[0][:100]
            if len(ep.task) > 100:
                task_title += "..."
                
            result += f"### 🗓️ {date}\n"
            if badges_str:
                result += f"**{badges_str}**\n\n"
            
            result += f"**Tarea:** {task_title}\n\n"
            result += f"**Tipo:** `{ep.episode_type.value}` | **Proyecto:** `{ep.project_name}`\n"
            
            if ep.tags:
                tags_str = ", ".join([f"`{t}`" for t in ep.tags])
                result += f"**Tags:** {tags_str}\n"
            
            result += "\n---\n\n"
        
        return result
    except Exception as e:
        return f"❌ Error al obtener timeline: {str(e)}"

def get_lessons_markdown(project_name: str = "") -> str:
    """Get aggregated lessons learned formatted as Markdown."""
    try:
        storage = get_storage()
        project = None if project_name in ["", "(Todos)"] else project_name.strip()
        lessons = storage.get_lessons_learned(project_name=project)
        
        if not lessons:
            return "ℹ️ No hay lecciones aprendidas registradas."
        
        result = ""
        for lesson_data in lessons:
            task = lesson_data.get('from_task', 'Tarea desconocida')[:60]
            lesson = lesson_data.get('lesson', '')
            tags = lesson_data.get('tags', [])
            timestamp = lesson_data.get('timestamp')
            date_str = timestamp.strftime('%Y-%m-%d') if timestamp else ''
            
            result += f"### 💡 {lesson}\n"
            result += f"_Aprendido en: {task}... ({date_str})_\n"
            if tags:
                tags_str = ", ".join([f"`{t}`" for t in tags])
                result += f"**Tags:** {tags_str}\n"
            result += "\n---\n"
        
        return result
    except Exception as e:
        return f"❌ Error al obtener lecciones: {str(e)}"

def get_statistics_markdown(project_name: str = "") -> str:
    """Get memory statistics formatted as Markdown."""
    try:
        storage = get_storage()
        project = None if project_name in ["", "(Todos)"] else project_name.strip()
        stats = storage.get_statistics(project_name=project)
        
        if stats.get('total_episodes', 0) == 0:
            return "ℹ️ No hay episodios registrados."
        
        # General Stats
        result = "### 📊 Resumen General\n"
        result += f"- **Total de episodios:** {stats['total_episodes']}\n"
        result += f"- **Indexados en ChromaDB:** {stats.get('chroma_count', 0)}\n\n"
        
        # By Type
        result += "### 🧩 Por Tipo de Episodio\n"
        by_type = stats.get('by_type', {})
        for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
            if count > 0:
                result += f"- **{t}:** {count}\n"
        
        # By Assistant
        result += "\n### 🤖 Por Asistente\n"
        by_assistant = stats.get('by_assistant', {})
        for a, count in sorted(by_assistant.items(), key=lambda x: -x[1]):
            if count > 0:
                result += f"- **{a}:** {count}\n"
        
        return result
    except Exception as e:
        return f"❌ Error al obtener estadísticas: {str(e)}"

def get_episodes_dataframe(project_name: str = "", limit: int = 50) -> pd.DataFrame:
    """Get episodes as a DataFrame for the management tab."""
    try:
        storage = get_storage()
        project = None if project_name in ["", "(Todos)"] else project_name.strip()
        episodes = storage.get_timeline(project_name=project, limit=limit)
        
        data = []
        for ep in episodes:
            data.append({
                "ID": str(ep.id),
                "Fecha": ep.timestamp.strftime("%Y-%m-%d %H:%M"),
                "Proyecto": ep.project_name,
                "Tarea": ep.task[:50] + "..." if len(ep.task) > 50 else ep.task,
                "Tipo": ep.episode_type.value,
                "Antipattern": "✅" if ep.is_antipattern else "",
                "Crítico": "⭐" if ep.is_critical else ""
            })
            
        if not data:
            return pd.DataFrame(columns=["ID", "Fecha", "Proyecto", "Tarea", "Tipo", "Antipattern", "Crítico"])
            
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error getting dataframe: {e}")
        return pd.DataFrame(columns=["Error"])

def delete_episode_action(episode_id: str) -> str:
    """Delete an episode by ID."""
    if not episode_id.strip():
        return "⚠️ Por favor, ingresa un ID válido."
    
    try:
        storage = get_storage()
        success = storage.delete_episode(episode_id.strip())
        if success:
            return f"✅ Episodio {episode_id} eliminado correctamente."
        else:
            return f"❌ No se encontró el episodio con ID {episode_id}."
    except Exception as e:
        return f"❌ Error al eliminar: {str(e)}"

# --- UI Construction ---

def create_gradio_interface():
    """Create the Gradio interface with a modern theme."""
    
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        font=["Inter", "sans-serif"]
    )

    with gr.Blocks(title="Memory Twin - Oráculo") as app:
        app.theme = theme
        
        gr.Markdown("# 🧠 Memory Twin | Oráculo")
        gr.Markdown("Sistema de Memoria Episódica para Asistentes de IA", elem_classes=["text-center"])
        
        # State for project selection (shared across tabs if needed, but kept simple here)
        projects = get_available_projects()
        
        with gr.Tabs():
            
            # --- Tab 1: Consultas (Chat/RAG) ---
            with gr.Tab("💬 Consultas", id="tab_chat"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔍 Realizar Consulta")
                        question_input = gr.Textbox(
                            label="Pregunta",
                            placeholder="Ej: ¿Por qué elegimos ChromaDB como base de datos vectorial?",
                            lines=4,
                            info="Consulta tu base de conocimiento consolidada."
                        )
                        
                        with gr.Row():
                            project_input = gr.Dropdown(
                                label="Filtrar por Proyecto",
                                choices=projects,
                                value="(Todos)",
                                interactive=True
                            )
                            num_episodes = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1,
                                label="Episodios a consultar"
                            )
                        
                        ask_btn = gr.Button("Consultar Memoria", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 💡 Respuesta Generada")
                        answer_output = gr.Markdown(
                            value="_La respuesta aparecerá aquí..._",
                            elem_classes=["markdown-text"]
                        )
                
                ask_btn.click(
                    fn=answer_question,
                    inputs=[question_input, project_input, num_episodes],
                    outputs=answer_output
                )

            # --- Tab 2: Gestión de Memorias (CRUD) ---
            with gr.Tab("🛠️ Gestión", id="tab_manage"):
                gr.Markdown("### 🗂️ Administrar Episodios de Memoria")
                gr.Markdown("Visualiza y elimina episodios incorrectos u obsoletos.")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        manage_project = gr.Dropdown(
                            label="Filtrar Proyecto",
                            choices=projects,
                            value="(Todos)",
                            interactive=True
                        )
                    with gr.Column(scale=1):
                        refresh_btn = gr.Button("🔄 Actualizar Tabla")
                
                episodes_table = gr.Dataframe(
                    headers=["ID", "Fecha", "Proyecto", "Tarea", "Tipo", "Antipattern", "Crítico"],
                    datatype=["str", "str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    label="Últimos Episodios",
                    wrap=True
                )
                
                gr.Markdown("---")
                gr.Markdown("### 🗑️ Eliminar Episodio")
                
                with gr.Row():
                    id_to_delete = gr.Textbox(
                        label="ID del Episodio",
                        placeholder="Pega aquí el ID del episodio a eliminar (columna ID de la tabla)",
                        lines=1
                    )
                    delete_btn = gr.Button("🗑️ Eliminar Episodio", variant="stop")
                
                delete_output = gr.Markdown()
                
                # Events
                # Load table on click refresh or change project
                refresh_btn.click(
                    fn=get_episodes_dataframe,
                    inputs=[manage_project],
                    outputs=episodes_table
                )
                
                # Also load on tab select (simulated by loading on launch/change)
                manage_project.change(
                    fn=get_episodes_dataframe,
                    inputs=[manage_project],
                    outputs=episodes_table
                )
                
                # Delete action
                delete_btn.click(
                    fn=delete_episode_action,
                    inputs=[id_to_delete],
                    outputs=delete_output
                ).then( # Refresh table after delete
                    fn=get_episodes_dataframe,
                    inputs=[manage_project],
                    outputs=episodes_table
                )

            # --- Tab 3: Timeline ---
            with gr.Tab("📅 Timeline", id="tab_timeline"):
                with gr.Row():
                    timeline_project = gr.Dropdown(
                        label="Proyecto",
                        choices=projects,
                        value="(Todos)",
                        interactive=True
                    )
                    timeline_limit = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=20,
                        step=10,
                        label="Límite de episodios"
                    )
                    timeline_btn = gr.Button("Ver Timeline", variant="secondary")
                
                timeline_output = gr.Markdown()
                timeline_btn.click(
                    fn=get_timeline_markdown,
                    inputs=[timeline_project, timeline_limit],
                    outputs=timeline_output
                )

            # --- Tab 4: Lecciones ---
            with gr.Tab("📚 Lecciones", id="tab_lessons"):
                with gr.Row():
                    lessons_project = gr.Dropdown(
                        label="Proyecto",
                        choices=projects,
                        value="(Todos)",
                        interactive=True
                    )
                    lessons_btn = gr.Button("Ver Lecciones Aprendidas", variant="secondary")
                
                lessons_output = gr.Markdown()
                lessons_btn.click(
                    fn=get_lessons_markdown,
                    inputs=[lessons_project],
                    outputs=lessons_output
                )

            # --- Tab 5: Estadísticas ---
            with gr.Tab("📊 Estadísticas", id="tab_stats"):
                with gr.Row():
                    stats_project = gr.Dropdown(
                        label="Proyecto",
                        choices=projects,
                        value="(Todos)",
                        interactive=True
                    )
                    stats_btn = gr.Button("Ver Estadísticas", variant="secondary")
                
                stats_output = gr.Markdown()
                stats_btn.click(
                    fn=get_statistics_markdown,
                    inputs=[stats_project],
                    outputs=stats_output
                )
                
    return app

def main():
    """Main entry point."""
    settings = get_settings()
    app = create_gradio_interface()
    print(f"🚀 Iniciando Oráculo en http://0.0.0.0:{settings.gradio_server_port}")
    app.launch(
        server_name="0.0.0.0",
        server_port=settings.gradio_server_port,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
