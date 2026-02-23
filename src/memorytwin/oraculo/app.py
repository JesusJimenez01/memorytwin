"""
Oráculo - Gradio Interface for Memory Twin
==========================================

Modern graphical interface for interacting with the memory system.
Features:
1. Query memories (RAG)
2. Visualize timeline and lessons
3. Manage memories (delete episodes)
4. View system statistics
"""


import gradio as gr
import pandas as pd

from memorytwin.config import get_settings
from memorytwin.escriba import MemoryStorage
from memorytwin.oraculo import RAGEngine

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
        return ["(All)"] + projects
    except Exception:
        return ["(All)"]

def answer_question(question: str, project_name: str = "", num_episodes: int = 5) -> str:
    """Answer a question using RAG over memory episodes."""
    if not question.strip():
        return "⚠️ Please enter a question."

    try:
        rag = get_rag_engine()
        project = None if project_name in ["", "(All)"] else project_name.strip()
        response = rag.query_sync(question, project_name=project, top_k=num_episodes)
        return response["answer"]
    except Exception as e:
        return f"❌ Error processing the question: {str(e)}"

def get_timeline_markdown(project_name: str = "", limit: int = 20) -> str:
    """Get timeline of episodes formatted as Markdown."""
    try:
        storage = get_storage()
        project = None if project_name in ["", "(All)"] else project_name.strip()
        episodes = storage.get_timeline(project_name=project, limit=limit)

        if not episodes:
            return "ℹ️ No episodes recorded."

        result = ""
        for ep in episodes:
            date = ep.timestamp.strftime("%Y-%m-%d %H:%M")

            # Badges
            badges = []
            if ep.is_antipattern:
                badges.append("🔴 ANTIPATTERN")
            if ep.is_critical:
                badges.append("⭐ CRITICAL")
            badges_str = " ".join(badges)

            # Content
            task_title = ep.task.split('\n')[0][:100]
            if len(ep.task) > 100:
                task_title += "..."

            result += f"### 🗓️ {date}\n"
            if badges_str:
                result += f"**{badges_str}**\n\n"

            result += f"**Task:** {task_title}\n\n"
            result += f"**Type:** `{ep.episode_type.value}` | **Project:** `{ep.project_name}`\n"

            if ep.tags:
                tags_str = ", ".join([f"`{t}`" for t in ep.tags])
                result += f"**Tags:** {tags_str}\n"

            result += "\n---\n\n"

        return result
    except Exception as e:
        return f"❌ Error getting timeline: {str(e)}"

def get_lessons_markdown(project_name: str = "") -> str:
    """Get aggregated lessons learned formatted as Markdown."""
    try:
        storage = get_storage()
        project = None if project_name in ["", "(All)"] else project_name.strip()
        lessons = storage.get_lessons_learned(project_name=project)

        if not lessons:
            return "ℹ️ No lessons learned recorded."

        result = ""
        for lesson_data in lessons:
            task = lesson_data.get('from_task', 'Unknown task')[:60]
            lesson = lesson_data.get('lesson', '')
            tags = lesson_data.get('tags', [])
            timestamp = lesson_data.get('timestamp')
            date_str = timestamp.strftime('%Y-%m-%d') if timestamp else ''

            result += f"### 💡 {lesson}\n"
            result += f"_Learned from: {task}... ({date_str})_\n"
            if tags:
                tags_str = ", ".join([f"`{t}`" for t in tags])
                result += f"**Tags:** {tags_str}\n"
            result += "\n---\n"

        return result
    except Exception as e:
        return f"❌ Error getting lessons: {str(e)}"

def get_statistics_markdown(project_name: str = "") -> str:
    """Get memory statistics formatted as Markdown."""
    try:
        storage = get_storage()
        project = None if project_name in ["", "(All)"] else project_name.strip()
        stats = storage.get_statistics(project_name=project)

        if stats.get('total_episodes', 0) == 0:
            return "ℹ️ No episodes recorded."

        # General Stats
        result = "### 📊 General Summary\n"
        result += f"- **Total episodes:** {stats['total_episodes']}\n"
        result += f"- **Indexed in ChromaDB:** {stats.get('chroma_count', 0)}\n\n"

        # By Type
        result += "### 🧩 By Episode Type\n"
        by_type = stats.get('by_type', {})
        for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
            if count > 0:
                result += f"- **{t}:** {count}\n"

        # By Assistant
        result += "\n### 🤖 By Assistant\n"
        by_assistant = stats.get('by_assistant', {})
        for a, count in sorted(by_assistant.items(), key=lambda x: -x[1]):
            if count > 0:
                result += f"- **{a}:** {count}\n"

        return result
    except Exception as e:
        return f"❌ Error getting statistics: {str(e)}"

def get_episodes_dataframe(project_name: str = "", limit: int = 50) -> pd.DataFrame:
    """Get episodes as a DataFrame for the management tab."""
    try:
        storage = get_storage()
        project = None if project_name in ["", "(All)"] else project_name.strip()
        episodes = storage.get_timeline(project_name=project, limit=limit)

        data = []
        for ep in episodes:
            data.append({
                "ID": str(ep.id),
                "Date": ep.timestamp.strftime("%Y-%m-%d %H:%M"),
                "Project": ep.project_name,
                "Task": ep.task[:50] + "..." if len(ep.task) > 50 else ep.task,
                "Type": ep.episode_type.value,
                "Antipattern": "✅" if ep.is_antipattern else "",
                "Critical": "⭐" if ep.is_critical else ""
            })

        if not data:
            return pd.DataFrame(columns=["ID", "Date", "Project", "Task", "Type", "Antipattern", "Critical"])

        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error getting dataframe: {e}")
        return pd.DataFrame(columns=["Error"])

def delete_episode_action(episode_id: str) -> str:
    """Delete an episode by ID."""
    if not episode_id.strip():
        return "⚠️ Please enter a valid ID."

    try:
        storage = get_storage()
        success = storage.delete_episode(episode_id.strip())
        if success:
            return f"✅ Episode {episode_id} deleted successfully."
        else:
            return f"❌ Episode with ID {episode_id} not found."
    except Exception as e:
        return f"❌ Error deleting: {str(e)}"

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
        gr.Markdown("Episodic Memory System for AI Assistants", elem_classes=["text-center"])

        # State for project selection (shared across tabs if needed, but kept simple here)
        projects = get_available_projects()

        with gr.Tabs():

            # --- Tab 1: Queries (Chat/RAG) ---
            with gr.Tab("💬 Queries", id="tab_chat"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔍 Ask a Question")
                        question_input = gr.Textbox(
                            label="Question",
                            placeholder="E.g.: Why did we choose ChromaDB as the vector database?",
                            lines=4,
                            info="Query your consolidated knowledge base."
                        )

                        with gr.Row():
                            project_input = gr.Dropdown(
                                label="Filter by Project",
                                choices=projects,
                                value="(All)",
                                interactive=True
                            )
                            num_episodes = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1,
                                label="Episodes to query"
                            )

                        ask_btn = gr.Button("Query Memory", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        gr.Markdown("### 💡 Generated Answer")
                        answer_output = gr.Markdown(
                            value="_The answer will appear here..._",
                            elem_classes=["markdown-text"]
                        )

                ask_btn.click(
                    fn=answer_question,
                    inputs=[question_input, project_input, num_episodes],
                    outputs=answer_output
                )

            # --- Tab 2: Memory Management (CRUD) ---
            with gr.Tab("🛠️ Management", id="tab_manage"):
                gr.Markdown("### 🗂️ Manage Memory Episodes")
                gr.Markdown("View and delete incorrect or obsolete episodes.")

                with gr.Row():
                    with gr.Column(scale=3):
                        manage_project = gr.Dropdown(
                            label="Filter Project",
                            choices=projects,
                            value="(All)",
                            interactive=True
                        )
                    with gr.Column(scale=1):
                        refresh_btn = gr.Button("🔄 Refresh Table")

                episodes_table = gr.Dataframe(
                    headers=["ID", "Date", "Project", "Task", "Type", "Antipattern", "Critical"],
                    datatype=["str", "str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    label="Recent Episodes",
                    wrap=True
                )

                gr.Markdown("---")
                gr.Markdown("### 🗑️ Delete Episode")

                with gr.Row():
                    id_to_delete = gr.Textbox(
                        label="Episode ID",
                        placeholder="Paste the episode ID to delete here (ID column from the table)",
                        lines=1
                    )
                    delete_btn = gr.Button("🗑️ Delete Episode", variant="stop")

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
                        label="Project",
                        choices=projects,
                        value="(All)",
                        interactive=True
                    )
                    timeline_limit = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=20,
                        step=10,
                        label="Episode limit"
                    )
                    timeline_btn = gr.Button("View Timeline", variant="secondary")

                timeline_output = gr.Markdown()
                timeline_btn.click(
                    fn=get_timeline_markdown,
                    inputs=[timeline_project, timeline_limit],
                    outputs=timeline_output
                )

            # --- Tab 4: Lessons ---
            with gr.Tab("📚 Lessons", id="tab_lessons"):
                with gr.Row():
                    lessons_project = gr.Dropdown(
                        label="Project",
                        choices=projects,
                        value="(All)",
                        interactive=True
                    )
                    lessons_btn = gr.Button("View Lessons Learned", variant="secondary")

                lessons_output = gr.Markdown()
                lessons_btn.click(
                    fn=get_lessons_markdown,
                    inputs=[lessons_project],
                    outputs=lessons_output
                )

            # --- Tab 5: Statistics ---
            with gr.Tab("📊 Statistics", id="tab_stats"):
                with gr.Row():
                    stats_project = gr.Dropdown(
                        label="Project",
                        choices=projects,
                        value="(All)",
                        interactive=True
                    )
                    stats_btn = gr.Button("View Statistics", variant="secondary")

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
    print(f"🚀 Starting Oráculo at http://0.0.0.0:{settings.gradio_server_port}")
    app.launch(
        server_name="0.0.0.0",
        server_port=settings.gradio_server_port,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
