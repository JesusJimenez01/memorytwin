# 🧠 The Memory Twin

> **Agente de Memoria Episódica para Desarrollo de Software**

Sistema de arquitectura dual (Escriba + Oráculo) diseñado para mitigar la "amnesia técnica" en equipos de desarrollo. Captura, procesa y almacena el razonamiento ("thinking") detrás de las decisiones de código tomadas por asistentes de IA.

## 🎯 Valor Diferencial

- **Memoria de Razonamiento**: Captura el "porqué" (thinking), no solo el "qué" (código final)
- **Agnóstico del Modelo**: Funciona con cualquier asistente (Copilot, Claude, Cursor)
- **Onboarding Automatizado**: Reduce el tiempo de aprendizaje en proyectos legacy
- **RAG sobre Decisiones**: Consulta contextual sobre la historia técnica
- **Colaborativo**: Soporte para base de datos compartida en equipos

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                      Memory Twin                            │
├────────────────────────┬────────────────────────────────────┤
│     ESCRIBA            │           ORÁCULO                  │
│   (Backend/Ingesta)    │       (Frontend/Consulta)          │
├────────────────────────┼────────────────────────────────────┤
│ • Captura thinking     │ • Q&A Contextual (RAG)             │
│ • Procesa con LLM      │ • Timeline de Decisiones           │
│ • Genera embeddings    │ • Lecciones Aprendidas             │
│ • Almacena episodios   │ • Interfaz Gradio                  │
├────────────────────────┴────────────────────────────────────┤
│                     MCP Server                              │
│            (Model Context Protocol)                         │
├─────────────────────────────────────────────────────────────┤
│                 Storage Backend (Strategy)                  │
│      ┌─────────────────────────┬──────────────────────┐     │
│      │         Local           │       Server         │     │
│      │ (SQLite + ChromaDir)    │ (ChromaDB Server)    │     │
│      └─────────────────────────┴──────────────────────┘     │
├─────────────────────────────────────────────────────────────┤
│                Langfuse (Observabilidad)                    │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Instalación y Configuración

### 1. Instalación

```bash
# Clonar el repositorio
cd memorytwin

# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Instalar dependencias y el paquete en modo editable
pip install -e .
```

### 2. Configuración Inicial

Memory Twin incluye un comando de configuración automática que prepara tu entorno de desarrollo.

```bash
# Configura el entorno, crea archivos de configuración y prepara la integración con VS Code
mt setup
```

Este comando:
1.  Crea el archivo `.env` si no existe (deberás editarlo con tu `GOOGLE_API_KEY`).
2.  Genera `.github/copilot-instructions.md` con las instrucciones para tu agente de IA.
3.  Genera `.vscode/mcp.json` configurado automáticamente para usar el servidor MCP de Memory Twin en este proyecto.

### 3. Variables de Entorno

Edita el archivo `.env` generado con tus credenciales:

```env
# Requerido: API Key de Google Gemini
GOOGLE_API_KEY=tu_api_key_de_gemini

# Opcional: Configuración de Almacenamiento (por defecto 'local')
STORAGE_BACKEND=local
# STORAGE_BACKEND=chromadb_server
# CHROMA_SERVER_HOST=localhost
# CHROMA_SERVER_PORT=8000

# Opcional: Observabilidad con Langfuse
# LANGFUSE_PUBLIC_KEY=...
# LANGFUSE_SECRET_KEY=...
# LANGFUSE_HOST=...
```

## 🚀 Uso

### Integración con VS Code y Copilot

Gracias al comando `mt setup`, tu VS Code ya debería estar configurado.

1.  **Reinicia VS Code** para que cargue la configuración de MCP.
2.  Abre el chat de Copilot y verás disponibles las herramientas de Memory Twin.
3.  Copilot usará automáticamente estas herramientas siguiendo las instrucciones en `.github/copilot-instructions.md`.

#### Herramientas MCP Disponibles

| Herramienta | Descripción |
|-------------|-------------|
| `get_project_context` | ⭐ **Principal**. Obtiene contexto inteligente del proyecto. Usar al inicio de cada tarea. |
| `capture_thinking` | Captura y almacena el razonamiento de decisiones técnicas. |
| `query_memory` | Consulta memorias usando RAG. Ej: "¿Por qué elegimos X?" |
| `search_episodes` | Búsqueda semántica de episodios por término. |
| `get_episode` | Obtiene el contenido completo de un episodio por ID. |
| `get_lessons` | Obtiene lecciones aprendidas agregadas. |
| `get_timeline` | Timeline cronológico de decisiones técnicas. |
| `get_statistics` | Estadísticas de la base de memoria. |
| `onboard_project` | Analiza un proyecto existente y crea un episodio inicial. |

### CLI (Línea de Comandos)

Puedes usar el comando `mt` directamente en tu terminal:

```bash
# Capturar un pensamiento desde un archivo
mt capture --file thinking.txt --assistant copilot --project mi-proyecto

# Capturar desde el portapapeles
mt capture --clipboard --assistant claude

# Buscar en la memoria
mt search "autenticación JWT"

# Ver lecciones aprendidas
mt lessons --project mi-proyecto

# Ver estadísticas
mt stats --project mi-proyecto
```

### Onboarding de Proyectos Existentes

Si empiezas a trabajar en un proyecto que **ya existe** y no tiene historial en Memory Twin, puedes ejecutar un análisis inicial que crea una "memoria base" con la estructura, stack y convenciones del proyecto:

```bash
# Analizar el proyecto actual
mt onboard

# O especificar una ruta
mt onboard /ruta/a/mi-proyecto

# Ver el análisis completo
mt onboard --verbose
```

Esto genera un episodio de tipo "onboarding" que incluye:
- **Stack tecnológico** detectado (Python, Node.js, etc.)
- **Patrones arquitectónicos** (MVC, DDD, etc.)
- **Dependencias principales**
- **Convenciones** de linting, testing, etc.

El agente de IA puede consultar esta información para entender el proyecto desde el primer momento.

### Interfaz Web (Oráculo)

Para explorar la base de conocimiento visualmente:

```bash
# Iniciar interfaz web
python -m memorytwin.oraculo.app
# Abre http://localhost:7860
```

## 🧪 Desarrollo y Tests

Para asegurar que todo funciona correctamente, puedes ejecutar los tests:

```bash
# Instalar dependencias de test
pip install pytest pytest-asyncio

# Ejecutar tests
pytest
```

## 📁 Estructura del Proyecto

```
memorytwin/
├── src/memorytwin/
│   ├── escriba/            # Agente de Ingesta y CLI
│   ├── oraculo/            # Agente de Consulta y Web UI
│   ├── mcp_server/         # Servidor MCP
│   ├── models.py           # Modelos de datos
│   ├── config.py           # Configuración
│   └── observability.py    # Integración Langfuse
├── scripts/                # Scripts de utilidad
├── data/                   # Datos persistentes (modo local)
├── tests/                  # Tests unitarios y de integración
├── pyproject.toml          # Configuración del proyecto y dependencias
└── README.md
```

## 📄 Licencia

MIT License

---

