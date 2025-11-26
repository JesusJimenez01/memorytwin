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

## 📦 Instalación

### Requisitos Previos

- Python 3.10+
- API Key de Google (Gemini)
- (Opcional) Docker (para modo servidor compartido)
- (Opcional) Cuenta en Langfuse para observabilidad

### Instalación Rápida

```bash
# Clonar el repositorio
cd memorytwin

# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# O instalar como paquete editable
pip install -e .

# Configurar variables de entorno
copy .env.example .env
# Editar .env con tus API keys
```

### Configuración

Edita el archivo `.env` con tus credenciales y preferencias:

```env
# Requerido
GOOGLE_API_KEY=tu_api_key_de_gemini

# Configuración de Almacenamiento
# Opciones: 'local' (por defecto) o 'chromadb_server'
STORAGE_BACKEND=local

# Si usas STORAGE_BACKEND=chromadb_server
CHROMA_SERVER_HOST=localhost
CHROMA_SERVER_PORT=8000

# Opcional - Langfuse para observabilidad
LANGFUSE_PUBLIC_KEY=tu_public_key
LANGFUSE_SECRET_KEY=tu_secret_key
```

## 🚀 Uso

### 1. Integración con VS Code y Copilot (Recomendado)

Memory Twin expone sus herramientas a través del protocolo MCP.

1.  Asegúrate de tener el archivo `mcp.json` configurado en tu carpeta `.vscode` global o del proyecto.
2.  Copia el archivo `.github/copilot-instructions.md` a tu proyecto para que Copilot sepa cuándo usar la memoria.

**Flujo Automático:**
- Copilot detectará decisiones complejas o errores y usará `capture_thinking` automáticamente.
- Antes de responder dudas técnicas, consultará `query_memory` para ver si ya se resolvió antes.

### 2. Trabajo en Equipo (Base de Datos Compartida)

Para que todo el equipo comparta la misma memoria:

1.  Levanta el servidor de ChromaDB con Docker:
    ```bash
    docker run -p 8000:8000 chromadb/chroma
    ```
2.  Configura `.env` en las máquinas de todos los desarrolladores:
    ```env
    STORAGE_BACKEND=chromadb_server
    CHROMA_SERVER_HOST=ip_del_servidor
    CHROMA_SERVER_PORT=8000
    ```
3.  (Opcional) Migra tus memorias locales al servidor:
    ```bash
    python scripts/migrate_to_server.py
    ```

### 3. Escriba - Capturar Pensamientos (CLI)

Si has añadido `scripts/mt.bat` a tu PATH, puedes usar el comando `mt` desde cualquier lugar.

```bash
# Capturar desde archivo
mt capture --file thinking.txt --assistant copilot --project mi-proyecto

# Capturar desde clipboard
mt capture --clipboard --assistant claude

# Ver estadísticas
mt stats --project mi-proyecto

# Buscar en memoria
mt search "autenticación JWT"

# Ver lecciones aprendidas
mt lessons --project mi-proyecto
```

### 4. Oráculo - Visualizar Memorias

Interfaz web para explorar la base de conocimiento.

```bash
# Iniciar interfaz web
python -m memorytwin.oraculo.app
# Abre http://localhost:7860
```

## 📊 Observabilidad con Langfuse

Para habilitar trazabilidad completa:

1. Crea una cuenta en [Langfuse](https://langfuse.com)
2. Configura las API keys en `.env`
3. Las trazas se enviarán automáticamente

## 📁 Estructura del Proyecto

```
memorytwin/
├── src/memorytwin/
│   ├── __init__.py
│   ├── models.py           # Modelos Pydantic
│   ├── config.py           # Configuración centralizada
│   ├── observability.py    # Integración Langfuse
│   ├── escriba/            # Agente de Ingesta
│   │   ├── __init__.py
│   │   ├── processor.py    # Procesamiento LLM
│   │   ├── storage_interface.py # Interfaz Strategy
│   │   ├── storage.py      # Backend Local
│   │   ├── storage_chromadb_server.py # Backend Servidor
│   │   ├── escriba.py      # Agente principal
│   │   └── cli.py          # CLI
│   ├── oraculo/            # Agente de Consulta
│   │   ├── __init__.py
│   │   ├── rag_engine.py   # Motor RAG
│   │   ├── oraculo.py      # Agente principal
│   │   └── app.py          # Interfaz Gradio
│   └── mcp_server/         # Servidor MCP
│       ├── __init__.py
│       └── server.py
├── scripts/                # Scripts de utilidad (migración, setup)
├── data/                   # Datos persistentes (modo local)
├── tests/
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

## 🔄 Flujo de Trabajo Típico

1. **Durante el desarrollo**: Copilot detecta una decisión clave y la guarda automáticamente.
2. **Revisión**: Usas `mt lessons` para ver qué aprendió el equipo recientemente.
3. **Consulta**: Abres el Oráculo para entender por qué se tomó una decisión hace meses.
4. **Onboarding**: Nuevo miembro revisa el timeline del proyecto en el Oráculo.

## 📝 Esquema de Episodio

Cada episodio de memoria contiene:

```json
{
  "id": "uuid",
  "timestamp": "2024-01-15T10:30:00",
  "task": "Implementar autenticación JWT",
  "context": "Módulo auth/ en proyecto FastAPI",
  "reasoning_trace": {
    "raw_thinking": "Texto del razonamiento...",
    "alternatives_considered": ["Sessions", "OAuth2"],
    "decision_factors": ["Stateless", "Escalabilidad"],
    "confidence_level": 0.85
  },
  "solution": "Código implementado...",
  "solution_summary": "JWT con PyJWT, tokens de 24h",
  "episode_type": "feature",
  "tags": ["auth", "security", "jwt"],
  "lessons_learned": ["Validar siempre el algoritmo JWT"],
  "source_assistant": "copilot",
  "project_name": "mi-proyecto"
}
```

## 🛣️ Roadmap

- [x] **Fase 1**: Prototipo con CLI y Gradio
- [x] **Fase 2**: Servidor MCP para integración
- [x] **Fase 3**: Soporte para Base de Datos Compartida (Team Mode)
- [x] **Fase 4**: Automatización con Copilot Instructions
- [ ] **Fase 5**: Dashboard de analytics avanzado
- [ ] **Fase 6**: Extensión nativa de VS Code

## 📄 Licencia

MIT License

---

**Memory Twin** - Porque el conocimiento del equipo no debería perderse con cada sesión de desarrollo.
