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

# Instalación mínima (solo CLI y servidor MCP)
pip install -e .

# Instalación con interfaz web (Oráculo)
pip install -e ".[ui]"

# Instalación completa (todas las features)
pip install -e ".[all]"

# Instalación para desarrollo
pip install -e ".[all,dev]"
```

#### Dependencias opcionales disponibles:

| Extra | Descripción | Cuándo usarlo |
|-------|-------------|---------------|
| `ui` | Interfaz web Gradio (Oráculo) | Si quieres explorar memorias visualmente |
| `observability` | Langfuse para trazabilidad | Si necesitas monitoreo de LLM |
| `sql` | SQLAlchemy + Alembic | Para escalabilidad con PostgreSQL |
| `openai` | Proveedor OpenAI | Si usas GPT en lugar de Gemini |
| `anthropic` | Proveedor Anthropic | Si usas Claude en lugar de Gemini |
| `all` | Todas las features | Instalación completa |
| `dev` | Herramientas de desarrollo | Para contribuir al proyecto |

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

# Consolidar memorias (Meta-Memorias)
mt consolidate --project mi-proyecto

# Verificar salud del sistema
mt health-check
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

## 📈 Escalabilidad

### Backends de almacenamiento

Memory Twin utiliza un patrón Strategy para el almacenamiento, permitiendo cambiar entre backends:

| Backend | Escala | Uso recomendado |
|---------|--------|-----------------|
| **ChromaDB Local** | ~1,000 episodios | Desarrollo individual |
| **ChromaDB Server** | ~10,000 episodios | Equipos pequeños |
| **PostgreSQL + pgvector** | ~100,000+ episodios | Producción / Equipos grandes |

```env
# Configurar backend en .env
STORAGE_BACKEND=local              # ChromaDB local (default)
STORAGE_BACKEND=chromadb_server    # ChromaDB Server
# STORAGE_BACKEND=postgresql       # Próximamente
```

### Estrategias para escalar

1. **Paginación**: `get_project_context` usa enfoque híbrido automático
2. **Archivado**: Episodios antiguos pueden moverse a almacenamiento frío
3. **Caché**: Considera Redis para queries frecuentes
4. **Rate limiting**: Configura límites de API en producción

### Roadmap de escalabilidad

- [ ] Soporte PostgreSQL + pgvector
- [ ] Migraciones con Alembic
- [ ] Caché inteligente con Redis
- [ ] Rate limiting configurable
- [ ] Archivado automático de episodios antiguos

## 🧠 Memoria Cognitiva Avanzada

Memory Twin incluye características inspiradas en la neurociencia para simular el comportamiento de la memoria humana.

### Curva de Olvido (Forgetting Curve)

Inspirada en la curva de olvido de Ebbinghaus, los episodios tienen un **score híbrido** que combina:

```
final_score = semantic_score × decay × boost × importance_score
```

| Factor | Fórmula | Descripción |
|--------|---------|-------------|
| `semantic_score` | Similitud coseno | Relevancia semántica con la query |
| `decay` | `exp(-0.05 × días)` | Decaimiento temporal (episodios viejos se "olvidan") |
| `boost` | `1 + 0.1 × accesos` | Episodios consultados frecuentemente se refuerzan |
| `importance_score` | 0.0 - 1.0 | Relevancia base del episodio |

**Ejemplo práctico:**
- Un episodio de hace 30 días tiene ~22% de "frescura" (`exp(-0.05 × 30) ≈ 0.22`)
- Si fue consultado 10 veces, obtiene un boost de 2x (`1 + 0.1 × 10 = 2.0`)
- Resultado: se mantiene relevante a pesar del tiempo

### Meta-Memorias (Consolidación)

Similar a la consolidación de la memoria durante el sueño, Memory Twin puede **consolidar episodios relacionados en meta-memorias**:

```bash
# Consolidar episodios de un proyecto
mt consolidate --project mi-proyecto

# Con más detalle
mt consolidate --project mi-proyecto --verbose

# Ajustar mínimo de episodios por cluster
mt consolidate --project mi-proyecto --min-cluster 5
```

Una **MetaMemory** representa conocimiento consolidado:

| Campo | Descripción |
|-------|-------------|
| `pattern` | Patrón común identificado |
| `lessons` | Lecciones aprendidas consolidadas |
| `best_practices` | Mejores prácticas derivadas |
| `antipatterns` | Errores comunes a evitar |
| `exceptions` | Casos donde el patrón no aplica |
| `edge_cases` | Casos límite descubiertos |
| `confidence` | Confianza en la consolidación (0-1) |
| `source_episode_ids` | IDs de episodios fuente |

**Proceso de consolidación:**
1. **Clustering**: Agrupa episodios similares usando DBSCAN sobre embeddings
2. **Síntesis**: Un LLM analiza el cluster y extrae patrones comunes
3. **Almacenamiento**: La meta-memoria se guarda con trazabilidad a episodios fuente

### Integración en RAG

El sistema RAG prioriza las meta-memorias sobre episodios individuales:

1. **Buscar en meta-memorias** (conocimiento consolidado, más confiable)
2. **Complementar con episodios** (detalles específicos)
3. **Combinar contexto** para generar respuesta

## 🛡️ Resiliencia y Recuperación de Errores

### Fallos de API de LLM

Memory Twin incluye estrategias de retry automático para llamadas a LLM:

```python
# Configuración actual en processor.py
@retry(
    stop=stop_after_attempt(3),           # Máximo 3 intentos
    wait=wait_exponential(min=2, max=10)  # Espera exponencial: 2s, 4s, 8s
)
async def process_thought(...):
```

**Configuración recomendada en `.env`:**

```env
# Rate limiting (próximamente)
LLM_MAX_REQUESTS_PER_MINUTE=60
LLM_TIMEOUT_SECONDS=30

# Fallback a modelo local (próximamente)
LLM_FALLBACK_ENABLED=true
LLM_FALLBACK_MODEL=ollama/llama3
```

### Consistencia de datos

Memory Twin usa almacenamiento dual (ChromaDB + SQLite). Para evitar inconsistencias:

```bash
# Verificar integridad de la base de datos
mt health-check
```

**Roadmap de mantenimiento (Próximamente):**

```bash
# Sincronizar ChromaDB con SQLite
mt sync --repair

# Backup completo (SQLite + ChromaDB)
mt backup --output ./backups/$(date +%Y%m%d).tar.gz

# Restaurar desde backup
mt restore --input ./backups/20251127.tar.gz
```

### Recuperación de embeddings

Si los embeddings se corrompen o cambias de modelo (Próximamente):

```bash
# Regenerar todos los embeddings desde SQLite
mt rebuild-embeddings

# Regenerar solo para un proyecto específico
mt rebuild-embeddings --project mi-proyecto
```

### Migración de schemas

Para futuras migraciones de base de datos:

```bash
# Instalar dependencia de migraciones
pip install -e ".[sql]"

# Crear nueva migración
alembic revision --autogenerate -m "descripcion"

# Aplicar migraciones pendientes
alembic upgrade head

# Rollback a versión anterior
alembic downgrade -1
```

### Roadmap de resiliencia

- [x] Retry automático con exponential backoff (LLM)
- [x] Comando `mt health-check` para verificar integridad
- [ ] Comando `mt backup/restore` para backups
- [ ] Comando `mt rebuild-embeddings` para regenerar vectores
- [ ] Transacciones atómicas SQLite + ChromaDB
- [ ] Fallback a modelo local (Ollama)
- [ ] Migraciones con Alembic

## 📄 Licencia

MIT License

---

