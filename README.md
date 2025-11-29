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

## 📦 Instalación Rápida

### Opción 1: Con pip (Recomendada)

```bash
# Clonar el repositorio
git clone https://github.com/JesusJimenez01/memorytwin.git
cd memorytwin

# Crear y activar entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Instalar Memory Twin
pip install -e .

# Configurar tu proyecto (crea .env, mcp.json, instrucciones)
mt setup

# Editar .env con tu API Key de Google Gemini
# Obtén una gratis en: https://aistudio.google.com/apikey
```

### Opción 2: Con uv (Más rápido)

```bash
# Instalar uv si no lo tienes
pip install uv

# Clonar e instalar
git clone https://github.com/JesusJimenez01/memorytwin.git
cd memorytwin
uv venv && uv pip install -e .

# Configurar
uv run mt setup
```

### ¡Listo!

Después de `mt setup`:
1. Edita `.env` y añade tu `GOOGLE_API_KEY`
2. Reinicia VS Code
3. Copilot ahora usa Memory Twin automáticamente 🧠

### Extras opcionales

```bash
# Interfaz web para explorar memorias
pip install -e ".[ui]"

# Todas las features
pip install -e ".[all]"

# Desarrollo (tests, linters)
pip install -e ".[all,dev]"
```

| Extra | Descripción |
|-------|-------------|
| `ui` | Interfaz web Gradio |
| `observability` | Langfuse para trazabilidad |
| `openai` | Soporte para GPT |
| `anthropic` | Soporte para Claude |
| `all` | Todo incluido |
| `dev` | Herramientas de desarrollo |

## 🔧 Configuración

El comando `mt setup` crea automáticamente:

| Archivo | Propósito |
|---------|-----------|
| `.env` | Tu API Key y configuración |
| `.vscode/mcp.json` | Integración con VS Code/Copilot |
| `.github/copilot-instructions.md` | Instrucciones para el agente |
| `.gitignore` | Ignora `.env` y `data/` |

### Variables de Entorno (.env)

```env
# Requerido: API Key de Google Gemini
# Obtén una gratis en: https://aistudio.google.com/apikey
GOOGLE_API_KEY=tu_api_key_aqui

# Opcional: Rutas de datos (por defecto usa ./data/)
# CHROMA_PERSIST_DIR=./data/chroma
# SQLITE_DB_PATH=./data/memory.db
```

## 🚀 Uso

### Integración con VS Code y Copilot

Después de `mt setup` y reiniciar VS Code:
- Copilot tendrá acceso a las herramientas de Memory Twin
- Usará automáticamente la memoria del proyecto
- Capturará decisiones técnicas importantes

#### Herramientas MCP Disponibles

| Herramienta | Descripción |
|-------------|-------------|
| `get_project_context` | ⭐ **Principal**. Obtiene contexto del proyecto |
| `capture_thinking` | Captura razonamiento de decisiones |
| `query_memory` | Consultas RAG: "¿Por qué elegimos X?" |
| `search_episodes` | Búsqueda semántica de episodios |
| `get_episode` | Contenido completo de un episodio |
| `get_lessons` | Lecciones aprendidas agregadas |
| `get_timeline` | Timeline cronológico |
| `get_statistics` | Estadísticas de la memoria |
| `onboard_project` | Análisis inicial de proyecto |
| `consolidate_memories` | Crear meta-memorias |
| `check_consolidation_status` | Verificar si necesita consolidación |
| `mark_episode` | Marcar antipatterns/críticos |

### CLI (Línea de Comandos)

```bash
# Configurar Memory Twin en tu proyecto
mt setup

# Buscar en la memoria
mt search "autenticación JWT"

# Consulta RAG (respuesta generada por LLM)
mt query "¿por qué elegimos JWT para autenticación?"

# Ver lecciones aprendidas
mt lessons --project mi-proyecto

# Ver estadísticas
mt stats

# Consolidar memorias (crea meta-memorias)
mt consolidate --project mi-proyecto

# Verificar salud del sistema
mt health-check

# Analizar proyecto existente
mt onboard /ruta/proyecto

# Capturar pensamiento desde archivo
mt capture --file thinking.txt --project mi-proyecto
```

### Interfaz Web (requiere `pip install -e ".[ui]"`)

```bash
python -m memorytwin.oraculo.app
# Abre http://localhost:7860
```

## 🧪 Tests

```bash
pip install -e ".[dev]"
pytest
```

## 📁 Estructura del Proyecto

```
memorytwin/
├── src/memorytwin/
│   ├── escriba/            # Ingesta y CLI
│   ├── oraculo/            # Consulta y Web UI
│   ├── mcp_server/         # Servidor MCP
│   ├── models.py           # Modelos de datos
│   ├── scoring.py          # Sistema de relevancia
│   ├── consolidation.py    # Meta-memorias
│   └── config.py           # Configuración
├── data/                   # Datos persistentes
├── tests/                  # Tests
└── pyproject.toml          # Dependencias
```

## 📈 Escalabilidad

| Backend | Escala | Uso |
|---------|--------|-----|
| **ChromaDB Local** | ~1,000 episodios | Individual |
| **ChromaDB Server** | ~10,000 episodios | Equipos |
| **PostgreSQL** | ~100,000+ | Producción |
3. **Caché**: Considera Redis para queries frecuentes
4. **Rate limiting**: Configura límites de API en producción

### Roadmap de escalabilidad

- [ ] Soporte PostgreSQL + pgvector
- [ ] Migraciones con Alembic
- [ ] Caché inteligente con Redis
- [ ] Rate limiting configurable
- [ ] Archivado automático de episodios antiguos

## 🧠 Memoria Cognitiva Avanzada

Memory Twin incluye características inspiradas en la neurociencia para gestionar la relevancia de las memorias.

### Sistema de Refuerzo (Sin Olvido)

A diferencia de sistemas que penalizan memorias antiguas, Memory Twin usa un enfoque de **"refuerzo sin olvido"**: todas las memorias persisten indefinidamente, pero las más consultadas ganan relevancia.

```
final_score = semantic_score × boost × importance_score × modifiers
```

| Factor | Fórmula | Descripción |
|--------|---------|-------------|
| `semantic_score` | Similitud coseno | Relevancia semántica con la query |
| `boost` | `1 + 0.1 × accesos` | Episodios consultados frecuentemente se refuerzan |
| `importance_score` | 0.0 - 1.0 | Relevancia base del episodio |
| `critical_modifier` | 1.5x | Episodios marcados como críticos |
| `antipattern_modifier` | 0.3x | Antipatterns aparecen al final, no se excluyen |

**Beneficios del enfoque:**
- ✅ Las memorias antiguas pero valiosas nunca se "olvidan"
- ✅ El uso frecuente refuerza naturalmente lo importante
- ✅ Los antipatterns siguen visibles como advertencias
- ✅ Las meta-memorias consolidan patrones recurrentes

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

- [ ] Comando `mt backup/restore` para backups
- [ ] Comando `mt rebuild-embeddings` para regenerar vectores
- [ ] Transacciones atómicas SQLite + ChromaDB
- [ ] Fallback a modelo local (Ollama)
- [ ] Migraciones con Alembic

## 📄 Licencia

MIT License

---

