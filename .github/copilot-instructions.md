# Memory Twin - Instrucciones para Agentes IA

---
## 🚨 FLUJO OBLIGATORIO PARA CADA TAREA 🚨

```
┌─────────────────────────────────────────────────────────────────┐
│  1. CONSULTAR    →   2. PLANIFICAR   →   3. EJECUTAR   →   4. REGISTRAR  │
│     memorias          con contexto        el plan            resultado     │
└─────────────────────────────────────────────────────────────────┘
```

### PASO 1: CONSULTAR MEMORIAS (OBLIGATORIO)
**ANTES de responder CUALQUIER tarea técnica:**

```
get_project_context(topic="<términos de la tarea>", include_reasoning=true)
```

⚠️ **SI HAY WARNINGS DE ANTIPATTERNS**: 
- DEBES leerlos antes de continuar
- Si decides ignorarlos, JUSTIFICA por qué

### PASO 2: PLANIFICAR CON CONTEXTO
Basándote en las memorias recuperadas:
- ¿Hay soluciones previas que puedo reutilizar?
- ¿Hay errores previos que debo evitar?
- ¿Qué lecciones aplican a esta tarea?

### PASO 3: EJECUTAR EL PLAN
Implementa la solución teniendo en cuenta el contexto.

### PASO 4: REGISTRAR RESULTADO (OBLIGATORIO)
**DESPUÉS de cada tarea técnica:**

```
capture_thinking(
    thinking_text="## Tarea\n...\n## Decisiones\n...\n## Resultado\n...",
    project_name="<proyecto>"
)
```

Si algo FALLÓ o fue un error, también usa:
```
mark_episode(episode_id="<id>", is_antipattern=true)
```

---

## ¿Qué es Memory Twin?
Sistema de memoria episódica que captura el razonamiento técnico para evitar "amnesia técnica" en proyectos.

## ⚡ PROTOCOLO DE CONSULTA

### `get_project_context` - ⭐ HERRAMIENTA PRINCIPAL
**Usar SIEMPRE al inicio de cada tarea.**

Priorización de resultados:
0. ⚠️ **ANTIPATTERNS**: Advertencias de errores previos (revisar OBLIGATORIO)
1. **META-MEMORIAS**: Conocimiento consolidado y patrones
2. **EPISODIOS**: Decisiones individuales relevantes

Parámetros:
- `topic` (recomendado): Palabras clave de la tarea
- `include_reasoning` (opcional): Si true, incluye razonamiento completo
- `project_name` (opcional): Filtrar por proyecto

Ejemplos:
- Usuario pregunta sobre autenticación → `get_project_context(topic="autenticación login JWT")`
- Usuario tiene un error → `get_project_context(topic="error <tipo de error>")`
- Usuario quiere implementar feature → `get_project_context(topic="<nombre feature>")`

### PASO 2: Analizar resultados
- **Si hay WARNINGS**: Leerlos ANTES de proceder
- Si hay memorias relevantes → Incorporar ese conocimiento
- Si no hay resultados → Proceder, pero capturar después

### PASO 3: Responder mencionando contexto
- "Según memorias previas del proyecto..."
- "⚠️ Hay un antipattern registrado sobre esto, evitaré..."
- "No encontré decisiones previas, procedo con..."

### Cuándo consultar (SIEMPRE que aplique):
| Situación | Consulta obligatoria |
|-----------|---------------------|
| Pregunta técnica | `get_project_context(topic="<tema>")` |
| Error/Bug | `get_project_context(topic="error <descripción>")` |
| Nueva feature | `get_project_context(topic="<feature>")` |
| Decisión de arquitectura | `get_project_context(topic="arquitectura", include_reasoning=true)` |
| Primera vez en proyecto | `onboard_project("<ruta>")` |

## Herramientas MCP Disponibles

### `get_project_context` - ⭐ HERRAMIENTA PRINCIPAL
**Usar SIEMPRE antes de cualquier respuesta técnica.**

Comportamiento inteligente:
- **Pocas memorias (<20)**: Devuelve TODO el contexto completo
- **Muchas memorias (>=20)**: Devuelve estadísticas + 5 recientes + 5 relevantes al topic

**Nota Importante:**
Esta herramienta activa el mecanismo de **"Forgetting Curve"**. Al consultar memorias relevantes, incrementas su `access_count`, evitando que el sistema las "olvide" con el tiempo. ¡Consultar es reforzar!

Parámetros:
- `topic` (opcional): Tema para búsqueda semántica
- `include_reasoning` (opcional): Incluir raw_thinking completo
- `project_name` (opcional): Filtrar por proyecto

### `capture_thinking` - 🔴 CAPTURA OBLIGATORIA
**DEBO usar esta herramienta AUTOMÁTICAMENTE después de cualquier tarea técnica.**

**IMPORTANTE:** Esta es la **ÚNICA** forma de persistir conocimiento. Si no la uso, el trabajo se pierde.

#### ✅ CAPTURAR SIEMPRE (sin excepción):
- Resolví un bug o error (cualquiera, no importa si es "simple")
- Tomé una decisión técnica (librería, patrón, enfoque)
- Comparé alternativas antes de elegir
- Descubrí algo inesperado (gotcha, edge case, comportamiento raro)
- Modifiqué código existente (refactor, mejora, fix)
- Implementé una feature nueva
- Configuré algo (entorno, herramientas, dependencias)
- Investigué documentación o código para entender algo
- El usuario me pidió hacer algo y lo completé
- Encontré un problema en documentación/código y lo corregí

#### ❌ NO capturar SOLO cuando:
- Respuesta puramente informativa sin acción (ej: "¿qué hora es?")
- Conversación casual sin contenido técnico
- El usuario explícitamente dice "no guardes esto"

#### 🎯 REGLA DE ORO: Ante la duda, CAPTURAR
Es mejor tener una memoria "de más" que perder conocimiento valioso.

Parámetros:
- `thinking_text` (requerido): Texto de razonamiento del modelo
- `user_prompt` (opcional): Prompt original del usuario
- `code_changes` (opcional): Cambios de código asociados
- `source_assistant` (opcional): copilot, claude, cursor, etc.
- `project_name` (opcional): Nombre del proyecto

### `query_memory` - Consultar memorias con RAG
Usar cuando:
- El usuario pregunta "¿por qué hicimos X?"
- El usuario pregunta "¿cómo resolvimos algo similar?"
- Antes de tomar una decisión importante (consultar precedentes)

Parámetros:
- `question` (requerido): Pregunta a responder
- `project_name` (opcional): Filtrar por proyecto
- `num_episodes` (opcional): Número de episodios a consultar (1-10, default: 5)

### `search_episodes` - Búsqueda semántica de episodios
Usar para búsquedas específicas de temas o tecnologías.
Devuelve los episodios más relevantes para un término de búsqueda.
*Nota: Los resultados consultados reciben un boost de relevancia para el futuro.*

Parámetros:
- `query` (requerido): Término de búsqueda
- `project_name` (opcional): Filtrar por proyecto
- `top_k` (opcional): Número de resultados (default: 5)

### `get_episode` - Obtener episodio completo
Usar cuando necesitas profundizar en los detalles de una decisión específica.
Devuelve el contenido COMPLETO: thinking, alternativas, factores de decisión, contexto y lecciones.

Parámetros:
- `episode_id` (requerido): UUID del episodio a recuperar

### `get_lessons` - Lecciones aprendidas
Usar para:
- Onboarding de nuevos miembros
- Revisión antes de empezar feature similar
- El usuario pide "¿qué hemos aprendido sobre X?"

Parámetros:
- `project_name` (opcional): Filtrar por proyecto
- `tags` (opcional): Array de tags para filtrar

### `get_timeline` - Ver historial cronológico
Usar para ver evolución cronológica del proyecto y entender qué se hizo cuándo.

Parámetros:
- `project_name` (opcional): Filtrar por proyecto
- `limit` (opcional): Máximo de episodios a retornar (default: 20)

### `get_statistics` - Estadísticas de la memoria
Obtiene estadísticas de la base de memoria: total de episodios, distribución por tipo y asistente.

Parámetros:
- `project_name` (opcional): Filtrar por proyecto

### `mark_episode` - 🚨 Marcar episodios como antipattern o crítico
**Usar SIEMPRE que algo haya fallado o sea un error a evitar.**

Permite marcar episodios existentes como:
- **Antipattern**: Errores, fallos, enfoques que NO funcionaron
- **Critical**: Decisiones importantes que deben preservarse siempre

También permite marcar episodios como superseded (reemplazados por uno nuevo).

Parámetros:
- `episode_id` (requerido): UUID del episodio a marcar
- `is_antipattern` (opcional): true si es un error a evitar
- `is_critical` (opcional): true si es conocimiento crítico
- `superseded_by` (opcional): UUID del episodio que lo reemplaza
- `deprecation_reason` (opcional): Razón por la que ya no aplica

Ejemplo de uso después de un error:
```
mark_episode(episode_id="abc-123", is_antipattern=true)
```

### `onboard_project` - Onboarding de proyecto existente
Usar cuando:
- ✅ Es la primera vez que trabajo en este proyecto
- ✅ El usuario pide "analiza el proyecto", "conoce el código"
- ✅ Necesito entender la estructura antes de hacer cambios grandes
- ✅ No hay memorias previas y quiero crear contexto inicial

Genera automáticamente un episodio con:
- Stack tecnológico detectado
- Patrones arquitectónicos
- Dependencias principales
- Convenciones de código

Parámetros:
- `project_path` (requerido): Ruta absoluta al proyecto
- `project_name` (opcional): Nombre del proyecto (se detecta automáticamente)

### `check_consolidation_status` - Verificar necesidad de consolidación
Usar para:
- Determinar si hay suficientes episodios para consolidar
- Ver estadísticas de episodios con alto access_count
- Decidir si ejecutar `consolidate_memories`

Parámetros:
- `project_name` (opcional): Proyecto a verificar

### `consolidate_memories` - Consolidar episodios en meta-memorias
Usar cuando:
- El sistema indica que hay episodios sin consolidar
- Hay muchos episodios (>20) en un proyecto
- Quieres crear conocimiento consolidado de patrones recurrentes

Parámetros:
- `project_name` (requerido): Proyecto a consolidar
- `min_cluster_size` (opcional): Mínimo de episodios por cluster (default: 3)

## Flujo de Trabajo OBLIGATORIO

### 🔄 CICLO COMPLETO (SIEMPRE):
```
1. INICIO: get_project_context(topic="...") 
2. TRABAJO: Realizar la tarea solicitada
3. FIN: capture_thinking(thinking_text="...") 
```

### Checklist antes de terminar respuesta:
- [ ] ¿Consulté la memoria al inicio? Si no → HACERLO AHORA
- [ ] ¿Hice algo técnico? Si sí → CAPTURAR MEMORIA
- [ ] ¿Resolví un problema? Si sí → CAPTURAR MEMORIA  
- [ ] ¿Tomé una decisión? Si sí → CAPTURAR MEMORIA
- [ ] ¿Modifiqué código? Si sí → CAPTURAR MEMORIA
- [ ] ¿Investigué algo? Si sí → CAPTURAR MEMORIA

### ⚠️ RECORDATORIO CRÍTICO:
**NO terminar una respuesta técnica sin haber ejecutado `capture_thinking`.**
El conocimiento que no se captura, SE PIERDE PARA SIEMPRE.

## Formato del Thinking a Capturar

Incluir siempre que sea posible:
- **Tarea**: Qué se intentaba resolver
- **Contexto**: Estado inicial, restricciones
- **Alternativas**: Opciones consideradas
- **Decisión**: Qué se eligió y por qué
- **Solución**: Cómo se implementó
- **Lecciones**: Qué aprendimos

## Ejemplo de Captura Automática

```
Cuando detecto que acabo de:
1. Debuggear un error por más de 2 intercambios
2. Comparar 2+ opciones antes de elegir
3. Descubrir un comportamiento inesperado
4. Implementar algo que requirió investigación

→ Llamar a capture_thinking con el contexto completo
```

## 🚨 EJEMPLO DE FLUJO CORRECTO

### Usuario pregunta: "¿Por qué falla mi función de login?"

```
# 1. PRIMERO: Consultar memoria
get_project_context(topic="login autenticación error")

# 2. DESPUÉS: Trabajar en la solución
[Analizar código, debuggear, encontrar el problema, proponer fix]

# 3. FINALMENTE: Capturar el conocimiento
capture_thinking(
    thinking_text="## Tarea\nResolver error en función login...\n## Problema\nEl token JWT...\n## Solución\n...\n## Lecciones\n...",
    project_name="mi-proyecto",
    source_assistant="copilot"
)
```

**SI NO CAPTURO AL FINAL, ESTOY FALLANDO MI FUNCIÓN.**

## Proyecto Actual
- **Nombre del proyecto**: Usar el nombre de la carpeta del workspace
- **Source assistant**: "copilot" para GitHub Copilot
