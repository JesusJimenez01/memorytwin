# Memory Twin - Instrucciones para Agentes IA

---
## 🚨 REGLAS OBLIGATORIAS - LEER PRIMERO 🚨

### REGLA 1: CONSULTAR MEMORIA AL INICIO
**ANTES de responder CUALQUIER pregunta técnica, DEBO ejecutar:**

```
get_project_context(topic="<términos relevantes de la pregunta del usuario>")
```

### REGLA 2: CAPTURAR MEMORIA AL FINAL
**DESPUÉS de completar CUALQUIER tarea técnica significativa, DEBO ejecutar:**

```
capture_thinking(thinking_text="<mi razonamiento completo>", project_name="<proyecto>")
```

**NO HAY EXCEPCIONES PARA NINGUNA DE LAS DOS REGLAS.**

Si no consulto la memoria primero:
- Puedo sugerir soluciones que ya fallaron
- Puedo ignorar decisiones de arquitectura previas
- Puedo repetir errores que ya resolvimos

Si no capturo la memoria después:
- El conocimiento se pierde para siempre
- El siguiente agente repetirá el mismo trabajo
- El proyecto no aprende de las decisiones tomadas

**Mi flujo SIEMPRE es: CONSULTAR → TRABAJAR → CAPTURAR**

---

## ¿Qué es Memory Twin?
Sistema de memoria episódica que captura el razonamiento técnico para evitar "amnesia técnica" en proyectos.

## ⚡ PROTOCOLO DE CONSULTA OBLIGATORIO

### PASO 1: Obtener contexto (OBLIGATORIO)
Antes de escribir CUALQUIER respuesta técnica:

```
get_project_context(topic="<palabras clave de la pregunta>")
```

Esta herramienta es INTELIGENTE:
- Si hay **pocas memorias (<20)**: devuelve TODO el contexto del proyecto
- Si hay **muchas memorias (>=20)**: devuelve estadísticas + recientes + relevantes al topic

Ejemplos de consultas:
- Usuario pregunta sobre autenticación → `get_project_context(topic="autenticación login JWT")`
- Usuario tiene un error → `get_project_context(topic="error <tipo de error>")`
- Usuario quiere implementar feature → `get_project_context(topic="<nombre feature>")`
- Usuario pregunta arquitectura → `get_project_context(topic="arquitectura diseño")`

### PASO 2: Analizar resultados
- Si hay memorias relevantes → Incorporar ese conocimiento
- Si no hay resultados → Proceder normalmente, pero considerar capturar después

### PASO 3: Responder mencionando contexto
- "Según memorias previas del proyecto..."
- "No encontré decisiones previas sobre esto, procedo con..."
- "Esto ya se resolvió anteriormente, la solución fue..."

### Cuándo consultar (SIEMPRE que aplique):
| Situación | Consulta obligatoria |
|-----------|---------------------|
| Pregunta técnica | `get_project_context(topic="<tema>")` |
| Error/Bug | `get_project_context(topic="error <descripción>")` |
| Nueva feature | `get_project_context(topic="<feature>")` + `get_lessons()` |
| Decisión de arquitectura | `query_memory("<pregunta>")` |
| Primera vez en proyecto | `onboard_project("<ruta>")` |
| Elegir librería/enfoque | `get_project_context(topic="<opciones>")` |

## Herramientas MCP Disponibles

### `get_project_context` - ⭐ HERRAMIENTA PRINCIPAL
**Usar SIEMPRE antes de cualquier respuesta técnica.**

Comportamiento inteligente:
- **Pocas memorias (<20)**: Devuelve TODO el contexto completo
- **Muchas memorias (>=20)**: Devuelve estadísticas + 5 recientes + 5 relevantes al topic

Parámetros:
- `topic` (opcional): Tema para búsqueda semántica
- `project_name` (opcional): Filtrar por proyecto

### `capture_thinking` - 🔴 CAPTURA OBLIGATORIA
**DEBO usar esta herramienta AUTOMÁTICAMENTE después de cualquier tarea técnica.**

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
