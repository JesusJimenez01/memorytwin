# Memory Twin - Instrucciones para Agentes IA

## ¿Qué es Memory Twin?
Sistema de memoria episódica que captura el razonamiento técnico para evitar "amnesia técnica" en proyectos.

## ⚡ COMPORTAMIENTO AUTOMÁTICO (IMPORTANTE)

### Antes de responder cualquier pregunta técnica:
1. **SIEMPRE** usar `query_memory` o `search_episodes` para buscar contexto relevante
2. Si hay memorias relacionadas, incorporar ese conocimiento en la respuesta
3. Mencionar brevemente si encontré información útil de sesiones anteriores

### Consultas automáticas recomendadas:
- Antes de sugerir una solución → buscar si ya resolvimos algo similar
- Antes de elegir una librería/enfoque → buscar decisiones previas
- Cuando el usuario menciona un error → buscar si lo vimos antes
- Al empezar una nueva feature → consultar lecciones aprendidas relevantes

### Ejemplo de flujo:
```
Usuario: "Tengo un error de autenticación"
Yo (internamente): 
  1. search_episodes("autenticación error")
  2. Encuentro que ya resolvimos algo similar
  3. Respondo incorporando ese contexto
```

## Herramientas MCP Disponibles

### `capture_thinking` - Capturar razonamiento
Usar cuando:
- ✅ Se resuelve un bug no trivial
- ✅ Se toma una decisión de arquitectura
- ✅ Se comparan alternativas y se elige una
- ✅ Se descubre algo inesperado (gotcha, edge case)
- ✅ El usuario dice "guarda esto", "recuerda esto", "captura esto"

NO usar cuando:
- ❌ Cambios triviales (typos, formateo)
- ❌ Preguntas simples sin razonamiento complejo
- ❌ Código boilerplate sin decisiones

### `query_memory` - Consultar memorias
Usar cuando:
- El usuario pregunta "¿por qué hicimos X?"
- El usuario pregunta "¿cómo resolvimos algo similar?"
- Antes de tomar una decisión importante (consultar precedentes)

### `get_lessons` - Lecciones aprendidas
Usar para:
- Onboarding de nuevos miembros
- Revisión antes de empezar feature similar
- El usuario pide "¿qué hemos aprendido sobre X?"

### `search_episodes` - Buscar episodios
Usar para búsquedas específicas de temas o tecnologías.

### `get_timeline` - Ver historial
Usar para ver evolución cronológica del proyecto.

## Flujo de Trabajo Recomendado

### Durante desarrollo:
1. Cuando resuelvas algo complejo → `capture_thinking` automáticamente
2. Incluir: contexto, alternativas consideradas, decisión final, lecciones

### Antes de empezar tarea:
1. `query_memory` para ver si hay contexto relevante
2. `get_lessons` para evitar errores pasados

### Al final de sesión (opcional):
El usuario puede pedir: "resume lo que trabajamos" → capturar resumen

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

## Proyecto Actual
- **Nombre del proyecto**: Usar el nombre de la carpeta del workspace
- **Source assistant**: "copilot" para GitHub Copilot
