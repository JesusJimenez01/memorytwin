# Memory Twin - Setup para nuevos proyectos
# ==========================================
# Ejecuta este script en la raíz de tu nuevo proyecto:
#   irm https://raw.githubusercontent.com/.../setup.ps1 | iex
# O simplemente copia el archivo copilot-instructions.md

param(
    [string]$ProjectPath = "."
)

$InstructionsContent = @'
# Memory Twin - Instrucciones para Agentes IA

## ¿Qué es Memory Twin?
Sistema de memoria episódica que captura el razonamiento técnico para evitar "amnesia técnica" en proyectos.

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

## Proyecto Actual
- **Nombre del proyecto**: Usar el nombre de la carpeta del workspace
- **Source assistant**: "copilot" para GitHub Copilot
'@

# Crear directorio .github si no existe
$GithubDir = Join-Path $ProjectPath ".github"
if (-not (Test-Path $GithubDir)) {
    New-Item -ItemType Directory -Path $GithubDir -Force | Out-Null
}

# Escribir archivo de instrucciones
$InstructionsPath = Join-Path $GithubDir "copilot-instructions.md"
$InstructionsContent | Out-File -FilePath $InstructionsPath -Encoding utf8

Write-Host "✓ Memory Twin configurado en: $InstructionsPath" -ForegroundColor Green
Write-Host ""
Write-Host "El agente ahora capturará razonamiento automáticamente en este proyecto." -ForegroundColor Cyan
