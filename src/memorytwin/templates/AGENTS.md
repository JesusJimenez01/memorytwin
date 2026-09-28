# Memory Twin - Instructions for AI Agents

---
## 🚨 MANDATORY RULE - READ FIRST 🚨

**BEFORE answering ANY technical question, I MUST run:**

```
get_project_context(topic="<relevant terms from the user's question>")
```

**NO EXCEPTIONS.** If I don't check memory first, I may:
- Suggest solutions that already failed
- Ignore previous architecture decisions
- Repeat errors we already solved
- Contradict the project's style/conventions

**My first action is ALWAYS to get project context. ALWAYS.**

## 🛠️ Tool Availability
These instructions assume the MemoryTwin MCP server is active.
- If the `get_project_context` tool fails or is not available:
1. NOTIFY ME immediately: "⚠️ I cannot access my memory. Check the MCP server."
2. Ask me to paste the context manually if it's critical.

---

## What is Memory Twin?
Episodic memory system that captures technical reasoning to prevent "technical amnesia" in projects.

## ⚡ MANDATORY QUERY PROTOCOL

### STEP 1: Get context (MANDATORY)
Before writing ANY technical response:

```
get_project_context(topic="<keywords from the question>")
```

This tool is SMART:
- If there are **few memories (<20)**: returns ALL project context
- If there are **many memories (>=20)**: returns statistics + recent + relevant to the topic

**Important Note:**
This tool activates the **"Forgetting Curve"** mechanism. By querying relevant memories, you increase their `access_count`, preventing the system from "forgetting" them over time. Querying is reinforcing!

Query examples:
- User asks about authentication → `get_project_context(topic="authentication login JWT")`
- User has an error → `get_project_context(topic="error <error type>")`
- User wants to implement a feature → `get_project_context(topic="<feature name>")`
- User asks about architecture → `get_project_context(topic="architecture design")`

### STEP 2: Analyze results
- If there are relevant memories → Incorporate that knowledge
- If no results → Proceed normally, but consider capturing afterwards

### STEP 3: Respond mentioning context
- "According to previous project memories..."
- "I found no previous decisions about this, proceeding with..."
- "This was already solved before, the solution was..."

### When to query (ALWAYS when applicable):
| Situation | Mandatory query |
|-----------|---------------------|
| Technical question | `get_project_context(topic="<topic>")` |
| Error/Bug | `get_project_context(topic="error <description>")` |
| New feature | `get_project_context(topic="<feature>")` + `get_lessons()` |
| Architecture decision | `query_memory("<question>")` |
| First time in project | `onboard_project("<path>")` |
| Choose library/approach | `get_project_context(topic="<options>")` |

## Available MCP Tools

### `get_project_context` - ⭐ MAIN TOOL
**ALWAYS use before any technical response.**

Smart behavior:
- **Few memories (<20)**: Returns ALL complete context
- **Many memories (>=20)**: Returns statistics + 5 recent + 5 relevant to the topic

**Important Note:**
This tool activates the **"Forgetting Curve"** mechanism. By querying relevant memories, you increase their `access_count`, preventing the system from "forgetting" them over time. Querying is reinforcing!

Parameters:
- `topic` (optional): Topic for semantic search
- `project_name` (optional): Filter by project

### `capture_thinking` - 🔴 MANDATORY CAPTURE
**I MUST use this tool AUTOMATICALLY after any technical task.**

**IMPORTANT:** This is the **ONLY** way to persist knowledge. If I don't use it, the work is lost.

**💡 TIP:** There are 3 ways to capture, choose the most convenient:
1. `capture_quick` - ⚡ The fastest (only what + why)
2. `capture_decision` - 🎯 For decisions (task + decision + reasoning)
3. `capture_thinking` - 📝 For extensive free text

#### ✅ ALWAYS CAPTURE (no exceptions):
- Fixed a bug or error (any, no matter if it's "simple")
- Made a technical decision (library, pattern, approach)
- Compared alternatives before choosing
- Discovered something unexpected (gotcha, edge case, weird behavior)
- Modified existing code (refactor, improvement, fix)
- Implemented a new feature
- Configured something (environment, tools, dependencies)
- Researched documentation or code to understand something
- The user asked me to do something and I completed it
- Found a problem in documentation/code and corrected it

#### ❌ Do NOT capture ONLY when:
- Purely informational response without action (e.g., "what time is it?")
- Casual conversation without technical content
- The user explicitly says "don't save this"

#### 🎯 GOLDEN RULE: When in doubt, CAPTURE
It's better to have an "extra" memory than to lose valuable knowledge.

Parameters:
- `thinking_text` (required): Model reasoning text
- `user_prompt` (optional): Original user prompt
- `code_changes` (optional): Associated code changes
- `source_assistant` (optional): copilot, claude, cursor, etc.
- `project_name` (optional): Project name

### `capture_decision` - 🎯 STRUCTURED CAPTURE (PREFERRED)
**Most convenient way to capture technical decisions.**

Use when you have data organized in separate fields. More convenient than writing free text.

Parameters:
- `task` (required): Brief description of the task or problem
- `decision` (required): The decision or solution taken
- `reasoning` (required): Why this decision was made
- `alternatives` (optional): Array of alternatives considered
- `lesson` (optional): Lesson learned for the future
- `context` (optional): Additional context
- `project_name` (optional): Project name

**Example:**
```
capture_decision(
    task="Choose database",
    decision="PostgreSQL",
    alternatives=["MongoDB", "MySQL"],
    reasoning="We need ACID and complex queries",
    lesson="For relational data with transactions, SQL > NoSQL"
)
```

### `capture_quick` - ⚡ QUICK CAPTURE (MINIMUM EFFORT)
**The simplest way to capture. Only 2 required fields.**

Use for quick captures without much detail. Ideal when you're in a hurry.

Parameters:
- `what` (required): What did you do? (action performed)
- `why` (required): Why did you do it? (reason)
- `lesson` (optional but recommended): Lesson learned
- `project_name` (optional): Project name

**Examples:**
```
capture_quick(
    what="Added retry logic to HTTP client",
    why="API calls were failing intermittently"
)

capture_quick(
    what="Switched from axios to fetch",
    why="Reduce dependencies, native fetch is sufficient",
    lesson="Always evaluate if a dependency is really necessary"
)
```

### `query_memory` - Query memories with RAG
Use when:
- The user asks "why did we do X?"
- The user asks "how did we solve something similar?"
- Before making an important decision (check precedents)

Parameters:
- `question` (required): Question to answer
- `project_name` (optional): Filter by project
- `num_episodes` (optional): Number of episodes to query (1-10, default: 5)

### `search_episodes` - Semantic episode search
Use for specific searches on topics or technologies.
Returns the most relevant episodes for a search term.
*Note: Queried results receive a relevance boost for the future.*

Parameters:
- `query` (required): Search term
- `project_name` (optional): Filter by project
- `top_k` (optional): Number of results (default: 5)

### `get_episode` - Get complete episode
Use when you need to dive into the details of a specific decision.
Returns the COMPLETE content: thinking, alternatives, decision factors, context and lessons.

Parameters:
- `episode_id` (required): UUID of the episode to retrieve

### `get_lessons` - Lessons learned
Use for:
- Onboarding new members
- Review before starting a similar feature
- The user asks "what have we learned about X?"

Parameters:
- `project_name` (optional): Filter by project
- `tags` (optional): Array of tags to filter

### `get_timeline` - View chronological history
Use to see chronological evolution of the project and understand what was done when.

Parameters:
- `project_name` (optional): Filter by project
- `limit` (optional): Maximum episodes to return (default: 20)

### `get_statistics` - Memory statistics
Gets memory database statistics: total episodes, distribution by type and assistant.

Parameters:
- `project_name` (optional): Filter by project

### `onboard_project` - Onboard existing project
Use when:
- ✅ It's the first time I work on this project
- ✅ The user asks "analyze the project", "learn the code"
- ✅ I need to understand the structure before making big changes
- ✅ There are no previous memories and I want to create initial context

Automatically generates an episode with:
- Detected technology stack
- Architectural patterns
- Main dependencies
- Code conventions

Parameters:
- `project_path` (required): Absolute path to the project
- `project_name` (optional): Project name (auto-detected)

### `mark_episode` - Curate memory
Use when:
- A previous solution turned out to be wrong → `is_antipattern=true` (it will appear as a WARNING)
- A decision is essential for the project → `is_critical=true` (ranked higher in searches)
- A newer episode replaces an old one → `superseded_by="<episode UUID>"`

Parameters:
- `episode_id` (required): UUID of the episode to mark
- `is_antipattern`, `is_critical` (optional): Boolean flags
- `superseded_by`, `deprecation_reason` (optional): Replacement episode and reason

### `consolidate_memories` / `check_consolidation_status` - Meta-memories
Groups similar episodes (DBSCAN over embeddings) and asks the LLM to synthesize
patterns, lessons and anti-patterns. Run it when `get_project_context` recommends it.

Parameters:
- `project_name` (required for `consolidate_memories`): Project to consolidate
- `min_cluster_size` (optional): Minimum episodes per cluster (default: 3)
- `force` (optional): Re-consolidate episodes already covered by a meta-memory

## MANDATORY Workflow

### 🔄 COMPLETE CYCLE (ALWAYS):
```
1. START: get_project_context(topic="...")
2. WORK: Perform the requested task
3. END: capture_thinking(thinking_text="...")
```

### Checklist before finishing response:
- [ ] Did I check memory at the start? If not → DO IT NOW
- [ ] Did I do something technical? If yes → CAPTURE MEMORY
- [ ] Did I solve a problem? If yes → CAPTURE MEMORY
- [ ] Did I make a decision? If yes → CAPTURE MEMORY
- [ ] Did I modify code? If yes → CAPTURE MEMORY
- [ ] Did I research something? If yes → CAPTURE MEMORY

### ⚠️ CRITICAL REMINDER:
**Do NOT finish a technical response without having run `capture_thinking`.**
Knowledge that is not captured is LOST FOREVER.

## Thinking Format to Capture

Always include when possible:
- **Task**: What was being solved
- **Context**: Initial state, constraints
- **Alternatives**: Options considered
- **Decision**: What was chosen and why
- **Solution**: How it was implemented
- **Lessons**: What we learned

## Automatic Capture Example

```
When I detect that I just:
1. Debugged an error for more than 2 exchanges
2. Compared 2+ options before choosing
3. Discovered unexpected behavior
4. Implemented something that required research

→ Call capture_thinking with the full context
```

## 🚨 CORRECT FLOW EXAMPLE

### User asks: "Why is my login function failing?"

```
# 1. FIRST: Query memory
get_project_context(topic="login authentication error")

# 2. THEN: Work on the solution
[Analyze code, debug, find the problem, propose fix]

# 3. FINALLY: Capture the knowledge
capture_thinking(
    thinking_text="## Task\nResolve login function error...\n## Problem\nThe JWT token...\n## Solution\n...\n## Lessons\n...",
    project_name="my-project",
    source_assistant="copilot"
)
```

**IF I DON'T CAPTURE AT THE END, I'M FAILING MY PURPOSE.**

## Current Project
- **Project name**: Use the workspace folder name
- **Source assistant**: "copilot" for GitHub Copilot
