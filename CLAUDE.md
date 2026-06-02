# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A benchmark comparing three multi-agent workflow architectures for the same task: an
interactive **demo-script writer** that researches a product (Tavily web search) and
produces a Tell-Show-Tell presentation script. The three implementations are compared on
latency, token cost, output quality (LLM judges), and ease of evaluation.

## Commands

```bash
uv sync                 # install dependencies
uv run main.py          # run the interactive REPL (type 'quit' to exit)
```

Requires `.env` (gitignored) with `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and
`TAVILY_API_KEY`. No tests or lint configured yet.

## The three architectures

Each lives in its own module and **exports a single `agent` object** with the same
LangChain `create_agent` interface (`.stream(...)`). They are interchangeable — `main.py`
selects one via its import line (`from sub_agent import agent`). To benchmark a different
architecture, change that import.

- **`single_agent.py`** — one agent, one mega-prompt. Discovery and scripting guidelines
  are fused into a single `system_prompt`; the model decides when to stop researching and
  start writing. Simplest baseline.

- **`sub_agent.py`** — conversation manager + sub-agent. A GPT manager owns the dialogue
  and calls `write_script`, a tool that invokes a **separate Claude model** to do the
  actual writing. The two models have different prompts; the writer never sees the user
  directly. This is the only architecture that mixes model providers by design.

- **`handoff_agent.py`** — single agent with a state machine. One model, but its prompt
  and available tools are swapped per step (`discovery` → `writer` → `refinement`) based
  on `current_step` in a custom `DemoScriptState`. `transfer_to_*` tools return a
  LangGraph `Command` that mutates state to drive handoffs; an `@wrap_model_call`
  middleware (`apply_step_config`) reads `current_step` and overrides the system prompt +
  tool set on every model call.

## Key conventions

- The **demo-script writer prompt** (Tell-Show-Tell structure, audience tone rules,
  output sections, formatting rules) is duplicated across all three modules so each is
  self-contained and independently comparable. If you change the script spec, change it in
  all three — divergence would invalidate the comparison.
- Prompts in `handoff_agent.py` are filled with `.replace("{writer_prompt}", ...)` rather
  than `str.format()` on purpose: the prompts contain literal `{` / `}` (markdown, JSON
  examples) that would break `format`.
- `_latest_script()` in `handoff_agent.py` scans messages backward rather than trusting
  `messages[-1]`, because the writer may emit the script and call the transfer tool on
  separate turns.
- State is in-memory only (`InMemorySaver`); conversations don't persist across runs.
