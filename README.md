# Multi-Agent Architecture Comparison

A benchmark of three multi-agent workflow architectures built on LangChain, all
solving the same task: an interactive **demo-script writer** that researches a
product (Tavily web search) and produces a Tell-Show-Tell presentation script.

A Streamlit app runs all three side by side over independent conversations, then
a two-layer LLM judge scores each output and picks a winner so you can compare
them on speed, cost, and quality.

## The three architectures

Each lives in its own module and exports a single `agent` object behind the same
LangChain `.stream(...)` interface, so they're interchangeable.

- **`single_agent.py`** — one GPT agent, one mega-prompt. Discovery and scripting
  fused into a single system prompt; the model decides when to stop researching
  and start writing. Simplest baseline.
- **`sub_agent.py`** — a GPT manager owns the dialogue and delegates writing to a
  `write_script` tool that invokes a separate Claude model. The only architecture
  that mixes providers by design.
- **`handoff_agent.py`** — one GPT agent with a state machine. Its prompt and
  available tools are swapped per step (`discovery` → `writer` → `refinement`)
  via `transfer_to_*` tools that mutate custom state, with middleware overriding
  the system prompt and tool set on every call.

## Comparison dimensions

- **Latency** and **token cost** — captured live per message (including the
  sub-agent's hidden Claude writer, via a usage-metadata callback) and
  price-weighted into dollars across providers (`scoring.py`).
- **Output quality** — two judge layers (`judge.py`), both Claude Opus, both
  grading against the same shared criteria (writing quality + Tell-Show-Tell
  structure):
  1. **Per-script scoring** — each finished script is judged in isolation and
     given 0-100 scores on overall quality and structure adherence, with a
     rationale.
  2. **Comparative winner pick** — a head-judge pass sees all candidate scripts
     side by side and picks the single strongest. This is a pairwise-style
     evaluation: rather than trusting absolute scores (which drift between
     calls), it compares the outputs directly against each other on the same
     criteria, which is more reliable for ranking.

  Using Opus keeps both layers independent of the GPT manager and Claude writer
  they grade.
- **Ease of evaluation** — how observable and instrumentable each architecture is.

## Running it

```bash
uv sync                          # install dependencies
uv run streamlit run streamlit_app.py   # the side-by-side comparison UI
uv run main.py                   # optional: single-agent REPL (type 'quit' to exit)
```

In the app, message any of the three columns to drive that architecture, then open
the **Summary** page and click *Analyze performance* to judge the scripts and chart
cost / latency / quality.

Requires `.env` (gitignored) with `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and
`TAVILY_API_KEY`.

## Layout

| File | Role |
|------|------|
| `streamlit_app.py` | Side-by-side comparison UI; streams all three agents |
| `pages/2_Summary.py` | Runs the judge and charts the three indicators |
| `single_agent.py` / `sub_agent.py` / `handoff_agent.py` | The architectures |
| `judge.py` | Two-layer LLM judge: per-script scores + comparative winner pick |
| `scoring.py` | Pure cost/latency/quality scoring over chat history |
| `main.py` | CLI REPL for a single architecture |

The demo-script spec (Tell-Show-Tell structure, tone rules, output sections) is
duplicated across all three agent modules on purpose, so each is self-contained
and independently comparable. If you change the spec, change it in all three.
