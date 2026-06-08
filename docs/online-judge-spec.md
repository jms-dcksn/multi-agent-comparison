# Spec: Online LLM Judge + Comparison Summary

## Goal

Add an **online LLM judge** that scores each agent's demo-script output the moment it's
produced, then surface a **summary page** that compares all three architectures on three
normalized indicators: **cost**, **latency**, and **quality** — three bars per agent,
each 0–100.

The judge starts narrow (script quality + Tell-Show-Tell structure adherence) and is built
to grow into more dimensions later.

**Trigger: a manual "Analyze performance" button.** No inline judging during streaming, no
heuristics to detect when a script was produced. The user runs conversations, then clicks
the button on the summary page to score the latest script from each agent. Simple,
predictable, and zero wasted judge calls on clarifying-question turns.

---

## Recommendations up front

1. **Judge model: Claude Opus 4.8 (`claude-opus-4-8`), called via the Anthropic SDK with
   structured outputs.** One strong, consistent judge across all three agents keeps the
   comparison fair, and it's a *different* model from both the GPT manager and the Claude
   Sonnet writer, so the judge isn't grading its own homework. Structured outputs
   (`messages.parse()` + Pydantic) give us a guaranteed 0–100 integer with no parsing.
2. **Cost = price-weighted dollars, not raw tokens.** Raw token counts are unfair when the
   sub-agent burns Claude Sonnet tokens (pricier) while the others use GPT. We already have
   per-model usage in the callback — capture the input/output split and apply per-model
   pricing. (Decision card below if you'd rather keep it as raw tokens for v1.)
3. **Relative scores use best-as-100 inverse ratio**, not min-max. Min-max forces the worst
   agent to 0 even when all three are close, which misleads. Inverse ratio
   (`100 * best / value`) keeps the gaps honest.
4. **Separate page = the Streamlit `pages/` convention.** Drop a file in `pages/` and it
   becomes a second page in the sidebar nav automatically. `st.session_state` is shared
   across pages, so the summary page just reads the history the main page already built.
   No routing, no new framework.
5. **Manual trigger.** The judge runs on demand from a button, not inline per turn. The
   only "did a script get produced" logic we need is picking *which* message to judge — the
   latest script in each agent's history — not an auto-trigger.

---

## Component 1 — `judge.py` (the online judge)

A single self-contained module. One function: take a finished script, return a structured
verdict.

```python
# judge.py
import anthropic
from pydantic import BaseModel, Field

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from .env (already loaded)

class Verdict(BaseModel):
    is_script: bool = Field(description="True only if the text is an actual demo script, "
                                        "not a clarifying question or discovery turn.")
    overall: int = Field(ge=0, le=100, description="Overall demo-script quality, 0-100.")
    structure: int = Field(ge=0, le=100, description="Adherence to Tell-Show-Tell + required "
                                                     "sections (Limbic Opening, Initial Tell, "
                                                     "exactly 3 Key Ideas in Tell-Show-Tell, "
                                                     "Closing Tell, Prep Checklist).")
    rationale: str = Field(description="2-3 sentences justifying the scores.")

JUDGE_PROMPT = """You are an expert evaluator of B2B demo scripts...
Score on two axes:
- overall: writing quality, audience calibration, narrative tension, concreteness.
- structure: strict adherence to the Tell-Show-Tell spec (sections in order, exactly 3
  Key Ideas, each with TELL/SHOW/TELL, stage directions, time markers).
If the text is not a script (e.g. a clarifying question), set is_script=false and score 0.
"""

def judge_script(script_text: str) -> Verdict:
    resp = client.messages.parse(
        model="claude-opus-4-8",
        max_tokens=2000,
        system=JUDGE_PROMPT,
        messages=[{"role": "user", "content": f"Evaluate this output:\n\n{script_text}"}],
        output_format=Verdict,
    )
    return resp.parsed_output
```

Notes:
- `is_script` lets us **skip discovery/clarifying turns** from the quality aggregate — only
  real scripts get a quality score.
- The rubric in `JUDGE_PROMPT` should quote the same Tell-Show-Tell spec the agents use, so
  the judge measures against the exact contract. Keep it in one place here; do not duplicate
  the writer prompt — paraphrase the *checklist*.
- Structured outputs are supported on Opus 4.8. The Pydantic model is the schema.
- Alternative considered: reuse `langchain_anthropic.ChatAnthropic` for codebase
  consistency. Rejected — the raw SDK's `messages.parse()` is the cleanest path to a
  guaranteed-shape verdict, and the judge is conceptually separate from the agent stack.

---

## Component 2 — scoring & normalization

Lives in a small `scoring.py` (or inline in the summary page). Pure functions, no I/O.

### Raw metrics per agent (`build_scores(history, verdicts)`)
- **tokens / cost** — sum across that agent's assistant messages.
- **latency** — sum (or mean) of `latency` across assistant messages.
- **quality** — the judge `overall` from that agent's verdict (the latest script only). No
  averaging over turns; one script judged per agent per analysis run. `structure` rides
  along for the raw table.

### Cost (price-weighted)
The usage callback already exposes per-model `input_tokens` / `output_tokens` — we currently
throw away the split and keep only `total_tokens`. Capture the split (see change in
`streamlit_app.py` below), then:

```python
# $ per 1M tokens. Fill in the OpenAI numbers for your actual gpt-5.4-mini pricing.
PRICING = {
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "gpt-5.4-mini":      {"in": 0.00, "out": 0.00},  # TODO: real prices
}

def message_cost(by_model: dict) -> float:
    total = 0.0
    for model, u in by_model.items():
        p = PRICING.get(model)
        if not p:
            continue
        total += u["input"] / 1e6 * p["in"] + u["output"] / 1e6 * p["out"]
    return total
```

### Normalize to 0–100 (best = 100)
Cost and latency are "lower is better"; quality is already absolute 0–100.

```python
def normalize_lower_better(values: dict[str, float]) -> dict[str, float]:
    """100 for the cheapest/fastest; others scaled by inverse ratio."""
    best = min(v for v in values.values() if v > 0)
    return {k: (100 * best / v if v > 0 else 0.0) for k, v in values.items()}

# quality passes through unchanged (already 0-100)
```

Result: each agent gets `{cost: 0-100, latency: 0-100, quality: 0-100}`.

---

## Component 3 — the summary page

`pages/2_Summary.py` (the numeric prefix orders it in the sidebar; rename the main file to
`pages/1_Compare.py` or leave the main script as the landing page — see build steps).

The page does nothing until the user clicks **Analyze performance**. On click, it finds the
latest script per agent, judges each, computes scores, and caches the result in
`st.session_state` so reruns don't re-trigger the (paid) judge calls.

```python
import pandas as pd
import streamlit as st
from judge import judge_script
from scoring import build_scores, latest_script  # both read st.session_state.history

st.title("Comparison Summary")

if st.button("Analyze performance"):
    with st.spinner("Judging scripts…"):
        verdicts = {}
        for key in KEYS:
            script = latest_script(st.session_state.history[key])  # None if no script yet
            verdicts[key] = judge_script(script) if script else None
        st.session_state.verdicts = verdicts

verdicts = st.session_state.get("verdicts")
if not verdicts:
    st.info("Run some conversations on the main page, then click Analyze performance.")
    st.stop()

scores = build_scores(st.session_state.history, verdicts)  # {agent_key: {cost, latency, quality}}
df = pd.DataFrame([
    {"agent": title_for(k), "metric": m, "score": v}
    for k, metrics in scores.items() for m, v in metrics.items()
])

# Simplest "three bars per agent" visual — st.bar_chart does grouped bars natively in 1.58.
st.bar_chart(df, x="agent", y="score", color="metric", stack=False)

# raw numbers + judge rationale under the chart for credibility
st.dataframe(raw_metrics_table(st.session_state.history, verdicts))
```

`latest_script(history)` scans the agent's messages backward and returns the text of the most
recent assistant turn that looks like a script (Tell-Show-Tell section markers present), or
`None` — the same backward-scan idea already in `handoff_agent._latest_script()`. This is the
*only* place script-detection is needed; it picks the judge's input, it does not gate timing.

If grouped `st.bar_chart` ever fights you, the dead-simple fallback is three labeled
`st.progress` bars per agent inside `st.columns(3)` — no dataframe, no chart lib. Either way,
zero new dependencies.

---

## Changes to `streamlit_app.py`

The judge does **not** run here — it's triggered manually from the summary page. The main
page only needs two small changes:

1. **Capture input/output token split** (for price-weighted cost). In `stream_events`,
   replace the `total_tokens`-only line with the full split per model:
   ```python
   by_model = {
       m: {"input": u.get("input_tokens", 0),
           "output": u.get("output_tokens", 0),
           "total": u.get("total_tokens", 0)}
       for m, u in cb.usage_metadata.items()
   }
   ```
   (Adjust `model_split` / `render_meta` to read `["total"]`.)
2. **Link to the summary page** — `st.page_link("pages/2_Summary.py", label="View summary")`.

Nothing else in the streaming/rendering path changes. Latency and tokens are already recorded
per message today, so the summary page has everything it needs the moment the button is clicked.

---

## Decision cards

`[DECISION] Judge model | Rec: Claude Opus 4.8 via Anthropic SDK | Risk: adds ~$/call and a
few seconds per turn; one judge could systematically favor one writing style | Reversible? Yes`

`[DECISION] Cost basis | Rec: price-weighted dollars (capture in/out split) | Alt: raw total
tokens (simpler, no pricing table, but unfair across providers) | Risk: gpt-5.4-mini pricing
must be filled in | Reversible? Yes`

`[DECISION] What gets judged | Rec: the latest script per agent, on button click | Alt: judge
every revision and average | Risk: a single latest-script score ignores earlier drafts |
Reversible? Yes`

---

## Build sequence

1. `judge.py` — judge + Pydantic verdict; smoke-test on a pasted script.
2. Capture input/output token split in `streamlit_app.py`; verify `render_meta` still works.
3. `scoring.py` — `latest_script`, `build_scores`, normalization (pure functions, unit-testable).
4. `pages/2_Summary.py` — "Analyze performance" button → judge → grouped bar chart + raw table.
5. Fill in real `PRICING`, sanity-check the summary against a few live runs.

## Out of scope (later)

- Additional judge dimensions (audience calibration, factual grounding vs. research,
  hallucination checks) — the `Verdict` model is designed to grow.
- Persisting scores across runs (state is in-memory only today).
- Multi-judge ensembles or self-consistency.
```
