"""Pure scoring helpers for the comparison summary.

No I/O, no Streamlit -- just functions over the in-memory chat history and judge
verdicts. Cost is price-weighted dollars (fair across providers); cost and latency
are normalized so the best agent scores 100 (inverse ratio, not min-max).
"""

from __future__ import annotations

# $ per 1M tokens, per model. Both rates confirmed.
PRICING = {
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "gpt-5.4-mini": {"in": 0.75, "out": 4.50},
}

# A produced script contains these spec-defined section headers; a clarifying
# question does not. Used only to pick which message to judge.
SCRIPT_MARKERS = ("LIMBIC OPENING", "PREPARATION CHECKLIST")


def assistant_text(message: dict) -> str:
    """Concatenate the text parts of an assistant history message."""
    return "".join(
        p["content"] for p in message.get("parts", []) if p.get("type") == "text"
    )


def looks_like_script(text: str) -> bool:
    return all(marker in text for marker in SCRIPT_MARKERS)


def latest_script(history: list[dict]) -> str | None:
    """Most recent assistant turn that looks like a script, or None.

    Scans backward because refinement turns and tool calls mean the script isn't
    always the final message.
    """
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            text = assistant_text(msg)
            if looks_like_script(text):
                return text
    return None


def message_cost(by_model: dict) -> float:
    """Dollar cost of one assistant message from its per-model token split."""
    total = 0.0
    for model, usage in by_model.items():
        price = PRICING.get(model)
        if not price or not isinstance(usage, dict):
            continue
        total += usage.get("input", 0) / 1e6 * price["in"]
        total += usage.get("output", 0) / 1e6 * price["out"]
    return total


def agent_cost(messages: list[dict]) -> float:
    return sum(
        message_cost(m.get("by_model", {}))
        for m in messages
        if m.get("role") == "assistant"
    )


def agent_latency(messages: list[dict]) -> float:
    return sum(
        m.get("latency", 0.0) for m in messages if m.get("role") == "assistant"
    )


def normalize_lower_better(values: dict[str, float]) -> dict[str, float]:
    """100 for the cheapest/fastest; others scaled by inverse ratio (best / value)."""
    positives = [v for v in values.values() if v > 0]
    if not positives:
        return {k: 0.0 for k in values}
    best = min(positives)
    return {k: (100 * best / v if v > 0 else 0.0) for k, v in values.items()}


def build_scores(history: dict[str, list], verdicts: dict) -> dict[str, dict]:
    """Per-agent {cost, latency, quality}, each 0-100."""
    costs = {k: agent_cost(msgs) for k, msgs in history.items()}
    lats = {k: agent_latency(msgs) for k, msgs in history.items()}
    cost_n = normalize_lower_better(costs)
    lat_n = normalize_lower_better(lats)

    scores = {}
    for k in history:
        verdict = verdicts.get(k)
        scores[k] = {
            "cost": round(cost_n[k], 1),
            "latency": round(lat_n[k], 1),
            "quality": float(verdict.overall) if verdict else 0.0,
        }
    return scores
