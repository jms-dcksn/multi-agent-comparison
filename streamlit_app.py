"""Side-by-side benchmark of the three multi-agent architectures.

Each column drives one architecture through the same LangChain `.stream(...)` API
and owns an **independent** chat input, so you can carry three isolated
conversations in parallel. Submitting any column's input reruns the script and
streams only that agent, leaving the other two conversations untouched.
"""

import time

import streamlit as st
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import AIMessageChunk, ToolMessage

from single_agent import agent as single_agent
from sub_agent import agent as sub_agent
from handoff_agent import agent as handoff_agent
from uuid_utils import uuid7


# (key, agent, column title, one-line description)
AGENTS = [
    ("single", single_agent, "Single Agent", "One GPT agent, one mega-prompt"),
    ("sub", sub_agent, "Manager + Sub-Agent", "GPT manager delegates writing to Claude"),
    ("handoff", handoff_agent, "Handoff State Machine", "One GPT agent, prompt/tools swap per step"),
]
KEYS = [a[0] for a in AGENTS]
AGENT_BY_KEY = {a[0]: a[1] for a in AGENTS}


def to_text(content) -> str:
    """Flatten message content (str, or list of content blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, str):
                out.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                out.append(block.get("text", ""))
        return "".join(out)
    return str(content)


def stream_events(agent, thread_id: str, prompt: str):
    """Yield (kind, data) events for one agent run, then a final ('done', meta).

    The usage callback intercepts on_llm_end for every model call in scope via a
    contextvar hook -- including the Claude writer that sub_agent invokes inside
    its write_script tool, whose usage never reaches the message stream.
    """
    config = {"configurable": {"thread_id": thread_id}}
    start = time.time()
    try:
        with get_usage_metadata_callback() as cb:
            for mode, payload in agent.stream(
                {"messages": [{"role": "user", "content": prompt}]},
                config=config,
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    chunk, metadata = payload
                    # The sub-agent's Claude writer streams up this same channel from
                    # inside write_script (the `tools` node). Skip it so the script isn't
                    # shown twice -- it already renders collapsed in the tool-call expander.
                    if metadata.get("langgraph_node") == "tools":
                        continue
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        text = to_text(chunk.content)
                        if text:
                            yield ("text", text)
                elif mode == "updates":
                    for _node, data in payload.items():
                        for msg in data.get("messages", []) or []:
                            for call in getattr(msg, "tool_calls", []) or []:
                                yield ("tool_call", {
                                    "id": call.get("id"),
                                    "name": call.get("name", "tool"),
                                    "args": call.get("args", {}),
                                })
                            if isinstance(msg, ToolMessage):
                                yield ("tool_result", {
                                    "id": msg.tool_call_id,
                                    "content": to_text(msg.content),
                                })
            by_model = {
                m: {
                    "input": u.get("input_tokens", 0),
                    "output": u.get("output_tokens", 0),
                    "total": u.get("total_tokens", 0),
                }
                for m, u in cb.usage_metadata.items()
            }
            tokens = sum(u["total"] for u in by_model.values())
        yield ("done", {"latency": time.time() - start, "tokens": tokens, "by_model": by_model})
    except Exception as exc:  # surface failures in the column rather than crashing
        yield ("error", str(exc))


def append_text(parts: list, text: str) -> None:
    """Append streamed text, extending the current text block if there is one."""
    if parts and parts[-1]["type"] == "text":
        parts[-1]["content"] += text
    else:
        parts.append({"type": "text", "content": text})


def render_parts(parts: list) -> None:
    """Render an ordered list of text / tool-call blocks into the current container."""
    for part in parts:
        if part["type"] == "text":
            if part["content"].strip():
                st.markdown(part["content"])
        else:
            label = f"Tool call · {part['name']}"
            with st.expander(label, expanded=False):
                st.caption("Arguments")
                st.json(part["args"])
                if part.get("result") is None:
                    st.caption("Running…")
                else:
                    st.caption("Result")
                    st.code(part["result"][:2000], language="json")


def handle_event(parts: list, meta: dict, event) -> bool:
    """Apply one event to a column's parts/meta. Returns True when the agent is done."""
    kind, data = event
    if kind == "text":
        append_text(parts, data)
    elif kind == "tool_call":
        parts.append({
            "type": "tool", "id": data["id"], "name": data["name"],
            "args": data["args"], "result": None,
        })
    elif kind == "tool_result":
        for part in parts:
            if part["type"] == "tool" and part["id"] == data["id"]:
                part["result"] = data["content"]
                break
    elif kind == "done":
        meta.update(data)
        return True
    elif kind == "error":
        append_text(parts, f"\n\n**Error:** {data}")
        meta.update({"latency": 0.0, "tokens": 0, "by_model": {}})
        return True
    return False


def model_split(by_model: dict) -> str:
    """One-line per-model token breakdown, e.g. 'gpt-5.4-mini 1,200 · claude 4,800'."""
    return " · ".join(f"{m} {u['total']:,}" for m, u in by_model.items())


def render_meta(meta: dict) -> None:
    """Render the per-message token/latency captions."""
    st.caption(f"{meta.get('tokens', 0):,} tokens · {meta.get('latency', 0.0):.1f}s")
    split = model_split(meta.get("by_model", {}))
    if split:
        st.caption(split)


# --- Page setup & state -----------------------------------------------------

st.set_page_config(page_title="Multi-Agent Comparison", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = {k: [] for k in KEYS}
    st.session_state.threads = {k: str(uuid7()) for k in KEYS}

st.title("Multi-Agent Architecture Comparison")
st.caption("Three architectures, three independent conversations. Compare speed, cost, and output quality side by side.")
if st.button("Reset all conversations"):
    st.session_state.history = {k: [] for k in KEYS}
    st.session_state.threads = {k: str(uuid7()) for k in KEYS}
    st.session_state.pop("verdicts", None)
    st.rerun()

st.page_link("pages/2_Summary.py", label="View performance summary →")

cols = st.columns(len(AGENTS), border=True)

# Render each column's header, metrics, history, a streaming slot, and its own input.
placeholders, inputs = {}, {}
for (key, _agent, title, desc), col in zip(AGENTS, cols):
    with col:
        st.subheader(title)
        st.caption(desc)
        history = st.session_state.history[key]
        total_tokens = sum(m.get("tokens", 0) for m in history if m["role"] == "assistant")
        total_latency = sum(m.get("latency", 0.0) for m in history if m["role"] == "assistant")
        m1, m2 = st.columns(2)
        m1.metric("Total tokens", f"{total_tokens:,}")
        m2.metric("Total latency", f"{total_latency:.1f}s")
        for msg in history:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                else:
                    render_parts(msg["parts"])
                    render_meta(msg)
        # Slot for the in-flight exchange, kept above this column's input.
        placeholders[key] = st.empty()
        inputs[key] = st.chat_input(f"Message {title}…", key=f"input_{key}")


# --- Handle a submission: only the submitted column reruns and streams ------

for key in KEYS:
    prompt = inputs[key]
    if not prompt:
        continue

    st.session_state.history[key].append({"role": "user", "content": prompt})
    parts, meta = [], {}
    with placeholders[key].container():
        with st.chat_message("user"):
            st.markdown(prompt)
        slot = st.empty()
        for event in stream_events(AGENT_BY_KEY[key], st.session_state.threads[key], prompt):
            done = handle_event(parts, meta, event)
            with slot.container():
                with st.chat_message("assistant"):
                    render_parts(parts)
                    if done:
                        render_meta(meta)

    st.session_state.history[key].append({
        "role": "assistant",
        "parts": parts,
        "tokens": meta.get("tokens", 0),
        "latency": meta.get("latency", 0.0),
        "by_model": meta.get("by_model", {}),
    })
    st.rerun()
