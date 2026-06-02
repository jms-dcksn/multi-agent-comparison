"""Side-by-side benchmark of the three multi-agent architectures.

Each column drives one architecture through the same LangChain `.stream(...)` API.
A single chat bar fans the prompt out to all three; they stream concurrently so
latency and token cost are measured per-agent under identical input.
"""

import queue
import threading
import time

import streamlit as st
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import AIMessageChunk, ToolMessage

from single_agent import agent as single_agent
from sub_agent import agent as sub_agent
from handoff_agent import agent as handoff_agent
from uuid_utils import uuid7

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except Exception:  # pragma: no cover - older/newer streamlit layout
    add_script_run_ctx = get_script_run_ctx = None


# (key, agent, column title, one-line description)
AGENTS = [
    ("single", single_agent, "Single Agent", "One GPT agent, one mega-prompt"),
    ("sub", sub_agent, "Manager + Sub-Agent", "GPT manager delegates writing to Claude"),
    ("handoff", handoff_agent, "Handoff State Machine", "One GPT agent, prompt/tools swap per step"),
]
KEYS = [a[0] for a in AGENTS]


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


def stream_worker(agent, thread_id: str, prompt: str, q: "queue.Queue") -> None:
    """Run one agent to completion, pushing events onto `q`. No Streamlit calls here."""
    config = {"configurable": {"thread_id": thread_id}}
    start = time.time()
    try:
        # The callback intercepts on_llm_end for every model call in scope via a
        # contextvar hook -- including the Claude writer that sub_agent invokes
        # inside its write_script tool, whose usage never reaches the message stream.
        with get_usage_metadata_callback() as cb:
            for mode, payload in agent.stream(
                {"messages": [{"role": "user", "content": prompt}]},
                config=config,
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    chunk, _ = payload
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        text = to_text(chunk.content)
                        if text:
                            q.put(("text", text))
                elif mode == "updates":
                    for _node, data in payload.items():
                        for msg in data.get("messages", []) or []:
                            for call in getattr(msg, "tool_calls", []) or []:
                                q.put(("tool_call", {
                                    "id": call.get("id"),
                                    "name": call.get("name", "tool"),
                                    "args": call.get("args", {}),
                                }))
                            if isinstance(msg, ToolMessage):
                                q.put(("tool_result", {
                                    "id": msg.tool_call_id,
                                    "content": to_text(msg.content),
                                }))
            by_model = {m: u.get("total_tokens", 0) for m, u in cb.usage_metadata.items()}
            tokens = sum(by_model.values())
        q.put(("done", {"latency": time.time() - start, "tokens": tokens, "by_model": by_model}))
    except Exception as exc:  # surface failures in the column rather than crashing
        q.put(("error", str(exc)))


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
    return " · ".join(f"{m} {t:,}" for m, t in by_model.items())


# --- Page setup & state -----------------------------------------------------

st.set_page_config(page_title="Multi-Agent Comparison", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = {k: [] for k in KEYS}
    st.session_state.threads = {k: str(uuid7()) for k in KEYS}

st.title("Multi-Agent Architecture Comparison")
st.caption("Same demo-script task, three architectures. Compare speed, cost, and output quality.")
if st.button("Reset all conversations"):
    st.session_state.history = {k: [] for k in KEYS}
    st.session_state.threads = {k: str(uuid7()) for k in KEYS}
    st.rerun()

cols = st.columns(len(AGENTS), border=True)

# Render headers, cumulative metrics, and conversation history per column.
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
                    st.caption(f"{msg.get('tokens', 0):,} tokens · {msg.get('latency', 0.0):.1f}s")
                    split = model_split(msg.get("by_model", {}))
                    if split:
                        st.caption(split)


# --- Handle a new prompt: stream all three concurrently ---------------------

prompt = st.chat_input("Describe the product and demo you want a script for…")
if prompt:
    for key in KEYS:
        st.session_state.history[key].append({"role": "user", "content": prompt})

    queues, threads, placeholders = {}, {}, {}
    parts = {k: [] for k in KEYS}
    meta = {k: {} for k in KEYS}

    for (key, agent, _t, _d), col in zip(AGENTS, cols):
        with col:
            with st.chat_message("user"):
                st.markdown(prompt)
            placeholders[key] = st.empty()
        q = queue.Queue()
        queues[key] = q
        t = threading.Thread(
            target=stream_worker,
            args=(agent, st.session_state.threads[key], prompt, q),
            daemon=True,
        )
        if add_script_run_ctx and get_script_run_ctx:
            add_script_run_ctx(t, get_script_run_ctx())
        threads[key] = t
        t.start()

    active = set(KEYS)
    while active:
        for key in list(active):
            dirty = False
            try:
                while True:
                    event = queues[key].get_nowait()
                    if handle_event(parts[key], meta[key], event):
                        active.discard(key)
                    dirty = True
            except queue.Empty:
                pass
            if dirty:
                with placeholders[key].container():
                    with st.chat_message("assistant"):
                        render_parts(parts[key])
                        if key not in active:
                            mt = meta[key]
                            st.caption(f"{mt.get('tokens', 0):,} tokens · {mt.get('latency', 0.0):.1f}s")
                            split = model_split(mt.get("by_model", {}))
                            if split:
                                st.caption(split)
        time.sleep(0.05)

    for key in KEYS:
        st.session_state.history[key].append({
            "role": "assistant",
            "parts": parts[key],
            "tokens": meta[key].get("tokens", 0),
            "latency": meta[key].get("latency", 0.0),
            "by_model": meta[key].get("by_model", {}),
        })
    st.rerun()
