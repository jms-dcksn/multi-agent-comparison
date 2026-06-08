"""Comparison summary -- runs the LLM judge on demand and charts the three indicators.

A separate page via Streamlit's `pages/` convention; `st.session_state` is shared with
the main page, so this reads the history that page already built. Nothing happens until
the user clicks "Analyze performance".
"""

import pandas as pd
import streamlit as st

from judge import judge_script, pick_winner
from scoring import latest_script, build_scores, agent_cost, agent_latency

AGENTS = [
    ("single", "Single Agent"),
    ("sub", "Manager + Sub-Agent"),
    ("handoff", "Handoff State Machine"),
]
TITLES = dict(AGENTS)
KEYS = [k for k, _ in AGENTS]

st.set_page_config(page_title="Summary", layout="wide")
st.title("Comparison Summary")
st.caption("Cost, latency, and quality — each scored 0–100, best in class = 100.")

history = st.session_state.get("history")
if not history or not any(history.get(k) for k in KEYS):
    st.info("Run some conversations on the main page first, then come back and analyze.")
    st.stop()

if st.button("Analyze performance", type="primary"):
    with st.spinner("Judging scripts…"):
        scripts = {k: latest_script(history.get(k, [])) for k in KEYS}
        verdicts = {k: judge_script(s) if s else None for k, s in scripts.items()}
        st.session_state.verdicts = verdicts
        st.session_state.winner = pick_winner(scripts)

verdicts = st.session_state.get("verdicts")
if not verdicts:
    st.stop()

scores = build_scores({k: history.get(k, []) for k in KEYS}, verdicts)

# Lead with the head judge's overall pick.
winner = st.session_state.get("winner")
if winner and winner.winner in TITLES:
    st.subheader("Overall winner")
    left, right = st.columns([1, 2])
    with left:
        st.success(f"**{TITLES[winner.winner]}**")
        v = verdicts.get(winner.winner)
        if v:
            st.metric("Quality", v.overall)
            st.metric("Structure", v.structure)
    with right:
        st.markdown("**Why this script wins**")
        st.markdown(winner.reasoning)
    st.divider()

# Three bars per agent — st.bar_chart does grouped bars natively (stack=False).
chart_df = pd.DataFrame(
    [
        {"agent": TITLES[k], "metric": metric, "score": scores[k][metric]}
        for k in KEYS
        for metric in ("cost", "latency", "quality")
    ]
)
st.bar_chart(chart_df, x="agent", y="score", color="metric", stack=False)

# Raw numbers + judge rationale, for credibility.
raw_df = pd.DataFrame(
    [
        {
            "Agent": TITLES[k],
            "Cost ($)": round(agent_cost(history.get(k, [])), 4),
            "Latency (s)": round(agent_latency(history.get(k, [])), 1),
            "Quality": verdicts[k].overall if verdicts.get(k) else "—",
            "Structure": verdicts[k].structure if verdicts.get(k) else "—",
            "Verdict": verdicts[k].rationale if verdicts.get(k) else "no script produced yet",
        }
        for k in KEYS
    ]
)
st.dataframe(raw_df, width="stretch", hide_index=True)
