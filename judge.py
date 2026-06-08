"""Online LLM judge for demo scripts.

A single Claude Opus call scores a finished script on two axes -- overall quality
and Tell-Show-Tell structure adherence -- and returns a structured verdict. One
consistent judge across all three agents keeps the comparison fair, and Opus is a
different model from both the GPT manager and the Claude Sonnet writer, so it never
grades its own output.
"""

import anthropic
from pydantic import BaseModel, Field
import dotenv

dotenv.load_dotenv()

client = anthropic.Anthropic()


class Verdict(BaseModel):
    is_script: bool = Field(
        description="True only if the text is an actual demo script, not a clarifying "
        "question or discovery turn."
    )
    overall: int = Field(
        ge=0, le=100,
        description="Overall demo-script quality, 0-100: writing quality, audience "
        "calibration, narrative tension, and concreteness.",
    )
    structure: int = Field(
        ge=0, le=100,
        description="Adherence to the required Tell-Show-Tell structure, 0-100.",
    )
    rationale: str = Field(description="2-3 sentences justifying the two scores.")


class WinnerVerdict(BaseModel):
    winner: str = Field(
        description="The id of the single best script, copied exactly from the "
        "candidate header it appears under."
    )
    reasoning: str = Field(
        description="3-5 sentences explaining why this script wins over the others, "
        "citing concrete quality and Tell-Show-Tell structure differences."
    )


# Shared evaluation criteria — both the per-script judge and the comparative judge
# grade on the same two axes so the winner pick is consistent with the scores.
CRITERIA = """== quality (0-100) ==
Writing quality and persuasive craft:
- Audience calibration: tone matches the stated audience (executive vs. practitioner).
- Narrative tension: a clear pain -> transformation arc, not a feature list.
- Concreteness: specific screens, metrics, and stage directions -- not vague claims.
- Discipline: no hype/buzzwords for executive audiences; every line earns its place.

== structure (0-100) ==
Strict adherence to the Tell-Show-Tell specification. Penalize each deviation:
- Sections present and in order: LIMBIC OPENING, INITIAL TELL, SHOW: KEY IDEAS,
  CLOSING TELL, PREPARATION CHECKLIST.
- Limbic Opening hooks emotionally and does NOT name the product yet.
- EXACTLY 3 Key Ideas -- not 2, not 4.
- Each Key Idea follows TELL -> SHOW -> TELL, with [STAGE DIRECTION: ...] in the SHOW.
- Approximate time markers on every section; second-person voice throughout."""


JUDGE_PROMPT = f"""You are an expert evaluator of B2B demo scripts used by enterprise \
sales teams. You grade a single produced script on two independent axes.

{CRITERIA}

If the text is NOT a script (e.g. a clarifying question or research summary), set
is_script=false and score both axes 0. Be a discerning, calibrated grader: reserve
90+ for scripts that are genuinely presentation-ready."""


WINNER_PROMPT = f"""You are the head judge of a demo-script competition. Several agents \
each produced a script for the same task; you must choose the SINGLE best one.

Use the same two axes your panel uses, weighing persuasive writing quality and strict
Tell-Show-Tell structural adherence together:

{CRITERIA}

Compare the candidates head to head. Pick the one that is most presentation-ready
overall. Return the winning candidate's id exactly as it appears in its header, and
explain in 3-5 sentences why it beats the others -- cite specific quality and structure
differences, not vague praise."""


def judge_script(script_text: str) -> Verdict:
    """Score one demo script. Returns a structured, validated Verdict."""
    resp = client.messages.parse(
        model="claude-opus-4-8",
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=JUDGE_PROMPT,
        messages=[{"role": "user", "content": f"Evaluate this output:\n\n{script_text}"}],
        output_format=Verdict,
    )
    return resp.parsed_output


def pick_winner(scripts: dict[str, str]) -> WinnerVerdict | None:
    """Compare candidate scripts and return the winning agent id + reasoning.

    `scripts` maps agent id -> script text; entries without a script are skipped.
    The returned `winner` is guaranteed to be one of the supplied ids.
    """
    candidates = {k: v for k, v in scripts.items() if v}
    if not candidates:
        return None

    blocks = "\n\n".join(
        f"=== CANDIDATE id={k} ===\n{text}" for k, text in candidates.items()
    )
    resp = client.messages.parse(
        model="claude-opus-4-8",
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=WINNER_PROMPT,
        messages=[{"role": "user", "content": f"Candidate scripts:\n\n{blocks}"}],
        output_format=WinnerVerdict,
    )
    verdict = resp.parsed_output
    # Guard against a hallucinated id: fall back to the first candidate.
    if verdict.winner not in candidates:
        verdict.winner = next(iter(candidates))
    return verdict
