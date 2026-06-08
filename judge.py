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


JUDGE_PROMPT = """You are an expert evaluator of B2B demo scripts used by enterprise \
sales teams. You grade a single produced script on two independent axes.

== overall (0-100) ==
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
- Approximate time markers on every section; second-person voice throughout.

If the text is NOT a script (e.g. a clarifying question or research summary), set
is_script=false and score both axes 0. Be a discerning, calibrated grader: reserve
90+ for scripts that are genuinely presentation-ready."""


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
