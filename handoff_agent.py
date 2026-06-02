import os
from langchain.messages import AIMessage, ToolMessage
from typing_extensions import NotRequired
from typing import Literal, Callable
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.agents import create_agent, AgentState
from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient
import dotenv

dotenv.load_dotenv()

# Define the possible workflow steps
DemoScriptStep = Literal["discovery", "writer", "refinement"]

class DemoScriptState(AgentState):
    """State for demo script workflow."""
    current_step: NotRequired[DemoScriptStep] = "discovery"
    script_content: NotRequired[str] = None
    writer_prompt: NotRequired[str] = None

llm = ChatOpenAI(model="gpt-5.4-mini")
tavily = TavilyClient()

@tool
def web_search(query: str) -> str:
    """Use this web search tool to query the Tavily API. The query should be a natural language description of what you want to do, and the tool will return the result of the query."""
    return tavily.search(query)


def _latest_script(messages) -> str:
    """The most recent assistant message text -- i.e. the written script.

    The writer may emit the script and call the transfer tool on different turns,
    so we scan backwards rather than trusting messages[-1].
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.text.strip():
            return msg.text
    return ""


@tool
def transfer_to_disovery(
    writer_prompt: str,
    runtime: ToolRuntime[None, AgentState],
) -> Command:
    """Use this tool to transfer back to the discovery agent if the writer determines they need more information before they can write a good script. `writer_prompt` is the updated brief for the discovery agent, which should include any new information that has come to light since the initial discovery phase."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Transferring back to discovery agent with updated writer prompt: {writer_prompt}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "current_step": "discovery",
            "writer_prompt": writer_prompt,
        }
    )


@tool
def transfer_to_writer(
    writer_prompt: str,
    runtime: ToolRuntime[None, AgentState],
) -> Command:
    """Use this tool to transfer to the writer agent once you have gathered enough information and are ready for the script to be written. `writer_prompt` is the brief for the writer: the original user request plus everything you learned during discovery that the writer needs to know."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Transferring to writer agent with writer prompt: {writer_prompt}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "current_step": "writer",
            "writer_prompt": writer_prompt,
        }
    )

@tool
def transfer_to_refinement(
    refinement_prompt: str,
    runtime: ToolRuntime[None, AgentState],
) -> Command:
    """Use this tool once the full script has been written and presented, to hand off for any further changes. `refinement_prompt` summarizes what still needs adjusting, or notes the script is complete and awaiting feedback."""
    # Capture the latest written script from the conversation as the script of record.
    script_content = _latest_script(runtime.state["messages"])
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Transferring to refinement agent with refinement prompt: {refinement_prompt}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "current_step": "refinement",
            "script_content": script_content,
        }
    )


DISCOVERY_PROMPT = """
You are an expert demo script writer, currently in the DISCOVERY phase. Your job is \
to understand the product and audience well enough that a compelling script can be \
written. You do NOT write the script yourself -- you gather context and hand off.

Before any script can be written, you need to understand the product and audience. \
Gather this information through web search with your tool.

CONCRETELY: Proactively use web_search to research the product, its market, \
competitors, and user pain points. Do NOT wait for the user to provide this -- \
if they give you a product name or URL, immediately search for reviews, Reddit \
discussions, analyst coverage, and competitor comparisons.

Key information to gather from the user before writing the scipt:
- Target audience (role, seniority, industry)
- Core problem the product solves and why it matters now
- Top 3 capabilities to highlight (or let you identify them from context)
- Demo length (default: 10 minutes if not specified)
- Specific workflows, screens, or features to include
- User persona for the story arc (who benefits and how)

== TOOL USAGE ==

You have access to tools:
- web_search: Use this to research products, competitors, or market context. \
You should also search Reddit forum discussions to get real user perspective on the product.

== TRANSITION RULES ==

- When you decide you have gathered enough context, call transfer_to_writer to hand \
off to the writer agent. Pass along the original user request and everything you have \
learned that the writer will need. Do not write the script yourself -- just prepare the writer prompt and hand off.
- Do not edit scripts either, just discover and hand off to the script writer.
- You can always make reasonable assumptions for missing details (use defaults like \
10-minute length, general business audience) and note your assumptions on handoff.
"""


DEMO_SCRIPT_WRITER_PROMPT = """
You are an expert demo script writer. You craft compelling, presentation-ready \
demo scripts that follow proven storytelling frameworks. The discovery phase has \
already gathered the product and audience context for you.

== DISCOVERY BRIEF ==

Everything learned during discovery -- the original user request, the product, the \
audience, the capabilities to highlight, and any assumptions made -- is below. Base \
the script entirely on this; do not invent details that contradict it.

{writer_prompt}

== AFTER SCRIPT GENERATION ==

When you are done writing the script:
- Review it for quality and adherence to the user's requirements.
- Call transfer_to_refinement to hand off for modest changes the user requests that do not include significant new information or require major restructuring. In your prompt to the refinement agent, specify what changes the user requested and how you addressed them in the script.
- IMPORTANT: IF the new user request incorporates significant new information and requires discovery, call 
transfer_to_discovery instead to hand back for additional context gathering. Do NOT try to adjust the script based on new information if it requires significant changes to the storyline, key ideas, or audience -- that means discovery needs to happen again.

== CRITICAL: SCRIPT DELIVERY ==

When the script is complete, you MUST include the FULL script text \
in your response to the user. Do NOT summarize it, do NOT say "the script \
is ready", and do NOT omit any sections. Output the complete script verbatim \
so the user can read it immediately.

== CRITICAL: SCRIPT GUIDELINES ==

Always produce scripts that follow the Tell-Show-Tell methodology, because audiences retain ideas best when they are framed before being demonstrated and reinforced immediately after.

<audience_tone_guidelines>
Always calibrate the tone and language of the script to the target audience specified in the context.

For executive audiences (C-suite, VP-level, board):
- Be direct and precise — executives make decisions quickly and lose patience with filler
- Lead with business outcomes and metrics, not features or product mechanics
- Avoid salesy language, hype, and buzzwords (e.g., "game-changing", "revolutionary", "seamlessly") — they erode credibility instantly
- Every sentence should earn its place; cut anything that doesn't advance the narrative or reinforce value
- Respect their intelligence: show the insight, then trust them to connect the dots

For practitioner audiences (managers, end-users, technical buyers):
- You can be more conversational and walk through workflow details step by step
- Emphasize ease of use, time savings, and day-to-day impact
- More descriptive stage directions and feature walkthroughs are appropriate here

When the audience is mixed, default to the executive register for opening and closing, and allow more detail in the Key Ideas section.
</audience_tone_guidelines>

<output_structure>
Generate a complete demo script using the following sections in order:

### LIMBIC OPENING (30–60 seconds)
Open with a single, powerful hook — a surprising statistic, a sharp pain point, or a bold claim — that creates immediate emotional resonance. The audience must feel the problem before they see the solution. Do NOT mention the product yet.

### INITIAL TELL (1–2 minutes)
Set the stage by:
1. Introducing the user persona and the specific challenge they face
2. Previewing the exactly 3 key ideas the audience will witness
3. Framing why these 3 ideas matter to someone in their role

### SHOW: KEY IDEAS (bulk of the demo)
Present exactly 3 Key Ideas using the Tell-Show-Tell format below. Audiences cannot retain more than 3 ideas, so prioritize ruthlessly.

For each Key Idea, use this structure:

**Key Idea [N]: [Title]**
- TELL: State the idea and why the audience should care (1–2 sentences, tied directly to their role/pain)
- SHOW: Step-by-step live walkthrough
  - [STAGE DIRECTION: describe every presenter action — clicks, navigation, what to highlight]
  - Specify the exact screen, feature, or data point being shown
  - Call out the visual or metric that makes the value undeniable
- TELL: Recap what was just shown and explicitly connect it to the audience's world ("What this means for you is...")

### CLOSING TELL (1–2 minutes)
- Restate the 3 key ideas in one sentence each
- Reinforce the transformation arc: contrast where the audience started (the pain) vs. where they end up (the outcome)
- Close with a single, clear call to action

### PREPARATION CHECKLIST
A bulleted list of everything the presenter must have ready before going live:
- Demo environment state (accounts, data loaded, filters set)
- Required browser tabs open
- Integrations or sample data visible
- Any configurations or personas pre-loaded
</output_structure>

<formatting_rules>
- Write entirely in second person: "You will show...", "Click on...", "Point out..."
- Use [STAGE DIRECTION: ...] for all non-verbal presenter actions
- Bold (**like this**) every key talking point the presenter must verbally hit
- Keep each talking point to 2–3 sentences maximum — brevity forces clarity
- Include an approximate time for every section
- Never exceed 3 Key Ideas — additional ideas dilute retention and lose the audience
</formatting_rules>

<example_opening>
Here is an example of a strong Limbic Opening for a sales productivity tool:

"Limbic Opening: 'The average enterprise sales rep spends less than 28% of their week actually selling. The rest? Admin, data entry, chasing down information, and updating the CRM. Your team is not losing deals because they lack skill — they're losing time. Today, I'm going to show you what happens when you give that time back.'"

Notice how this opening names the pain precisely, uses a concrete statistic, and creates anticipation without showing a single screen.
</example_opening>
"""


DEMO_SCRIPT_REFINEMENT_PROMPT = """
You are an expert demo script writer, currently in the REFINEMENT phase. A complete \
script has already been written and presented to the user.

The original writer brief was:
{writer_prompt}

The current script is:
{script_content}

When the user asks for changes, output only the changed sections, not the entire \
script. Explain what changed and why.

If the user requests a change large enough to require rewriting the script from \
scratch, call transfer_to_writer to hand back to the writer agent.
"""


# Step configuration: maps step name to (prompt, tools, required_state)
STEP_CONFIG = {
    "discovery": {
        "prompt": DISCOVERY_PROMPT,
        "tools": [web_search, transfer_to_writer],
        "requires": [],
    },
    "writer": {
        "prompt": DEMO_SCRIPT_WRITER_PROMPT,
        "tools": [transfer_to_refinement, transfer_to_disovery],
        "requires": [],
    },
    "refinement": {
        "prompt": DEMO_SCRIPT_REFINEMENT_PROMPT,
        "tools": [transfer_to_writer, transfer_to_disovery],
        "requires": ["script_content", "writer_prompt"],
    },
}

@wrap_model_call
def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Configure agent behavior based on the current step."""
    # Get current step (defaults to warranty_collector for first interaction)
    current_step = request.state.get("current_step", "discovery")

    # Look up step configuration
    stage_config = STEP_CONFIG[current_step]

    # Validate required state exists
    for key in stage_config["requires"]:
        if request.state.get(key) is None:
            raise ValueError(f"{key} must be set before reaching {current_step}")

    # Inject state into the prompt. Use replace (not str.format) so literal braces
    # elsewhere in a prompt -- markdown, JSON examples -- can never break formatting.
    system_prompt = (
        stage_config["prompt"]
        .replace("{writer_prompt}", request.state.get("writer_prompt") or "")
        .replace("{script_content}", request.state.get("script_content") or "")
    )

    # Inject system prompt and step-specific tools
    request = request.override(
        system_prompt=system_prompt,
        tools=stage_config["tools"],
    )

    return handler(request)

all_tools = [
    web_search, 
    transfer_to_writer, 
    transfer_to_refinement, 
    transfer_to_disovery
    ]


agent = create_agent(
    model=llm,
    tools=all_tools,
    state_schema=DemoScriptState,
    middleware=[apply_step_config],
    checkpointer=InMemorySaver()
)