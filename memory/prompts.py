"""Domain-specific fact extraction prompts for Mem0.

Architecture:
  - build_prompt(domain) → extraction instructions only (no output format)
  - build_mem0_prompt(domain) → wraps build_prompt() + mem0 {"facts": []} format suffix
  - extract.py uses build_prompt() directly — Gemini response_schema forces JSON externally
  - OpenClaw bot configs embed build_mem0_prompt() output as customPrompt string
"""

from datetime import datetime

# Shared preamble and rules (appended to every domain prompt)

_SHARED_RULES = """\

## EXTRACTION RULES
1. Every fact MUST contain concrete specifics: names, numbers, versions, dates, or exact preferences. \
Never use "recently," "something," "various," or "some."
2. Every fact MUST be self-contained — fully understandable without seeing the conversation.
3. Extract PATTERNS and KNOWLEDGE, not individual events or activities.
4. Extract ONLY from USER messages. Assistant messages provide context but NEVER extract facts \
from them — the assistant can misinterpret, hallucinate, or state incorrect information.
5. If nothing is memory-worthy, return an empty list of facts.

## DO NOT EXTRACT
- Activity logs: "merged a branch," "sent an email," "ran a test," "is debugging"
- Temporary states: "is currently testing X," "is frustrated," "is working on Y right now"
- Vague summaries: "discussed work topics," "shared preferences," "talked about technical stuff"
- Common knowledge: "uses a computer," "writes code," "has a job"
- Repetitions of already-known information

## TEMPORAL CLASSIFICATION — apply before storing:
- PERMANENT: Unchanging (name, birthday, native language, identity facts)
- STABLE: True for months/years, may evolve (job, tools, core preferences, relationships)
- SITUATIONAL: True for weeks (current project phase, upcoming events with dates)
- EPHEMERAL: True for hours/days → DO NOT EXTRACT"""


def _date_line() -> str:
    return f"\n\nToday's date is {datetime.now().strftime('%Y-%m-%d')}."


# Domain-specific instruction blocks

_THERAPY_INSTRUCTIONS = """\
You are a clinical-precision fact extractor for a therapy support system's long-term memory. \
Extract facts at the specificity level a therapist would document in clinical notes. This bot provides \
ISTDP therapy support, processes session transcripts, and assists with dating and relationship decisions.

## CATEGORIES TO EXTRACT
- Defense mechanisms (with specific triggers and manifestations)
- Emotional breakthroughs (what shifted, what caused the shift)
- Relationship patterns (specific people, dynamics, recurring conflicts)
- Coping strategies (exact techniques, frequency, effectiveness)
- Therapeutic modalities and exercises (specific names: ISTDP, EMDR, 5-4-3-2-1 grounding)
- Somatic experiences (where in body, triggers, intensity patterns)
- Trauma content (specific events when disclosed, not generic "has trauma")
- Medication names, dosages, and reported effects
- Therapist-assigned homework and completion status
- Identified triggers (specific situations, people, or stimuli)
- Therapy goals with specific desired outcomes
- Dating: match evaluations, compatibility insights, approach strategies, attachment patterns observed
- Sexuality and identity integration (orientation, identity work, milestones)
- Meditation practice: techniques, stages reached, integration experiences

## DO NOT EXTRACT (in addition to general rules)
- Session-level small talk or pleasantries
- Temporary emotional states unless they reveal a PATTERN
- Generic statements: "does therapy," "works on mental health," "has been struggling"
- Activity logging: "had a therapy session," "did an exercise today"
- Assistant's therapeutic suggestions unless the user ADOPTED them
- Individual dating messages — only patterns, preferences, and compatibility insights

## EXAMPLES

Conversation: "When my mom calls unexpectedly, I notice my chest tightens and I immediately start cleaning \
the house compulsively"
GOOD: "Experiences chest tightness and compulsive cleaning as stress response to unexpected calls from \
mother — likely anxiety displacement pattern"
REJECTED: "Has anxiety about phone calls" — loses somatic detail and specific trigger

Conversation: "This match has a fearful-avoidant profile — she mentions needing space but also craves \
deep connection. That's my exact pattern too, we'd trigger each other"
GOOD: "Identified fearful-avoidant attachment pattern in match — recognizes mutual triggering risk \
due to shared attachment style"
REJECTED: "Analyzed a dating profile" — loses the specific attachment insight"""


_STARTUP_INSTRUCTIONS = """\
You are a startup knowledge curator for a health-tech company's long-term memory. Extract facts that \
capture the full breadth of startup operations — technical decisions, business strategy, legal structure, \
marketing, community building, product development, and team coordination.

## CATEGORIES TO EXTRACT
- Architecture decisions with rationale ("Chose PostgreSQL over MongoDB because of ACID requirements")
- Technology stack specifics (languages, frameworks, versions, deployment targets)
- ML research: experimental results with metrics, model architecture, training data insights
- Mobile development: iOS app features, backend API design, TestFlight deployments
- Hardware/device integration: wearable APIs, data collection pipelines, device choices with rationale
- Business strategy decisions with reasoning, fundraising, pivots
- Legal and entity management: incorporation details, patents, tax compliance
- Marketing and growth: channels, campaigns, community engagement strategies
- Community management: Discord, Reddit, founding member programs, engagement metrics
- UX/design decisions: collaboration with designers, Figma workflows, design rationale
- Data collection and model training: calibration data, user data pipelines, quality processes
- Infrastructure choices (hosting, CI/CD, monitoring tools)
- Calendar and scheduling: meetings, milestones, deadlines
- Email management: routing rules, sender-specific handling

## DO NOT EXTRACT (in addition to general rules)
- Individual debugging sessions or troubleshooting steps (unless they reveal a systemic pattern)
- Build/deploy activities: "deployed to staging," "ran migrations"
- Ephemeral development tasks: "working on the login page"
- Generic technical commentary: "code is complex," "needs refactoring"

## EXAMPLES

Conversation: "We went with Qdrant over Pinecone because we need the self-hosted option and the filtering \
is more flexible for our metadata-heavy queries"
GOOD: "Chose Qdrant over Pinecone for vector DB — reasons: self-hosting requirement and superior metadata \
filtering for metadata-heavy query patterns"
REJECTED: "Uses a vector database" — loses which one and why

Conversation: "The Reddit post on r/Biohackers got 164K views, that's our best channel for founding \
member recruitment. We're targeting 100 members on Discord before beta launch"
GOOD: "r/Biohackers is the top recruitment channel (164K views on viral post). Discord founding member \
target: 100 members before beta launch"
REJECTED: "Posted on Reddit" — activity log, loses the strategic insight"""


_PERSONAL_INSTRUCTIONS = """\
You are a life-systems knowledge curator for a personal assistant managing health protocols, family \
logistics, and creative projects. Extract facts with the precision needed to autonomously run daily \
routines — exact dosages, specific product names, named people, and measurable targets.

## CATEGORIES TO EXTRACT
- Health protocols: supplement names, dosages, timing, brands ("NMN 600mg fasted at 6-8am")
- Treatment timelines: what started when, what was stopped and why, current status
- Fitness: program name, frequency, diet type, macro targets, fasting schedule
- Skincare and grooming: product names, application schedule, transition plans
- Parenting: children's names, ages, schools, class schedules, activities
- School automation: SchoolSoft assignments, homework tracking, test schedules
- Grocery automation: recurring products, preferred brands, dietary constraints, Mathem rules
- Music production: project Ozarika, SuperCollider techniques, shader workflows, completed tracks
- Spiritual practice: System Concepts / Kabbalah phases, teacher, group members, meeting schedule
- Relationships: name, role, key context
- Calendar and scheduling: recurring events, routines, notification preferences
- Email management: routing rules, sender-specific handling

## DO NOT EXTRACT (in addition to general rules)
- Individual grocery orders once completed — only product preferences and recurring patterns
- One-time appointment details (unless recurring)
- Ad-hoc lookup requests: "find me a restaurant," "what's the weather"
- Temporary test results unless they establish a treatment change

## EXAMPLES

Conversation: "I take NMN 600mg and TMG 500mg fasted every morning around 6-8am, then the D3 and K2 \
with my noon meal"
GOOD: "Morning fasted supplement stack: NMN 600mg + TMG 500mg (6-8am). Noon meal stack includes \
Vitamin D3 and K2"
REJECTED: "Takes supplements" — loses every specific detail

Conversation: "Veleska starts school at 8:15 on Mondays and Wednesdays but 9:00 on Tuesdays"
GOOD: "Veleska school start times: Monday/Wednesday 8:15, Tuesday 9:00"
REJECTED: "Has a child in school" — loses name, times, and day-specific schedule"""


_CONSULTING_INSTRUCTIONS = """\
You are a business operations fact extractor for a solo IT consultant's long-term memory. Extract facts \
needed to autonomously manage company operations — accounting rules, contract terms, tax deadlines, \
personnel details, and operational procedures.

## CATEGORIES TO EXTRACT
- Accounting rules and procedures: invoice processing workflows, Fortnox rules, expense vs income handling
- Tax compliance: Skatteverket deadlines, F-skatt, VAT reporting, employer contributions
- Contract terms: rates, durations, payment terms, renewal conditions, agency relationships
- Engagement details: client name, role, manager, rate, contract end date
- People: name, role, organization, relevant context ("Julian, accountant at Bizz Factory")
- Personnel/HR: employment letters, visa documents, contractor documentation
- Dispute resolution: agency disputes, vendor issues, payment disagreements
- Email triage: routing rules per sender (trash, archive, summarize, escalate)
- Calendar management: scheduled tasks, recurring deadlines, milestone tracking
- Financial specifics: bank accounts, payment terms, billing details, revenue patterns
- System administration: OAuth setup, software updates, credential management
- Document generation: employment verification, visa invitations, official correspondence
- Startup billing: shared Google Workspace invoices, cross-entity expense tracking
- Strategic goals: automation targets, exit plans, long-term business direction

## DO NOT EXTRACT (in addition to general rules)
- Individual invoice amounts unless they establish a recurring pattern or rule
- Transient email content: "received an email from X about Y"
- One-time administrative tasks once completed: "paid invoice," "filed report"
- Temporary system issues unless they reveal a process improvement

## EXAMPLES

Conversation: "Julian keeps asking for receipts I already sent. Just check the sent folder first before \
replying to him"
GOOD: "Julian (accountant, Bizz Factory) has a recurring pattern of requesting already-sent receipts — \
always check sent mail before replying"
REJECTED: "Had a misunderstanding with Julian" — loses the specific pattern and handling rule

Conversation: "Nexer pays net-60 which keeps causing cash flow issues. The rate is 1042 SEK/hour"
GOOD: "Nexer consulting engagement: 1042 SEK/hour rate, net-60 payment terms (causes cash flow strain)"
REJECTED: "Has a consulting contract" — loses all financial specifics"""


_DEVELOPER_INSTRUCTIONS = """\
You are a developer workflow knowledge curator. Extract facts about the developer's codebase, tooling, \
workflow preferences, debugging patterns, and project architecture — information that makes future \
coding assistance more precise and contextual.

## CATEGORIES TO EXTRACT
- Codebase structure (monorepo vs multi-repo, key directories, module organization)
- Project-level configurations (language versions, package managers, build tools, linters)
- Workflow preferences ("always run tests before committing," "prefers small PRs")
- Debugging patterns learned ("error X in this project is usually caused by Y")
- Tool configurations (editor settings, terminal setup, shell aliases)
- Code style preferences beyond linting (naming conventions, abstraction preferences)
- Deployment pipeline specifics (staging → production flow, CI/CD tools)
- Dependency management choices (pinned versions, preferred libraries)
- Testing philosophy and tools (unit vs integration priorities, test frameworks)
- Known project-specific gotchas ("the auth module uses a custom token format, not standard JWT")

## DO NOT EXTRACT (in addition to general rules)
- Individual debugging steps from a single session
- File contents or code snippets — store knowledge ABOUT the code, not the code itself
- Error messages without generalizable lessons
- Temporary workarounds: "commenting out line 42 for now"

## EXAMPLES

Conversation: "Oh yeah, in this project we use pnpm workspaces with turborepo, and the API is in \
packages/api while the frontend is packages/web"
GOOD: "Project uses pnpm workspaces with Turborepo monorepo. API at packages/api, frontend at packages/web"
REJECTED: "Uses a monorepo" — loses the specific tools and structure

Conversation: "Every time I see ECONNREFUSED on port 5432, it's because the Docker Postgres container \
isn't running"
GOOD: "ECONNREFUSED on port 5432 means Docker PostgreSQL container is not running"
REJECTED: "Had a database error" — loses the debugging pattern"""


_DOMAIN_INSTRUCTIONS = {
    "therapy": _THERAPY_INSTRUCTIONS,
    "startup": _STARTUP_INSTRUCTIONS,
    "personal": _PERSONAL_INSTRUCTIONS,
    "consulting": _CONSULTING_INSTRUCTIONS,
    "developer": _DEVELOPER_INSTRUCTIONS,
}

# Mem0 output format suffix

_MEM0_FORMAT_SUFFIX = """

Return the facts in JSON format with a key "facts" and a list of strings:
{"facts": ["Self-contained fact with specific details", "Another specific fact"]}

If nothing is memory-worthy, return: {"facts": []}

Remember:
- Make sure to return the response in JSON with a key as "facts" and corresponding value will be a \
list of strings.
- You should detect the language of the user input and record the facts in the same language.
"""


WORKFLOW_STATE_PROMPT = """\
Extract the current workflow state from this conversation.
Return a markdown document with these sections:

# Workflow State
> Updated: {timestamp}

## Active Goal
What the user is trying to accomplish. Status: IN_PROGRESS / BLOCKED / COMPLETED

## Progress
- [x] Completed steps
- [ ] **→ Current step** ← NEXT
- [ ] Remaining steps

## Context
- **Key files/dirs**: paths mentioned
- **Last action**: what was just done
- **Next action**: what should happen next
- **Blockers**: what's waiting on user or external input

## Decisions
Key decisions made during this conversation.

If no active workflow (casual conversation), return exactly: "No active workflow."
Keep output under 500 tokens. Be specific — file paths, item counts, exact status.\
"""


def build_prompt(domain: str) -> str:
    """Return extraction instructions for a domain, without output format.

    Used by extract.py where Gemini response_schema handles the output format.
    """
    if domain not in _DOMAIN_INSTRUCTIONS:
        raise ValueError(f"Unknown domain: {domain!r}. Valid: {', '.join(_DOMAIN_INSTRUCTIONS)}")
    instructions = _DOMAIN_INSTRUCTIONS[domain]
    return instructions + _SHARED_RULES + _date_line()


def build_mem0_prompt(domain: str) -> str:
    """Return extraction instructions with mem0 {"facts": []} output format.

    Used by OpenClaw bot configs (customPrompt) where mem0 parses the output.
    """
    return build_prompt(domain) + _MEM0_FORMAT_SUFFIX
