# FinAgent

A multi-agent financial research assistant. Given a research question, three
coordinating agents — **Planner**, **Analyst**, **Critic** — produce a
grounded, structured answer, with a custom NLI-based verification layer
(Phase 2+) checking that every claim the Analyst makes is actually supported
by the data it retrieved.

Orchestration between agents is hand-rolled (no LangGraph/CrewAI/AutoGen) so
every part of the control flow is explicit and easy to reason about.

## Status: Phase 1 complete

Agent skeleton, hand-rolled orchestration loop, and a stubbed data source.
No real data ingestion or groundedness verification yet — those are Phase 2+.

## Architecture

```
finagent/
  agents/           Planner, Analyst, Critic — each a thin wrapper around
                     one Gemini call, forced to respond via a single
                     structured function call (no prose parsing).
  orchestration/
    context.py       ResearchContext — shared mutable state threaded through
                      the loop (plan, analyst outputs, critic reviews).
    events.py         Event schema + EventEmitter. Every step emits a
                       structured event; Phase 1 just prints them, Phase 5
                       streams the same events over SSE.
    loop.py           The orchestration loop itself: Planner -> Analyst ->
                       Critic -> (revise up to N times) -> next task.
  data/
    stub_source.py    Placeholder "financial news" source. Its `fetch(query)
                       -> list[DataSnippet]` interface is the contract Phase
                       3's real RSS ingester must match.
  verification/       Phase 2: NLI groundedness checker goes here.
  api/                Phase 5: FastAPI + SSE endpoint goes here.
  config.py           Env-based settings (API key, model, revision cap).
scripts/
  run_research.py     CLI entry point — runs one research question and
                       prints the full event trace + final answer.
```

### Design decisions

**State passing: shared mutable context, not an immutable event chain.**
A single `ResearchContext` object accumulates the plan, analyst outputs, and
critic reviews as the loop runs, and is passed by reference. Agents read
`context.analyst_outputs` / `context.latest_review_for(task_id)` directly
instead of replaying an event log to reconstruct state. The event log in
`events.py` still exists, but it's a side channel for observability/streaming,
not the mechanism agents use to see prior state — that keeps the two concerns
(what agents need to function vs. what the UI needs to display) decoupled.

**Structured output via forced function calling, not prompt-and-parse.**
Every agent call sets Gemini's `function_calling_config` to `mode=ANY` with
`allowed_function_names` restricted to one function, so the model *must*
respond with a single, schema-validated function call. This eliminates
brittle "parse JSON out of prose" logic entirely. Gemini accepts plain JSON
Schema dicts via `parameters_json_schema`, so no proprietary schema dialect
had to be introduced.

**Revision loop: fixed cap, then force-accept.**
`FINAGENT_MAX_REVISIONS` (default 2) bounds how many times the Critic can
send a sub-task's analysis back to the Analyst. After the cap is hit, the
loop accepts the latest Analyst output regardless of verdict and moves on —
chosen for predictable cost/latency over a "fail loudly" alternative, since a
partially-caveated answer is more useful to a user than no answer.

**Event schema.** Every step emits one of: `research_started`,
`planner_done`, `analyst_step_started`, `analyst_step_done`,
`critic_review`, `revision_requested`, `research_completed` (plus
`verification_done`, reserved for Phase 2). Each event is `{type, data,
timestamp}` and JSON-serializable by construction — this is the exact shape
Phase 5 will push over SSE, so nothing about the event schema needs to change
when streaming is added, only the sink (`print_sink` -> an SSE queue sink).

**LLM provider: Gemini (`google-genai`), not Anthropic.** Originally built
against the Anthropic API; switched to Gemini (`gemini-3.5-flash` by default)
because that's the API key available for this project. The provider is
isolated entirely inside `agents/base.py::BaseAgent._call_tool` — swapping
providers again means changing one file, not the three agent classes, since
they only ever call `self._call_tool(...)` with a JSON Schema.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
cp .env.example .env  # then fill in GEMINI_API_KEY
```

Get a key at https://aistudio.google.com/apikey. The free tier caps
`gemini-3.5-flash` at 5 requests/minute — `BaseAgent` retries once on a 429,
honoring the server's suggested `retryDelay`, which is enough for a single
research run end-to-end but will add latency.

## Running it

```bash
python scripts/run_research.py "Why are AI data center buildouts accelerating, and what could constrain them?"
```

With no argument it runs a default sample question. Output is the full event
trace (one JSON block per event, to stdout) followed by a `FINAL RESULT`
section with each sub-task's summary and cited claims.

## Interface for Phase 2 (NLI groundedness verification)

Phase 2 slots a verifier between `analyst_step_done` and the Critic call in
`orchestration/loop.py`. It will consume an `AnalystOutput` (specifically
`claims: list[Claim]`, where each `Claim` has `text` and
`supporting_snippet_ids`) plus the `snippets_used: list[DataSnippet]` those
IDs point into, and check whether each claim's text is actually entailed by
the snippet(s) it cites. That result becomes a new `verification_done` event
and additional signal fed to the Critic — no changes needed to `Planner`,
`Analyst`, or the `ResearchContext` shape to support this.

## Interface for Phase 3 (RSS ingestion)

Phase 3 replaces `finagent/data/stub_source.py` with a real implementation
of the same signature:

```python
def fetch(query: str, limit: int = 3) -> list[DataSnippet]
```

`DataSnippet` (`id`, `source`, `title`, `text`, `published_at`, `url`) is
already source-agnostic — an RSS item maps onto it directly (item guid/link
-> `id`, feed name -> `source`, item title/description -> `title`/`text`,
pubDate -> `published_at`, link -> `url`). `Analyst` and the orchestration
loop only ever call `fetch(query)`, so this should be a swap of one import in
`scripts/run_research.py` (and wherever `Analyst` is constructed later), not
a change to `Analyst` itself.
