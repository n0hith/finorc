# FinAgent

A multi-agent financial research assistant. Given a research question, three
coordinating agents — **Planner**, **Analyst**, **Critic** — produce a
grounded, structured answer, with a custom NLI-based verification layer
(Phase 2+) checking that every claim the Analyst makes is actually supported
by the data it retrieved.

Orchestration between agents is hand-rolled (no LangGraph/CrewAI/AutoGen) so
every part of the control flow is explicit and easy to reason about.

## Status: Phase 5 complete

Agent skeleton, hand-rolled orchestration loop, NLI-based groundedness
verification between the Analyst and Critic, real financial news ingestion
from RSS feeds, FinBERT sentiment scoring on that ingested news, and the
FastAPI backend now streams every orchestration event live over SSE instead
of blocking on one big JSON response. No eval harness or CI yet — that's
Phase 6.

A FastAPI backend and static frontend were added ahead of Phase 7's "real"
frontend so the pipeline could be demoed in a browser instead of a terminal.
As of Phase 5 the backend streams live, matching the plan; the frontend
itself is still intentionally simple (vanilla HTML/JS, no framework) — see
"Frontend / API" below for what it is and isn't.

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
                       structured event; the CLI prints them, the API (Phase
                       5) streams the same events over SSE - same schema,
                       different sink.
    loop.py           The orchestration loop itself: Planner -> Analyst ->
                       Critic -> (revise up to N times) -> next task.
  data/
    stub_source.py    Placeholder "financial news" source (4 fixed fake
                       articles). Still used by `tests/test_groundedness.py`
                       for a deterministic, network-free check of the
                       verifier; no longer wired into the live pipeline.
    rss_source.py     Phase 3's real data source: pulls recent headlines from
                       a fixed set of financial RSS feeds (CNBC, MarketWatch,
                       Yahoo Finance, Investing.com) and ranks them against
                       the query by keyword overlap, same as the stub. Same
                       `fetch(query, limit) -> list[DataSnippet]` signature,
                       so Analyst/orchestration code didn't change at all.
  verification/
    nli.py             Local NLI model wrapper (entailment/neutral/
                        contradiction between a source snippet and a claim).
    groundedness.py     Runs NLI over every Analyst claim against the
                        snippet(s) it cites; flags claims not clearly
                        entailed. Non-LLM, deterministic, testable in
                        isolation.
  analysis/
    finbert.py          Local FinBERT sentiment classifier (positive/
                        negative/neutral). Same shape as verification/nli.py:
                        standalone, non-LLM, loaded once per process.
  api/
    main.py            FastAPI app. POST /api/research streams the
                       orchestration event trace live as SSE (Phase 5).
    static/index.html   Single-page vanilla HTML/JS frontend that consumes
                        the SSE stream and renders results once it ends.
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
against the Anthropic API; switched to Gemini because that's the API key
available for this project. The provider is isolated entirely inside
`agents/base.py::BaseAgent._call_tool` — swapping providers again means
changing one file, not the three agent classes, since they only ever call
`self._call_tool(...)` with a JSON Schema. Default model is
`gemini-flash-lite-latest` (see "A note on free-tier quotas" below for why).

**Groundedness verification: a separate NLI step, not something inside the
Critic.** `finagent/orchestration/loop.py` calls
`verification.groundedness.verify(output)` right after the Analyst runs and
before the Critic does. It runs local NLI (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`,
via `transformers`) between each claim and the snippet(s) it cites, and flags
any claim not clearly *entailed* (neutral, contradicted, or citing nothing at
all counts as unsupported). The result is hard evidence handed to the Critic
as part of its prompt, not something the Critic has to notice on its own.
This was chosen over folding the check into the Critic's own call because it
keeps the verifier deterministic, non-LLM, and unit-testable in isolation
(`tests/test_groundedness.py` exercises it with zero Gemini calls) — and
because "the LLM grades its own homework" is a weaker groundedness signal
than an independent classifier the LLM then has to explain away.

### Phase 3: RSS ingestion

**Fixed feed list, not search.** `rss_source.py` pulls the latest ~20 items
from each of four general financial-news RSS feeds (CNBC, MarketWatch, Yahoo
Finance, Investing.com) — none of which support server-side query search —
then ranks the combined pool against the query by keyword overlap, exactly
like the Phase 1 stub did. The upgrade from Phase 1 isn't smarter ranking,
it's that the pool being ranked is real, current news instead of four fixed
fake articles. Feed URLs are hardcoded rather than env-configurable: there's
no deployer-side reason to swap them, so adding config surface here would be
unearned complexity.

**Fails open to an empty list, not to the stub.** If a feed is down, slow, or
malformed, it's logged and skipped — one bad feed doesn't sink a research
run. If every feed fails, or nothing in the combined pool overlaps the query
at all, `fetch()` returns `[]` rather than quietly substituting stub data.
This was a deliberate choice over "fall back to stub so there's always an
answer": a demo silently showing fabricated articles as if they were live
data is worse than the Analyst honestly reporting it has no relevant source
data for that sub-task (which is what happens in practice — see the example
below). `tests/test_groundedness.py` still imports `stub_source` directly for
its own deterministic, network-free assertions; the stub itself is otherwise
unused now.

**Short in-process cache, no persistence.** One research question fans out
into several Analyst sub-tasks, each calling `fetch()` independently within
seconds of each other. A feed-URL-keyed, 10-minute TTL in-memory cache avoids
re-downloading the same four feeds multiple times per run, without needing
any external cache (Redis, disk, etc.) for what is still a single-process
demo app.

**Observed effect on output quality.** General headline feeds don't always
cover a narrow sub-task (e.g. a Planner-generated "OPEC+ supply dynamics"
task may have zero matching live headlines on a given day). In that case the
Analyst correctly reports it has no relevant data rather than inventing a
claim, the groundedness verifier has nothing ungrounded to catch, and the
Critic approves the honest non-answer. This is real behavior, not a
hypothetical — it's a decent interview example of the pipeline doing the
right thing specifically *because* the Analyst is barred from using
knowledge outside the retrieved snippets.

**Known rough edge, not fixed in Phase 3.** The NLI model sometimes verdicts
a clearly-supported claim as `neutral` (flagged) when the Analyst paraphrases
the source heavily, since DeBERTa-MNLI's entailment judgment is sensitive to
surface wording, not just meaning. The Critic sees both the verdict and the
underlying snippet text, and in testing correctly overrides a spurious
`neutral` flag when the source obviously supports the claim — but this is the
verifier being a noisy signal the Critic weighs, not a guarantee. Worth
knowing for Phase 6's eval harness (groundedness *verifier* accuracy vs.
*pipeline* accuracy are different numbers).

### Phase 4: FinBERT sentiment scoring

**Scored at the source, not lazily at the point of use.** `rss_source.fetch()`
runs every snippet's title+text through FinBERT once, right when it's parsed,
and stores the result on `DataSnippet.sentiment` (a new optional field —
`None` for sources that don't score it, namely `stub_source`). The
alternative — scoring on-demand wherever a snippet is read — was rejected
because it would mean every consumer of `DataSnippet` needs to know sentiment
scoring exists and remember to call it; scoring at the source means
`sentiment` is just always populated for anything that came from a live feed,
with no second call for a consumer to forget.

**Sentiment is a labeled model output, not freeform text.** `SentimentScore`
(`label`: positive/negative/neutral, `score`: softmax confidence) is a small
dataclass, not a prose description — exactly like NLI's `verdict`/`score`
shape in Phase 2. This keeps it structured signal the Analyst's prompt can
present compactly (`[sentiment: negative 0.95]`) rather than another block of
text to parse.

**The Analyst is told, explicitly, that sentiment describes tone, not fact.**
FinBERT's output is scoring how an article is *written*, not asserting
anything that happened — "oil prices fell" being tagged `negative` doesn't
mean the fall is disputed, it means the article reads negatively. The
Analyst's system prompt says this directly: sentiment is secondary signal for
sub-tasks about market mood/outlook, never something to cite as a claim or
repeat as if it were a fact from the article. This boundary matters for
Phase 2's groundedness verifier too — a claim like "sentiment is negative" has
nothing in the article text to be NLI-entailed against, so letting the
Analyst state sentiment as a claim would produce a false hallucination flag
on something that's actually true, just not text-grounded in the normal
sense.

**Model choice: ProsusAI/finbert, not a general sentiment model.** Financial
language doesn't carry ordinary-English sentiment polarity ("earnings beat",
"guidance cut" aren't obviously positive/negative to a general model), so a
model fine-tuned specifically on financial text was used instead of a
generic one — same reasoning as Phase 2 picking an NLI model trained across
MNLI/FEVER/ANLI rather than a narrower one.

No new dependency was needed — `transformers`/`torch` were already pulled in
for Phase 2's NLI model, and FinBERT loads through the same
`AutoModelForSequenceClassification` API.

## Frontend / API

`finagent/api/main.py` exposes one endpoint, `POST /api/research`, that now
(Phase 5) **streams** the orchestration event trace to the client as Server-
Sent Events as the pipeline runs, instead of blocking on one big JSON
response. `finagent/api/static/index.html` is still a single vanilla
HTML/JS/CSS page (no build step, no framework, by design) — it posts a
question, reads the streamed response incrementally, updates a one-line
status as events arrive, and renders the plan/claims/verdicts once the
stream ends. It's a reasonable stand-in for the UI, but a real reasoning-
trace view (showing the Planner/Analyst/Critic timeline live, not just a
status line) is still Phase 7's job.

**Bridging sync orchestration with an async server.** `run_research()` and
the agents beneath it make blocking calls (Gemini, RSS), so making the
*whole* pipeline async would mean rewriting every agent and the orchestration
loop. Instead, each request's pipeline run happens on a background thread
that pushes `Event`s onto a `queue.Queue`; the response body is a plain
*synchronous* generator that blocks on `queue.get()`, which Starlette detects
and iterates in a thread pool automatically. The server's event loop is never
blocked, and `orchestration/loop.py` didn't need to change at all - it still
just takes a plain callback `event_sink`, which is now `queue.put` instead of
`print`.

**POST + manually-read stream, not GET + the browser's EventSource API.**
`EventSource` only supports GET requests with no body, which doesn't fit a
free-text research question. Instead the frontend calls `fetch()` and reads
`response.body` as a `ReadableStream`, buffering incoming text and splitting
on blank lines to recover each `data: <json>` message — the same
`text/event-stream` wire format, just parsed by hand instead of by a browser
API built around GET.

**A `research_error` event exists only at this layer.** `run_research()`
itself never catches exceptions - a CLI caller just lets them propagate and
crash normally, which is correct for a script. But an unhandled exception on
the background thread would otherwise vanish silently (nothing reads a
thread's return value), so `main.py` wraps the pipeline call, catches any
exception, and emits a `research_error` event with the message before closing
the stream, letting the frontend show it instead of hanging.

Run it:

```bash
uvicorn finagent.api.main:app --reload
```

Then open `http://localhost:8000`.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
cp .env.example .env  # then fill in GEMINI_API_KEY
```

Get a key at https://aistudio.google.com/apikey.

### A note on free-tier quotas

Free-tier Gemini keys have both a per-minute rate limit and a **separate,
much stricter per-day quota, tracked per model**. `gemini-3.5-flash`'s free
tier is 20 requests/day, which a single 4-sub-task research run (~9 calls)
burns through in two or three tries. `BaseAgent._generate_with_retry` retries
on 429s using the server's suggested `retryDelay`, which smooths over the
per-minute limit, but there's no working around an exhausted daily quota —
retries just fail again until the next day. The default model is therefore
`gemini-flash-lite-latest`, which sits in a separate daily-quota bucket and
held up fine under testing. If you hit `429 RESOURCE_EXHAUSTED` mentioning
`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, that's this limit, not a
bug — either wait for the daily reset or point `FINAGENT_MODEL` at a
different model with its own quota.

## Running it

```bash
python scripts/run_research.py "Why are AI data center buildouts accelerating, and what could constrain them?"
```

With no argument it runs a default sample question. Output is the full event
trace (one JSON block per event, to stdout) followed by a `FINAL RESULT`
section with each sub-task's summary and cited claims.

## Testing the groundedness verifier in isolation

```bash
python tests/test_groundedness.py
```

Builds a synthetic `AnalystOutput` against Phase 1's stubbed snippets with
one verbatim claim, one fabricated claim, and one uncited claim, and asserts
the verifier accepts the first and flags the other two. No Gemini calls
involved — this only exercises `finagent/verification/`. First run downloads
the ~370MB NLI model from HuggingFace and caches it locally.

## Testing the sentiment classifier in isolation

```bash
python tests/test_sentiment.py
```

Runs FinBERT on one clearly positive, one negative, and one neutral sentence
and asserts each gets the expected label. No Gemini calls, no RSS fetch —
only exercises `finagent/analysis/finbert.py`.

## Interface for Phase 6 (eval harness + CI)

Phase 6 needs a small, fixed benchmark set of research questions (with some
notion of expected plan coverage / known-answerable sub-topics) to run the
pipeline against repeatedly and score. Three things it can measure using
pieces that already exist, without new instrumentation:

- **Groundedness verifier accuracy** — `finagent/verification/groundedness.py`
  already returns a structured `GroundednessReport`; a benchmark with
  hand-labeled entailed/unsupported claims can check it against known-good
  verdicts directly (no Gemini call needed, same as `tests/test_groundedness.py`).
- **Critic catch rate** — how often the Critic's `verdict` agrees with the
  groundedness report's `flagged` claims, and how often it overrides a false
  flag correctly (the `neutral`-but-actually-fine case documented above is a
  real example worth building a fixture from).
- **Plan quality** — whether `Planner`'s sub-tasks are each answerable from
  what `rss_source.fetch()` can realistically return, which the "Analyst
  reports no data" pattern already surfaces as a signal rather than something
  to special-case.

This is a real design question Phase 6 should start with (what counts as a
benchmark "case," and whether grading is rule-based vs. another LLM call) —
flagging it now rather than deciding silently when that phase starts.
