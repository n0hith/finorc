"""Stubbed financial data source.

This is the Phase 1 placeholder for what Phase 3 will replace with a real RSS
ingester. The Analyst and orchestration code only ever call `fetch(query)` and
consume `DataSnippet` objects, so the swap in Phase 3 should be a one-file
change: implement the same `fetch(query) -> list[DataSnippet]` signature
against real feeds and point the Analyst at it instead.
"""

from __future__ import annotations

from finagent.orchestration.context import DataSnippet

_FAKE_ARTICLES: list[DataSnippet] = [
    DataSnippet(
        id="stub-1",
        source="Reuters (stub)",
        title="Nvidia raises data center revenue guidance for Q3",
        text=(
            "Nvidia told investors it expects data center segment revenue to grow "
            "roughly 12% quarter-over-quarter in Q3, citing continued demand for "
            "its Blackwell-generation AI accelerators from hyperscale cloud customers."
        ),
        published_at="2026-08-10",
        url="https://example.com/stub/nvidia-q3-guidance",
    ),
    DataSnippet(
        id="stub-2",
        source="Bloomberg (stub)",
        title="TSMC advanced packaging capacity remains the binding constraint",
        text=(
            "TSMC executives said CoWoS advanced packaging capacity, not wafer "
            "fabrication, remains the primary bottleneck limiting how much AI "
            "accelerator supply can reach the market through 2026."
        ),
        published_at="2026-08-08",
        url="https://example.com/stub/tsmc-packaging",
    ),
    DataSnippet(
        id="stub-3",
        source="WSJ (stub)",
        title="Hyperscalers' 2026 capex plans point to continued AI infrastructure spend",
        text=(
            "The four largest US hyperscalers collectively guided to a further "
            "increase in 2026 capital expenditures, with executives attributing the "
            "bulk of the increase to AI training and inference infrastructure."
        ),
        published_at="2026-08-05",
        url="https://example.com/stub/hyperscaler-capex",
    ),
    DataSnippet(
        id="stub-4",
        source="FT (stub)",
        title="Analysts flag power availability as an emerging constraint on AI data centers",
        text=(
            "Several sell-side analysts noted that grid interconnection delays, rather "
            "than chip supply, are becoming the limiting factor for how quickly new AI "
            "data center capacity can come online in parts of the US."
        ),
        published_at="2026-08-01",
        url="https://example.com/stub/power-constraint",
    ),
]


def fetch(query: str, limit: int = 3) -> list[DataSnippet]:
    """Return the top `limit` stubbed snippets 'relevant' to `query`.

    Phase 1 relevance is intentionally trivial (keyword overlap) since the
    point of the stub is to exercise the Analyst/verifier plumbing, not to
    simulate a real ranking algorithm.
    """
    query_terms = set(query.lower().split())

    def score(snippet: DataSnippet) -> int:
        text = f"{snippet.title} {snippet.text}".lower()
        return sum(1 for term in query_terms if term in text)

    ranked = sorted(_FAKE_ARTICLES, key=score, reverse=True)
    return ranked[:limit]
