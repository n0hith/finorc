"""Real financial news data source, backed by a fixed set of RSS feeds.

This is the Phase 3 replacement for `stub_source.py`. It implements the exact
same `fetch(query, limit) -> list[DataSnippet]` signature, so the Analyst and
orchestration loop don't change at all - only the import in
`scripts/run_research.py` / `finagent/api/main.py` points here instead.

Design notes:

- **Fixed feed list, not configurable.** The feeds below are hardcoded rather
  than pulled from env/config. These are general financial news sources (not
  per-ticker or per-topic feeds), so there's nothing a deployer would
  realistically need to override - keeping them in code matches the stub's
  self-contained style and avoids config surface this project doesn't need.
- **Ranking, not search.** None of these feeds support server-side query
  search; they're just "latest headlines" firehoses. So, like the stub, this
  pulls a batch of recent entries from each feed and ranks them against the
  query by keyword overlap, then returns the top `limit`. The data is real;
  the relevance ranking is still a simple heuristic, same as Phase 1.
- **Fails open to an empty list, per feed and overall.** A dead/slow/malformed
  feed is logged and skipped rather than raising - one bad feed shouldn't
  break a research run. If every feed fails, or nothing overlaps the query,
  `fetch` returns `[]` rather than silently falling back to fake data: the
  Analyst then has to say "no relevant data" honestly instead of the pipeline
  pretending a live run succeeded when it didn't.
- **Short in-process cache per feed URL.** One research question fans out
  into several Analyst sub-tasks, each calling `fetch()` independently. Without
  caching that means re-downloading the same handful of feeds N times in a
  few seconds. A simple TTL cache keyed by feed URL avoids that without
  needing any external cache.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.request
from html import unescape

import feedparser

from finagent.orchestration.context import DataSnippet

logger = logging.getLogger(__name__)

_FEEDS: list[tuple[str, str]] = [
    ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("MarketWatch", "http://feeds.marketwatch.com/marketwatch/topstories/"),
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
    ("Investing.com", "https://www.investing.com/rss/news.rss"),
]

_ENTRIES_PER_FEED = 20
"""How many recent items to pull from each feed before ranking."""

_FETCH_TIMEOUT_SECONDS = 8
_CACHE_TTL_SECONDS = 600
_USER_AGENT = "Mozilla/5.0 (compatible; FinAgentBot/1.0)"

_HTML_TAG_RE = re.compile(r"<[^>]+>")

_feed_cache: dict[str, tuple[float, list[DataSnippet]]] = {}
"""feed_url -> (fetched_at_monotonic, snippets). Process-local, not persisted."""


def _strip_html(text: str) -> str:
    return unescape(_HTML_TAG_RE.sub("", text)).strip()


def _parse_published_at(entry) -> str:
    parsed = entry.get("published_parsed")
    if parsed is not None:
        return time.strftime("%Y-%m-%d", parsed)
    return entry.get("published", "")


def _fetch_one_feed(name: str, url: str) -> list[DataSnippet]:
    cached = _feed_cache.get(url)
    if cached is not None:
        fetched_at, snippets = cached
        if time.monotonic() - fetched_at < _CACHE_TTL_SECONDS:
            return snippets

    try:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_SECONDS) as response:
            raw = response.read()
        parsed_feed = feedparser.parse(raw)
        feed_source = parsed_feed.feed.get("title") or name

        snippets: list[DataSnippet] = []
        for entry in parsed_feed.entries[:_ENTRIES_PER_FEED]:
            link = entry.get("link", "")
            entry_id = entry.get("id") or link
            if not entry_id:
                continue
            snippets.append(
                DataSnippet(
                    id=entry_id,
                    source=feed_source,
                    title=_strip_html(entry.get("title", "")),
                    text=_strip_html(entry.get("summary", "")),
                    published_at=_parse_published_at(entry),
                    url=link,
                )
            )
    except Exception as exc:
        logger.warning("RSS fetch failed for %s (%s): %s", name, url, exc)
        return []

    _feed_cache[url] = (time.monotonic(), snippets)
    return snippets


def fetch(query: str, limit: int = 3) -> list[DataSnippet]:
    """Fetch recent articles across all configured feeds and rank by relevance to `query`.

    Same keyword-overlap ranking as `stub_source.fetch` - see that module's
    docstring. Returns `[]` if every feed fails or nothing overlaps the query.
    """
    query_terms = set(query.lower().split())
    all_snippets: list[DataSnippet] = []
    for name, url in _FEEDS:
        all_snippets.extend(_fetch_one_feed(name, url))

    seen_ids: set[str] = set()
    deduped: list[DataSnippet] = []
    for snippet in all_snippets:
        if snippet.id in seen_ids:
            continue
        seen_ids.add(snippet.id)
        deduped.append(snippet)

    def score(snippet: DataSnippet) -> int:
        text = f"{snippet.title} {snippet.text}".lower()
        return sum(1 for term in query_terms if term in text)

    ranked = sorted(deduped, key=score, reverse=True)
    relevant = [s for s in ranked if score(s) > 0]
    return relevant[:limit]
