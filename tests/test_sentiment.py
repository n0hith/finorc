"""Isolated check that the FinBERT sentiment classifier produces sane labels
on clearly positive, negative, and neutral financial text - no Gemini calls,
no RSS fetch required.

Mirrors tests/test_groundedness.py's style: a plain runnable script with
assertions (Phase 6's eval harness / CI will formalize this later).

Run with: python tests/test_sentiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finagent.analysis import finbert


def main() -> None:
    cases = [
        ("Nvidia raised its revenue guidance, citing strong demand for AI chips.", "positive"),
        ("The company missed earnings estimates and cut its full-year outlook.", "negative"),
        ("The committee will meet next Tuesday to discuss the agenda.", "neutral"),
    ]

    print("Loading FinBERT (first run downloads the model from HuggingFace)...")
    for text, expected_label in cases:
        result = finbert.classify(text)
        status = "ok" if result.label == expected_label else "MISMATCH"
        print(f"  [{status:8s}] expected={expected_label:8s} got={result.label:8s} (score={result.score:.2f})  {text}")
        assert result.label == expected_label, f"Expected {expected_label!r} for {text!r}, got {result.label!r}"

    print("\nAll assertions passed: positive, negative, and neutral text classified correctly.")


if __name__ == "__main__":
    main()
