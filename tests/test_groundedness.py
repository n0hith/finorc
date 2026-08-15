"""Isolated check that the NLI groundedness verifier actually catches an
unsupported claim, using Phase 1's stubbed data - no Gemini calls, no live
orchestration run required.

No test framework is wired up yet (that's Phase 6's eval harness / CI), so
this is a plain runnable script with assertions, consistent with the rest of
the project at this stage.

Run with: python tests/test_groundedness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finagent.data import stub_source
from finagent.orchestration.context import AnalystOutput, Claim
from finagent.verification.groundedness import verify


def main() -> None:
    snippets = stub_source.fetch("AI data center demand")
    snippet_by_id = {s.id: s for s in snippets}

    grounded_claim = Claim(
        text=snippet_by_id["stub-1"].text,
        supporting_snippet_ids=["stub-1"],
    )
    hallucinated_claim = Claim(
        text="Nvidia announced a $50 billion stock buyback program this quarter.",
        supporting_snippet_ids=["stub-1"],
    )
    uncited_claim = Claim(
        text="TSMC will build a new fab in Arizona by 2027.",
        supporting_snippet_ids=[],
    )

    output = AnalystOutput(
        task_id="test_task",
        summary="Test output mixing a grounded claim with an unsupported one.",
        claims=[grounded_claim, hallucinated_claim, uncited_claim],
        snippets_used=snippets,
    )

    print("Loading NLI model (first run downloads ~370MB from HuggingFace)...")
    report = verify(output)

    print("\nResults:")
    for v in report.claim_verifications:
        status = "FLAGGED" if v.flagged else "ok"
        print(f"  [{status:7s}] {v.verdict:12s} (score={v.score:.2f})  {v.claim_text[:70]}")

    grounded_result = report.claim_verifications[0]
    hallucinated_result = report.claim_verifications[1]
    uncited_result = report.claim_verifications[2]

    assert not grounded_result.flagged, "Expected the verbatim-source claim to be entailed, not flagged"
    assert grounded_result.verdict == "entailment"

    assert hallucinated_result.flagged, "Expected the fabricated claim to be flagged"
    assert hallucinated_result.verdict != "entailment"

    assert uncited_result.flagged, "Expected the uncited claim to be flagged"
    assert uncited_result.verdict == "uncited"

    print("\nAll assertions passed: grounded claim accepted, hallucinated and uncited claims flagged.")


if __name__ == "__main__":
    main()
