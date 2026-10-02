"""Local FinBERT sentiment classifier.

Loads a pretrained 3-way sentiment model (positive / negative / neutral) fine-
tuned on financial text and exposes a single `classify(text) -> SentimentScore`
call. Deliberately mirrors `verification/nli.py`'s shape: a standalone,
non-LLM, locally-run classifier loaded once per process and reused, so
sentiment scoring doesn't cost a Gemini call per snippet and is independently
testable.

Model choice: ProsusAI/finbert - a BERT-base model fine-tuned specifically for
sentiment on financial news/text (as opposed to a general-purpose sentiment
model), which matters because financial language ("guidance cut", "beat
estimates") doesn't carry the same sentiment polarity in everyday text.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from finagent.orchestration.context import SentimentScore

_MODEL_NAME = "ProsusAI/finbert"


@lru_cache(maxsize=1)
def _load_model() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
    model.eval()
    return tokenizer, model


def classify(text: str) -> SentimentScore:
    """Classify the financial sentiment of `text` as positive, negative, or neutral."""
    tokenizer, model = _load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    best_idx = int(torch.argmax(probs))
    label = model.config.id2label[best_idx].lower()
    return SentimentScore(label=label, score=float(probs[best_idx]))
