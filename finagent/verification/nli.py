"""Local NLI (Natural Language Inference) model wrapper.

Loads a pretrained 3-way NLI classifier (entailment / neutral / contradiction)
from HuggingFace and exposes a single `classify(premise, hypothesis)` call.
This is deliberately a standalone, non-LLM component: given a piece of source
text (premise) and a claim (hypothesis), it answers "does the premise support
the hypothesis?" without an API call, a prompt, or any dependency on the
Gemini agents. That's what makes it usable as an independent groundedness
check rather than just another LLM asking another LLM if it did a good job.

Model choice: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli - a mid-size
(~370MB) DeBERTa-v3 model fine-tuned specifically for NLI across MNLI, FEVER,
and ANLI, which is a common choice for exactly this "is this claim entailed
by this source text" use case. The model is loaded once (module-level cache)
and reused across calls - the first call in a process pays the load cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


@dataclass
class NLIResult:
    label: str
    """One of 'entailment', 'neutral', 'contradiction'."""
    score: float
    """Softmax probability of the predicted label, in [0, 1]."""


@lru_cache(maxsize=1)
def _load_model() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
    model.eval()
    return tokenizer, model


def classify(premise: str, hypothesis: str) -> NLIResult:
    """Classify whether `premise` entails, contradicts, or is neutral toward `hypothesis`."""
    tokenizer, model = _load_model()
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    best_idx = int(torch.argmax(probs))
    label = model.config.id2label[best_idx].lower()
    return NLIResult(label=label, score=float(probs[best_idx]))
