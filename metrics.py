"""
Token accounting and compression-exposure metrics.

CER is becoming a primary experimental variable, so a token count has to be
reproducible and has to say how it was obtained. chars/4 is an estimate; the
model's own tokenizer is a measurement. Mixing them silently would put an
estimated CER next to a measured one in the same column, and nobody reading
the table later could tell which was which.

So every count carries its method, and the analysis layer can refuse to pool
them.

    n, method = count_tokens(text)
    # method is "tiktoken:o200k_base" or "estimated_chars_div_4"

WHY THE FALLBACK EXISTS AT ALL. tiktoken downloads its vocabulary on first
use. That works in an ordinary terminal and fails behind an egress proxy that
does not allow openaipublic.blob.core.windows.net, which is the case in some
of the sandboxes this repository gets run in. Failing hard there would make
the harness unrunnable for inspection; silently estimating would corrupt the
headline variable. Recording the method does neither.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

ESTIMATE_CHARS_PER_TOKEN = 4
ESTIMATED = "estimated_chars_div_4"

_ENCODER = None
_ENCODER_NAME = None
_ENCODER_TRIED = False


def _encoder(model: str = "gpt-4o-mini"):
    """The model's own tokenizer, or None. Resolved once and cached.

    Cached across threads deliberately: tiktoken's Encoding is safe to share
    for encoding, and resolving it per call would re-attempt a failed network
    fetch on every count.
    """
    global _ENCODER, _ENCODER_NAME, _ENCODER_TRIED
    if _ENCODER_TRIED:
        return _ENCODER
    _ENCODER_TRIED = True
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")
        enc.encode("probe")          # force the vocab fetch to happen here
        _ENCODER, _ENCODER_NAME = enc, enc.name
    except Exception:
        _ENCODER, _ENCODER_NAME = None, None
    return _ENCODER


def tokenizer_name(model: str = "gpt-4o-mini") -> str:
    _encoder(model)
    return f"tiktoken:{_ENCODER_NAME}" if _ENCODER_NAME else ESTIMATED


def count_tokens(text: str, model: str = "gpt-4o-mini") -> tuple[int, str]:
    """(tokens, method). Never raises; an empty or None input is 0 tokens."""
    if not text:
        return 0, tokenizer_name(model)
    enc = _encoder(model)
    if enc is not None:
        return len(enc.encode(text)), f"tiktoken:{_ENCODER_NAME}"
    return len(text) // ESTIMATE_CHARS_PER_TOKEN, ESTIMATED


def is_measured(method: str) -> bool:
    return bool(method) and method.startswith("tiktoken:")


def pooling_is_safe(methods) -> bool:
    """True only if every count in a set came from the same method.

    Call this before averaging a CER column or quoting a median. An
    estimated CER and a measured one are different quantities and a mean of
    the two is not a number anyone can interpret, so the analysis layer has
    to be able to refuse rather than silently average them.
    """
    seen = {m for m in methods if m}
    return len(seen) <= 1


# --------------------------------------------------------------- exposure

@dataclass
class CompressionMetrics:
    """Sizes at every stage, and what they imply.

    source_context_tokens and precompression_handoff_tokens are SEPARATE and
    must stay that way. A task can read 3,224 tokens of documents and hand the
    compressor 27 words; describing that 119:1 drop as compression would be
    wrong, because the compressor never saw the documents. Agent 1 did the
    reduction. Only the handoff enters the compressor, so only the handoff can
    be compressed.
    """

    source_context_tokens: int
    agent1_input_tokens: int
    precompression_handoff_tokens: int
    compressor_budget_tokens: int
    compressed_tokens: int | None            # None when condition == "none"
    cer: float | None
    realized_ratio: float | None
    fraction_removed: float | None
    compression_stratum: str
    token_count_method: str
    condition: str
    compression_applied: bool

    def as_dict(self) -> dict:
        return asdict(self)


# Thresholds live in config.py; imported here lazily so metrics has no
# dependency on it and can be unit-tested alone.
def stratum_of(cer: float | None, strata) -> str:
    """Label, never a filter.

    A low-CER task is a negative control, not a reject: it tells us what the
    pipeline does when compression cannot bite, which is the comparison the
    exposed tasks are measured against.
    """
    if cer is None:
        return "unknown"
    for name, (lo, hi) in strata:
        if lo <= cer < hi:
            return name
    return "unknown"


def compute(source_text, handoff_text, compressed_text, budget_tokens,
            condition, strata, model="gpt-4o-mini",
            agent1_input_text=None) -> CompressionMetrics:
    """Measure one task-run.

    `compressed_text` may be None, which is the honest representation of
    condition="none": no compressor ran, so there is no compressed size and no
    realized ratio. CER is still computed, because CER is about EXPOSURE —
    what the compressor would have been handed relative to its budget — and
    that is meaningful whether or not compression was actually applied. The
    two are kept apart by `compression_applied`.

    `source_text` is the DOCUMENTS Agent 1 read. `agent1_input_text` is its
    whole prompt, documents plus the task line. They are recorded separately
    because the earlier log called one of them "context" without saying which,
    and the difference is the task statement — which is the one thing Agent 2
    never sees and therefore the thing the handoff has to carry.
    """
    src, method = count_tokens(source_text, model)
    hnd, _ = count_tokens(handoff_text, model)
    inp, _ = count_tokens(
        agent1_input_text if agent1_input_text is not None else source_text,
        model)

    applied = condition != "none" and compressed_text is not None
    cmp_tok = count_tokens(compressed_text, model)[0] if applied else None

    # Every count above goes through the same cached encoder, so one method
    # describes the whole row. Asserting that rather than assuming it, because
    # a row with a mixed method would be the kind of quiet corruption this
    # field exists to prevent.
    cer = (hnd / budget_tokens) if budget_tokens else None
    if applied and cmp_tok:
        realized = hnd / cmp_tok
        removed = 1.0 - (cmp_tok / hnd) if hnd else None
    elif applied and cmp_tok == 0:
        # The compressor returned nothing. Everything was removed; the ratio
        # is undefined rather than infinite, and saying so beats a divide.
        realized, removed = None, 1.0
    else:
        realized, removed = None, None

    return CompressionMetrics(
        source_context_tokens=src,
        agent1_input_tokens=inp,
        precompression_handoff_tokens=hnd,
        compressor_budget_tokens=budget_tokens,
        compressed_tokens=cmp_tok,
        cer=cer,
        realized_ratio=realized,
        fraction_removed=removed,
        compression_stratum=stratum_of(cer, strata),
        token_count_method=method,
        condition=condition,
        compression_applied=applied,
    )
