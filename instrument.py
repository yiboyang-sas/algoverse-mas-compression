"""
Per-run state, and a backend wrapper that records what actually happened.

Two problems this fixes, both of which have already cost real runs.

THE FIRST IS SHARED MUTABLE STATE. `pipeline.AGENT2_SYSTEM` is a module
global that the runner used to set once per task. In one thread that is
correct. In eight, task A writes it, task B overwrites it, and task A's
Agent 2 is then told it holds task B's tools — workspace tasks came back
calling `post_webpage`. It invalidated every parallel run, and it was
invisible because the scores it produced were plausible. Anything that
varies per task now lives in a TaskRunContext held in a thread-local, so
two workers cannot see each other's values.

THE SECOND IS THAT THE HARNESS THREW AWAY THE EVIDENCE. A refusal,
unparseable JSON, a call to a tool the agent does not hold, a timeout and
genuine silence all look identical once the run is scored. Five rounds of
this project were spent diagnosing from inference and four of the five
diagnoses were wrong. Every model call now writes a line to calls.jsonl
with the verbatim output, whether the transport succeeded, how many retries
it took and whether the result parsed.

The distinction that matters most in that record: `succeeded` is about the
TRANSPORT. A model that refuses in prose succeeded. An out-of-credit run
scored 45%, 59% and 73% across three identical repeats purely because 18,
then 13, then 0 calls were refused, and it was read as a baseline. Keeping
transport apart from content is what stops that happening again.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

# --------------------------------------------------------------- stages
#
# Agent 1, the compressor and Agent 2 all go through the same backend
# object, so a call record has to say which stage it came from. The
# compressor sets its stage explicitly (see compressors._prompted), because
# both it and Agent 2 would otherwise fall into the same default branch.
# Agent 1 and Agent 2 are identified from their system prompt, which is a
# prose assumption and therefore a tested one: tests/test_instrument.py
# asserts every Agent 1 regime still starts with the string below. If
# someone rewrites a prompt, that test fails instead of the traces quietly
# mislabelling half a run.

AGENT1_PREFIX = "You are Agent 1"
AGENT2_PREFIX = "You are Agent 2"

STAGE_AGENT1 = "agent1"
STAGE_AGENT2 = "agent2"
STAGE_COMPRESSOR = "compressor"


def stage_from_system(system: str) -> str:
    s = system or ""
    if s.startswith(AGENT1_PREFIX):
        return STAGE_AGENT1
    if s.startswith(AGENT2_PREFIX):
        return STAGE_AGENT2
    return STAGE_COMPRESSOR


# ------------------------------------------------------------- the context

@dataclass
class TaskRunContext:
    """Everything that varies per task-run, in one object, per thread.

    `budget_tokens` is here rather than read from compressors.BUDGET_TOKENS
    because that module global is read once at import and baked into the
    prompt strings, which makes a budget sweep inside one process impossible
    and makes the number in a result unverifiable. CER divides by this, so it
    has to be the value the run actually used.
    """

    run_id: str
    task_id: str
    repeat: int = 0
    budget_tokens: int = 150
    condition: str = "none"
    model: str | None = None
    temperature: float | None = None
    agent2_prompt: str | None = None
    stage_override: str | None = None
    calls: list = field(default_factory=list)
    parse_events: list = field(default_factory=list)
    trace: object | None = None

    # set by the retry wrapper so the trace can tell a slow network apart
    # from a model that answered first time
    retries: int = 0


_local = threading.local()


def current() -> TaskRunContext | None:
    return getattr(_local, "ctx", None)


def current_budget(default: int) -> int:
    ctx = current()
    return ctx.budget_tokens if ctx is not None else default


@contextmanager
def run_context(ctx: TaskRunContext):
    """Install a context for the duration of one task-run on this thread.

    Nested runs are not a thing here, but the previous value is restored
    anyway: swallowing it would make a future nesting bug silent, which is
    the category of bug this module exists to stop.
    """
    prev = getattr(_local, "ctx", None)
    _local.ctx = ctx
    try:
        yield ctx
    finally:
        _local.ctx = prev


@contextmanager
def stage(name: str):
    """Label the calls made inside this block explicitly."""
    ctx = current()
    if ctx is None:
        yield
        return
    prev = ctx.stage_override
    ctx.stage_override = name
    try:
        yield
    finally:
        ctx.stage_override = prev


def note_retry(n: int = 1) -> None:
    ctx = current()
    if ctx is not None:
        ctx.retries += n


# ------------------------------------------------------- traced backend

_INSTALLED = "_mas_traced"


def traced(backend):
    """Wrap `backend.chat` once so every call writes a trace line.

    INSTALLED ONCE, DELIBERATELY. The first version of the raw capture set
    `backend.chat` inside each run. With eight workers they overwrote each
    other and four different tasks' records all came back holding bk_5's
    output. The wrapper goes on once and every piece of per-call state it
    touches lives in the context, which is thread-local.

    This wrapper sits OUTSIDE the retry wrapper, so what it records is the
    outcome after retries. The attempt count comes from note_retry, which
    the retry wrapper calls.
    """
    if getattr(backend, _INSTALLED, False):
        return backend

    inner = backend.chat

    def chat(system, user):
        ctx = current()
        started = time.time()
        before = ctx.retries if ctx is not None else 0
        try:
            out = inner(system, user)
        except Exception as e:
            if ctx is not None:
                _record(ctx, system, user, None, started,
                        api_error=f"{type(e).__name__}: {e}",
                        retries=(ctx.retries - before))
            raise
        if ctx is not None:
            _record(ctx, system, user, out, started,
                    retries=(ctx.retries - before))
        return out

    backend.chat = chat
    setattr(backend, _INSTALLED, True)
    return backend


def _record(ctx, system, user, raw, started, api_error=None, retries=0):
    import metrics
    from tracing import call_record

    stg = ctx.stage_override or stage_from_system(system)
    # Sizes on the call itself, not just on the run. A handoff that came back
    # short is one thing; a handoff that came back short because Agent 1 was
    # handed 12,000 tokens and ran out of room is another, and the run record
    # alone cannot tell them apart.
    pt, method = metrics.count_tokens((system or "") + "\n" + (user or ""),
                                      ctx.model or "gpt-4o-mini")
    ot, _ = metrics.count_tokens(raw or "", ctx.model or "gpt-4o-mini")
    rec = call_record(
        run_id=ctx.run_id, task_id=ctx.task_id, stage=stg,
        model=ctx.model, temperature=ctx.temperature,
        system=system, user=user, raw=raw,
        api_error=api_error, retries=retries,
        started=started, finished=time.time(),
        prompt_tokens=pt, output_tokens=ot, token_count_method=method,
    )
    rec["repeat"] = ctx.repeat
    ctx.calls.append(rec)
    if ctx.trace is not None:
        ctx.trace.write("calls", rec)


# -------------------------------------------------------------- the audit
#
# Written down here rather than in a commit message because the next person
# to add a worker pool needs to see it. Every module-level name that gets
# ASSIGNED after import, and what makes it safe or not:
#
#   pipeline.AGENT2_SYSTEM      written per task. Now holds a _PerThreadSystem
#                               whose .format() answers from a thread-local,
#                               so the object is shared and the VALUE is not.
#                               Still a global write in a concurrent path;
#                               tests/test_concurrency.py is what keeps it
#                               honest.
#   pipeline.AGENT1_SYSTEM      written once at import from MAS_AGENT1. Does
#                               not vary per task. Safe.
#   pipeline.{run_pipeline,_parse,_replay,_build_agent1_input,score_run,
#             TOOL_REGISTRY}    all written once at import by baseline.py's
#                               patches. Order is load-bearing but there is
#                               no concurrency at import. Safe.
#   compressors.BUDGET_TOKENS   read at import and baked into four prompt
#                               strings. Not written per task, so not a race,
#                               but it made the budget unsweepable in one
#                               process and unverifiable in a result. Now a
#                               parameter, defaulting through this context.
#   baseline._inner_chat        set once by the capture installer, guarded by
#                               a flag. Replaced by `traced` above.
#   baseline._a2.prompt         thread-local. Correct, and now tested.
#   baseline._cap.buf           thread-local. Replaced by ctx.calls.
#   models.OpenAIBackend        holds no state between calls, so one instance
#                               is shared across workers. Verified by reading
#                               chat(): it builds its own request and returns
#                               the response.
#   metrics._ENCODER            resolved once and cached across threads on
#                               purpose. tiktoken's Encoding is safe to share
#                               for encoding, and re-resolving would re-try a
#                               failed network fetch on every count.

SHARED_STATE_AUDIT = {
    "per_task_globals": ["pipeline.AGENT2_SYSTEM"],
    "import_time_globals": [
        "pipeline.AGENT1_SYSTEM", "pipeline.run_pipeline", "pipeline._parse",
        "pipeline._replay", "pipeline._build_agent1_input", "pipeline.score_run",
        "pipeline.TOOL_REGISTRY", "compressors.BUDGET_TOKENS",
    ],
    "thread_locals": ["baseline._a2.prompt", "instrument._local.ctx"],
    "shared_read_only": ["models.OpenAIBackend", "metrics._ENCODER"],
}
