"""
Raw traces, written as JSONL, one object per line.

Five rounds of this project were spent diagnosing from inference because the
harness threw the models' output away: a refusal, unparseable JSON, a call to
a tool the agent did not hold, and genuine silence all look identical once
_replay has finished. Four wrong fixes came out of that. Aggregate scores are
not evidence.

Two streams, because they answer different questions:

  calls.jsonl   one line per MODEL CALL. Was it requested, did it succeed,
                what came back verbatim, did it parse, how many retries. This
                is what distinguishes an API failure from a model failure.

  runs.jsonl    one line per TASK RUN. Sizes at every stage, CER, stratum,
                scoring, and a reference to the calls it made.

Append-only and flushed per line, so a run killed halfway still leaves usable
evidence on disk — the interrupted out-of-credit run lost everything.
"""

from __future__ import annotations

import json
import os
import threading
import time


class TraceWriter:
    """Thread-safe append-only JSONL. One lock, held only around the write."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._files = {}

    def _fh(self, name: str):
        fh = self._files.get(name)
        if fh is None:
            fh = open(os.path.join(self.run_dir, f"{name}.jsonl"), "a")
            self._files[name] = fh
        return fh

    def write(self, stream: str, record: dict) -> None:
        line = json.dumps(record, default=str)
        with self._lock:
            fh = self._fh(stream)
            fh.write(line + "\n")
            fh.flush()          # a killed run must still leave evidence

    def close(self) -> None:
        with self._lock:
            for fh in self._files.values():
                try:
                    fh.close()
                except Exception:
                    pass
            self._files.clear()


def call_record(*, run_id, task_id, stage, model, temperature, system, user,
                raw, parsed=None, parse_error=None, api_error=None,
                retries=0, started, finished, prompt_tokens=None,
                output_tokens=None, token_count_method=None) -> dict:
    """One model call.

    `requested` is always true: the record exists because a call was made.
    `succeeded` is about the TRANSPORT, not the content — a model that
    refuses in prose succeeded. Keeping those apart is what stops an
    out-of-credit run being read as the model getting worse.
    """
    return {
        "kind": "call",
        "run_id": run_id,
        "task_id": task_id,
        "stage": stage,                     # agent1 | compressor | agent2
        "model": model,
        "temperature": temperature,
        "requested": True,
        "succeeded": api_error is None,
        "api_error": api_error,
        "retries": retries,
        "parse_status": ("ok" if parse_error is None and raw is not None
                         else ("unparsed" if raw is not None else "no_output")),
        "parse_error": parse_error,
        "system_prompt": system,
        "user_input": user,
        "raw_output": raw,
        "parsed_output": parsed,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "token_count_method": token_count_method,
        "started_at": started,
        "finished_at": finished,
        "duration_s": (finished - started) if (started and finished) else None,
    }


def run_record(*, cfg, task, repeat, metrics, scoring, failure_stage=None,
               api_failure=False, parse_failure=False, n_calls=0,
               calls=None, extra=None) -> dict:
    """One task run: everything needed to put a row in the analysis table."""
    rec = {
        "kind": "run",
        "run_id": cfg.run_id,
        "repeat": repeat,
        "task_id": task.get("id"),
        "suite": task.get("domain"),
        "experiment_mode": cfg.experiment_mode,
        "handoff_regime": cfg.handoff_regime,
        "condition": cfg.condition,
        "model": cfg.model,
        "temperature": cfg.temperature,
        "git_commit": cfg.git_commit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "api_failure": api_failure,
        "parse_failure": parse_failure,
        "n_tool_calls": n_calls,
        "tool_calls": calls or [],
        "failure_stage": failure_stage,
    }
    rec.update(metrics.as_dict() if hasattr(metrics, "as_dict") else metrics)
    rec.update(scoring or {})
    rec.update(extra or {})
    return rec
