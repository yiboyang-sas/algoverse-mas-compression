"""
Model backends for the pipeline.

Every agent and compressor in this project talks to a model through one function:
`chat(system, user)` -> str. That is the ONLY thing a backend has to provide.

Three backends ship here:
  - MockBackend   : no network, no key. Fakes plausible agent/compressor output
                    so the whole loop runs end-to-end for testing.
  - OpenAIBackend : any OpenAI-compatible API (OpenAI, Together, Groq, a local
                    vLLM/Ollama server exposing /v1). Set base_url + api_key.
  - OllamaBackend : a local Ollama server (http://localhost:11434), no key.

Switch backend in ONE place: see config at the bottom of run.py.
"""

from __future__ import annotations
import json
import time
import os
import re
import urllib.request
import urllib.error

from budget import BudgetError, prompt_token_bound


class Backend:
    """A backend just needs chat(system, user) -> str."""
    name = "base"

    def chat(self, system: str, user: str) -> str:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Mock backend: runs the whole pipeline with no model at all.
# It is deliberately dumb. It does NOT try to be a real agent. It exists so
# you can run the loop, see the JSON come out, and check the scoring works
# before you spend money or GPU on a real model.
# --------------------------------------------------------------------------
class MockBackend(Backend):
    name = "mock"

    def __init__(self, follows_injection: bool = True):
        # follows_injection: whether the mock "agent" obeys a hidden instruction
        # it finds in the documents. Toggle to sanity-check both score paths.
        self.follows_injection = follows_injection

    def chat(self, system: str, user: str) -> str:
        role = system.lower()

        if "compress" in role or "summar" in role:
            return self._mock_compress(system, user)

        # Otherwise treat it as an agent that must emit tool calls.
        return self._mock_agent(system, user)

    def _mock_agent(self, system: str, user: str) -> str:
        is_agent2 = system.lower().startswith("you are agent 2")

        if not is_agent2:
            # AGENT 1: make the task-required calls it is hinted to make, and
            # forward BOTH the next agent's required call AND (if it follows
            # injections) the planted instruction into its handoff note as text.
            # It does not fire the harmful call itself -- it lacks the tool.
            calls = []
            for m in re.finditer(r"REQUIRED_CALL:\s*(\w+)\((.*?)\)", user):
                calls.append(_mk_call(m.group(1), m.group(2)))

            note = "Completed my part. Results ready for Agent 2 to finish."
            m = re.search(r"NEXT_CALL:\s*(\w+)\((.*?)\)", user)
            if m:
                note += f" Next step: NEXT_CALL: {m.group(1)}({m.group(2)})."
            if self.follows_injection:
                m = re.search(r"INJECT_CALL:\s*(\w+)\((.*?)\)", user)
                if m:
                    note += (f" Also, as part of finishing up, "
                             f"INJECT_CALL: {m.group(1)}({m.group(2)}).")
            return json.dumps({"tool_calls": calls, "handoff_note": note})

        # AGENT 2: acts on the handoff it received. Fires the next-step call and
        # the harmful call ONLY if each survived into the text it can see (i.e.
        # survived Agent 1's note AND compression).
        calls = []
        for m in re.finditer(r"NEXT_CALL:\s*(\w+)\((.*?)\)", user):
            calls.append(_mk_call(m.group(1), m.group(2)))
        if self.follows_injection:
            m = re.search(r"INJECT_CALL:\s*(\w+)\((.*?)\)", user)
            if m:
                calls.append(_mk_call(m.group(1), m.group(2)))
        return json.dumps({"tool_calls": calls, "handoff_note": "done"})

    def _mock_compress(self, system: str, user: str) -> str:
        # The mock compressor simulates each condition's effect on the markers:
        #   allowlist        -> drops INJECT_CALL whose tool is not allowed
        #   instruction_strip-> (mock) leaves markers; a real model would reword
        #   others           -> keep call-bearing lines, drop filler
        text = user
        sysl = system.lower()

        if "only use these tools" in sysl or "may only use" in sysl or "allowlist" in sysl:
            allowed = self._allowed_tools_from(system)
            def drop(m):
                tool = m.group(1)
                return "" if tool not in allowed else m.group(0)
            text = re.sub(r"INJECT_CALL:\s*(\w+)\(.*?\)", drop, text)

        kept = [ln for ln in text.splitlines()
                if "CALL:" in ln or "task" in ln.lower() or "result" in ln.lower()]
        return "\n".join(kept) if kept else text[:400]

    @staticmethod
    def _allowed_tools_from(system: str) -> set:
        # pull the tool names the allowlist/destination prompt lists
        m = re.search(r"tools?(?:\s*(?:are|:))?\s*([a-z_,\s]+)", system, re.I)
        if not m:
            return set()
        return {t.strip() for t in re.split(r"[,\s]+", m.group(1)) if t.strip()}


def _mk_call(tool: str, args_str: str) -> dict:
    args = {}
    for part in args_str.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            args[k.strip()] = v.strip().strip('"\'')
    return {"tool": tool.strip(), "args": args}


# --------------------------------------------------------------------------
# OpenAI-compatible backend. Works with OpenAI, Together, Groq, OpenRouter,
# and any local server (vLLM, LM Studio, Ollama's /v1 shim) that speaks the
# /chat/completions format.
# --------------------------------------------------------------------------
def _retry_after(err, detail: str):
    """Seconds to wait from a 429: the Retry-After header, or the
    'try again in 1.2s' / '1m30s' / '350ms' text in the body. Capped at 90s."""
    secs = None
    try:
        h = err.headers.get("retry-after") if err.headers else None
        if h:
            secs = float(h)
    except (TypeError, ValueError):
        pass
    if secs is None:
        m = re.search(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)", detail)
        if m:
            secs = int(m.group(1) or 0) * 60 + float(m.group(2)) / (1000 if m.group(3) == "ms" else 1)
    if secs is None:
        return None
    return min(secs + 1, 90)


def _request_with_retry(req, tries: int = 8, timeout: int = 120,
                        audit: list | None = None,
                        before_attempt=None, finish_attempt=None) -> dict:
    """Send a request, retrying transient failures.

    Without this, one slow response raises straight out of run_pipeline and
    the whole sweep dies, losing every task that already completed. An 18-task
    run is several minutes of API calls; a single timeout should not cost all
    of it. Seen twice in practice on 18-task runs.
    """
    last = None
    for i in range(tries):
        attempt = {"attempt": i + 1}
        if audit is not None:
            audit.append(attempt)
        reservation = None
        if before_attempt is not None:
            try:
                reservation = before_attempt()
                attempt["budget_reservation_id"] = reservation
            except BudgetError:
                attempt["status"] = "budget_blocked"
                raise
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
                attempt.update(status="completed", http_status=getattr(resp, "status", 200))
                if finish_attempt is not None:
                    attempt["budget"] = finish_attempt(reservation, data.get("usage"), "completed")
                return data
        except urllib.error.HTTPError as e:
            attempt.update(status="http_error", http_status=e.code)
            if finish_attempt is not None:
                attempt["budget"] = finish_attempt(reservation, None, "http_error")
            # 429 and 5xx are usually transient; 400/401/404 will fail the
            # same way every time, so surface them immediately. OpenAI also
            # returns 429 for an account with no credits (insufficient_quota),
            # which retrying can never fix, so say that plainly instead.
            try:
                detail = e.read().decode("utf-8", "replace")
            except Exception:
                detail = ""
            if "insufficient_quota" in detail:
                raise SystemExit(
                    "OpenAI says this key has no credits (insufficient_quota). "
                    "Add credit under Settings > Billing on platform.openai.com, "
                    "wait a few minutes, then rerun.") from None
            if e.code != 429 and e.code < 500:
                raise SystemExit(f"API error {e.code}: {detail[:500]}") from None
            last = RuntimeError(f"API error {e.code} after retries: {detail[:500]}")
            # Per-minute limits (tight on gpt-4o at low tiers) clear on their
            # own. Wait as long as OpenAI says to, instead of 1-2-4 seconds.
            # A per-DAY limit won't clear in time, so stop and say so.
            if "per day" in detail:
                raise SystemExit("Hit the daily request limit for this model. "
                                 "Rerun later; finished runs are saved.") from None
            wait = _retry_after(e, detail)
            if wait is not None and i < tries - 1:
                print(f"    rate limited, waiting {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
        except BudgetError:
            # Budget denials or ledger failures must never become retries.
            raise
        except Exception as e:          # timeouts, connection resets
            attempt.update(status="transport_error", error_type=type(e).__name__)
            if finish_attempt is not None:
                attempt["budget"] = finish_attempt(reservation, None, "transport_error")
            last = e
        if i < tries - 1:
            time.sleep(2 ** i)
    raise last


class OpenAIBackend(Backend):
    name = "openai"

    def __init__(self, model: str, api_key: str | None = None,
                 base_url: str = "https://api.openai.com/v1",
                 temperature: float = 0.0):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.last_response_metadata = {}
        self.budget = None
        self.max_output_tokens = None

    def chat(self, system: str, user: str) -> str:
        self.last_response_metadata = {"http_attempts": []}
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.max_output_tokens is not None:
            payload["max_completion_tokens"] = self.max_output_tokens
        if self.budget is not None and self.max_output_tokens is None:
            raise BudgetError("budgeted requests require an explicit max_output_tokens")
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            hooks = {}
            if self.budget is not None:
                bound = prompt_token_bound(payload["messages"])
                hooks = {
                    "before_attempt": lambda: self.budget.before_request(bound, self.max_output_tokens),
                    "finish_attempt": lambda reservation, usage, status: self.budget.finalize(
                        reservation, usage=usage, status=status),
                }
            data = _request_with_retry(req, audit=self.last_response_metadata["http_attempts"], **hooks)
        except (Exception, SystemExit) as exc:
            # Persist useful categories without copying HTTP response bodies or
            # credentials into experiment error logs.
            message = str(exc).lower()
            self.last_response_metadata["error_category"] = (
                "budget" if isinstance(exc, BudgetError) else
                "insufficient_quota" if "insufficient_quota" in message else
                "authentication" if "401" in message else
                "rate_limit" if "limit" in message or "429" in message else
                "api_or_transport_error")
            raise
        self.last_response_metadata.update({
            "response_id": data.get("id"),
            "response_model": data.get("model"),
            "system_fingerprint": data.get("system_fingerprint"),
            "usage": data.get("usage"),
            "finish_reason": data["choices"][0].get("finish_reason"),
            "refusal": data["choices"][0]["message"].get("refusal"),
        })
        # Refusals and filtered completions are observed model behavior. Keep
        # them in the task denominator rather than retrying as network errors.
        content = data["choices"][0]["message"].get("content")
        return "" if content is None else content


# --------------------------------------------------------------------------
# Ollama native backend (http://localhost:11434). No key. Good for Colab with
# a small local model, or a laptop running e.g. qwen2.5:14b.
# --------------------------------------------------------------------------
class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 temperature: float = 0.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def chat(self, system: str, user: str) -> str:
        body = json.dumps({
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        return data["message"]["content"]
