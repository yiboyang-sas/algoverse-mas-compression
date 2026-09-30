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
import os
import re
import urllib.request
import urllib.error


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
class OpenAIBackend(Backend):
    name = "openai"

    def __init__(self, model: str, api_key: str | None = None,
                 base_url: str = "https://api.openai.com/v1",
                 temperature: float = 0.0):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def chat(self, system: str, user: str) -> str:
        body = json.dumps({
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]


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
