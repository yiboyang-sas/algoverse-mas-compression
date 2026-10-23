"""Passive stage tracing. Tokenized text lengths are distinct from billed usage."""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import tempfile
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache


@lru_cache(maxsize=8)
def _encoding(model):
    try:
        import tiktoken
        name = "o200k_base" if model == "mock" else tiktoken.model.encoding_name_for_model(model)
        # Preflight and offline fixtures must never initiate a tokenizer
        # download. These public-data hashes are from pinned tiktoken 0.12.0.
        known = {
            "o200k_base": "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
            "cl100k_base": "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        }
        if name not in known:
            return None
        url = f"https://openaipublic.blob.core.windows.net/encodings/{name}.tiktoken"
        cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR", os.environ.get(
            "DATA_GYM_CACHE_DIR", os.path.join(tempfile.gettempdir(), "data-gym-cache")))
        cache = Path(cache_dir) / hashlib.sha1(url.encode()).hexdigest()
        if not cache_dir or not cache.is_file() or hashlib.sha256(cache.read_bytes()).hexdigest() != known[name]:
            return None
        return tiktoken.get_encoding(name)
    except Exception:
        # Offline installations may lack either the library or its BPE cache.
        # Do not silently relabel character estimates as token measurements.
        return None


def text_size(text: str, model: str) -> dict:
    enc = _encoding(model)
    return {"characters": len(text), "utf8_bytes": len(text.encode("utf-8")),
            "tokens": len(enc.encode(text, disallowed_special=())) if enc else None,
            "tokenizer": enc.name if enc else "unavailable"}


class PipelineExecutionError(RuntimeError):
    """An infrastructure error, with the completed portion of a trial attached."""
    def __init__(self, stage, cause, partial_trial):
        super().__init__(f"{stage}: {cause}")
        self.stage = stage
        self.partial_trial = partial_trial


class TracedBackend:
    def __init__(self, backend, stage, requests):
        self.backend, self.stage, self.requests = backend, stage, requests

    def chat(self, system, user):
        model = getattr(self.backend, "model", self.backend.name)
        request = {"stage": self.stage, "backend": self.backend.name,
                   "request_id": uuid.uuid4().hex,
                   "started_at": datetime.now(timezone.utc).isoformat(),
                   "model": model, "system": system, "user": user,
                   "input_text": {"system": text_size(system, model),
                                  "user": text_size(user, model)},
                   "status": "started"}
        self.requests.append(request)
        start = time.monotonic()
        try:
            response = self.backend.chat(system, user)
            if not isinstance(response, str):
                raise ValueError("backend returned a non-text response")
            request.update(status="completed", response=response,
                           output_text=text_size(response, model))
            return response
        except (Exception, SystemExit, KeyboardInterrupt) as exc:
            request.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "infrastructure_error",
                           error_type=type(exc).__name__)
            raise
        finally:
            request["elapsed_seconds"] = time.monotonic() - start
            request["finished_at"] = datetime.now(timezone.utc).isoformat()
            request["provider"] = copy.deepcopy(
                getattr(self.backend, "last_response_metadata", {}))


def usage_totals(requests):
    """Null, rather than zero, when provider usage is absent for any request."""
    totals = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [(r.get("provider", {}).get("usage") or {}).get(key) for r in requests]
        totals[key] = sum(values) if values and all(isinstance(v, int) for v in values) else None
    return totals
