"""Durable per-HTTP-attempt spending reservations for explicitly approved runs.

Prices are supplied by the operator in USD per million tokens. They are never
inferred from a model alias. Before each HTTP attempt (including retries), reserve
the maximum estimated cost and fsync it. A missing/failed response keeps that
reservation because the provider may have billed it. Returned usage releases the
reservation only when both token counters are trustworthy nonnegative integers.

This enforces the configured accounting cap, conditional on correct prices and
provider token bounds. It cannot impose a billing cap on the provider's account.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import json
import os
from pathlib import Path
import uuid


MILLION = Decimal(1_000_000)


class BudgetError(RuntimeError):
    """Accounting/configuration failure: never retry it as a network error."""


class BudgetExceeded(BudgetError):
    """No HTTP attempt may start under the remaining configured budget."""


def _money(value, label, *, allow_zero=False):
    if isinstance(value, bool):
        raise BudgetError(f"{label} must be a finite USD amount")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise BudgetError(f"{label} must be a finite USD amount") from None
    if not number.is_finite() or number < 0 or (number == 0 and not allow_zero):
        raise BudgetError(f"{label} must be {'nonnegative' if allow_zero else 'positive'} and finite")
    return number


def _decimal_string(value):
    return format(value.normalize(), "f")


def _tokens(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0):
        raise BudgetError(f"{label} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def prompt_token_bound(messages):
    """Conservative GPT-4o text-message estimate, without a BPE download.

    UTF-8 bytes upper-bound ordinary byte-pair text tokens; counting the complete
    JSON also reserves role/key punctuation. Add 256 tokens per request and 64
    per message for chat framing. This is deliberately larger than a tokenizer
    estimate. It is only for this harness's text messages, without tool schemas,
    images, audio, hidden reasoning, or provider-added context. A returned usage
    counter above the bound stops further attempts for explicit review.
    """
    if not isinstance(messages, list) or not messages:
        raise BudgetError("messages must be a nonempty list of text messages")
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise BudgetError("each message must have a text role")
        if not isinstance(message.get("content"), str):
            raise BudgetError("budget bound supports text message content only")
    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return len(serialized) + 256 + 64 * len(messages)


def _now():
    return datetime.now(timezone.utc).isoformat()


class BudgetLedger:
    """Append-only accounting with an interprocess lock and immutable prices.

    Every before_request call authorizes exactly one network attempt. Never
    reuse its reservation for a retry. An optional request_id must be unique;
    duplicate IDs fail closed, including after process restarts. finalize is
    idempotent for the same status/usage and never releases unknown charges.
    """

    def __init__(self, path, *, cap_usd, input_per_million_usd,
                 output_per_million_usd):
        self.path = Path(path)
        self.cap = _money(cap_usd, "cap_usd", allow_zero=True)
        self.input_price = _money(input_per_million_usd, "input_per_million_usd")
        self.output_price = _money(output_per_million_usd, "output_per_million_usd")
        self.config = {
            "cap_usd": _decimal_string(self.cap),
            "input_per_million_usd": _decimal_string(self.input_price),
            "output_per_million_usd": _decimal_string(self.output_price),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        with self._locked():
            if not self.path.exists():
                self._append({"event": "configuration", "schema_version": 1,
                              "created_at": _now(), **self.config})
            self._read()

    @contextmanager
    def _locked(self):
        with self.lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _append(self, event):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _cost(self, prompt, completion):
        return (Decimal(prompt) * self.input_price + Decimal(completion) * self.output_price) / MILLION

    @staticmethod
    def _usage(usage):
        """Only explicit token counts can release a reservation."""
        if not isinstance(usage, dict):
            return None
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0
               for v in (prompt, completion)):
            return None
        total = usage.get("total_tokens")
        if total is not None and (isinstance(total, bool) or not isinstance(total, int)
                                  or total != prompt + completion):
            return None
        return {"prompt_tokens": prompt, "completion_tokens": completion,
                "total_tokens": prompt + completion}

    def _read(self):
        raw = self.path.read_bytes()
        # Fail closed on a partial append. An earlier reservation may represent
        # already billed work; guessing away corruption can authorize overspend.
        if not raw or not raw.endswith(b"\n"):
            raise BudgetError("budget ledger has an incomplete append; manual accounting review required")
        try:
            events = [json.loads(line) for line in raw.splitlines()]
        except (ValueError, UnicodeDecodeError):
            raise BudgetError("budget ledger is malformed; manual accounting review required") from None
        first = events[0]
        if not isinstance(first, dict) or first.get("event") != "configuration" or first.get("schema_version") != 1:
            raise BudgetError("budget ledger configuration is missing or unsupported")
        if any(first.get(key) != value for key, value in self.config.items()):
            raise BudgetError("budget cap/prices differ from the existing ledger; resume with the original configuration")
        reservations, finalized, request_ids = {}, {}, set()
        for event in events[1:]:
            if not isinstance(event, dict):
                raise BudgetError("budget ledger event is malformed")
            reservation = event.get("reservation_id")
            if event.get("event") == "reservation":
                request_id = event.get("request_id")
                if (not isinstance(reservation, str) or not reservation
                        or reservation in reservations or not isinstance(request_id, str)
                        or not request_id or request_id in request_ids):
                    raise BudgetError("budget ledger has duplicate or malformed attempt IDs")
                prompt = _tokens(event.get("prompt_token_bound"), "prompt_token_bound", positive=True)
                completion = _tokens(event.get("completion_token_bound"), "completion_token_bound", positive=True)
                if event.get("reserved_usd") != _decimal_string(self._cost(prompt, completion)):
                    raise BudgetError("budget reservation arithmetic is inconsistent")
                reservations[reservation] = event
                request_ids.add(request_id)
            elif event.get("event") == "finalization":
                if reservation not in reservations or reservation in finalized:
                    raise BudgetError("budget ledger has an unknown or duplicate finalization")
                normalized = self._usage(event.get("usage"))
                if event.get("usage") is not None and normalized != event.get("usage"):
                    raise BudgetError("budget ledger has invalid usage counters")
                if not isinstance(event.get("status"), str):
                    raise BudgetError("budget ledger has malformed finalization status")
                finalized[reservation] = event
            else:
                raise BudgetError("budget ledger contains an unknown event")
        return reservations, finalized, request_ids

    def _snapshot(self, reservations, finalized):
        committed, reserved, all_estimates = Decimal(0), Decimal(0), Decimal(0)
        prompt_tokens = completion_tokens = pending = unknown = measured = 0
        reserved_prompt_tokens = reserved_completion_tokens = 0
        estimated_prompt_tokens = estimated_completion_tokens = 0
        bound_exceeded = False
        for reservation, event in reservations.items():
            estimate = Decimal(event["reserved_usd"])
            all_estimates += estimate
            estimated_prompt_tokens += event["prompt_token_bound"]
            estimated_completion_tokens += event["completion_token_bound"]
            final = finalized.get(reservation)
            usage = final.get("usage") if final else None
            if usage is None:
                reserved += estimate
                reserved_prompt_tokens += event["prompt_token_bound"]
                reserved_completion_tokens += event["completion_token_bound"]
                if final:
                    unknown += 1
                else:
                    pending += 1
            else:
                measured += 1
                prompt_tokens += usage["prompt_tokens"]
                completion_tokens += usage["completion_tokens"]
                committed += self._cost(usage["prompt_tokens"], usage["completion_tokens"])
                if (usage["prompt_tokens"] > event["prompt_token_bound"]
                        or usage["completion_tokens"] > event["completion_token_bound"]):
                    bound_exceeded = True
        accounted = committed + reserved
        return {
            **self.config,
            "committed_usd": _decimal_string(committed),
            "rate_priced_usage_cost_usd": _decimal_string(committed),
            "reserved_usd": _decimal_string(reserved),
            "accounted_usd": _decimal_string(accounted),
            "available_usd": _decimal_string(max(Decimal(0), self.cap - accounted)),
            "estimated_reservation_usd": _decimal_string(all_estimates),
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "reserved_prompt_token_bound": reserved_prompt_tokens,
            "reserved_completion_token_bound": reserved_completion_tokens,
            "estimated_prompt_token_bound": estimated_prompt_tokens,
            "estimated_completion_token_bound": estimated_completion_tokens,
            "requests_reserved": len(reservations), "requests_finalized": len(finalized),
            "requests_with_usage": measured, "requests_with_unknown_usage": unknown,
            "pending_requests": pending,
            "token_bound_exceeded": bound_exceeded, "over_cap": accounted > self.cap,
            "cost_interpretation": "Configured-price estimates, not a provider invoice; cached input uses the full configured input rate; unresolved attempts retain their full reservation",
        }

    def snapshot(self):
        with self._locked():
            reservations, finalized, _ = self._read()
            return self._snapshot(reservations, finalized)

    def before_request(self, prompt_token_bound, completion_bound, *, request_id=None):
        prompt = _tokens(prompt_token_bound, "prompt_token_bound", positive=True)
        completion = _tokens(completion_bound, "completion_bound", positive=True)
        request_id = str(uuid.uuid4()) if request_id is None else request_id
        if not isinstance(request_id, str) or not request_id.strip():
            raise BudgetError("request_id must be a nonempty unique string")
        with self._locked():
            reservations, finalized, request_ids = self._read()
            current = self._snapshot(reservations, finalized)
            if request_id in request_ids:
                raise BudgetError("request_id was already reserved; retries need a new reservation")
            if current["token_bound_exceeded"]:
                raise BudgetExceeded("provider usage exceeded a reserved token bound; review accounting before further requests")
            estimate = self._cost(prompt, completion)
            if self.cap == 0 or Decimal(current["accounted_usd"]) + estimate > self.cap:
                raise BudgetExceeded("next HTTP attempt would exceed the configured USD cap; no request authorized")
            reservation = str(uuid.uuid4())
            self._append({"event": "reservation", "reservation_id": reservation,
                          "request_id": request_id, "created_at": _now(),
                          "prompt_token_bound": prompt, "completion_token_bound": completion,
                          "reserved_usd": _decimal_string(estimate)})
            return reservation

    def finalize(self, reservation, usage=None, status="completed"):
        if not isinstance(status, str) or not status.strip():
            raise BudgetError("status must be a nonempty string")
        normalized = self._usage(usage)
        with self._locked():
            reservations, finalized, _ = self._read()
            if reservation not in reservations:
                raise BudgetError("unknown reservation ID")
            if reservation in finalized:
                previous = finalized[reservation]
                if previous["usage"] != normalized or previous["status"] != status:
                    raise BudgetError("conflicting finalization for an already finalized attempt")
                return self._snapshot(reservations, finalized)
            event = {"event": "finalization", "reservation_id": reservation,
                     "finished_at": _now(), "status": status, "usage": normalized,
                     "usage_status": "measured" if normalized is not None else "unknown_reservation_retained"}
            self._append(event)
            finalized[reservation] = event
            return self._snapshot(reservations, finalized)
