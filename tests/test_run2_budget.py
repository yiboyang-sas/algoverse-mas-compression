"""Budget reservation behavior; no HTTP requests or credentials involved."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from budget import BudgetError, BudgetExceeded, BudgetLedger, prompt_token_bound


def ledger(tmp_path, cap="0.2", input_price="1", output_price="1"):
    return BudgetLedger(tmp_path / "budget.jsonl", cap_usd=cap,
                        input_per_million_usd=input_price,
                        output_per_million_usd=output_price)


def usage(prompt=20_000, completion=10_000):
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion}


def test_zero_budget_never_authorizes_a_request(tmp_path):
    account = ledger(tmp_path, cap="0")
    with pytest.raises(BudgetExceeded):
        account.before_request(1, 1)
    assert account.snapshot()["requests_reserved"] == 0


def test_exact_reservation_boundary_and_decimal_accounting(tmp_path):
    account = ledger(tmp_path)
    account.before_request(50_000, 50_000, request_id="attempt-1")
    account.before_request(50_000, 50_000, request_id="attempt-2")
    with pytest.raises(BudgetExceeded):
        account.before_request(1, 1, request_id="attempt-3")
    assert account.snapshot()["accounted_usd"] == "0.2"
    assert account.snapshot()["available_usd"] == "0"


def test_returned_usage_releases_only_unused_reservation(tmp_path):
    account = ledger(tmp_path)
    reservation = account.before_request(50_000, 50_000)
    snapshot = account.finalize(reservation, usage(), status="completed")
    assert snapshot["committed_usd"] == "0.03"
    assert snapshot["rate_priced_usage_cost_usd"] == "0.03"
    assert snapshot["reserved_usd"] == "0"
    assert snapshot["available_usd"] == "0.17"
    assert snapshot["prompt_tokens"] == 20_000
    assert snapshot["completion_tokens"] == 10_000
    assert snapshot["requests_with_usage"] == 1
    account.before_request(80_000, 90_000)
    assert account.snapshot()["accounted_usd"] == "0.2"


def test_failed_request_and_retry_reserve_separately(tmp_path):
    account = ledger(tmp_path)
    first = account.before_request(50_000, 50_000, request_id="attempt-1")
    account.finalize(first, status="transport_error")
    second = account.before_request(50_000, 50_000, request_id="attempt-2")
    assert second != first
    account.finalize(second, usage(), status="completed")
    snapshot = account.snapshot()
    assert snapshot["accounted_usd"] == "0.13"
    assert snapshot["reserved_usd"] == "0.1"
    assert snapshot["requests_with_unknown_usage"] == 1
    assert snapshot["requests_with_usage"] == 1
    assert snapshot["requests_reserved"] == 2
    assert snapshot["reserved_prompt_token_bound"] == 50_000
    assert snapshot["reserved_completion_token_bound"] == 50_000
    assert snapshot["estimated_prompt_token_bound"] == 100_000
    with pytest.raises(BudgetExceeded):
        account.before_request(50_000, 50_000, request_id="attempt-3")


def test_crash_and_process_restart_keep_unknown_charge_reserved(tmp_path):
    account = ledger(tmp_path, cap="0.1")
    account.before_request(50_000, 50_000, request_id="crashed-attempt")
    script = (
        "from budget import BudgetLedger, BudgetExceeded; import sys; "
        "b=BudgetLedger(sys.argv[1],cap_usd='0.1',input_per_million_usd='1',output_per_million_usd='1'); "
        "b.before_request(1,1,request_id='new-process')"
    )
    result = subprocess.run([sys.executable, "-c", script, str(account.path)],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and "BudgetExceeded" in result.stderr
    restored = ledger(tmp_path, cap="0.1").snapshot()
    assert restored["reserved_usd"] == "0.1"
    assert restored["pending_requests"] == 1


def test_unique_attempt_id_cannot_authorize_a_second_network_request(tmp_path):
    account = ledger(tmp_path)
    reservation = account.before_request(1, 1, request_id="unique")
    account.finalize(reservation, usage(1, 1))
    with pytest.raises(BudgetError, match="already reserved"):
        ledger(tmp_path).before_request(1, 1, request_id="unique")


def test_finalize_is_idempotent_but_never_releases_unknown_usage_later(tmp_path):
    account = ledger(tmp_path)
    reservation = account.before_request(100, 100)
    first = account.finalize(reservation, status="transport_error")
    assert account.finalize(reservation, status="transport_error") == first
    with pytest.raises(BudgetError, match="conflicting"):
        account.finalize(reservation, usage(1, 1), status="completed")
    assert account.snapshot()["reserved_usd"] == "0.0002"


@pytest.mark.parametrize("returned", [
    None, {}, {"prompt_tokens": 1},
    {"prompt_tokens": -1, "completion_tokens": 1},
    {"prompt_tokens": True, "completion_tokens": 1},
    {"prompt_tokens": 1.0, "completion_tokens": 1},
    {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 99},
])
def test_missing_or_untrustworthy_provider_usage_keeps_reserve(tmp_path, returned):
    account = ledger(tmp_path)
    reservation = account.before_request(100, 100)
    snapshot = account.finalize(reservation, returned)
    assert snapshot["reserved_usd"] == "0.0002"
    assert snapshot["committed_usd"] == "0"
    assert snapshot["requests_with_unknown_usage"] == 1


def test_bound_violation_records_usage_and_stops_future_attempts(tmp_path):
    account = ledger(tmp_path)
    reservation = account.before_request(100, 100)
    snapshot = account.finalize(reservation, usage(101, 1))
    assert snapshot["token_bound_exceeded"] is True
    assert snapshot["rate_priced_usage_cost_usd"] == "0.000102"
    with pytest.raises(BudgetExceeded, match="token bound"):
        account.before_request(1, 1)


def test_parallel_attempts_cannot_collectively_exceed_cap(tmp_path):
    account = ledger(tmp_path, cap="0.3")
    def reserve(index):
        try:
            ledger(tmp_path, cap="0.3").before_request(50_000, 50_000,
                                                     request_id=f"attempt-{index}")
            return True
        except BudgetExceeded:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = list(pool.map(reserve, range(16)))
    assert sum(accepted) == 3
    snapshot = account.snapshot()
    assert snapshot["accounted_usd"] == "0.3"
    assert snapshot["requests_reserved"] == 3


def test_independent_processes_share_the_same_reservation_lock(tmp_path):
    account = ledger(tmp_path, cap="0.3")
    script = """
import sys
from budget import BudgetLedger, BudgetExceeded
account = BudgetLedger(sys.argv[1], cap_usd='0.3',
                       input_per_million_usd='1', output_per_million_usd='1')
try:
    account.before_request(50_000, 50_000, request_id=sys.argv[2])
except BudgetExceeded:
    sys.exit(3)
"""
    processes = [subprocess.Popen([sys.executable, "-c", script, str(account.path),
                                   f"process-{i}"], cwd=ROOT,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                 for i in range(8)]
    results = []
    for process in processes:
        _, err = process.communicate(timeout=10)
        assert process.returncode in (0, 3), err.decode()
        results.append(process.returncode)
    assert results.count(0) == 3
    assert results.count(3) == 5
    assert account.snapshot()["accounted_usd"] == "0.3"


@pytest.mark.parametrize("argument,value", [
    ("cap_usd", "-1"), ("cap_usd", "NaN"),
    ("input_per_million_usd", "0"), ("output_per_million_usd", "Infinity"),
    ("cap_usd", True),
])
def test_invalid_or_missing_rates_fail_closed(tmp_path, argument, value):
    kwargs = dict(cap_usd="1", input_per_million_usd="1", output_per_million_usd="1")
    kwargs[argument] = value
    with pytest.raises(BudgetError):
        BudgetLedger(tmp_path / "budget.jsonl", **kwargs)


@pytest.mark.parametrize("cap,input_price,output_price", [
    ("0.3", "1", "1"), ("0.2", "2", "1"), ("0.2", "1", "2"),
])
def test_resume_cannot_change_cap_or_prices(tmp_path, cap, input_price, output_price):
    ledger(tmp_path)
    with pytest.raises(BudgetError, match="differ"):
        ledger(tmp_path, cap=cap, input_price=input_price, output_price=output_price)


@pytest.mark.parametrize("tail", [b'{"event":', b'{broken}\n'])
def test_corrupt_ledger_does_not_guess_away_reserved_charges(tmp_path, tail):
    account = ledger(tmp_path)
    account.before_request(100, 100)
    with account.path.open("ab") as handle:
        handle.write(tail)
    with pytest.raises(BudgetError, match="review"):
        account.snapshot()


def test_custom_prices_and_cached_tokens_use_full_configured_input_rate(tmp_path):
    account = ledger(tmp_path, cap="1", input_price="2", output_price="7")
    reservation = account.before_request(1000, 1000)
    returned = usage(100, 10)
    returned["prompt_tokens_details"] = {"cached_tokens": 100}
    result = account.finalize(reservation, returned)
    assert Decimal(result["rate_priced_usage_cost_usd"]) == Decimal("0.00027")
    assert result["estimated_reservation_usd"] == "0.009"


def test_prompt_token_bound_covers_utf8_message_bytes_and_chat_framing():
    messages = [{"role": "system", "content": "Read only."},
                {"role": "user", "content": "Résumé: 日本語 🚀\n" * 100}]
    actual_bytes = sum(len(m["content"].encode("utf-8")) for m in messages)
    assert prompt_token_bound(messages) > actual_bytes + 256 + 64 * 2


@pytest.mark.parametrize("messages", [None, [], [{"role": "user", "content": None}],
                                       [{"role": "user", "content": [{"type": "image"}]}]])
def test_prompt_bound_rejects_unsupported_multimodal_requests(messages):
    with pytest.raises(BudgetError):
        prompt_token_bound(messages)


@pytest.mark.parametrize("prompt,completion", [(0, 1), (1, 0), (True, 1), (1, -1)])
def test_invalid_token_bounds_cannot_authorize_attempts(tmp_path, prompt, completion):
    account = ledger(tmp_path)
    with pytest.raises(BudgetError):
        account.before_request(prompt, completion)
    assert account.snapshot()["requests_reserved"] == 0
