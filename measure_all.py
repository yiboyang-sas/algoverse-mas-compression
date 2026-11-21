"""
How much context each candidate task actually hands Agent 1, per suite, BEFORE
any expansion.

This decides how much filler each suite needs. Workspace documents were around
800 tokens and needed heavy expansion to make the handoff cap bite. Slack and
banking environments are smaller, so they may need more, or may be unusable.

Measured by running the ground truth's read steps for real. Estimating from
store size is useless: a task that searches for one channel gets one channel
back, not the whole workspace.

    python3 measure_all.py
"""

import json
import sys
from statistics import median

from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite

VERSION = "v1.2.1"

SPLIT = {
    "workspace": {
        "read": ["get_unread_emails", "get_sent_emails", "get_received_emails",
                 "get_draft_emails", "search_emails", "search_contacts_by_name",
                 "search_contacts_by_email", "get_current_day",
                 "search_calendar_events", "get_day_calendar_events",
                 "search_files_by_filename", "get_file_by_id", "list_files",
                 "search_files"],
        "write": ["send_email", "delete_email", "create_calendar_event",
                  "cancel_calendar_event", "reschedule_calendar_event",
                  "add_calendar_event_participants", "append_to_file",
                  "create_file", "delete_file", "share_file"],
    },
    "slack": {
        "read": ["get_channels", "get_users_in_channel", "get_webpage",
                 "read_channel_messages", "read_inbox"],
        "write": ["add_user_to_channel", "invite_user_to_slack", "post_webpage",
                  "remove_user_from_slack", "send_channel_message",
                  "send_direct_message"],
    },
    "banking": {
        "read": ["get_balance", "get_iban", "get_most_recent_transactions",
                 "get_scheduled_transactions", "get_user_info", "read_file"],
        "write": ["schedule_transaction", "send_money", "update_password",
                  "update_scheduled_transaction", "update_user_info"],
    },
    "travel": {
        "read": ["check_restaurant_opening_hours",
                 "get_all_car_rental_companies_in_city", "get_all_hotels_in_city",
                 "get_all_restaurants_in_city", "get_car_fuel_options",
                 "get_car_price_per_day", "get_car_rental_address",
                 "get_car_types_available", "get_contact_information_for_restaurants",
                 "get_cuisine_type_for_restaurants", "get_day_calendar_events",
                 "get_dietary_restrictions_for_all_restaurants",
                 "get_flight_information", "get_hotels_address", "get_hotels_prices",
                 "get_price_for_restaurants", "get_rating_reviews_for_car_rental",
                 "get_rating_reviews_for_hotels", "get_rating_reviews_for_restaurants",
                 "get_restaurants_address", "get_user_information",
                 "search_calendar_events"],
        "write": ["cancel_calendar_event", "create_calendar_event",
                  "reserve_car_rental", "reserve_hotel", "reserve_restaurant",
                  "send_email"],
    },
}


def tier_of(tokens):
    if tokens < 1000:
        return "thin"
    if tokens <= 16000:
        return "core"
    return "oversized"


def measure(suite_name):
    cfg = SPLIT[suite_name]
    reads, writes = set(cfg["read"]), set(cfg["write"])

    suite = get_suite(VERSION, suite_name)
    env = suite.load_and_inject_default_environment({})
    runtime = FunctionsRuntime(suite.tools)

    rows = []
    for tid, t in suite.user_tasks.items():
        try:
            gt = t.ground_truth(env)
        except Exception:
            continue
        if not gt or gt[-1].function not in writes:
            continue

        total = 0
        for step in gt:
            if step.function not in reads:
                continue
            try:
                out, _ = runtime.run_function(env, step.function, step.args,
                                              raise_on_error=False)
                total += len(str(out))
            except Exception:
                pass
        tok = total // 4

        final = gt[-1]
        critical = {k: v for k, v in final.args.items()
                    if k not in ("body", "description", "content", "message")
                    and v not in (None, [], "")}

        rows.append({
            "suite": suite_name, "task_id": tid, "tokens": tok,
            "tier": tier_of(tok), "n_steps": len(gt),
            "n_read_steps": sum(1 for s in gt if s.function in reads),
            "final_write": final.function,
            "has_critical": bool(critical),
            "critical_keys": list(critical),
            "prompt": t.PROMPT[:90],
        })
    return rows


def main():
    allrows = []
    for name in ["workspace", "slack", "banking", "travel"]:
        rows = measure(name)
        allrows += rows
        toks = [r["tokens"] for r in rows] or [0]
        print("=" * 74)
        print(f"{name.upper():<12} {len(rows)} tasks ending in a write")
        print("=" * 74)
        print(f"  context tokens: min {min(toks)}, median {int(median(toks))}, "
              f"max {max(toks)}")
        for tier in ("thin", "core", "oversized"):
            n = sum(1 for r in rows if r["tier"] == tier)
            if n:
                print(f"    {tier:<10} {n}")
        nocrit = [r["task_id"] for r in rows if not r["has_critical"]]
        print(f"  no extractable handoff_critical: {nocrit or 'none'}")
        print(f"\n  {'task':<16}{'tok':>7}{'rd':>4}  {'tier':<10}{'final write':<26}crit")
        print("  " + "-" * 70)
        for r in sorted(rows, key=lambda x: x["tokens"]):
            print(f"  {r['task_id']:<16}{r['tokens']:>7}{r['n_read_steps']:>4}  "
                  f"{r['tier']:<10}{r['final_write']:<26}"
                  f"{','.join(r['critical_keys'])[:22]}")
        print()

    print("=" * 74)
    core = sum(1 for r in allrows if r["tier"] == "core")
    thin = sum(1 for r in allrows if r["tier"] == "thin")
    over = sum(1 for r in allrows if r["tier"] == "oversized")
    print(f"ALL FOUR SUITES, unexpanded: {len(allrows)} candidates")
    print(f"  core {core}, thin {thin}, oversized {over}")
    print("  thin is the number expansion has to fix.")
    print("=" * 74)

    with open("context_survey.json", "w") as fh:
        json.dump(allrows, fh, indent=2, default=str)
    print("\nwrote context_survey.json")


if __name__ == "__main__":
    sys.exit(main())
