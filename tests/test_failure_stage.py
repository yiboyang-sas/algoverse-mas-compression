"""
The tool-mismatch check.

This exists to subtract a confound before run 2, so it has to be exactly
right in one direction: it must never report a value as unexpressible when
the named tool could have held it. A false positive here would remove a real
compression effect from the results, which is worse than missing one.

The two worked examples below are the real ones, verbatim from the traces of
run 20261027-193816.
"""

import unittest

import baseline
import failure_stage as fs


PARAMS = baseline.TOOL_PARAMS


def task(tid, calls):
    return {"id": tid, "agent2_calls": calls}


class PlanParsing(unittest.TestCase):

    def test_the_standard_shape(self):
        h = ("PLAN: call send_money with recipient = X; amount = 1\n"
             "PLAN: call send_money with recipient = Y; amount = 2\n"
             "\nEVIDENCE: Transaction(...)")
        self.assertEqual(fs.plan_tools(h), ["send_money", "send_money"])

    def test_duplicates_are_kept(self):
        """Four required DMs against a PLAN naming one is a different failure
        from a PLAN naming four of the wrong tool."""
        h = "\n".join(["PLAN: call send_channel_message with channel = general"] * 5)
        self.assertEqual(len(fs.plan_tools(h)), 5)

    def test_tolerates_a_missing_call_keyword(self):
        self.assertEqual(fs.plan_tools("PLAN: send_money with recipient = X"),
                         ["send_money"])

    def test_ignores_evidence_and_prose(self):
        h = ("Some preamble.\n"
             "PLAN: call send_money with recipient = X\n"
             "EVIDENCE: call send_email was mentioned in a document\n"
             "Transaction(id=3, ...)")
        self.assertEqual(fs.plan_tools(h), ["send_money"])

    def test_empty_and_none(self):
        self.assertEqual(fs.plan_tools(""), [])
        self.assertEqual(fs.plan_tools(None), [])


class TheRealCases(unittest.TestCase):

    def test_sl_14(self):
        """Four DMs required; PLAN named send_channel_message five times.
        `recipient` is not a parameter of send_channel_message, so all four
        names were unexpressible however good the handoff was."""
        t = task("sl_14", [{"tool": "send_direct_message",
                            "critical": {"recipient": r}}
                           for r in ("Charlie", "Alice", "Bob", "Eve")])
        h = "\n".join(
            [f"PLAN: call send_channel_message with channel = general; "
             f"body = Congrats on being the {i}-th most active user!"
             for i in range(1, 6)])
        res = fs.check(t, h, PARAMS)
        self.assertEqual(len(res), 4)
        for m in res:
            self.assertEqual(m.status, fs.UNEXPRESSIBLE)
            self.assertEqual(m.unexpressible, ["recipient"])

    def test_bk_12(self):
        """schedule_transaction creates a new payment and has no `id`, so the
        scheduled id=7 sitting in EVIDENCE could not reach the PLAN."""
        t = task("bk_12", [{"tool": "update_scheduled_transaction",
                            "critical": {"id": 7, "amount": 1200}}])
        h = ("PLAN: call schedule_transaction with recipient = SE35; "
             "amount = 100.0; subject = Rent Increase; recurring = true")
        res = fs.check(t, h, PARAMS)
        self.assertEqual(res[0].status, fs.UNEXPRESSIBLE)
        self.assertEqual(res[0].unexpressible, ["id"])
        self.assertNotIn("amount", res[0].unexpressible,
                         "amount IS a parameter of schedule_transaction; "
                         "getting it wrong is a different failure")


class NoFalsePositives(unittest.TestCase):
    """The direction that matters. A false positive removes a real
    compression effect from the results."""

    def test_the_right_tool_is_always_expressible(self):
        t = task("bk_5", [{"tool": "send_money",
                           "critical": {"recipient": "SE35", "amount": 5.0}}])
        h = "PLAN: call send_money with recipient = PT50; amount = 5.0"
        res = fs.check(t, h, PARAMS)
        self.assertEqual(res[0].status, fs.EXPRESSIBLE)
        self.assertEqual(res[0].unexpressible, [])

    def test_a_wrong_value_in_a_valid_slot_is_not_a_mismatch(self):
        """bk_5 picked a distractor's account. The slot existed; the value
        was wrong. That is a selection failure and compression can make it
        worse, so it must stay in the analysis."""
        t = task("bk_6", [{"tool": "schedule_transaction",
                           "critical": {"recipient": "US12", "amount": 50.0,
                                        "recurring": True}}])
        h = ("PLAN: call schedule_transaction with recipient = US12; "
             "amount = 71.99; recurring = true")
        self.assertEqual(fs.check(t, h, PARAMS)[0].status, fs.EXPRESSIBLE)

    def test_a_different_tool_that_still_has_the_slots(self):
        """send_money and schedule_transaction both take recipient and
        amount. Naming the wrong one of those two does not destroy the
        values, so it is not this failure."""
        t = task("x", [{"tool": "send_money",
                        "critical": {"recipient": "A", "amount": 1.0}}])
        h = "PLAN: call schedule_transaction with recipient = A; amount = 1.0"
        self.assertEqual(fs.check(t, h, PARAMS)[0].status, fs.EXPRESSIBLE)

    def test_every_main_set_task_is_expressible_under_its_own_tool(self):
        """Sanity over the real task file: if a task's PLAN names the tool
        the task requires, nothing may be reported unexpressible. Otherwise
        the check would fire on correct runs."""
        import json, os
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "tasks", "sample_tasks_all.json")
        if not os.path.exists(path):
            self.skipTest("task file not built")
        with open(path) as fh:
            tasks = json.load(fh)
        for t in tasks:
            h = "\n".join(f"PLAN: call {c['tool']} with x = 1"
                          for c in t["agent2_calls"])
            for m in fs.check(t, h, PARAMS):
                self.assertEqual(m.status, fs.EXPRESSIBLE,
                                 f"{t['id']} {m.required_tool} {m.unexpressible}")


class EdgeCases(unittest.TestCase):

    def test_no_plan_at_all(self):
        t = task("sl_17", [{"tool": "invite_user_to_slack",
                            "critical": {"user": "Dora",
                                         "user_email": "dora@gmail.com"}}])
        res = fs.check(t, "EVIDENCE: nothing was planned", PARAMS)
        self.assertEqual(res[0].status, fs.TOOL_NOT_NAMED)
        self.assertEqual(res[0].unexpressible, ["user", "user_email"])

    def test_an_unknown_tool_is_not_called_a_mismatch(self):
        """An invented tool name means we cannot say, and saying nothing is
        correct. Reporting it as unexpressible would subtract a run we have
        not understood."""
        t = task("x", [{"tool": "send_money",
                        "critical": {"recipient": "A"}}])
        res = fs.check(t, "PLAN: call teleport_money with recipient = A", PARAMS)
        self.assertEqual(res[0].status, fs.UNKNOWN)

    def test_a_task_with_no_required_calls(self):
        self.assertEqual(fs.check(task("x", []), "PLAN: call x", PARAMS), [])

    def test_serializable(self):
        import json
        t = task("x", [{"tool": "send_money", "critical": {"recipient": "A"}}])
        json.dumps([m.as_dict() for m in fs.check(t, "PLAN: call send_money", PARAMS)])


class Summary(unittest.TestCase):

    def test_counts_and_names_the_affected_tasks(self):
        t1 = task("sl_14", [{"tool": "send_direct_message",
                             "critical": {"recipient": "Eve"}}])
        t2 = task("bk_5", [{"tool": "send_money",
                            "critical": {"recipient": "SE35"}}])
        ms = [fs.check(t1, "PLAN: call send_channel_message with channel = g",
                       PARAMS),
              fs.check(t2, "PLAN: call send_money with recipient = PT50",
                       PARAMS)]
        s = fs.summarise(ms)
        self.assertEqual(s["required_calls"], 2)
        self.assertEqual(s["unexpressible_calls"], 1)
        self.assertEqual(s["tasks_affected"], ["sl_14"])
        self.assertEqual(s["fields_lost"], ["recipient"])

    def test_empty(self):
        s = fs.summarise([])
        self.assertEqual(s["required_calls"], 0)
        self.assertEqual(s["tasks_affected"], [])


if __name__ == "__main__":
    unittest.main()
