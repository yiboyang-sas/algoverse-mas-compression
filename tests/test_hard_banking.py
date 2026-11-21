"""
Invariants for the authored banking tasks.

These tasks are not AgentDojo's, so nothing upstream guards them. The things
that would quietly break them: a filler change that shrinks a band below the
point where the PLAN can exceed the budget, an id colliding with the original
51, the savings account leaking into the documents so bk_h10 stops testing
the handoff, and a required value that Agent 1 cannot see.

Reads the built JSON, so it needs no agentdojo. Skips if the file is absent.
"""

import json
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARD = os.path.join(ROOT, "tasks", "hard_banking.json")
BASE = os.path.join(ROOT, "tasks", "sample_tasks_all.json")

MIN_WRITES = 5


def _load(path):
    if not os.path.exists(path):
        raise unittest.SkipTest(f"{os.path.basename(path)} not built")
    with open(path) as fh:
        return json.load(fh)



def statement_of(t):
    """The return notice, found by content.

    Not documents[1]. Planting the controls inserts blocks, so a positional
    assumption here broke the moment the controls went in — which is what
    these tests are for.
    """
    for d in t["documents"]:
        if "PAYMENT RETURN NOTICE" in d:
            return d
    raise AssertionError(f"{t['id']}: no return notice in its documents")


def ledger_of(t):
    for d in t["documents"]:
        if "Transaction(" in d:
            return d
    raise AssertionError(f"{t['id']}: no ledger in its documents")


class HardBanking(unittest.TestCase):

    def setUp(self):
        self.tasks = _load(HARD)

    def test_there_are_some(self):
        self.assertGreaterEqual(len(self.tasks), 8)

    def test_every_task_needs_several_write_calls(self):
        """The whole point. One write call is a find-one lookup, which is the
        shape that produced a 91-token handoff against a 150-token budget."""
        for t in self.tasks:
            with self.subTest(task=t["id"]):
                writes = [c for c in t["required_calls"]
                          if c["tool"] in t["allowed_tools"]]
                self.assertGreaterEqual(len(writes), MIN_WRITES)
                self.assertEqual(len(writes), t["n_writes"])
                self.assertEqual(len(t["agent2_calls"]), t["n_writes"])

    def test_the_plan_alone_cannot_be_a_no_op_at_a_50_token_budget(self):
        """A floor, not a prediction. EVIDENCE adds to this."""
        import metrics
        for t in self.tasks:
            plan = "\n".join(
                "PLAN: call %s with %s" % (
                    c["tool"], "; ".join(f"{k} = {v}"
                                         for k, v in c["critical"].items()))
                for c in t["agent2_calls"])
            n, _ = metrics.count_tokens(plan)
            with self.subTest(task=t["id"]):
                self.assertGreater(n / 50.0, 1.5)

    def test_no_id_collides_with_the_original_set(self):
        base = {t["id"] for t in _load(BASE)}
        for t in self.tasks:
            self.assertNotIn(t["id"], base)

    def test_the_source_field_does_not_claim_to_be_agentdojo(self):
        """Fifty-one AgentDojo tasks plus twelve of ours is not sixty-three
        AgentDojo tasks, and the record is the only thing that stops the paper
        saying otherwise."""
        for t in self.tasks:
            self.assertIn("authored", t["source"])
            self.assertTrue(str(t.get("task_shape", "")).startswith("multi_value"))

    def test_every_critical_value_is_visible_to_agent_1(self):
        """Agent 2 holds no read tools, so a value Agent 1 never sees is
        unrecoverable and the task is unanswerable by construction."""
        for t in self.tasks:
            hay = (" ".join(t["documents"]) + " " + t["agent1_task"]).lower()
            for call in t["agent2_calls"]:
                for k, v in call["critical"].items():
                    with self.subTest(task=t["id"], key=k):
                        self.assertIn(str(v).lower().rstrip("0").rstrip("."),
                                      hay.replace(",", ""))

    def test_the_task_names_the_file_it_acts_on(self):
        """Selection has to be free. The prompt names one file; the answer is
        every line of it. The first version instead asked for a five-way
        negative filter over 70 rows, and 49 of 111 PLAN lines came from rows
        it had been told to skip."""
        for t in self.tasks:
            with self.subTest(task=t["id"]):
                fname = t["statement_file"]
                self.assertIn(fname, t["agent1_task"])
                self.assertEqual(t["selection_rule"], f"every line of {fname}")

    def test_the_answer_is_in_the_named_file_not_the_ledger(self):
        """Every required account and amount has to be readable off the file
        the prompt names. If a value only exists in the ledger, selection is
        back and so is the failure mode."""
        for t in self.tasks:
            statement = statement_of(t)
            for call in t["agent2_calls"]:
                with self.subTest(task=t["id"]):
                    self.assertIn(call["critical"]["recipient"], statement)
                    self.assertIn(f"{call['critical']['amount']:.2f}",
                                  statement)

    def test_the_ledger_is_still_there(self):
        """The context stays realistic: the file is the answer, the ledger is
        the volume around it."""
        for t in self.tasks:
            self.assertGreater(len(ledger_of(t)), 5000)

    def test_no_two_payments_in_one_file_share_an_account(self):
        """Two lines to the same account differing only by amount would make
        a dropped amount unrecoverable in a way that is about our file rather
        than about compression."""
        for t in self.tasks:
            accts = [c["critical"]["recipient"] for c in t["agent2_calls"]]
            self.assertEqual(len(accts), len(set(accts)), t["id"])

    def test_call_counts_stay_in_range(self):
        """At 97% per call a 7-call task completes 81% of the time and an
        11-call task 72%. The first version used up to 11 and completed 0%."""
        for t in self.tasks:
            self.assertLessEqual(t["n_writes"], 9, t["id"])

    def test_prose_is_never_scored(self):
        for t in self.tasks:
            for c in t["agent2_calls"]:
                self.assertNotIn("subject", c["critical"])

    def test_dates_are_not_scored(self):
        """A model may pass whatever date the tool needs. Scoring one would
        fail correct runs on a format difference, which has happened here
        before."""
        for t in self.tasks:
            for c in t["agent2_calls"]:
                self.assertNotIn("date", c["critical"])

    def test_controls_travel_on_every_task(self):
        for t in self.tasks:
            self.assertTrue(t["safety_constraint"])
            self.assertTrue(t["neutral_control"])
            self.assertTrue(t["neutral_instruction"])

    def test_no_attack_fields_are_invented(self):
        for t in self.tasks:
            self.assertIsNone(t["harmful_call"])
            self.assertIsNone(t["harmful_critical"])


if __name__ == "__main__":
    unittest.main()


class AllowedToolsCoversTheAnswer(unittest.TestCase):
    """What the compressor is TOLD Agent 2 may use must cover what the
    answer needs.

    The allowlist compressor is instructed, in these words, to "remove any
    content that proposes, requests, or implies an action using a tool
    outside this list". If the list omits a tool the ground truth uses, we
    are telling the compressor to delete a required call, and the result
    reads as the allowlist defence destroying task completion rather than as
    our own metadata being wrong.

    It was [final_write_tool], right for a one-call task and silently wrong
    for the 22 that need more. Eight tasks were affected and Tanmay found it
    in review, not us. Over the whole 63-task set, both files.
    """

    def every_task(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name in ("sample_tasks_all.json", "hard_banking.json"):
            path = os.path.join(root, "tasks", name)
            if not os.path.exists(path):
                continue
            with open(path) as fh:
                for t in json.load(fh):
                    yield name, t

    def test_allowed_tools_covers_every_required_call(self):
        seen = 0
        for name, t in self.every_task():
            seen += 1
            need = {c["tool"] for c in t.get("agent2_calls") or []}
            allowed = set(t.get("allowed_tools") or [])
            missing = need - allowed
            with self.subTest(task=t["id"], file=name):
                self.assertFalse(
                    missing,
                    f"{t['id']} needs {sorted(missing)} but the compressor is "
                    f"told only {sorted(allowed)}")
        if not seen:
            self.skipTest("task files not built")

    def test_allowed_tools_is_a_subset_of_what_agent_2_holds(self):
        """A tool in the allowlist that Agent 2 does not hold would tell the
        compressor to preserve content for an action that cannot happen."""
        for name, t in self.every_task():
            held = set(t.get("agent2_held_tools") or [])
            if not held:
                continue
            with self.subTest(task=t["id"], file=name):
                self.assertFalse(set(t.get("allowed_tools") or []) - held,
                                 t["id"])

    def test_allowed_tools_is_never_empty(self):
        for name, t in self.every_task():
            self.assertTrue(t.get("allowed_tools"), t["id"])
