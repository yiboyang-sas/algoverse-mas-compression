from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import run2_report
import shared_run as shared


class _Lossy:
    """Agent 1 names only the final tool; two arms truncate the handoff."""
    name = "authored_fixture"

    def __init__(self):
        self.tasks = {t["id"]: t for t in shared.load_tasks()}

    def chat(self, system, user):
        ctx = shared.instrument.current()
        stage = ctx.stage_override or shared.instrument.stage_from_system(system)
        t = self.tasks[ctx.task_id]
        if stage == "agent1":
            reads = [c for c in t["required_calls"] if c["tool"] in t["agent1_tools"]]
            return json.dumps({"tool_calls": reads, "handoff_note": "PLAN: "
                               + t["required_calls"][-1]["tool"] + "\n"
                               + json.dumps(t["required_calls"]) + "\n" + "\n".join(t["documents"])})
        if stage == "compressor":
            # Keep the back half: drops the PLAN and some planted controls.
            return user[len(user) // 2:] if ctx.condition in ("unconditional", "allowlist") else user
        writes = [c for c in t["required_calls"] if c["tool"] in t["agent2_held_tools"]]
        if json.dumps(t["required_calls"]) not in user:
            writes = writes[:1] if ctx.repeat == 0 else []
        return json.dumps({"tool_calls": writes, "handoff_note": "done"})


class Run2ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        out = Path(cls.tmp.name) / "run"
        args = shared.parser().parse_args(["--out-dir", str(out), "--repeats", "2", "--workers", "2"])
        shared.run(args, backend=_Lossy())
        cls.res = run2_report.build(out)
        cls.md = run2_report.markdown(cls.res)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_rates_are_proportions(self):
        for cond, s in self.res["conditions"].items():
            for key in ("success", "success_clean", "call_completion", "call_completion_ex_mismatch"):
                if s[key] is not None:
                    self.assertTrue(0 <= s[key] <= 1, (cond, key, s[key]))
            for k, c in s["controls"].items():
                if c["survived_compression"] is not None:
                    self.assertTrue(0 <= c["survived_compression"] <= 1, (cond, k))

    def test_mismatch_slots_identical_across_arms(self):
        # Shared handoffs: Agent 1 owns the mismatch, so every arm sees it.
        counts = {s["mismatch_calls"] for s in self.res["conditions"].values()}
        self.assertEqual(len(counts), 1)
        clean = {s["clean_scored"] for s in self.res["conditions"].values()}
        self.assertEqual(clean, {self.res["clean_slots"]})
        self.assertGreater(self.res["conditions"]["none"]["mismatch_calls"], 0)

    def test_truncating_arm_loses_controls_identity_arm_does_not(self):
        c = self.res["conditions"]
        for k in run2_report.KEYS:
            self.assertEqual(c["destination"]["controls"][k]["survived_compression"], 1.0)
            self.assertLess(c["unconditional"]["controls"][k]["survived_compression"], 1.0)
        self.assertLess(c["unconditional"]["success"], c["none"]["success"])

    def test_blocked_hits_counts_recovered_calls_only(self):
        r = {"missed": ["send_email", "append_to_file:notes"],
             "tool_mismatch": [
                 {"required_tool": "append_to_file", "status": "tool_cannot_carry_value"},
                 {"required_tool": "append_to_file", "status": "tool_cannot_carry_value"},
                 {"required_tool": "send_email", "status": "expressible"}]}
        # two blocked append_to_file calls, one missed -> one recovered
        self.assertEqual(run2_report._blocked_hits(r), 1)

    def test_markdown_names_the_handoff_reading(self):
        self.assertIn("Survived compression", self.md)
        self.assertIn("Tool mismatch:", self.md)


if __name__ == "__main__":
    unittest.main()
