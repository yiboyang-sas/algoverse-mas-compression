"""
The run 3 to run 4 selection.

What has to hold here is not arithmetic, it is that the selection stays
defensible. Three ways it could stop being so, all of them quiet:

  it selects on a treatment arm instead of the control, which biases the
  comparison run 4 exists to make;

  it treats an API failure as a defended attack, which is the same error as
  counting an unplanted control as a dropped one;

  it drops the fact that the resulting rates are conditional, so someone
  later pairs a run 4 rate with run 3's 26% as though the denominators
  matched.

The last one is not a code property, so the provenance block is asserted
instead: whatever is written has to carry the rule it was selected by.
"""

import json
from pathlib import Path
import tempfile
import unittest

import carry_forward


def row(spec_id, fired, *, condition="none", api_failure=False, kind="run"):
    return {"kind": kind, "condition": condition, "attack_spec_id": spec_id,
            "attack_fired_any": fired, "api_failure": api_failure}


def spec(task, injection, wrapper, suite="banking", relation="tool_novel"):
    return {"task_id": task, "injection": injection, "wrapper": wrapper,
            "suite": suite, "tool_relation": relation}


class Run:
    """A finished attack run on disk, as much of one as this needs."""

    def __init__(self, rows, condition="none"):
        self.temp = tempfile.TemporaryDirectory(prefix="carry-test-")
        self.dir = Path(self.temp.name) / "run"
        arm = self.dir / "arms" / condition
        arm.mkdir(parents=True)
        (arm / "runs.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
        (self.dir / "manifest.json").write_text(json.dumps(
            {"experiment_id": "abc123", "manifest_sha256": "f" * 64}))

    def cleanup(self):
        self.temp.cleanup()


class Reading(unittest.TestCase):

    def run_with(self, rows):
        r = Run(rows)
        self.addCleanup(r.cleanup)
        return r

    def test_repeats_are_kept_separate_not_pooled(self):
        """The per-spec pattern is the point. A spec that fired once out of
        three is a different thing from one that fired three times, and the
        status summary cannot tell them apart."""
        r = self.run_with([row("a", True), row("a", False), row("a", False),
                           row("b", False), row("b", False), row("b", False)])
        fired = carry_forward.fired_by_spec(r.dir)
        self.assertEqual(fired["a"], [True, False, False])
        self.assertEqual(fired["b"], [False, False, False])

    def test_api_failures_are_dropped_not_counted_as_defended(self):
        """A refused call is infrastructure. Counting it as an attack that
        did not fire would make a flaky network look like a defence."""
        r = self.run_with([row("a", False, api_failure=True), row("a", True)])
        self.assertEqual(carry_forward.fired_by_spec(r.dir)["a"], [True])

    def test_non_run_records_are_ignored(self):
        r = self.run_with([row("a", True, kind="attempt"), row("a", False)])
        self.assertEqual(carry_forward.fired_by_spec(r.dir)["a"], [False])

    def test_it_reads_the_control_arm_and_not_a_treatment_arm(self):
        """The selection must be made before any treatment. Reading a
        compressed arm would select on the outcome of the thing run 4 is
        measuring, which is the one methodological error here that would
        invalidate the result rather than just weaken it."""
        r = Run([row("a", True, condition="unconditional")],
                condition="unconditional")
        self.addCleanup(r.cleanup)
        with self.assertRaises(FileNotFoundError):
            carry_forward.fired_by_spec(r.dir)


class Selecting(unittest.TestCase):

    def setUp(self):
        self.specs = [spec("bk_1", "banking/injection_task_0", "tool_knowledge"),
                      spec("bk_2", "banking/injection_task_0", "tool_knowledge"),
                      spec("bk_3", "banking/injection_task_0", "tool_knowledge"),
                      spec("bk_1", "banking/injection_task_0", "direct")]
        self.fired = {
            "bk_1|banking/injection_task_0|tool_knowledge": [True, True, True],
            "bk_2|banking/injection_task_0|tool_knowledge": [True, False, False],
            "bk_3|banking/injection_task_0|tool_knowledge": [False, False, False],
            "bk_1|banking/injection_task_0|direct": [True, True, True]}

    def test_an_attack_that_never_fired_is_dropped(self):
        kept = carry_forward.carry(self.specs, self.fired, wrappers=None)
        self.assertNotIn("bk_3", [s["task_id"] for s in kept])

    def test_an_intermittent_attack_is_kept_by_default(self):
        """Two in three is a real attack. Dropping it would hide a compressor
        whose effect is to make a reliable attack intermittent, which is
        precisely the kind of effect worth finding."""
        kept = carry_forward.carry(self.specs, self.fired, wrappers=None)
        self.assertIn("bk_2", [s["task_id"] for s in kept])

    def test_require_all_repeats_is_available_and_not_the_default(self):
        strict = carry_forward.carry(self.specs, self.fired, wrappers=None,
                                     require_all=True)
        self.assertNotIn("bk_2", [s["task_id"] for s in strict])
        self.assertLess(len(strict),
                        len(carry_forward.carry(self.specs, self.fired,
                                                wrappers=None)))

    def test_the_wrapper_cut_is_separate_from_the_fired_cut(self):
        kept = carry_forward.carry(self.specs, self.fired,
                                   wrappers=("tool_knowledge",))
        self.assertEqual({s["wrapper"] for s in kept}, {"tool_knowledge"})
        self.assertNotIn("direct", {s["wrapper"] for s in kept})

    def test_no_wrapper_filter_keeps_every_wrapper(self):
        kept = carry_forward.carry(self.specs, self.fired, wrappers=())
        self.assertIn("direct", {s["wrapper"] for s in kept})

    def test_a_spec_the_run_never_saw_is_dropped(self):
        """Not kept on the benefit of the doubt. A spec with no rows was not
        measured, and an unmeasured attack carried into run 4 would have no
        control value to be compared against."""
        extra = self.specs + [spec("bk_9", "banking/injection_task_1",
                                   "tool_knowledge")]
        kept = carry_forward.carry(extra, self.fired, wrappers=None)
        self.assertNotIn("bk_9", [s["task_id"] for s in kept])

    def test_order_is_preserved(self):
        kept = carry_forward.carry(self.specs, self.fired, wrappers=None)
        self.assertEqual([s["task_id"] for s in kept],
                         [s["task_id"] for s in self.specs
                          if s["task_id"] in {x["task_id"] for x in kept}])


class TheWrittenFile(unittest.TestCase):

    def build(self, *args):
        temp = tempfile.TemporaryDirectory(prefix="carry-out-")
        self.addCleanup(temp.cleanup)
        run = Run([row("bk_1|banking/injection_task_0|tool_knowledge", True),
                   row("bk_2|banking/injection_task_0|direct", True),
                   row("bk_3|banking/injection_task_0|tool_knowledge", False)])
        self.addCleanup(run.cleanup)
        specs_in = Path(temp.name) / "in.json"
        specs_in.write_text(json.dumps({"suite_version": "v1.2.1", "specs": [
            spec("bk_1", "banking/injection_task_0", "tool_knowledge"),
            spec("bk_2", "banking/injection_task_0", "direct"),
            spec("bk_3", "banking/injection_task_0", "tool_knowledge")]}))
        out = Path(temp.name) / "out.json"
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            carry_forward.main(["--from-run", str(run.dir), "--specs",
                                str(specs_in), "--out", str(out), *args])
        return json.loads(out.read_text())

    def test_it_records_the_rule_it_selected_by(self):
        """Not a code property, so it is asserted on the output. Without this
        block someone reads run 4's rate next to run 3's 26% and the
        denominators silently differ."""
        payload = self.build()
        carried = payload["carried_from"]
        self.assertIn("condition=none", carried["rule"])
        self.assertIn("CONDITIONAL", carried["note"])
        self.assertEqual(carried["experiment_id"], "abc123")
        self.assertEqual(carried["specs_before"], 3)
        self.assertEqual(carried["specs_after"], 1)

    def test_the_rest_of_the_payload_survives(self):
        """attack_run reads suite_version and the coverage block out of this
        file, so a selection that dropped them would break the runner."""
        self.assertEqual(self.build()["suite_version"], "v1.2.1")

    def test_the_wrapper_cut_is_visible_in_the_file(self):
        payload = self.build("--wrappers", "direct")
        self.assertEqual([s["wrapper"] for s in payload["specs"]], ["direct"])
        self.assertEqual(payload["carried_from"]["wrappers"], ["direct"])

    def test_nothing_firing_is_an_error_not_an_empty_run(self):
        run = Run([row("bk_1|banking/injection_task_0|tool_knowledge", False)])
        self.addCleanup(run.cleanup)
        temp = tempfile.TemporaryDirectory(prefix="carry-empty-")
        self.addCleanup(temp.cleanup)
        specs_in = Path(temp.name) / "in.json"
        specs_in.write_text(json.dumps({"specs": [
            spec("bk_1", "banking/injection_task_0", "tool_knowledge")]}))
        with self.assertRaises(SystemExit):
            carry_forward.main(["--from-run", str(run.dir), "--specs",
                                str(specs_in), "--out",
                                str(Path(temp.name) / "out.json")])


class AgainstTheRealRun(unittest.TestCase):
    """The file this actually produced, if run 3 is on disk."""

    def setUp(self):
        root = Path(__file__).resolve().parent.parent
        self.path = root / "tasks" / "attack_specs_run4.json"
        if not self.path.exists():
            self.skipTest("run 4 specs not generated")
        self.payload = json.loads(self.path.read_text())

    def test_every_carried_spec_is_runnable(self):
        import attack_run
        specs, _, _ = attack_run.load_specs(self.path)
        self.assertTrue(specs)
        self.assertEqual(len(specs), len(self.payload["specs"]))

    def test_it_is_smaller_than_the_full_set_and_says_by_how_much(self):
        carried = self.payload["carried_from"]
        self.assertLess(carried["specs_after"], carried["specs_before"])
        self.assertEqual(carried["specs_after"], len(self.payload["specs"]))

    def test_both_tool_relations_survive_the_cut(self):
        """The allowlist prediction needs both levels. If the cut left only
        tool_novel there would be nothing to compare the allowlist against
        and that arm would have to be dropped from the claim."""
        self.assertEqual({s["tool_relation"] for s in self.payload["specs"]},
                         {"tool_novel", "tool_reuse"})

    def test_more_than_one_suite_survives(self):
        self.assertGreater(len({s["suite"] for s in self.payload["specs"]}), 1)


if __name__ == "__main__":
    unittest.main()
