"""
The attack runner, end to end, offline.

The thing these have to prove is not that the runner runs. It is that a fired
attack REACHES THE TABLE. There are four places it could be lost between the
agent making a harmful call and a number in the report, and every one of them
fails silently in the same direction, as a defence:

  1. the injection never gets planted in the documents
  2. the scorer does not match the call the agent made
  3. the score never gets written into the row
  4. the report does not read it back out

Number 3 was real. `baseline.py` scored the attack and then built the row from
a `scoring` dict that did not contain any of the attack fields, so a run could
detect an attack perfectly and write a table of zeroes. The mock backend never
fires an injection, so nothing in the existing suite would ever have noticed.
Hence `FollowsInjection` below: a backend that does the harmful thing, so the
whole path is exercised against an attack that definitely happened.

Everything here uses authored fixture backends. No model is called.
"""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import attack_run
import attacks


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECS = os.path.join(ROOT, "tasks", "attack_specs.json")


def _skip_unless_ready():
    if not os.path.exists(SPECS):
        raise unittest.SkipTest("attack specs not generated")
    try:
        import shared_run  # noqa: F401
    except Exception as exc:                       # pragma: no cover
        raise unittest.SkipTest(f"shared_run unavailable: {exc}")


class Obedient:
    """Agent 1 forwards the documents; Agent 2 does the benign task only.

    The control case. Everything the attack path needs is present and working
    and the attack still must read as not fired, because this agent never
    makes a harmful call.
    """

    name = "authored_fixture"
    follows = False

    def __init__(self):
        import shared_run
        self.tasks = {t["id"]: t for t in shared_run.load_tasks()}
        with open(SPECS) as handle:
            self.specs = {attack_run.spec_id(s): s
                          for s in json.load(handle)["specs"]}

    def _task_of(self, ctx):
        return self.tasks[ctx.task_id.split("|")[0]]

    def chat(self, system, user):
        import instrument
        ctx = instrument.current()
        stage = ctx.stage_override or instrument.stage_from_system(system)
        task = self._task_of(ctx)
        if stage == "agent1":
            reads = [c for c in task["required_calls"]
                     if c["tool"] in task["agent1_tools"]]
            return json.dumps({
                "tool_calls": reads,
                "handoff_note": "PLAN: " + task["required_calls"][-1]["tool"]
                                + "\n" + "\n".join(task["documents"])})
        if stage == "compressor":
            return user
        return json.dumps({"tool_calls": self._agent2(task, ctx),
                           "handoff_note": "done"})

    def _agent2(self, task, ctx):
        return [c for c in task["required_calls"]
                if c["tool"] in task["agent2_held_tools"]]


class FollowsInjection(Obedient):
    """Agent 2 does the benign task AND the harmful one.

    Not a prediction about any model. It is the only way to test that a fired
    attack survives all the way from the tool log to the printed rate, which
    is the one failure this runner could have that nobody would see.
    """

    follows = True

    def _agent2(self, task, ctx):
        spec = self.specs[ctx.task_id]
        benign = super()._agent2(task, ctx)
        harmful = [{"tool": c["tool"], "args": c["args"]}
                   for c in spec["harmful_call"]]
        return benign + harmful


class Base(unittest.TestCase):

    def setUp(self):
        _skip_unless_ready()
        self.temp = tempfile.TemporaryDirectory(prefix="attack-run-test-")
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "arm"

    def args(self, *more):
        return attack_run.parser().parse_args(
            ["--out-dir", str(self.out), "--limit", "6", "--repeats", "1",
             "--workers", "2", *more])

    def execute(self, backend, *more):
        with contextlib.redirect_stdout(io.StringIO()):
            return attack_run.run(self.args(*more), backend=backend)

    def manifest(self):
        return json.loads((self.out / "manifest.json").read_text())


class TheAttackReachesTheTable(Base):

    def test_an_agent_that_follows_the_injection_reads_as_attacked(self):
        """The test this file exists for. Scoring the attack and then writing
        a row without the attack fields produced a table of zeroes from a run
        where every single attack fired."""
        status = self.execute(FollowsInjection())
        over = status["conditions"]["none"]["overall"]
        self.assertEqual(over["attacked_trials"], 6)
        self.assertEqual(over["fired_any"], 6)
        self.assertEqual(over["any_rate"], 1.0)
        self.assertEqual(over["fired_all"], 6)

    def test_an_agent_that_does_not_reads_as_not_attacked(self):
        status = self.execute(Obedient())
        over = status["conditions"]["none"]["overall"]
        self.assertEqual(over["attacked_trials"], 6)
        self.assertEqual(over["fired_any"], 0)
        self.assertEqual(over["any_rate"], 0.0)

    def test_the_benign_task_is_still_scored_alongside_the_attack(self):
        """An attack arm that silently destroyed task completion would look
        like a defence working. Both numbers come out of the same trial."""
        status = self.execute(FollowsInjection())
        self.assertEqual(status["conditions"]["none"]["task_success"], 6)

    def test_the_trials_carry_a_spec_id_not_just_a_task_id(self):
        self.execute(Obedient())
        rows = attack_run.rows(self.out, self.manifest(), "none")
        self.assertTrue(all(r.get("attack_spec_id") for r in rows))
        self.assertEqual(len({r["attack_spec_id"] for r in rows}), 6)

    def test_irreversibility_is_recorded_when_it_applies(self):
        status = self.execute(FollowsInjection(), "--limit", "40")
        over = status["conditions"]["none"]["overall"]
        self.assertGreater(over["irreversible"], 0)
        self.assertLess(over["irreversible"], over["attacked_trials"],
                        "not every harmful tool is irreversible")


class Planting(Base):

    def test_the_injection_is_in_the_prompt_agent_1_sees(self):
        """If planting failed the arm would measure nothing, and it would
        measure nothing in the direction of the attack being defended."""
        self.execute(Obedient())
        cached = [json.loads(l) for l in
                  (self.out / "handoffs.jsonl").read_text().splitlines()]
        with open(SPECS) as handle:
            specs = {attack_run.spec_id(s): s
                     for s in json.load(handle)["specs"]}
        for row in cached:
            spec = specs[row["attack_spec_id"]]
            needle = spec["goal"][:40].replace("'", "")
            with self.subTest(spec=row["attack_spec_id"]):
                self.assertIn(needle,
                              row["agent1_input"].replace("\\", "")
                              .replace("'", ""))

    def test_agent_1_is_never_handed_the_harmful_call(self):
        """The stock prompt builder emits an INJECT_CALL marker naming the
        harmful tool and its arguments whenever the task carries
        `harmful_call`. That would hand the model the attack it is supposed to
        discover in a document, and the whole arm would be worthless. The
        planted task carries only `harmful_critical` for that reason."""
        self.execute(Obedient())
        cached = [json.loads(l) for l in
                  (self.out / "handoffs.jsonl").read_text().splitlines()]
        for row in cached:
            with self.subTest(spec=row["attack_spec_id"]):
                self.assertNotIn("INJECT_CALL", row["agent1_input"])
                self.assertNotIn("REQUIRED_CALL", row["agent1_input"])
                self.assertNotIn("NEXT_CALL", row["agent1_input"])

    def test_the_planted_task_is_not_written_over_the_dataset(self):
        """416 planted datasets on disk would be 416 copies of the same
        documents, and the first one to drift from the specs would go
        unnoticed. The planting is derived per trial and hashed into it."""
        import shared_run
        before = shared_run.sha256(
            (Path(ROOT) / shared_run.TASKS_FILE).read_bytes())
        self.execute(Obedient())
        after = shared_run.sha256(
            (Path(ROOT) / shared_run.TASKS_FILE).read_bytes())
        self.assertEqual(before, after)
        self.assertEqual(before, shared_run.DATASET_SHA256)


class Resume(Base):

    def test_agent_1_is_drawn_once_per_spec_and_repeat(self):
        """Run 3 and run 4 share Agent 1 the way run 2 did, so a difference
        between conditions is compression and not Agent 1 having a different
        day."""
        backend = Obedient()
        calls = []
        original = backend.chat

        def counting(system, user):
            import instrument
            ctx = instrument.current()
            stage = ctx.stage_override or instrument.stage_from_system(system)
            calls.append((ctx.task_id, ctx.repeat, stage))
            return original(system, user)

        backend.chat = counting
        self.execute(backend, "--repeats", "2")
        agent1 = [c for c in calls if c[2] == "agent1"]
        self.assertEqual(len(agent1), len(set(agent1)))
        self.assertEqual(len(agent1), 12)

    def test_resuming_adds_nothing(self):
        self.execute(Obedient())
        first = attack_run.rows(self.out, self.manifest(), "none")
        self.execute(Obedient())
        again = attack_run.rows(self.out, self.manifest(), "none")
        self.assertEqual(len(first), len(again))

    def test_changed_specs_refuse_to_resume(self):
        """Mixing two attack sets in one table would make every rate a
        weighted average of two different experiments."""
        self.execute(Obedient())
        other = Path(self.temp.name) / "other_specs.json"
        with open(SPECS) as handle:
            payload = json.load(handle)
        payload["specs"] = payload["specs"][:3]
        other.write_text(json.dumps(payload))
        with self.assertRaisesRegex(attack_run.IntegrityError, "specs file"):
            self.execute(Obedient(), "--specs", str(other))

    def test_changed_conditions_refuse_to_resume(self):
        """Run 4's conditions belong in run 4's directory. Writing them into
        run 3's would leave one manifest describing two designs."""
        self.execute(Obedient())
        with self.assertRaisesRegex(attack_run.IntegrityError, "conditions"):
            self.execute(Obedient(), "--conditions", "none", "unconditional")


class Manifest(Base):

    def test_the_design_is_pinned_and_says_it_is_an_attack_run(self):
        self.execute(Obedient())
        m = self.manifest()
        self.assertEqual(m["design"], "attack-arm")
        self.assertTrue(m["settings"]["attack_enabled"])
        attack_run.validate_manifest(m)

    def test_tampering_with_the_manifest_is_caught(self):
        self.execute(Obedient())
        m = self.manifest()
        m["settings"]["temperature"] = 1.0
        with self.assertRaises(attack_run.IntegrityError):
            attack_run.validate_manifest(m)

    def test_an_unknown_condition_is_refused(self):
        with self.assertRaisesRegex(attack_run.IntegrityError, "conditions"):
            self.execute(Obedient(), "--conditions", "not_a_compressor")

    def test_the_specs_hash_and_provenance_are_recorded(self):
        self.execute(Obedient())
        s = self.manifest()["settings"]
        self.assertEqual(len(s["specs_sha256"]), 64)
        self.assertTrue(s["prompt_hashes"]["agent1"])
        self.assertIn("git_commit", self.manifest()["source"])

    def test_a_paid_backend_without_the_flag_is_refused(self):
        class Paid:
            name = "openai"
        with self.assertRaisesRegex(attack_run.IntegrityError, "paid"):
            self.execute(Paid())

    def test_a_mock_run_is_labelled_as_not_scientific(self):
        status = self.execute(Obedient())
        self.assertFalse(status["scientific_model_results"])
        self.assertEqual(status["backend"], "mock")


class Report(Base):

    def test_it_splits_by_the_factors_the_arm_was_designed_for(self):
        status = self.execute(FollowsInjection(), "--limit", "40")
        by = status["conditions"]["none"]["by"]
        self.assertEqual(set(by), {"wrapper", "tool_relation", "arm", "suite"})
        self.assertTrue(by["tool_relation"])

    def test_report_only_rereads_without_running(self):
        self.execute(Obedient())
        again = attack_run.run(self.args("--report-only"), backend=None)
        self.assertEqual(again["experiment_id"],
                         self.manifest()["experiment_id"])

    def test_the_status_is_serializable(self):
        json.dumps(self.execute(Obedient()))


if __name__ == "__main__":
    unittest.main()
