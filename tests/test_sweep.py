"""
The budget sweep.

It decides the n the paper reports, so the two definitions it combines have
to be exactly right: EXPOSED is about the handoff against the budget,
RELIABLE is about the task completing without compression, and the analysis
set is the intersection. Getting either wrong changes the headline number
without changing anything that was measured.
"""

import json
import os
import tempfile
import unittest

import sweep
from config import PRIMARY_CER_MIN


def write_run(d, rows):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "runs.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return d


def row(task, suite, handoff, ok, repeat=0, **kw):
    r = {"task_id": task, "suite": suite, "repeat": repeat,
         "precompression_handoff_tokens": handoff, "task_success": ok,
         "token_count_method": "tiktoken:o200k_base"}
    r.update(kw)
    return r


class Loading(unittest.TestCase):

    def test_median_handoff_across_repeats(self):
        """Not the last one. sl_20 came back 340, 582 and 12,267 tokens on
        three identical runs; the last draw is not the task."""
        with tempfile.TemporaryDirectory() as d:
            write_run(d, [row("a", "slack", 340, True, 0),
                          row("a", "slack", 582, True, 1),
                          row("a", "slack", 12267, True, 2)])
            median, _, _, _, _ = sweep.load([d])
            self.assertEqual(median["a"], 582)

    def test_reliable_means_every_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d, [row("a", "slack", 900, True, 0),
                          row("a", "slack", 900, True, 1),
                          row("b", "slack", 900, True, 0),
                          row("b", "slack", 900, False, 1)])
            _, _, reliable, _, _ = sweep.load([d])
            self.assertEqual(reliable, {"a"})

    def test_api_failures_are_not_counted_either_way(self):
        """A refused call is infrastructure. Counting it as a failure would
        shrink the analysis set for a reason that is not about the model."""
        with tempfile.TemporaryDirectory() as d:
            write_run(d, [row("a", "slack", 900, True, 0),
                          {"task_id": "a", "suite": "slack", "repeat": 1,
                           "api_failure": True, "task_success": None}])
            _, _, reliable, total, _ = sweep.load([d])
            self.assertEqual(total["a"], 1)
            self.assertEqual(reliable, {"a"})

    def test_several_run_directories_combine(self):
        with tempfile.TemporaryDirectory() as root:
            a = write_run(os.path.join(root, "r1"),
                          [row("x", "workspace", 900, True)])
            b = write_run(os.path.join(root, "r2"),
                          [row("y", "banking", 600, True)])
            median, suite, reliable, _, _ = sweep.load([a, b])
            self.assertEqual(set(median), {"x", "y"})
            self.assertEqual(suite["y"], "banking")

    def test_a_missing_directory_is_skipped_not_fatal(self):
        median, _, _, _, _ = sweep.load(["/nonexistent/path"])
        self.assertEqual(median, {})

    def test_a_corrupt_line_does_not_lose_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d, [row("a", "slack", 900, True)])
            with open(os.path.join(d, "runs.jsonl"), "a") as fh:
                fh.write("{not json\n")
                fh.write(json.dumps(row("b", "slack", 800, True)) + "\n")
            median, _, _, _, _ = sweep.load([d])
            self.assertEqual(set(median), {"a", "b"})

    def test_mixed_token_methods_are_surfaced(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d, [row("a", "slack", 900, True),
                          row("b", "slack", 900, True,
                              token_count_method="estimated_chars_div_4")])
            *_, method = sweep.load([d])
            self.assertEqual(len(method), 2)


class Table(unittest.TestCase):

    def fixture(self):
        median = {"big": 900, "mid": 300, "small": 90}
        suite = {"big": "workspace", "mid": "slack", "small": "banking"}
        return median, suite

    def test_a_bigger_budget_never_grows_the_analysis_set(self):
        median, suite = self.fixture()
        rows, _ = sweep.table(median, suite, set(median),
                              [50, 100, 200, 400], PRIMARY_CER_MIN)
        sizes = [r["analysis_set"] for r in rows]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_exposure_is_handoff_over_budget(self):
        median, suite = self.fixture()
        # at 200, only `big` clears 1.5x (900/200 = 4.5); mid is 1.5 exactly
        rows, _ = sweep.table(median, suite, set(median), [200],
                              PRIMARY_CER_MIN)
        self.assertEqual(rows[0]["exposed"], 2)

    def test_the_boundary_is_inclusive(self):
        """CER exactly at the threshold is in. The strata are half-open
        [lo, hi), and the primary set starts where `moderate` starts."""
        rows, _ = sweep.table({"t": 150}, {"t": "slack"}, {"t"}, [100],
                              1.5)
        self.assertEqual(rows[0]["exposed"], 1)

    def test_unreliable_tasks_are_excluded_however_exposed(self):
        median, suite = self.fixture()
        rows, _ = sweep.table(median, suite, {"mid"}, [50], PRIMARY_CER_MIN)
        self.assertEqual(rows[0]["exposed"], 3)
        self.assertEqual(rows[0]["analysis_set"], 1)

    def test_suite_columns_add_up(self):
        median, suite = self.fixture()
        rows, suites = sweep.table(median, suite, set(median), [50],
                                   PRIMARY_CER_MIN)
        self.assertEqual(sum(rows[0][s] for s in suites),
                         rows[0]["analysis_set"])


class Floor(unittest.TestCase):

    def test_an_infeasible_task_leaves_the_analysis_set(self):
        """A task whose ANSWER does not fit the budget is impossible, not
        compressed. It drops out for the opposite reason to an unexposed one,
        and counting it either way would be wrong."""
        median = {"heavy": 3000, "light": 900}
        suite = {"heavy": "banking", "light": "workspace"}
        floors = {"heavy": 224, "light": 20}
        rows, _ = sweep.table(median, suite, set(median), [150, 300],
                              PRIMARY_CER_MIN, floors)
        at150, at300 = rows
        self.assertEqual(at150["infeasible"], 1)
        self.assertEqual(at150["analysis_set"], 1)
        self.assertEqual(at300["infeasible"], 0)
        self.assertEqual(at300["analysis_set"], 2)

    def test_no_floors_means_nothing_is_infeasible(self):
        rows, _ = sweep.table({"a": 900}, {"a": "slack"}, {"a"}, [150],
                              PRIMARY_CER_MIN, {})
        self.assertEqual(rows[0]["infeasible"], 0)
        self.assertEqual(rows[0]["analysis_set"], 1)

    def test_the_floor_is_the_heaviest_answer(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as fh:
            json.dump([
                {"id": "light", "agent2_calls": [
                    {"tool": "send_money", "critical": {"amount": 5.0}}]},
                {"id": "heavy", "agent2_calls": [
                    {"tool": "send_money",
                     "critical": {"recipient": "GB29NWBK60161331926819",
                                  "amount": 594.29}}] * 9},
            ], fh)
            path = fh.name
        try:
            f = sweep.answer_floors(path)
            self.assertGreater(f["heavy"], f["light"])
            self.assertGreater(f["heavy"], 50)
        finally:
            os.unlink(path)

    def test_a_missing_task_file_is_not_fatal(self):
        self.assertEqual(sweep.answer_floors("/nonexistent.json"), {})


class EndToEnd(unittest.TestCase):

    def test_main_runs_and_writes_a_csv(self):
        import contextlib, io
        with tempfile.TemporaryDirectory() as root:
            d = write_run(os.path.join(root, "run"),
                          [row("a", "workspace", 1200, True, i)
                           for i in range(3)] +
                          [row("b", "banking", 90, True, i) for i in range(3)])
            out = os.path.join(root, "x.csv")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = sweep.main([d, "--csv", out, "--budgets", "50,150"])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(out))
            text = buf.getvalue()
            self.assertIn("REPORT THE WHOLE SWEEP", text)

    def test_no_runs_is_an_error_not_a_crash(self):
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = sweep.main(["/nonexistent"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
