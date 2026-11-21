"""
The traces, which are the only evidence a finished run leaves behind.

Two properties matter. One: a run killed halfway must still have usable
lines on disk — the out-of-credit run was interrupted and lost everything,
and that is why nobody could tell for two days whether the model had got
worse or the account had run dry. Two: eight workers writing at once must
not interleave into invalid JSON, because a corrupt line is indistinguishable
from a line that was never written.

Also here: the transport/content distinction. `succeeded` is about whether
the call went through, not whether the answer was any good. A model that
refuses in prose succeeded. Collapsing the two is what turned an account
recovering its credit into a baseline climbing from 45% to 73%.
"""

import json
import os
import tempfile
import threading
import time
import unittest

import instrument
import tracing


class Writer(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.w = tracing.TraceWriter(self.dir)

    def tearDown(self):
        self.w.close()

    def lines(self, stream="calls"):
        p = os.path.join(self.dir, f"{stream}.jsonl")
        if not os.path.exists(p):
            return []
        with open(p) as fh:
            return [json.loads(x) for x in fh if x.strip()]

    def test_creates_the_directory(self):
        d = os.path.join(self.dir, "deeper", "still")
        tracing.TraceWriter(d).close()
        self.assertTrue(os.path.isdir(d))

    def test_one_line_per_record(self):
        for i in range(5):
            self.w.write("calls", {"i": i})
        self.assertEqual([r["i"] for r in self.lines()], list(range(5)))

    def test_readable_before_the_writer_closes(self):
        """A killed run must leave evidence, so every line is flushed."""
        self.w.write("calls", {"i": 0})
        self.assertEqual(len(self.lines()), 1)

    def test_streams_are_separate_files(self):
        self.w.write("calls", {"a": 1})
        self.w.write("runs", {"b": 2})
        self.assertEqual(len(self.lines("calls")), 1)
        self.assertEqual(len(self.lines("runs")), 1)

    def test_eight_writers_produce_valid_lines(self):
        def go(i):
            for j in range(50):
                self.w.write("calls", {"who": i, "j": j,
                                       "text": "x" * 500})

        ts = [threading.Thread(target=go, args=(i,)) for i in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        rows = self.lines()
        self.assertEqual(len(rows), 400)
        self.assertEqual(len({(r["who"], r["j"]) for r in rows}), 400)

    def test_unserializable_values_do_not_lose_the_line(self):
        """default=str. A record that raises on dump is a record that is gone,
        and the run that needed it is the run that already failed."""
        self.w.write("calls", {"when": time.gmtime(), "obj": object()})
        self.assertEqual(len(self.lines()), 1)


class CallRecords(unittest.TestCase):

    def rec(self, **kw):
        base = dict(run_id="r", task_id="bk_5", stage="agent1",
                    model="gpt-4o-mini", temperature=0.0,
                    system="You are Agent 1", user="TASK: x",
                    raw='{"tool_calls": []}', started=1.0, finished=2.5)
        base.update(kw)
        return tracing.call_record(**base)

    def test_a_record_exists_because_a_call_was_made(self):
        self.assertTrue(self.rec()["requested"])

    def test_transport_success_is_not_content_success(self):
        refused = self.rec(raw="I will not do that.")
        self.assertTrue(refused["succeeded"],
                        "a prose refusal is a successful call")
        self.assertIsNone(refused["api_error"])

    def test_api_failure_is_marked_as_a_failure(self):
        r = self.rec(raw=None, api_error="HTTPError: 429 no credits")
        self.assertFalse(r["succeeded"])
        self.assertEqual(r["parse_status"], "no_output")

    def test_parse_status(self):
        self.assertEqual(self.rec()["parse_status"], "ok")
        self.assertEqual(self.rec(parse_error="x")["parse_status"], "unparsed")
        self.assertEqual(self.rec(raw=None)["parse_status"], "no_output")

    def test_the_verbatim_output_is_kept(self):
        raw = 'PLAN: call send_money with recipient = XX00\nEVIDENCE: ...'
        self.assertEqual(self.rec(raw=raw)["raw_output"], raw)

    def test_duration_is_recorded(self):
        self.assertAlmostEqual(self.rec()["duration_s"], 1.5)

    def test_retries_are_recorded(self):
        self.assertEqual(self.rec(retries=3)["retries"], 3)

    def test_serializable(self):
        json.dumps(self.rec(), default=str)


class TracedBackend(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.w = tracing.TraceWriter(self.dir)

    def tearDown(self):
        self.w.close()

    def ctx(self, **kw):
        base = dict(run_id="r", task_id="t", trace=self.w)
        base.update(kw)
        return instrument.TaskRunContext(**base)

    def test_installs_once(self):
        class B:
            calls = 0

            def chat(self, s, u):
                return "ok"

        b = B()
        first = instrument.traced(b).chat
        second = instrument.traced(b).chat
        self.assertIs(first, second, "double-wrapping would double-count")

    def test_records_the_call(self):
        class B:
            def chat(self, s, u):
                return "the answer"

        b = instrument.traced(B())
        c = self.ctx()
        with instrument.run_context(c):
            b.chat("You are Agent 1", "TASK: x")
        self.assertEqual(len(c.calls), 1)
        self.assertEqual(c.calls[0]["raw_output"], "the answer")
        self.assertEqual(c.calls[0]["stage"], "agent1")
        self.assertEqual(c.calls[0]["task_id"], "t")

    def test_records_an_exception_and_re_raises_it(self):
        class B:
            def chat(self, s, u):
                raise TimeoutError("too slow")

        b = instrument.traced(B())
        c = self.ctx()
        with instrument.run_context(c):
            with self.assertRaises(TimeoutError):
                b.chat("You are Agent 2", "handoff")
        self.assertEqual(len(c.calls), 1)
        self.assertFalse(c.calls[0]["succeeded"])
        self.assertIn("TimeoutError", c.calls[0]["api_error"])

    def test_works_with_no_context_at_all(self):
        """selftest and the mock path run outside a context; they must not
        crash, they just leave no trace."""
        class B:
            def chat(self, s, u):
                return "ok"

        b = instrument.traced(B())
        self.assertEqual(b.chat("You are Agent 1", "x"), "ok")

    def test_retry_count_reaches_the_record(self):
        """The wrapper goes on OUTSIDE the retry wrapper.

        Order matters and this test pins it. In main the backend is built as
        traced(_Retrying(OpenAIBackend)), so by the time `traced` sees a
        result the retries have already happened inside it — which is why the
        count is taken as a delta across the call rather than read after it.
        Two calls in one run must not each report the other's retries.
        """
        class Flaky:
            """Stands in for _Retrying: retries inside, reports as it goes."""

            def chat(self, s, u):
                for _ in range(2):
                    instrument.note_retry()
                return "ok"

        b = instrument.traced(Flaky())
        c = self.ctx()
        with instrument.run_context(c):
            b.chat("You are Agent 1", "x")
            b.chat("You are Agent 2", "x")
        self.assertEqual(c.calls[0]["retries"], 2)
        self.assertEqual(c.calls[1]["retries"], 2,
                         "the second call must not inherit the first's count")
        self.assertEqual(c.retries, 4, "the run total is still cumulative")


if __name__ == "__main__":
    unittest.main()
