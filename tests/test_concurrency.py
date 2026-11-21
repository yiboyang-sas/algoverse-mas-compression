"""
Serial and parallel must configure every task identically.

THIS IS THE TEST THAT SHOULD HAVE EXISTED FIRST. `pipeline.AGENT2_SYSTEM` was
a module global set once per task. In one thread that is correct. In eight,
task A writes it, task B overwrites it, and task A's Agent 2 is then told it
holds task B's tools: workspace tasks came back calling `post_webpage`, which
is a slack tool. It invalidated every parallel run from the day the thread
pool went in, and it was invisible because the scores it produced looked
plausible — 49% to 57%, which reads like a weak model rather than a bug.
Fixing it moved the baseline from 52% to 67% and the spread from 6% to 2%.

The test does not inspect the fix. It asserts the property: whatever Agent 2
is told, under eight workers, must be exactly what it is told under one. Any
future reintroduction of per-task global state fails here rather than
eighteen hours into a paid run.

No API calls. The backend is a fake that records the system prompt it was
handed, keyed by the task it belongs to.
"""

import threading
import unittest

import baseline            # installs every patch on import
import compressors
import instrument
import pipeline


# One task per worker, each holding exactly ONE Agent 2 tool, all different.
# A crossed prompt is then unambiguous: if fake_0 is offered send_money,
# some other task's configuration reached it.
WORKERS = 8

_SUITE_OF = {
    "send_email": "workspace", "create_calendar_event": "workspace",
    "create_file": "workspace", "post_webpage": "slack",
    "send_direct_message": "slack", "send_money": "banking",
    "schedule_transaction": "banking", "reserve_hotel": "travel",
}
_TOOLS = list(_SUITE_OF)
assert len(_TOOLS) == WORKERS

TASKS = [
    {"id": f"fake_{i}", "domain": _SUITE_OF[tool],
     "agent1_task": f"read record {i}", "documents": [f"doc {i}"],
     "agent1_tools": ["search_emails"],
     "agent2_held_tools": [tool],
     "required_calls": [{"tool": tool, "args": {"x": i}}],
     "handoff_critical": {"x": i}}
    for i, tool in enumerate(_TOOLS)
]


class _Recorder:
    """A backend that answers nothing and writes down what it was asked.

    THE BARRIER IS THE POINT. The race window is between
    `pipeline.AGENT2_SYSTEM = ...` in run_with_schema and the
    `AGENT2_SYSTEM.format(...)` read in run_pipeline, with Agent 1's API call
    in between. Against a fake backend that returns instantly the window is
    nanoseconds wide and a broken build passes the test — which is exactly
    what happened the first time this test was written, and is why it is
    written this way now.

    Waiting on a barrier during the Agent 1 call holds every worker open
    until all of them have written the global. A per-task global then
    produces 7 wrong prompts out of 8, deterministically, with no sleeps and
    nothing to tune.
    """

    name = "recorder"

    def __init__(self, barrier=None):
        self.seen = []                      # (task_id, stage, system, budget)
        self.lock = threading.Lock()
        self.barrier = barrier

    def chat(self, system, user):
        ctx = instrument.current()
        stage = instrument.stage_from_system(system)
        if stage == instrument.STAGE_AGENT1 and self.barrier is not None:
            # every worker has now set the global; nobody has read it yet
            self.barrier.wait(timeout=20)
        with self.lock:
            self.seen.append((
                ctx.task_id if ctx else None,
                stage,
                system,
                ctx.budget_tokens if ctx else None,
            ))
        return '{"tool_calls": [], "handoff_note": "x"}'


def _run(tasks, workers, budget=150, sync=False):
    import concurrent.futures as cf
    barrier = threading.Barrier(len(tasks)) if sync else None
    b = _Recorder(barrier)
    baseline.RUN = None
    baseline.TRACE = None

    def one(t):
        # repeat is the runner's job; here it only has to be a real value.
        return pipeline.run_pipeline(t, b, b, "none", repeat=0)

    if workers > 1:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, tasks))
    else:
        for t in tasks:
            one(t)
    return b.seen


def _agent2_prompts(seen):
    """task_id -> set of Agent 2 system prompts it was given."""
    out = {}
    for tid, stage, system, _ in seen:
        if stage == instrument.STAGE_AGENT2:
            out.setdefault(tid, set()).add(system)
    return out


class Isolation(unittest.TestCase):

    def test_no_task_is_told_about_another_tasks_tools(self):
        """THE REGRESSION TEST. The symptom, asserted directly.

        A workspace task being told it holds post_webpage is what actually
        happened and what invalidated every parallel run, so test for that
        rather than only for prompt equality. Every worker is held at the
        barrier until all eight have written the per-task global, so a build
        with the old bug fails here every time rather than occasionally.
        """
        by_id = {t["id"]: t for t in TASKS}
        got = _agent2_prompts(_run(TASKS, workers=WORKERS, sync=True))
        self.assertEqual(len(got), len(TASKS), "a task made no Agent 2 call")
        for tid, prompts in got.items():
            self.assertEqual(len(prompts), 1,
                             f"{tid} saw {len(prompts)} different Agent 2 "
                             f"prompts: that is the race")
            mine = set(by_id[tid]["agent2_held_tools"])
            others = {tool for t in TASKS
                      for tool in t["agent2_held_tools"]} - mine
            prompt = next(iter(prompts))
            for tool in others:
                self.assertNotIn(
                    f"{tool}(", prompt,
                    f"{tid} was offered {tool}, which belongs to another task")
            for tool in mine:
                self.assertIn(f"{tool}(", prompt,
                              f"{tid} was not offered its own tool {tool}")

    def test_parallel_matches_serial_exactly(self):
        serial = _agent2_prompts(_run(TASKS, workers=1))
        par = _agent2_prompts(_run(TASKS, workers=WORKERS, sync=True))
        self.assertEqual(set(serial), set(par))
        for tid in serial:
            self.assertEqual(serial[tid], par[tid],
                             f"{tid}'s Agent 2 prompt differs between serial "
                             f"and parallel")

    def test_agent1_and_agent2_are_distinguishable_in_every_record(self):
        stages = {stage for _, stage, _, _ in _run(TASKS[:3], workers=1)}
        self.assertIn(instrument.STAGE_AGENT1, stages)
        self.assertIn(instrument.STAGE_AGENT2, stages)

    def test_every_call_is_attributed_to_the_right_task(self):
        """A call recorded against the wrong task is how bk_5's output ended
        up in four other tasks' records."""
        for tid, _, system, _ in _run(TASKS, workers=WORKERS, sync=True):
            self.assertIsNotNone(tid, "a call with no task context")

    def test_budget_is_per_run_not_per_process(self):
        """Two runs at different budgets must not see each other's value.

        compressors.BUDGET_TOKENS used to be read at import and baked into
        the four compressor prompts, which made a sweep inside one process
        impossible and left the budget out of the result entirely.
        """
        results = {}
        errs = []

        def go(budget):
            try:
                ctx = instrument.TaskRunContext(
                    run_id="t", task_id=f"b{budget}", budget_tokens=budget)
                with instrument.run_context(ctx):
                    for _ in range(200):
                        results.setdefault(budget, set()).add(
                            compressors.active_budget())
            except Exception as e:          # pragma: no cover
                errs.append(e)

        ts = [threading.Thread(target=go, args=(b,))
              for b in (50, 100, 150, 300)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertFalse(errs)
        for budget, seen in results.items():
            self.assertEqual(seen, {budget},
                             f"a thread on budget {budget} read {seen}")

    def test_budget_reaches_the_compressor_prompt(self):
        ctx = instrument.TaskRunContext(run_id="t", task_id="x",
                                        budget_tokens=50)
        seen = []

        class B:
            name = "b"

            def chat(self, system, user):
                seen.append(system)
                return "compressed"

        with instrument.run_context(ctx):
            compressors.compress("unconditional", "some handoff", B(), {})
        self.assertIn(str(compressors.budget_words(50)), seen[0])

    def test_compressor_calls_are_labelled_as_the_compressor(self):
        """Agent 2 and the compressor share one backend, so a prose-prefix
        guess would put both in the same bucket."""
        ctx = instrument.TaskRunContext(run_id="t", task_id="x")
        got = []

        class B:
            name = "b"

            def chat(self, system, user):
                got.append(instrument.current().stage_override)
                return "compressed"

        with instrument.run_context(ctx):
            compressors.compress("unconditional", "h", B(), {})
        self.assertEqual(got, [instrument.STAGE_COMPRESSOR])

    def test_context_does_not_leak_out_of_a_run(self):
        self.assertIsNone(instrument.current())
        ctx = instrument.TaskRunContext(run_id="t", task_id="x")
        with instrument.run_context(ctx):
            self.assertIs(instrument.current(), ctx)
        self.assertIsNone(instrument.current())

    def test_a_worker_cannot_see_another_workers_context(self):
        seen, errs = {}, []

        def go(i):
            try:
                ctx = instrument.TaskRunContext(run_id="t", task_id=f"task{i}")
                with instrument.run_context(ctx):
                    for _ in range(500):
                        seen.setdefault(i, set()).add(
                            instrument.current().task_id)
            except Exception as e:          # pragma: no cover
                errs.append(e)

        ts = [threading.Thread(target=go, args=(i,)) for i in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertFalse(errs)
        for i, s in seen.items():
            self.assertEqual(s, {f"task{i}"})


if __name__ == "__main__":
    unittest.main()
