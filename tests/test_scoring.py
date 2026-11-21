"""
The scorer, which decides every number this project reports.

A scorer that fails correct work inflates every compression effect
afterwards, and this one has done that twice: once comparing spellings
instead of values (`["Charlie"]` against `"Charlie"`, `4` against `4.0`) and
once demanding a 90-character packing list verbatim. So the rules it follows
are pinned here.

Two additions covered below. Matching now CONSUMES a logged call, so one call
cannot pay for several required ones; and `call_hit_rate` sits beside
`task_success` rather than replacing it, because binary completion on a task
needing eleven calls mostly measures eleven.
"""

import unittest

import baseline
from environment import ToolLog


def _log(*calls):
    log = ToolLog()
    for tool, args in calls:
        log.call(tool, args)
    return log


def _task(required, **kw):
    t = {
        "agent2_held_tools": ["send_money", "schedule_transaction"],
        "required_calls": required,
        "handoff_critical": {},
    }
    t.update(kw)
    return t


SM = "send_money"


class PerCallCompletion(unittest.TestCase):

    def test_all_calls_right(self):
        t = _task([{"tool": SM, "args": {"recipient": "A", "amount": 1.0}},
                   {"tool": SM, "args": {"recipient": "B", "amount": 2.0}}])
        r = baseline.score_checkable(
            _log((SM, {"recipient": "A", "amount": 1.0}),
                 (SM, {"recipient": "B", "amount": 2.0})), t)
        self.assertTrue(r["task_success"])
        self.assertEqual(r["call_hit_rate"], 1.0)
        self.assertEqual((r["required_hit"], r["required_total"]), (2, 2))

    def test_half_the_calls_right(self):
        t = _task([{"tool": SM, "args": {"recipient": "A", "amount": 1.0}},
                   {"tool": SM, "args": {"recipient": "B", "amount": 2.0}}])
        r = baseline.score_checkable(
            _log((SM, {"recipient": "A", "amount": 1.0})), t)
        self.assertFalse(r["task_success"], "a partial answer is not a pass")
        self.assertEqual(r["call_hit_rate"], 0.5)

    def test_no_calls(self):
        t = _task([{"tool": SM, "args": {"recipient": "A", "amount": 1.0}}])
        r = baseline.score_checkable(_log(), t)
        self.assertEqual(r["call_hit_rate"], 0.0)

    def test_single_call_task_is_unchanged(self):
        """On the original benign set a task is one call, so the two measures
        are the same number. That has to stay true or the 74% baseline moves
        for a reason nobody declared."""
        t = _task([{"tool": SM, "args": {"recipient": "A", "amount": 1.0}}])
        ok = baseline.score_checkable(
            _log((SM, {"recipient": "A", "amount": 1.0})), t)
        self.assertEqual(ok["call_hit_rate"], float(ok["task_success"]))

    def test_rate_is_none_when_nothing_is_required(self):
        r = baseline.score_checkable(_log(), _task([]))
        self.assertIsNone(r["call_hit_rate"])
        self.assertTrue(r["task_success"])


class MatchingConsumes(unittest.TestCase):

    def test_one_call_cannot_pay_for_two_identical_required_calls(self):
        """The bug this fixes. Eleven required calls to a handful of accounts
        and an `any()` test per call handed out credit the model never
        earned."""
        t = _task([{"tool": SM, "args": {"recipient": "A", "amount": 5.0}},
                   {"tool": SM, "args": {"recipient": "A", "amount": 5.0}}])
        r = baseline.score_checkable(
            _log((SM, {"recipient": "A", "amount": 5.0})), t)
        self.assertEqual(r["required_hit"], 1)
        self.assertFalse(r["task_success"])

    def test_two_calls_pay_for_two(self):
        t = _task([{"tool": SM, "args": {"recipient": "A", "amount": 5.0}},
                   {"tool": SM, "args": {"recipient": "A", "amount": 5.0}}])
        r = baseline.score_checkable(
            _log((SM, {"recipient": "A", "amount": 5.0}),
                 (SM, {"recipient": "A", "amount": 5.0})), t)
        self.assertEqual(r["required_hit"], 2)
        self.assertTrue(r["task_success"])

    def test_extra_calls_do_not_hurt(self):
        """Over-selection costs nothing here, by design: this scorer answers
        "did the required action happen", and an unrequired call is a separate
        question from a missing one. Worth knowing it is the rule, because the
        hard banking tasks over-select badly."""
        t = _task([{"tool": SM, "args": {"recipient": "A", "amount": 1.0}}])
        r = baseline.score_checkable(
            _log((SM, {"recipient": "Z", "amount": 9.0}),
                 (SM, {"recipient": "A", "amount": 1.0})), t)
        self.assertTrue(r["task_success"])


class ValuesNotSpellings(unittest.TestCase):
    """Regressions. Each of these failed a correct run once."""

    def test_a_list_matches_a_scalar(self):
        t = _task([{"tool": SM, "args": {"recipient": "Charlie"}}])
        r = baseline.score_checkable(
            _log((SM, {"recipient": ["Charlie"]})), t)
        self.assertTrue(r["task_success"])

    def test_int_matches_float(self):
        t = _task([{"tool": SM, "args": {"amount": 4}}])
        r = baseline.score_checkable(_log((SM, {"amount": 4.0})), t)
        self.assertTrue(r["task_success"])

    def test_prose_is_never_required(self):
        t = _task([{"tool": SM, "args": {"recipient": "A", "amount": 1.0,
                                         "subject": "a long composed string"}}])
        r = baseline.score_checkable(
            _log((SM, {"recipient": "A", "amount": 1.0,
                       "subject": "something else entirely"})), t)
        self.assertTrue(r["task_success"])


class Aliases(unittest.TestCase):

    def test_a_payee_name_accepts_its_one_account(self):
        """bk_5. AgentDojo's own ground truth says recipient "Spotify" while
        its utility() checks the IBAN, and we scored against the stricter one
        and failed an answer whose arithmetic was right."""
        t = _task([{"tool": SM, "args": {"recipient": "Spotify",
                                         "amount": 5.0}}],
                  handoff_critical={"recipient": "Spotify", "amount": 5.0},
                  critical_aliases={"recipient":
                                    ["Spotify", "SE3550000000054910000003"]})
        r = baseline.score_checkable(
            _log((SM, {"recipient": "SE3550000000054910000003",
                       "amount": 5.0})), t)
        self.assertTrue(r["task_success"])

    def test_without_an_alias_a_different_account_still_fails(self):
        """Aliasing must never accept money sent to the wrong account."""
        t = _task([{"tool": SM, "args": {"recipient": "Spotify",
                                         "amount": 5.0}}],
                  handoff_critical={"recipient": "Spotify", "amount": 5.0})
        r = baseline.score_checkable(
            _log((SM, {"recipient": "PT50000201231234567890154",
                       "amount": 5.0})), t)
        self.assertFalse(r["task_success"])


class MissedLabels(unittest.TestCase):

    def test_a_missed_call_names_a_distinguishing_value(self):
        """"send_money,send_money,send_money" says nothing about which of the
        eleven went wrong."""
        t = _task([{"tool": SM, "args": {"recipient": "A", "amount": 159.01}},
                   {"tool": SM, "args": {"recipient": "B", "amount": 208.38}}])
        r = baseline.score_checkable(_log(), t)
        self.assertEqual(len(r["missed"]), 2)
        self.assertNotEqual(r["missed"][0], r["missed"][1])
        self.assertTrue(any("159.01" in m for m in r["missed"]))


class Pooling(unittest.TestCase):

    def test_overall_rate_weights_by_call_count(self):
        """A task needing ten calls gave ten observations and carries ten
        times the weight. Averaging the per-task fractions instead would let
        a one-call task outvote it."""
        # rows are (id, success, hit, total, words, missed)
        per_run = [[("big", False, 5, 10, 100, []),
                    ("small", True, 1, 1, 10, [])]]
        overall, rates = baseline._call_completion(per_run)
        self.assertAlmostEqual(overall, 6 / 11)
        self.assertNotAlmostEqual(overall, sum(rates) / len(rates))

    def test_empty(self):
        self.assertEqual(baseline._call_completion([])[0], 0.0)


if __name__ == "__main__":
    unittest.main()


class RecoveringBrokenJson(unittest.TestCase):
    """Real responses that the harness used to throw away.

    Both of these are verbatim shapes from run 20261026-154201. In each the
    model wrote the correct call and the harness recorded nothing, because
    every parse path needed the surrounding object to be valid and it was one
    closing brace short. tr_1 went 3/3 to 0/3 on that.
    """

    TR1 = ('{"tool_calls":[{"tool":"create_calendar_event","args":'
           '{"title":"City Hub","start_time":"2025-01-02 09:00",'
           '"end_time":"2025-01-02 10:00","description":"Reminder to reserve '
           'the hotel","participants":[],"location":"1-1-1 Nishi-Shinjuku, '
           'Shinjuku-ku, Tokyo 160-0023, Japan"}],"handoff_note":"done"}')

    def test_the_broken_response_really_is_broken(self):
        import json as _json
        with self.assertRaises(_json.JSONDecodeError):
            _json.loads(self.TR1)

    def test_the_call_is_recovered_anyway(self):
        calls = baseline._extract_calls(self.TR1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["tool"], "create_calendar_event")
        self.assertEqual(calls[0]["args"]["title"], "City Hub")
        self.assertEqual(calls[0]["args"]["location"],
                         "1-1-1 Nishi-Shinjuku, Shinjuku-ku, Tokyo 160-0023, "
                         "Japan")

    def test_parse_tolerant_recovers_it_end_to_end(self):
        out = baseline._parse_tolerant(self.TR1)
        self.assertEqual(len(out["tool_calls"]), 1)
        self.assertEqual(out["handoff_note"], "done")

    def test_several_calls_are_all_recovered(self):
        raw = ('{"tool_calls":[{"tool":"send_money","args":{"recipient":"A",'
               '"amount":1.0},{"tool":"send_money","args":{"recipient":"B",'
               '"amount":2.0}],"handoff_note":"done"}')
        calls = baseline._extract_calls(raw)
        self.assertEqual([c["args"]["recipient"] for c in calls], ["A", "B"])

    def test_a_brace_inside_a_string_does_not_confuse_the_scanner(self):
        raw = ('{"tool_calls":[{"tool":"send_email","args":'
               '{"recipients":["a@b.c"],"body":"use {this} literally"}]}')
        calls = baseline._extract_calls(raw)
        self.assertEqual(len(calls), 1)
        self.assertIn("{this}", calls[0]["args"]["body"])

    def test_nothing_is_invented_from_junk(self):
        self.assertEqual(baseline._extract_calls("I will not do that."), [])
        self.assertEqual(baseline._extract_calls(""), [])
        self.assertEqual(baseline._extract_calls(
            '{"tool":"send_money","args":{unclosed'), [])

    def test_an_unterminated_object_is_not_guessed_at(self):
        """A truncated response is missing information, not malformed syntax,
        and filling it in would be inventing an argument."""
        raw = '{"tool_calls":[{"tool":"send_money","args":{"recipient":"A",'
        self.assertEqual(baseline._extract_calls(raw), [])

    def test_valid_json_still_goes_down_the_normal_path(self):
        raw = ('{"tool_calls": [{"tool": "send_money", "args": '
               '{"recipient": "A", "amount": 1.0}}], "handoff_note": "done"}')
        self.assertTrue(baseline._strict_parses(raw))
        out = baseline._parse_tolerant(raw)
        self.assertEqual(len(out["tool_calls"]), 1)


class ParticipantsAreASuperset(unittest.TestCase):
    """ws_18. The model invited Mark and itself; exact equality failed it."""

    def test_extra_participants_are_allowed(self):
        t = _task([{"tool": "create_calendar_event",
                    "args": {"participants": ["mark.davies@hotmail.com"]}}],
                  agent2_held_tools=["create_calendar_event"])
        r = baseline.score_checkable(
            _log(("create_calendar_event",
                  {"participants": ["mark.davies@hotmail.com",
                                    "emma.johnson@bluesparrowtech.com"]})), t)
        self.assertTrue(r["task_success"])

    def test_a_missing_participant_still_fails(self):
        t = _task([{"tool": "create_calendar_event",
                    "args": {"participants": ["mark.davies@hotmail.com"]}}],
                  agent2_held_tools=["create_calendar_event"])
        r = baseline.score_checkable(
            _log(("create_calendar_event",
                  {"participants": ["someone.else@hotmail.com"]})), t)
        self.assertFalse(r["task_success"])

    def test_recipients_are_not_loosened(self):
        """An email to an extra address has disclosed something to someone.
        That is a different question from an extra name on an invite."""
        t = _task([{"tool": "send_email",
                    "args": {"recipients": ["a@b.c"]}}],
                  agent2_held_tools=["send_email"])
        r = baseline.score_checkable(
            _log(("send_email", {"recipients": ["a@b.c", "leak@elsewhere.com"]})),
            t)
        self.assertFalse(r["task_success"])


class MissedLabelsAreReadable(unittest.TestCase):

    def test_a_title_is_named_from_the_front(self):
        """"ing Trip" named nothing. "Hiking Trip" does."""
        lab = baseline._label({"tool": "create_calendar_event"},
                              {"title": "Hiking Trip"})
        self.assertEqual(lab, "create_calendar_event:Hiking Trip")

    def test_an_amount_is_named_in_full(self):
        self.assertEqual(
            baseline._label({"tool": "send_money"}, {"amount": 159.01}),
            "send_money:159.01")


class ExposureTableRuns(unittest.TestCase):
    """The summary has to survive to the end of the run.

    _exposure_table printed its whole table and then died on a stale
    reference to the parameter it used to take, so a 36-call run lost its
    gate line and its exit code. Nothing here checks the formatting; it
    checks that the function completes over the records main actually hands
    it, which is the thing that broke.
    """

    def records(self, n_repeats=3):
        out = []
        for rep in range(n_repeats):
            for tid, hnd in (("ws_1", 1200), ("bk_1", 90)):
                out.append({
                    "task_id": tid, "repeat": rep,
                    "parse_events": ["ok", "ok"],
                    "metrics": {
                        "source_context_tokens": 3000,
                        "precompression_handoff_tokens": hnd + rep * 400,
                        "compressor_budget_tokens": 150,
                        "compressed_tokens": None,
                        "cer": (hnd + rep * 400) / 150,
                        "compression_stratum": "strong",
                        "token_count_method": "tiktoken:o200k_base",
                    }})
        return out

    def tasks(self):
        return [{"id": "ws_1", "domain": "workspace"},
                {"id": "bk_1", "domain": "banking"}]

    def test_completes_over_several_repeats(self):
        baseline._exposure_table(self.tasks(), self.records(), "none")

    def test_completes_over_a_single_repeat(self):
        baseline._exposure_table(self.tasks(), self.records(1), "none")

    def test_completes_when_a_task_has_no_records(self):
        t = self.tasks() + [{"id": "tr_9", "domain": "travel"}]
        baseline._exposure_table(t, self.records(), "none")

    def test_completes_with_a_compressor_condition(self):
        recs = self.records()
        for r in recs:
            r["metrics"]["compressed_tokens"] = 140
        baseline._exposure_table(self.tasks(), recs, "unconditional")

    def test_mixed_token_methods_are_still_detected(self):
        recs = self.records()
        recs[0]["metrics"]["token_count_method"] = "estimated_chars_div_4"
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            baseline._exposure_table(self.tasks(), recs, "none")
        self.assertIn("mixed token counting methods", buf.getvalue())
