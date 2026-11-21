"""
Token counting, CER and strata.

CER is the primary experimental variable now, so the arithmetic that
produces it is the arithmetic most worth testing. Every case below is one
that would have gone unnoticed in a results table: a zero budget, an empty
handoff, a compressor that returned nothing, a boundary that puts a task in
the wrong stratum, and an estimate quietly sitting in a column of
measurements.
"""

import unittest

import metrics
from config import CER_STRATA


class TokenCounting(unittest.TestCase):

    def test_empty_is_zero_and_still_reports_a_method(self):
        n, method = metrics.count_tokens("")
        self.assertEqual(n, 0)
        self.assertTrue(method, "a count with no method is unusable later")

    def test_none_does_not_raise(self):
        self.assertEqual(metrics.count_tokens(None)[0], 0)

    def test_method_is_consistent_across_calls(self):
        a = metrics.count_tokens("hello world")[1]
        b = metrics.count_tokens("something else entirely")[1]
        self.assertEqual(a, b, "two rows of one run must share a method")

    def test_estimate_is_labelled_as_an_estimate(self):
        # Force the fallback the way a blocked egress proxy does.
        saved = (metrics._ENCODER, metrics._ENCODER_NAME,
                 metrics._ENCODER_TRIED)
        metrics._ENCODER, metrics._ENCODER_NAME = None, None
        metrics._ENCODER_TRIED = True
        try:
            n, method = metrics.count_tokens("x" * 400)
            self.assertEqual(n, 100)
            self.assertEqual(method, metrics.ESTIMATED)
            self.assertFalse(metrics.is_measured(method))
        finally:
            (metrics._ENCODER, metrics._ENCODER_NAME,
             metrics._ENCODER_TRIED) = saved

    def test_measured_is_labelled_as_measured(self):
        method = metrics.tokenizer_name()
        if not metrics.is_measured(method):
            self.skipTest(f"no tokenizer here ({method}); the estimate path "
                          f"is covered above")
        self.assertTrue(method.startswith("tiktoken:"))
        self.assertGreater(metrics.count_tokens("hello world")[0], 0)

    def test_pooling_guard(self):
        self.assertTrue(metrics.pooling_is_safe(["tiktoken:o200k_base"] * 3))
        self.assertTrue(metrics.pooling_is_safe([]))
        self.assertFalse(metrics.pooling_is_safe(
            ["tiktoken:o200k_base", metrics.ESTIMATED]))


class Strata(unittest.TestCase):
    """Boundaries, because a task in the wrong stratum is a silent error.

    Ranges are half-open [lo, hi), so 1.0 is weak rather than unexposed and
    1.5 is moderate rather than weak. Writing that down is the point: 1.5 is
    also PRIMARY_CER_MIN, so an off-by-one here moves tasks in and out of the
    primary effect set.
    """

    def test_boundaries(self):
        cases = [(0.0, "unexposed"), (0.56, "unexposed"), (0.999, "unexposed"),
                 (1.0, "weak"), (1.49, "weak"),
                 (1.5, "moderate"), (2.99, "moderate"),
                 (3.0, "strong"), (11.1, "strong"), (1e9, "strong")]
        for cer, want in cases:
            with self.subTest(cer=cer):
                self.assertEqual(metrics.stratum_of(cer, CER_STRATA), want)

    def test_unknown_when_there_is_no_cer(self):
        self.assertEqual(metrics.stratum_of(None, CER_STRATA), "unknown")

    def test_strata_cover_the_line_with_no_gaps(self):
        edges = [r for _, r in CER_STRATA]
        for (_, hi), (lo, _) in zip(edges, edges[1:]):
            self.assertEqual(hi, lo, "a gap here loses tasks silently")
        self.assertEqual(edges[0][0], 0.0)
        self.assertEqual(edges[-1][1], float("inf"))


class Compute(unittest.TestCase):

    def m(self, **kw):
        base = dict(source_text="doc " * 100, handoff_text="word " * 50,
                    compressed_text=None, budget_tokens=150,
                    condition="none", strata=CER_STRATA)
        base.update(kw)
        return metrics.compute(**base)

    def test_source_and_handoff_stay_separate(self):
        """The 119:1 mistake. bk_5 reads 3,224 tokens and hands over 27 words.

        If these two ever collapse into one field, that drop gets reported as
        compression, and the compressor never saw the documents.
        """
        m = self.m(source_text="x " * 3000, handoff_text="short handoff")
        self.assertGreater(m.source_context_tokens, 1000)
        self.assertLess(m.precompression_handoff_tokens, 20)
        self.assertNotEqual(m.source_context_tokens,
                            m.precompression_handoff_tokens)

    def test_agent1_input_is_recorded_separately_from_the_documents(self):
        m = self.m(source_text="doc " * 10,
                   agent1_input_text="TASK: do the thing\n" + "doc " * 10)
        self.assertGreater(m.agent1_input_tokens, m.source_context_tokens)

    def test_cer_is_handoff_over_budget(self):
        m = self.m(handoff_text="w " * 300, budget_tokens=150)
        self.assertAlmostEqual(
            m.cer, m.precompression_handoff_tokens / 150, places=6)

    def test_condition_none_records_that_nothing_was_compressed(self):
        m = self.m(condition="none", compressed_text=None)
        self.assertFalse(m.compression_applied)
        self.assertIsNone(m.compressed_tokens)
        self.assertIsNone(m.realized_ratio)
        self.assertIsNone(m.fraction_removed)
        # ...but exposure is still a number. That is the whole distinction.
        self.assertIsNotNone(m.cer)
        self.assertEqual(m.condition, "none")

    def test_realized_ratio_only_when_a_compressor_ran(self):
        m = self.m(condition="unconditional", handoff_text="w " * 400,
                   compressed_text="w " * 100)
        self.assertTrue(m.compression_applied)
        self.assertGreater(m.realized_ratio, 3.0)
        self.assertAlmostEqual(m.fraction_removed, 0.75, delta=0.05)

    def test_compressor_returning_nothing_is_not_an_infinite_ratio(self):
        m = self.m(condition="unconditional", compressed_text="")
        # Empty output with a non-none condition: everything was removed.
        self.assertIsNone(m.realized_ratio)
        self.assertEqual(m.fraction_removed, 1.0)

    def test_zero_budget_does_not_divide(self):
        m = self.m(budget_tokens=0)
        self.assertIsNone(m.cer)
        self.assertEqual(m.compression_stratum, "unknown")

    def test_empty_handoff_does_not_divide(self):
        m = self.m(handoff_text="", condition="unconditional",
                   compressed_text="anything")
        self.assertEqual(m.precompression_handoff_tokens, 0)
        self.assertEqual(m.cer, 0.0)
        self.assertIsNone(m.fraction_removed)

    def test_method_travels_with_the_row(self):
        m = self.m()
        self.assertTrue(m.token_count_method)
        self.assertIn("token_count_method", m.as_dict())

    def test_as_dict_is_json_serializable(self):
        import json
        json.dumps(self.m().as_dict())


if __name__ == "__main__":
    unittest.main()
