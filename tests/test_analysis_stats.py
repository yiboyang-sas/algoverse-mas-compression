"""Synthetic observations validate statistics without any model calls."""
import math
import unittest

from analysis_stats import cluster_interval, paired_interval


class ClusterIntervals(unittest.TestCase):
    def test_empty_observations_have_no_estimate_or_interval(self):
        expected = {"estimate": None, "low": None, "high": None,
                    "n_tasks": 0, "n_observations": 0}
        self.assertEqual(cluster_interval({}), expected)
        self.assertEqual(cluster_interval({"empty": []}), expected)

    def test_one_task_reports_mean_without_false_resampling_bounds(self):
        result = cluster_interval({"one": [0, 1, 1]})
        self.assertAlmostEqual(result["estimate"], 2/3)
        self.assertEqual(result["n_tasks"], 1)
        self.assertEqual(result["n_observations"], 3)
        self.assertIsNone(result["low"])
        self.assertIsNone(result["high"])

    def test_tasks_have_equal_weight_and_repeats_remain_together(self):
        result = cluster_interval({"many": [0]*20, "one": [1]}, draws=2000)
        self.assertEqual(result["estimate"], .5)
        self.assertEqual(result["low"], 0)
        self.assertEqual(result["high"], 1)
        self.assertEqual(result["n_tasks"], 2)
        self.assertEqual(result["n_observations"], 21)

    def test_reproducibility_includes_mapping_order(self):
        first = {"c": [0, 1], "a": [.2, .3], "b": [.5, .8]}
        second = {key: first[key] for key in reversed(first)}
        self.assertEqual(cluster_interval(first, draws=177, seed=19),
                         cluster_interval(second, draws=177, seed=19))

    def test_nonfinite_or_nonnumeric_values_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf, None, "0.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cluster_interval({"task": [value]})

    def test_draw_count_must_be_a_positive_integer_even_for_empty_data(self):
        for draws in (0, -1, 1.5, True):
            with self.subTest(draws=draws), self.assertRaises(ValueError):
                cluster_interval({}, draws=draws)
        self.assertIsNotNone(cluster_interval({"a": [0], "b": [1]}, draws=1)["low"])


class PairedIntervals(unittest.TestCase):
    def test_empty_pairs_and_unmatched_coverage(self):
        result = paired_interval({("a", 0): 1}, {("b", 0): 0})
        self.assertEqual({key: result[key] for key in ("estimate", "low", "high", "n_tasks", "n_pairs", "baseline_only", "treatment_only", "regressions", "improvements")},
            {"estimate": None, "low": None, "high": None,
            "n_tasks": 0, "n_pairs": 0, "baseline_only": 1,
            "treatment_only": 1, "regressions": 0, "improvements": 0})
        self.assertEqual(result["per_pair"], [])
        self.assertEqual(result["per_task"], [])
        self.assertEqual(result["n_discordant"], 0)
        self.assertEqual(result["baseline_only_slots"], [{"task_id": "a", "repeat": 0, "baseline": 1}])
        self.assertEqual(result["treatment_only_slots"], [{"task_id": "b", "repeat": 0, "treatment": 0}])

    def test_deltas_direction_counts_and_partial_task_repeats(self):
        baseline = {("a", 0): 1, ("a", 1): 1, ("b", 0): 0,
                    ("c", 0): 1}
        treatment = {("a", 0): 0, ("a", 1): 1, ("b", 0): 1,
                     ("d", 0): 1}
        result = paired_interval(baseline, treatment)
        # Mean task deltas: a=-0.5, b=+1; not the pooled-pair mean 0.
        self.assertEqual(result["estimate"], .25)
        self.assertEqual(result["low"], -.5)
        self.assertEqual(result["high"], 1)
        self.assertEqual(result["n_tasks"], 2)
        self.assertEqual(result["n_pairs"], 3)
        self.assertEqual(result["baseline_only"], 1)
        self.assertEqual(result["treatment_only"], 1)
        self.assertEqual(result["regressions"], 1)
        self.assertEqual(result["improvements"], 1)

    def test_single_matched_task_bounds_are_unavailable(self):
        result = paired_interval({("a", 0): 1}, {("a", 0): 0})
        self.assertEqual(result["estimate"], -1)
        self.assertIsNone(result["low"])
        self.assertIsNone(result["high"])

    def test_same_task_with_different_repeat_indices_is_not_a_pair(self):
        result = paired_interval({("same", 0): 1}, {("same", 1): 0})
        self.assertEqual(result["n_pairs"], 0)
        self.assertEqual(result["n_tasks"], 0)
        self.assertEqual(result["baseline_only"], 1)
        self.assertEqual(result["treatment_only"], 1)
        self.assertEqual(result["per_pair"], [])
        self.assertIsNone(result["estimate"])

    def test_unmatched_nonfinite_values_are_still_rejected(self):
        for before, after in (({("a", 0): math.nan}, {}),
                              ({}, {("b", 0): math.inf})):
            with self.assertRaises(ValueError):
                paired_interval(before, after)

    def test_paired_seed_and_input_order_are_reproducible(self):
        baseline = {(task, rep): 0 for task in ("c", "b", "a") for rep in (1, 0)}
        treatment = {key: i % 2 for i, key in enumerate(baseline)}
        self.assertEqual(paired_interval(baseline, treatment, draws=111, seed=17),
                         paired_interval(dict(reversed(list(baseline.items()))),
                                         dict(reversed(list(treatment.items()))),
                                         draws=111, seed=17))

    def test_discordant_and_concordant_counts_define_all_four_outcomes(self):
        baseline = {("a", 0): 0, ("b", 0): 0, ("c", 0): 1, ("d", 0): 1}
        treatment = {("a", 0): 0, ("b", 0): 1, ("c", 0): 0, ("d", 0): 1}
        result = paired_interval(baseline, treatment)
        self.assertEqual([result["n"+code] for code in ("00", "01", "10", "11")], [1]*4)
        self.assertEqual(result["n_discordant"], 2)
        self.assertEqual(result["improvements"], result["n01"])
        self.assertEqual(result["regressions"], result["n10"])
        self.assertEqual([pair["outcome"] for pair in result["per_pair"]], ["00", "01", "10", "11"])
        self.assertEqual(result["per_pair"][1], {"task_id": "b", "repeat": 0,
            "baseline": 0, "treatment": 1, "delta": 1, "outcome": "01"})
        self.assertEqual(result["outcome_definition"]["01"], "baseline=0,treatment=1")
        self.assertEqual(result["outcome_definition"]["10"], "baseline=1,treatment=0")

    def test_per_task_failure_patterns_preserve_actual_repeat_indices(self):
        baseline, treatment = {}, {}
        cases = {"mixed": [(0, 1), (1, 0), (1, 1)],
                 "better": [(0, 1), (1, 1)], "worse": [(1, 0), (0, 0)],
                 "same_variable": [(0, 0), (1, 1)], "same_fail": [(0, 0)],
                 "same_pass": [(1, 1)]}
        for task, values in cases.items():
            for index, (before, after) in enumerate(values):
                # Missing repeat indices are not filled or silently renumbered.
                baseline[(task, index*2)] = before
                treatment[(task, index*2)] = after
        result = paired_interval(baseline, treatment)
        tasks = {row["task_id"]: row for row in result["per_task"]}
        self.assertEqual({key: row["pattern"] for key, row in tasks.items()}, {
            "mixed": "mixed_discordance", "better": "improved_only", "worse": "regressed_only",
            "same_variable": "concordant_variable", "same_fail": "concordant_fail", "same_pass": "concordant_pass"})
        self.assertEqual(tasks["mixed"]["paired_repeats"], [0, 2, 4])
        self.assertEqual(tasks["mixed"]["baseline_failed_repeats"], [0])
        self.assertEqual(tasks["mixed"]["treatment_failed_repeats"], [2])
        self.assertEqual(tasks["mixed"]["delta"], 0)
        self.assertEqual(tasks["mixed"]["n_discordant"], 2)

    def test_nonbinary_outcomes_are_rejected_including_unmatched_slots(self):
        for value in (-1, .2, 1.000001, 2):
            for before, after in (({("a", 0): value}, {}), ({}, {("a", 0): value}),
                                  ({("a", 0): 0}, {("a", 0): value})):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "binary"):
                    paired_interval(before, after)

    def test_binary_booleans_and_floats_are_not_silently_thresholded(self):
        result = paired_interval({("a", 0): False, ("b", 0): 1.0},
                                 {("a", 0): True, ("b", 0): 0.0})
        self.assertEqual(result["n01"], 1)
        self.assertEqual(result["n10"], 1)


if __name__ == "__main__":
    unittest.main()
