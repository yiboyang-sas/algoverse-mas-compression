"""Plot semantics tested with synthetic fixtures, never scientific results."""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
import unittest

from scripts.plot_run2 import main, make_figures, task_token_ratios


class TokenRatioTests(unittest.TestCase):
    def test_missing_token_counts_are_not_character_estimates(self):
        data = {"per_task": [{"condition": "none", "task_id": "one", "token_counts": {
            "handoff": {"characters": 100, "tokens": None, "tokenizer": "unavailable"},
            "compressed": {"characters": 50, "tokens": None, "tokenizer": "unavailable"}}}]}
        self.assertEqual(task_token_ratios(data), [])

    def test_actual_ratios_are_pooled_by_task_preserving_repeats(self):
        rows = []
        for compressed in [40, 60, 80]:
            rows.append({"condition": "destination", "task_id": "one", "token_counts": json.dumps({
                "handoff": {"tokens": 100, "tokenizer": "o200k_base"},
                "compressed": {"tokens": compressed, "tokenizer": "o200k_base"}})})
        ratios = task_token_ratios({"per_task": rows})
        self.assertEqual(ratios[0]["observed_repeats"], 3)
        self.assertAlmostEqual(ratios[0]["mean_retained_token_ratio"], .6)


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib optional plotting dependency absent")
class FigureTests(unittest.TestCase):
    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_empty_analysis_has_no_fabricated_data_marks(self):
        figures = make_figures({"n_records": 0, "mock": False, "per_task": [], "summary": []})
        overview = figures["run2_overview"]
        self.assertEqual(len(overview.axes), 3)
        for axes in overview.axes:
            self.assertEqual(len(axes.lines), 0)
            self.assertEqual(len(axes.collections), 0)
            self.assertEqual(len(axes.patches), 0)
            self.assertEqual(len(axes.get_xticklabels()), 6)
        self.assertTrue(any("No measured GPT-4o results" in t.get_text() for t in overview.texts))

    def test_mock_watermark_is_present_in_every_figure(self):
        figures = make_figures({"n_records": 1, "mock": True, "per_task": [], "summary": [], "paired_tasks": []})
        for figure in figures.values():
            self.assertTrue(any("MOCK — INFRASTRUCTURE ONLY" == t.get_text() for t in figure.texts))

    def test_exports_vector_and_raster_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            analysis_dir, output_dir = Path(temp) / "analysis", Path(temp) / "figures"
            analysis_dir.mkdir()
            (analysis_dir / "analysis.json").write_text(json.dumps({"n_records": 0, "mock": False,
                "summary": [], "per_task": [], "paired_tasks": []}))
            main(["--analysis-dir", str(analysis_dir), "--out-dir", str(output_dir)])
            for name in ("run2_overview", "task_success", "paired_delta", "token_retention"):
                for extension in ("pdf", "svg", "png"):
                    self.assertGreater((output_dir / f"{name}.{extension}").stat().st_size, 500)
            metadata = json.loads((output_dir / "figure_metadata.json").read_text())
            self.assertEqual(metadata["data_status"], "no_measured_results")


if __name__ == "__main__":
    unittest.main()
