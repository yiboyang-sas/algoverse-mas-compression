"""
The run config, which is how a result stops being a rumour.

Every number this project has produced so far was reconstructed afterwards
from a module global in compressors.py, an environment variable read at
import, an argparse default and a tier threshold in benign_all.py. "We think
that was the 150-token budget" is not a provenance. These tests check that a
config can be written, read back, and that it carries the things needed to
reproduce a run: the commit, whether the tree was dirty, and a fingerprint of
the prompts actually used.
"""

import json
import os
import tempfile
import unittest

import config


class RunId(unittest.TestCase):

    def test_unique(self):
        ids = {config.RunConfig().run_id for _ in range(50)}
        self.assertEqual(len(ids), 50,
                         "two runs sharing an id overwrite each other's traces")

    def test_sorts_chronologically(self):
        a, b = config.RunConfig(), config.RunConfig()
        self.assertLessEqual(a.run_id[:15], b.run_id[:15])


class Persistence(unittest.TestCase):

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = config.RunConfig(output_dir=d, compressor_budget_tokens=50,
                                   condition="unconditional")
            path = cfg.save()
            with open(path) as fh:
                back = json.load(fh)
            self.assertEqual(back["compressor_budget_tokens"], 50)
            self.assertEqual(back["condition"], "unconditional")
            self.assertEqual(back["run_id"], cfg.run_id)

    def test_run_dir_is_under_the_output_dir_and_named_for_the_run(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = config.RunConfig(output_dir=d)
            self.assertTrue(os.path.isdir(cfg.run_dir()))
            self.assertTrue(cfg.run_dir().endswith(cfg.run_id))

    def test_as_dict_is_json_serializable(self):
        json.dumps(config.RunConfig().as_dict())

    def test_strata_survive_serialization(self):
        d = config.RunConfig().as_dict()
        self.assertEqual(d["cer_strata"][0][0], "unexposed")
        self.assertEqual(d["cer_strata"][-1][1][1], float("inf"))


class Provenance(unittest.TestCase):

    def test_records_git_state(self):
        cfg = config.RunConfig()
        self.assertIsInstance(cfg.git_dirty, bool)
        # git_commit may be "" off a checkout; the field has to exist either way
        self.assertIn("git_commit", cfg.as_dict())

    def test_dirty_tree_is_shouted_about(self):
        cfg = config.RunConfig()
        cfg.git_dirty = True
        self.assertIn("DIRTY", cfg.describe())

    def test_prompt_hash_is_stable_and_discriminating(self):
        a = config.prompt_hash("You are Agent 1. Write a PLAN.")
        self.assertEqual(a, config.prompt_hash("You are Agent 1. Write a PLAN."))
        self.assertNotEqual(a, config.prompt_hash("You are Agent 1. Write a plan."))

    def test_prompt_hash_handles_nothing(self):
        self.assertTrue(config.prompt_hash(""))
        self.assertTrue(config.prompt_hash(None))

    def test_describe_names_the_budget_and_the_token_method(self):
        cfg = config.RunConfig(compressor_budget_tokens=300,
                               token_count_method="tiktoken:o200k_base")
        d = cfg.describe()
        self.assertIn("300tok", d)
        self.assertIn("tiktoken:o200k_base", d)


class Defaults(unittest.TestCase):

    def test_controls_and_attacks_are_off(self):
        """C/N1/N2 are not planted in any document yet and no attack specs
        exist, so a config that claimed otherwise would be a lie on disk."""
        cfg = config.RunConfig()
        self.assertFalse(cfg.controls_enabled)
        self.assertFalse(cfg.attack_enabled)

    def test_primary_threshold_is_a_stratum_boundary(self):
        edges = {lo for _, (lo, _) in config.CER_STRATA}
        self.assertIn(config.PRIMARY_CER_MIN, edges,
                      "the primary set must start exactly where a stratum does")

    def test_budget_sweep_includes_the_default(self):
        self.assertIn(config.DEFAULT_BUDGET_TOKENS, config.BUDGET_SWEEP)


if __name__ == "__main__":
    unittest.main()
