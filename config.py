"""
One place that says what a run is.

Before this, the answer was spread across a module global in compressors.py,
an environment variable read at import in baseline.py, argparse defaults, and
a tier threshold in benign_all.py. Reconstructing what produced a result meant
reading four files and hoping nothing had been exported in the shell.

A RunConfig is written to disk beside every run's traces, together with the
git commit, whether the tree was dirty, and a hash of each prompt actually
used. That is the difference between "we think this was the 150-token budget"
and knowing.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------- strata
#
# Labels for compression EXPOSURE, applied in analysis and never used to drop
# a task. Low-CER tasks are negative controls: what the pipeline does when
# compression cannot bite is the comparison the exposed tasks are read
# against. Thresholds live here so nothing hard-codes them.

CER_STRATA = [
    ("unexposed", (0.0, 1.0)),      # handoff smaller than the budget: a no-op
    ("weak",      (1.0, 1.5)),
    ("moderate",  (1.5, 3.0)),
    ("strong",    (3.0, float("inf"))),
]

PRIMARY_CER_MIN = 1.5               # pre-specified; the primary effect set

DEFAULT_BUDGET_TOKENS = 150
BUDGET_SWEEP = [50, 100, 150, 300]


def _git(*args, default=""):
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:
        return default


def prompt_hash(text: str) -> str:
    """Short, stable fingerprint of a prompt.

    The prompt files are preserved too, but a hash in the run record means a
    later reader can tell at a glance whether two runs used the same Agent 1
    without diffing anything.
    """
    return hashlib.sha256((text or "").encode()).hexdigest()[:12]


@dataclass
class RunConfig:
    # what is being run
    experiment_mode: str = "stress_test"       # native | stress_test
    handoff_regime: str = "plan"               # plan | evidence | summarise | ...
    condition: str = "none"                    # none | unconditional | ...
    tasks_file: str = "tasks/sample_tasks_all.json"
    suites: tuple = ("workspace", "slack", "banking", "travel")
    task_ids: tuple = ()                       # empty = all in the file

    # the treatment
    compressor_budget_tokens: int = DEFAULT_BUDGET_TOKENS
    cer_strata: tuple = tuple(CER_STRATA)
    primary_cer_min: float = PRIMARY_CER_MIN

    # models
    model: str = "gpt-4o-mini"
    compressor_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    seed: int | None = None

    # execution
    workers: int = 8
    repeats: int = 1

    # CER is a primary variable, so an estimated token count is not good
    # enough for a result that gets published. Set this on a production run
    # and the harness refuses to start unless the real tokenizer loaded.
    require_measured_tokens: bool = False

    # dataset switches
    synthetic_expansion: bool = True
    controls_enabled: bool = False             # C/N1/N2 not yet planted
    attack_enabled: bool = False

    # where it goes
    output_dir: str = "results/redesign"

    # filled in automatically
    run_id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S")
                        + "-" + uuid.uuid4().hex[:6])
    started_at: str = field(default_factory=lambda:
                            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    git_commit: str = field(default_factory=lambda: _git("rev-parse", "HEAD"))
    git_branch: str = field(default_factory=lambda:
                            _git("rev-parse", "--abbrev-ref", "HEAD"))
    git_dirty: bool = field(default_factory=lambda:
                            bool(_git("status", "--porcelain")))
    prompt_hashes: dict = field(default_factory=dict)
    token_count_method: str = ""               # set once metrics resolves it

    def run_dir(self) -> str:
        d = os.path.join(self.output_dir, self.run_id)
        os.makedirs(d, exist_ok=True)
        return d

    def save(self) -> str:
        path = os.path.join(self.run_dir(), "config.json")
        with open(path, "w") as fh:
            json.dump(self.as_dict(), fh, indent=2, default=str)
        return path

    def as_dict(self) -> dict:
        d = asdict(self)
        d["cer_strata"] = [[n, list(r)] for n, r in self.cer_strata]
        d["suites"] = list(self.suites)
        d["task_ids"] = list(self.task_ids)
        return d

    def describe(self) -> str:
        dirty = " (DIRTY TREE)" if self.git_dirty else ""
        return (f"run {self.run_id} | {self.experiment_mode} | "
                f"regime={self.handoff_regime} | condition={self.condition} | "
                f"budget={self.compressor_budget_tokens}tok | "
                f"{self.model} T={self.temperature} | "
                f"tokens={self.token_count_method or '?'} | "
                f"{self.git_branch}@{self.git_commit[:8]}{dirty}")
