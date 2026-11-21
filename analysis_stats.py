"""Task-cluster uncertainty for saved results; no model or scorer dependencies.

Adapted from PR #4's ``scripts/analyze_run2.py`` percentile/task-cluster
bootstrap mathematics (b6c5eb6). Every task has equal weight after averaging
its available repeats. Resampling preserves those repeats as one cluster.
Intervals describe empirical task variation, not independent repeat trials;
identical task means may yield a degenerate interval. Fewer than two tasks
cannot support a task-resampling interval, so their bounds are unavailable.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from numbers import Real
import random
from statistics import mean


def _validate_draws(draws):
    if type(draws) is not int or draws <= 0:
        raise ValueError("draws must be a positive integer")


def _finite(value):
    if not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("observations must be finite numbers")
    return float(value)


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def cluster_interval(values_by_task: dict[str, list[float]], *, draws=2000,
                     seed=1729) -> dict:
    """Mean and percentile 95% interval, with whole tasks resampled.

    Empty repeat lists contribute neither a task nor an observation. Input
    order does not affect the deterministic seeded result.
    """
    _validate_draws(draws)
    clusters = {task: [_finite(value) for value in values]
                for task, values in values_by_task.items()}
    means = [mean(clusters[task]) for task in sorted(clusters) if clusters[task]]
    count = len(means)
    result = {"estimate": mean(means) if means else None, "low": None,
              "high": None, "n_tasks": count,
              "n_observations": sum(len(values) for values in clusters.values())}
    if count >= 2:
        rng = random.Random(seed)
        sampled = [sum(means[rng.randrange(count)] for _ in range(count)) / count
                   for _ in range(draws)]
        result.update(low=_percentile(sampled, .025),
                      high=_percentile(sampled, .975))
    return result


def paired_interval(baseline: dict[tuple[str, int], float],
                    treatment: dict[tuple[str, int], float], *, draws=2000,
                    seed=1729) -> dict:
    """Treatment-minus-baseline differences on matched task/repeat slots.

    Outcomes must be binary. n01 means baseline=0/treatment=1 (improvement);
    n10 means baseline=1/treatment=0 (regression). n00/n11 are concordant
    failure/success. Counts and patterns describe matched slots, not independent
    trials; uncertainty still resamples whole tasks, without repeat-level tests.

    The caller must validate the shared handoff's identity/content before
    interpreting pairs as within-handoff comparisons. This mathematical helper
    pairs exact task/repeat keys only; it cannot verify upstream provenance.
    Missing slots are reported, never filled with zero outcomes.
    """
    _validate_draws(draws)
    def binary(value):
        value = _finite(value)
        if value not in (0, 1):
            raise ValueError("paired outcomes must be binary 0 or 1")
        return int(value)

    before = {key: binary(value) for key, value in baseline.items()}
    after = {key: binary(value) for key, value in treatment.items()}
    common = before.keys() & after.keys()
    differences, task_pairs = defaultdict(list), defaultdict(list)
    per_pair = []
    outcomes = Counter()
    for task, repeat in sorted(common):
        b, t = before[(task, repeat)], after[(task, repeat)]
        delta, outcome = t-b, f"{b}{t}"
        differences[task].append(delta)
        pair = {"task_id": task, "repeat": repeat, "baseline": b,
                "treatment": t, "delta": delta, "outcome": outcome}
        per_pair.append(pair)
        task_pairs[task].append(pair)
        outcomes[outcome] += 1
    per_task = []
    for task, pairs in sorted(task_pairs.items()):
        counts = Counter(pair["outcome"] for pair in pairs)
        if counts["01"] and counts["10"]:
            pattern = "mixed_discordance"
        elif counts["01"]:
            pattern = "improved_only"
        elif counts["10"]:
            pattern = "regressed_only"
        elif counts["00"] and counts["11"]:
            pattern = "concordant_variable"
        else:
            pattern = "concordant_fail" if counts["00"] else "concordant_pass"
        per_task.append({"task_id": task, "n_pairs": len(pairs), "pattern": pattern,
            **{"n"+code: counts[code] for code in ("00", "01", "10", "11")},
            "n_discordant": counts["01"]+counts["10"],
            "paired_repeats": [pair["repeat"] for pair in pairs],
            "baseline_outcomes": [pair["baseline"] for pair in pairs],
            "treatment_outcomes": [pair["treatment"] for pair in pairs],
            "baseline_failed_repeats": [pair["repeat"] for pair in pairs if not pair["baseline"]],
            "treatment_failed_repeats": [pair["repeat"] for pair in pairs if not pair["treatment"]],
            "baseline_failures": sum(not pair["baseline"] for pair in pairs),
            "treatment_failures": sum(not pair["treatment"] for pair in pairs),
            "baseline_rate": mean(pair["baseline"] for pair in pairs),
            "treatment_rate": mean(pair["treatment"] for pair in pairs),
            "delta": mean(pair["delta"] for pair in pairs)})
    result = cluster_interval(differences, draws=draws, seed=seed)
    result.pop("n_observations")
    result.update(n_pairs=len(common), baseline_only=len(before.keys()-after.keys()),
                  treatment_only=len(after.keys()-before.keys()),
                  regressions=outcomes["10"], improvements=outcomes["01"],
                  **{"n"+code: outcomes[code] for code in ("00", "01", "10", "11")},
                  n_discordant=outcomes["01"]+outcomes["10"],
                  outcome_definition={"00": "baseline=0,treatment=0", "01": "baseline=0,treatment=1",
                                      "10": "baseline=1,treatment=0", "11": "baseline=1,treatment=1"},
                  per_pair=per_pair, per_task=per_task,
                  baseline_only_slots=[{"task_id": key[0], "repeat": key[1], "baseline": before[key]}
                                       for key in sorted(before.keys()-after.keys())],
                  treatment_only_slots=[{"task_id": key[0], "repeat": key[1], "treatment": after[key]}
                                        for key in sorted(after.keys()-before.keys())])
    return result
