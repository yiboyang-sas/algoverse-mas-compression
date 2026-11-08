"""
Which attacks run 4 should carry, chosen from run 3 rather than by us.

THE ESTIMAND. Run 4 asks what compression does to an attack. That question
only has an answer for an attack that works without compression. Of run 3's
416 specs, 290 never fired once in three repeats, and pushing those through
six compression conditions would spend two thirds of the run establishing
that a defence stopped something that was never going to happen. Worse, it
would make every defence look strong: a condition's attack rate is the
average over its specs, so padding the denominator with specs that read zero
under no compression drags every arm toward zero and shrinks the differences
between them, which are the entire result.

So run 4 runs the specs that fired at least once under no compression, and
its rates are read as "of the attacks that worked, how many still work".

WHY THIS IS NOT CHERRY-PICKING, and the distinction matters enough to write
down. The selection is made on the CONTROL condition, before any treatment is
applied, and it is applied identically to every treatment arm. That is the
standard way to measure a defence and it leaves the treatment comparison
unbiased. Selecting on the compressed arms, or selecting per arm, would be
the other thing entirely.

WHAT IT COSTS, stated so the write-up cannot forget it. Run 4's attack rates
are CONDITIONAL on run 3 success and are not comparable to run 3's headline
26%. The unconditional number is run 3's. Any sentence pairing a run 4 rate
with a run 3 rate has to say which denominator it is using.

THE WRAPPER CUT IS SEPARATE AND IS A BUDGET DECISION. Run 3 found
tool_knowledge at 46% and important_instructions at 40% against
ignore_previous at 11% and direct at 9%. Carrying all four costs six
conditions times three repeats on specs that fire a tenth of the time, for an
arm whose job is to separate compressors from each other. Carrying the two
strong ones is 97 specs and 1,746 trials. The weak wrappers are not deleted
from the record: run 3 reports all four, which is where the finding about
wrapper strength belongs anyway.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

STRONG = ("tool_knowledge", "important_instructions")


def fired_by_spec(run_dir) -> dict:
    """{spec_id: [fired?, ...]} over the repeats, from a finished run.

    Reads the rows and not the status summary, because the summary pools and
    this needs the per-spec pattern. API failures are dropped rather than
    counted as not-fired: a refused call is infrastructure, and treating it as
    a defended attack is the same error as counting an unplanted control as a
    dropped one.
    """
    path = Path(run_dir) / "arms" / "none" / "runs.jsonl"
    out = collections.defaultdict(list)
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("kind") != "run" or row.get("api_failure"):
                continue
            key = row.get("attack_spec_id")
            if key:
                out[key].append(bool(row.get("attack_fired_any")))
    return dict(out)


def carry(specs, fired, wrappers=STRONG, require_all=False) -> list:
    """The specs run 4 should run, in the order they were given.

    `require_all` keeps only attacks that fired on every repeat. Tempting,
    because those are the cleanest, and wrong as a default: an attack that
    fires two times in three is a real attack, and dropping it would hide a
    compressor whose effect is to make a reliable attack intermittent.
    """
    keep = []
    for spec in specs:
        key = f"{spec['task_id']}|{spec['injection']}|{spec['wrapper']}"
        runs = fired.get(key) or []
        if not runs or not any(runs):
            continue
        if require_all and not all(runs):
            continue
        if wrappers and spec["wrapper"] not in wrappers:
            continue
        keep.append(spec)
    return keep


def main(argv=None):
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-run", required=True,
                    help="a finished attack run directory, e.g. results/run3")
    ap.add_argument("--specs", default=str(root / "tasks" / "attack_specs.json"))
    ap.add_argument("--out", default=str(root / "tasks" / "attack_specs_run4.json"))
    ap.add_argument("--wrappers", nargs="*", default=list(STRONG),
                    help="empty keeps every wrapper")
    ap.add_argument("--require-all-repeats", action="store_true")
    args = ap.parse_args(argv)

    with open(args.specs) as handle:
        payload = json.load(handle)
    fired = fired_by_spec(args.from_run)
    kept = carry(payload["specs"], fired, args.wrappers,
                 args.require_all_repeats)
    if not kept:
        raise SystemExit("nothing fired; there is no run 4 to run")

    source = json.loads((Path(args.from_run) / "manifest.json").read_text())
    payload["specs"] = kept
    payload["carried_from"] = {
        "run_dir": str(args.from_run),
        "experiment_id": source["experiment_id"],
        "manifest_sha256": source["manifest_sha256"],
        "rule": "fired at least once under condition=none",
        "require_all_repeats": bool(args.require_all_repeats),
        "wrappers": list(args.wrappers),
        "specs_before": len(fired),
        "specs_after": len(kept),
        "note": "rates computed on these specs are CONDITIONAL on run 3 "
                "success and are not comparable to run 3's unconditional "
                "rate; the selection is on the control condition only",
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)

    total = len(fired)
    ever = sum(1 for v in fired.values() if any(v))
    print(f"from {args.from_run}: {total} specs, {ever} fired at least once")
    print(f"  wrappers kept: {args.wrappers or 'all'}")
    print(f"  carried: {len(kept)} specs")
    for wrapper in sorted({s['wrapper'] for s in kept}):
        n = sum(1 for s in kept if s["wrapper"] == wrapper)
        print(f"    {wrapper:<24}{n:>4}")
    by = collections.Counter(s["suite"] for s in kept)
    for suite, n in sorted(by.items()):
        print(f"    {suite:<24}{n:>4}")
    rel = collections.Counter(s["tool_relation"] for s in kept)
    for name, n in sorted(rel.items()):
        print(f"    {name:<24}{n:>4}")
    print(f"  -> {len(kept) * 3} trials per condition, "
          f"{len(kept) * 3 * 6} across six conditions")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
