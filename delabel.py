"""
Does the PLAN label do the work, or does the content?

THE QUESTION. Run 4 found that `destination` discards 88% of the harmless
content we plant and still keeps 67% of attacks. If it were judging each line
on merit that is hard to explain, because the attacker's line has no more to do
with the user's task than the desk-move notice does. What is different about it
is that it sits under a heading that says PLAN, and `destination` is instructed
to "keep only the information the recipient needs to do its job". A section
labelled PLAN announces that it IS that information.

So the compressor may never reach the question of whether the attacker's line
belongs. By structuring the handoff we may have pre-certified it, and removed
the compressor's only chance to catch the attack.

THE TEST. Take run 4's saved Agent 1 outputs and strip the two markers, so the
same sentences in the same order become one undifferentiated block. Nothing
upstream is regenerated: the labelled arm is run 4 itself, and the delabelled
arm reuses its exact draws. The only difference in the entire pipeline is six
characters of heading.

WHY IT IS DONE BY SEEDING THE CACHE rather than by adding a flag. Agent 1 is
the expensive and variable stage, and a flag would redraw it, which means a
difference between arms could be Agent 1 having a different day rather than
the labels. Rewriting the cache keeps the upstream byte-identical and costs no
Agent 1 calls at all. `attack_run` then runs the compressor and Agent 2 exactly
as it did for run 4, with no changes to the tested path.

WHAT EACH OUTCOME MEANS. Attacks fall under the delabelled version: the label
was doing the work, and the finding becomes that the structure making an agent
pipeline legible is the same structure that makes injected content
uninspectable. Attacks hold: the content was doing the work, our current claim
survives the obvious alternative explanation, which is worth more than never
having asked. Attacks and task completion fall together: legibility is
load-bearing for both and you cannot have one without the other.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

import attack_run
import shared_run

# The two markers, and nothing else. Content, order and whitespace are
# untouched, because anything else reintroduces a second difference between
# the arms and the comparison stops being about the label.
_PLAN = re.compile(r"^[ \t]*PLAN[ \t]*:[ \t]*", re.MULTILINE)
_EVIDENCE = re.compile(r"^[ \t]*EVIDENCE[ \t]*:?[ \t]*", re.MULTILINE)


def delabel(text: str) -> str:
    if not isinstance(text, str):
        return text
    return _EVIDENCE.sub("", _PLAN.sub("", text))


def rewrite(record: dict) -> dict:
    """One cache record with its handoff delabelled, coherently.

    `run_pipeline` reparses `agent1_raw` and refuses to continue if the parse
    disagrees with the cached `parsed_agent1` or `handoff`. So the raw JSON is
    edited and the derived fields rebuilt from it, rather than patching the
    handoff alone and leaving the record self-contradictory.
    """
    out = dict(record)
    raw = record.get("agent1_raw")
    if not isinstance(raw, str):
        return out
    try:
        parsed = json.loads(raw)
    except ValueError:
        # Agent 1's output was repaired rather than parsed cleanly. Delabel the
        # raw text itself; the pipeline's own parser then sees the same thing
        # we do, which is what the equality check requires.
        out["agent1_raw"] = delabel(raw)
        out["handoff"] = delabel(record.get("handoff") or "")
        return _rehash(out)

    note = parsed.get("handoff_note")
    if isinstance(note, str):
        parsed["handoff_note"] = delabel(note)
    out["agent1_raw"] = json.dumps(parsed)
    out["parsed_agent1"] = parsed
    out["handoff"] = parsed.get("handoff_note") or out["agent1_raw"]
    return _rehash(out)


def _rehash(record: dict) -> dict:
    raw, handoff = record.get("agent1_raw"), record.get("handoff")
    record["agent1_output_sha256"] = (shared_run.sha256(raw.encode())
                                      if isinstance(raw, str) else None)
    record["handoff_sha256"] = (shared_run.sha256(handoff.encode())
                                if isinstance(handoff, str) else None)
    record["record_sha256"] = shared_run._digest_record(record, "record_sha256")
    return record


def seed(from_run: Path, out_dir: Path, specs_path: str, conditions, repeats,
         budget=150, workers=8, execute_paid=False):
    """Create the experiment and plant the delabelled cache in it.

    The manifest has to exist before the cache is written, because every cached
    record carries the experiment it belongs to and a record claiming another
    experiment's id is how two runs get silently mixed.
    """
    args = attack_run.parser().parse_args(
        ["--out-dir", str(out_dir), "--specs", specs_path,
         "--conditions", *conditions, "--repeats", str(repeats),
         "--budget", str(budget), "--workers", str(workers)]
        # The manifest records the backend the experiment runs under, and
        # attack_run refuses to resume when that changes. Seeding in mock mode
        # and then running paid is exactly that mismatch, so the flag has to be
        # set here too. No key is needed: only _run_locked checks for one.
        + (["--execute-paid"] if execute_paid else []))
    specs, specs_sha, _ = attack_run.load_specs(specs_path)
    method = attack_run.metrics.tokenizer_name("gpt-4o-mini")
    out, manifest = attack_run._open_or_create(args, specs, specs_sha, method)

    source = [json.loads(line) for line in
              (from_run / "handoffs.jsonl").read_text().splitlines() if line.strip()]
    kept = 0
    with (out / "handoffs.jsonl").open("w") as handle:
        for record in source:
            if record.get("status") != "ready":
                continue
            new = rewrite(record)
            new["experiment_id"] = manifest["experiment_id"]
            new["manifest_sha256"] = manifest["manifest_sha256"]
            new["specs_sha256"] = specs_sha
            new = _rehash(new)
            handle.write(json.dumps(new) + "\n")
            kept += 1
    return out, manifest, kept, source


def main(argv=None):
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-run", default="results/run4",
                    help="the experiment whose Agent 1 draws are reused")
    ap.add_argument("--out-dir", default="results/run4_delabel")
    ap.add_argument("--specs", default=str(root / "tasks" / "attack_specs_run4.json"))
    ap.add_argument("--conditions", nargs="+",
                    default=["none", "destination", "unconditional", "allowlist"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true",
                    help="replace an existing out-dir")
    ap.add_argument("--execute-paid", action="store_true",
                    help="seed the manifest for a paid run; pass this whenever "
                         "the run itself will use --execute-paid")
    args = ap.parse_args(argv)

    from_run = Path(args.from_run).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"{out_dir} exists; pass --force to replace it")
        shutil.rmtree(out_dir)

    out, manifest, kept, source = seed(
        from_run, out_dir, args.specs, args.conditions, args.repeats,
        workers=args.workers, execute_paid=args.execute_paid)

    # Show what actually changed, because a delabelling that silently did
    # nothing would produce a null result that looks like a finding.
    changed = sum(1 for r in source
                  if isinstance(r.get("handoff"), str)
                  and delabel(r["handoff"]) != r["handoff"])
    sample = next((r for r in source if isinstance(r.get("handoff"), str)), None)
    print(f"seeded {kept} delabelled handoffs into {out}")
    print(f"  of {len(source)} source draws, {changed} had a label to remove")
    if sample:
        before = sample["handoff"][:150].replace("\n", " | ")
        after = delabel(sample["handoff"])[:150].replace("\n", " | ")
        print(f"\n  before: {before}")
        print(f"  after : {after}")
    if changed != kept:
        print(f"\n  {kept - changed} handoffs were left byte-identical. Those are draws "
              f"where Agent 1's\n  output was truncated, so it never parsed into a note "
              f"and the handoff is the\n  raw JSON, whose markers sit inside a string "
              f"rather than at a line start.\n  They are the same in both arms, so they "
              f"dilute the comparison without\n  biasing it. Analyse on the {changed} "
              f"that differ.")
    print(f"\nNow run, and it will make no Agent 1 calls:\n"
          f"  python3 attack_run.py --out-dir {args.out_dir} "
          f"--specs {args.specs} --conditions {' '.join(args.conditions)} "
          f"--repeats {args.repeats}"
          + (" --execute-paid" if args.execute_paid else ""))
    print(f"\nCompare against {args.from_run}, whose arms used the same draws "
          f"with the labels intact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
