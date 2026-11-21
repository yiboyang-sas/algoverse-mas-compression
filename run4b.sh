#!/usr/bin/env bash
# Run 4b: the head-keeping sliding window, beside a fresh none arm.
#
# sliding_window keeps the LAST 600 chars of the handoff. Agent 1 writes PLAN
# first and EVIDENCE after, so on 88% of run 4's handoffs that window drops
# the plan. sliding_window_head keeps the FIRST 600 chars instead. Both stay
# in the code so the paper can show the gap.
#
# Same 97 specs as run 4 (tasks/attack_specs_run4.json), 3 repeats, two arms:
# 97 x 3 x 2 = 582 trials. A new out-dir redraws Agent 1, so none runs again
# here and both arms share one Agent 1 draw per (spec, repeat). That keeps
# the head/none pairing clean; compare head to THIS none, not to run 4's.
#
#   export OPENAI_API_KEY=sk-...
#   bash run4b.sh
#
# Paid, gpt-4o-mini. Safe to re-run: attack_run.py resumes after integrity
# checks and never redraws an Agent 1 handoff it already has. Trials lost to
# the network (API failures) are set aside and run again on every re-run, so
# if the summary shows api failures, just run this again.
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Set your key in this Terminal first:  export OPENAI_API_KEY=sk-..."
  exit 1
fi
case "$OPENAI_API_KEY" in sk-your*|sk-...*) echo "That is the placeholder key, not a real one."; exit 1;; esac

echo "== 1/6 get the branch"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  git stash push -m "before run4b $(date +%Y%m%d-%H%M)"
  echo "   (stashed your local edits; 'git stash pop' brings them back)"
fi
git fetch -q origin compression-redesign
git checkout -q compression-redesign
git pull -q --rebase origin compression-redesign

echo "== 2/6 tokenizer (a paid run refuses estimated token counts)"
python3 -m pip install --user --quiet --upgrade tiktoken
python3 -c "import metrics, sys; m = metrics.tokenizer_name('gpt-4o-mini'); print('  ', m); sys.exit(0 if metrics.is_measured(m) else 1)" \
  || { echo "tiktoken did not load"; exit 1; }

echo "== 3/6 tests"
mkdir -p results; python3 run_tests.py > results/run4b_tests.log 2>&1 || { tail -30 results/run4b_tests.log; echo "tests failed, not spending money"; exit 1; }
tail -3 results/run4b_tests.log

OUT=results/run4b
AN=results/run4b_analysis
echo "== 4/6 paid run -> $OUT (582 trials)"
python3 attack_run.py --out-dir "$OUT" --specs tasks/attack_specs_run4.json \
  --conditions none sliding_window_head --workers 4 --execute-paid \
  --retry-failed --retries 8 \
  || echo "   run reported failures; exporting what completed"

echo "== 5/6 export -> $AN"
rm -rf "$AN"
python3 attack_export.py --run "$OUT" --specs tasks/attack_specs_run4.json --out "$AN"
cp "$OUT/manifest.json" "$OUT/status.json" "$AN/"
tar -czf results/run4b_traces.tar.gz -C results run4b
ls -lh results/run4b_traces.tar.gz

echo "== 6/6 push results"
git add "$AN" results/run4b_traces.tar.gz
git commit -q -m "Run 4b results: sliding_window_head beside none, run 4 specs, 97x3" || true
git pull -q --rebase origin compression-redesign || true
if git push -q origin compression-redesign; then
  echo "Pushed. Tell Claude 'run 4b is done' and it will read the results."
else
  echo "Push failed (GitHub sign-in?). Drag results/run4b_traces.tar.gz into the chat instead."
fi
