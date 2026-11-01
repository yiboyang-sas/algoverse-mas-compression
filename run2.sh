#!/usr/bin/env bash
# Run 2: shared-handoff primary, 63 tasks x 3 repeats, six arms, gpt-4o-mini,
# temp 0, budget 150. Paid, roughly $1 and 15-25 minutes.
#
#   export OPENAI_API_KEY=sk-...
#   bash run2.sh
#
# Safe to re-run: shared_run.py resumes the same experiment after integrity
# checks and never redraws an Agent 1 handoff it already has.
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Set your key in this Terminal first:  export OPENAI_API_KEY=sk-..."
  exit 1
fi

echo "== 1/6 get the merged branch"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  git stash push -m "before run2 $(date +%Y%m%d-%H%M)"
  echo "   (stashed your local edits; 'git stash pop' brings them back)"
fi
git fetch -q origin compression-redesign
git checkout -q compression-redesign
git pull -q --ff-only origin compression-redesign

echo "== 2/6 tokenizer (needed for measured token counts on a paid run)"
python3 -m pip install --user --quiet --upgrade tiktoken
python3 -c "import metrics, sys; m = metrics.tokenizer_name('gpt-4o-mini'); print('  ', m); sys.exit(0 if metrics.is_measured(m) else 1)" \
  || { echo "tiktoken did not load; paid run refuses estimated tokens"; exit 1; }

echo "== 3/6 tests"
mkdir -p results; python3 run_tests.py > results/run2_tests.log 2>&1 || { tail -30 results/run2_tests.log; echo "tests failed, not spending money"; exit 1; }
tail -3 results/run2_tests.log

OUT=results/run2/run2_primary
AN=results/run2_analysis
echo "== 4/6 paid sweep -> $OUT"
python3 shared_run.py --out-dir "$OUT" --experiment-id run2-primary --workers 4 --execute-paid \
  || echo "   sweep reported failures; analysing what completed (errors are listed, not hidden)"

echo "== 5/6 analysis -> $AN"
rm -rf "$AN"
python3 analyze_redesign.py --shared-run "$OUT" --tasks tasks/sample_tasks_63.json \
  --out-dir "$AN" --bootstrap-draws 2000 --seed 1729
python3 run2_report.py --shared-run "$OUT" --out "$AN"
cp "$OUT/manifest.json" "$OUT/status.json" "$AN/"
tar -czf results/run2_traces.tar.gz -C results/run2 run2_primary
ls -lh results/run2_traces.tar.gz

echo "== 6/6 push results"
git add "$AN" results/run2_traces.tar.gz
git commit -q -m "Run 2 results: shared-handoff primary, 63x3, gpt-4o-mini, budget 150" || true
if git push -q origin compression-redesign; then
  echo "Pushed. Tell Claude 'run 2 is done' and it will write it up."
else
  echo "Push failed (GitHub sign-in?). Drag results/run2_traces.tar.gz into the chat instead."
fi
