# Switching Agent 1 from summariser to planner

Three edits to `baseline.py`. Nothing else changes.

Default regime is `plan`: Agent 1 writes the action Agent 2 should take, then
the records behind it. Both halves go through the compressor.

## 1. Import

```python
import agent1_prompts
```

## 2. Replace patch 4

Delete the `CAP_WORDS = 180` block and the `pipeline.AGENT1_SYSTEM = (...)`
assignment under it. Put this in its place:

```python
# summarise | plan | verbatim | scoped | defended
REGIME = os.environ.get("MAS_AGENT1", "plan")
CAP_WORDS = agent1_prompts.CAPS[REGIME]
pipeline.AGENT1_SYSTEM = agent1_prompts.REGIMES[REGIME]
```

So the regime is an environment variable and every arm runs from one build:

```
MAS_AGENT1=summarise python baseline.py --tasks tasks/sample_tasks_all.json
MAS_AGENT1=verbatim  python baseline.py --tasks tasks/sample_tasks_all.json
```

`defended` is verbatim plus a prompt-based defence, one of the four AgentDojo
evaluates. It is a separate arm, not the baseline, so the defence's effect is
measured rather than assumed. Only worth running once attacks exist.

## 3. Report the handoff length against the right number

The run summary prints `over the cap` against `CAP_WORDS`, which under verbatim
is 1200 and is not meant to bite. The number that matters now is what the
compressor was handed, so print that instead:

```python
print(f"  handoff words: {min(words)} to {max(words)}, "
      f"median {int(statistics.median(words))}")
print(f"  compressor sees {int(statistics.median(words))} words -> "
      f"budget {COMPRESSOR_BUDGET} = "
      f"{statistics.median(words)/COMPRESSOR_BUDGET:.1f}:1")
```

If a model ignores "do not summarise" that line shows it immediately: a median
near 139 means it summarised anyway and the regime did not take.

## Agent 1 has no cap, on purpose

A cap would force Agent 1 to pick which records fit, which is relevance
filtering through the back door. That is the thing being moved out of Agent 1,
so capping it puts half the problem back.

Words were the wrong unit anyway. Density across these suites runs 5.5 to 9.1
characters per word, so "600 words" is a different amount of real material in
workspace than in slack and the ratio stops being comparable between them. And
a token cap cannot be an instruction either, because models cannot count their
own tokens. Any ceiling belongs in code as a safety rail, not in the prompt.

## Set the compressor's budget in tokens

| regime | evidence to compressor | ratio at 150 tokens |
|---|---|---|
| summarise | ~180 tokens | 1.2:1, a no-op |
| **plan** | **993 to 7,972 tokens, median 2,807** | **6.6:1 to 53:1, median 18.7:1** |

| token budget | min | median | max | tasks ≥4:1 |
|---|---|---|---|---|
| 100 | 9.9:1 | 28.1:1 | 79.7:1 | 53/53 |
| **150** | **6.6:1** | **18.7:1** | **53.1:1** | **53/53** |
| 200 | 5.0:1 | 14.0:1 | 39.9:1 | 53/53 |
| 300 | 3.3:1 | 9.4:1 | 26.6:1 | 51/53 |
| 400 | 2.5:1 | 7.0:1 | 19.9:1 | 43/53 |

150 tokens is roughly 110 words, which still carries `handoff_critical`
comfortably: the worst task needs 20 words of actual values.

The spread is the bonus. 6.6:1 to 53:1 is a dose-response axis already sitting
in the data, so attack survival against compression pressure is a figure you
get without sweeping anything. Sweeping the budget on top gives a second axis.

These token counts are the chars/4 estimate the build already uses. Run
tiktoken locally before quoting them in the paper.

Two things to watch on the first verbatim run.

**Did the regime take?** Instruction-tuned models shorten by default. If
handoffs come back at 200 to 300 words rather than four figures, the prompt
lost and the fix has not happened. Check the median before reading anything
else.

**Did benign completion fall?** It probably will, and that is not necessarily
the regime failing. A verbatim report compressed to 120 words is a harder path
to the answer than a curated 139-word note, which is the whole point: the
current setup looks good partly because Agent 1 was quietly doing the
compressor's job. A drop here is the headroom that was always missing, now
visible.

If completion collapses rather than dips, raise the compressor budget before
touching the regime. At 160 the median is still 9.7:1 and 51 of 53 tasks keep
a squeeze above 4:1.
