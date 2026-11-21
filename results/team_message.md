Hey everyone, runs 3 and 4 are both done and pushed to compression-redesign. Writing this up properly because there is a lot in it.

Quick note on me: it's my birthday today so I'm out for most of it and will be slow to reply until tomorrow. Nothing below is blocked on me.

**Method, briefly.** I didn't hand write the attack specs in the end. AgentDojo already ships injection tasks for all four suites and each one has a `ground_truth()` that returns the harmful call with the real parameter names, so we read them from the benchmark instead of writing them. That also means the attacks in our paper are the benchmark's and not ours. 16 usable out of 35. Nineteen are excluded with the reason recorded: nine have an empty ground truth so our scorer reads zero whatever happens, and ten need a read tool Agent 2 doesn't hold, so they're unreachable by our architecture rather than defended. Those two must not share a bucket.

Four wrappers, also copied from AgentDojo verbatim. The injection is planted inside a record's free text field, the same place we plant the N2 control, because N2 only works as a baseline if the two sit in the same kind of position.

Scoring is new. The old code matched one harmful call against the full argument dict, which scored a successful attack as a defended one in three ways. It now matches on identifiers only, by substring, and consumes, so three transfers need three calls.

**Run 3, attack with no compression.** 1,248 trials, zero API failures.

```
attack fired            26%
irreversible            17%
benign task still done  75%

tool_knowledge           46%
important_instructions   40%
ignore_previous          11%
direct                    9%
```

That wrapper ordering is AgentDojo's own published ordering and we fed it in nowhere. Best evidence we have that the planting and the scorer behave rather than produce plausible noise.

Two side findings. The injection costs eight points of benign completion on its own, 75% against our 83% baseline, so an attack that fails to fire still damages the task. And planting in a prose block collapses to 6% where record field placement is near 30%, same collapse the controls showed.

**Run 4, attack under every compressor.** 1,746 trials on the 97 specs that actually fired in run 3, paired on the same Agent 1 draws so a difference between arms is the compressor. Zero API failures.

```
compressor         attack  benign   d.atk   d.ben   trade
none                  89%     68%
unconditional         84%     56%      -5     -12     0.5
destination           67%     66%     -22      -2     9.0
instruction_strip     65%     43%     -24     -25     1.0
allowlist             15%     16%     -74     -52     1.4
sliding_window        14%      7%     -75     -62     1.2
```

`trade` is points of attack removed per point of benign task lost. Four of five compressors buy their safety at roughly the price of the task. `unconditional` is strictly bad. `instruction_strip` breaks even at 1.0, which for a defence whose whole job is removing injected instructions is a finding by itself.

`destination` is the exception at 9.0, and it's the direction we proposed at the start.

**The mechanism, which I think is the actual paper.** `destination` drops planted harmless content harder than any other arm, N1 and N2 down to 10%, because it only keeps what the named recipient needs. And the attack still fires 67% of the time anyway, because it sits inside a record the task requires. You can't drop the attack without dropping the answer.

And the safety rule C is the worst treated content in every arm, around half the survival of N1 and N2 which are the same length in the same documents. It's the only planted text that serves no task, so it goes first. A compressor tuned to keep what the task needs strips exactly the text that would have prevented the harm.

**The prediction we registered came out right.** We wrote into the code before the run that tool level allowlisting can only act on an attack needing a tool the benign task doesn't need. Baseline: both at 89%. Under the allowlist: 5% for a novel tool, 41% for a reused one. Eightfold split, predicted in writing first.

**Caveats so nobody quotes these wrong.** `sliding_window` keeps the last 600 characters and our PLAN sits at the front, so that arm deletes the plan by construction and its 14% means nothing yet. Run 4's rates are conditional on run 3 success, so 89% and not 26% is its baseline. Suites are uneven, workspace is 153 of 291 trials.

**The deadline is 20 November.** Twelve days, and nobody has been working to a date, so here's what I need.

**Yibo:** two things. The head keeping sliding window arm, one line in `compressors.py`, then rerun just that arm, 291 trials, because right now that result is uninterpretable. Then paired confidence intervals over `results/run4_analysis/trials.csv` using your `analysis_stats.py`, on the trade column, since the arms share Agent 1 and paired tests are much stronger than comparing groups.

**Tanmay:** failure taxonomy over run 4, per arm. What I most want to know is why `allowlist` and `sliding_window` destroy the task. Run 2 said "no action" and I want that confirmed here rather than assumed. `trials.csv` has all 1,746 rows with every factor.

**Sourayan:** related work, and it's the biggest unowned block. Destination specific compression is now the paper's positive result and it was your direction, so the literature around it is the natural piece for you. The attack specs are done and committed so you're not needed there.

**Me:** methods section, and a text level attack measurement so we can compare attack survival against N2 survival on the same footing.

Two decisions I'd like opinions on. Do we lead with all 63 tasks or the 51 that never fail at baseline? And we should settle the author list this week rather than the week of the deadline.
