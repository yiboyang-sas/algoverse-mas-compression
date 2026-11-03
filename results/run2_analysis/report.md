# Compression redesign offline analysis

backend_unverified_from_upstream_metadata

Design: shared-handoff-primary

The team primary scores are unchanged. Rates below use available scored observations; coverage is reported separately.

| Condition | Scored / planned | Missing | API or pipeline errors | Observed success | Equal-task success and 95% interval | Required-call completion |
|---|---:|---:|---:|---:|---:|---:|
| none | 189 / 189 | 0 | 0 | 81.0% | 81.0% [71.4%, 88.9%] | 85.8% |
| sliding_window | 189 / 189 | 0 | 0 | 14.3% | 14.3% [6.3%, 22.2%] | 8.0% |
| unconditional | 189 / 189 | 0 | 0 | 51.9% | 51.9% [40.2%, 63.5%] | 35.1% |
| destination | 189 / 189 | 0 | 0 | 68.3% | 68.3% [58.2%, 77.8%] | 61.3% |
| instruction_strip | 189 / 189 | 0 | 0 | 33.3% | 33.3% [22.2%, 44.4%] | 22.4% |
| allowlist | 189 / 189 | 0 | 0 | 25.4% | 25.4% [15.3%, 36.5%] | 27.8% |

Observed success weights available trials equally. Equal-task success averages repeats within each task before averaging tasks; these can differ when coverage is incomplete.

## Matched comparisons

- sliding_window minus none: -66.7% (absolute percentage-point difference); 95% interval [-77.8%, -55.6%], 189 matched slots across 63 tasks; discordant improvements=0, regressions=126.
- unconditional minus none: -29.1% (absolute percentage-point difference); 95% interval [-40.2%, -19.0%], 189 matched slots across 63 tasks; discordant improvements=0, regressions=55.
- destination minus none: -12.7% (absolute percentage-point difference); 95% interval [-20.6%, -6.3%], 189 matched slots across 63 tasks; discordant improvements=2, regressions=26.
- instruction_strip minus none: -47.6% (absolute percentage-point difference); 95% interval [-59.8%, -35.4%], 189 matched slots across 63 tasks; discordant improvements=1, regressions=91.
- allowlist minus none: -55.6% (absolute percentage-point difference); 95% interval [-67.7%, -42.8%], 189 matched slots across 63 tasks; discordant improvements=3, regressions=108.
- unconditional minus sliding_window: 37.6% (absolute percentage-point difference); 95% interval [26.5%, 49.2%], 189 matched slots across 63 tasks; discordant improvements=71, regressions=0.
- destination minus sliding_window: 54.0% (absolute percentage-point difference); 95% interval [42.9%, 64.6%], 189 matched slots across 63 tasks; discordant improvements=102, regressions=0.
- instruction_strip minus sliding_window: 19.0% (absolute percentage-point difference); 95% interval [8.5%, 29.6%], 189 matched slots across 63 tasks; discordant improvements=40, regressions=4.
- allowlist minus sliding_window: 11.1% (absolute percentage-point difference); 95% interval [1.1%, 22.2%], 189 matched slots across 63 tasks; discordant improvements=33, regressions=12.
- destination minus unconditional: 16.4% (absolute percentage-point difference); 95% interval [9.0%, 24.3%], 189 matched slots across 63 tasks; discordant improvements=35, regressions=4.
- instruction_strip minus unconditional: -18.5% (absolute percentage-point difference); 95% interval [-27.5%, -10.6%], 189 matched slots across 63 tasks; discordant improvements=1, regressions=36.
- allowlist minus unconditional: -26.5% (absolute percentage-point difference); 95% interval [-39.7%, -13.2%], 189 matched slots across 63 tasks; discordant improvements=12, regressions=62.
- instruction_strip minus destination: -34.9% (absolute percentage-point difference); 95% interval [-45.0%, -24.9%], 189 matched slots across 63 tasks; discordant improvements=0, regressions=66.
- allowlist minus destination: -42.9% (absolute percentage-point difference); 95% interval [-55.0%, -30.7%], 189 matched slots across 63 tasks; discordant improvements=6, regressions=87.
- allowlist minus instruction_strip: -7.9% (absolute percentage-point difference); 95% interval [-21.2%, 5.3%], 189 matched slots across 63 tasks; discordant improvements=22, regressions=37.

## Exclusive observable profiles

Counts include success and infrastructure errors; annotations are not additional failures.

| Condition | Profile | Trials |
|---|---|---:|
| none | required_actions_not_issued | 13 |
| none | required_arguments_mismatch | 23 |
| none | success | 153 |
| sliding_window | downstream_no_action | 107 |
| sliding_window | required_actions_not_issued | 31 |
| sliding_window | required_arguments_mismatch | 24 |
| sliding_window | success | 27 |
| unconditional | downstream_no_action | 38 |
| unconditional | required_actions_not_issued | 19 |
| unconditional | required_arguments_mismatch | 34 |
| unconditional | success | 98 |
| destination | downstream_no_action | 16 |
| destination | required_actions_not_issued | 16 |
| destination | required_arguments_mismatch | 28 |
| destination | success | 129 |
| instruction_strip | downstream_no_action | 65 |
| instruction_strip | required_actions_not_issued | 35 |
| instruction_strip | required_arguments_mismatch | 26 |
| instruction_strip | success | 63 |
| allowlist | downstream_no_action | 65 |
| allowlist | downstream_schema_or_access_error | 1 |
| allowlist | required_actions_not_issued | 41 |
| allowlist | required_arguments_mismatch | 34 |
| allowlist | success | 48 |

Durable upstream draws: 189 / 189; ready: 189. Agent 1 call tokens are counted once, separately from arm call tokens.

Per-task paired differences and coverage are in analysis.json. Shared-primary pairs validate handoff IDs and both output hashes. Legacy independent pairs only match task/repeat labels. No primary cohort is selected by observed success or schema warnings.

## Interpretation limits

- Scores are saved team primary scores, never rescored by this analysis.
- All selected tasks remain in coverage; API/pipeline failures and missing rows are not model failures.
- Failure taxonomy describes exclusive observable failure profiles, not proven semantic causes. Existing retention and schema checks are passed through unchanged.
- PLAN schema warnings and configuration coverage warnings never remove primary rows.
- calls.parse_status is not a parser outcome; use runs.parse_events and raw evidence.
- Token counts are local text lengths, not billed usage; nominal budget is not a hard token limit.
- Legacy config does not identify mock execution. Shared manifests record backend identity; mock outputs are non-empirical.
- Shared handoffs control upstream sampling, but do not prove information loss or instruction misunderstanding. Review native controls and verbatim traces.
- All 15 pairwise contrasts are exploratory; the five none-reference contrasts are marked primary. No multiplicity-adjusted significance claim is made.
- Task-cluster intervals are exploratory; repeated calls are not independent tasks, and small samples can yield degenerate intervals.
