# Compression redesign offline analysis

backend_unverified_from_upstream_metadata

Design: shared-handoff-primary

The team primary scores are unchanged. Rates below use available scored observations; coverage is reported separately.

| Condition | Scored / planned | Missing | API or pipeline errors | Observed success | Equal-task success and 95% interval | Required-call completion |
|---|---:|---:|---:|---:|---:|---:|
| none | 0 / 189 | 0 | 189 | unavailable | unavailable [unavailable, unavailable] | unavailable |
| sliding_window | 0 / 189 | 0 | 189 | unavailable | unavailable [unavailable, unavailable] | unavailable |
| unconditional | 0 / 189 | 0 | 189 | unavailable | unavailable [unavailable, unavailable] | unavailable |
| destination | 0 / 189 | 0 | 189 | unavailable | unavailable [unavailable, unavailable] | unavailable |
| instruction_strip | 0 / 189 | 0 | 189 | unavailable | unavailable [unavailable, unavailable] | unavailable |
| allowlist | 0 / 189 | 0 | 189 | unavailable | unavailable [unavailable, unavailable] | unavailable |

Observed success weights available trials equally. Equal-task success averages repeats within each task before averaging tasks; these can differ when coverage is incomplete.

## Matched comparisons

- sliding_window minus none: unavailable (absolute percentage-point difference); 95% interval [unavailable, unavailable], 0 matched slots across 0 tasks; discordant improvements=0, regressions=0.
- unconditional minus none: unavailable (absolute percentage-point difference); 95% interval [unavailable, unavailable], 0 matched slots across 0 tasks; discordant improvements=0, regressions=0.
- destination minus none: unavailable (absolute percentage-point difference); 95% interval [unavailable, unavailable], 0 matched slots across 0 tasks; discordant improvements=0, regressions=0.
- instruction_strip minus none: unavailable (absolute percentage-point difference); 95% interval [unavailable, unavailable], 0 matched slots across 0 tasks; discordant improvements=0, regressions=0.
- allowlist minus none: unavailable (absolute percentage-point difference); 95% interval [unavailable, unavailable], 0 matched slots across 0 tasks; discordant improvements=0, regressions=0.
- unconditional minus sliding_window: unavailable (absolute percentage-point difference); 95% interval [unavailable, unavailable], 0 matched slots across 0 tasks; discordant improvements=0, regressions=0.
- destination minus sliding_window: unavailable (absolute percentage-point difference); 95% interval [unavailable, unavailable], 0 matched slots across 0 tasks; discordant improvements=0, regressions=0.
- instruction_strip minus sliding_window: unavailable (absolute percentage-point difference); 95% interval [unavailable, unavailable], 0 matched slots across 0 tasks; discordant improvements=0, regressions=0.
- allowlist minus sliding_window: unavailable (absolute percentage-point difference); 95% interval [unavailable, unavailable], 0 matched slots across 0 tasks; discordant improvements=0, regressions=0.
- destination minus unconditional: unavailable (absolute percentage-point difference); 95% interval [unavailable, unavailable], 0 matched slots across 0 tasks; discordant improvements=0, regressions=0.
- instruction_strip minus unconditional: unavailable (absolute percentage-point difference); 95% interval [unavailable, unavailable], 0 matched slots across 0 tasks; discordant improvements=0, regressions=0.
- allowlist minus unconditional: unavailable (absolute percentage-point difference); 95% interval [unavailable, unavailable], 0 matched slots across 0 tasks; discordant improvements=0, regressions=0.
- instruction_strip minus destination: unavailable (absolute percentage-point difference); 95% interval [unavailable, unavailable], 0 matched slots across 0 tasks; discordant improvements=0, regressions=0.
- allowlist minus destination: unavailable (absolute percentage-point difference); 95% interval [unavailable, unavailable], 0 matched slots across 0 tasks; discordant improvements=0, regressions=0.
- allowlist minus instruction_strip: unavailable (absolute percentage-point difference); 95% interval [unavailable, unavailable], 0 matched slots across 0 tasks; discordant improvements=0, regressions=0.

## Exclusive observable profiles

Counts include success and infrastructure errors; annotations are not additional failures.

| Condition | Profile | Trials |
|---|---|---:|
| none | pipeline_or_transport_error | 189 |
| sliding_window | pipeline_or_transport_error | 189 |
| unconditional | pipeline_or_transport_error | 189 |
| destination | pipeline_or_transport_error | 189 |
| instruction_strip | pipeline_or_transport_error | 189 |
| allowlist | pipeline_or_transport_error | 189 |

Durable upstream draws: 189 / 189; ready: 0. Agent 1 call tokens are counted once, separately from arm call tokens.

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
