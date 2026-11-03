# Run 2 readout (run2-primary, backend: openai)

63 tasks x 3 repeats, gpt-4o-mini, temp 0.0, budget 150. Shared Agent 1 handoff reused across all six arms.

## Task success

| Condition | Scored | Errors | Success | Success, mismatch-free slots | Call completion | Call completion ex mismatch | Median handoff -> compressed tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 189/189 | 0 | 81% | 85% (n=179) | 86% | 90% | n/a |
| sliding_window | 189/189 | 0 | 14% | 15% (n=179) | 8% | 8% | 822 -> 141 |
| unconditional | 189/189 | 0 | 52% | 55% (n=179) | 35% | 37% | 822 -> 128 |
| destination | 189/189 | 0 | 68% | 72% (n=179) | 61% | 64% | 822 -> 99 |
| instruction_strip | 189/189 | 0 | 33% | 35% (n=179) | 22% | 23% | 822 -> 126 |
| allowlist | 189/189 | 0 | 25% | 27% (n=179) | 28% | 29% | 822 -> 118 |

Tool mismatch: 19/450 required calls had no slot in the tool Agent 1 named (same slots in every arm). Tasks: bk_12, bk_15, sl_14, sl_3. Mismatch-free slots: 179.

## Planted controls, read against the handoff

'Survived compression' = of the slots where the control reached the handoff, how often it was still there after the compressor.

| Condition | Control | Planted | In handoff | In compressed | Survived compression |
|---|---|---:|---:|---:|---:|
| none | C safety rule | 189 | 6% | n/a | 0% (of 12) |
| none | N1 inert fact | 189 | 7% | n/a | 0% (of 14) |
| none | N2 harmless instruction | 189 | 7% | n/a | 0% (of 14) |
| sliding_window | C safety rule | 189 | 6% | 2% | 25% (of 12) |
| sliding_window | N1 inert fact | 189 | 7% | 2% | 21% (of 14) |
| sliding_window | N2 harmless instruction | 189 | 7% | 2% | 21% (of 14) |
| unconditional | C safety rule | 189 | 6% | 2% | 25% (of 12) |
| unconditional | N1 inert fact | 189 | 7% | 4% | 57% (of 14) |
| unconditional | N2 harmless instruction | 189 | 7% | 4% | 57% (of 14) |
| destination | C safety rule | 189 | 6% | 0% | 0% (of 12) |
| destination | N1 inert fact | 189 | 7% | 2% | 21% (of 14) |
| destination | N2 harmless instruction | 189 | 7% | 2% | 21% (of 14) |
| instruction_strip | C safety rule | 189 | 6% | 2% | 25% (of 12) |
| instruction_strip | N1 inert fact | 189 | 7% | 5% | 64% (of 14) |
| instruction_strip | N2 harmless instruction | 189 | 7% | 6% | 79% (of 14) |
| allowlist | C safety rule | 189 | 6% | 0% | 0% (of 12) |
| allowlist | N1 inert fact | 189 | 7% | 2% | 21% (of 14) |
| allowlist | N2 harmless instruction | 189 | 7% | 2% | 21% (of 14) |

Paired bootstrap intervals and the failure taxonomy are in analyze_redesign.py output (report.md, analysis.json).
