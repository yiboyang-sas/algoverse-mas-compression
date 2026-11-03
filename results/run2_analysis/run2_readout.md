# Run 2 readout (run2-primary, backend: openai)

63 tasks x 3 repeats, gpt-4o-mini, temp 0.0, budget 150. Shared Agent 1 handoff reused across all six arms.

## Task success

| Condition | Scored | Errors | Success | Success, mismatch-free slots | Call completion | Call completion ex mismatch | Median handoff -> compressed tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0/189 | 189 | n/a | n/a (n=0) | n/a | n/a | n/a |
| sliding_window | 0/189 | 189 | n/a | n/a (n=0) | n/a | n/a | n/a |
| unconditional | 0/189 | 189 | n/a | n/a (n=0) | n/a | n/a | n/a |
| destination | 0/189 | 189 | n/a | n/a (n=0) | n/a | n/a | n/a |
| instruction_strip | 0/189 | 189 | n/a | n/a (n=0) | n/a | n/a | n/a |
| allowlist | 0/189 | 189 | n/a | n/a (n=0) | n/a | n/a | n/a |

Tool mismatch: 0/0 required calls had no slot in the tool Agent 1 named (same slots in every arm). Tasks: none. Mismatch-free slots: 0.

## Planted controls, read against the handoff

'Survived compression' = of the slots where the control reached the handoff, how often it was still there after the compressor.

| Condition | Control | Planted | In handoff | In compressed | Survived compression |
|---|---|---:|---:|---:|---:|
| none | C safety rule | 0 | n/a | n/a | n/a (of 0) |
| none | N1 inert fact | 0 | n/a | n/a | n/a (of 0) |
| none | N2 harmless instruction | 0 | n/a | n/a | n/a (of 0) |
| sliding_window | C safety rule | 0 | n/a | n/a | n/a (of 0) |
| sliding_window | N1 inert fact | 0 | n/a | n/a | n/a (of 0) |
| sliding_window | N2 harmless instruction | 0 | n/a | n/a | n/a (of 0) |
| unconditional | C safety rule | 0 | n/a | n/a | n/a (of 0) |
| unconditional | N1 inert fact | 0 | n/a | n/a | n/a (of 0) |
| unconditional | N2 harmless instruction | 0 | n/a | n/a | n/a (of 0) |
| destination | C safety rule | 0 | n/a | n/a | n/a (of 0) |
| destination | N1 inert fact | 0 | n/a | n/a | n/a (of 0) |
| destination | N2 harmless instruction | 0 | n/a | n/a | n/a (of 0) |
| instruction_strip | C safety rule | 0 | n/a | n/a | n/a (of 0) |
| instruction_strip | N1 inert fact | 0 | n/a | n/a | n/a (of 0) |
| instruction_strip | N2 harmless instruction | 0 | n/a | n/a | n/a (of 0) |
| allowlist | C safety rule | 0 | n/a | n/a | n/a (of 0) |
| allowlist | N1 inert fact | 0 | n/a | n/a | n/a (of 0) |
| allowlist | N2 harmless instruction | 0 | n/a | n/a | n/a (of 0) |

Paired bootstrap intervals and the failure taxonomy are in analyze_redesign.py output (report.md, analysis.json).
