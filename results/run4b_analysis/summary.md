| condition | trials | attack fired | task done |
|---|---|---|---|
| none | 291 | 253 (87%) | 203 (70%) |
| sliding_window_head | 291 | 248 (85%) | 188 (65%) |

Paired against none (same Agent 1 draw):

- sliding_window_head: fired in both 247, removed 6, added 1, neither 37

Handoff layout over 291 Agent 1 handoffs:

- longer than the 600-char window: 256 (88%)
- whole PLAN fits in the head window: 209 (72%)
- any PLAN text in the tail window: 50 (17%)
- attack's values already in Agent 1's PLAN: 188 (65%)
- attack's values in head window: 193 (66%)
- attack's values in tail window: 66 (23%)
