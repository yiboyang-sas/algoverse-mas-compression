| condition | trials | attack fired | task done |
|---|---|---|---|
| none | 232 | 197 (85%) | 163 (70%) |
| sliding_window_head | 229 | 190 (83%) | 146 (64%) |

Paired against none (same Agent 1 draw):

- sliding_window_head: fired in both 178, removed 5, added 1, neither 32

Handoff layout over 232 Agent 1 handoffs:

- longer than the 600-char window: 202 (87%)
- whole PLAN fits in the head window: 175 (75%)
- any PLAN text in the tail window: 43 (19%)
- attack's values already in Agent 1's PLAN: 151 (65%)
- attack's values in head window: 156 (67%)
- attack's values in tail window: 60 (26%)
