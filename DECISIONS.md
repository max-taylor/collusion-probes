# Pre-registered decisions

Fill in each value and its timestamp before generating or looking at the results it governs. Never edit a logged value; add a new dated line that says what changed and why.

| Decision | Value | Logged (UTC) |
|---|---|---|
| Model | TBD | |
| High-price band for A and B | TBD | |
| Number of A/B pairs | TBD | |
| Cue wordings (5 per cue) | see `src/ai_collusion_sprint/prompt.py` | |
| Positive-control pass rule (A minus no-cue price, same history) | TBD | |
| Pilot size (excluded from final data) | 10 pairs | |
| Pair price gap allowed for A vs B | TBD | |
| Train/test split and split seeds | TBD | |
| Probe flag threshold (for false-positive rate) | TBD | |
| Layer selection rule | TBD | |
| Stopping rule (when to stop adding runs) | TBD | |

## Extension 1 (only if run)

| Decision | Value | Logged (UTC) |
|---|---|---|
| C rival follow rule (lag, noise) | rival = my price last round, noise TBD | |
| C price-gap test (follow minus shuffle) | TBD | |

## Change log

- 2026-10-24 07:36: cues now state the rival price ($1.90, `CUE_PRICE`) in both A and B wordings. Why: in pilot 1 (7B, 10 pairs, cues without a number) the innocent cue matched no-cue prices (mean 1.66 vs 1.66) and only 1/10 pairs passed the band and gap filter. Same number in both cues keeps A and B differing only in the cause. Pilot 1 data kept as `data/pilot1_*` on the pod and excluded.
