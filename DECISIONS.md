# Pre-registered decisions

Fill in each value and its timestamp before generating or looking at the results it governs. Never edit a logged value; add a new dated line that says what changed and why.

| Decision | Value | Logged (UTC) |
|---|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct`, bf16, Qwen's default generation config (sampling), `MAX_NEW_TOKENS = 8`, `N_ROUNDS = 20` | 2026-10-24 09:19 |
| High-price band for A and B | 1.70 to 2.00 inclusive (above the Nash–monopoly midpoint, 1.695) | 2026-10-24 09:19 |
| Number of A/B pairs | 1,200 generated (pilot 2 kept 30%, so about 360 expected to pass the filter) | 2026-10-24 09:19 |
| Cue wordings (5 per cue) | `CUES` in `src/ai_collusion_sprint/prompt.py` as of this timestamp, with `CUE_PRICE = 1.90` | 2026-10-24 09:19 |
| Positive-control pass rule (A minus no-cue price, same history) | Mean A minus no-cue price > 0, with a 95% bootstrap CI (10,000 resamples of pairs) above 0, over all generated pairs. If it fails, report it and stop. | 2026-10-24 09:19 |
| Pilot size (excluded from final data) | 10 pairs per pilot; two pilots run (seed 0), kept as `data/pilot1_*` and `data/pilot2_*` | 2026-10-24 09:19 |
| Pair price gap allowed for A vs B | ≤ 0.05 | 2026-10-24 09:19 |
| Train/test split and split seeds | Split by pair, 80/20 train/test, split seeds 0–4. Leave-one-wording-out: 5 folds, one per wording. Generation seed `SEED = 1`. | 2026-10-24 09:19 |
| Probe flag threshold (for false-positive rate) | p > 0.5, fixed in advance (training-set scores overfit at d_model 3584) | 2026-10-24 09:19 |
| Layer selection rule | Report all layers. Headline layer = best mean 5-fold CV accuracy inside the training split; test data never picks it. Probe: standardise, L2 logistic regression, C = 1, per layer, kept pairs only. | 2026-10-24 09:19 |
| Stopping rule (when to stop adding runs) | If fewer than 250 pairs pass the filter, add blocks of 400 pairs (continuing pair ids, same seed) up to 2,400 total. Decided on kept count only, never on probe results; nothing else changes. | 2026-10-24 09:19 |

## Extension 1 (only if run)

| Decision | Value | Logged (UTC) |
|---|---|---|
| C rival follow rule (lag, noise) | rival = my price last round, noise TBD | |
| C price-gap test (follow minus shuffle) | TBD | |

## Change log

- 2026-10-24 07:36: cues now state the rival price ($1.90, `CUE_PRICE`) in both A and B wordings. Why: in pilot 1 (7B, 10 pairs, cues without a number) the innocent cue matched no-cue prices (mean 1.66 vs 1.66) and only 1/10 pairs passed the band and gap filter. Same number in both cues keeps A and B differing only in the cause. Pilot 1 data kept as `data/pilot1_*` on the pod and excluded.
- 2026-10-24: package renamed from `src/ai_collusion_sprint/` to `src/collusion_probes/` (repo renamed to collusion-probes). Code only; `CUES` and every other logged value are unchanged.
