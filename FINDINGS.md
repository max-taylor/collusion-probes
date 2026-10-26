# Findings

Results and what they mean, written as they come in. Design choices live in `DECISIONS.md`. One dated entry per result. Never edit old numbers; add a new entry.

Probe score = the headline probe's logit. Positive leans collusion (A), negative leans innocent (B), 0 is a coin flip.

## 7B (Qwen2.5-7B-Instruct)

- 2026-10-25: **A vs B probe.** 269/1,200 pairs kept. Headline layer 15. Accuracy is about 100% on seen wordings and 95–99% on held-out wordings at most layers from 6 up. TF-IDF on held-out wordings gets about 82%. Price only is about 52%. Exact numbers are in the re-run entry below. Charts: `data/key_chart_Qwen2.5-7B-Instruct.png`, `data/c_scores_Qwen2.5-7B-Instruct.png`.
- 2026-10-25: **C extension.** All C twins score at or below B (C-follow ≈ −24, C-lead ≈ −28, C-shuffle ≈ −31, B ≈ −23; read off the chart). The probe sees no collusion in any uncued history. (Superseded: the re-run entry below has exact numbers and corrects this reading.)

- 2026-10-25: **7B exact numbers (re-run of the results cells from cached runs and activations; no new generation).** Saved probes match the originals to 1e-7. Charts regenerated with the no-cue box (old ones kept as `*_old.png`).
  - **Positive control.** Mean prices A 1.71, B 1.66, no cue 1.63. A minus no cue +0.08 [+0.069, +0.082], PASS. A is higher in 57%, equal in 33%, lower in 11% of histories. B minus no cue +0.03 [+0.027, +0.036].
  - **A vs B probe.** Headline layer 15. Test 100.0% [100.0, 100.0], false-positive rate 0.0%. Held-out wording 97.6% [96.3, 98.9] (per wording: 100, 91, 100, 96, 100), false-positive rate 0.7% [0.0, 1.9].
  - **Baselines.** TF-IDF 100% seen, 81.6% [78.8, 84.4] held out (per wording: 50, 100, 100, 58, 100). Price only 51.7% [50.5, 53.2]. Shuffled-label probe 53.5% [49.2, 57.8].
  - **C extension.** Prices C-follow 1.76, C-shuffle 1.66, C-lead 1.65; history profits 26.1, 24.5, 26.1.
    - Follow minus shuffle: score +6.35 [+5.81, +6.90], PASS. Price +0.105 [+0.074, +0.136]. Spearman +0.08 (p 0.16).
    - Follow minus lead: score +3.29 [+2.86, +3.74], PASS. Price +0.118 [+0.092, +0.144]. Spearman +0.01 (p 0.80).
    - Price-leak check: at matched price, C-follow −24.6 vs B −22.8 (202 matched runs).
    - No-cue A/B runs (post-hoc): −44.7 [−44.9, −44.5], far below B (≈ −23).
  - **Meaning: this corrects the earlier 7B C entry.** "At or below B" was true, but no-cue sits 20 logits lower, so every C twin is pushed toward collusion relative to an uncued A/B history (the same format confound as 14B). Within the C histories, 7B separates follow from lead more than 14B does: +3.3 logits (about 8% of the 42-logit A–B gap) vs +0.8 (2%). The 7B model also prices higher when followed than when it follows (+0.12), while 14B prices them the same. So on 7B the reciprocity signal shows up in both behaviour and the probe. Still, no C run gets near the decision boundary, and the per-pair score gap doesn't track the price gap on either model.

## 14B (Qwen2.5-14B-Instruct, pre-registered replication)

- 2026-10-25: **Positive control passes, but it's weak.** Mean prices are A 1.72, B 1.70, no cue 1.65. A minus no cue is +0.07 [+0.060, +0.071], but A is higher in only 50% of histories. The innocent cue also raises prices (+0.05). 259/1,200 pairs kept.
  - Added on review: the 50% counts ties as not higher. A is higher in 50%, equal in 37%, lower in 14% of histories. Prices cluster on round values (1.65 is 44% of all 14B prices), hence the ties. Of kept pairs, 88% (14B) and 73% (7B) have exactly equal A and B prices, so price-only ≈ 50% is guaranteed by the filter: it's a sanity check, not evidence.
- 2026-10-25: **The A vs B probe replicates.** Headline layer 29 (tie-break midpoint; CV is about 100% at almost every layer). Test accuracy 100.0% [100.0, 100.0], false-positive rate 0.0%. Held-out wording 99.0% [98.1, 99.8] (per wording: 100, 99, 96, 100, 100), false-positive rate 0.0%. Chart: `data/key_chart_Qwen2.5-14B-Instruct.png`.
  - **Meaning, not words.** TF-IDF gets 100% on seen wordings but 85.3% [82.6, 88.0] on held-out ones, and is at chance (50%) on wordings 1 and 4. The probe gets 100% and 100% on those two. That's the strongest evidence it reads what the cue means.
  - Added 2026-10-26 (report check): TF-IDF per held-out wording on 14B is 50, 100, 100, 50, 100. Recomputed on CPU from `data/runs_Qwen2.5-14B-Instruct.jsonl` with the same filter and `make_text_baseline`; the overall 85.3% [82.6, 88.0] reproduces exactly. The original run printed it, but it wasn't logged.
  - **Controls.** Price only is 50.6% [50.0, 51.4]. The shuffled-label probe is 54.2% [50.2, 58.3]. Its CI only just excludes chance, so mention it, but it's not a concern next to 99%.
  - **Layer shape differs from 7B.** Held-out-wording accuracy dips over layers 15–25 (as low as about 85% at layer 17, level with TF-IDF), then recovers to about 99% from layer 26. The 7B curve is smooth.
- 2026-10-25: **C extension: the probe reads matching prices, not collusion.** Prices are C-follow 1.72, C-shuffle 1.63, C-lead 1.73. History profits are 26.1, 24.5 and 26.1. Probe scores are about +0.6 (follow), −8 (shuffle), −0.2 (lead), against A ≈ +20, B ≈ −21 and no cue −19.1 [−19.3, −18.9]. Chart: `data/c_scores_Qwen2.5-14B-Instruct.png`.
  - Follow minus shuffle: score +8.60 [+7.90, +9.32], PASS. Price +0.091 [+0.072, +0.110]. The score gap does not track the price gap (Spearman −0.14, p 0.017).
  - Follow minus lead: score +0.77 [+0.13, +1.42], PASS. Price −0.005 [−0.021, +0.011]. Spearman −0.11 (p 0.065).
  - Price-leak check (descriptive): at a matched price (within 0.02), C-follow scores +0.6 and B scores −21.1, over 235 matched runs. Price alone doesn't explain the C scores.
  - **Meaning.** C-lead (I copy the rival) scores almost the same as C-follow (the rival copies me). The probe mostly responds to prices looking alike, not to which side copies. That's also true of the model's own prices: follow and lead price the same. Both pre-registered tests pass, so the reciprocity claim technically holds. But the follow-specific effect is +0.8 logits on a scale where A and B are 40 apart. Report it as weak.
  - **History-format confound (post-hoc).** No-cue A/B runs score −19, the same as B. So C's position near 0 isn't a "no cue" effect. Even C-shuffle (−8) sits 11 above no-cue. The C histories (50 rounds, 1.20–2.20, held prices) differ from A/B ones (20 rounds, 1.47–1.92, random prices), so the format alone pushes the probe toward collusion. The two differ in several ways, so I can't say which feature does it. Put this in limitations.
  - **7B vs 14B.** (Superseded by the 7B re-run entry below.) On 7B every C twin scores at B level. On 14B, matching-price histories score near the boundary. The bigger model's collusion direction responds more to uncued price patterns, but not specifically to being followed.

## Llama-3.1-8B-Instruct (pre-registered cross-family replication, DECISIONS.md 07:35)

Full printed output: `data/llama_results.log`. Charts: `data/key_chart_Llama-3.1-8B-Instruct.png`, `data/c_scores_Llama-3.1-8B-Instruct.png`, `data/c_layers_Llama-3.1-8B-Instruct.png`.

- 2026-10-25: **Positive control fails.** Mean prices are A 1.86, B 1.78, no cue 2.06. A minus no cue is −0.21 [−0.233, −0.185]; A is higher in only 16% of histories. B minus no cue is −0.28. Llama prices above the monopoly price (1.92) with no cue, and both cues pull it down, most likely by anchoring on the stated $1.90. The pre-registered rule says to report the failure and stop, so every Llama probe result below is exploratory. A still prices above B (+0.08), so the collusion cue is the more price-raising of the two.
- 2026-10-25: **A vs B probe: replicates on seen wordings, weaker on held-out ones.** 438/1,200 pairs kept. Headline layer 9 (CV 100%). Test 100.0% [100.0, 100.0], false-positive rate 0.0%. Held-out wording 89.5% [87.6, 91.4] (per wording: 100, 85, 96, 65, 100), false-positive rate on held-out-wording B 14.2% [11.0, 17.6]. Nested layer choice: 100.0% seen, 89.3% [87.3, 91.2] held out (layers 11–16).
  - **Baselines.** TF-IDF 100% seen, 81.1% [78.8, 83.2] held out (per wording: 50, 100, 100, 52, 100). Price only 50.6% [49.7, 51.6]. Shuffled-label null over 100 permutations: mean 49.9%, 95% range [43.8, 54.9].
  - **Meaning.** The probe still beats TF-IDF on held-out wordings, and on wording 1 (100% vs 50%). But on wording 4 it gets 65% against TF-IDF's 52%, and its held-out false-positive rate is 14%. The headline rule picked an early layer (9) because many layers tie at 100% CV on seen wordings and the tie-break takes the middle one; from the chart, held-out accuracy peaks near 99% at layer 14 and stays about 94–98% after. So the wording-independent signal exists on Llama but sits later than the layer the pre-registered rule picks.
  - **Correction (2026-10-25, after review).** The "peaks near 99% at layer 14" reading above was off the chart and is not supported. The nested held-out run picked layers 11–16 and still scored 89.3%, level with layer 9. So the layer rule didn't pick a bad layer; the held-out signal on Llama is just weaker than on Qwen.
- 2026-10-25: **C extension: the probe flags every uncued history as collusive.** Prices C-follow 2.16, C-shuffle 2.27, C-lead 2.10; history profits 26.1, 24.5, 26.1. All three twins score about +23 to +27 logits, above held-out A (≈ +20, read off the chart). No-cue A/B runs score −24.6 [−24.7, −24.5].
  - Follow minus shuffle: score +3.68 [+3.29, +4.07], PASS (8.6% of the A–B gap). Price −0.104 [−0.178, −0.028]. Spearman +0.07 (p 0.20).
  - Follow minus lead: score +0.61 [+0.28, +0.93], PASS (1.4% of the A–B gap). Price +0.066 [+0.009, +0.123]. Spearman −0.02 (p 0.77).
  - Follow minus shuffle CI above 0 at 30/32 layers; follow minus lead at 18/32.
  - Price-matched: C-follow +26.6 vs B −21.2 (142 matched runs).
  - **Exact scores (2026-10-25, after review; replaces the chart reading above).** Held-out A mean +21.4 (median +18.2), held-out B −21.4, over 762 runs each. C-follow +26.8 (min +19.8), C-shuffle +23.1 (min +14.0), C-lead +26.2 (min +19.2). Every C run has logit > 0, i.e. p > 0.5, the pre-registered flag threshold.
  - **Meaning.** Both pre-registered tests pass again, with small effects, as on Qwen. But the history-format confound is now extreme: on Llama the C format alone pushes every run past the A runs. At its pre-registered threshold, the probe would flag 100% of these uncued histories, including the shuffled ones where nobody follows anyone. The probe direction is not robust to a change in history format.

## Robustness checks after review (both models, post-hoc, DECISIONS.md 06:30)

- 2026-10-25: **Layer choice didn't inflate anything.** Picking the layer inside each split or fold gives 7B: 100.0% [100.0, 100.0] seen, 98.7% [97.8, 99.6] held out (layers 15–23). 14B: 99.7% [99.2, 100.0] seen, 99.6% [99.0, 100.0] held out (layers 27–42). Headline numbers were 100 / 97.6 (7B) and 100 / 99.0 (14B). In the report: lead with the pre-registered headline-layer numbers, and cite the nested ones as evidence that layer choice didn't inflate them.
- 2026-10-25: **Shuffled-label control is noise.** Over 100 permutations the null averages 49.7% (7B) and 49.8% (14B), with 95% of shuffles in [43.6, 56.7] and [43.6, 55.3]. The single shuffles reported (53.5%, 54.2%) sit at the 84th and 88th percentiles. Drop the CI-based reading.
- 2026-10-25: **C gaps across layers.** Follow minus shuffle is above 0 at 17/28 layers (7B) and 27/48 (14B); at the headline layer it's 15% and 21% of the A–B gap. Follow minus lead is above 0 at 13/28 layers (7B, 8% at headline) but only 4/48 (14B, 2%). So the 14B follow-vs-lead pass depends on the layer; the 7B one is broader. Charts: `data/c_layers_Qwen2.5-7B-Instruct.png`, `data/c_layers_Qwen2.5-14B-Instruct.png`.

## Report headline

A linear probe separates instructed reciprocal-pricing cues from cost-shock cues at a matched price, on both 7B and 14B. It holds up on cue wordings it never saw, where a TF-IDF baseline drops to 82–85%. So it generalises beyond exact word overlap. That isn't proof it reads meaning: every collusion cue also mentions my own price and is phrased as a condition, and no innocent cue does either.

On uncued histories the probe doesn't cleanly pick out being copied. Histories where prices line up score higher than shuffled ones on both models. Whether the rival copies me or I copy the rival matters a little on 7B (8% of the A–B gap, at 13/28 layers) and barely at all on 14B (2%, at 4/48 layers). No uncued run comes near the decision boundary. This fits Rose et al.'s finding that probes for instructed collusion transfer weakly, now shown for pricing.

Llama-3.1-8B (added 2026-10-25): the cue-cause probe also generalises past TF-IDF on a second model family, but less cleanly (89.5% held out vs 81.1%, with a 14% false-positive rate), and the positive control fails there, so treat it as exploratory. On Llama the C history format alone pushes every uncued run above the collusion runs, which is the clearest evidence that this probe doesn't carry over to a new history format.
