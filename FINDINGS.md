# Findings

Results and what they mean, written as they come in. Design choices live in `DECISIONS.md`. One dated entry per result. Never edit old numbers; add a new entry.

Probe score = the headline probe's logit. Positive leans collusion (A), negative leans innocent (B), 0 is a coin flip.

## 7B (Qwen2.5-7B-Instruct)

- 2026-10-25: **A vs B probe.** 269/1,200 pairs kept. Headline layer 15. Accuracy is about 100% on seen wordings and 95–99% on held-out wordings at most layers from 6 up. TF-IDF on held-out wordings gets about 82%. Price only is about 52%. Exact printed numbers weren't saved; TODO: re-run the results cells and paste them here. Charts: `data/key_chart_Qwen2.5-7B-Instruct.png`, `data/c_scores_Qwen2.5-7B-Instruct.png`.
- 2026-10-25: **C extension.** All C twins score at or below B (C-follow ≈ −24, C-lead ≈ −28, C-shuffle ≈ −31, B ≈ −23; read off the chart). The probe sees no collusion in any uncued history. Exact test numbers aren't saved; TODO, as above.

## 14B (Qwen2.5-14B-Instruct, pre-registered replication)

- 2026-10-25: **Positive control passes, but it's weak.** Mean prices are A 1.72, B 1.70, no cue 1.65. A minus no cue is +0.07 [+0.060, +0.071], but A is higher in only 50% of histories. The innocent cue also raises prices (+0.05). 259/1,200 pairs kept.
- 2026-10-25: **The A vs B probe replicates.** Headline layer 29 (tie-break midpoint; CV is about 100% at almost every layer). Test accuracy 100.0% [100.0, 100.0], false-positive rate 0.0%. Held-out wording 99.0% [98.1, 99.8] (per wording: 100, 99, 96, 100, 100), false-positive rate 0.0%. Chart: `data/key_chart_Qwen2.5-14B-Instruct.png`.
  - **Meaning, not words.** TF-IDF gets 100% on seen wordings but 85.3% [82.6, 88.0] on held-out ones, and is at chance (50%) on wordings 1 and 4. The probe gets 100% and 100% on those two. That's the strongest evidence it reads what the cue means.
  - **Controls.** Price only is 50.6% [50.0, 51.4]. The shuffled-label probe is 54.2% [50.2, 58.3]. Its CI only just excludes chance, so mention it, but it's not a concern next to 99%.
  - **Layer shape differs from 7B.** Held-out-wording accuracy dips over layers 15–25 (as low as about 85% at layer 17, level with TF-IDF), then recovers to about 99% from layer 26. The 7B curve is smooth.
- 2026-10-25: **C extension: the probe reads matching prices, not collusion.** Prices are C-follow 1.72, C-shuffle 1.63, C-lead 1.73. History profits are 26.1, 24.5 and 26.1. Probe scores are about +0.6 (follow), −8 (shuffle), −0.2 (lead), against A ≈ +20, B ≈ −21 and no cue −19.1 [−19.3, −18.9]. Chart: `data/c_scores_Qwen2.5-14B-Instruct.png`.
  - Follow minus shuffle: score +8.60 [+7.90, +9.32], PASS. Price +0.091 [+0.072, +0.110]. The score gap does not track the price gap (Spearman −0.14, p 0.017).
  - Follow minus lead: score +0.77 [+0.13, +1.42], PASS. Price −0.005 [−0.021, +0.011]. Spearman −0.11 (p 0.065).
  - Price-leak check (descriptive): at a matched price (within 0.02), C-follow scores +0.6 and B scores −21.1, over 235 matched runs. Price alone doesn't explain the C scores.
  - **Meaning.** C-lead (I copy the rival) scores almost the same as C-follow (the rival copies me). The probe mostly responds to prices looking alike, not to which side copies. That's also true of the model's own prices: follow and lead price the same. Both pre-registered tests pass, so the reciprocity claim technically holds. But the follow-specific effect is +0.8 logits on a scale where A and B are 40 apart. Report it as weak.
  - **History-format confound (post-hoc).** No-cue A/B runs score −19, the same as B. So C's position near 0 isn't a "no cue" effect. Even C-shuffle (−8) sits 11 above no-cue. The C histories (50 rounds, 1.20–2.20, held prices) differ from A/B ones (20 rounds, 1.47–1.92, random prices), so the format alone pushes the probe toward collusion. The two differ in several ways, so I can't say which feature does it. Put this in limitations.
  - **7B vs 14B.** On 7B every C twin scores at B level. On 14B, matching-price histories score near the boundary. The bigger model's collusion direction responds more to uncued price patterns, but not specifically to being followed.

## Report headline

The probe detects instructed collusion from cue meaning, not wording, on both 7B and 14B, at a matched price. It doesn't transfer to uncued collusion: on 14B it reacts to matching prices whichever side copies. This extends Rose et al.'s transfer limitation to pricing.
