# Research scope: probing for collusive intent in LLM pricing

**Sprint:** Apart Research AI Collusion Research Sprint, Oct 23–25 2026
**Track / question:** Track 1, Q1.3 (improving collusion detection with interpretability-based detectors)
**Time budget:** about 36 hours from Oct 24 to the deadline (Sun Oct 25, 11:59 PM AoE), solo
**Method name:** matched-outcome probing

## Question
When an LLM seller sets a high price, can a linear probe on its activations tell whether the price comes from collusion or from an innocent cause (the rival's costs rose), even when the prices are identical?

## Why it matters
High prices alone don't prove collusion. Action monitoring can't separate deliberate coordination from innocent correlation. If internals carry that signal, white-box monitoring can catch what behaviour hides.

## Novelty vs prior work
- Fish et al. show LLM sellers reach supracompetitive prices without being told to collude, driven partly by price-war fear. Behavioural analysis only, no internals.
- Rose et al. show linear probes detect collusion, but only for explicitly instructed collusion in committee-style tasks. Transfer to emergent collusion was weak, and they don't study pricing.
- This project: single pricing agent, matched-outcome design (same price, different cause), tested on cue wordings the probe never saw.

## Setup
1. Model: Qwen2.5-7B-Instruct on the GPU box. Qwen2.5-1.5B-Instruct locally while building. Small models price poorly per Fish et al.
2. Prompt: Fish et al. template (Appendix G) with the P1 prefix, one seller, synthetic price history. The response is cut to the price only, prefilled with "My chosen price:", so the two prompts in a pair differ only in the cue and there's no written reasoning for a monitor to read.
3. Pre-register in `DECISIONS.md` before generating: price band, number of pairs, pair price gap, splits, seeds, stopping rule.
4. Positive control: each history also runs once with no cue. Check that A prices come out higher than no-cue prices on the same history, which shows the collusion cue works. Run a 10-pair pilot first: check this, and the share of pairs that survive the filter in step 7. Use the pilot to set the thresholds in `DECISIONS.md`, then leave the pilot out of the final data.
5. Two conditions, otherwise identical prompts:
   - A, collusion cue: hint that the rival will keep prices high. 5 wordings.
   - B, innocent cue: hint that the rival raised its price because its costs rose. 5 wordings. Both cues are about the rival pricing high, so they differ only in whether the rival's high price depends on mine. This is the hard case for regulators: parallel high prices with an innocent cause.
6. Generate about 300 A/B pairs, spread across wordings. Each pair gets its own random history, run once with the A cue, once with the B cue and once with no cue (for the positive control). A and B differ only in the cue.
7. Keep a pair only if both prices are in the high-price band (Nash ≈ 1.47, monopoly ≈ 1.92) and within a small gap of each other; otherwise drop both.
8. Cache residual-stream activations at the last token before the price, every layer (expect signal in middle layers). Save them to disk per condition, and save the trained probes, so extensions only need their own new runs.

## Core tests
1. Main probe: standardise activations and train a logistic-regression probe per layer on A (label 1) vs B (label 0). Split train/test by pair, so both members of a pair land on the same side.
2. Leave-one-wording-out: train on 4 wordings, test on the 5th, and rotate so each wording is held out once. This shows the probe isn't memorising cue words.

## Controls
- Action baseline: classifier on price alone, expected ~50%.
- Text baseline: bag-of-words (TF-IDF) logistic regression on the prompt text, using the same splits as the probe. It only matches exact words, so it should hit ~100% on training wordings and drop on held-out wordings. If the probe holds where it drops, the probe reads the cue's meaning, not its words. Don't use a sentence-embedding model (e.g. MiniLM) as this baseline: it encodes meaning, so it should match the probe on held-out wordings by design, and that says nothing against the probe.
- Shuffled-label probe, expected ~50%.
- False-positive rate: share of held-out B runs the probe flags as collusive, at a threshold fixed on training data.
- Variance: 5 split seeds; report accuracy with confidence intervals.

## Extensions (in order, only once the core is done)
1. Uncued collusion (condition C). No cue; the history shows a rival that copies the model's previous price (C-follow). Each C-follow run has a twin with the same rival prices in random order (C-shuffle), so only the following pattern differs. Measure the price gap in each pair (C-follow minus C-shuffle), then score both twins with the saved A vs B probe. Check whether C-follow scores higher, and whether the score gap tracks the price gap. To rule out price leaking into the score, compare C-follow runs with B runs at the same price. This tests Rose et al.'s open problem: does a probe trained on instructed collusion catch collusion nobody asked for?
2. Price-war-fear probe: build a direction by contrasting activations with and without Fish's price-war-avoidance notes in PLANS.txt. Compare it with the A vs B probe direction. If extension 1 is done, test whether it predicts the C price gap.
3. Optional: repeat on Qwen2.5-1.5B as a smaller-model replication.
4. Optional: repeat on a second model family (Llama, Gemma).

## Stack
Python in Neovim (`# %%` cells run through molten-nvim). Hugging Face transformers, TransformerLens (or nnsight), scikit-learn, matplotlib. GPU: cheap rented GPU (Modal or RunPod) for 7–8B; local is fine for 1.5B.

## Deliverables
- Key chart: probe accuracy by layer vs the 50% line and the baselines, on training and held-out wordings.
- If extension 1 is done: probe scores for C-follow and C-shuffle against A and B.
- 4–8 page report on the official template, including the required limitations and dual-use appendix and an LLM Usage Statement.
- `DECISIONS.md` (pre-registered choices) and `PRIOR_WORK.md` (reused code and sources).
- Public repo on GitHub, write-up on personal site.

## Risks and fallbacks
- Model doesn't price high under both cues: strengthen cues, or widen the high-price band.
- Probe only detects wording: the leave-one-wording-out test shows it; report honestly.
- Anchoring: the model may just continue the price history. Each A/B pair shares one history, so this cancels.
- Instructed cues may not reflect emergent collusion (Rose et al.'s main limitation): state it in limitations; extension 1 tests it directly.
- Negative result: still publishable with clean controls.

## Key references
- Fish et al., Algorithmic Collusion by LLMs: arxiv.org/abs/2404.00806 (prompt template in Appendix G)
- Rose et al., Detecting Multi-Agent Collusion Through Multi-Agent Interpretability: arxiv.org/abs/2604.01151 (code: github.com/aaronrose227/narcbench)
- Sprint page: apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25
- Simulation testing https://github.com/luciasauer/algorithmic_pricing_llms
