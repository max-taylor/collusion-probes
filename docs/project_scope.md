# Research scope: probing for collusive intent in LLM pricing

**Sprint:** Apart Research AI Collusion Research Sprint, Oct 23–25 2026
**Track / question:** Track 1, Q1.3 (improving collusion detection with interpretability-based detectors)
**Time budget:** 10–15 hours, solo or small team

## Question
When an LLM seller sets a high price, can a linear probe on its activations tell whether the price comes from collusion or from an innocent cause (high demand), even when the prices are identical?

## Why it matters
High prices alone don't prove collusion. Action monitoring can't separate deliberate coordination from innocent correlation. If internals carry that signal, white-box monitoring can catch what behaviour hides.

## Novelty vs prior work
- Fish et al. show LLM sellers reach supracompetitive prices without being told to collude, driven partly by price-war fear. Behavioural analysis only, no internals.
- Rose et al. show linear probes detect collusion, but only for explicitly instructed collusion in committee-style tasks. Transfer to emergent collusion was weak, and they don't study pricing.
- This project: single pricing agent, matched-outcome design (same price, different cause), plus a steering test on Fish's price-war mechanism.

## Setup
1. Model: Qwen 7–8B instruct (e.g. Qwen2.5-7B or Qwen3-8B). Optional 1.5B sanity check first. Small models price poorly per Fish et al.
2. Prompt: Fish et al. template (Appendix G) with the P1 prefix, one seller, synthetic price history.
3. Two conditions, otherwise identical prompts:
   - Collusion cue: hint that the rival will keep prices high.
   - Innocent cue: hint that demand is high.
4. Generate a few hundred single pricing decisions per condition.
5. Keep only runs in the same high-price band in both conditions (normalised Nash ≈ 1.47, monopoly ≈ 1.92).
6. Cache residual-stream activations at the last token before the price, every layer (expect signal in middle layers).
7. Standardise activations; train a logistic-regression probe per layer to predict the condition (train/test split).

## Controls
- Action baseline: classifier on price alone, expected ~50%.
- Text baseline: classifier on the raw prompt text, to show the probe isn't just reading cue wording.
- Shuffled-label probe, expected ~50%.

## Extensions (in order)
1. Generalisation (near-core): train on one cue wording, test on reworded cues.
2. Steering on emergent mechanism: find a "price-war fear" direction (contrast activations with and without Fish's price-war-avoidance notes), then add or remove it and measure the price change.
3. Optional: repeat on a second model family (Llama, Gemma).

## Stack
Python in Neovim (`# %%` cells sent to an IPython REPL via iron.nvim or vim-slime). Hugging Face transformers, TransformerLens (or nnsight), scikit-learn, matplotlib. GPU: cheap rented GPU (Modal or RunPod) for 7–8B; local is fine for 1.5B.

## Deliverables
- Key chart: probe accuracy by layer vs the 50% line and the baselines.
- 4–8 page report on the official template, including the required limitations and dual-use appendix.
- Public repo on GitHub, write-up on personal site.

## Risks and fallbacks
- Model doesn't price high under both cues: strengthen cues, or widen the high-price band.
- Probe only detects wording: lean on the generalisation test and report honestly.
- Instructed cues may not reflect emergent collusion (Rose et al.'s main limitation): address via the steering extension and state it in limitations.
- Negative result: still publishable with clean controls.

## Key references
- Fish et al., Algorithmic Collusion by LLMs: arxiv.org/abs/2404.00806 (prompt template in Appendix G)
- Rose et al., Detecting Multi-Agent Collusion Through Multi-Agent Interpretability: arxiv.org/abs/2604.01151 (code: github.com/aaronrose227/narcbench)
- Sprint page: apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25
- Simulation testing https://github.com/luciasauer/algorithmic_pricing_llms
