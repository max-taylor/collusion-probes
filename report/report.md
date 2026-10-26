# Same Price, Different Reason: Probing LLM Pricing Agents for Instructed Collusion Cues

**Max Taylor** — Independent [CHECK affiliation]

Apart Research AI Collusion Research Sprint, October 2026. Track 1, Q1.3.

<!-- Markers: [CHECK] = confirm before submitting. -->

## Abstract

A high price alone is insufficient to prove collusion; a competitive seller could price high for many reasons: market demand or supply-chain issues. I test whether a linear probe on a pricing agent's activations can tell a collusion cue from an innocent cue when the prices match.

In this method, which I call **matched-outcome probing**, the agent gets either a collusion cue (the rival will hold $1.90 if I keep my price high) or an innocent cue (the rival charges $1.90 because its costs rose). I keep only the pairs where both runs price high and within $0.05 of each other, so price cannot separate them.

A logistic-regression probe trained on the model's residual-stream activations separates the cue types with 100% accuracy on seen cue wordings, and 97.6% (Qwen2.5-7B) and 99.0% (Qwen2.5-14B) on wordings it never saw, where a TF-IDF baseline drops to 82–85%. On Llama-3.1-8B the probe still beats TF-IDF on held-out wordings (89.5% vs 81.1%), but the positive control fails there, so that result is exploratory.

The probe's transfer to uncued collusion is weak. When the price history shows the rival copying the agent's previous price, the probe barely separates this from the reverse, where the agent copies the rival, though it does separate it from shuffled prices. On Llama, a change in history format alone pushes every uncued run past the flag threshold.

The instructed-collusion probes separate cues beyond exact word overlap, but they don't catch collusion nobody asked for.

*(Word count: [CHECK] — keep 150–250.)*

