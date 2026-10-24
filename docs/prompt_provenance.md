# Prompt and market provenance

`src/collusion_probes/prompt.py` and `market.py` follow two references as closely as a single-decision, synthetic-history setup allows.

- **Fish et al. 2024** ([arXiv 2404.00806](https://arxiv.org/abs/2404.00806)): primary source.
- **Sauer, Romero and Peist** ([algorithmic_pricing_llms](https://github.com/luciasauer/algorithmic_pricing_llms)): a reimplementation of Fish. It's used to settle details the paper leaves out.

## Taken directly

| Item | Value | Source |
|---|---|---|
| Prefix P1 | Verbatim | Fish App. G.1 |
| Prompt template, input sections | Verbatim | Fish App. G.2.1 |
| Price line of the response template ("My chosen price:") | Verbatim | Fish App. G.2.1 |
| Market-data line format (`Round N:`, `- My price:` …) | Verbatim | Fish App. G.2.2 example |
| History order | Newest round first | Sauer `prompt_manager.py` (Fish doesn't say) |
| Marginal cost | $1 (α = 1) | Fish §2.1, Sauer `duopoly.py` |
| Max willingness to pay | $4.51 | Fish App. G.2.2, Sauer `duopoly.py` |
| Demand | Logit: a = 2, a₀ = 0, μ = 0.25, β = 100 | Fish §2.1, Sauer `experiment.py` |
| Nash / monopoly price | 1.47 / 1.92 | Computed from the demand above |

Check: `get_quantity(1.5, 2.76)` gives 87.58, which matches the quantity in Fish's worked example.

## Our deviations

- **Cues:** the collusion and innocent cue sentences are ours. Each has 5 wordings, and every wording states the same rival price (`CUE_PRICE`, $1.90), so A and B differ only in why the rival prices high, not in the number. The first pilot used cues without a number; the innocent cue then left prices unchanged and only 1 of 10 pairs survived the filter. They go in INSIGHTS.txt and are the experimental manipulation. No-cue runs (the positive control, and the C extension) leave INSIGHTS.txt empty, so their prompt is Fish's unchanged.
- **Price-only response:** Fish's response template asks for observations, new PLANS.txt and new INSIGHTS.txt before the price. We keep only the price line, and drop the closing note about overwriting files. Otherwise the model's own written reasoning would sit between the cue and the price, so A and B runs would differ in more than the cue, and a monitor could just read the reasoning. The file descriptions in the input stay verbatim, even though the model no longer writes to the files.
- **Prefill:** the assistant turn starts with "My chosen price:", so the model answers with the number only. Activations are read at the last prefill token.
- **Empty PLANS.txt:** this matches Fish's first round, before the model has written any plans.
- **Synthetic history:** each round draws both prices uniformly between the Nash and monopoly prices. The C extension, if run, uses the same range but makes the rival copy my previous round's price (C-follow), plus a twin with those rival prices shuffled into random order (C-shuffle). Quantities and profits come from the real demand formula. Fish and Sauer use history from a live simulation.
- **History length:** `N_ROUNDS` defaults to 20 so local runs stay fast. Fish uses 100.
- **One decision per prompt:** there is no multi-round loop and no competitor agent. The competitor appears only in the history.
- **Placeholders:** `<fill in here>` replaces Fish's `¡fill in here¿`, a LaTeX rendering artefact of angle brackets.

## Sauer choices we did not follow

Sauer swaps "files" for "sections", drops the response template and reads the price through Mistral's structured output. We keep Fish's wording and his price line, and parse the price from text.
