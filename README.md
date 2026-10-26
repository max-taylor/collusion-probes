# Matched-outcome probing

Can a linear probe on an LLM pricing agent's activations tell a collusion cue from an innocent cue when both lead to the same price? Apart Research AI Collusion Research Sprint, October 2026 (Track 1, Q1.3). The report is `report/report.md` (PDF: `report/report.pdf`).

## Prior work disclosure

No code in this repo predates the sprint (Oct 23–25 2026). Reused external material, all disclosed in `PRIOR_WORK.md`:

- Prompt template, prefix P1 and market parameters from Fish et al. 2024 (arXiv 2404.00806).
- History order and demand constants from Romero, Sauer and Peist, [`algorithmic_pricing_llms`](https://github.com/luciasauer/algorithmic_pricing_llms).
- Unmodified Hugging Face weights: Qwen2.5-1.5B/7B/14B-Instruct, Llama-3.1-8B-Instruct.

`docs/prompt_provenance.md` lists every deviation from Fish et al.

## Architecture

One pricing decision per prompt. Each random price history runs three times: with a collusion cue (A), an innocent cue (B) and no cue. Pairs where A and B price the same are kept. The residual stream at the last prefill token is cached at every layer, and a logistic-regression probe is trained per layer on A vs B. An extension scores uncued histories (rival follows me, shuffled, I follow the rival) with the saved probe.

## Setup

```bash
uv sync
```

Python 3.11–3.12. Locked versions in `uv.lock` (torch 2.14.1, transformers 5.20.2, scikit-learn 1.9.1). The report's runs used bf16 on RunPod: 1× RTX PRO 4500 (32 GB) for 7B and Llama-3.1-8B, 1× RTX PRO 6000 (96 GB) for 14B. `docs/runpod_setup.md` covers the pod. Llama needs an accepted Meta licence on Hugging Face.

## Reproduction

`src/main.py` is a notebook-style file split into `# %%` cells. Run it in any Jupyter-compatible editor (VS Code, molten-nvim, `jupytext`), top to bottom.

1. Set `MODEL_ID` in the Config cell (`Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct` or `meta-llama/Llama-3.1-8B-Instruct`). Every other value is pre-registered in `DECISIONS.md`; leave it.
2. Run the cells in order. Generation and activation caching resume from `data/` if interrupted.
3. Appendix D of the report maps each table and figure to its cell.

Outputs go to `data/` (gitignored, about 4.3 GB for all models):

| File | Contents |
|---|---|
| `runs_<model>.jsonl`, `runs_C_<model>.jsonl` | Prompts, completions and prices for A/B/no-cue and C runs |
| `acts_<model>.npz`, `acts_C_<model>.npz` | Activations at every layer |
| `probes_<model>.pkl` | One saved probe per layer, plus the headline layer |
| `lowo_curve_<model>.npz` | Held-out-wording accuracy at every layer, for the model comparison chart |
| `c_groups_<model>.npz` | Probe scores for A, B, no-cue and the C twins, for the C comparison chart |
| `key_chart_<model>.png`, `c_scores_<model>.png`, `c_layers_<model>.png`, `model_comparison.png`, `c_scores_comparison.png` | Report figures |

The runs, activations, probes and C scores for all three models are on Hugging Face: [`maxtaylordev/collusion-probes`](https://huggingface.co/datasets/maxtaylordev/collusion-probes). Download them into `data/` to skip generation (`hf download maxtaylordev/collusion-probes --repo-type dataset --local-dir data`). The logged seeds also regenerate them.

## Key concepts

- **Pre-registration.** `DECISIONS.md` holds every design choice with a UTC timestamp, logged before the data it governs. Changes are dated change-log lines, never edits.
- **Findings log.** `FINDINGS.md` has one dated entry per result.

## Project structure

```
src/main.py               experiment cells
src/collusion_probes/     helpers: prompt, market, runs, probe
docs/                     scope, prompt provenance, pod setup
report/                   report (Markdown and PDF) and figures
scripts/start_jupyter.sh  starts Jupyter on the pod
```
