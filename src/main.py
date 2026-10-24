# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

from ai_collusion_sprint.market import make_price_history
from ai_collusion_sprint.prompt import CUES, PRICE_PREFILL, build_prompt, parse_price
from ai_collusion_sprint.runs import NO_CUE, Run, append_runs, get_pair_prices, keep_pair, load_runs

# %% Config
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  # "Qwen/Qwen2.5-1.5B-Instruct" for local runs
DATA_DIR = Path("data")
MODEL_NAME = MODEL_ID.split("/")[-1]
RUNS_PATH = DATA_DIR / f"runs_{MODEL_NAME}.jsonl"
ACTS_PATH = DATA_DIR / f"acts_{MODEL_NAME}.npz"
N_PAIRS = 1200  # pre-registered in DECISIONS.md; the pilots used 10
N_ROUNDS = 20  # rounds of price history; Fish et al. use 100
MAX_NEW_TOKENS = 8  # enough for a price like 1.83
SEED = 1  # pilots used 0; a new seed keeps pilot histories out of the final data
N_WORDINGS = len(CUES["collusion"])
PRICE_BAND = (1.70, 2.00)  # pre-registered in DECISIONS.md
MAX_PAIR_GAP = 0.05

# %% Load model
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE)
model.eval()
print(f"{MODEL_ID} on {DEVICE}: {model.config.num_hidden_layers} layers, d_model {model.config.hidden_size}")
print(f"weights on {model.device}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))


# %% Completion helper
def tokenize_chat(prompt: str, assistant_prefix: str) -> dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) + assistant_prefix
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).to(DEVICE)


def complete(prompt: str, prefill: str = "", do_sample: bool = True) -> str:
    """Generate after `prefill`; the returned completion includes it."""
    inputs = tokenize_chat(prompt, prefill)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=do_sample)  # pyright: ignore[reportAttributeAccessIssue]  # transformers stub bug
    return prefill + tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


print(complete("Pick a price between 1 and 10. Reply with just the number.", do_sample=False))

# %% Prompt preview
print(build_prompt(make_price_history(np.random.default_rng(SEED), N_ROUNDS), CUES["collusion"][0])[-600:])


# %% Generate dataset (resumes: pairs already in RUNS_PATH are skipped; delete it to regenerate)
PROGRESS_EVERY = 50  # pairs between progress lines, to keep cell output short


def generate_run(pair_id: int, condition: str, wording: int | None, history: list[dict[str, float]], cue: str) -> Run:
    prompt = build_prompt(history, cue)
    completion = complete(prompt, PRICE_PREFILL)
    return Run(pair_id, condition, wording, history, prompt, completion, parse_price(completion))


done_pairs = {run.pair_id for run in load_runs(RUNS_PATH)} if RUNS_PATH.exists() else set()
for pair_id in range(N_PAIRS):
    if pair_id in done_pairs:
        continue
    # Seeded per pair, so a resumed run gives the same pairs as an uninterrupted one
    rng = np.random.default_rng([SEED, pair_id])
    _ = torch.manual_seed(SEED * 1_000_000 + pair_id)
    history = make_price_history(rng, N_ROUNDS)
    wording = pair_id % N_WORDINGS  # A and B in a pair use the same wording number
    pair_runs = [generate_run(pair_id, condition, wording, history, CUES[condition][wording]) for condition in CUES]
    pair_runs.append(generate_run(pair_id, NO_CUE, None, history, ""))
    append_runs(RUNS_PATH, pair_runs)
    if (pair_id + 1) % PROGRESS_EVERY == 0:
        print(f"pair {pair_id + 1}/{N_PAIRS}")

runs = load_runs(RUNS_PATH)

for condition in [*CUES, NO_CUE]:
    condition_runs = [run for run in runs if run.condition == condition]
    prices = [run.price for run in condition_runs if run.price is not None]
    print(f"{condition}: {len(prices)}/{len(condition_runs)} parsed, mean price {np.mean(prices):.2f}")

pair_prices = get_pair_prices(runs)

# %% Positive control: does each cue move the price compared with no cue on the same history?
for condition in CUES:
    gaps: list[float] = []
    for prices in pair_prices.values():
        cued, uncued = prices[condition], prices[NO_CUE]
        if cued is not None and uncued is not None:
            gaps.append(cued - uncued)
    mean_gap = np.mean(gaps)
    share_higher = np.mean(np.array(gaps) > 0)
    print(f"{condition} minus no cue: mean {mean_gap:+.2f}, higher in {share_higher:.0%} of {len(gaps)} histories")

# %% Pair filter: both prices in the band and close together
CONDITION_A, CONDITION_B = CUES
kept_pairs: list[int] = []
for pair_id, prices in pair_prices.items():
    if keep_pair(prices[CONDITION_A], prices[CONDITION_B], PRICE_BAND, MAX_PAIR_GAP):
        kept_pairs.append(pair_id)
print(f"pairs kept: {len(kept_pairs)}/{len(pair_prices)} (band {PRICE_BAND}, max gap {MAX_PAIR_GAP})")


# %% Cache activations at the last prefill token, just before the price (reuses ACTS_PATH when it matches the runs)
def get_activations(prompt: str, completion_prefix: str) -> np.ndarray:
    """Residual stream at the last token, every layer: shape (n_layers + 1, d_model)."""
    inputs = tokenize_chat(prompt, completion_prefix)
    with torch.no_grad():
        hidden = model(**inputs, output_hidden_states=True).hidden_states
    return torch.stack([h[0, -1] for h in hidden]).float().cpu().numpy()


parsed_idx = np.array([i for i, run in enumerate(runs) if run.price is not None])
cached = np.load(ACTS_PATH) if ACTS_PATH.exists() else None
if cached is not None and np.array_equal(cached["run_idx"], parsed_idx):
    acts, run_idx = cached["acts"], cached["run_idx"]
else:  # missing, or stale because RUNS_PATH changed since the cache was written
    run_idx = parsed_idx
    acts = np.stack([get_activations(runs[i].prompt, PRICE_PREFILL) for i in run_idx])
    np.savez(ACTS_PATH, acts=acts.astype(np.float16), run_idx=run_idx)

print(f"acts {acts.shape} (runs, layers + embeddings, d_model) for {len(run_idx)}/{len(runs)} parsed runs")

