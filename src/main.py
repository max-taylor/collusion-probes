# %% Imports
import pickle
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.core.getipython import get_ipython
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# Re-import edited helpers in src/collusion_probes/ on every cell run, without a kernel restart
ipython = get_ipython()
if ipython is not None:
    _ = ipython.run_line_magic("load_ext", "autoreload")
    _ = ipython.run_line_magic("autoreload", "2")

from collusion_probes.market import make_price_history
from collusion_probes.probe import get_mean_ci, make_probe, make_text_baseline
from collusion_probes.prompt import CUES, PRICE_PREFILL, build_prompt, parse_price
from collusion_probes.runs import NO_CUE, Run, append_runs, get_pair_prices, keep_pair, load_runs

# %% Config
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  # "Qwen/Qwen2.5-1.5B-Instruct" for local runs
DATA_DIR = Path("data")
MODEL_NAME = MODEL_ID.split("/")[-1]
RUNS_PATH = DATA_DIR / f"runs_{MODEL_NAME}.jsonl"
ACTS_PATH = DATA_DIR / f"acts_{MODEL_NAME}.npz"
PROBES_PATH = DATA_DIR / f"probes_{MODEL_NAME}.pkl"
N_PAIRS = 1200  # pre-registered in DECISIONS.md; the pilots used 10
N_ROUNDS = 20  # rounds of price history; Fish et al. use 100
MAX_NEW_TOKENS = 8  # enough for a price like 1.83
SEED = 1  # pilots used 0; a new seed keeps pilot histories out of the final data
CONDITION_A, CONDITION_B = CUES  # collusion, innocent
N_WORDINGS = len(CUES[CONDITION_A])
PRICE_BAND = (1.70, 2.00)  # pre-registered in DECISIONS.md
MAX_PAIR_GAP = 0.05
N_BOOTSTRAP = 10_000  # positive-control CI resamples, pre-registered in DECISIONS.md
MIN_KEPT_PAIRS = 250  # stopping rule in DECISIONS.md: below this, add blocks of 400 pairs
MIN_PARSE_RATE = 0.95  # lower means the prefill or price parser broke
SPLIT_SEEDS = range(5)  # probe train/test splits, pre-registered in DECISIONS.md
TEST_SIZE = 0.2
N_CV_FOLDS = 5  # CV inside the training split picks the headline layer

# %% Load model (lazily: only on first generation or activation call, so cached data needs no download)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@cache
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    print(f"{MODEL_ID} on {DEVICE}: {model.config.num_hidden_layers} layers, d_model {model.config.hidden_size}")
    print(f"weights on {model.device}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
    return tokenizer, model


# %% Completion helper
def tokenize_chat(prompt: str, assistant_prefix: str) -> dict[str, torch.Tensor]:
    tokenizer, _ = load_model()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) + assistant_prefix
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).to(DEVICE)


def complete(prompt: str, prefill: str = "", do_sample: bool = True) -> str:
    """Generate after `prefill`; the returned completion includes it."""
    tokenizer, model = load_model()
    inputs = tokenize_chat(prompt, prefill)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=do_sample)  # pyright: ignore[reportAttributeAccessIssue]  # transformers stub bug
    return prefill + tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

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
    assert len(prices) >= MIN_PARSE_RATE * len(condition_runs), f"{condition}: parse rate below {MIN_PARSE_RATE:.0%}"

pair_prices = get_pair_prices(runs)
assert len(runs) == len(pair_prices) * (len(CUES) + 1), "a pair is missing a run or has a duplicate"
assert all(set(prices) == {*CUES, NO_CUE} for prices in pair_prices.values()), "a pair is missing a condition"
pair_histories: dict[int, list[dict[str, float]]] = {}
assert all(pair_histories.setdefault(run.pair_id, run.history) == run.history for run in runs), "runs in a pair differ in history"

# %% Positive control: does each cue move the price compared with no cue on the same history?
for condition in CUES:
    gaps: list[float] = []
    for prices in pair_prices.values():
        cued, uncued = prices[condition], prices[NO_CUE]
        if cued is not None and uncued is not None:
            gaps.append(cued - uncued)
    mean_gap = np.mean(gaps)
    share_higher = np.mean(np.array(gaps) > 0)
    boot_rng = np.random.default_rng(SEED)
    boot_means = boot_rng.choice(gaps, size=(N_BOOTSTRAP, len(gaps))).mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    print(f"{condition} minus no cue: mean {mean_gap:+.2f}, higher in {share_higher:.0%} of {len(gaps)} histories")
    # Pre-registered pass rule applies to the collusion cue only: CI entirely above 0
    verdict = f" {'PASS' if ci_low > 0 else 'FAIL'}" if condition == CONDITION_A else ""
    print(f"  95% bootstrap CI [{ci_low:+.3f}, {ci_high:+.3f}]{verdict}")

# %% Pair filter: both prices in the band and close together
kept_pairs: list[int] = []
for pair_id, prices in pair_prices.items():
    if keep_pair(prices[CONDITION_A], prices[CONDITION_B], PRICE_BAND, MAX_PAIR_GAP):
        kept_pairs.append(pair_id)
enough_pairs = "PASS" if len(kept_pairs) >= MIN_KEPT_PAIRS else f"FAIL: below {MIN_KEPT_PAIRS}, add 400 pairs"
print(f"pairs kept: {len(kept_pairs)}/{len(pair_prices)} (band {PRICE_BAND}, max gap {MAX_PAIR_GAP}) {enough_pairs}")


# %% Cache activations at the last prefill token, just before the price (reuses ACTS_PATH when it matches the runs)
def get_activations(prompt: str, completion_prefix: str) -> np.ndarray:
    """Residual stream at the last token, every layer: shape (n_layers + 1, d_model)."""
    _, model = load_model()
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
kept_set = set(kept_pairs)
kept_ab_idx = {i for i, run in enumerate(runs) if run.pair_id in kept_set and run.condition != NO_CUE}
assert kept_ab_idx <= set(run_idx.tolist()), "a kept A/B run has no cached activations"


# %% Main probe: per layer, collusion (1) vs innocent (0) on kept pairs, split by pair (takes a few minutes)
row_of_run = {run_i: row for row, run_i in enumerate(run_idx.tolist())}
probe_runs = [i for i, run in enumerate(runs) if run.pair_id in kept_set and run.condition != NO_CUE]
probe_acts = acts[[row_of_run[i] for i in probe_runs]].astype(np.float32)  # (runs, layers + embeddings, d_model)
probe_labels = np.array([runs[i].condition == CONDITION_A for i in probe_runs], dtype=int)
probe_pairs = np.array([runs[i].pair_id for i in probe_runs])
probe_wordings = np.array([runs[i].wording for i in probe_runs])
probe_prompts = np.array([runs[i].prompt for i in probe_runs])
n_layers = probe_acts.shape[1]

cv_acc = np.zeros((len(SPLIT_SEEDS), n_layers))  # 5-fold CV inside the training split
test_acc = np.zeros((len(SPLIT_SEEDS), n_layers))
for seed_i, split_seed in enumerate(SPLIT_SEEDS):
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=split_seed)
    train, test = next(splitter.split(probe_acts, probe_labels, groups=probe_pairs))
    assert not set(probe_pairs[train]) & set(probe_pairs[test]), "a pair is on both sides of the split"
    for layer in range(n_layers):
        X = probe_acts[:, layer]
        cv_scores = cross_val_score(
            make_probe(), X[train], probe_labels[train], groups=probe_pairs[train], cv=GroupKFold(N_CV_FOLDS), n_jobs=-1
        )
        cv_acc[seed_i, layer] = cv_scores.mean()
        test_acc[seed_i, layer] = make_probe().fit(X[train], probe_labels[train]).score(X[test], probe_labels[test])


# %% Main probe results: headline layer and accuracy by layer
cv_mean = cv_acc.mean(axis=0)
tied_layers = np.flatnonzero(np.isclose(cv_mean, cv_mean.max()))
headline_layer = int(tied_layers[len(tied_layers) // 2])  # CV only picks it; ties go to the middle tied layer
mean, low, high = get_mean_ci(test_acc[:, headline_layer])
print(f"{len(probe_runs)} runs from {len(kept_set)} pairs, {len(SPLIT_SEEDS)} split seeds")
print(f"headline layer {headline_layer} (CV {cv_acc[:, headline_layer].mean():.0%}): test {mean:.0%} [{low:.0%}, {high:.0%}]")

test_ci = np.array([get_mean_ci(test_acc[:, layer]) for layer in range(n_layers)])
_ = plt.figure(figsize=(6, 2.5))
_ = plt.plot(test_ci[:, 0], label="probe (test)")
_ = plt.fill_between(range(n_layers), test_ci[:, 1], test_ci[:, 2], alpha=0.3)
_ = plt.axhline(0.5, color="grey", linestyle="--", label="chance")
_ = plt.axvline(headline_layer, color="black", linewidth=0.8)
_ = plt.xlabel("layer (0 = embeddings)")
_ = plt.ylabel("accuracy")
_ = plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% Save probes: one per layer, fit on every kept pair, for scoring new runs (extensions)
probes = [make_probe().fit(probe_acts[:, layer], probe_labels) for layer in range(n_layers)]
with PROBES_PATH.open("wb") as f:
    pickle.dump({"probes": probes, "headline_layer": headline_layer}, f)
print(f"saved {len(probes)} probes to {PROBES_PATH}")


# %% Leave-one-wording-out: train on 4 cue wordings, test on the 5th, every layer (takes a few minutes)
lowo_acc = np.zeros((N_WORDINGS, n_layers))
for wording in range(N_WORDINGS):
    train, test = np.flatnonzero(probe_wordings != wording), np.flatnonzero(probe_wordings == wording)
    for layer in range(n_layers):
        X = probe_acts[:, layer]
        lowo_acc[wording, layer] = make_probe().fit(X[train], probe_labels[train]).score(X[test], probe_labels[test])

mean, low, high = get_mean_ci(lowo_acc[:, headline_layer])
per_wording = ", ".join(f"{acc:.0%}" for acc in lowo_acc[:, headline_layer])
print(f"held-out wording, layer {headline_layer}: {mean:.0%} [{low:.0%}, {high:.0%}] (per wording: {per_wording})")


# %% Text baseline: TF-IDF logistic regression on the prompt, same splits as the probe
text_split_acc = np.zeros(len(SPLIT_SEEDS))
for seed_i, split_seed in enumerate(SPLIT_SEEDS):
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=split_seed)
    train, test = next(splitter.split(probe_prompts, probe_labels, groups=probe_pairs))
    text_split_acc[seed_i] = make_text_baseline().fit(probe_prompts[train], probe_labels[train]).score(probe_prompts[test], probe_labels[test])

text_lowo_acc = np.zeros(N_WORDINGS)
for wording in range(N_WORDINGS):
    train, test = np.flatnonzero(probe_wordings != wording), np.flatnonzero(probe_wordings == wording)
    text_lowo_acc[wording] = make_text_baseline().fit(probe_prompts[train], probe_labels[train]).score(probe_prompts[test], probe_labels[test])

for name, accs in [("seen wordings", text_split_acc), ("held-out wording", text_lowo_acc)]:
    mean, low, high = get_mean_ci(accs)
    print(f"TF-IDF, {name}: {mean:.0%} [{low:.0%}, {high:.0%}]")
print("TF-IDF per held-out wording: " + ", ".join(f"{acc:.0%}" for acc in text_lowo_acc))

lowo_ci = np.array([get_mean_ci(lowo_acc[:, layer]) for layer in range(n_layers)])
_ = plt.figure(figsize=(6, 2.5))
_ = plt.plot(lowo_ci[:, 0], label="probe (held-out wording)")
_ = plt.fill_between(range(n_layers), lowo_ci[:, 1], lowo_ci[:, 2], alpha=0.3)
_ = plt.axhline(text_lowo_acc.mean(), color="tab:orange", label="TF-IDF (held-out wording)")
_ = plt.axhline(0.5, color="grey", linestyle="--", label="chance")
_ = plt.xlabel("layer (0 = embeddings)")
_ = plt.ylabel("accuracy")
_ = plt.legend(fontsize=8)
plt.tight_layout()
plt.show()
