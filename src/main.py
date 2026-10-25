# %% Imports
import hashlib
import pickle
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.core.getipython import get_ipython
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# Re-import edited helpers in src/collusion_probes/ on every cell run, without a kernel restart
ipython = get_ipython()
if ipython is not None:
    _ = ipython.run_line_magic("load_ext", "autoreload")
    _ = ipython.run_line_magic("autoreload", "2")

from collusion_probes.market import make_follow_history, make_price_history, shuffle_rival_prices, swap_roles
from collusion_probes.probe import get_bootstrap_ci, get_pair_accuracy_ci, make_probe, make_text_baseline, pick_headline_layer
from collusion_probes.prompt import CUES, PRICE_PREFILL, build_prompt, parse_price
from collusion_probes.runs import NO_CUE, Run, append_runs, get_pair_prices, keep_pair, load_runs

# %% Config
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"  # "Qwen/Qwen2.5-1.5B-Instruct" for local runs
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
    ci_low, ci_high = get_bootstrap_ci(gaps, N_BOOTSTRAP, SEED)
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


def load_activations(runs: list[Run], path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Activations and run indices for every parsed run, from `path` if it matches the runs, else computed and saved."""
    parsed_idx = np.array([i for i, run in enumerate(runs) if run.price is not None])
    content = MODEL_ID + PRICE_PREFILL + "".join(runs[i].prompt for i in parsed_idx)
    key = hashlib.sha256(content.encode()).hexdigest()  # catches changed prompts or model with the same parsed rows
    cached = np.load(path) if path.exists() else None
    # Caches written before the key existed are checked on run indices only
    if cached is not None and np.array_equal(cached["run_idx"], parsed_idx) and str(cached.get("key", key)) == key:
        return cached["acts"], cached["run_idx"]
    # missing, or stale because the runs, prompts or model changed since the cache was written
    acts = np.stack([get_activations(runs[i].prompt, PRICE_PREFILL) for i in parsed_idx])
    np.savez(path, acts=acts.astype(np.float16), run_idx=parsed_idx, key=key)
    return acts, parsed_idx


acts, run_idx = load_activations(runs, ACTS_PATH)
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
probe_prices = np.array([[runs[i].price] for i in probe_runs])
n_layers = probe_acts.shape[1]
splits = [
    next(GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed).split(probe_acts, groups=probe_pairs))
    for seed in SPLIT_SEEDS
]
assert all(not set(probe_pairs[train]) & set(probe_pairs[test]) for train, test in splits), "a pair is on both sides"

cv_acc = np.zeros((len(SPLIT_SEEDS), n_layers))  # 5-fold CV inside the training split
test_correct = np.full((len(SPLIT_SEEDS), len(probe_runs), n_layers), np.nan)  # NaN where the run is in training
for seed_i, (train, test) in enumerate(splits):
    for layer in range(n_layers):
        X = probe_acts[:, layer]
        cv_scores = cross_val_score(
            make_probe(), X[train], probe_labels[train], groups=probe_pairs[train], cv=GroupKFold(N_CV_FOLDS), n_jobs=-1
        )
        cv_acc[seed_i, layer] = cv_scores.mean()
        test_correct[seed_i, test, layer] = make_probe().fit(X[train], probe_labels[train]).predict(X[test]) == probe_labels[test]


def get_split_accuracy_ci(correct: np.ndarray, keep: np.ndarray | None = None) -> tuple[float, float, float]:
    """Pair-bootstrap accuracy over every seed's test runs; correct is (seeds, runs), NaN for training runs."""
    tested = ~np.isnan(correct) & (True if keep is None else keep)
    return get_pair_accuracy_ci(correct[tested], np.broadcast_to(probe_pairs, correct.shape)[tested], N_BOOTSTRAP, SEED)


# %% Main probe results: headline layer, accuracy and false-positive rate
cv_mean = cv_acc.mean(axis=0)
headline_layer = pick_headline_layer(cv_mean)  # CV only picks it; ties go to the middle tied layer
test_ci = np.array([get_split_accuracy_ci(test_correct[:, :, layer]) for layer in range(n_layers)])
mean, low, high = test_ci[headline_layer]
print(f"{len(probe_runs)} runs from {len(kept_set)} pairs, {len(SPLIT_SEEDS)} split seeds, pair-bootstrap CIs")
print(f"headline layer {headline_layer} (CV {cv_mean[headline_layer]:.0%}): test {mean:.1%} [{low:.1%}, {high:.1%}]")
# False-positive rate: held-out B runs flagged collusive at the pre-registered p > 0.5 (predict's threshold)
mean, low, high = get_split_accuracy_ci(test_correct[:, :, headline_layer], keep=probe_labels == 0)
print(f"false-positive rate on held-out B: {1 - mean:.1%} [{1 - high:.1%}, {1 - low:.1%}]")

# %% Save probes: one per layer, fit on every kept pair, for scoring new runs (extensions)
probes = [make_probe().fit(probe_acts[:, layer], probe_labels) for layer in range(n_layers)]
with PROBES_PATH.open("wb") as f:
    pickle.dump({"probes": probes, "headline_layer": headline_layer}, f)
print(f"saved {len(probes)} probes to {PROBES_PATH}")


# %% Leave-one-wording-out: train on 4 cue wordings, test on the 5th, every layer (takes a few minutes)
lowo_correct = np.zeros((len(probe_runs), n_layers))  # every run is tested once, when its wording is held out
for wording in range(N_WORDINGS):
    train, test = np.flatnonzero(probe_wordings != wording), np.flatnonzero(probe_wordings == wording)
    for layer in range(n_layers):
        X = probe_acts[:, layer]
        lowo_correct[test, layer] = make_probe().fit(X[train], probe_labels[train]).predict(X[test]) == probe_labels[test]

lowo_ci = np.array([get_pair_accuracy_ci(lowo_correct[:, layer], probe_pairs, N_BOOTSTRAP, SEED) for layer in range(n_layers)])
mean, low, high = lowo_ci[headline_layer]
per_wording = ", ".join(f"{lowo_correct[probe_wordings == w, headline_layer].mean():.0%}" for w in range(N_WORDINGS))
print(f"held-out wording, layer {headline_layer}: {mean:.1%} [{low:.1%}, {high:.1%}] (per wording: {per_wording})")
is_b = probe_labels == 0
mean, low, high = get_pair_accuracy_ci(lowo_correct[is_b, headline_layer], probe_pairs[is_b], N_BOOTSTRAP, SEED)
print(f"false-positive rate on held-out-wording B: {1 - mean:.1%} [{1 - high:.1%}, {1 - low:.1%}]")


# %% Nested layer choice (post-hoc, DECISIONS.md): each outer split / held-out wording picks its own layer without its test data
split_layers = [pick_headline_layer(cv_acc[seed_i]) for seed_i in range(len(SPLIT_SEEDS))]
nested_correct = np.stack([test_correct[seed_i, :, layer] for seed_i, layer in enumerate(split_layers)])
mean, low, high = get_split_accuracy_ci(nested_correct)
print(f"nested test: layers {split_layers}, {mean:.1%} [{low:.1%}, {high:.1%}]")

# Inner CV leaves one of the 4 training wordings out, so the layer is chosen for carrying over to new wordings
fold_layers: list[int] = []
nested_lowo_correct = np.zeros(len(probe_runs))
for wording in range(N_WORDINGS):
    train, test = np.flatnonzero(probe_wordings != wording), probe_wordings == wording
    inner_acc = np.array([
        cross_val_score(
            make_probe(), probe_acts[train, layer], probe_labels[train], groups=probe_wordings[train], cv=GroupKFold(N_WORDINGS - 1), n_jobs=-1
        ).mean()
        for layer in range(n_layers)
    ])
    fold_layers.append(pick_headline_layer(inner_acc))
    nested_lowo_correct[test] = lowo_correct[test, fold_layers[-1]]
mean, low, high = get_pair_accuracy_ci(nested_lowo_correct, probe_pairs, N_BOOTSTRAP, SEED)
print(f"nested held-out wording: layers {fold_layers}, {mean:.1%} [{low:.1%}, {high:.1%}]")


# %% Baselines on the same splits: TF-IDF on the prompt, price alone, and the probe with shuffled labels
text_correct = np.full((len(SPLIT_SEEDS), len(probe_runs)), np.nan)
price_correct = np.full((len(SPLIT_SEEDS), len(probe_runs)), np.nan)
shuffled_correct = np.full((len(SPLIT_SEEDS), len(probe_runs)), np.nan)
shuffled_labels = np.random.default_rng(SEED).permutation(probe_labels)
X = probe_acts[:, headline_layer]
for seed_i, (train, test) in enumerate(splits):
    model = make_text_baseline().fit(probe_prompts[train], probe_labels[train])
    text_correct[seed_i, test] = model.predict(probe_prompts[test]) == probe_labels[test]
    price_correct[seed_i, test] = make_probe().fit(probe_prices[train], probe_labels[train]).predict(probe_prices[test]) == probe_labels[test]
    shuffled_correct[seed_i, test] = make_probe().fit(X[train], shuffled_labels[train]).predict(X[test]) == probe_labels[test]

text_lowo_correct = np.zeros(len(probe_runs))
for wording in range(N_WORDINGS):
    train, test = np.flatnonzero(probe_wordings != wording), np.flatnonzero(probe_wordings == wording)
    model = make_text_baseline().fit(probe_prompts[train], probe_labels[train])
    text_lowo_correct[test] = model.predict(probe_prompts[test]) == probe_labels[test]

text_ci = get_split_accuracy_ci(text_correct)
text_lowo_ci = get_pair_accuracy_ci(text_lowo_correct, probe_pairs, N_BOOTSTRAP, SEED)
price_ci = get_split_accuracy_ci(price_correct)
for name, (mean, low, high) in [
    ("TF-IDF, seen wordings", text_ci),
    ("TF-IDF, held-out wording", text_lowo_ci),
    ("price only", price_ci),
    (f"shuffled-label probe, layer {headline_layer}", get_split_accuracy_ci(shuffled_correct)),
]:
    print(f"{name}: {mean:.1%} [{low:.1%}, {high:.1%}]")
print("TF-IDF per held-out wording: " + ", ".join(f"{text_lowo_correct[probe_wordings == w].mean():.0%}" for w in range(N_WORDINGS)))


# %% Shuffled-label null distribution (post-hoc, DECISIONS.md): many permutations, not one, at the headline layer
N_PERMUTATIONS = 100


def get_shuffled_accuracy(permutation: int) -> float:
    labels = np.random.default_rng([SEED, permutation]).permutation(probe_labels)
    correct = np.full((len(SPLIT_SEEDS), len(probe_runs)), np.nan)
    for seed_i, (train, test) in enumerate(splits):
        correct[seed_i, test] = make_probe().fit(X[train], labels[train]).predict(X[test]) == probe_labels[test]
    return get_split_accuracy_ci(correct)[0]


null_acc = np.array(Parallel(n_jobs=-1)(delayed(get_shuffled_accuracy)(p) for p in range(N_PERMUTATIONS)))
observed = get_split_accuracy_ci(shuffled_correct)[0]
null_low, null_high = np.percentile(null_acc, [2.5, 97.5])
print(f"shuffled-label null, {N_PERMUTATIONS} permutations: mean {null_acc.mean():.1%}, 95% range [{null_low:.1%}, {null_high:.1%}]")
print(f"  single shuffle {observed:.1%} sits at percentile {(null_acc < observed).mean():.0%}; real probe {test_ci[headline_layer, 0]:.1%}")


# %% Key chart: probe accuracy by layer against chance and the baselines (saved for the report)
_ = plt.figure(figsize=(6, 2.8))
for ci, label, color in [(test_ci, "probe, seen wordings", "tab:blue"), (lowo_ci, "probe, held-out wording", "tab:green")]:
    _ = plt.plot(ci[:, 0], label=label, color=color)
    _ = plt.fill_between(range(n_layers), ci[:, 1], ci[:, 2], alpha=0.25, color=color)
_ = plt.axhline(text_lowo_ci[0], color="tab:orange", label="TF-IDF, held-out wording")
_ = plt.axhline(price_ci[0], color="tab:red", linestyle=":", label="price only")
_ = plt.axhline(0.5, color="grey", linestyle="--", label="chance")
_ = plt.axvline(headline_layer, color="black", linewidth=0.8)
_ = plt.xlabel("layer (0 = embeddings)")
_ = plt.ylabel("accuracy")
_ = plt.legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.savefig(DATA_DIR / f"key_chart_{MODEL_NAME}.png", dpi=200)
plt.show()

# %% Extension 1 config: uncued collusion, C-follow vs C-shuffle and C-lead twins (pre-registered in DECISIONS.md)
C_PILOT = False  # 10-pair pilot (seed 3, excluded); set False for the 300 pre-registered pairs (seed 2)
C_SEED, N_C_PAIRS = (3, 10) if C_PILOT else (2, 300)
C_PREFIX = "pilotC4_" if C_PILOT else ""  # pilotC_ to pilotC3_ are superseded pilots (DECISIONS.md change log)
C_N_ROUNDS = 50  # longer and wider than A/B so the following is visible; pilot 1 at 20 rounds showed no reaction
C_PRICE_RANGE = (1.20, 2.20)
C_HOLD_ROUNDS = (3, 6)  # I hold each price this many rounds, so the rival visibly matches each change a round later
C_RUNS_PATH = DATA_DIR / f"{C_PREFIX}runs_C_{MODEL_NAME}.jsonl"
C_ACTS_PATH = DATA_DIR / f"{C_PREFIX}acts_C_{MODEL_NAME}.npz"
C_FOLLOW, C_SHUFFLE, C_LEAD = "follow", "shuffle", "lead"
C_CONTROLS = (C_SHUFFLE, C_LEAD)  # shuffle breaks the copying; lead keeps equal prices but I copy the rival
PRICE_MATCH = 0.02  # price-leak check: B runs within this of a C-follow price count as matched

# %% Extension 1: generate C twins, no cue (resumes like the A/B generation)
done_c_pairs = {run.pair_id for run in load_runs(C_RUNS_PATH)} if C_RUNS_PATH.exists() else set()
for pair_id in range(N_C_PAIRS):
    if pair_id in done_c_pairs:
        continue
    rng = np.random.default_rng([C_SEED, pair_id])
    _ = torch.manual_seed(C_SEED * 1_000_000 + pair_id)
    follow_history = make_follow_history(rng, C_N_ROUNDS, C_PRICE_RANGE, C_HOLD_ROUNDS)
    twin_histories = {
        C_FOLLOW: follow_history,
        C_SHUFFLE: shuffle_rival_prices(rng, follow_history),
        C_LEAD: swap_roles(follow_history),
    }
    append_runs(C_RUNS_PATH, [generate_run(pair_id, name, None, history, "") for name, history in twin_histories.items()])
    if (pair_id + 1) % PROGRESS_EVERY == 0:
        print(f"C pair {pair_id + 1}/{N_C_PAIRS}")

c_runs = load_runs(C_RUNS_PATH)
c_pair_prices = get_pair_prices(c_runs)
assert len(c_runs) == 3 * len(c_pair_prices), "a C pair is missing a twin or has a duplicate"
for name in (C_FOLLOW, *C_CONTROLS):
    twin_runs = [run for run in c_runs if run.condition == name]
    prices = [run.price for run in twin_runs if run.price is not None]
    profit = np.mean([r["profit"] for run in twin_runs for r in run.history])
    print(f"C-{name}: {len(prices)}/{len(twin_runs)} parsed, mean price {np.mean(prices):.2f}, mean history profit {profit:.1f}")
    assert len(prices) >= MIN_PARSE_RATE * len(twin_runs), f"C-{name}: parse rate below {MIN_PARSE_RATE:.0%}"

# %% Extension 1: score C twins and out-of-sample A/B runs with the saved headline-layer probe
# Score = probe logit (decision_function): P(collusion) saturates at 0 or 1, which hid any C difference in pilot 1
c_acts, c_run_idx = load_activations(c_runs, C_ACTS_PATH)
with PROBES_PATH.open("rb") as f:
    saved = pickle.load(f)
c_layer = saved["headline_layer"]
c_probe = saved["probes"][c_layer]
c_score = dict(zip(c_run_idx.tolist(), c_probe.decision_function(c_acts[:, c_layer].astype(np.float32))))

# The saved probes were fit on every kept pair, so only A/B runs from dropped pairs are out of sample
oos_rows = [row for row, i in enumerate(run_idx.tolist()) if runs[i].pair_id not in kept_set and runs[i].condition in CUES]
oos_scores = c_probe.decision_function(acts[oos_rows, c_layer].astype(np.float32))
oos_conditions = np.array([runs[run_idx[row]].condition for row in oos_rows])
oos_prices = np.array([runs[run_idx[row]].price for row in oos_rows])
# Post-hoc uncued reference: no-cue runs never trained the probe, so every one is out of sample
no_cue_rows = [row for row, i in enumerate(run_idx.tolist()) if runs[i].condition == NO_CUE]
no_cue_scores = c_probe.decision_function(acts[no_cue_rows, c_layer].astype(np.float32))
print(f"layer {c_layer}: scored {len(c_score)} C runs, {len(oos_rows)} out-of-sample A/B runs and {len(no_cue_rows)} no-cue runs")

# %% Extension 1 results: pre-registered tests (DECISIONS.md)
twin_score: dict[int, dict[str, float]] = {}
twin_price: dict[int, dict[str, float]] = {}
for i, run in enumerate(c_runs):
    if i in c_score and run.price is not None:
        twin_score.setdefault(run.pair_id, {})[run.condition] = c_score[i]
        twin_price.setdefault(run.pair_id, {})[run.condition] = run.price
both_ids = [pid for pid, s in twin_score.items() if len(s) == 1 + len(C_CONTROLS)]
for control in C_CONTROLS:
    score_gaps = np.array([twin_score[pid][C_FOLLOW] - twin_score[pid][control] for pid in both_ids])
    price_gaps = np.array([twin_price[pid][C_FOLLOW] - twin_price[pid][control] for pid in both_ids])
    low, high = get_bootstrap_ci(score_gaps, N_BOOTSTRAP, C_SEED)
    print(f"score gap (follow - {control}): {score_gaps.mean():+.3f} [{low:+.3f}, {high:+.3f}] over {len(both_ids)} sets {'PASS' if low > 0 else 'FAIL'}")
    low, high = get_bootstrap_ci(price_gaps, N_BOOTSTRAP, C_SEED)
    print(f"  price gap: {price_gaps.mean():+.3f} [{low:+.3f}, {high:+.3f}]")
    rho, p_value = stats.spearmanr(score_gaps, price_gaps)
    higher = price_gaps > 0
    print(f"  score vs price gap: Spearman {rho:+.2f} (p {p_value:.3f}); score gap where follow priced higher {score_gaps[higher].mean():+.3f} ({higher.sum()})")

# Price-leak check, descriptive only (histories differ): C-follow vs out-of-sample B runs at a matched price
oos_b_prices, oos_b_scores = oos_prices[oos_conditions == CONDITION_B], oos_scores[oos_conditions == CONDITION_B]
matched = [
    (twin_score[pid][C_FOLLOW], oos_b_scores[near].mean())
    for pid in both_ids
    if (near := np.abs(oos_b_prices - twin_price[pid][C_FOLLOW]) <= PRICE_MATCH).any()
]
if matched:
    follow_matched, b_matched = np.array(matched).T
    print(f"price-matched: C-follow {follow_matched.mean():.3f} vs B {b_matched.mean():.3f} ({len(matched)} C-follow runs matched)")
else:
    print(f"price-matched: no out-of-sample B run within {PRICE_MATCH} of any C-follow price")
low, high = get_bootstrap_ci(no_cue_scores, N_BOOTSTRAP, C_SEED)
print(f"no-cue A/B histories (post-hoc): mean logit {no_cue_scores.mean():+.3f} [{low:+.3f}, {high:+.3f}]")

groups = {
    "A (oos)": oos_scores[oos_conditions == CONDITION_A],
    "B (oos)": oos_b_scores,
    "no cue": no_cue_scores,
    "C-follow": [twin_score[pid][C_FOLLOW] for pid in both_ids],
    **{f"C-{control}": [twin_score[pid][control] for pid in both_ids] for control in C_CONTROLS},
}
_ = plt.figure(figsize=(6, 2.5))
_ = plt.boxplot(list(groups.values()), tick_labels=list(groups))
_ = plt.ylabel(f"probe logit, layer {c_layer}")
plt.tight_layout()
plt.savefig(DATA_DIR / f"c_scores_{MODEL_NAME}.png", dpi=200)
plt.show()

# %% C score gaps at every layer (post-hoc, DECISIONS.md), as a share of that layer's A minus B gap so layers compare
c_index = {(run.pair_id, run.condition): i for i, run in enumerate(c_runs)}
c_row_of_run = {i: row for row, i in enumerate(c_run_idx.tolist())}
twin_rows = {name: [c_row_of_run[c_index[pid, name]] for pid in both_ids] for name in (C_FOLLOW, *C_CONTROLS)}
c_layers = range(1, n_layers)  # layer 0 is the embedding of the same last token in every run, so its probe is degenerate
layer_gaps = np.zeros((len(C_CONTROLS), n_layers, 3))  # (control, layer, mean / CI low / CI high)
for layer in c_layers:
    probe = saved["probes"][layer]
    layer_score = {name: probe.decision_function(c_acts[rows, layer].astype(np.float32)) for name, rows in twin_rows.items()}
    oos_layer = probe.decision_function(acts[oos_rows, layer].astype(np.float32))
    ab_gap = oos_layer[oos_conditions == CONDITION_A].mean() - oos_layer[oos_conditions == CONDITION_B].mean()
    for k, control in enumerate(C_CONTROLS):
        gaps = (layer_score[C_FOLLOW] - layer_score[control]) / ab_gap
        layer_gaps[k, layer] = (gaps.mean(), *get_bootstrap_ci(gaps, N_BOOTSTRAP, C_SEED))

_ = plt.figure(figsize=(6, 2.5))
for k, (control, color) in enumerate(zip(C_CONTROLS, ["tab:purple", "tab:brown"])):
    passing = [layer for layer in c_layers if layer_gaps[k, layer, 1] > 0]
    print(f"follow - {control}: CI above 0 at {len(passing)}/{len(c_layers)} layers; headline layer {layer_gaps[k, c_layer, 0]:+.1%} of A-B gap")
    _ = plt.plot(c_layers, layer_gaps[k, 1:, 0], label=f"C-follow minus C-{control}", color=color)
    _ = plt.fill_between(c_layers, layer_gaps[k, 1:, 1], layer_gaps[k, 1:, 2], alpha=0.25, color=color)
_ = plt.axhline(0, color="grey", linestyle="--")
_ = plt.axvline(c_layer, color="black", linewidth=0.8)
_ = plt.xlabel("layer")
_ = plt.ylabel("score gap / A-B gap")
_ = plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(DATA_DIR / f"c_layers_{MODEL_NAME}.png", dpi=200)
plt.show()








