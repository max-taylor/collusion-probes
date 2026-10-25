"""Linear probes on cached activations: one per layer, collusion (label 1) vs innocent (label 0)."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

PROBE_C = 1.0  # inverse L2 strength, pre-registered in DECISIONS.md
MAX_ITER = 5000  # high-dimensional fits need more than sklearn's default 100


def make_probe() -> Pipeline:
    """Standardise, then L2 logistic regression, as pre-registered."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=PROBE_C, max_iter=MAX_ITER))


def pick_headline_layer(cv_acc: np.ndarray) -> int:
    """Layer with the best CV accuracy; ties go to the middle tied layer (DECISIONS.md change log)."""
    tied_layers = np.flatnonzero(np.isclose(cv_acc, cv_acc.max()))
    return int(tied_layers[len(tied_layers) // 2])


def get_bootstrap_ci(values: list[float] | np.ndarray, n_resamples: int, seed: int) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean, resampling values (one per pair) with replacement."""
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(values, size=(n_resamples, len(values))).mean(axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(low), float(high)


def get_pair_accuracy_ci(
    correct: np.ndarray, pairs: np.ndarray, n_resamples: int, seed: int
) -> tuple[float, float, float]:
    """Accuracy with a 95% CI that resamples pairs, not runs: a pair's runs (and its repeats across split seeds) are one unit."""
    _, pair_of_run = np.unique(pairs, return_inverse=True)
    pair_acc = np.bincount(pair_of_run, weights=correct) / np.bincount(pair_of_run)
    low, high = get_bootstrap_ci(pair_acc, n_resamples, seed)
    return float(pair_acc.mean()), low, high


def make_text_baseline() -> Pipeline:
    """TF-IDF on the prompt text, then logistic regression: matches exact words only, not meaning."""
    return make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=MAX_ITER))
