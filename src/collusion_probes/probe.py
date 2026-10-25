"""Linear probes on cached activations: one per layer, collusion (label 1) vs innocent (label 0)."""

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

PROBE_C = 1.0  # inverse L2 strength, pre-registered in DECISIONS.md
MAX_ITER = 5000  # high-dimensional fits need more than sklearn's default 100


def make_probe() -> Pipeline:
    """Standardise, then L2 logistic regression, as pre-registered."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=PROBE_C, max_iter=MAX_ITER))


def get_mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean and t-distribution confidence interval, e.g. of accuracy across split seeds."""
    mean = float(np.mean(values))
    half_width = float(stats.sem(values) * stats.t.ppf((1 + confidence) / 2, len(values) - 1))
    return mean, mean - half_width, mean + half_width


def make_text_baseline() -> Pipeline:
    """TF-IDF on the prompt text, then logistic regression: matches exact words only, not meaning."""
    return make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=MAX_ITER))
