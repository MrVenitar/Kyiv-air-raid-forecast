"""Small transforms used inside saved model pipelines (kept in an importable module so pickles load anywhere)."""
import numpy as np


def signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))


def clip5(X):
    """Bound standardised inputs: a value never seen in training must not become a huge z-score."""
    return np.clip(X, -5, 5)
