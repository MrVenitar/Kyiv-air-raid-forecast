"""Small transforms used inside saved model pipelines (kept in an importable module so pickles load anywhere)."""
import numpy as np


def signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))


def clip5(X):
    """Bound standardised inputs: a value never seen in training must not become a huge z-score."""
    return np.clip(X, -5, 5)


# ------------------------------------------------------------------ version-independent logistic regression
def export_logreg(pipe, columns) -> dict:
    """The fitted pipeline (median imputer -> signed log -> standard scaler -> clip ±5 -> logistic regression)
    as plain numbers, so forecasts do not depend on the scikit-learn version that trained it."""
    imp, scaler, lr = pipe.steps[0][1], pipe.steps[2][1], pipe.steps[-1][1]
    stats = np.asarray(imp.statistics_, float)
    keep = np.ones(len(stats), bool) if getattr(imp, "keep_empty_features", False) else ~np.isnan(stats)
    return {"type": "logreg", "columns": list(columns), "keep": keep.tolist(),
            "median": np.where(np.isnan(stats), 0.0, stats).tolist(),
            "mean": np.asarray(scaler.mean_, float).tolist(), "scale": np.asarray(scaler.scale_, float).tolist(),
            "coef": np.asarray(lr.coef_[0], float).tolist(), "intercept": float(lr.intercept_[0])}


def predict_logreg(m: dict, X) -> np.ndarray:
    """Probability of the positive class for rows of X (columns in m['columns'] order); numpy only."""
    X = np.asarray(X, float)
    X = np.where(np.isnan(X), np.asarray(m["median"]), X)[:, np.asarray(m["keep"], bool)]
    z = clip5((signed_log(X) - np.asarray(m["mean"])) / np.asarray(m["scale"]))
    logit = np.clip(z @ np.asarray(m["coef"]) + m["intercept"], -60, 60)
    return 1.0 / (1.0 + np.exp(-logit))
