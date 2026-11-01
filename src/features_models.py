"""Features and classifier (BCI Workshop notebooks 05–06).

Log band power: mean Welch PSD over 1-Hz bins 8–12 Hz (mu) and 13–30 Hz (beta) at C3, Cz, C4 -> 6 features.

Variants
  A  raw log power      — the standard pipeline
  C  baseline-relative  — log power in 3.5–5.5 s minus the trial's own log power in 0.5–1.5 s
  D  session z-scored   — each feature standardised with the mean/SD of all trials of its own session (no labels)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import BANDS


def logbp(psd, freqs, bands=BANDS):
    """(trials, channels, freqs) PSD -> (trials, channels × bands) natural-log band power."""
    return np.concatenate([np.log(psd[:, :, (freqs >= lo) & (freqs < hi)].mean(-1)) for lo, hi in bands], axis=1)


def features(rec, freqs, variant="A", version="eog", zmask=None):
    """Feature matrix of one session record (from preprocess.process_session).

    zmask: optional function rec -> boolean mask of the trials used for the z-scoring statistics of variant D
    (e.g. only the first run, for the realistic online version).
    """
    X = logbp(rec[f"psd_{version}"], freqs)
    if variant == "C":
        X = X - logbp(rec[f"psdbase_{version}"], freqs)
    elif variant == "D":
        m = np.ones(len(X), bool) if zmask is None else zmask(rec)
        X = (X - X[m].mean(0)) / X[m].std(0)
    return X


def make_clf():
    """Scaler + L2 logistic regression; refitted inside every fold."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def strat_sub(idx, y, n, rng):
    """n indices from idx keeping the class proportions of y[idx] (used to match training-set sizes)."""
    yi = y[idx]
    classes, counts = np.unique(yi, return_counts=True)
    quota = np.floor(counts / len(yi) * n).astype(int)
    quota[np.argsort(-(counts / len(yi) * n - quota))[: n - quota.sum()]] += 1
    return np.concatenate([rng.choice(idx[yi == c], q, replace=False) for c, q in zip(classes, quota)])
