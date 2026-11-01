"""Data loading, analysis settings and data audit for BNCI2014-004 (BCI Competition IV 2b).

The competition .mat files are read directly (as in the course lecture *Motor-Imagery BCI decoding with CSP and
LDA*) instead of MOABB's ready-made epochs, so every field the authors provide is kept: signal, trial onsets,
labels, expert artifact flags, age and gender. MOABB is used only to download the files.
"""
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat

# ---------------------------------------------------------------- analysis settings (fixed in the analysis plan)
SEED = 42
SUBJECTS = list(range(1, 10))
SESSIONS = [1, 2, 3, 4, 5]
SESS_TYPE = {1: "screening", 2: "screening", 3: "feedback", 4: "feedback", 5: "feedback"}
RUN_LEN = {"screening": 20, "feedback": 40}       # trials per run (block) in recording order
CH_EEG, CH_EOG = ["C3", "Cz", "C4"], ["EOG1", "EOG2", "EOG3"]

L_FREQ, H_FREQ = 1.0, 40.0          # band-pass of the continuous signal
EPOCH_LEN = 7.5                     # s, epochs 0–7.5 s from trial start
FEAT_WIN = (3.5, 5.5)               # s, feature window (imagery)
BASE_WIN = (0.5, 1.5)               # s, pre-cue baseline for fix C
MIN_CALIB_S = 60                    # s, minimum pre-trial EOG calibration segment
ROBUST_Z = 5.0                      # robust-SD limit for trial log-variance outliers

BANDS = [(8, 13), (13, 31)]         # 1-Hz bins 8–12 Hz and 13–30 Hz inclusive (masks are half-open [lo, hi))
FEATURE_NAMES = [f"{b}_{ch}" for b in ("mu", "beta") for ch in CH_EEG]
TRANSFERS = {"S1→S2": (1, [2]), "S3→S4/S5": (3, [4, 5])}      # (training session, scored sessions)
VARIANTS = {"A": "raw log power", "C": "baseline-relative", "D": "session z-scored"}
N_REP = 20                          # repeated class-stratified subsamples when matching training size
N_SHUFFLE = 20
ALPHA = 0.05


def mat_files(subject):
    """Return [B0xT.mat, B0xE.mat]: use local copies if present, otherwise download through MOABB."""
    root = Path(mne.get_config("MNE_DATA") or Path.home() / "mne_data")
    local = [root / "MNE-bnci-data" / "~bci" / "database" / "004-2014" / f"B{subject:02d}{s}.mat" for s in "TE"]
    if all(p.exists() for p in local):
        return local
    from moabb.datasets import BNCI2014_004
    return [Path(p) for p in BNCI2014_004().data_path(subject)]


def load_subject(subject):
    """Load all five sessions of one subject. Returns {session: dict(X, onsets, y, artifacts, fs, ...)}."""
    out = {}
    s = 1
    for f in mat_files(subject):
        runs = loadmat(f, struct_as_record=False, squeeze_me=True)["data"]
        runs = runs if isinstance(runs, np.ndarray) else [runs]
        for r in runs:
            assert f.name.endswith("T.mat") == (s <= 3), "session order: B0xT must hold sessions 1–3, B0xE sessions 4–5"
            trial = np.atleast_1d(r.trial).astype(int) - 1          # MATLAB 1-based -> 0-based samples
            if trial.size == 0:                                      # pre-trial EOG calibration runs have no trials
                continue
            out[s] = dict(X=np.asarray(r.X, float), onsets=trial, y=np.atleast_1d(r.y).astype(int) - 1,  # 0 left, 1 right
                          artifacts=np.atleast_1d(r.artifacts).astype(bool), fs=float(r.fs),
                          gender=str(r.gender), age=int(r.age), file=f.name)
            s += 1
    assert sorted(out) == SESSIONS, f"subject {subject}: found sessions {sorted(out)}"
    return out


def load_all(subjects=SUBJECTS):
    return {subj: load_subject(subj) for subj in subjects}


def audit_table(data):
    """One row per subject and session: trials, class balance, expert flags, length of the pre-trial calibration."""
    rows = []
    for subj, sessions in data.items():
        for s, d in sessions.items():
            rows.append(dict(subject=subj, session=s, type=SESS_TYPE[s], file=d["file"], fs=d["fs"],
                             minutes=len(d["X"]) / d["fs"] / 60, trials=len(d["y"]), left=int((d["y"] == 0).sum()),
                             right=int((d["y"] == 1).sum()), expert_flagged=int(d["artifacts"].sum()),
                             first_trial_s=d["onsets"][0] / d["fs"], age=d["age"], gender=d["gender"]))
    audit = pd.DataFrame(rows)
    assert (audit.fs == 250).all() and (audit.left == audit.right).all()
    return audit
