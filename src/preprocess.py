"""Preprocessing, epoching and trial quality control (BCI Workshop notebooks 01–04).

For every session: µV -> V and an MNE RawArray (bipolar channels, no re-referencing), zero-phase FIR band-pass
1–40 Hz, and EOG regression whose weights are fitted on the pre-trial eye-movement calibration recording.
No step uses class labels.
"""
import mne
import numpy as np
from mne.preprocessing import EOGRegression
from mne.time_frequency import psd_array_welch
from scipy.fft import next_fast_len
from scipy.signal import hilbert

from .data import (BASE_WIN, CH_EEG, CH_EOG, EPOCH_LEN, FEAT_WIN, H_FREQ, L_FREQ, MIN_CALIB_S, ROBUST_Z,
                   RUN_LEN, SESS_TYPE)

TF_BANDS = [(lo, lo + 4) for lo in range(6, 31, 2)]            # 4-Hz bands, centres 8–32 Hz, for time–frequency maps
ERD_BANDS = {"mu 8–12": (8, 12), "beta 13–30": (13, 30)}
DS = 10                                                          # envelope downsampling: 250 Hz -> 25 Hz
N_EP = int(EPOCH_LEN * 250) + 1
TF_TIMES = (np.arange(N_EP // DS) * DS + DS / 2) / 250


def to_raw(d):
    info = mne.create_info(CH_EEG + CH_EOG, d["fs"], ["eeg"] * 3 + ["eog"] * 3)
    assert not np.isnan(d["X"]).any(), "missing samples found — interpolate before filtering"
    raw = mne.io.RawArray(d["X"].T * 1e-6, info, verbose=False)
    raw.set_eeg_reference(ref_channels=[], verbose=False)
    return raw


def preprocess(d, eog_regression=True):
    """Filter and (optionally) EOG-correct one session. Returns (raw, info) with where the EOG model was fitted."""
    raw = to_raw(d)
    raw.filter(L_FREQ, H_FREQ, picks="all", fir_design="firwin", verbose=False)
    info = {"eog_fit_on": "none", "coef": np.full((3, 3), np.nan)}
    if eog_regression:
        calib_end = d["onsets"][0] / d["fs"] - 2.0
        if calib_end >= MIN_CALIB_S:
            fit_raw, info["eog_fit_on"] = raw.copy().crop(0, calib_end), f"calibration ({calib_end:.0f} s)"
        else:
            fit_raw, info["eog_fit_on"] = raw, "whole session (no calibration)"
        model = EOGRegression(picks="eeg", picks_artifact="eog", proj=False).fit(fit_raw)
        raw = model.apply(raw)
        info["coef"] = model.coef_
    return raw, info


def envelope_power(raw_eeg_data, band, fs, onsets):
    """Hilbert-envelope power in a band, epoched and downsampled to 25 Hz: (trials, channels, time)."""
    x = mne.filter.filter_data(raw_eeg_data, fs, band[0], band[1], fir_design="firwin", verbose=False)
    n = x.shape[-1]
    p = np.abs(hilbert(x, N=next_fast_len(n), axis=-1)[..., :n]) ** 2
    idx = onsets[:, None] + np.arange(N_EP)[None, :]
    ep = p[:, idx].transpose(1, 0, 2)
    return ep[:, :, : (N_EP // DS) * DS].reshape(len(onsets), 3, -1, DS).mean(-1)


def window_psd(ep, fs, win):
    """Welch PSD (1-Hz resolution, 1–40 Hz) of epochs restricted to a time window."""
    a, b = int(win[0] * fs), int(win[1] * fs) + 1
    return psd_array_welch(ep[:, :, a:b], fs, fmin=1, fmax=40, n_fft=250, n_per_seg=250, n_overlap=125, verbose=False)


def signed_r2(L, y):
    """Signed r² between a feature array (trials, ...) and the label (right = 1)."""
    Lc = L - L.mean(0)
    yc = (y - y.mean()).reshape((-1,) + (1,) * (L.ndim - 1))
    r = (Lc * yc).sum(0) / np.sqrt((Lc ** 2).sum(0) * (yc ** 2).sum())
    return np.sign(r) * r ** 2


def robust_outliers(logvar, z=ROBUST_Z):
    """Trials whose log-variance is > z robust SDs (1.4826 × MAD) from the session median on any channel."""
    med = np.median(logvar, 0)
    mad = 1.4826 * np.median(np.abs(logvar - med), 0)
    return (np.abs(logvar - med) / mad > z).any(1)


def process_session(d, session, keep_erd=True):
    """Everything the analysis needs from one session, for both preprocessing versions ('eog' and 'noeog').

    Cleaning rule (applied to TRAINING trials only): a trial is left out if the dataset authors flagged it or it is
    a log-variance outlier. Scored trials are never removed.
    Returns (rec, freqs, eog_info).
    """
    fs, on = d["fs"], d["onsets"]
    rec = {"y": d["y"], "expert": d["artifacts"], "block": np.arange(len(d["y"])) // RUN_LEN[SESS_TYPE[session]]}
    if SESS_TYPE[session] == "feedback":             # runs are separated by ~23-s pauses: check the reconstruction
        gaps = np.diff(on) / fs
        big = set(np.where(gaps > 2 * np.median(gaps))[0])
        assert big == set(np.where(np.diff(rec["block"]) == 1)[0]), "run reconstruction mismatch"
    eog_info = None
    for version, use_eog in [("eog", True), ("noeog", False)]:
        raw, info = preprocess(d, eog_regression=use_eog)
        eeg = raw.get_data(picks="eeg")
        ep = eeg[:, on[:, None] + np.arange(N_EP)[None, :]].transpose(1, 0, 2) * 1e6     # µV, trials × ch × time
        rec[f"psd_{version}"], freqs = window_psd(ep, fs, FEAT_WIN)
        rec[f"psdbase_{version}"], _ = window_psd(ep, fs, BASE_WIN)
        rec[f"logvar_{version}"] = np.log(ep.var(-1))
        if use_eog:
            eog_info = info
            if keep_erd:
                rec["tf_r2"] = np.stack([signed_r2(np.log(envelope_power(eeg, b, fs, on)), d["y"]) for b in TF_BANDS])
                rec["erd"] = {k: envelope_power(eeg, b, fs, on) for k, b in ERD_BANDS.items()}
    for version in ("eog", "noeog"):
        rec[f"outlier_{version}"] = robust_outliers(rec[f"logvar_{version}"])
        rec[f"keep_{version}"] = ~(rec["expert"] | rec[f"outlier_{version}"])
    rec["logvar"], rec["outlier"], rec["keep"] = rec["logvar_eog"], rec["outlier_eog"], rec["keep_eog"]
    return rec, freqs, eog_info


def erd_curve(p, y, cls, base=(0.5, 2.5)):
    """Classic ERD: average power over trials of one class first, then % change vs the pre-cue period."""
    m = p[y == cls].mean(0)
    b = m[:, (TF_TIMES >= base[0]) & (TF_TIMES <= base[1])].mean(-1, keepdims=True)
    return 100 * (m / b - 1)
