"""Leak-free within- vs cross-session evaluation and the hypothesis tests.

For a training session a and a scored session b:
  * within: leave-one-run-out CV inside b; training folds use only cleaned trials; every held-out trial is scored
  * cross:  fit on the cleaned trials of a; score every trial of b
  * both sides are scored run by run on exactly the same trials (AUC per run, then averaged)
  * both decoders get the same number of training trials, drawn class-stratified N_REP times
drift cost = within-session AUC − cross-session AUC.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import balanced_accuracy_score

from .data import ALPHA, N_REP, SEED, SUBJECTS, TRANSFERS
from .features_models import features, make_clf, strat_sub


def fast_auc(y, s):
    """ROC AUC via the Mann–Whitney rank formula (identical to sklearn's roc_auc_score)."""
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bacc0(y, s):
    """Balanced accuracy at the default decision threshold (decision function > 0)."""
    return balanced_accuracy_score(y, (s > 0).astype(int))


def per_run(fn, y, s, blk, runs):
    vals = [fn(y[blk == g], s[blk == g]) for g in runs if len(np.unique(y[blk == g])) == 2]
    return float(np.mean(vals))


def compare(S, freqs, subj, s_tr, s_te, variant="A", version="eog", clean=True, matched=True, first_block=False,
            n_rep=N_REP, train_override=None, shuffle_within=False, pooled=False, seed_offset=0):
    """Within-session vs cross-session decoding of session s_te for one subject.

    S: {(subject, session): rec} from preprocess.process_session.
    train_override: (X, y, keep) to train the cross decoder on something other than session s_tr.
    pooled: v1 method (one AUC over pooled out-of-fold scores) — kept as sensitivity analysis (f).
    """
    zmask = (lambda r: r["block"] == 0) if first_block else None      # first-run statistics, scored day only
    rt, re_ = (S[(subj, s_tr)] if s_tr else None), S[(subj, s_te)]
    if train_override is None:
        Xtr, ytr = features(rt, freqs, variant, version), rt["y"]
        keep_tr = rt[f"keep_{version}"] if clean else np.ones(len(ytr), bool)
    else:
        Xtr, ytr, keep_tr = train_override
    Xte, yte, blk = features(re_, freqs, variant, version, zmask), re_["y"], re_["block"]
    keep_te = re_[f"keep_{version}"] if clean else np.ones(len(yte), bool)
    scored = (blk >= 1) if first_block else np.ones(len(yte), bool)
    folds = [g for g in np.unique(blk) if g >= 1 or not first_block]
    tr_w = {g: np.where((blk != g) & keep_te)[0] for g in folds}
    for g in folds:                                                     # V1: a run is never on both sides
        assert not np.isin(np.where(blk == g)[0], tr_w[g]).any()
    n = int(min(min(len(v) for v in tr_w.values()), keep_tr.sum())) if matched else None
    rng = np.random.default_rng([SEED, subj, s_tr, s_te, seed_offset])   # own stream per comparison
    reps = n_rep if matched else 1
    idx_tr = np.where(keep_tr)[0]
    out = {k: [] for k in ("within", "cross", "within_bacc", "cross_bacc")}
    cross_scores = np.zeros(len(yte))
    for _ in range(reps):
        oof = np.full(len(yte), np.nan)
        for g in folds:
            idx = tr_w[g] if n is None else strat_sub(tr_w[g], yte, n, rng)
            y_fit = rng.permutation(yte[idx]) if shuffle_within else yte[idx]
            oof[blk == g] = make_clf().fit(Xte[idx], y_fit).decision_function(Xte[blk == g])
        idx = idx_tr if n is None else strat_sub(idx_tr, ytr, n, rng)
        sc = make_clf().fit(Xtr[idx], ytr[idx]).decision_function(Xte)
        cross_scores += sc / reps
        if pooled:
            m = scored
            out["within"].append(fast_auc(yte[m], oof[m])); out["cross"].append(fast_auc(yte[m], sc[m]))
            out["within_bacc"].append(bacc0(yte[m], oof[m])); out["cross_bacc"].append(bacc0(yte[m], sc[m]))
        else:
            out["within"].append(per_run(fast_auc, yte, oof, blk, folds))
            out["cross"].append(per_run(fast_auc, yte, sc, blk, folds))
            out["within_bacc"].append(per_run(bacc0, yte, oof, blk, folds))
            out["cross_bacc"].append(per_run(bacc0, yte, sc, blk, folds))
    res = {k: float(np.mean(v)) for k, v in out.items()}
    res.update(drift=res["within"] - res["cross"], drift_bacc=res["within_bacc"] - res["cross_bacc"], n_train=n,
               scores=cross_scores[scored], y=yte[scored])
    return res


def transfer(S, freqs, subj, name, **kw):
    """One transfer (e.g. 'S3→S4/S5'), averaged over its scored sessions."""
    s_tr, scored = TRANSFERS[name]
    rs = [compare(S, freqs, subj, s_tr, s, **kw) for s in scored]
    keys = ("within", "cross", "drift", "within_bacc", "cross_bacc", "drift_bacc")
    out = {k: float(np.mean([r[k] for r in rs])) for k in keys}
    out.update(n_train=" / ".join(str(r["n_train"]) for r in rs), scores=np.concatenate([r["scores"] for r in rs]),
               y=np.concatenate([r["y"] for r in rs]), per_session={s: r for s, r in zip(scored, rs)})
    return out


def table(R, variant, tname, key, subjects=SUBJECTS):
    return pd.Series({subj: R[(variant, tname, subj)][key] for subj in subjects})


def test_h1(R):
    """H1a/H1b: one-sided Wilcoxon (drift cost > 0) over subjects, for AUC and balanced accuracy."""
    rows = []
    for tname in TRANSFERS:
        d, db = table(R, "A", tname, "drift"), table(R, "A", tname, "drift_bacc")
        rows.append(dict(test="H1a (primary)" if tname == "S1→S2" else "H1b (secondary)", transfer=tname,
                         within_auc=table(R, "A", tname, "within").median(),
                         cross_auc=table(R, "A", tname, "cross").median(),
                         drift_median=d.median(), n_positive=int((d > 0).sum()),
                         p=wilcoxon(d, alternative="greater").pvalue,
                         drift_bacc_median=db.median(), p_bacc=wilcoxon(db, alternative="greater").pvalue))
    tab = pd.DataFrame(rows).set_index("test")
    tab["supported"] = tab.p < ALPHA
    return tab


def holm(pvalues):
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.minimum(1, np.maximum.accumulate(p[order] * (m - np.arange(m))))
    out = np.empty(m)
    out[order] = adj
    return out


def avg_over_transfers(R, variant, key, subjects=SUBJECTS):
    return pd.Series({subj: np.mean([R[(variant, t, subj)][key] for t in TRANSFERS]) for subj in subjects})


def test_h2(R, variants=("C", "D")):
    """H2: does a fix lower the drift cost (one-sided Wilcoxon, Holm) without lowering cross-session AUC?"""
    dA, cA, wA = (avg_over_transfers(R, "A", k) for k in ("drift", "cross", "within"))
    rows = []
    for v in variants:
        dV, cV, wV = (avg_over_transfers(R, v, k) for k in ("drift", "cross", "within"))
        rows.append(dict(fix=v, median_within_change=(wV - wA).median(), median_drift_A=dA.median(),
                         median_drift_fix=dV.median(), median_reduction=(dA - dV).median(),
                         n_reduced=int((dA - dV > 0).sum()), p_reduction=wilcoxon(dA - dV, alternative="greater").pvalue,
                         median_cross_change=(cV - cA).median(), p_cross_change=wilcoxon(cV - cA).pvalue))
    tab = pd.DataFrame(rows).set_index("fix")
    tab["p_reduction_holm"] = holm(tab.p_reduction.values)
    tab["supported"] = (tab.p_reduction_holm < ALPHA) & ~((tab.p_cross_change < ALPHA) & (tab.median_cross_change < 0))
    return tab
