"""
Tsetlin Machine classification of the 5 ACDC classes (NOR, MINF, DCM, HCM, RV)
from the merged + de-duplicated feature set produced by reduce_features.py.

Protocol
  * Training / model selection: patients 1-100 only.
  * 5-fold stratified CV on 1-100 selects: number of thermometer bins, number of
    clauses, s, and the best epoch (highest mean validation accuracy over folds).
  * Thermometer thresholds (quantiles) are fitted on the training part of each fold
    only -> no information from validation/test patients leaks into the encoding.
  * Final model: retrained on all of 1-100 with the selected settings, evaluated once
    on the unseen test patients 101-150. The TM is stochastic, so the final training
    is repeated N_RUNS times and mean +/- std is reported.

The same pipeline is run on three feature sets for comparison:
  reduced   – all four sources after redundancy removal   (main model)
  all       – all four sources concatenated, no removal
  clinical  – reduced set without the Cetin radiomics (W + K + I only)
"""
import json
import sys
import time
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

CLASSES = ["NOR", "MINF", "DCM", "HCM", "RV"]
N_FOLDS = 5
MAX_EPOCHS = 40
N_RUNS = 10
GRID = {"n_bins": [5, 10], "clauses": [200, 500], "s": [3.0, 8.0]}
SEED = 42


# ------------------------------------------------------------------ encoding
class Thermometer:
    """Each feature -> (n_bins-1) bits: bit_k = [x >= q_k], q_k = training quantiles."""

    def __init__(self, n_bins):
        self.n_bins = n_bins

    def fit(self, X):
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.thresholds_ = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        bits = [(X[:, [j]] >= t[None, :]) for j, t in enumerate(self.thresholds_)]
        return np.hstack(bits).astype(np.uint32)

    def bit_names(self, feature_names):
        return [(f, t) for f, ts in zip(feature_names, self.thresholds_) for t in ts]


FEATURE_SETS = ["reduced", "clinical", "all"]


def load(feature_set):
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}; choose from {FEATURE_SETS}")
    df = pd.read_csv("merged_features_reduced.csv" if feature_set != "all" else "merged_features_all.csv",
                     index_col="pid")
    feats = [c for c in df.columns if c != "group"]
    if feature_set == "clinical":
        feats = [c for c in feats if not c.startswith("C__")]
    X = df[feats].to_numpy(float)
    y = df["group"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy(np.uint32)
    tr = df.index <= 100
    return X[tr], y[tr], X[~tr], y[~tr], feats


# ------------------------------------------------------------------ CV for one config
def cv_curve(args):
    X, y, n_bins, clauses, s = args
    T = clauses // 4
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    curves = np.zeros((N_FOLDS, MAX_EPOCHS))
    for k, (tr, va) in enumerate(skf.split(X, y)):
        enc = Thermometer(n_bins).fit(X[tr])
        Xtr, Xva = enc.transform(X[tr]), enc.transform(X[va])
        tm = MultiClassTsetlinMachine(clauses, T, s)
        for e in range(MAX_EPOCHS):
            tm.fit(Xtr, y[tr], epochs=1, incremental=True)
            curves[k, e] = accuracy_score(y[va], tm.predict(Xva))
    return {"n_bins": n_bins, "clauses": clauses, "T": T, "s": s, "curve": curves.mean(0).tolist(),
            "curve_std": curves.std(0).tolist()}


def explain(tm, enc, feats, Xtr_bits, ytr, top_clauses=3, top_feats=8):
    """Readable rules: most class-discriminative positive clauses + most-included features."""
    names = enc.bit_names(feats)
    n_bits, n_clauses = len(names), tm.number_of_clauses
    out = {}
    clause_out = tm.transform(Xtr_bits, inverted=False).reshape(len(ytr), len(CLASSES), n_clauses)
    for c, cname in enumerate(CLASSES):
        feat_count = {}
        scored = []
        for j in range(0, n_clauses, 2):  # even index = positive polarity
            lits = []
            for b in range(n_bits):
                f, t = names[b]
                if tm.ta_action(c, j, b):
                    lits.append(f"{f} >= {t:.4g}")
                    feat_count[f] = feat_count.get(f, 0) + 1
                if tm.ta_action(c, j, b + n_bits):
                    lits.append(f"{f} < {t:.4g}")
                    feat_count[f] = feat_count.get(f, 0) + 1
            if not lits:
                continue
            fires = clause_out[:, c, j]
            score = fires[ytr == c].mean() - fires[ytr != c].mean()
            scored.append((score, fires[ytr == c].mean(), fires[ytr != c].mean(), lits))
        scored.sort(key=lambda z: -z[0])
        out[cname] = {
            "top_features": sorted(feat_count.items(), key=lambda z: -z[1])[:top_feats],
            "top_clauses": [{"fires_on_class": round(a, 2), "fires_on_others": round(b, 2),
                             "rule": " AND ".join(l)} for _, a, b, l in scored[:top_clauses]],
        }
    return out


def run(feature_set):
    Xtr, ytr, Xte, yte, feats = load(feature_set)
    print(f"\n=== {feature_set}: {len(feats)} features, train {len(ytr)}, test {len(yte)} ===", flush=True)
    t0 = time.time()
    jobs = [(Xtr, ytr, nb, cl, s) for nb, cl, s in product(*GRID.values())]
    with Pool(2) as p:
        results = p.map(cv_curve, jobs)
    best = max(((r, e) for r in results for e in range(MAX_EPOCHS)),
               key=lambda z: (round(z[0]["curve"][z[1]], 6), -z[1]))
    cfg, ep = best[0], best[1] + 1
    cv_acc, cv_std = cfg["curve"][best[1]], cfg["curve_std"][best[1]]
    print(f"CV done in {time.time()-t0:.0f}s. Best: bins={cfg['n_bins']} clauses={cfg['clauses']} "
          f"T={cfg['T']} s={cfg['s']} epochs={ep}  CV acc={cv_acc:.3f}±{cv_std:.3f}", flush=True)

    # final: retrain on all training patients, test on 101-150
    enc = Thermometer(cfg["n_bins"]).fit(Xtr)
    Btr, Bte = enc.transform(Xtr), enc.transform(Xte)
    accs, preds, models = [], [], []
    for r in range(N_RUNS):
        tm = MultiClassTsetlinMachine(cfg["clauses"], cfg["T"], cfg["s"])
        tm.fit(Btr, ytr, epochs=ep)
        p = tm.predict(Bte)
        accs.append(accuracy_score(yte, p)); preds.append(p); models.append(tm)
    accs = np.array(accs)
    med = int(np.argsort(accs)[len(accs) // 2])  # median run for the confusion matrix / rules
    print(f"Test acc over {N_RUNS} runs: {accs.mean():.3f} ± {accs.std():.3f} "
          f"(min {accs.min():.2f}, max {accs.max():.2f})")
    print(classification_report(yte, preds[med], target_names=CLASSES, digits=3, zero_division=0))
    cm = confusion_matrix(yte, preds[med], labels=range(5))
    print(pd.DataFrame(cm, index=[f"true {c}" for c in CLASSES], columns=CLASSES))

    res = {
        "feature_set": feature_set, "n_features": len(feats), "n_bits": int(Btr.shape[1]),
        "best_config": {k: cfg[k] for k in ("n_bins", "clauses", "T", "s")}, "best_epoch": ep,
        "cv_accuracy": cv_acc, "cv_accuracy_std": cv_std,
        "test_accuracy_runs": accs.tolist(), "test_accuracy_mean": accs.mean(),
        "test_accuracy_std": accs.std(), "median_run_confusion_matrix": cm.tolist(),
        "median_run_report": classification_report(yte, preds[med], target_names=CLASSES,
                                                   output_dict=True, zero_division=0),
        "cv_grid": results,
    }
    if feature_set == "reduced":
        res["rules"] = explain(models[med], enc, feats, Btr, ytr)
    return res


if __name__ == "__main__":
    # Run all three feature sets by default. Only recognised names are taken from the
    # command line, so the extra arguments Jupyter/Colab/Spyder pass (e.g. "-f kernel.json")
    # are ignored. Examples:  python tsetlin_acdc.py            -> all three
    #                         python tsetlin_acdc.py clinical   -> only the 57-feature set
    sets = [a for a in sys.argv[1:] if a in FEATURE_SETS] or FEATURE_SETS
    allres = {s: run(s) for s in sets}
    with open("tsetlin_results.json", "w") as f:
        json.dump(allres, f, indent=1, default=float)
    print("\nSummary")
    for s, r in allres.items():
        print(f"  {s:9s} {r['n_features']:4d} feats | CV {r['cv_accuracy']:.3f} | "
              f"test {r['test_accuracy_mean']:.3f} ± {r['test_accuracy_std']:.3f}")
