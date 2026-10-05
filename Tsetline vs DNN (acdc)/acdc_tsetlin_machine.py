"""
Tsetlin Machine classifier for ACDC cardiac pathology (5 classes: DCM, HCM, MINF, NOR, RV)
using the ORIGINAL ACDC release (NIfTI + Info.cfg).

Expected folder layout (as in ACDC-dataset-cine-CMR-):
    <root>/training/patient001/Info.cfg
    <root>/training/patient001/patient001_frame01_gt.nii.gz   (ED mask)
    <root>/training/patient001/patient001_frame12_gt.nii.gz   (ES mask)
    ...
    <root>/testing/patient101/...                              (patients 101-150)

Pipeline
--------
1. Info.cfg gives the label (Group), ED/ES frame numbers, Height and Weight.
   NOTE: in the testing folder the classes are NOT ordered by patient ID, so the
   label is always read from Info.cfg, never inferred from the ID.
2. From the ground-truth masks (0=bg, 1=RV, 2=MYO, 3=LV) and the real voxel spacing
   we compute clinical features in mL / g / mm:
     volumes, ejection fractions, myocardial mass, ratios, BSA-indexed volumes,
     and myocardial wall thickness (max / mean / std across slices at ED and ES).
3. Booleanization: per-feature quantile thermometer encoding, thresholds fitted on
   training data only.
4. 5-fold stratified CV on patients 001-100 selects n_bins and the best epoch.
5. Retrain on all 100 training patients and evaluate ONCE on unseen patients 101-150.

Install:  pip install nibabel scikit-learn scikit-image matplotlib
          pip install git+https://github.com/cair/tmu.git
Run:      python acdc_tsetlin_machine.py --data_root "path/to/ACDC-dataset-cine-CMR-"
"""

import os
import json
import argparse

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tmu.models.classification.vanilla_classifier import TMClassifier

CLASSES = ["DCM", "HCM", "MINF", "NOR", "RV"]
MYO_DENSITY = 1.05  # g/mL


# ----------------------------------------------------------------------------
# Data loading & feature extraction
# ----------------------------------------------------------------------------
def read_info(path):
    info = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
    return info


def wall_thickness(myo_mask, spacing_xy):
    """Per-slice myocardial thickness (mm): 2 x distance-to-boundary on the skeleton."""
    per_slice_mean, per_slice_max = [], []
    for z in range(myo_mask.shape[2]):
        sl = myo_mask[:, :, z]
        if sl.sum() < 10:
            continue
        dt = distance_transform_edt(sl, sampling=spacing_xy)
        skel = skeletonize(sl)
        if skel.sum() == 0:
            continue
        t = 2.0 * dt[skel]
        per_slice_mean.append(t.mean())
        per_slice_max.append(t.max())
    if not per_slice_mean:
        return 0.0, 0.0, 0.0
    return float(np.max(per_slice_max)), float(np.mean(per_slice_mean)), float(np.std(per_slice_mean))


def frame_measures(gt_path):
    img = nib.load(gt_path)
    mask = np.asarray(img.dataobj).astype(np.uint8)
    sp = img.header.get_zooms()[:3]
    voxel_ml = float(np.prod(sp)) / 1000.0
    myo = mask == 2
    tmax, tmean, tstd = wall_thickness(myo, sp[:2])
    return {
        "RV": (mask == 1).sum() * voxel_ml,
        "MYO": myo.sum() * voxel_ml,
        "LV": (mask == 3).sum() * voxel_ml,
        "T_max": tmax, "T_mean": tmean, "T_std": tstd,
    }


def patient_features(pdir):
    pid = os.path.basename(pdir)
    info = read_info(os.path.join(pdir, "Info.cfg"))
    ed = frame_measures(os.path.join(pdir, f"{pid}_frame{int(info['ED']):02d}_gt.nii.gz"))
    es = frame_measures(os.path.join(pdir, f"{pid}_frame{int(info['ES']):02d}_gt.nii.gz"))
    h, w = float(info["Height"]), float(info["Weight"])
    bsa = np.sqrt(h * w / 3600.0)  # Mosteller, m^2
    eps = 1e-6

    feats = {
        "LV_EDV": ed["LV"], "LV_ESV": es["LV"],
        "LV_EF": (ed["LV"] - es["LV"]) / (ed["LV"] + eps),
        "RV_EDV": ed["RV"], "RV_ESV": es["RV"],
        "RV_EF": (ed["RV"] - es["RV"]) / (ed["RV"] + eps),
        "MYO_mass_ED": ed["MYO"] * MYO_DENSITY, "MYO_vol_ES": es["MYO"],
        "LV_EDV_idx": ed["LV"] / bsa, "LV_ESV_idx": es["LV"] / bsa,
        "RV_EDV_idx": ed["RV"] / bsa, "RV_ESV_idx": es["RV"] / bsa,
        "MYO_mass_idx": ed["MYO"] * MYO_DENSITY / bsa,
        "RV_LV_EDV_ratio": ed["RV"] / (ed["LV"] + eps),
        "RV_LV_ESV_ratio": es["RV"] / (es["LV"] + eps),
        "MYO_LV_ED_ratio": ed["MYO"] / (ed["LV"] + eps),
        "MYO_LV_ES_ratio": es["MYO"] / (es["LV"] + eps),
        "Thick_max_ED": ed["T_max"], "Thick_mean_ED": ed["T_mean"], "Thick_std_ED": ed["T_std"],
        "Thick_max_ES": es["T_max"], "Thick_mean_ES": es["T_mean"], "Thick_std_ES": es["T_std"],
        "Height": h, "Weight": w, "BSA": bsa,
    }
    return feats, CLASSES.index(info["Group"])


def load_split(split_dir):
    pdirs = sorted(d for d in (os.path.join(split_dir, x) for x in os.listdir(split_dir))
                   if os.path.isdir(d) and os.path.basename(d).startswith("patient"))
    rows, ys, ids = [], [], []
    for d in pdirs:
        f, y = patient_features(d)
        rows.append(f); ys.append(y); ids.append(os.path.basename(d))
    names = list(rows[0].keys())
    X = np.array([[r[n] for n in names] for r in rows], dtype=np.float64)
    return X, np.array(ys, dtype=np.uint32), ids, names


# ----------------------------------------------------------------------------
# Booleanization: per-feature quantile thermometer encoding
# ----------------------------------------------------------------------------
class QuantileThermometer:
    def __init__(self, n_bins):
        self.n_bins = n_bins
        self.thresholds = None

    def fit(self, X):
        qs = np.linspace(0, 1, self.n_bins + 2)[1:-1]
        self.thresholds = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        cols = [(X[:, [j]] > thr[None, :]) for j, thr in enumerate(self.thresholds)]
        return np.hstack(cols).astype(np.uint32)


def make_tm(args, seed):
    return TMClassifier(number_of_clauses=args.clauses, T=args.T, s=args.s,
                        platform="CPU", weighted_clauses=True, seed=seed)


def train_curve(Xtr, ytr, Xev, yev, args, seed):
    tm = make_tm(args, seed)
    accs = []
    for _ in range(args.epochs):
        tm.fit(Xtr, ytr)
        accs.append(accuracy_score(yev, tm.predict(Xev)))
    return np.array(accs)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="Folder that contains 'training' and 'testing'")
    ap.add_argument("--bins_grid", type=int, nargs="+", default=[3, 5, 8, 10, 15])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--clauses", type=int, default=200)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="tm_results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Extracting features...")
    X_train, y_train, train_ids, names = load_split(os.path.join(args.data_root, "training"))
    X_test, y_test, test_ids, _ = load_split(os.path.join(args.data_root, "testing"))
    print(f"Train: {len(train_ids)} patients | Test: {len(test_ids)} patients | {len(names)} features")
    print("Train class counts:", dict(zip(CLASSES, np.bincount(y_train, minlength=5))))
    print("Test  class counts:", dict(zip(CLASSES, np.bincount(y_test, minlength=5))))

    # save the feature tables (useful for the report / other models)
    for X, y, ids, fn in [(X_train, y_train, train_ids, "features_train.csv"),
                          (X_test, y_test, test_ids, "features_test.csv")]:
        with open(os.path.join(args.out, fn), "w") as f:
            f.write("patient,label," + ",".join(names) + "\n")
            for pid, lab, row in zip(ids, y, X):
                f.write(f"{pid},{CLASSES[lab]}," + ",".join(f"{v:.4f}" for v in row) + "\n")

    # ---- 5-fold CV: choose n_bins and best epoch ----------------------------
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    cv_curves = {}
    for n_bins in args.bins_grid:
        fold_curves = []
        for k, (tr, va) in enumerate(skf.split(X_train, y_train)):
            enc = QuantileThermometer(n_bins).fit(X_train[tr])
            fold_curves.append(train_curve(enc.transform(X_train[tr]), y_train[tr],
                                           enc.transform(X_train[va]), y_train[va],
                                           args, seed=args.seed + k))
        m, s = np.mean(fold_curves, axis=0), np.std(fold_curves, axis=0)
        cv_curves[n_bins] = (m, s)
        b = int(np.argmax(m))
        print(f"n_bins={n_bins:>3}: best mean CV acc {m[b]:.3f} ± {s[b]:.3f} at epoch {b + 1}")

    best_bins = max(cv_curves, key=lambda b: cv_curves[b][0].max())
    best_epoch = int(np.argmax(cv_curves[best_bins][0])) + 1
    best_cv = float(cv_curves[best_bins][0].max())
    print(f"\nSelected n_bins={best_bins}, epochs={best_epoch} (CV acc {best_cv:.3f})")

    # ---- Retrain on all training patients, test once on 101-150 -------------
    enc = QuantileThermometer(best_bins).fit(X_train)
    Xtr_b, Xte_b = enc.transform(X_train), enc.transform(X_test)
    tm = make_tm(args, seed=args.seed)
    for _ in range(best_epoch):
        tm.fit(Xtr_b, y_train)
    y_pred = tm.predict(Xte_b)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"\nTEST accuracy (patients 101-150): {test_acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=CLASSES, digits=3))
    cm = confusion_matrix(y_test, y_pred, labels=range(5))
    print("Confusion matrix (rows=true, cols=pred):\n", cm)
    wrong = [(pid, CLASSES[t], CLASSES[p]) for pid, t, p in zip(test_ids, y_test, y_pred) if t != p]
    print("Misclassified:", wrong)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({
            "features": names, "best_n_bins": best_bins, "best_epoch": best_epoch,
            "cv_accuracy": best_cv, "test_accuracy": test_acc,
            "confusion_matrix": cm.tolist(), "misclassified": wrong,
            "hyperparams": {"clauses": args.clauses, "T": args.T, "s": args.s},
            "thresholds": {n: t.tolist() for n, t in zip(names, enc.thresholds)},
        }, f, indent=2)

    # ---- Plots --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ep = np.arange(1, args.epochs + 1)
    for b, (m, s) in cv_curves.items():
        ax.plot(ep, m, label=f"n_bins={b}", lw=2.2 if b == best_bins else 1)
        if b == best_bins:
            ax.fill_between(ep, m - s, m + s, alpha=0.15)
    ax.axvline(best_epoch, ls="--", c="gray")
    ax.set(xlabel="Epoch", ylabel="Mean validation accuracy (5-fold)",
           title="Tsetlin Machine – CV selection of n_bins and epoch")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "cv_curves.png"), dpi=200)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5), CLASSES); ax.set_yticks(range(5), CLASSES)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xlabel="Predicted", ylabel="True", title=f"Test set (101-150), acc = {test_acc:.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "test_confusion.png"), dpi=200)
    print(f"\nSaved results, feature CSVs and figures to ./{args.out}/")


if __name__ == "__main__":
    main()
