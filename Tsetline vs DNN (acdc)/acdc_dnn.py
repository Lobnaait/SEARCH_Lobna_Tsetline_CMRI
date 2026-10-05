"""
DNN (MLP) classifier for ACDC cardiac pathology on the SAME features used by the
Tsetlin Machine (features_train.csv / features_test.csv written by acdc_tsetlin_machine.py).

Architecture (as requested):
    Linear(in, h) -> BatchNorm1d(h) -> ReLU -> Dropout(p)
    Linear(h, h//2) -> BatchNorm1d(h//2) -> ReLU -> Dropout(p)
    Linear(h//2, 5)

Protocol (mirrors the Tsetlin Machine so the comparison is fair):
1. Features standardised with mean/std of the training data only (inside each CV fold
   the scaler is fitted on the training part of the fold).
2. 5-fold stratified CV on patients 001-100 selects the capacity (hidden_dim) and the
   best epoch = epoch with the MINIMUM mean validation loss.
3. Retrain on all 100 training patients for that number of epochs and evaluate ONCE on
   the unseen test patients 101-150.
4. Calibration: softmax probabilities, plus temperature scaling fitted on the
   out-of-fold logits of the training set (test set never used to fit anything).

Outputs (in --out):
  dnn_capacity.png        CV loss / accuracy vs hidden_dim (under/overfitting check)
  dnn_loss_curves.png     train vs validation loss and accuracy per epoch (selected model)
  dnn_test_confusion.png  confusion matrix on the test set
  dnn_calibration.png     top-label reliability diagram (raw vs temperature-scaled)
  dnn_results.json, dnn_test_probabilities.csv

Run:
  python acdc_dnn.py --features_dir tm_results
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

CLASSES = ["DCM", "HCM", "MINF", "NOR", "RV"]


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_features, hidden_dim, n_classes=5, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def standardise(Xtr, *others):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return [(Xtr - mu) / sd] + [(X - mu) / sd for X in others]


def evaluate(model, X, y, loss_fn):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        return loss_fn(logits, y).item(), (logits.argmax(1) == y).float().mean().item(), logits.numpy()


def train(Xtr, ytr, hidden_dim, args, seed, epochs, Xva=None, yva=None):
    """Train for `epochs`; if a validation set is given, record curves per epoch."""
    set_seed(seed)
    Xtr_t, ytr_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)
    model = MLP(Xtr.shape[1], hidden_dim, len(CLASSES), args.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    if Xva is not None:
        Xva_t, yva_t = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.long)

    n = len(ytr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            if len(idx) < 2:          # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            loss_fn(model(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
        if Xva is not None:
            tl, ta, _ = evaluate(model, Xtr_t, ytr_t, loss_fn)
            vl, va, _ = evaluate(model, Xva_t, yva_t, loss_fn)
            for k, v in zip(hist, (tl, ta, vl, va)):
                hist[k].append(v)
    return model, {k: np.array(v) for k, v in hist.items()}


def predict_logits(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


# ----------------------------------------------------------------------------
# Calibration helpers
# ----------------------------------------------------------------------------
def nll(probs, y):
    return float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))


def fit_temperature(logits, y):
    f = lambda lt: nll(softmax(logits / np.exp(lt), axis=1), y)
    return float(np.exp(minimize_scalar(f, bounds=(np.log(0.05), np.log(20)), method="bounded").x))


def ece_rows(probs, y, n_bins):
    conf, correct = probs.max(1), (probs.argmax(1) == y).astype(float)
    order, ece, rows = np.argsort(conf), 0.0, []
    for idx in np.array_split(order, n_bins):
        c, a = conf[idx].mean(), correct[idx].mean()
        ece += len(idx) / len(y) * abs(a - c)
        rows.append((c, a, len(idx)))
    return ece, np.array(rows)


def brier(probs, y):
    return float(np.mean(np.sum((probs - np.eye(probs.shape[1])[y]) ** 2, axis=1)))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default="tm_results",
                    help="Folder with features_train.csv and features_test.csv")
    ap.add_argument("--hidden_grid", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_epochs", type=int, default=600)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calib_bins", type=int, default=5)
    ap.add_argument("--out", default="dnn_results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    tr_df = pd.read_csv(os.path.join(args.features_dir, "features_train.csv"))
    te_df = pd.read_csv(os.path.join(args.features_dir, "features_test.csv"))
    feat_names = [c for c in tr_df.columns if c not in ("patient", "label")]
    X_train, X_test = tr_df[feat_names].values.astype(np.float64), te_df[feat_names].values.astype(np.float64)
    y_train = tr_df["label"].map(CLASSES.index).values
    y_test = te_df["label"].map(CLASSES.index).values
    print(f"Train {len(y_train)} | Test {len(y_test)} | {len(feat_names)} features")

    # ---- 5-fold CV: capacity (hidden_dim) and best epoch ---------------------
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = list(skf.split(X_train, y_train))
    cv = {}
    for h in args.hidden_grid:
        hists = []
        for k, (tr, va) in enumerate(folds):
            Xtr, Xva = standardise(X_train[tr], X_train[va])
            _, hist = train(Xtr, y_train[tr], h, args, args.seed + k, args.max_epochs, Xva, y_train[va])
            hists.append(hist)
        mean = {key: np.mean([hh[key] for hh in hists], axis=0) for key in hists[0]}
        std = {key: np.std([hh[key] for hh in hists], axis=0) for key in hists[0]}
        best_ep = int(np.argmin(mean["val_loss"]))
        cv[h] = {"mean": mean, "std": std, "best_epoch": best_ep + 1,
                 "val_loss": float(mean["val_loss"][best_ep]), "val_acc": float(mean["val_acc"][best_ep]),
                 "train_loss": float(mean["train_loss"][best_ep]), "train_acc": float(mean["train_acc"][best_ep])}
        print(f"hidden_dim={h:>4}: min CV val loss {cv[h]['val_loss']:.3f} at epoch {best_ep + 1:>3} "
              f"| val acc {cv[h]['val_acc']:.3f} | train acc {cv[h]['train_acc']:.3f}")

    best_h = min(cv, key=lambda h: cv[h]["val_loss"])
    best_epoch = cv[best_h]["best_epoch"]
    print(f"\nSelected hidden_dim={best_h}, epochs={best_epoch} "
          f"(CV val loss {cv[best_h]['val_loss']:.3f}, CV acc {cv[best_h]['val_acc']:.3f})")

    # ---- Out-of-fold logits with the selected config (for temperature scaling) ----
    oof = np.zeros((len(y_train), len(CLASSES)))
    for k, (tr, va) in enumerate(folds):
        Xtr, Xva = standardise(X_train[tr], X_train[va])
        m, _ = train(Xtr, y_train[tr], best_h, args, args.seed + k, best_epoch)
        oof[va] = predict_logits(m, Xva)
    tau = fit_temperature(oof, y_train)

    # ---- Final model on all 100 training patients -> test once ---------------
    Xtr_s, Xte_s = standardise(X_train, X_test)
    model, _ = train(Xtr_s, y_train, best_h, args, args.seed, best_epoch)
    logits = predict_logits(model, Xte_s)
    y_pred = logits.argmax(1)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nTEST accuracy (patients 101-150): {test_acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=CLASSES, digits=3))
    cm = confusion_matrix(y_test, y_pred, labels=range(5))
    print("Confusion matrix (rows=true, cols=pred):\n", cm)
    wrong = [(p, CLASSES[t], CLASSES[q]) for p, t, q in zip(te_df["patient"], y_test, y_pred) if t != q]
    print("Misclassified:", wrong)

    p_raw, p_cal = softmax(logits, axis=1), softmax(logits / tau, axis=1)
    ece_raw, rows_raw = ece_rows(p_raw, y_test, args.calib_bins)
    ece_cal, rows_cal = ece_rows(p_cal, y_test, args.calib_bins)
    print(f"\nCalibration (test): tau = {tau:.3f} (fitted on training out-of-fold)")
    print(f"  raw:    ECE {ece_raw:.3f} | Brier {brier(p_raw, y_test):.3f} | NLL {nll(p_raw, y_test):.3f}")
    print(f"  scaled: ECE {ece_cal:.3f} | Brier {brier(p_cal, y_test):.3f} | NLL {nll(p_cal, y_test):.3f}")

    # ---- Save ----------------------------------------------------------------
    with open(os.path.join(args.out, "dnn_results.json"), "w") as f:
        json.dump({
            "features": feat_names, "best_hidden_dim": best_h, "best_epoch": best_epoch,
            "cv": {h: {k: v for k, v in d.items() if k not in ("mean", "std")} for h, d in cv.items()},
            "test_accuracy": test_acc, "confusion_matrix": cm.tolist(), "misclassified": wrong,
            "hyperparams": {k: getattr(args, k) for k in ("dropout", "lr", "weight_decay", "batch_size", "seed")},
            "calibration": {"temperature": tau, "ece_raw": ece_raw, "brier_raw": brier(p_raw, y_test),
                            "ece_scaled": ece_cal, "brier_scaled": brier(p_cal, y_test)},
        }, f, indent=2)
    with open(os.path.join(args.out, "dnn_test_probabilities.csv"), "w") as f:
        f.write("patient,true,pred," + ",".join(f"p_{c}" for c in CLASSES) + "\n")
        for pid, t, q, pr in zip(te_df["patient"], y_test, y_pred, p_cal):
            f.write(f"{pid},{CLASSES[t]},{CLASSES[q]}," + ",".join(f"{v:.4f}" for v in pr) + "\n")

    # ---- Plots ---------------------------------------------------------------
    hs = list(cv)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(hs, [cv[h]["train_loss"] for h in hs], "o-", label="Train loss")
    a1.plot(hs, [cv[h]["val_loss"] for h in hs], "s-", label="Validation loss")
    a2.plot(hs, [cv[h]["train_acc"] for h in hs], "o-", label="Train accuracy")
    a2.plot(hs, [cv[h]["val_acc"] for h in hs], "s-", label="Validation accuracy")
    for a, yl in ((a1, "Cross-entropy (at best epoch)"), (a2, "Accuracy (at best epoch)")):
        a.axvline(best_h, ls="--", c="gray")
        a.set_xscale("log", base=2); a.set_xticks(hs); a.set_xticklabels(hs)
        a.set(xlabel="hidden_dim", ylabel=yl); a.grid(alpha=0.3); a.legend()
    fig.suptitle("DNN capacity selection (5-fold CV, patients 001-100)")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "dnn_capacity.png"), dpi=200); plt.close(fig)

    m, s = cv[best_h]["mean"], cv[best_h]["std"]
    ep = np.arange(1, args.max_epochs + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for key, lab, a in (("train_loss", "Train", a1), ("val_loss", "Validation", a1),
                        ("train_acc", "Train", a2), ("val_acc", "Validation", a2)):
        a.plot(ep, m[key], label=lab)
        a.fill_between(ep, m[key] - s[key], m[key] + s[key], alpha=0.15)
    for a, yl in ((a1, "Cross-entropy loss"), (a2, "Accuracy")):
        a.axvline(best_epoch, ls="--", c="gray", label=f"Best epoch = {best_epoch}")
        a.set(xlabel="Epoch", ylabel=yl); a.grid(alpha=0.3); a.legend()
    fig.suptitle(f"DNN learning curves, hidden_dim={best_h} (mean ± std over 5 folds)")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "dnn_loss_curves.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5), CLASSES); ax.set_yticks(range(5), CLASSES)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xlabel="Predicted", ylabel="True", title=f"DNN – test set (101-150), acc = {test_acc:.2f}")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "dnn_test_confusion.png"), dpi=200); plt.close(fig)

    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6, 6.8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1, label="Perfect calibration")
    for p, rows, e, name, col in ((p_raw, rows_raw, ece_raw, "Raw softmax", "#D55E00"),
                                  (p_cal, rows_cal, ece_cal, f"Temperature-scaled (τ={tau:.2f})", "#0072B2")):
        ax.plot(rows[:, 0], rows[:, 1], "o-", c=col, lw=2,
                label=f"{name} — ECE {e:.3f}, Brier {brier(p, y_test):.3f}")
        axh.hist(p.max(1), bins=np.linspace(0, 1, 21), color=col, alpha=0.55)
    ax.set(ylabel="Observed accuracy", xlim=(0, 1.02), ylim=(0, 1.04),
           title=f"DNN calibration on test set ({args.calib_bins} equal-count bins)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    axh.set(xlabel="Predicted confidence (max class probability)", ylabel="Count"); axh.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "dnn_calibration.png"), dpi=200); plt.close(fig)

    print(f"\nSaved results and figures to ./{args.out}/")


if __name__ == "__main__":
    main()
