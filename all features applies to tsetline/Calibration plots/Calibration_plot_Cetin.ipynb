import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---- data & final model (RBF, C=1, 9 features selected by SFS: from classification.json) ----
cfg = json.load(open("classification.json"))
feats = cfg["kernels"][cfg["kernel"]]["selected"]
gt = pd.read_csv("radiomics_gt.csv")
tr = gt[gt.pid <= 100]
te = pd.read_csv("radiomics_isensee.csv")            # automatic (Isensee) masks, test cases 101-150
CLASSES = ["NOR", "DCM", "HCM", "MINF", "RV"]
ytr = tr.group.map(CLASSES.index).values
yte = te.group.map(CLASSES.index).values

# probability=True -> Platt scaling (internal 5-fold CV) so the SVM outputs class probabilities
model = make_pipeline(StandardScaler(),
                      SVC(kernel=cfg["kernel"], C=1.0, probability=True, random_state=0))
model.fit(tr[feats].values, ytr)
P = model.predict_proba(te[feats].values)             # (50, 5), columns follow model.classes_ = 0..4
print("acc argmax(proba): %.2f | acc predict(): %.2f" % (
    (P.argmax(1) == yte).mean(), (model.predict(te[feats].values) == yte).mean()))

# ---- reliability diagram ----
N_BINS = 5
conf = P.max(1)
ok = (P.argmax(1) == yte).astype(float)
edges = np.linspace(1 / P.shape[1], 1.0, N_BINS + 1)
ctr = (edges[:-1] + edges[1:]) / 2
idx = np.clip(np.digitize(conf, edges) - 1, 0, N_BINS - 1)
n = np.array([(idx == b).sum() for b in range(N_BINS)])
acc = np.array([ok[idx == b].mean() if n[b] else np.nan for b in range(N_BINS)])
cm = np.array([conf[idx == b].mean() if n[b] else np.nan for b in range(N_BINS)])
ece = np.nansum(n / n.sum() * np.abs(acc - cm))

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.5, 6.5), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.3})
w = (edges[1] - edges[0]) * 0.95
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax.bar(ctr, np.nan_to_num(acc), width=w, color="#4C78A8", alpha=.6, edgecolor="k", label="Accuracy per bin")
ax.plot(cm, acc, "o-", color="#E45756", label="Model")
ax.set_xlim(edges[0], 1); ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy"); ax.grid(alpha=.3)
ax.legend(fontsize=8, loc="upper left")
ax.set_title("Calibration – final model (SVM-RBF, SFS features), test set\n"
             f"ECE={ece:.3f}, accuracy={ok.mean():.2f}, mean confidence={conf.mean():.2f} (n={len(yte)})", fontsize=10)
ax2.bar(ctr, n, width=w, color="gray", edgecolor="k")
for x, v in zip(ctr, n):
    ax2.text(x, v, str(v), ha="center", va="bottom", fontsize=8)
ax2.set_xlim(edges[0], 1); ax2.set_ylim(0, n.max() * 1.25)
ax2.set_xlabel("Top-label confidence"); ax2.set_ylabel("# cases")
fig.savefig("calibration_cetin_test.png", dpi=160, bbox_inches="tight")
print(n, np.round(acc, 2), np.round(cm, 2), round(ece, 3))
