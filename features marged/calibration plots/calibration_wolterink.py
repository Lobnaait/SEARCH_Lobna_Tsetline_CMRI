"""Calibration plot (test set only) of the final Wolterink model: random forest, 1000 full-depth trees,
14 features, trained on the 100 training cases (features from out-of-fold CNN segmentations),
evaluated on the 50 test cases (features from the CNN snapshot ensemble).
Run from the folder containing results/features_pred.csv (output of classify.py)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

RES = "results/"          # change if the CSV is elsewhere
SOURCE = "pred"           # "pred" = automatic segmentations, "gt" = reference segmentations
FEATURES = ['weight', 'height', 'LV_ED', 'RV_ED', 'MYO_ED', 'LV_ES', 'RV_ES', 'MYO_ES', 'EF_LV', 'EF_RV',
            'RV_LV_ED', 'RV_LV_ES', 'MYO_LV_ED', 'MYO_LV_ES']

df = pd.read_csv(RES + "features_%s.csv" % SOURCE)
tr, te = df[df.pid <= 100], df[df.pid > 100]

# identical to classify.py: rf(0) fitted on all 100 training cases
m = RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=0, n_jobs=8)
m.fit(tr[FEATURES], tr.group)
P = m.predict_proba(te[FEATURES])                       # (50, 5), columns = m.classes_
pred = m.classes_[P.argmax(1)]
y = te.group.values

# ---- reliability diagram ----
N_BINS = 5
conf = P.max(1)
ok = (pred == y).astype(float)
edges = np.linspace(1 / P.shape[1], 1.0, N_BINS + 1)
ctr = (edges[:-1] + edges[1:]) / 2
idx = np.clip(np.digitize(conf, edges) - 1, 0, N_BINS - 1)
n = np.array([(idx == b).sum() for b in range(N_BINS)])
acc = np.array([ok[idx == b].mean() if n[b] else np.nan for b in range(N_BINS)])
cm = np.array([conf[idx == b].mean() if n[b] else np.nan for b in range(N_BINS)])
ece = np.nansum(n / n.sum() * np.abs(acc - cm))
print("test accuracy %.2f | ECE %.3f | mean conf %.2f | bin counts %s" % (ok.mean(), ece, conf.mean(), n))
print("accuracy per bin:", np.round(acc, 2), "| mean conf per bin:", np.round(cm, 2))

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.5, 6.5), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.3})
w = (edges[1] - edges[0]) * 0.95
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax.bar(ctr, np.nan_to_num(acc), width=w, color="#4C78A8", alpha=.6, edgecolor="k", label="Accuracy per bin")
ax.plot(cm, acc, "o-", color="#E45756", label="Model")
ax.set_xlim(edges[0], 1); ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy"); ax.grid(alpha=.3)
ax.legend(fontsize=8, loc="upper left")
ax.set_title("Calibration – final model (Wolterink, RF-1000), test set\n"
             f"ECE={ece:.3f}, accuracy={ok.mean():.2f}, mean confidence={conf.mean():.2f} (n={len(y)})", fontsize=10)
ax2.bar(ctr, n, width=w, color="gray", edgecolor="k")
for x, v in zip(ctr, n):
    ax2.text(x, v, str(v), ha="center", va="bottom", fontsize=8)
ax2.set_xlim(edges[0], 1); ax2.set_ylim(0, max(n.max(), 1) * 1.25)
ax2.set_xlabel("Top-label confidence"); ax2.set_ylabel("# cases")
fig.savefig("calibration_wolterink_test.png", dpi=160, bbox_inches="tight")
