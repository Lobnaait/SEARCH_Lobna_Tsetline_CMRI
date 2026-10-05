"""Calibration plot (test set only) of the final Khened model: stage-1 soft-voting ensemble + stage-2 DCM/MINF expert.
Run from the folder containing features.py and results/features_gt_train.csv, results/features_pred_test.csv."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from features import MEDIA_FEATURES, STAGE2_FEATURES

RES = "results/"          # change if the CSVs are elsewhere
SEED = 0


def media_stage1(seed=0):                 # identical to classify.py
    return make_pipeline(StandardScaler(), VotingClassifier([
        ('mlp', MLPClassifier(hidden_layer_sizes=(100, 100), random_state=1, max_iter=1000)),
        ('gnb', GaussianNB()),
        ('svm', SVC(kernel='rbf', probability=True, random_state=seed)),
        ('rf', RandomForestClassifier(n_estimators=1000, random_state=seed))], voting='soft'))


def media_stage2():                       # identical to classify.py
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100, 100), random_state=1, max_iter=1000))


# ---- fit on the 100 training cases (GT features), predict the 50 test cases (automatic-segmentation features) ----
tr = pd.read_csv(RES + "features_gt_train.csv")
te = pd.read_csv(RES + "features_pred_test.csv")
y = te.group.values

s1 = media_stage1(SEED).fit(tr[MEDIA_FEATURES].values, tr.group.values)
cl = list(s1.classes_)                                    # DCM, HCM, MINF, NOR, RV
P = s1.predict_proba(te[MEDIA_FEATURES].values)           # stage-1 probabilities
pred1 = s1.predict(te[MEDIA_FEATURES].values)

sub = tr[tr.group.isin(["DCM", "MINF"])]
s2 = media_stage2().fit(sub[STAGE2_FEATURES].values, sub.group.values)
m = np.isin(pred1, ["DCM", "MINF"])                       # cases re-classified by stage 2
iD, iM = cl.index("DCM"), cl.index("MINF")
if m.any():
    # Probabilities for the 2-stage model (NOT in the original pipeline): the stage-1 mass on DCM+MINF
    # is split between the two classes according to the stage-2 expert; other classes keep stage-1 values.
    q = s2.predict_proba(te[STAGE2_FEATURES].values[m])   # columns: DCM, MINF
    mass = P[m][:, [iD, iM]].sum(1)
    P[np.ix_(m, [iD, iM])] = q * mass[:, None]
pred = np.array(cl)[P.argmax(1)]                          # equals the 2-stage prediction of classify.py

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

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.5, 6.5), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.3})
w = (edges[1] - edges[0]) * 0.95
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax.bar(ctr, np.nan_to_num(acc), width=w, color="#4C78A8", alpha=.6, edgecolor="k", label="Accuracy per bin")
ax.plot(cm, acc, "o-", color="#E45756", label="Model")
ax.set_xlim(edges[0], 1); ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy"); ax.grid(alpha=.3)
ax.legend(fontsize=8, loc="upper left")
ax.set_title("Calibration – final model (Khened, 2-stage), test set\n"
             f"ECE={ece:.3f}, accuracy={ok.mean():.2f}, mean confidence={conf.mean():.2f} (n={len(y)})", fontsize=10)
ax2.bar(ctr, n, width=w, color="gray", edgecolor="k")
for x, v in zip(ctr, n):
    ax2.text(x, v, str(v), ha="center", va="bottom", fontsize=8)
ax2.set_xlim(edges[0], 1); ax2.set_ylim(0, max(n.max(), 1) * 1.25)
ax2.set_xlabel("Top-label confidence"); ax2.set_ylabel("# cases")
fig.savefig("calibration_khened_test.png", dpi=160, bbox_inches="tight")
