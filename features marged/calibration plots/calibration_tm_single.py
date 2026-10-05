"""
Calibration plot of the best Tsetlin Machine (57 clinical/geometric features) on the
test set (patients 101-150), drawn in the same format as the MLP+RF ensemble figure:
top = accuracy per confidence bin + model curve, bottom = number of cases per bin.

Probabilities = temperature-scaled class sums, tau taken from calibration_results.json
(fitted on out-of-fold CV predictions of patients 1-100 only). The TM is stochastic, so
N_RUNS models are trained and the run with the median test accuracy is plotted (n = 50).
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from pyTsetlinMachine.tm import MultiClassTsetlinMachine

from tsetlin_acdc import CLASSES, Thermometer, load

cfg_all = json.load(open("tsetlin_results.json"))["clinical"]
cfg, EPOCHS = cfg_all["best_config"], cfg_all["best_epoch"]
NB, CL, T, S = cfg["n_bins"], cfg["clauses"], cfg["T"], cfg["s"]
Xtr, ytr, Xte, yte, feats = load("clinical")


def train(Xa, ya):
    enc = Thermometer(NB).fit(Xa)
    tm = MultiClassTsetlinMachine(CL, T, S)
    tm.fit(enc.transform(Xa), ya, epochs=EPOCHS)
    return tm, enc


def class_sums(tm, B):
    out = tm.transform(B, inverted=False).reshape(len(B), len(CLASSES), CL).astype(int)
    sign = np.where(np.arange(CL) % 2 == 0, 1, -1)
    return np.clip((out * sign).sum(-1), -T, T)


def p_temp(v, tau):
    z = v / (T * tau)
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

N_RUNS = 10
EDGES = np.linspace(0.2, 1.0, 6)  # 5 bins; 0.2 = minimum possible top-label confidence (5 classes)

tau = json.load(open("calibration_results.json"))["tau"]

runs = []
for _ in range(N_RUNS):
    tm, enc = train(Xtr, ytr)
    p = p_temp(class_sums(tm, enc.transform(Xte)), tau)
    runs.append((np.mean(p.argmax(1) == yte), p))
runs.sort(key=lambda r: r[0])
acc, P = runs[len(runs) // 2]

conf = P.max(1)
correct = (P.argmax(1) == yte).astype(float)
idx = np.clip(np.digitize(conf, EDGES) - 1, 0, len(EDGES) - 2)
centers, width = (EDGES[:-1] + EDGES[1:]) / 2, EDGES[1] - EDGES[0]
counts = np.array([(idx == b).sum() for b in range(len(centers))])
bin_acc = np.array([correct[idx == b].mean() if counts[b] else np.nan for b in range(len(centers))])
bin_conf = np.array([conf[idx == b].mean() if counts[b] else np.nan for b in range(len(centers))])
ece = np.nansum(counts * np.abs(bin_acc - bin_conf)) / len(conf)

fig, (ax, axh) = plt.subplots(2, 1, figsize=(5.4, 7.6), gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.32})
has = counts > 0
ax.bar(centers[has], bin_acc[has], width=width * 0.95, color="#92aecb", edgecolor="#4a4a4a",
       label="Accuracy per bin", zorder=2)
ax.plot([0.2, 1.0], [0.2, 1.0], "k--", lw=1, label="Perfect calibration", zorder=3)
ax.plot(bin_conf[has], bin_acc[has], "o-", color="#e05252", lw=2, ms=7, label="Model", zorder=4)
ax.set(xlim=(0.2, 1.0), ylim=(0, 1.05), ylabel="Accuracy")
ax.grid(alpha=0.35, zorder=0)
h, l = ax.get_legend_handles_labels()
order = [l.index(x) for x in ("Perfect calibration", "Model", "Accuracy per bin")]
ax.legend([h[i] for i in order], [l[i] for i in order], loc="center left", bbox_to_anchor=(0.0, 0.66), fontsize=8.5)
ax.set_title(f"Calibration – final model (Tsetlin Machine), test set\n"
             f"ECE={ece:.3f}, accuracy={correct.mean():.2f}, mean confidence={conf.mean():.2f} (n={len(conf)})")

axh.bar(centers, counts, width=width * 0.95, color="grey", edgecolor="black")
for c, n in zip(centers, counts):
    if n:
        axh.text(c, n, str(n), ha="center", va="bottom", fontsize=9)
axh.set(xlim=(0.2, 1.0), ylim=(0, max(counts) * 1.25), xlabel="Top-label confidence", ylabel="# cases")

fig.savefig("calibration_tm_testset.png", dpi=200, bbox_inches="tight")
print(f"median run: acc {correct.mean():.2f}, mean conf {conf.mean():.3f}, ECE {ece:.3f}, "
      f"counts {counts.tolist()}, bin acc {np.round(bin_acc, 2).tolist()}")
print("all runs acc:", [round(r[0], 2) for r in runs])
