import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# P: (50, 5) ensemble probabilities on the test set, 0.5 * (p_mlp + p_rf)
# y: (50,) true class indices, in the same column order as P
P = np.load("probs_test_ensemble.npy")
y = np.load("labels_test.npy")

N_BINS = 5                                   # keep it low: only 50 cases
conf = P.max(1)                              # top-label confidence
ok = (P.argmax(1) == y).astype(float)        # 1 if correct

edges = np.linspace(1 / P.shape[1], 1.0, N_BINS + 1)   # confidence >= 1/5
ctr = (edges[:-1] + edges[1:]) / 2
idx = np.clip(np.digitize(conf, edges) - 1, 0, N_BINS - 1)

n = np.array([(idx == b).sum() for b in range(N_BINS)])
acc = np.array([ok[idx == b].mean() if n[b] else np.nan for b in range(N_BINS)])
cm = np.array([conf[idx == b].mean() if n[b] else np.nan for b in range(N_BINS)])
ece = np.nansum(n / n.sum() * np.abs(acc - cm))        # expected calibration error

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.5, 6.5),
                              gridspec_kw={"height_ratios": [3, 1], "hspace": 0.3})
w = (edges[1] - edges[0]) * 0.95
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax.bar(ctr, np.nan_to_num(acc), width=w, color="#4C78A8", alpha=.6,
       edgecolor="k", label="Accuracy per bin")
ax.plot(cm, acc, "o-", color="#E45756", label="Model")
ax.set_xlim(edges[0], 1); ax.set_ylim(0, 1.05)
ax.set_ylabel("Accuracy"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
ax.set_title("Calibration – final model (MLP+RF ensemble), test set\n"
             f"ECE={ece:.3f}, accuracy={ok.mean():.2f}, "
             f"mean confidence={conf.mean():.2f} (n={len(y)})", fontsize=10)

ax2.bar(ctr, n, width=w, color="gray", edgecolor="k")
for x, v in zip(ctr, n):
    ax2.text(x, v, str(v), ha="center", va="bottom", fontsize=8)
ax2.set_xlim(edges[0], 1); ax2.set_ylim(0, n.max() * 1.25)
ax2.set_xlabel("Top-label confidence"); ax2.set_ylabel("# cases")

fig.savefig("calibration_final_model_test.png", dpi=160, bbox_inches="tight")