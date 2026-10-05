"""
Merge the Wolterink, Khened, Isensee and Cetin feature sets (ACDC, 150 patients)
and remove redundant features in two stages:

  Stage 1 – physiological rules: when several sources measure the SAME physiological
            quantity (e.g. LV end-diastolic volume), keep one representative and drop
            the others. Also drop features that are exact mathematical functions of
            features already kept (e.g. Variance = StdDev^2, Sphericity vs Compactness).
  Stage 2 – data check: among the survivors, any pair with |Spearman rho| >= 0.95 on the
            TRAINING patients only (pid 1-100) is considered the same information; the
            feature with higher clinical priority is kept.

Outputs:
  merged_features_reduced.csv   pid, group, selected features (one prefix per source)
  merged_features_all.csv       all features before reduction (same prefixes)
  feature_reduction_report.csv  every original feature with its decision and reason
"""
import re
import numpy as np
import pandas as pd

DATA = "data"
SOURCES = {
    "W": f"{DATA}/Wolterink_features.txt",
    "K": f"{DATA}/Khened_features.csv",
    "I": f"{DATA}/Isensee_feature.csv",
    "C": f"{DATA}/Cetin_features.csv",
}
TRAIN_PIDS = range(1, 101)
RHO_THRESHOLD = 0.95

# ---------------------------------------------------------------- load + merge
dfs = {p: pd.read_csv(f).set_index("pid").sort_index() for p, f in SOURCES.items()}
groups = [d["group"] for d in dfs.values()]
assert all((g == groups[0]).all() for g in groups), "labels disagree between files"
group = groups[0]

merged = pd.concat(
    [group] + [d.drop(columns="group").add_prefix(p + "__") for p, d in dfs.items()], axis=1
)
merged.to_csv("merged_features_all.csv")
features = [c for c in merged.columns if c != "group"]

decision = {}  # feature -> (kept?, reason, kept_instead)


def drop(feat, reason, instead=""):
    assert feat in features, feat
    decision[feat] = (False, reason, instead)


# ---------------------------------------------------------------- stage 1: rules
# 1) Anthropometrics: identical in all four files -> keep Wolterink copy only.
for p in "KIC":
    for f in ("height", "weight"):
        drop(f"{p}__{f}", "identical copy of patient " + f, f"W__{f}")
drop("I__BMI", "deterministic function of height and weight", "W__height, W__weight")

# 2) Chamber / myocardial volumes. Same quantity measured by 4 sources.
#    Keep the BSA-indexed volumes (Isensee) = clinical standard (ESC/ASE guidelines);
#    raw volumes = indexed volume x BSA(height, weight) -> same information.
vol_map = {  # (structure, phase) -> Isensee indexed representative
    ("LV", "ED"): "I__ED_LVC_volume_per_bsa", ("LV", "ES"): "I__ES_LVC_volume_per_bsa",
    ("RV", "ED"): "I__ED_RVC_volume_per_bsa", ("RV", "ES"): "I__ES_RVC_volume_per_bsa",
    ("MYO", "ED"): "I__ED_LVM_volume_per_bsa", ("MYO", "ES"): "I__ES_LVM_volume_per_bsa",
}
isensee_struct = {"LV": "LVC", "RV": "RVC", "MYO": "LVM"}
for (s, ph), keep in vol_map.items():
    why = f"{s} {ph} volume (same quantity as BSA-indexed volume)"
    drop(f"W__{s}_{ph}", why, keep)
    drop(f"K__{ph}_{s}", why, keep)
    drop(f"C__{s}_{ph}_shape_MeshVolume", why, keep)
    drop(f"C__{s}_{ph}_shape_VoxelVolume", why, keep)
drop("I__ED_LVM_mass", "myocardial mass = ED myocardial volume x 1.05 g/ml", "I__ED_LVM_volume_per_bsa")
drop("I__ES_LVM_mass", "myocardial mass = ES myocardial volume x 1.05 g/ml", "I__ES_LVM_volume_per_bsa")
for s in ("LVC", "RVC"):
    drop(f"I__dyn_{s}_vmax", "max volume over cardiac cycle = ED volume", f"I__ED_{s}_volume_per_bsa")
    drop(f"I__dyn_{s}_vmin", "min volume over cardiac cycle = ES volume", f"I__ES_{s}_volume_per_bsa")
drop("I__dyn_LVM_vmax", "max myocardial volume over cycle ~ myocardial volume", "I__ED_LVM_volume_per_bsa")
drop("I__dyn_LVM_vmin", "min myocardial volume over cycle ~ myocardial volume", "I__ES_LVM_volume_per_bsa")

# 3) Ejection fraction: three sources -> keep Wolterink.
#    (EF is kept although computable from ED/ES volumes: it is a distinct physiological
#     concept – systolic FUNCTION – and a TM cannot form ratios from thresholds.)
drop("K__EF_LV", "LV ejection fraction (duplicate)", "W__EF_LV")
drop("K__EF_RV", "RV ejection fraction (duplicate)", "W__EF_RV")
drop("I__dyn_LVC_dyn_EF", "LV ejection fraction from volume curve (duplicate)", "W__EF_LV")
drop("I__dyn_RVC_dyn_EF", "RV ejection fraction from volume curve (duplicate)", "W__EF_RV")

# 4) Volume ratios: Khened LV/RV is the inverse of Wolterink RV/LV; MYO/LV identical.
for ph in ("ED", "ES"):
    drop(f"K__{ph}_LV_RV", "LV/RV volume ratio = 1 / (RV/LV)", f"W__RV_LV_{ph}")
    drop(f"K__{ph}_MYO_LV", "MYO/LV volume ratio (duplicate)", f"W__MYO_LV_{ph}")
drop("I__dyn_ratio_vmin_LVC_RVC", "LV/RV ratio at end-systole = 1 / (RV/LV ES)", "W__RV_LV_ES")
drop("I__dyn_ratio_vmin_LVM_LVC", "MYO/LV ratio at end-systole (duplicate)", "W__MYO_LV_ES")
drop("I__dyn_ratio_vmin_RVC_LVM", "RV/MYO = (RV/LV)/(MYO/LV): function of kept ratios", "W__RV_LV_ES, W__MYO_LV_ES")

# 5) Myocardial wall thickness. Isensee gives max/min/mean/std (+septum);
#    Khened max_mean (mean of per-slice max) and std_mean (mean of per-slice std)
#    describe the same quantities. Khened *_std (variation ACROSS slices) is new
#    information (apex-to-base heterogeneity) and is kept.
for ph in ("ED", "ES"):
    drop(f"K__{ph}_MWT_max_mean", "maximal wall thickness (duplicate)", f"I__{ph}_LVM_max_thickness")
    drop(f"K__{ph}_MWT_std_mean", "in-slice wall-thickness variability (duplicate)", f"I__{ph}_LVM_std_thickness")

# 6) Cetin shape features: drop exact functions of other kept shape features.
for s in ("RV", "MYO", "LV"):
    for ph in ("ED", "ES"):
        b = f"C__{s}_{ph}_shape_"
        drop(b + "Compactness1", "function of Sphericity (same V/A^1.5 index)", b + "Sphericity")
        drop(b + "Compactness2", "= Sphericity^3", b + "Sphericity")
        drop(b + "SphericalDisproportion", "= 1 / Sphericity", b + "Sphericity")
        drop(b + "SurfaceVolumeRatio", "= SurfaceArea / Volume (both kept)", b + "SurfaceArea")
        drop(b + "MinorAxisLength", "= Elongation^2 x MajorAxisLength", b + "Elongation")
        drop(b + "LeastAxisLength", "= Flatness^2 x MajorAxisLength", b + "Flatness")
        drop(b + "Maximum2DDiameterRow", "long-axis maximal diameter, covered by 3D diameter", b + "Maximum3DDiameter")
        drop(b + "Maximum2DDiameterColumn", "long-axis maximal diameter, covered by 3D diameter", b + "Maximum3DDiameter")

# 7) Cetin first-order intensity features: exact algebraic redundancies.
for s in ("RV", "MYO", "LV"):
    for ph in ("ED", "ES"):
        b = f"C__{s}_{ph}_firstorder_"
        drop(b + "TotalEnergy", "= Energy x voxel volume", b + "Energy")
        drop(b + "Variance", "= StandardDeviation^2", b + "StandardDeviation")
        drop(b + "Range", "= Maximum - Minimum", b + "Maximum, " + b + "Minimum")
        drop(b + "RootMeanSquared", "= sqrt(Mean^2 + Variance)", b + "Mean, " + b + "StandardDeviation")
        g = f"C__{s}_{ph}_glcm_"
        drop(g + "SumAverage", "= 2 x JointAverage (symmetric GLCM)", g + "JointAverage")

# ---------------------------------------------------------------- stage 2: data check
stage1_kept = [f for f in features if f not in decision]


def priority(f):
    """Lower = more important. Clinical/physiological > Cetin shape > Cetin texture."""
    if not f.startswith("C__"):
        return 0
    if "_shape_" in f or f == "C__ED_ES_duration":
        return 1
    if "_firstorder_" in f:
        return 2
    return 3  # glcm / glrlm / glszm


Xtr = merged.loc[merged.index.isin(TRAIN_PIDS), stage1_kept]
# constant features carry no information
for f in stage1_kept:
    if Xtr[f].nunique() <= 1:
        drop(f, "constant on training set")
stage1_kept = [f for f in stage1_kept if f not in decision]

rho = Xtr[stage1_kept].rank().corr().abs()  # Spearman
order = sorted(stage1_kept, key=lambda f: (priority(f), stage1_kept.index(f)))
kept = []
for f in order:
    hits = [k for k in kept if rho.loc[f, k] >= RHO_THRESHOLD]
    if hits:
        best = max(hits, key=lambda k: rho.loc[f, k])
        drop(f, f"|Spearman rho| = {rho.loc[f, best]:.3f} with kept feature (train set)", best)
    else:
        kept.append(f)

for f in kept:
    decision[f] = (True, "kept", "")

# ---------------------------------------------------------------- save
kept_in_order = [f for f in features if decision[f][0]]
reduced = merged[["group"] + kept_in_order]
reduced.to_csv("merged_features_reduced.csv")

rep = pd.DataFrame(
    [
        {
            "feature": f,
            "source": {"W": "Wolterink", "K": "Khened", "I": "Isensee", "C": "Cetin"}[f[0]],
            "kept": decision[f][0],
            "stage": "" if decision[f][0] else ("2-correlation" if "Spearman" in decision[f][1] or "constant" in decision[f][1] else "1-physiology"),
            "reason": decision[f][1],
            "kept_instead": decision[f][2],
        }
        for f in features
    ]
)
rep.to_csv("feature_reduction_report.csv", index=False)

print(f"Original features : {len(features)}")
print(f"After stage 1     : {len(features) - (rep.stage == '1-physiology').sum()}")
print(f"After stage 2     : {len(kept_in_order)}")
print(rep.groupby(["source", "kept"]).size().unstack(fill_value=0))
print("\nKept by family:")
fam = lambda f: ("clinical" if not f.startswith("C__") else
                 ("shape" if "_shape_" in f else ("timing" if "duration" in f else
                  re.search(r"_(firstorder|glcm|glrlm|glszm)_", f).group(1))))
print(pd.Series([fam(f) for f in kept_in_order]).value_counts())
