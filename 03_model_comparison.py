"""
STEP 3: 9-Model Comparative Analysis (8 base + 1 stacking ensemble)
CO02: Supervised ML algorithms
CO03: Unsupervised (PCA, K-Means)
CO04: Error analysis + performance improvement
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.svm              import SVC
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.ensemble         import RandomForestClassifier, StackingClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.neural_network   import MLPClassifier
from sklearn.naive_bayes      import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition    import PCA
from sklearn.cluster          import KMeans
from sklearn.model_selection  import cross_val_score, StratifiedKFold
from sklearn.metrics          import (accuracy_score, f1_score,
                                      confusion_matrix, precision_score, recall_score)
from xgboost import XGBClassifier

FEAT_DIR   = "./features"
REPORT_DIR = "./reports"
import os; os.makedirs(REPORT_DIR, exist_ok=True)

# ── LOAD SPLITS ────────────────────────────────────────────────────────────────
print("Loading features and splits...")
with open(f"{FEAT_DIR}/splits.pkl", "rb") as f:
    data = pickle.load(f)

X_train = data["X_train"]
X_test  = data["X_test"]
y_train = data["y_train"]
y_test  = data["y_test"]
le      = data["label_encoder"]

n_classes = len(np.unique(y_train))
print(f"Train: {X_train.shape}, Test: {X_test.shape}, Classes: {n_classes}")

# ── CO03: UNSUPERVISED ANALYSIS ────────────────────────────────────────────────
print("\n=== CO03: Unsupervised Analysis ===")
pca_full = PCA().fit(X_train)
cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
n_95pct  = np.argmax(cumvar >= 0.95) + 1
print(f"  PCA: {n_95pct} components explain 95% variance")

pca_128     = PCA(n_components=min(128, X_train.shape[1]-1), random_state=42)
X_train_pca = pca_128.fit_transform(X_train)
X_test_pca  = pca_128.transform(X_test)

kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
kmeans.fit(X_train_pca)
print(f"  K-Means: {n_classes} clusters — inertia: {kmeans.inertia_:.2f}")

# ── FAR/FRR ───────────────────────────────────────────────────────────────────
def compute_far_frr(y_true, y_pred):
    FA, total_impostors = 0, 0
    for true, pred in zip(y_true, y_pred):
        if true != pred:
            total_impostors += 1
            FA += 1
    FAR = FA / max(total_impostors, 1)
    FRR = (len(y_true) - sum(y_true == y_pred)) / max(len(y_true), 1)
    return FAR, FRR

# ── 9 MODEL DEFINITIONS (8 base + 1 stacking) ─────────────────────────────────
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Base estimators for stacking
stack_estimators = [
    ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
    ("knn", KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1)),
    ("rf",  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ("mlp", MLPClassifier(hidden_layer_sizes=(256,), max_iter=100, random_state=42)),
    ("nb",  GaussianNB()),
    ("lda", LinearDiscriminantAnalysis()),
]

MODELS = {
    "SVM (RBF)": SVC(
        kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
    "KNN (k=5)": KNeighborsClassifier(
        n_neighbors=5, metric="cosine", n_jobs=-1),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1),
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", multi_class="multinomial", random_state=42),
    "Gradient Boosting": XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=4,
        tree_method="hist", random_state=42),
    "MLP Neural Net": MLPClassifier(
        hidden_layer_sizes=(512, 256), activation="relu",
        max_iter=200, early_stopping=True, random_state=42),
    "Naive Bayes": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(
        solver="svd", n_components=min(n_classes-1, 50)),
    "Stacking Ensemble": StackingClassifier(
        estimators=stack_estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=3, n_jobs=-1
    ),
}

# ── TRAIN + EVALUATE ALL 9 MODELS ─────────────────────────────────────────────
print("\n=== CO02 + CO04: Training and Evaluating 9 Models ===\n")

results = {}

for name, model in MODELS.items():
    print(f"  Training: {name}...")

    X_tr = X_train_pca if name in ("LDA", "Naive Bayes") else X_train
    X_te = X_test_pca  if name in ("LDA", "Naive Bayes") else X_test

    # Cross validation (skip for stacking to save time)
    if name != "Stacking Ensemble":
        cv_scores = cross_val_score(model, X_tr, y_train, cv=cv,
                                    scoring="accuracy", n_jobs=-1)
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    else:
        cv_mean, cv_std = 0.0, 0.0  # stacking already uses CV internally

    model.fit(X_tr, y_train)

    y_pred_train = model.predict(X_tr)
    y_pred_test  = model.predict(X_te)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test,  y_pred_test)
    f1        = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
    prec      = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
    rec       = recall_score(y_test, y_pred_test, average="weighted", zero_division=0)
    far, frr  = compute_far_frr(y_test, y_pred_test)
    gap       = train_acc - test_acc

    results[name] = {
        "train_acc": train_acc, "test_acc": test_acc,
        "cv_mean": cv_mean,     "cv_std": cv_std,
        "f1": f1, "precision": prec, "recall": rec,
        "FAR": far, "FRR": frr, "overfit_gap": gap,
        "y_pred": y_pred_test,
    }

    print(f"    Train: {train_acc:.3f} | Test: {test_acc:.3f} | "
          f"F1: {f1:.3f} | Gap: {gap:.3f}")

# ── SAVE RESULTS TO CSV + XLSX (sir's requirement) ────────────────────────────
print("\n=== Saving results to CSV and Excel ===")
model_names = list(results.keys())
results_df  = pd.DataFrame({
    "Model":       model_names,
    "Train_Acc":   [results[m]["train_acc"]   for m in model_names],
    "Test_Acc":    [results[m]["test_acc"]    for m in model_names],
    "F1_Score":    [results[m]["f1"]          for m in model_names],
    "CV_Mean":     [results[m]["cv_mean"]     for m in model_names],
    "CV_Std":      [results[m]["cv_std"]      for m in model_names],
    "FAR":         [results[m]["FAR"]         for m in model_names],
    "FRR":         [results[m]["FRR"]         for m in model_names],
    "Overfit_Gap": [results[m]["overfit_gap"] for m in model_names],
})
results_df = results_df.sort_values("Test_Acc", ascending=False)
results_df.to_csv(f"{REPORT_DIR}/model_results.csv",   index=False)
results_df.to_excel(f"{REPORT_DIR}/model_results.xlsx", index=False)
print(f"✓ Saved to {REPORT_DIR}/model_results.csv and .xlsx")

# ── SAVE MODEL RESULTS FOR WEBCAM APP ─────────────────────────────────────────
with open(f"{FEAT_DIR}/model_results.pkl", "wb") as f:
    pickle.dump(results, f)

# ── CO04: COMPARISON PLOTS ────────────────────────────────────────────────────
print("\n=== Generating Comparison Plots (CO04) ===")

train_accs = [results[m]["train_acc"]   for m in model_names]
test_accs  = [results[m]["test_acc"]    for m in model_names]
f1_scores_ = [results[m]["f1"]         for m in model_names]
cv_means   = [results[m]["cv_mean"]    for m in model_names]
cv_stds    = [results[m]["cv_std"]     for m in model_names]
far_vals   = [results[m]["FAR"]        for m in model_names]
frr_vals   = [results[m]["FRR"]        for m in model_names]
gaps       = [results[m]["overfit_gap"] for m in model_names]

short_names = [n.replace(" ", "\n") for n in model_names]
colors_bar  = ["#2196F3","#4CAF50","#FF9800","#E91E63",
               "#9C27B0","#00BCD4","#FF5722","#607D8B","#F44336"]

fig = plt.figure(figsize=(22, 14))
fig.suptitle("Facial Authentication: 9-Model Comparative Analysis\n"
             "(VGGFace2 Dataset, 512-d embeddings, Stacking Ensemble included)",
             fontsize=13, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, hspace=0.55, wspace=0.4)

# 1. Train vs Test
ax1 = fig.add_subplot(gs[0, :2])
x = np.arange(len(model_names)); w = 0.35
ax1.bar(x - w/2, train_accs, w, label="Train", color="#2196F3", alpha=0.85)
ax1.bar(x + w/2, test_accs,  w, label="Test",  color="#4CAF50", alpha=0.85)
ax1.set_xticks(x); ax1.set_xticklabels(short_names, fontsize=7)
ax1.set_ylabel("Accuracy"); ax1.set_title("Train vs Test Accuracy (CO02)")
ax1.legend(); ax1.set_ylim(0, 1.15)
for i, (tr, te) in enumerate(zip(train_accs, test_accs)):
    ax1.text(i-w/2, tr+0.01, f"{tr:.2f}", ha="center", fontsize=6)
    ax1.text(i+w/2, te+0.01, f"{te:.2f}", ha="center", fontsize=6)

# 2. CV scores
ax2 = fig.add_subplot(gs[0, 2])
ax2.barh(range(len(model_names)), cv_means, xerr=cv_stds,
         color=colors_bar, alpha=0.85, capsize=4)
ax2.set_yticks(range(len(model_names)))
ax2.set_yticklabels(model_names, fontsize=7)
ax2.set_xlabel("CV Accuracy (3-fold)")
ax2.set_title("Cross-Validation (CO04)")
ax2.set_xlim(0, 1.0)

# 3. F1 Score
ax3 = fig.add_subplot(gs[1, 0])
bars = ax3.bar(range(len(model_names)), f1_scores_, color=colors_bar, alpha=0.85)
ax3.set_xticks(range(len(model_names)))
ax3.set_xticklabels(short_names, fontsize=7)
ax3.set_ylabel("F1 Score"); ax3.set_title("Weighted F1 Score (CO04)")
ax3.set_ylim(0, 1.1)
for bar, val in zip(bars, f1_scores_):
    ax3.text(bar.get_x()+bar.get_width()/2, val+0.01,
             f"{val:.2f}", ha="center", fontsize=7)

# 4. FAR vs FRR
ax4 = fig.add_subplot(gs[1, 1])
x = np.arange(len(model_names)); w = 0.35
ax4.bar(x-w/2, far_vals, w, label="FAR", color="#E91E63", alpha=0.85)
ax4.bar(x+w/2, frr_vals, w, label="FRR", color="#FF9800", alpha=0.85)
ax4.set_xticks(x); ax4.set_xticklabels(short_names, fontsize=7)
ax4.set_ylabel("Error Rate"); ax4.set_title("FAR vs FRR (CO04)")
ax4.legend()

# 5. Overfitting Gap
ax5 = fig.add_subplot(gs[1, 2])
gap_colors = ["#E53935" if g > 0.1 else "#43A047" for g in gaps]
ax5.bar(range(len(model_names)), gaps, color=gap_colors, alpha=0.85)
ax5.axhline(0.1, color="red", linestyle="--", linewidth=1, label="10% threshold")
ax5.set_xticks(range(len(model_names)))
ax5.set_xticklabels(short_names, fontsize=7)
ax5.set_ylabel("Train - Test Acc")
ax5.set_title("Overfitting Gap (CO04)")
ax5.legend()

# 6. Confusion matrix — best model
best_name = max(results, key=lambda m: results[m]["test_acc"])
best_pred = results[best_name]["y_pred"]
cm        = confusion_matrix(y_test, best_pred)
ax6       = fig.add_subplot(gs[2, :2])
n_show    = min(20, cm.shape[0])
sns.heatmap(cm[:n_show, :n_show], ax=ax6, cmap="Blues",
            annot=n_show<=15, fmt="d", linewidths=0.5)
ax6.set_title(f"Confusion Matrix — Best Model: {best_name} (CO04)")
ax6.set_xlabel("Predicted"); ax6.set_ylabel("True")

# 7. PCA variance (CO03)
ax7 = fig.add_subplot(gs[2, 2])
ax7.plot(range(1, min(101, len(cumvar)+1)), cumvar[:100], "b-", linewidth=2)
ax7.axhline(0.95, color="r", linestyle="--", label="95% threshold")
ax7.axvline(n_95pct, color="g", linestyle="--", label=f"{n_95pct} components")
ax7.set_xlabel("# Components"); ax7.set_ylabel("Cumulative Variance")
ax7.set_title("PCA Variance Analysis (CO03)")
ax7.legend(fontsize=8)

plt.savefig(f"{REPORT_DIR}/model_comparison.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved to {REPORT_DIR}/model_comparison.png")

# ── PRINT SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "="*85)
print("FINAL RESULTS SUMMARY — CO04 Error Analysis")
print("="*85)
print(f"{'Model':<22} {'Train':>7} {'Test':>7} {'F1':>7} {'CV':>10} {'FAR':>7} {'FRR':>7} {'Gap':>7}")
print("-"*85)
for name in sorted(model_names, key=lambda m: results[m]["test_acc"], reverse=True):
    r = results[name]
    print(f"{name:<22} {r['train_acc']:>7.3f} {r['test_acc']:>7.3f} "
          f"{r['f1']:>7.3f} {r['cv_mean']:>6.3f}±{r['cv_std']:.2f} "
          f"{r['FAR']:>7.4f} {r['FRR']:>7.4f} {r['overfit_gap']:>7.3f}")

best = max(results, key=lambda m: results[m]["test_acc"])
print(f"\n★  Best Model: {best}")
print(f"   Test Accuracy: {results[best]['test_acc']:.4f}")
print(f"   F1 Score:      {results[best]['f1']:.4f}")
print(f"\nNext: run 04_webcam_app.py")
plt.show()
