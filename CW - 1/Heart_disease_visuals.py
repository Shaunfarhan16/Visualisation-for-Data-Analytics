# ---------- HEART-DISEASE VISUALS ----------
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 1 load + minimal preprocessing (matches pipeline in § 3)
hd = pd.read_csv("C:\Visualization of Data Analytics\Heart_disease_cleveland_new.csv")
for c in ("ca", "thal"):                       # impute mode
    hd[c].fillna(hd[c].mode()[0], inplace=True)
cont = ["age","trestbps","chol","thalach","oldpeak"]
hd[cont] = hd[cont].clip(hd[cont].quantile(.01),
                         hd[cont].quantile(.99), axis=1)
hd[cont] = StandardScaler().fit_transform(hd[cont])
hd = pd.get_dummies(hd,
                    columns=["cp","restecg","slope","sex","fbs","exang"],
                    drop_first=True)

# 2 split
X, y = hd.drop("target", axis=1), hd["target"]
Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=.2,
                                   stratify=y, random_state=42)

# 3 train models
lr = LogisticRegression(max_iter=300, solver="liblinear").fit(Xtr, ytr)
rf = RandomForestClassifier(n_estimators=300,
                            random_state=42).fit(Xtr, ytr)

# ---------- Fig 1  ROC curve ----------
probs = lr.predict_proba(Xte)[:,1]
fpr, tpr, _ = roc_curve(yte, probs)
plt.figure()
plt.plot(fpr, tpr, lw=2, label=f"LogReg  AUC = {auc(fpr,tpr):.2f}")
plt.plot([0,1],[0,1], "--", color="grey")
plt.xlabel("False-positive rate"); plt.ylabel("True-positive rate")
plt.title("ROC Curve – Logistic Regression")
plt.legend(); plt.tight_layout(); plt.savefig("Fig1_ROC.png")

# ---------- Fig 2  Confusion matrix (RF) ----------
cm = confusion_matrix(yte, rf.predict(Xte))
plt.figure(); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Fig 2  Confusion Matrix – Random Forest")
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
plt.savefig("Fig2_CM.png")

# ---------- Fig 3  Feature-importance (RF) ----------
imp = (pd.Series(rf.feature_importances_, index=X.columns)
       .nlargest(10)   # top-10 for readability
       .sort_values())
plt.figure(figsize=(6,4))
imp.plot.barh(); plt.title("Fig 3  Top-10 Feature Importances (RF)")
plt.xlabel("Mean decrease in impurity"); plt.tight_layout()
plt.savefig("Fig3_Imp.png")
plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score
probs = lr.predict_proba(Xte)[:,1]
prec, reca, _ = precision_recall_curve(yte, probs)
ap = average_precision_score(yte, probs)

plt.figure()
plt.plot(reca, prec, lw=2)
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title(f"PR Curve – Logistic Regression  (AP = {ap:.2f})")
plt.tight_layout()

from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(yte, probs, n_bins=6)
plt.figure()
plt.plot(prob_pred, prob_true, "s-")
plt.plot([0,1],[0,1], "--", color="grey")
plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
plt.title("Calibration – Logistic Regression")
plt.tight_layout()


