import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ── load ────────────────────────────────────────────────────────────────────────
hd_path = r"C:\\Visualization of Data Analytics\\Heart_disease_cleveland_new.csv"         # ← update if needed
hd = pd.read_csv(hd_path)

# ── 1. missing-value imputation (mode) ─────────────────────────────────────────
for col in ("ca", "thal"):
    if hd[col].isna().any():
        hd[col] = hd[col].fillna(hd[col].mode(dropna=True)[0])

# ── 2. outlier mitigation: clip at 1st / 99th percentiles ──────────────────────
cont = ["age", "trestbps", "chol", "thalach", "oldpeak"]
hd[cont] = hd[cont].clip(hd[cont].quantile(0.01),
                         hd[cont].quantile(0.99), axis=1)

# ── 3. scaling (z-score) ───────────────────────────────────────────────────────
hd[cont] = StandardScaler().fit_transform(hd[cont])

# ── 4. categorical one-hot encoding (drop_first to avoid dummy-trap) ───────────
cats = ["cp", "restecg", "slope", "sex", "fbs", "exang"]
hd = pd.get_dummies(hd, columns=cats, drop_first=True)

# ── 5. train / test split (80 : 20, stratified) ────────────────────────────────
X = hd.drop("target", axis=1)
y = hd["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print("Clean heart-disease shapes  –  train:", X_train.shape,
      "|  test:", X_test.shape)
