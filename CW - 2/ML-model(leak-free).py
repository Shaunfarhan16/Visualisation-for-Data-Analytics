#  Airbnb London – Linear, RF, XGB, LightGBM
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

plt.rcParams.update({"figure.dpi":300, "figure.figsize":(7,4),
                     "font.family":"DejaVu Sans"})
sns.set_style("whitegrid")

# ── 1. Load data & drop price‑leak columns ───────────────────────────
df = pd.read_csv(r"C:\Visualization of Data Analytics\Airbnb_clean.csv")

y = df["price"]
X = df.drop(columns="price")

# remove any column whose name contains 'price'
leak_cols = [c for c in X.columns if "price" in c.lower()]
print("Leaking columns removed:", leak_cols)
X = X.drop(columns=leak_cols)

# keep numeric / boolean only & drop all‑NaN cols
X = (X.select_dtypes(include=["number", "bool"])
       .dropna(axis=1, how="all"))

# drop rows without target
mask = ~y.isna()
X, y = X.loc[mask], y[mask]

# ── 2. Train–test split 
X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=.20, random_state=42)

# median imputer for any residual NaNs
num_cols = X_tr.columns
prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), num_cols)],
                         remainder="passthrough")

# ── 3. Model factory
models = {
    "Linear": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=400, learning_rate=.05, max_depth=6,
        subsample=.8, colsample_bytree=.8, n_jobs=-1, random_state=42),
    "LightGBM": LGBMRegressor(
        n_estimators=400, learning_rate=.05, num_leaves=31, random_state=42)
}

metrics = {}
pred_store = {}

for name, reg in models.items():
    pipe = Pipeline([("prep", prep), ("est", reg)])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    metrics[name] = (mean_absolute_error(y_te, pred),
                     np.sqrt(mean_squared_error(y_te, pred)),
                     r2_score(y_te, pred))
    pred_store[name] = pred

# ── 4. Metrics table
met = (pd.DataFrame(metrics, index=["MAE","RMSE","R2"])
         .T.sort_values("RMSE"))
print("\nLeak‑free test metrics:")
print(met.round(2))

# RMSE bar
plt.figure(figsize=(6,4))
sns.barplot(data=met.reset_index(), x="index", y="RMSE", color="steelblue")
plt.ylabel("RMSE (£)"); plt.xlabel("Model")
plt.title("RMSE Comparison – Leak‑Free Test Set")
plt.tight_layout(); plt.savefig("rmse_bar.png", dpi=300)
plt.show()

# ── 5. Diagnostics per model (parity & residual)
def diag_plots(name, y_true, y_pred):
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    # parity
    ax[0].scatter(y_true, y_pred, alpha=.3, s=12)
    lims = [min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max())]
    ax[0].plot(lims, lims, 'r--')
    ax[0].set_title(f"{name} – Predicted vs Actual")
    ax[0].set_xlabel("Actual £"); ax[0].set_ylabel("Predicted £")
    # residuals
    sns.residplot(x=y_pred, y=y_true - y_pred, lowess=True,
                  scatter_kws={"alpha":0.3, "s":12}, ax=ax[1])
    ax[1].set_title(f"{name} – Residuals")
    ax[1].set_xlabel("Predicted £"); ax[1].set_ylabel("Residual £")
    fig.tight_layout(); fig.savefig(f"diag_{name}.png", dpi=300)
    plt.show()

for name, pred in pred_store.items():
    diag_plots(name, y_te, pred)
