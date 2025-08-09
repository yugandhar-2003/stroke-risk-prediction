# heart_disease_model.py
# End-to-end: load -> preprocess -> train XGBoost -> calibrate -> evaluate -> SHAP

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix, precision_recall_fscore_support
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb

# ----------------- Load dataset -----------------
csv_path = "heart.csv"  # Ensure this is in the same folder
df = pd.read_csv(csv_path)
print("Shape:", df.shape)
print(df.head())

TARGET_COL = "target"  # Change if your CSV uses a different target name
FEATURES = [c for c in df.columns if c != TARGET_COL]

X = df[FEATURES]
y = df[TARGET_COL]
print("Positive class ratio:", y.mean())

# ----------------- Identify numeric/categorical -----------------
num_feats = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_feats = [c for c in X.columns if c not in num_feats]

# Optional: treat 'sex' as categorical if in numeric list
if "sex" in num_feats:
    num_feats.remove("sex")
    cat_feats.append("sex")

print("Numeric features:", num_feats)
print("Categorical features:", cat_feats)

# ----------------- Train/test split -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------- Preprocessing -----------------
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

if cat_feats:
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    preproc = ColumnTransformer([
        ("num", num_pipeline, num_feats),
        ("cat", cat_pipeline, cat_feats)
    ])
else:
    preproc = ColumnTransformer([("num", num_pipeline, num_feats)])

# ----------------- Model -----------------
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
pipe = Pipeline([
    ("preproc", preproc),
    ("clf", xgb_clf)
])

# Grid search for good params
param_grid = {
    "clf__n_estimators": [100, 250],
    "clf__max_depth": [3, 6],
    "clf__learning_rate": [0.05, 0.1]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=1)
search.fit(X_train, y_train)
best_model = search.best_estimator_
print("Best params:", search.best_params_)

# ----------------- Probability calibration -----------------
calibrator = CalibratedClassifierCV(best_model, cv=cv, method="isotonic")
calibrator.fit(X_train, y_train)

# ----------------- Evaluation -----------------
probs = calibrator.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)

auc = roc_auc_score(y_test, probs)
ap = average_precision_score(y_test, probs)
brier = brier_score_loss(y_test, probs)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary')
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

print(f"ROC AUC: {auc:.3f}")
print(f"PR AUC: {ap:.3f}")
print(f"Brier score: {brier:.3f}")
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
print("Confusion matrix: TP, FP, TN, FN ->", tp, fp, tn, fn)

# ----------------- Calibration curve -----------------
prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
plt.figure()
plt.plot(prob_pred, prob_true, marker='o', label='Observed')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect')
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Calibration curve")
plt.legend()
os.makedirs("output", exist_ok=True)
plt.savefig("output/calibration_curve.png", bbox_inches="tight")
plt.close()

# ----------------- SHAP explanations -----------------
final_xgb = best_model.named_steps["clf"]
X_test_pre = best_model.named_steps["preproc"].transform(X_test)

# Feature names after preprocessing
feature_names = num_feats
if cat_feats:
    ohe = best_model.named_steps["preproc"].named_transformers_["cat"].named_steps["ohe"]
    feature_names += list(ohe.get_feature_names_out(cat_feats))

explainer = shap.TreeExplainer(final_xgb)
shap_values = explainer.shap_values(X_test_pre)

# SHAP summary plot
shap.summary_plot(shap_values, X_test_pre, feature_names=feature_names, show=False)
plt.title("SHAP summary")
plt.savefig("output/shap_summary.png", bbox_inches="tight")
plt.close()

# ----------------- Save model -----------------
joblib.dump(calibrator, "output/calibrated_xgb_pipeline.joblib")
print("Saved model + plots in 'output/' folder")
