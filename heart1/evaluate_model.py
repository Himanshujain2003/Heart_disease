import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

DATA_PATH = "heart.csv"
if not os.path.exists(DATA_PATH):
    raise SystemExit(f"Data file not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
if 'target' not in df.columns:
    raise SystemExit("No 'target' column in CSV")

X = df.drop(columns=['target'])
y = df['target']

# use the same split as train.py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preds = model.predict(X_test)
probas = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else None

print("Evaluation on test set:")
print("Accuracy:", accuracy_score(y_test, preds))
print("Precision:", precision_score(y_test, preds))
print("Recall:", recall_score(y_test, preds))
print("F1:", f1_score(y_test, preds))
if probas is not None:
    print("ROC AUC:", roc_auc_score(y_test, probas))

print("\nConfusion matrix:\n", confusion_matrix(y_test, preds))
print("\nClassification report:\n", classification_report(y_test, preds))

print("\nDone. This script DOES NOT retrain the model; it only loads the saved model and evaluates it on the test split.")
