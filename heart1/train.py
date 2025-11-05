"""
Train and tune models on heart.csv and save the best tuned model and preprocessor.
Place heart.csv in the same folder as this script.
Outputs:
 - best_model.pkl
 - preprocessor.pkl
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

DATA_PATH = "heart.csv"   
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nStats:\n", df.describe().T)

plt.figure(figsize=(6,4))
plt.hist(df['age'], bins=12)
plt.title("Age distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
plt.bar(df['target'].value_counts().index.astype(str), df['target'].value_counts().values)
plt.title("Target distribution (0 = no disease, 1 = disease)")
plt.show()

X = df.drop(columns=['target'])
y = df['target']

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


pipelines = {
    'LogisticRegression': Pipeline(steps=[('preprocessor', preprocessor),
                                         ('classifier', LogisticRegression(max_iter=1000, random_state=42))]),
    'RandomForest': Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', RandomForestClassifier(random_state=42))]),
    'GradientBoosting': Pipeline(steps=[('preprocessor', preprocessor),
                                        ('classifier', GradientBoostingClassifier(random_state=42))])
}

results = {}
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probas = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, "predict_proba") else None
    results[name] = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall': recall_score(y_test, preds),
        'f1': f1_score(y_test, preds),
        'roc_auc': roc_auc_score(y_test, probas) if probas is not None else None,
        'model': pipe
    }

print("\nBaseline results:")
for name, r in results.items():
    print(f"{name}: Accuracy={r['accuracy']:.3f}, Precision={r['precision']:.3f}, Recall={r['recall']:.3f}, F1={r['f1']:.3f}, ROC_AUC={r['roc_auc']}")

for name, r in results.items():
    print(f"\n--- {name} Classification Report ---")
    print(classification_report(y_test, r['model'].predict(X_test)))


best_name = max(results.items(), key=lambda kv: kv[1]['f1'])[0]
print("\nSelected best model (by F1):", best_name)
best_pipeline = results[best_name]['model']

if best_name == 'RandomForest':
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 5, 10],
        'classifier__min_samples_split': [2, 5]
    }
elif best_name == 'GradientBoosting':
    param_grid = {
        'classifier__n_estimators': [50, 100, 150],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__max_depth': [3, 5]
    }
else: 
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10]
    }

grid = GridSearchCV(best_pipeline, param_grid, cv=4, scoring='f1', n_jobs=-1, verbose=1)
print("Running GridSearchCV (this can take a few minutes)...")
grid.fit(X_train, y_train)

print("GridSearch best params:", grid.best_params_)
print("GridSearch best CV f1:", grid.best_score_)

best_tuned = grid.best_estimator_
preds_tuned = best_tuned.predict(X_test)
probas_tuned = best_tuned.predict_proba(X_test)[:,1] if hasattr(best_tuned, "predict_proba") else None

print("\n--- Tuned model test metrics ---")
print("Accuracy:", accuracy_score(y_test, preds_tuned))
print("Precision:", precision_score(y_test, preds_tuned))
print("Recall:", recall_score(y_test, preds_tuned))
print("F1:", f1_score(y_test, preds_tuned))
if probas_tuned is not None:
    print("ROC AUC:", roc_auc_score(y_test, probas_tuned))

print("\nConfusion matrix:\n", confusion_matrix(y_test, preds_tuned))
print("\nClassification report:\n", classification_report(y_test, preds_tuned))


model_path = os.path.join(ARTIFACT_DIR, "best_model.pkl")
preprocessor_path = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")
joblib.dump(best_tuned, model_path)
try:
    joblib.dump(best_tuned.named_steps['preprocessor'], preprocessor_path)
except Exception:
    if hasattr(best_tuned, 'named_steps') and 'preprocessor' in best_tuned.named_steps:
        joblib.dump(best_tuned.named_steps['preprocessor'], preprocessor_path)

print(f"Saved tuned model to: {model_path}")
print(f"Saved preprocessor to: {preprocessor_path}")

metrics = {
    'selected_model': best_name,
    'grid_best_params': grid.best_params_,
    'grid_best_cv_f1': grid.best_score_,
    'test_accuracy': float(accuracy_score(y_test, preds_tuned)),
    'test_precision': float(precision_score(y_test, preds_tuned)),
    'test_recall': float(recall_score(y_test, preds_tuned)),
    'test_f1': float(f1_score(y_test, preds_tuned)),
}
metrics_file = os.path.join(ARTIFACT_DIR, 'metrics.json')
with open(metrics_file, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Saved metrics to: {metrics_file}")

cm = confusion_matrix(y_test, preds_tuned)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
cm_path = os.path.join(ARTIFACT_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, bbox_inches='tight')
plt.close()
print(f"Saved confusion matrix to: {cm_path}")

classif_report = classification_report(y_test, preds_tuned)
report_path = os.path.join(ARTIFACT_DIR, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write(classif_report)
print(f"Saved classification report to: {report_path}")

try:
    clf = best_tuned.named_steps.get('classifier') if hasattr(best_tuned, 'named_steps') else None
    if clf is not None and hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        fi_path = os.path.join(ARTIFACT_DIR, 'feature_importances.json')
        feature_names = numeric_features
        fi = {name: float(val) for name, val in zip(feature_names, importances)}
        with open(fi_path, 'w') as f:
            json.dump(fi, f, indent=2)
        print(f"Saved feature importances to: {fi_path}")
except Exception:
    pass

