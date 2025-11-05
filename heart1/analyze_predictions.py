import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = os.path.join('artifacts', 'best_model.pkl')
CSV_PATH = 'heart.csv'

print('Model path ->', MODEL_PATH)
print('CSV path ->', CSV_PATH)

if not os.path.exists(MODEL_PATH):
    raise SystemExit('Model file not found')
if not os.path.exists(CSV_PATH):
    raise SystemExit('CSV file not found')

model = joblib.load(MODEL_PATH)
print('\nLoaded model type:', type(model))

# If it's a pipeline, print steps
try:
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        print('Pipeline steps:')
        for name, step in model.steps:
            print(' -', name, type(step))
except Exception:
    pass

# Read CSV and prepare X,y
df = pd.read_csv(CSV_PATH)
if 'target' not in df.columns:
    raise SystemExit("CSV doesn't have 'target' column")

X = df.drop(columns=['target'])
y = df['target']
print('\nCSV columns:', list(X.columns))
print('First row example:')
print(X.iloc[0].to_dict())

# Try predicting
print('\nPredicting on first 10 rows...')
try:
    preds = model.predict(X.iloc[:10])
    print('Preds (first 10):', preds.tolist())
    print('Actuals (first 10):', y.iloc[:10].tolist())
except Exception as e:
    print('Error predicting on dataset:', e)

# Full report
try:
    preds_all = model.predict(X)
    print('\nAccuracy on CSV:', (preds_all == y).mean())
    print('\nClassification report:')
    print(classification_report(y, preds_all))
    print('\nConfusion matrix:')
    print(confusion_matrix(y, preds_all))

    # Show up to 10 mismatches
    mismatches = df[preds_all != y]
    print(f'\nFound {len(mismatches)} mismatches. Showing up to 10:')
    if len(mismatches) > 0:
        print(mismatches.head(10).to_string(index=False))
except Exception as e:
    print('Error computing full report:', e)

# If model doesn't support predict_proba, mention it
has_proba = hasattr(model, 'predict_proba')
print('\nModel has predict_proba:', has_proba)

print('\nDone')
