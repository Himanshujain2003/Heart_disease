import joblib
import pandas as pd
import os

model_path = os.path.join('artifacts', 'best_model.pkl')

if not os.path.exists(model_path):
    raise SystemExit(f'Model not found: {model_path}')

model = joblib.load(model_path)

sample_data = {
    'age': [57],
    'sex': [1],
    'cp': [2],
    'trestbps': [140],
    'chol': [240],
    'fbs': [0],
    'restecg': [1],
    'thalach': [150],
    'exang': [0],
    'oldpeak': [1.0],
    'slope': [2],
    'ca': [0],
    'thal': [2]
}

sample = pd.DataFrame(sample_data)

pred = model.predict(sample)
proba = model.predict_proba(sample)[:, 1] if hasattr(model, 'predict_proba') else None

print('Prediction:', int(pred[0]))
if proba is not None:
    print('Probability:', float(proba[0]))
