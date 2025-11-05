# Heart Disease Prediction (Streamlit + ML)

## Files
- `heart.csv` (dataset) — put this in same folder
- `train.py` — trains and tunes models, saves `best_model.pkl` and `preprocessor.pkl`
- `streamlit_app.py` — Streamlit UI to predict new cases
- `requirements.txt`

## Setup
1. Create a venv (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
## Training

Run `train.py` to train baseline models, run GridSearchCV on the best model by F1, and save artifacts to an `artifacts/` folder. Artifacts include:

- `artifacts/best_model.pkl`
- `artifacts/preprocessor.pkl`
- `artifacts/metrics.json`
- `artifacts/confusion_matrix.png`
- `artifacts/classification_report.txt`

Training may take a few minutes depending on your machine.

## Running the Streamlit app

1. If you already ran `train.py`, the Streamlit app will load `artifacts/best_model.pkl`. Otherwise you can use the quick-train button in the app to build a small demo model from `heart.csv`.
2. Start the app:

```powershell
streamlit run streamlit_app.py
```

