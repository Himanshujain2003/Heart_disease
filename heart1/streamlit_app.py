# streamlit_app.py
"""
Streamlit app to load best_model.pkl and make predictions for new inputs.
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_model.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("💓 Heart Disease Prediction")

if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False

@st.cache_resource
def load_and_cache_model(model_path):
    """Loads the model from disk and caches the result."""
    if not os.path.exists(model_path):
        return None

    with st.spinner(f"Loading best model from {model_path}..."):
        return joblib.load(model_path)

model = load_and_cache_model(MODEL_PATH)

def train_demo_model(data_path="heart.csv", model_path=MODEL_PATH):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.compose import ColumnTransformer

    if not os.path.exists(data_path):
        st.error(f"Data file '{data_path}' not found. Can't train demo model.")
        return None

    if os.path.exists(model_path):
        st.info(f"Model already exists at '{model_path}', skipping training.")
        try:
            return joblib.load(model_path)
        except Exception:
            pass

    df = pd.read_csv(data_path)
    if 'target' not in df.columns:
        st.error("'target' column not found in CSV. Can't train demo model.")
        return None

    X = df.drop(columns=['target'])
    y = df['target']

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000))])

    with st.spinner("Training demo model (this runs only when you click)..."):
        pipeline.fit(X, y)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(pipeline, model_path)

    st.success(f"Demo model trained and saved to '{model_path}'")
    st.session_state['model_trained'] = True
    return pipeline

if model is None:
    st.warning(f"Model file '{MODEL_PATH}' not found.")
    if not st.session_state.get('model_trained', False):
        if st.button("Train demo model now (quick)"):
            trained = train_demo_model()
            if trained is not None:
                load_and_cache_model.clear()
                st.experimental_rerun()
    else:
        st.info("Demo model was trained during this session. Reload the app to use it or run 'train.py' to create a persistent tuned model.")
        st.stop()

st.header("🧍 Enter Patient Details")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=57)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    cp = st.slider("Chest pain type (0–3)", 0, 3, 1)
    trestbps = st.number_input("Resting BP (mm Hg)", min_value=80, max_value=200, value=130)
with col2:
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=400, value=246)
    fbs = st.selectbox("Fasting blood sugar >120 mg/dl (1=Yes,0=No)", [1, 0])
    restecg = st.slider("Resting ECG Results (0–2)", 0, 2, 1)
    thalach = st.number_input("Max heart rate achieved", min_value=60, max_value=220, value=150)
with col3:
    exang = st.selectbox("Exercise induced angina (1=Yes,0=No)", [1, 0])
    oldpeak = st.number_input("ST depression", min_value=0.0, max_value=10.0, value=1.0, format="%.2f")
    slope = st.slider("Slope of ST segment (0–2)", 0, 2, 1)
    ca = st.slider("No. of major vessels (0–3)", 0, 3, 0)
    thal = st.slider("Thal (1=Normal,2=Fixed defect,3=Reversible defect)", 1, 3, 2)

# Prepare DataFrame
input_df = pd.DataFrame([{
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
    "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
    "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
}])

st.subheader("📋 Input Data")
st.dataframe(input_df, height=35, use_container_width=True, hide_index=True)


st.markdown("---")
st.write("") 
st.write("") 

if st.button("🔮 Predict"):
    
    with st.spinner("Making prediction..."):
        pred_numeric = int(model.predict(input_df)[0])
    
    st.subheader("🩺 Prediction Result")
    
    if pred_numeric == 1:
        st.error(f"Prediction: **YES** (Heart Disease Predicted)")
    else:
        st.success(f"Prediction: **NO** (No Heart Disease Predicted)")
        
   

st.caption("Disclaimer: This is a demo for educational purposes and should not be used for actual medical diagnosis.")