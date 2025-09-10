import sys
import os

# Fix import paths so 'src' package is found
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import streamlit as st
import joblib
import numpy as np
from src.utils import FEATURES
from src.recommendations import get_recommendations
# Load trained model
bundle = joblib.load("models/heart_pipeline.joblib")
pipe = bundle["pipeline"]

st.title("Heart Disease Risk Prediction & Recommendations")

# Input features dynamically
st.sidebar.header("Patient Features")
inputs = {}
for feat in FEATURES:
    inputs[feat] = st.sidebar.number_input(feat, value=0.0)

# Predict button
if st.button("Predict Risk"):
    X = np.array([[inputs[f] for f in FEATURES]], dtype=float)
    proba = pipe.predict_proba(X)[0, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X)[0]
    y_pred = int(proba >= 0.5)

    st.subheader("Prediction Result")
    st.write(f"Predicted class: **{y_pred}** (1 = disease, 0 = no disease)")
    st.write(f"Risk probability: **{proba:.3f}**")

    # Get recommendations
    advice = get_recommendations(y_pred)

    st.subheader("Recommendations")
    st.write(advice["message"])

    if advice["treatments"]:
        st.markdown("**Treatments / Medications:**")
        for t in advice["treatments"]:
            st.write(f"- {t}")

    st.markdown("**Lifestyle / Cautions:**")
    for l in advice["lifestyle_cautions"]:
        st.write(f"- {l}")
