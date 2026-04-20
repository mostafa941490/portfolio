import streamlit as st
import numpy as np
import pandas as pd
from model import predict_risk, train_model

st.set_page_config(page_title="Neural-Med V4", layout="wide")

st.title("🧠 NEURAL-MED V4 - AI Diabetes Diagnostic System")

# تحميل / تدريب الموديل أول مرة
@st.cache_resource
def load_model():
    return train_model()

model = load_model()

menu = st.sidebar.radio("Navigation", ["🔬 Predict", "📊 History (Demo)"])

# ===================== PREDICT =====================
if menu == "🔬 Predict":

    st.subheader("Patient Data Input")

    col1, col2 = st.columns(2)

    with col1:
        glucose = st.number_input("Glucose", 0, 300, 120)
        bp = st.number_input("Blood Pressure", 0, 200, 80)
        bmi = st.number_input("BMI", 0.0, 60.0, 25.0)

    with col2:
        age = st.number_input("Age", 0, 120, 30)
        insulin = st.number_input("Insulin", 0, 900, 80)

    if st.button("🧠 Analyze AI Risk"):

        features = np.array([glucose, bp, bmi, age, insulin])

        result, score, explanation = predict_risk(model, features)

        st.markdown("---")

        if result == 1:
            st.error(f"⚠ HIGH RISK ({score:.1f}%)")
        else:
            st.success(f"✅ LOW RISK ({score:.1f}%)")

        st.subheader("🧠 AI Explanation")

        for k, v in explanation.items():
            st.write(f"- **{k}** → {v}")

        st.progress(int(score))


# ===================== HISTORY (DEMO) =====================
if menu == "📊 History (Demo)":

    st.subheader("📊 Patient History (Demo Data)")

    df = pd.DataFrame({
        "Glucose": [120, 150, 90, 200],
        "BMI": [25, 30, 22, 35],
        "Risk": ["Low", "High", "Low", "High"]
    })

    st.dataframe(df)
