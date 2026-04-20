import streamlit as st
import numpy as np
import pandas as pd
# 1. تعديل الاستدعاء ليطابق الأسماء الجديدة في model.py
from model import predict, load_model 

st.set_page_config(page_title="Neural-Med V4", layout="wide")

st.title("🧠 NEURAL-MED V4 - AI Diabetes Diagnostic System")

# 2. تعديل الدالة لتخزين الموديل والـ scaler معاً
@st.cache_resource
def get_model_and_scaler():
    return load_model()

# استلام الموديل والـ scaler
model, scaler = get_model_and_scaler()

menu = st.sidebar.radio("Navigation", ["🔬 Predict", "📊 History (Demo)"])

# ===================== PREDICT =====================
if menu == "🔬 Predict":

    st.subheader("Patient Data Input")

    col1, col2 = st.columns(2)

    with col1:
        # ملاحظة: الترتيب هنا لازم يطابق ترتيب الأعمدة في ملف الـ CSV
        glucose = st.number_input("Glucose", 0, 300, 120)
        bp = st.number_input("Blood Pressure", 0, 200, 80)
        bmi = st.number_input("BMI", 0.0, 60.0, 25.0)

    with col2:
        age = st.number_input("Age", 0, 120, 30)
        insulin = st.number_input("Insulin", 0, 900, 80)

    if st.button("🧠 Analyze AI Risk"):
        
        # 3. تجميع البيانات (تأكد أن عددها مطابق لما يتوقعه الموديل)
        # إذا كان الموديل مدرب على 8 أعمدة، يجب إضافة القيم الناقصة هنا
        features = np.array([glucose, bp, bmi, age, insulin])

        # 4. استدعاء الدالة بالاسم الجديد predict وإرسال الـ scaler معها
        try:
            result, score, explanation = predict(model, scaler, features)

            st.markdown("---")

            if result == 1:
                st.error(f"⚠ HIGH RISK ({score:.1f}%)")
            else:
                st.success(f"✅ LOW RISK ({score:.1f}%)")

            st.subheader("🧠 AI Explanation")

            for k, v in explanation.items():
                st.write(f"- **{k}** → {v}")

            st.progress(int(score))
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("تأكد أن عدد المدخلات (5) يطابق عدد الأعمدة في ملف diabetes.csv")


# ===================== HISTORY (DEMO) =====================
if menu == "📊 History (Demo)":
    st.subheader("📊 Patient History (Demo Data)")
    df = pd.DataFrame({
        "Glucose": [120, 150, 90, 200],
        "BMI": [25, 30, 22, 35],
        "Risk": ["Low", "High", "Low", "High"]
    })
    st.dataframe(df)
