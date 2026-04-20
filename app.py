import streamlit as st
import numpy as np
import pandas as pd
from model import predict, load_model 

# إعدادات الصفحة
st.set_page_config(page_title="Neural-Med V4", layout="wide")

st.title("🧠 NEURAL-MED V4 - AI Diabetes Diagnostic System")

# تحميل الموديل والـ scaler مع التخزين المؤقت
@st.cache_resource
def get_model_and_scaler():
    return load_model()

model, scaler = get_model_and_scaler()

# القائمة الجانبية
menu = st.sidebar.radio("Navigation", ["🔬 Predict", "📊 History (Demo)"])

# ===================== PREDICT =====================
if menu == "🔬 Predict":

    st.subheader("Patient Data Input")
    st.info("الرجاء إدخال كافة البيانات المطلوبة لضمان دقة التشخيص الذكي")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies (عدد مرات الحمل)", 0, 20, 0)
        glucose = st.number_input("Glucose (الجلوكوز)", 0, 300, 120)
        bp = st.number_input("Blood Pressure (ضغط الدم)", 0, 200, 80)
        skinthickness = st.number_input("Skin Thickness (سمك الجلد)", 0, 100, 20)

    with col2:
        insulin = st.number_input("Insulin (الأنسولين)", 0, 900, 80)
        bmi = st.number_input("BMI (مؤشر كتلة الجسم)", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function (عامل الوراثة)", 0.0, 3.0, 0.5)
        age = st.number_input("Age (العمر)", 0, 120, 30)

    if st.button("🧠 Analyze AI Risk"):
        
        # تجميع الـ 8 بيانات بالترتيب الصحيح للموديل
        features = np.array([pregnancies, glucose, bp, skinthickness, insulin, bmi, dpf, age])

        try:
            # استدعاء دالة التوقع
            result, score, explanation = predict(model, scaler, features)

            st.markdown("---")

            # عرض النتيجة الأساسية
            if result == 1:
                st.error(f"⚠ HIGH RISK ({score:.1f}%)")
                st.warning("النتائج تشير إلى احتمالية وجود إصابة. يرجى استشارة طبيب متخصص.")
            else:
                st.success(f"✅ LOW RISK ({score:.1f}%)")
                st.info("النتائج تشير إلى انخفاض احتمالية الإصابة.")

            # عرض التفسير الذكي
            st.subheader("🧠 AI Explanation (تحليل العوامل)")
            cols = st.columns(len(explanation))
            for i, (k, v) in enumerate(explanation.items()):
                cols[i].metric(label=k, value=v)

            st.progress(int(score))

        except Exception as e:
            st.error(f"حدث خطأ أثناء التحليل: {e}")

# ===================== HISTORY (DEMO) =====================
if menu == "📊 History (Demo)":
    st.subheader("📊 Patient History (Demo Data)")
    df = pd.DataFrame({
        "Glucose": [120, 150, 90, 200],
        "BMI": [25, 30, 22, 35],
        "Risk": ["Low", "High", "Low", "High"]
    })
    st.dataframe(df, use_container_width=True)
