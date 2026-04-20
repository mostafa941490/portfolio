# ===================== PREDICT =====================
if menu == "🔬 Predict":

    st.subheader("Patient Data Input")

    col1, col2 = st.columns(2)

    with col1:
        # لازم الترتيب هنا يكون نفس ترتيب الأعمدة في ملف diabetes.csv
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
        
        # تجميع الـ 8 بيانات بالترتيب الصحيح
        features = np.array([pregnancies, glucose, bp, skinthickness, insulin, bmi, dpf, age])

        try:
            result, score, explanation = predict(model, scaler, features)
            # باقي كود عرض النتائج...
