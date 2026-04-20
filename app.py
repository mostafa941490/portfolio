import streamlit as st
import numpy as np
import pandas as pd
import time
from model import predict, load_model 

# 1. إعدادات الصفحة المتقدمة
st.set_page_config(
    page_title="Neural-Med V4 | AI Deep Diagnostic",
    page_icon="🧬",
    layout="wide"
)

# 2. حقن CSS احترافي (Dark Theme & Glassmorphism)
st.markdown("""
    <style>
    /* تحويل الخلفية للون الداكن العميق */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    
    /* تنسيق السايد بار */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* تأثير الزجاج على الحاويات (Cards) */
    div.stVerticalBlock > div > div.element-container {
        /* background: rgba(255, 255, 255, 0.03); */
        /* border-radius: 15px; */
    }
    
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* تنسيق المدخلات */
    .stNumberInput input {
        background-color: #0f172a !important;
        color: #00d1b2 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }

    /* زرار التحليل - Neon Style */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00d1b2 0%, #0097a7 100%);
        color: white;
        border: none;
        padding: 15px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 20px;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(0, 209, 178, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 209, 178, 0.5);
        border: none;
        color: white;
    }

    /* تنسيق التبويبات (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px 10px 0 0;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 209, 178, 0.2) !important;
        color: #00d1b2 !important;
        border-bottom: 2px solid #00d1b2 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. دالة تحميل الموديل
@st.cache_resource
def get_model_and_scaler():
    return load_model()

model, scaler = get_model_and_scaler()

# 4. الهيدر الاحترافي
col_logo, col_text = st.columns([1, 6])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/3843/3843187.png", width=90)
with col_text:
    st.markdown("<h1 style='text-align: left; margin-top: 10px; color: #00d1b2;'>NEURAL-MED V4</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Advanced AI-Driven Diabetes Prediction & Analytics System</p>", unsafe_allow_html=True)

st.write("---")

# 5. التبويبات
tab1, tab2, tab3 = st.tabs(["🚀 AI Diagnostic Hub", "📈 Insights & Analytics", "⚙️ System Engine"])

# ----------------- Tab 1: Diagnostic Hub -----------------
with tab1:
    st.markdown("### 🧬 Patient Bio-Metric Input")
    
    # تقسيم المدخلات لحاويات منظمة
    with st.form("diag_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🩸 Metabolic Stats")
            glucose = st.number_input("Glucose Level", 0, 300, 120, help="التركيز البلازمي للجلوكوز")
            insulin = st.number_input("Insulin Level", 0, 900, 80)
            preg = st.number_input("Pregnancies", 0, 20, 0)
            bp = st.number_input("Blood Pressure", 0, 200, 80)

        with c2:
            st.markdown("##### 📏 Physical Stats")
            bmi = st.number_input("BMI Index", 0.0, 70.0, 25.0)
            age = st.number_input("Patient Age", 0, 120, 30)
            dpf = st.number_input("Pedigree Function", 0.0, 3.0, 0.5, help="عامل الوراثة التاريخي")
            skin = st.number_input("Skin Thickness", 0, 100, 20)
            
        submit = st.form_submit_button("RUN AI DIAGNOSIS")

    if submit:
        features = np.array([preg, glucose, bp, skin, insulin, bmi, dpf, age])
        
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
            
        try:
            result, score, explanation = predict(model, scaler, features)

            st.markdown("<br>", unsafe_allow_html=True)
            
            # عرض النتائج في Card زجاجي
            res_box = st.container()
            with res_box:
                m1, m2, m3 = st.columns(3)
                if result == 1:
                    m1.error("RESULT: POSITIVE")
                    color = "#ff4b4b"
                else:
                    m1.success("RESULT: NEGATIVE")
                    color = "#00d1b2"
                
                m2.metric("Confidence Score", f"{score:.1f}%")
                m3.metric("AI Status", "Analyzing..." if score < 60 else "Stable")

                st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; border-left: 5px solid {color};">
                        <h4>AI Reasoning:</h4>
                        <p>The model detected <b>{'High' if result==1 else 'Low'}</b> risk based on the provided metrics.</p>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("##### 🔍 Factor Contribution")
                cols = st.columns(len(explanation))
                for i, (k, v) in enumerate(explanation.items()):
                    cols[i].write(f"**{k}**")
                    cols[i].code(v)

        except Exception as e:
            st.error(f"Execution Error: {e}")

# ----------------- Tab 2: Analytics -----------------
with tab2:
    st.markdown("### 📊 Distribution Trends")
    # توليد بيانات وهمية شكلها شيك للـ Analytics
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['Model Accuracy', 'Data Stability', 'Risk Variance']
    )
    st.line_chart(chart_data)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.bar_chart(np.random.randint(1, 10, size=(5, 1)))
    with col_b:
        st.markdown("""
        **System Logs:**
        - Model Version: V4.2.1-Stable
        - Engine: VotingClassifier(RF, GB, LR)
        - Last Sync: Today
        """)

# ----------------- Tab 3: System -----------------
with tab3:
    st.markdown("### ⚙️ Engine Specifications")
    st.json({
        "University": "New Mansoura University",
        "Faculty": "AI Engineering",
        "Developer": "Mostafa Khaled",
        "Models": ["RandomForest", "GradientBoosting", "LogisticRegression"],
        "Backend": "Python 3.10 / Streamlit Cloud"
    })
    st.markdown("---")
    st.button("Reset System Cache", on_click=st.cache_resource.clear)

# 6. السايد بار كـ Dashboard Controller
with st.sidebar:
    st.markdown("### 🛠️ Quick Settings")
    st.toggle("Auto-Refresh Data", value=True)
    st.select_slider("Risk Sensitivity", options=["Low", "Medium", "High"])
    st.divider()
    st.caption("© 2026 Neural-Med Systems. All rights reserved.")
    st.subheader("📊 Patient History (Demo Data)")
    df = pd.DataFrame({
        "Glucose": [120, 150, 90, 200],
        "BMI": [25, 30, 22, 35],
        "Risk": ["Low", "High", "Low", "High"]
    })
    st.dataframe(df, use_container_width=True)
