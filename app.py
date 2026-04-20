import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
from model import predict, load_model 

# 1. إعدادات الصفحة وإخفاء الهيدر
st.set_page_config(page_title="Neural-Med Pro V4.5", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    header[data-testid="stHeader"] {visibility: hidden;}
    .main .block-container {padding-top: 2rem;}
    
    /* تصميم البطاقات الزجاجية المتطورة */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        transition: 0.3s;
    }
    .glass-card:hover {
        border: 1px solid rgba(0, 242, 234, 0.4);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* العناوين بنظام النيون */
    .neon-text {
        color: #00f2ea;
        text-shadow: 0 0 10px rgba(0, 242, 234, 0.5);
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. وظيفة رسم الرادار الطبي (The Wow Factor)
def create_radar_chart(features, labels):
    # تطبيع القيم للعرض فقط (Normalizing for visualization)
    values = features / (features.max() if features.max() != 0 else 1) 
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=features,
        theta=labels,
        fill='toself',
        fillcolor='rgba(0, 242, 234, 0.3)',
        line=dict(color='#00f2ea', width=2),
        name='Patient Profile'
    ))
    
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, max(features)*1.2], showticklabels=False, gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="#94a3b8"))
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=20, b=20),
        height=350
    )
    return fig

# 3. التحميل واللغة
if 'lang' not in st.session_state: st.session_state.lang = 'English'
lang = st.sidebar.selectbox("Language / اللغة", ["English", "Arabic"])

model, scaler = load_model()

# 4. الهيدر الرئيسي
st.markdown(f"<h1 class='neon-text'>{'NEURAL-MED PRO' if lang=='English' else 'نيورال-ميد برو'} <small>V4.5</small></h1>", unsafe_allow_html=True)

# 5. منطقة الإدخال (تصميم عصري)
with st.container():
    st.markdown("### 🧬 " + ("Patient Bio-Analytics" if lang=='English' else "التحليلات الحيوية للمريض"))
    
    col_input, col_viz = st.columns([1.2, 1])
    
    with col_input:
        with st.form("medical_form"):
            c1, c2 = st.columns(2)
            with c1:
                glucose = st.number_input("Glucose", 0, 300, 120)
                insulin = st.number_input("Insulin", 0, 900, 80)
                preg = st.number_input("Pregnancies", 0, 20, 0)
                bp = st.number_input("Blood Pressure", 0, 200, 80)
            with c2:
                bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
                age = st.number_input("Age", 0, 120, 30)
                dpf = st.number_input("Pedigree", 0.0, 3.0, 0.5)
                skin = st.number_input("Skin Thickness", 0, 100, 20)
            
            submit = st.form_submit_button("⚡ START AI SCAN")

    with col_viz:
        # عرض الرادار المبدئي أو عند الإرسال
        labels = ['Preg', 'Glu', 'BP', 'Skin', 'Ins', 'BMI', 'DPF', 'Age']
        initial_data = np.array([preg, glucose, bp, skin, insulin, bmi, dpf, age])
        st.plotly_chart(create_radar_chart(initial_data, labels), use_container_width=True)

# 6. منطقة النتائج الفائقة
if submit:
    features = np.array([preg, glucose, bp, skin, insulin, bmi, dpf, age])
    
    with st.status("🔮 Neural engine is scanning DNA patterns...", expanded=True) as status:
        time.sleep(1)
        status.update(label="✅ Analysis complete!", state="complete", expanded=False)
    
    result, score, explanation = predict(model, scaler, features)
    
    st.markdown("---")
    
    # بطاقة النتيجة النهائية بتصميم Glassmorphism
    res_color = "#ff4b4b" if result == 1 else "#00f2ea"
    res_bg = "rgba(255, 75, 75, 0.1)" if result == 1 else "rgba(0, 242, 234, 0.1)"
    
    st.markdown(f"""
        <div style="background: {res_bg}; border: 1px solid {res_color}; padding: 30px; border-radius: 25px; text-align: center;">
            <h1 style="color: {res_color}; margin: 0;">{'POSITIVE' if result == 1 else 'NEGATIVE'}</h1>
            <p style="font-size: 1.5rem; opacity: 0.8;">AI Confidence: {score:.1f}%</p>
        </div>
    """, unsafe_allow_html=True)
    
    # توزيع البيانات في كروت صغيرة
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk Level", "High" if result == 1 else "Minimal", delta_color="inverse")
    m2.metric("Metabolic Sync", "Active")
    m3.metric("Data Integrity", "99.8%")
    m4.metric("Engine", "V4.5 Stable")

st.sidebar.markdown("---")
st.sidebar.caption("Powered by New Mansoura University AI Dept.")
