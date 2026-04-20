import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
from model import predict, load_model 

# 1. إعدادات المتصفح وإخفاء زوائد ستريمليت
st.set_page_config(page_title="Neural-Med V4.5 Pro", page_icon="🧬", layout="wide")

# 2. إدارة اللغات (Bilingual Logic)
if 'lang' not in st.session_state: st.session_state.lang = 'Arabic'

translations = {
    'English': {
        'dir': 'ltr',
        'title': 'NEURAL-MED V4.5 PRO',
        'sub': 'Advanced AI Diabetes Intelligence',
        'dev': 'Developed by: Mustafa Khaled',
        'lang_text': 'Language / اللغة',
        'group_m': 'Metabolic Stats',
        'group_p': 'Physical Stats',
        'btn': 'RUN AI SCAN',
        'conf': 'AI Confidence',
        'labels': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin', 'Insulin', 'BMI', 'Pedigree', 'Age']
    },
    'Arabic': {
        'dir': 'rtl',
        'title': 'نيورال-ميد V4.5 برو',
        'sub': 'ذكاء اصطناعي متطور لتشخيص السكري',
        'dev': 'تطوير: مصطفى خالد',
        'lang_text': 'Language / اللغة',
        'group_m': 'المؤشرات الأيضية',
        'group_p': 'القياسات الفيزيائية',
        'btn': 'بدء الفحص الذكي',
        'conf': 'نسبة ثقة الذكاء الاصطناعي',
        'labels': ['مرات الحمل', 'الجلوكوز', 'ضغط الدم', 'سمك الجلد', 'الأنسولين', 'BMI', 'عامل الوراثة', 'العمر']
    }
}

T = translations[st.session_state.lang]

# 3. محاكاة الألوان والتصميم (The Exact Visual Replica)
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&family=Orbitron:wght@700&display=swap');

    /* إخفاء الهيدر الأبيض تماماً */
    header[data-testid="stHeader"] {{visibility: hidden;}}
    [data-testid="stToolbar"] {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* خلفية التطبيق (Deep Navy Black) */
    .stApp {{
        background-color: #050a12;
        color: #ffffff;
        font-family: 'Cairo', sans-serif;
    }}

    /* السايد بار (Darker Sidebar) */
    [data-testid="stSidebar"] {{
        background-color: #0b1118 !important;
        border-right: 1px solid #1e293b;
    }}

    /* اسم المطور (Mustafa Khaled Style) */
    .dev-name {{
        font-family: 'Orbitron', 'Cairo', sans-serif;
        color: #00f2ea;
        text-shadow: 0 0 12px rgba(0, 242, 234, 0.4);
        font-size: 1.2rem;
        font-weight: bold;
    }}

    /* كروت المدخلات (The Cyan Border Box) */
    [data-testid="stForm"] {{
        background-color: #0c141d !important;
        border: 1px solid #00f2ea44 !important;
        border-radius: 20px !important;
        padding: 2rem !important;
    }}

    /* حقول الإدخال */
    .stNumberInput input {{
        background-color: #0f172a !important;
        color: #00f2ea !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }}

    /* زرار الـ Scan (Gradient Cyan) */
    .stButton>button {{
        width: 100%;
        background: linear-gradient(90deg, #00f2ea 0%, #00d1b2 100%);
        color: #050a12 !important;
        font-weight: bold;
        font-size: 1.3rem;
        border-radius: 12px;
        border: none;
        padding: 0.8rem;
        box-shadow: 0 4px 15px rgba(0, 242, 234, 0.2);
    }}

    /* كروت النتيجة */
    .result-card {{
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    </style>
    """, unsafe_allow_html=True)

# 4. السايد بار (تطوير مصطفى خالد)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3843/3843187.png", width=80)
    st.markdown(f"<div class='dev-name'>{T['dev'] if st.session_state.lang == 'Arabic' else 'Developed by: Mustafa Khaled'}</div>", unsafe_allow_html=True)
    st.markdown("<small style='color:#64748b'>Personalized Workspace</small>", unsafe_allow_html=True)
    st.divider()
    
    lang_choice = st.selectbox(T['lang_text'], ["Arabic", "English"])
    if lang_choice != st.session_state.lang:
        st.session_state.lang = lang_choice
        st.rerun()

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.caption("AI Engineering | NMU | © 2026")

# 5. الهيدر (العنوان الفرعي والرئيسي)
st.markdown(f"<h1 style='font-family:Orbitron; color:#00f2ea; margin:0;'>{T['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='opacity:0.7; font-size:1.1rem;'>{T['sub']}</p>", unsafe_allow_html=True)

# تحميل الموديل
model, scaler = load_model()

# 6. توزيع المحتوى (Inputs vs Radar)
col_input, col_viz = st.columns([1.2, 1])

with col_input:
    with st.form("replica_form"):
        st.markdown(f"### 🧬 {T['labels'][1]} & {T['labels'][5]} Analyze")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<small style='color:#00f2ea'>{T['group_1']}</small>", unsafe_allow_html=True)
            glu = st.number_input(T['labels'][1], 0, 300, 128)
            ins = st.number_input(T['labels'][4], 0, 900, 75)
            preg = st.number_input(T['labels'][0], 0, 20, 2)
            bp = st.number_input(T['labels'][2], 0, 200, 85)
        with c2:
            st.markdown(f"<small style='color:#00f2ea'>{T['group_2']}</small>", unsafe_allow_html=True)
            bmi = st.number_input(T['labels'][5], 0.0, 70.0, 26.5)
            age = st.number_input(T['labels'][7], 0, 120, 32)
            dpf = st.number_input(T['labels'][6], 0.0, 3.0, 0.62)
            skin = st.number_input(T['labels'][3], 0, 100, 21)
        
        submit = st.form_submit_button(T['btn'])

with col_viz:
    st.markdown(f"#### AI Profile Analytics")
    # محاكاة الرادار (Radar Chart)
    fig = go.Figure(data=go.Scatterpolar(
        r=[preg, glu, bp, skin, ins, bmi, dpf, age],
        theta=T['labels'],
        fill='toself',
        fillcolor='rgba(0, 242, 234, 0.15)',
        line=dict(color='#00f2ea', width=3)
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=False),
            angularaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#94a3b8", size=11))
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=60, t=20, b=20),
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

# 7. قسم النتائج (The Big Result Card)
if submit:
    features = np.array([preg, glu, bp, skin, ins, bmi, dpf, age])
    with st.status("🧠 Scanning Bio-Neural Patterns...") as s:
        time.sleep(1.2)
        result, score, explanation = predict(model, scaler, features)
        s.update(label="Scan Complete", state="complete")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    res_color = "#ff0055" if result == 1 else "#00f2ea"
    res_text = T['res_pos'] if result == 1 else T['res_neg']
    
    st.markdown(f"""
        <div style="border: 2px solid {res_color}; background: {res_color}11; padding: 40px; border-radius: 25px; text-align: center;">
            <h1 style="color: {res_color}; background:none; -webkit-text-fill-color:{res_color}; margin:0;">
                {res_text}
            </h1>
            <p style="font-size: 1.5rem; margin-top: 10px;">{T['conf']}: <b>{score:.1f}%</b></p>
            <div style="width: 100%; background: #1e293b; border-radius: 10px; height: 10px; margin-top: 15px;">
                <div style="width: {score}%; background: {res_color}; height: 10px; border-radius: 10px; box-shadow: 0 0 10px {res_color};"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)
