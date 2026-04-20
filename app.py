import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
from model import predict, load_model 

# 1. إعدادات الصفحة
st.set_page_config(page_title="Neural-Med V4.5 Pro", page_icon="🧬", layout="wide")

# 2. اللغات
if 'lang' not in st.session_state: st.session_state.lang = 'Arabic'
translations = {
    'English': {
        'dir': 'ltr', 'font': 'Inter', 'title': 'NEURAL-MED V4.5 PRO', 'dev': 'Developed by: Mustafa Khaled',
        'btn': 'START NEURAL SCAN', 'group_1': 'Metabolic Stats', 'group_2': 'Physical Stats',
        'labels': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin', 'Insulin', 'BMI', 'Pedigree', 'Age']
    },
    'Arabic': {
        'dir': 'rtl', 'font': 'Cairo', 'title': 'نيورال-ميد V4.5 برو', 'dev': 'تطوير: مصطفى خالد',
        'btn': 'إجراء فحص ذكي الآن', 'group_1': 'المؤشرات الأيضية', 'group_2': 'القياسات الفيزيائية',
        'labels': ['مرات الحمل', 'الجلوكوز', 'ضغط الدم', 'سمك الجلد', 'الأنسولين', 'BMI', 'عامل الوراثة', 'العمر']
    }
}
T = translations[st.session_state.lang]

# 3. الـ CSS الاحترافي (الزرار الأحمر "بيلمع")
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&family=Orbitron:wght@700&display=swap');

    header[data-testid="stHeader"], [data-testid="stToolbar"], footer {{visibility: hidden;}}

    .stApp {{
        background-color: #050a12;
        color: #ffffff;
        font-family: '{T['font']}', sans-serif;
        direction: {T['dir']};
    }}

    [data-testid="stSidebar"] {{ background-color: #0b1118 !important; border-right: 1px solid #1e293b; }}
    .dev-name {{ font-family: 'Orbitron', 'Cairo', sans-serif; color: #00f2ea; text-shadow: 0 0 12px #00f2ea44; font-size: 1.2rem; font-weight: bold; }}

    [data-testid="stForm"] {{
        background-color: #0c141d !important;
        border: 1px solid #1e293b !important;
        border-radius: 25px !important;
        padding: 2.5rem !important;
    }}

    /* --- كود الزرار الأحمر اللي بيلمع --- */
    @keyframes redGlow {{
        0% {{ box-shadow: 0 0 5px #ff4b4b; transform: scale(1); }}
        50% {{ box-shadow: 0 0 25px #ff0055; transform: scale(1.01); }}
        100% {{ box-shadow: 0 0 5px #ff4b4b; transform: scale(1); }}
    }}

    .stButton>button {{
        width: 100%;
        background: linear-gradient(90deg, #ff4b4b 0%, #ff0055 100%) !important;
        color: white !important;
        font-weight: bold;
        font-size: 1.4rem;
        border-radius: 15px;
        border: none;
        padding: 1.2rem;
        animation: redGlow 2s infinite ease-in-out; /* حركة اللمعان */
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        background: #ff0000 !important;
        box-shadow: 0 0 40px #ff0000 !important;
    }}
    /* ---------------------------------- */

    .stNumberInput input {{ background-color: #0f172a !important; color: #00f2ea !important; border: 1px solid #1e293b !important; border-radius: 10px !important; }}
    </style>
    """, unsafe_allow_html=True)

# 4. السايد بار
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3843/3843187.png", width=80)
    st.markdown(f"<div class='dev-name'>{T['dev']}</div>", unsafe_allow_html=True)
    st.divider()
    lang_choice = st.selectbox("Language", ["Arabic", "English"], index=0 if st.session_state.lang == 'Arabic' else 1)
    if lang_choice != st.session_state.lang:
        st.session_state.lang = lang_choice
        st.rerun()

# 5. الهيدر
st.markdown(f"<h1 style='font-family:Orbitron; color:#00f2ea; margin:0;'>{T['title']}</h1>", unsafe_allow_html=True)

model, scaler = load_model()

# 6. المحتوى
col_input, col_viz = st.columns([1.3, 1])
with col_input:
    with st.form("mustafa_red_form"):
        st.markdown(f"### 🩺 {T['group_1']} & {T['group_2']}")
        c1, c2 = st.columns(2)
        with c1:
            glu = st.number_input(T['labels'][1], 0, 300, 128)
            ins = st.number_input(T['labels'][4], 0, 900, 75)
            preg = st.number_input(T['labels'][0], 0, 20, 2)
            bp = st.number_input(T['labels'][2], 0, 200, 85)
        with c2:
            bmi = st.number_input(T['labels'][5], 0.0, 70.0, 26.5)
            age = st.number_input(T['labels'][7], 0, 120, 32)
            dpf = st.number_input(T['labels'][6], 0.0, 3.0, 0.62)
            skin = st.number_input(T['labels'][3], 0, 100, 21)
        submit = st.form_submit_button(T['btn'])

with col_viz:
    fig = go.Figure(data=go.Scatterpolar(
        r=[preg, glu, bp, skin, ins, bmi, dpf, age], theta=T['labels'],
        fill='toself', fillcolor='rgba(0, 242, 234, 0.1)', line=dict(color='#00f2ea', width=3)
    ))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=False), angularaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#64748b", size=10))),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", height=450
    )
    st.plotly_chart(fig, use_container_width=True)

# 7. النتائج
if submit:
    features = np.array([preg, glu, bp, skin, ins, bmi, dpf, age])
    with st.status("🧠 Scanning Bio-Neural Patterns...") as s:
        time.sleep(1)
        result, score, explanation = predict(model, scaler, features)
        s.update(label="Complete", state="complete")
    
    color = "#ff0055" if result == 1 else "#00f2ea"
    st.markdown(f"""
        <div style="border: 2px solid {color}; background: {color}11; padding: 30px; border-radius: 20px; text-align: center;">
            <h1 style="color:{color}; background:none; -webkit-text-fill-color:{color};">{T['res_pos'] if result == 1 else T['res_neg']}</h1>
            <h3>Confidence: {score:.1f}%</h3>
        </div>
    """, unsafe_allow_html=True)
