import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
from model import predict, load_model 

# 1. إعدادات الصفحة الفائقة
st.set_page_config(
    page_title="Neural-Med V4.5 Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. نظام اللغات (Translations)
if 'lang' not in st.session_state:
    st.session_state.lang = 'Arabic'

translations = {
    'English': {
        'dir': 'ltr', 'font': 'Inter', 'title': 'NEURAL-MED V4.5 PRO',
        'sub': 'Advanced AI Diabetes Intelligence', 'dev': 'Developed by: Mustafa Khaled',
        'lang_text': 'Language / اللغة', 'group_1': 'Metabolic Stats', 'group_2': 'Physical Stats',
        'btn': 'START SMART SCAN', 'conf': 'AI Confidence',
        'res_pos': 'HIGH RISK | DIABETIC', 'res_neg': 'LOW RISK | STABLE',
        'labels': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin', 'Insulin', 'BMI', 'Pedigree', 'Age']
    },
    'Arabic': {
        'dir': 'rtl', 'font': 'Cairo', 'title': 'نيورال-ميد V4.5 برو',
        'sub': 'ذكاء اصطناعي متطور لتشخيص السكري', 'dev': 'تطوير: مصطفى خالد',
        'lang_text': 'Language / اللغة', 'group_1': 'المؤشرات الأيضية', 'group_2': 'القياسات الفيزيائية',
        'btn': 'إجراء فحص ذكي الآن', 'conf': 'نسبة ثقة الذكاء الاصطناعي',
        'res_pos': 'خطر مرتفع | High Risk', 'res_neg': 'خطر منخفض | آمن',
        'labels': ['مرات الحمل', 'الجلوكوز', 'ضغط الدم', 'سمك الجلد', 'الأنسولين', 'BMI', 'عامل الوراثة', 'العمر']
    }
}

T = translations[st.session_state.lang]

# 3. الـ CSS الاحترافي (الأخضر المتوهج للمؤشرات + الأحمر النبضي للزر)
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

    /* --- البسملة المتوهجة --- */
    .basmala {{
        text-align: center;
        width: 100%;
        font-size: 2.5rem;
        color: #ffcc00;
        text-shadow: 0 0 15px #ffcc00, 0 0 30px #ffcc0066;
        padding: 20px 0;
        margin-bottom: 20px;
        border-bottom: 1px solid rgba(255, 204, 0, 0.1);
    }}

    /* السايد بار وتوهج اسمك */
    [data-testid="stSidebar"] {{ background-color: #0b1118 !important; border-right: 1.5px solid #00f2ea33; }}
    .dev-name {{ 
        font-family: 'Orbitron', 'Cairo', sans-serif; 
        color: #00f2ea; 
        text-shadow: 0 0 15px #00f2ea; 
        font-size: 1.3rem; font-weight: bold;
    }}

    /* --- الأخضر المتوهج للجلوكوز والمؤشرات --- */
    .stNumberInput label {{ 
        color: #39ff14 !important; 
        font-weight: bold !important;
        text-shadow: 0 0 10px #39ff14, 0 0 20px rgba(57, 255, 20, 0.3);
        font-size: 1.1rem !important;
        margin-bottom: 10px !important;
    }}

    /* المستطيلات الذهبية للعناوين */
    .yellow-box {{
        background: rgba(255, 204, 0, 0.03);
        border: 1px solid #ffcc00;
        box-shadow: 0 0 10px rgba(255, 204, 0, 0.2);
        padding: 10px;
        border-radius: 12px;
        margin-bottom: 25px;
        color: #ffcc00 !important;
        font-weight: bold;
        text-align: center;
        text-shadow: 0 0 8px #ffcc00;
    }}

    /* --- زر الفحص الأحمر المتوهج (نفس قوة توهج اسمك) --- */
    @keyframes redPulse {{
        0% {{ box-shadow: 0 0 10px #ff4b4b; transform: scale(1); }}
        50% {{ box-shadow: 0 0 35px #ff0000, 0 0 50px #ff000044; transform: scale(1.02); }}
        100% {{ box-shadow: 0 0 10px #ff4b4b; transform: scale(1); }}
    }}
    .stButton>button {{
        width: 100%;
        background: linear-gradient(90deg, #ff4b4b 0%, #cc0000 100%) !important;
        color: white !important;
        font-weight: bold;
        font-size: 1.6rem;
        border-radius: 15px;
        border: none;
        padding: 1.5rem;
        animation: redPulse 2s infinite ease-in-out;
        text-shadow: 0 0 10px #ffffff66;
    }}
    .stButton>button:hover {{
        background: #ff0000 !important;
        box-shadow: 0 0 60px #ff0000 !important;
    }}

    [data-testid="stForm"] {{ background-color: #0c141d !important; border: 1px solid #1e293b !important; border-radius: 25px !important; }}
    .stNumberInput input {{ background-color: #0f172a !important; color: #00f2ea !important; border: 1px solid #1e293b !important; }}
    </style>
    """, unsafe_allow_html=True)

# 4. البسملة في القمة
st.markdown("<div class='basmala'>بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ</div>", unsafe_allow_html=True)

# 5. السايد بار
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3843/3843187.png", width=80)
    st.markdown(f"<div class='dev-name'>{T['dev']}</div>", unsafe_allow_html=True)
    st.divider()
    lang_choice = st.selectbox("Language / اللغة", ["Arabic", "English"], index=0 if st.session_state.lang == 'Arabic' else 1)
    if lang_choice != st.session_state.lang:
        st.session_state.lang = lang_choice
        st.rerun()

# 6. الهيدر
st.markdown(f"<h1 style='font-family:Orbitron; color:#00f2ea; margin:0;'>{T['title']}</h1>", unsafe_allow_html=True)

model, scaler = load_model()

# 7. المحتوى الرئيسي
col_input, col_viz = st.columns([1.3, 1])

with col_input:
    with st.form("mustafa_neon_master_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<div class='yellow-box'>{T['group_1']}</div>", unsafe_allow_html=True)
            glu = st.number_input(T['labels'][1], 0, 300, 120)
            ins = st.number_input(T['labels'][4], 0, 900, 80)
            preg = st.number_input(T['labels'][0], 0, 20, 0)
            bp = st.number_input(T['labels'][2], 0, 200, 80)
        with c2:
            st.markdown(f"<div class='yellow-box'>{T['group_2']}</div>", unsafe_allow_html=True)
            bmi = st.number_input(T['labels'][5], 0.0, 70.0, 25.0)
            age = st.number_input(T['labels'][7], 0, 120, 30)
            dpf = st.number_input(T['labels'][6], 0.0, 3.0, 0.5)
            skin = st.number_input(T['labels'][3], 0, 100, 20)
        
        st.markdown("<br>", unsafe_allow_html=True)
        # زر الفحص الأحمر المتوهج
        submit = st.form_submit_button(T['btn'])

with col_viz:
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=[preg, glu, bp, skin, ins, bmi, dpf, age], theta=T['labels'],
        fill='toself', fillcolor='rgba(0, 242, 234, 0.1)', line=dict(color='#00f2ea', width=3)
    ))
    fig_radar.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=False), angularaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#64748b", size=10))),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", height=450
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# 8. قسم النتائج
if submit:
    features = np.array([preg, glu, bp, skin, ins, bmi, dpf, age])
    with st.status("🧠 Scanning Bio-Neural Patterns...") as s:
        time.sleep(1.2)
        result, score, explanation = predict(model, scaler, features)
        s.update(label="Complete", state="complete")
    
    color = "#ff0055" if result == 1 else "#00f2ea"
    st.markdown(f"""
        <div style="border: 2px solid {color}; background: {color}11; padding: 40px; border-radius: 25px; text-align: center; box-shadow: 0 0 35px {color}44;">
            <h1 style="color:{color}; background:none; -webkit-text-fill-color:{color}; font-size:3rem; margin:0;">
                {T['res_pos'] if result == 1 else T['res_neg']}
            </h1>
            <h2 style="margin:10px 0;">{T['conf']}: {score:.1f}%</h2>
        </div>
    """, unsafe_allow_html=True)
