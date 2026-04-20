import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
from model import predict, load_model 

# 1. إعدادات الصفحة (إخفاء العناصر الافتراضية)
st.set_page_config(
    page_title="Neural-Med V4.5 Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. نظام إدارة اللغات (قاموس شامل لضمان عدم حدوث KeyError)
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

# 3. الـ CSS الاحترافي (البسملة + الذهب + الأخضر المتوهج + الزر الأحمر النبضي)
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

    /* --- البسملة الذهبية المتوهجة --- */
    .basmala {{
        text-align: center;
        width: 100%;
        font-size: 2.6rem;
        color: #ffcc00;
        text-shadow: 0 0 20px #ffcc00, 0 0 40px #ffcc0066;
        padding: 20px 0;
        margin-bottom: 10px;
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

    /* --- الأخضر المتوهج للمؤشرات --- */
    .stNumberInput label {{ 
        color: #39ff14 !important; 
        font-weight: bold !important;
        text-shadow: 0 0 10px #39ff14;
        font-size: 1.1rem !important;
    }}

    /* المستطيلات الذهبية المتوهجة للعناوين */
    .yellow-glow-box {{
        background: rgba(255, 204, 0, 0.05);
        border: 1.5px solid #ffcc00;
        box-shadow: 0 0 15px rgba(255, 204, 0, 0.3);
        padding: 12px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: #ffcc00 !important;
        text-shadow: 0 0 10px #ffcc00;
        text-align: center;
    }}

    /* --- الزرار الأحمر المتوهج (طبق الأصل مع نبض نيون) --- */
    @keyframes redNeonPulse {{
        0% {{ box-shadow: 0 0 10px #ff4b4b; transform: scale(1); }}
        50% {{ box-shadow: 0 0 35px #ff0000, 0 0 60px #ff000044; transform: scale(1.02); }}
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
        animation: redNeonPulse 2s infinite ease-in-out;
        text-shadow: 0 0 10px #ffffff66;
    }}
    .stButton>button:hover {{
        background: #ff0000 !important;
        box-shadow: 0 0 70px #ff0000 !important;
    }}

    [data-testid="stForm"] {{ background-color: #0c141d !important; border: 1px solid #1e293b !important; border-radius: 25px !important; }}
    .stNumberInput input {{ background-color: #0f172a !important; color: #00f2ea !important; border: 1px solid #1e293b !important; border-radius: 10px !important; }}
    </style>
    """, unsafe_allow_html=True)

# 4. البسملة في أعلى الصفحة
st.markdown("<div class='basmala'>بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ</div>", unsafe_allow_html=True)

# 5. السايد بار
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3843/3843187.png", width=80)
    st.markdown(f"<div class='dev-name'>{T['dev']}</div>", unsafe_allow_html=True)
    st.divider()
    lang_choice = st.selectbox(T['lang_text'], ["Arabic", "English"], index=0 if st.session_state.lang == 'Arabic' else 1)
    if lang_choice != st.session_state.lang:
        st.session_state.lang = lang_choice
        st.rerun()
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("AI Engineering | NMU | © 2026")

# 6. الهيدر
st.markdown(f"<h1 style='font-family:Orbitron; color:#00f2ea; margin:0;'>{T['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='opacity:0.6; font-size:1.1rem; margin-top:-10px;'>{T['sub']}</p>", unsafe_allow_html=True)

# تحميل الموديل
model, scaler = load_model()

# 7. توزيع المحتوى (المدخلات مقابل الرادار)
col_input, col_viz = st.columns([1.3, 1])

with col_input:
    with st.form("mustafa_holy_pro_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<div class='yellow-glow-box'>{T['group_1']}</div>", unsafe_allow_html=True)
            glu = st.number_input(T['labels'][1], 0, 300, 120)
            ins = st.number_input(T['labels'][4], 0, 900, 80)
            preg = st.number_input(T['labels'][0], 0, 20, 0)
            bp = st.number_input(T['labels'][2], 0, 200, 80)
        with c2:
            st.markdown(f"<div class='yellow-glow-box'>{T['group_2']}</div>", unsafe_allow_html=True)
            bmi = st.number_input(T['labels'][5], 0.0, 70.0, 25.0)
            age = st.number_input(T['labels'][7], 0, 120, 30)
            dpf = st.number_input(T['labels'][6], 0.0, 3.0, 0.5)
            skin = st.number_input(T['labels'][3], 0, 100, 20)
        
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button(T['btn'])

with col_viz:
    st.markdown(f"#### AI Profile Analytics")
    fig = go.Figure(data=go.Scatterpolar(
        r=[preg, glu, bp, skin, ins, bmi, dpf, age], theta=T['labels'],
        fill='toself', fillcolor='rgba(0, 242, 234, 0.1)', line=dict(color='#00f2ea', width=3)
    ))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=False), 
                   angularaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#64748b", size=10))),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", height=450
    )
    st.plotly_chart(fig, use_container_width=True)

# 8. قسم النتائج
if submit:
    features = np.array([preg, glu, bp, skin, ins, bmi, dpf, age])
    with st.status("🧠 Scanning Bio-Neural Patterns...") as s:
        time.sleep(1.2)
        result, score, explanation = predict(model, scaler, features)
        s.update(label="Complete", state="complete")
    
    st.markdown("<br>", unsafe_allow_html=True)
    res_color = "#ff0055" if result == 1 else "#00f2ea"
    res_text = T['res_pos'] if result == 1 else T['res_neg']
    
    st.markdown(f"""
        <div style="border: 2px solid {res_color}; background: {res_color}11; padding: 40px; border-radius: 25px; text-align: center; box-shadow: 0 0 30px {res_color}44;">
            <h1 style="color: {res_color}; background:none; -webkit-text-fill-color:{res_color}; margin:0; font-size: 3rem;">
                {res_text}
            </h1>
            <p style="font-size: 1.6rem; margin-top: 15px; opacity: 0.9;">{T['conf']}: <b>{score:.1f}%</b></p>
            <div style="width: 100%; background: #1e293b; border-radius: 10px; height: 12px; margin-top: 20px;">
                <div style="width: {score}%; background: {res_color}; height: 12px; border-radius: 10px; box-shadow: 0 0 15px {res_color};"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)
