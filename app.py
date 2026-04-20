import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
from model import predict, load_model 

# 1. إعدادات الصفحة وإخفاء كل زوائد ستريمليت
st.set_page_config(page_title="Neural-Med Pro V4.5", page_icon="🧬", layout="wide")

# 2. قاموس الترجمة الشامل (عربي / إنجليزي)
if 'lang' not in st.session_state: st.session_state.lang = 'Arabic'

texts = {
    'English': {
        'dir': 'ltr',
        'title': 'NEURAL-MED V4.5 PRO',
        'subtitle': 'Next-Gen AI Diabetes Intelligence',
        'tab_diag': '🚀 Diagnostic Center',
        'tab_stats': '📊 Analytics Insight',
        'tab_engine': '⚙️ Core Engine',
        'header_input': '🧬 Patient Bio-Metric Scan',
        'group_1': '🩸 Metabolic Markers',
        'group_2': '📏 Physical Metrics',
        'btn': 'INITIALIZE AI SCAN',
        'scanning': 'Analyzing Bio-Patterns...',
        'result_pos': 'CRITICAL: DIABETIC RISK DETECTED',
        'result_neg': 'NORMAL: NO SIGNIFICANT RISK',
        'confidence': 'AI Confidence Score',
        'reasoning': 'Neural Explanation',
        'label_preg': 'Pregnancies',
        'label_glu': 'Glucose Level',
        'label_bp': 'Blood Pressure',
        'label_skin': 'Skin Thickness',
        'label_ins': 'Insulin',
        'label_bmi': 'BMI Index',
        'label_dpf': 'Pedigree Function',
        'label_age': 'Patient Age',
        'footer': 'Designed for NMU AI Engineering'
    },
    'Arabic': {
        'dir': 'rtl',
        'title': 'نيورال-ميد V4.5 برو',
        'subtitle': 'الجيل القادم من ذكاء تشخيص السكري',
        'tab_diag': '🚀 مركز التشخيص',
        'tab_stats': '📊 التحليلات البيانية',
        'tab_engine': '⚙️ المحرك الأساسي',
        'header_input': '🧬 مسح المؤشرات الحيوية',
        'group_1': '🩸 المؤشرات الأيضية',
        'group_2': '📏 القياسات الفيزيائية',
        'btn': 'بدء الفحص الذكي',
        'scanning': 'جاري تحليل الأنماط الحيوية...',
        'result_pos': 'تنبيه: تم رصد مخاطر إصابة',
        'result_neg': 'آمن: لا توجد مخاطر ملحوظة',
        'confidence': 'درجة ثقة الذكاء الاصطناعي',
        'reasoning': 'التفسير العصبي للنتائج',
        'label_preg': 'عدد مرات الحمل',
        'label_glu': 'مستوى الجلوكوز',
        'label_bp': 'ضغط الدم',
        'label_skin': 'سمك الجلد',
        'label_ins': 'الأنسولين',
        'label_bmi': 'مؤشر كتلة الجسم',
        'label_dpf': 'عامل الوراثة',
        'label_age': 'عمر المريض',
        'footer': 'تم التصميم لصالح هندسة الذكاء الاصطناعي بجامعة المنصورة الجديدة'
    }
}

T = texts[st.session_state.lang]

# 3. حقن CSS احترافي (الوضع المظلم الفخم)
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&family=Orbitron:wght@400;700&display=swap');

    /* إخفاء الهيدر الأبيض */
    header[data-testid="stHeader"] {{visibility: hidden;}}
    .main .block-container {{padding-top: 1rem; direction: {T['dir']};}}

    /* الخلفية والتنسيق العام */
    .stApp {{
        background-color: #050505;
        color: #ffffff;
        font-family: 'Cairo', sans-serif;
    }}

    /* حاويات المدخلات (Neon Glass) */
    [data-testid="stForm"] {{
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(0, 242, 234, 0.2) !important;
        border-radius: 25px !important;
        padding: 30px !important;
        box-shadow: 0 0 20px rgba(0, 242, 234, 0.05);
    }}

    /* تصميم الأزرار (Cyberpunk Button) */
    .stButton>button {{
        width: 100%;
        background: linear-gradient(90deg, #00f2ea 0%, #007bff 100%);
        color: white !important;
        border: none;
        padding: 20px;
        border-radius: 15px;
        font-weight: bold;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: 0.5s;
        box-shadow: 0 0 15px rgba(0, 242, 234, 0.3);
    }}
    .stButton>button:hover {{
        box-shadow: 0 0 30px rgba(0, 242, 234, 0.6);
        transform: translateY(-3px);
    }}

    /* تنسيق النصوص */
    h1, h2, h3 {{
        font-family: 'Orbitron', 'Cairo', sans-serif;
        background: linear-gradient(to right, #00f2ea, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    /* تنسيق حقول الأرقام */
    .stNumberInput label {{ color: #94a3b8 !important; font-weight: bold; }}
    input {{
        background-color: #0f172a !important;
        color: #00f2ea !important;
        border: 1px solid #1e293b !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# 4. السايد بار (Sidebar)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3843/3843187.png", width=80)
    st.markdown("### Neural-Med Console")
    lang_choice = st.selectbox("🌍 Language / اللغة", ["Arabic", "English"], index=0 if st.session_state.lang == 'Arabic' else 1)
    if lang_choice != st.session_state.lang:
        st.session_state.lang = lang_choice
        st.rerun()
    st.divider()
    st.caption(T['footer'])

# 5. الهيدر
st.markdown(f"<h1>{T['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color: #64748b; font-size: 1.1rem; margin-top:-10px;'>{T['subtitle']}</p>", unsafe_allow_html=True)

# استدعاء الموديل
model, scaler = load_model()

# 6. التبويبات (Tabs)
tab1, tab2, tab3 = st.tabs([T['tab_diag'], T['tab_stats'], T['tab_engine']])

with tab1:
    col_input, col_radar = st.columns([1.3, 1])
    
    with col_input:
        st.markdown(f"### {T['header_input']}")
        with st.form("pro_scan_form"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"<small style='color:#00f2ea'>{T['group_1']}</small>", unsafe_allow_html=True)
                glucose = st.number_input(T['label_glu'], 0, 300, 120)
                insulin = st.number_input(T['label_ins'], 0, 900, 80)
                preg = st.number_input(T['label_preg'], 0, 20, 0)
                bp = st.number_input(T['label_bp'], 0, 200, 80)
            with c2:
                st.markdown(f"<small style='color:#00f2ea'>{T['group_2']}</small>", unsafe_allow_html=True)
                bmi = st.number_input(T['label_bmi'], 0.0, 70.0, 25.0)
                age = st.number_input(T['label_age'], 0, 120, 30)
                dpf = st.number_input(T['label_dpf'], 0.0, 3.0, 0.5)
                skin = st.number_input(T['label_skin'], 0, 100, 20)
            
            submit = st.form_submit_button(T['btn'])

    with col_radar:
        # رسم الرادار التفاعلي
        labels = [T['label_preg'], T['label_glu'], T['label_bp'], T['label_skin'], T['label_ins'], T['label_bmi'], T['label_dpf'], T['label_age']]
        current_data = [preg, glucose, bp, skin, insulin, bmi, dpf, age]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=current_data,
            theta=labels,
            fill='toself',
            fillcolor='rgba(0, 242, 234, 0.2)',
            line=dict(color='#00f2ea', width=2)
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=False),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#64748b", size=10))
            ),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=40, t=40, b=40),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# 7. معالجة النتائج
if submit:
    features = np.array([preg, glucose, bp, skin, insulin, bmi, dpf, age])
    
    with st.status(T['scanning']) as status:
        time.sleep(1.5)
        result, score, explanation = predict(model, scaler, features)
        status.update(label="Scanning Complete", state="complete", expanded=False)
    
    st.markdown("---")
    
    # بطاقة النتيجة النهائية (Neon Alert)
    res_color = "#ff0055" if result == 1 else "#00f2ea"
    res_bg = "rgba(255, 0, 85, 0.05)" if result == 1 else "rgba(0, 242, 234, 0.05)"
    res_text = T['result_pos'] if result == 1 else T['result_neg']
    
    st.markdown(f"""
        <div style="background: {res_bg}; border: 2px solid {res_color}; padding: 40px; border-radius: 30px; text-align: center; box-shadow: 0 0 30px {res_color}33;">
            <h1 style="color: {res_color}; margin: 0; background:none; -webkit-text-fill-color: {res_color};">{res_text}</h1>
            <p style="font-size: 1.8rem; color: #fff; margin-top:10px;">{T['confidence']}: {score:.1f}%</p>
        </div>
    """, unsafe_allow_html=True)
