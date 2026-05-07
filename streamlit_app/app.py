"""
NeuralFinance · SaaS ML Platform
Lancer avec : streamlit run app.py
"""
import streamlit as st
import sys, os

st.set_page_config(
    page_title="NeuralFinance · SaaS ML Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL FUTURISTIC CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=Space+Grotesk:wght@400;700&display=swap');

:root {
    --bg-main:        #060b14;
    --bg-card:        rgba(13,25,48,0.85);
    --bg-sidebar:     #070d1a;
    --neon-blue:      #00d4ff;
    --neon-purple:    #7c3aed;
    --neon-green:     #00ff9d;
    --neon-pink:      #ff2d78;
    --neon-orange:    #ff6b2b;
    --text-primary:   #e8f4ff;
    --text-secondary: #7a9cc4;
    --border-glow:    rgba(0,212,255,0.2);
    --gradient-1:     linear-gradient(135deg,#00d4ff 0%,#7c3aed 100%);
    --gradient-2:     linear-gradient(135deg,#00ff9d 0%,#00d4ff 100%);
}

html, body, [data-testid="stApp"] {
    background: var(--bg-main) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

[data-testid="stApp"]::before {
    content:'';
    position:fixed;
    inset:0;
    background:
        radial-gradient(ellipse at 20% 50%,rgba(124,58,237,0.09) 0%,transparent 60%),
        radial-gradient(ellipse at 80% 20%,rgba(0,212,255,0.07) 0%,transparent 50%),
        radial-gradient(ellipse at 60% 80%,rgba(0,255,157,0.04) 0%,transparent 50%);
    pointer-events:none;
    z-index:0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid rgba(0,212,255,0.12) !important;
}
[data-testid="stSidebarNav"] { display:none; }
#MainMenu, footer, header { visibility:hidden; }
[data-testid="stToolbar"] { display:none; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg,rgba(0,212,255,0.06) 0%,rgba(124,58,237,0.06) 100%) !important;
    border: 1px solid rgba(0,212,255,0.18) !important;
    border-radius: 14px !important;
    padding: 1rem 1.3rem !important;
    transition: all 0.3s ease !important;
}
div[data-testid="metric-container"]:hover {
    border-color: var(--neon-blue) !important;
    box-shadow: 0 0 22px rgba(0,212,255,0.18) !important;
    transform: translateY(-2px) !important;
}
div[data-testid="metric-container"] label {
    color: var(--text-secondary) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--neon-blue) !important;
    font-family: 'Space Grotesk',sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.55rem !important;
}

/* Buttons */
.stButton > button {
    background: var(--gradient-1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.6rem 1.6rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 18px rgba(0,212,255,0.22) !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 30px rgba(0,212,255,0.48) !important;
    transform: translateY(-2px) !important;
}

/* Inputs */
.stTextInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div,
.stNumberInput input {
    background: rgba(13,25,48,0.9) !important;
    border: 1px solid rgba(0,212,255,0.22) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: var(--neon-blue) !important;
    box-shadow: 0 0 14px rgba(0,212,255,0.18) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(0,212,255,0.02) !important;
    border: 2px dashed rgba(0,212,255,0.28) !important;
    border-radius: 14px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--neon-blue) !important;
    background: rgba(0,212,255,0.05) !important;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,212,255,0.14) !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* Alerts */
.stSuccess  { background:rgba(0,255,157,0.07) !important; border-left:3px solid var(--neon-green) !important; }
.stWarning  { background:rgba(255,107,43,0.07) !important; border-left:3px solid var(--neon-orange) !important; }
.stError    { background:rgba(255,45,120,0.07) !important; border-left:3px solid var(--neon-pink) !important; }
.stInfo     { background:rgba(0,212,255,0.05) !important; border-left:3px solid var(--neon-blue) !important; }

/* Dividers */
hr { border-color: rgba(0,212,255,0.14) !important; }

/* Expanders */
details {
    background: rgba(13,25,48,0.6) !important;
    border: 1px solid rgba(0,212,255,0.14) !important;
    border-radius: 12px !important;
}

/* Sliders */
.stSlider > div > div > div > div { background: var(--gradient-1) !important; }

/* Radio buttons */
.stRadio > div { gap:0.3rem !important; }
.stRadio > div > label {
    background: rgba(13,25,48,0.6) !important;
    border: 1px solid rgba(0,212,255,0.1) !important;
    border-radius: 10px !important;
    padding: 0.55rem 1rem !important;
    transition: all 0.2s ease !important;
    color: var(--text-secondary) !important;
    cursor: pointer !important;
    width: 100% !important;
}
.stRadio > div > label:hover {
    border-color: var(--neon-blue) !important;
    color: var(--text-primary) !important;
    background: rgba(0,212,255,0.07) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:var(--bg-main); }
::-webkit-scrollbar-thumb { background:rgba(0,212,255,0.28); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:var(--neon-blue); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(13,25,48,0.7) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid rgba(0,212,255,0.12) !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    border-radius: 9px !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--gradient-1) !important;
    color: white !important;
}

/* Page titles */
h1 {
    font-family: 'Space Grotesk',sans-serif !important;
    font-weight: 900 !important;
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em !important;
}
h2, h3 {
    font-family: 'Space Grotesk',sans-serif !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Ajout du dossier courant au path ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── Import des pages ──────────────────────────────────────────────────────────
from pages import p1_data, p2_risk, p3_segmentation, p4_cashflow, p5_anomaly, p6_regression, p7_nlp

# ══════════════════════════════════════════════════════════════════════════════
# PAGES MAP
# ══════════════════════════════════════════════════════════════════════════════
PAGES = {
    "📂  Données & Entraînement":       p1_data,
    "💰  Risque de Paiement":           p2_risk,
    "👥  Segmentation Client (RFM)":    p3_segmentation,
    "📈  Prévision de Trésorerie":      p4_cashflow,
    "⚠️  Détection d'Anomalies":        p5_anomaly,
    "📉  Délai de Paiement":            p6_regression,
    "🧾  Catégorisation NLP":           p7_nlp,
}

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown("""
    <div style="padding:0.5rem 0 1.5rem 0;text-align:center;">
        <div style="
            font-family:'Space Grotesk',sans-serif;
            font-size:1.45rem;font-weight:900;
            background:linear-gradient(135deg,#00d4ff,#7c3aed);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            background-clip:text;letter-spacing:-0.02em;line-height:1.2;">
            ⚡ NeuralFinance
        </div>
        <div style="font-size:0.65rem;color:#7a9cc4;letter-spacing:0.14em;
                    text-transform:uppercase;margin-top:0.35rem;">
            SaaS ML Platform &middot; Tunisia v2.1
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(0,212,255,0.14);margin:0 0 1rem 0"/>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;
                color:#7a9cc4;margin-bottom:0.4rem;padding-left:0.2rem;">Navigation</div>
    """, unsafe_allow_html=True)

    page_name = st.radio("nav", list(PAGES.keys()), label_visibility="collapsed")

    st.markdown('<hr style="border-color:rgba(0,212,255,0.14);margin:1rem 0"/>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;
                color:#7a9cc4;margin-bottom:0.7rem;padding-left:0.2rem;">Statut des modèles</div>
    """, unsafe_allow_html=True)

    status_items = [
        ("UC1 – Risque paiement",  'metrics_uc1',    "#00d4ff"),
        ("UC2 – Segmentation RFM", 'rfm',            "#7c3aed"),
        ("UC3 – Trésorerie",       'df',             "#00ff9d"),
        ("UC4 – Anomalies",        'anomaly_labels', "#ff2d78"),
        ("UC5 – Régression",       'metrics_reg',    "#ff6b2b"),
        ("UC6 – NLP",              'acc_nlp',        "#00d4ff"),
    ]
    for label, key, color in status_items:
        ready = key in st.session_state
        dot   = color if ready else "#1e3050"
        txt   = "#c8dff5" if ready else "#3a5070"
        bg    = "rgba(0,212,255,0.04)" if ready else "transparent"
        bdr   = "1px solid rgba(0,212,255,0.1)" if ready else "1px solid transparent"
        glow  = f"box-shadow:0 0 8px {color};" if ready else ""
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.55rem;
                    padding:0.35rem 0.55rem;border-radius:8px;margin-bottom:0.2rem;
                    background:{bg};border:{bdr};">
            <div style="width:8px;height:8px;border-radius:50%;
                        background:{dot};{glow}flex-shrink:0;"></div>
            <span style="font-size:0.77rem;color:{txt};">{label}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(0,212,255,0.08);margin:1rem 0"/>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;font-size:0.6rem;color:#2a4060;letter-spacing:0.08em;">
        © 2026 NeuralFinance · All rights reserved
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RENDER
# ══════════════════════════════════════════════════════════════════════════════
PAGES[page_name].render()
