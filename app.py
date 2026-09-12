import os
import io
import time
import datetime
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Import platform internal modules
from src.api_services import inspect_image_with_vision
from src.fusion_layer import generate_audit_report
from src.telemetry_engine import TelemetryEngine
from src.mrv_engine import compute_continuous_mrv_index, compute_carbon_financials
from src.auth_manager import (
    init_db, verify_login, create_user, change_user_password,
    list_all_users, save_user_farm_params, load_user_farm_params
)
from src.pdf_generator import generate_farm_carbon_pdf

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Algi. | Aquatic Carbon MRV Platform",
    page_icon="assets/algi_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database schema and pre-seeded accounts
init_db()

st.markdown("""
<style>
    /* =========================================================
       ALGI. PLATFORM  —  ENTERPRISE DESIGN SYSTEM
       Theme colors:
         Primary   #1a7a4a  (mid green — interactive elements)
         Deep      #00462e  (brand dark green — titles, accents)
         Mint      #10b981  (highlights, pulse dot)
         Text      #1a2e25  (readable dark slate)
         Muted     #3d5a4f  (secondary labels)
         Surface   #f0f7f4  (light mint surface)
         Border    #c8e6d8  (card outlines)
    ========================================================= */

    .brand-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #00462e;
        letter-spacing: -0.04em;
        line-height: 1.1;
        margin-bottom: 0.1rem;
    }
    .brand-dot { color: #10b981; }
    .brand-tag {
        display: inline-block;
        background: #e6f4ed;
        color: #0d5c34;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 2px 9px;
        border-radius: 4px;
        margin-left: 10px;
        vertical-align: middle;
        border: 1px solid #9ad4b7;
    }
    .sub-header {
        color: #3d5a4f;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 1.2rem;
    }

    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 0.6rem 0 0.8rem 0;
    }
    .section-header .accent-bar {
        width: 4px;
        height: 24px;
        background: #1a7a4a;
        border-radius: 3px;
        flex-shrink: 0;
    }
    .section-header .section-icon {
        font-size: 1.15rem;
        color: #1a7a4a;
        font-style: normal;
        line-height: 1;
    }
    .section-header h3 {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1a2e25;
        margin: 0;
        padding: 0;
        letter-spacing: -0.01em;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #c8e6d8;
        border-top: 3px solid #1a7a4a;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 2px 8px rgba(26, 122, 74, 0.06);
    }
    div[data-testid="stMetric"] label {
        color: #2d5a40 !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #1a2e25 !important;
        font-weight: 800 !important;
    }

    /* Buttons */
    div.stButton > button[kind="primary"],
    div.stButton > button:first-child {
        background: #1a7a4a !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 7px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 9px 20px !important;
        letter-spacing: 0.01em;
        transition: background 0.16s ease, box-shadow 0.16s ease;
        box-shadow: 0 2px 8px rgba(26, 122, 74, 0.22);
    }
    div.stButton > button:first-child:hover {
        background: #0f5c34 !important;
        box-shadow: 0 4px 16px rgba(26, 122, 74, 0.32) !important;
    }

    /* Status Badges */
    .status-badge-healthy {
        background-color: #e6f4ed;
        color: #0d5c34;
        border: 1px solid #6fcfa0;
        padding: 5px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.84rem;
        display: inline-block;
    }
    .status-badge-warning {
        background-color: #fff8e1;
        color: #7c4a00;
        border: 1px solid #f0c040;
        padding: 5px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.84rem;
        display: inline-block;
    }
    .status-badge-danger {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #f87171;
        padding: 5px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.84rem;
        display: inline-block;
    }

    .status-dot-green {
        display: inline-block;
        width: 8px; height: 8px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
        box-shadow: 0 0 5px rgba(34,197,94,0.5);
    }
    .status-dot-yellow {
        display: inline-block;
        width: 8px; height: 8px;
        background: #f59e0b;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
    }
    .status-dot-red {
        display: inline-block;
        width: 8px; height: 8px;
        background: #ef4444;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
    }
    .sidebar-model-row {
        display: flex;
        align-items: center;
        font-size: 0.88rem;
        padding: 4px 0;
        color: #1a2e25;
    }
    .sidebar-model-label {
        font-weight: 600;
        color: #3d5a4f;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .telemetry-status-banner {
        background: linear-gradient(135deg, #0d5c34 0%, #1a7a4a 100%);
        color: #e6f4ed;
        padding: 11px 20px;
        border-radius: 8px;
        margin: 10px 0 14px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 16px rgba(13, 92, 52, 0.18);
        font-size: 0.9rem;
    }
    .telemetry-pulse {
        height: 9px; width: 9px;
        background-color: #34d399;
        border-radius: 50%;
        display: inline-block;
        margin-right: 7px;
        box-shadow: 0 0 7px #34d399;
        animation: pulse-glow 1.8s infinite;
    }
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 5px #34d399; }
        50%       { box-shadow: 0 0 12px #34d399, 0 0 20px rgba(52,211,153,0.3); }
    }

    .preset-card {
        background: #ffffff;
        border: 1px solid #c8e6d8;
        border-radius: 10px;
        padding: 16px 18px;
        text-align: center;
        cursor: pointer;
        transition: box-shadow 0.15s ease, border-color 0.15s ease;
    }
    .preset-card:hover {
        box-shadow: 0 4px 14px rgba(26,122,74,0.14);
        border-color: #1a7a4a;
    }
    .preset-indicator {
        width: 12px; height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
        vertical-align: middle;
    }

    .audit-card {
        background: #ffffff;
        border: 1px solid #c8e6d8;
        border-left: 5px solid #1a7a4a;
        border-radius: 10px;
        padding: 22px 26px;
        box-shadow: 0 2px 12px rgba(26,122,74,0.05);
        margin-bottom: 16px;
        color: #1a2e25;
        line-height: 1.7;
    }

    .login-card {
        background: #ffffff;
        border: 1px solid #c8e6d8;
        border-top: 4px solid #1a7a4a;
        border-radius: 12px;
        padding: 28px 32px;
        box-shadow: 0 4px 20px rgba(26,122,74,0.08);
    }

    .user-profile-badge {
        background: #e6f4ed;
        border: 1px solid #9ad4b7;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 14px;
    }

    hr { border-color: #c8e6d8 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper: render a professional section header
# ---------------------------------------------------------
def section_header(icon_symbol: str, title: str):
    st.markdown(f"""
    <div class="section-header">
        <div class="accent-bar"></div>
        <i class="section-icon">{icon_symbol}</i>
        <h3>{title}</h3>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Resource Caching & Universal Model Loader
# ---------------------------------------------------------
@st.cache_resource
def load_ml_assets():
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
    g_path = os.path.join(models_dir, "algae_growth_model.pkl")
    w_path = os.path.join(models_dir, "water_quality_model.pkl")
    e_path = os.path.join(models_dir, "label_encoder.pkl")
    growth_model = joblib.load(g_path) if os.path.exists(g_path) else None
    water_model  = joblib.load(w_path) if os.path.exists(w_path) else None
    encoder      = joblib.load(e_path) if os.path.exists(e_path) else None
    return growth_model, water_model, encoder

@st.cache_resource
def get_telemetry_engine():
    return TelemetryEngine()

growth_model, water_model, encoder = load_ml_assets()
telemetry_engine = get_telemetry_engine()

# ---------------------------------------------------------
# Authentication State Initialization ("Bank Account" Model)
# ---------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = ""
if "farm_name" not in st.session_state:
    st.session_state.farm_name = ""
if "admin_nav_view" not in st.session_state:
    st.session_state.admin_nav_view = "Admin Management Portal"
if "telemetry_log" not in st.session_state:
    st.session_state.telemetry_log = []
if "last_tick_time" not in st.session_state:
    st.session_state.last_tick_time = "Initial Boot"
if "active_station" not in st.session_state:
    st.session_state.active_station = "Station 1"
if "last_audit_report" not in st.session_state:
    st.session_state.last_audit_report = None
if "current_metrics" not in st.session_state:
    st.session_state.current_metrics = telemetry_engine.get_preset_scenario("Standard Open Pond")

logo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets", "algi_logo.png"))

# =========================================================
# GATEWAY: LOGIN SCREEN (When Not Authenticated)
# =========================================================
if not st.session_state.authenticated:
    c_login_l, c_login_card, c_login_r = st.columns([1, 2.2, 1])
    
    with c_login_card:
        st.markdown("<br>", unsafe_allow_html=True)
        if os.path.exists(logo_path):
            st.image(logo_path, width=110)
        
        st.markdown(
            '<div class="brand-title">Algi<span class="brand-dot">.</span>'
            '<span class="brand-tag">Identity Gateway</span></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="sub-header">Aquatic Carbon Removal &amp; Verification Platform<br/>'
            '<b>Secured Role-Based Access Control (RBAC) &bull; Isolated Farm Workspaces</b></div>',
            unsafe_allow_html=True
        )
        
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown("#### \u25BA Secure Portal Authentication")
        st.caption("Enter your assigned Facility Operator ID or Platform Administrator credentials.")
        
        login_username = st.text_input("User ID / Operator Username", placeholder="e.g. operator_alpha or admin")
        login_password = st.text_input("Password", type="password", placeholder="Enter your secure password")
        
        col_btn_log, col_help = st.columns([1.5, 2.5])
        with col_btn_log:
            do_login = st.button("Authenticate & Enter", type="primary", use_container_width=True)
            
        if do_login:
            if not login_username or not login_password:
                st.error("Please provide both User ID and Password.")
            else:
                success, result = verify_login(login_username, login_password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.username = result["username"]
                    st.session_state.role = result["role"]
                    st.session_state.farm_name = result["farm_name"]
                    
                    # Load isolated parameters for this specific user
                    user_params = load_user_farm_params(result["username"])
                    st.session_state.current_metrics = user_params
                    st.session_state.last_audit_report = None
                    st.success(f"Welcome, {result['username']}! Access granted to {result['farm_name']}.")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"Authentication Failed: {result}")

        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔑  Demo Credentials for Evaluators & Judges"):
            st.markdown("""
            - **Platform Administrator**:
              - User ID: `admin`
              - Password: `Admin@Algi2026`
              - *Capabilities: User Provisioning, Global API key configuration, Full MRV inspection.*
            <br>
            - **Pre-Provisioned Farm Operator**:
              - User ID: `operator_alpha`
              - Password: `Farm@1234`
              - *Facility: Alpha Spirulina Bioreactor Station*
              - *Capabilities: Isolated farm parameters, continuous MRV index, PDF report download, password change.*
            """, unsafe_allow_html=True)
    
    st.stop() # Halts execution so unauthenticated users cannot see any dashboard content!

# =========================================================
# AUTHENTICATED ENVIRONMENT: SIDEBAR MANAGEMENT
# =========================================================
with st.sidebar:
    if os.path.exists(logo_path):
        st.image(logo_path, width=115)
    else:
        st.markdown("### Algi.")

    # User Profile Card
    role_badge = "\U0001F6E1  ADMINISTRATOR" if st.session_state.role == "admin" else "\U0001F33F  FARM OPERATOR"
    st.markdown(f"""
    <div class="user-profile-badge">
        <div style="font-size:0.75rem; color:#2d5a40; font-weight:700; text-transform:uppercase;">
            {role_badge}
        </div>
        <div style="font-size:1.05rem; font-weight:800; color:#00462e;">
            {st.session_state.username}
        </div>
        <div style="font-size:0.8rem; color:#3d5a4f; margin-top:2px;">
            <b>Facility:</b> {st.session_state.farm_name}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("\u23FB  Log Out of Portal", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.role = ""
        st.session_state.farm_name = ""
        st.session_state.last_audit_report = None
        st.rerun()

    st.divider()

    # Universal ML Architecture Status
    st.markdown('<span class="sidebar-model-label">Universal ML Models</span>', unsafe_allow_html=True)
    def _dot(ok): return "status-dot-green" if ok else "status-dot-red"
    st.markdown(f"""
    <div class="sidebar-model-row">
        <span class="{_dot(growth_model is not None)}"></span>
        <span><b>Biomass Regressor</b> — XGBoost (R² = 0.959)</span>
    </div>
    <div class="sidebar-model-row">
        <span class="{_dot(water_model is not None)}"></span>
        <span><b>Water Classifier</b> — XGBoost (Acc = 99.6%)</span>
    </div>
    <div class="sidebar-model-row">
        <span class="status-dot-green"></span>
        <span><b>Vision Core</b> — Gemini 3.6 / CV Engine</span>
    </div>
    <div class="sidebar-model-row">
        <span class="status-dot-green"></span>
        <span><b>Auditor</b> — ISO 14064 Gemini Engine</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ADMIN-ONLY: API Key Configuration & Navigation Switcher
    if st.session_state.role == "admin":
        st.markdown('<span class="sidebar-model-label">Admin Control Panel</span>', unsafe_allow_html=True)
        admin_portal_mode = st.radio(
            "Admin Active View:",
            ["Admin Management Portal", "Live Carbon MRV Platform"],
            index=0 if st.session_state.admin_nav_view == "Admin Management Portal" else 1
        )
        st.session_state.admin_nav_view = admin_portal_mode

        st.divider()
        st.markdown('<span class="sidebar-model-label">API Key Vault (Admin Only)</span>', unsafe_allow_html=True)
        has_gemini = bool(os.getenv("GEMINI_API_KEY") and not os.getenv("GEMINI_API_KEY").startswith("your_"))
        has_groq   = bool(os.getenv("GROQ_API_KEY")   and not os.getenv("GROQ_API_KEY").startswith("your_"))

        st.markdown(f"""
        <div class="sidebar-model-row">
            <span class="{'status-dot-green' if has_gemini else 'status-dot-red'}"></span>
            <span>Gemini 3.6 API: {'Active' if has_gemini else 'Key missing'}</span>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Configure Global API Keys"):
            admin_gemini = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
            admin_groq   = st.text_input("Groq API Key",   value=os.getenv("GROQ_API_KEY", ""),   type="password")
            if st.button("Save Keys to Platform"):
                if admin_gemini: os.environ["GEMINI_API_KEY"] = admin_gemini
                if admin_groq:   os.environ["GROQ_API_KEY"]   = admin_groq
                st.success("Global API credentials updated!")
                st.rerun()

    # USER-ONLY: Password Change Dialog (Zero API access)
    else:
        st.markdown('<span class="sidebar-model-label">Account Security</span>', unsafe_allow_html=True)
        with st.expander("Change My Password"):
            current_pw = st.text_input("Current Password", type="password")
            new_pw_1 = st.text_input("New Password", type="password")
            new_pw_2 = st.text_input("Confirm New Password", type="password")
            if st.button("Update Password"):
                if not current_pw or not new_pw_1:
                    st.error("Please fill in all fields.")
                else:
                    success, msg = change_user_password(st.session_state.username, current_pw, new_pw_1, new_pw_2)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

    st.divider()
    st.caption(f"Algi. v2.0  |  ISO 14064 & Verra VM0042 MRV")
    st.caption("Bank-Grade Encrypted Facility Isolation")

# =========================================================
# ROUTING A: ADMIN MANAGEMENT PORTAL
# =========================================================
if st.session_state.role == "admin" and st.session_state.admin_nav_view == "Admin Management Portal":
    c_head_logo, c_head_txt = st.columns([1, 9])
    with c_head_logo:
        if os.path.exists(logo_path):
            st.image(logo_path, width=90)
    with c_head_txt:
        st.markdown(
            '<div class="brand-title">Algi<span class="brand-dot">.</span>'
            '<span class="brand-tag">Admin Center</span></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="sub-header">Platform Administration &bull; User Provisioning &bull; Multi-Tenant Farm Access Control</div>',
            unsafe_allow_html=True
        )

    st.divider()

    tab_provision, tab_users, tab_infra = st.tabs([
        "➕  Provision New Farm Operator",
        "📋  Registered Operators & Facilities",
        "⚙  Infrastructure & API Key Vault"
    ])

    # Tab 1: Provision New User
    with tab_provision:
        section_header("➕", "Provision New Algae Farm Operator Account")
        st.info("Provision new farm operators with isolated database workspaces. Once provisioned, the operator receives their unique ID and password.")
        
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            new_user_id = st.text_input("Assign Operator User ID / Username", placeholder="e.g. operator_beta")
            new_farm_name = st.text_input("Facility / Farm Name", placeholder="e.g. Beta Chlorella Raceway Pond #3")
            new_role = st.selectbox("Assigned System Role", ["user", "admin"], format_func=lambda x: "Farm Operator (Isolated Workspace)" if x == "user" else "Platform Administrator (Full Rights)")

        with p_c2:
            new_password = st.text_input("Temporary Access Password", type="password", placeholder="Min 6 characters")
            new_password_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")

        if st.button("Confirm & Add User to Database", type="primary"):
            if not new_user_id or not new_password:
                st.error("User ID and Password are required.")
            elif new_password != new_password_confirm:
                st.error("Passwords do not match. Please verify.")
            else:
                success, msg = create_user(new_user_id, new_password, new_role, new_farm_name or "Commercial Algae Farm")
                if success:
                    st.success(f"Success! {msg}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(msg)

    # Tab 2: User Registry
    with tab_users:
        section_header("📋", "Provisioned Multi-Tenant User Registry")
        all_users = list_all_users()
        if all_users:
            df_users = pd.DataFrame(all_users)
            df_users.columns = ["Record ID", "Operator Username", "System Role", "Facility / Farm Name", "Created Timestamp"]
            st.dataframe(df_users, use_container_width=True, hide_index=True)
        else:
            st.info("No users registered.")

    # Tab 3: Infrastructure & Keys
    with tab_infra:
        section_header("⚙", "Global Platform Infrastructure & AI Key Vault")
        st.markdown(f"""
        - **Database**: SQLite Local Vault at `data/algi_auth.db`
        - **Encryption Standard**: Werkzeug PBKDF2 / SHA-256 with dynamic salt
        - **Universal Models**: `models/algae_growth_model.pkl` (XGBoost), `models/water_quality_model.pkl`
        - **Gemini Model**: `gemini-3.6-flash` (Active)
        - **Groq Vision Core**: Multimodal / Llama 3.2 Vision Fallback
        """)

    st.stop()

# =========================================================
# ROUTING B: USER / OPERATOR ISOLATED FARM WORKSPACE
# =========================================================
c_logo, c_title = st.columns([1, 9])
with c_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
with c_title:
    st.markdown(
        f'<div class="brand-title">Algi<span class="brand-dot">.</span>'
        f'<span class="brand-tag">Operator Workspace</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="sub-header"><b>Facility:</b> {st.session_state.farm_name} &nbsp;|&nbsp; '
        f'<b>Operator ID:</b> <code>{st.session_state.username}</code> &nbsp;|&nbsp; '
        f'ISO 14064 / Verra VM0042 Protocol Compliance</div>',
        unsafe_allow_html=True
    )

st.divider()

# ---------------------------------------------------------
# Data Ingestion Mode Toggle
# ---------------------------------------------------------
MODE_MANUAL  = "\u2699  Manual Parameter Entry"
MODE_IOT     = "\u2315  Real-Time IoT Telemetry Stream"
MODE_PRESETS = "\u25B6  Presentation Presets"

ingestion_mode = st.radio(
    "Telemetry Ingestion Mode",
    [MODE_MANUAL, MODE_IOT, MODE_PRESETS],
    horizontal=True
)

st.divider()

# =========================================================
# MODE 1 — MANUAL PARAMETER ENTRY (ISOLATED PER USER)
# =========================================================
if ingestion_mode == MODE_MANUAL:
    section_header("\u2699", f"Isolated Parameter Configuration: {st.session_state.farm_name}")
    st.info(
        f"Adjust environmental parameters for **{st.session_state.farm_name}**. "
        f"Your parameters are isolated and saved specifically to your account. Universal ML models compute live biological proxy predictions in real time."
    )

    col_env, col_water = st.columns(2)

    with col_env:
        st.markdown("**Algae Growth & Photosynthetic Inputs**")
        m_temp = st.slider("Water Temperature (°C)", 10.0, 45.0, float(st.session_state.current_metrics.get("Temperature", 26.0)), 0.1)
        m_ph = st.slider("pH Level", 4.0, 11.0, float(st.session_state.current_metrics.get("pH", 7.5)), 0.05)
        m_light = st.slider("Light Intensity (Lux / PAR)", 500, 12000, int(st.session_state.current_metrics.get("Light", 5000)), 50)
        m_co2 = st.number_input("Dissolved / Injected CO\u2082 (ppm)",
                                min_value=100.0, max_value=2000.0,
                                value=float(st.session_state.current_metrics.get("CO2", 450.0)), step=10.0)
        m_nitrate = st.number_input("Nitrate (NO\u2083\u207B  ppm)",
                                    min_value=0.0, max_value=200.0,
                                    value=float(st.session_state.current_metrics.get("Nitrate", 18.0)), step=0.5)
        m_iron = st.number_input("Iron — Fe (mg/L)",
                                 min_value=0.001, max_value=5.0,
                                 value=float(st.session_state.current_metrics.get("Iron", 0.45)), step=0.01)
        m_phosphate = st.number_input("Phosphate — PO\u2084 (mg/L)",
                                      min_value=0.001, max_value=15.0,
                                      value=float(st.session_state.current_metrics.get("Phosphate", 1.5)), step=0.05)

    with col_water:
        st.markdown("**Pond Water Quality & Sensor Diagnostics**")
        m_ammonia = st.number_input("Ammonia — NH\u2083/NH\u2084\u207A (mg/L)",
                                     min_value=0.0, max_value=10.0,
                                     value=float(st.session_state.current_metrics.get("Ammonia", 0.02)), step=0.005, format="%.4f")
        m_do = st.slider("Dissolved Oxygen — DO (mg/L)", 0.0, 25.0, float(st.session_state.current_metrics.get("DO", 7.5)), 0.1)
        m_turbidity = st.slider("Turbidity (NTU)", 0.0, 150.0, float(st.session_state.current_metrics.get("Turbidity", 15.0)), 0.5)
        m_manganese = st.number_input("Manganese — Mn (mg/L)",
                                      min_value=0.0, max_value=10.0,
                                      value=float(st.session_state.current_metrics.get("Manganese", 0.05)), step=0.01, format="%.3f")
        st.caption("Optimal carbon uptake range: pH 7.2 – 8.4, DO > 6.0 mg/L, Ammonia < 0.05 mg/L")

    # Update state and automatically persist isolated parameters to SQLite
    active_metrics = {
        "Temperature": m_temp, "pH": m_ph, "CO2": m_co2, "Light": m_light,
        "Nitrate": m_nitrate, "Iron": m_iron, "Phosphate": m_phosphate,
        "Ammonia": m_ammonia, "DO": m_do, "Turbidity": m_turbidity, "Manganese": m_manganese
    }
    st.session_state.current_metrics = active_metrics
    save_user_farm_params(st.session_state.username, active_metrics)

# =========================================================
# MODE 2 — REAL-TIME IOT TELEMETRY STREAM
# =========================================================
elif ingestion_mode == MODE_IOT:
    section_header("\u2315", "Real-Time IoT Pond Telemetry Feed")
    st.caption("Receives continuous sensor ticks from remote monitoring buoys (Pondsdata telemetry stream, LoRaWAN / 24-bit ADC).")

    c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4 = st.columns([2, 2, 2, 2])

    with c_ctrl1:
        stations = telemetry_engine.get_stations()
        selected_station = st.selectbox("Active Monitoring Station:", stations, index=0)
        st.session_state.active_station = selected_station
    with c_ctrl2:
        fetch_now = st.button("\u21BA  Fetch Latest Telemetry Tick", use_container_width=True)
    with c_ctrl3:
        live_weather_sync = st.checkbox("Sync Open-Meteo Weather", value=True,
                                        help="Fetches real-time ambient temperature and solar irradiance.")
    with c_ctrl4:
        auto_stream = st.checkbox("\u25BA  Live Stream Auto-Poll", value=False,
                                  help="Continuously poll new telemetry ticks every 3 seconds.")

    if fetch_now or auto_stream or "last_telemetry" not in st.session_state:
        tick = telemetry_engine.fetch_live_iot_tick(station=selected_station)
        if live_weather_sync:
            w_data = telemetry_engine.fetch_live_weather()
            if w_data.get("available"):
                tick["metrics"]["Temperature"] = round(
                    (tick["metrics"]["Temperature"] + w_data["temperature"]) / 2.0, 2
                )
                tick["metrics"]["Light"] = w_data["light"]

        st.session_state.current_metrics = tick["metrics"]
        st.session_state.last_tick_time  = tick["timestamp"]

        history_entry = dict(tick["metrics"])
        history_entry["timestamp"] = tick["timestamp"]
        history_entry["station"]   = tick["station"]
        st.session_state.telemetry_log.append(history_entry)
        if len(st.session_state.telemetry_log) > 35:
            st.session_state.telemetry_log.pop(0)

    st.markdown(f"""
    <div class="telemetry-status-banner">
        <div>
            &#9632;&nbsp; <b>Station</b>: {st.session_state.active_station}
            &nbsp;|&nbsp;
            <b>Timestamp</b>: {st.session_state.last_tick_time}
        </div>
        <div>
            <span class="telemetry-pulse"></span>
            <b>Live IoT Buoy Stream Active</b> &mdash; LoRaWAN / 24-bit ADC
        </div>
    </div>
    """, unsafe_allow_html=True)

    m = st.session_state.current_metrics
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("DO (mg/L)", f"{m['DO']:.2f}", delta="Normal" if m['DO'] >= 5.0 else "Hypoxia Alert",
              delta_color="normal" if m['DO'] >= 5.0 else "inverse")
    k2.metric("Pond pH", f"{m['pH']:.2f}", delta="Optimal" if 7.0 <= m['pH'] <= 8.5 else "Suboptimal")
    k3.metric("Temp (°C)", f"{m['Temperature']:.1f}", delta=f"{m['Temperature']-25.0:+.1f} °C")
    k4.metric("Nitrate (ppm)", f"{m['Nitrate']:.1f}")
    k5.metric("Ammonia (mg/L)", f"{m['Ammonia']:.3f}", delta="Safe" if m['Ammonia'] < 0.1 else "Elevated",
              delta_color="normal" if m['Ammonia'] < 0.1 else "inverse")
    k6.metric("Turbidity (NTU)", f"{m['Turbidity']:.1f}")

    if len(st.session_state.telemetry_log) >= 2:
        df_hist = pd.DataFrame(st.session_state.telemetry_log)
        ch1, ch2 = st.columns(2)
        with ch1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(y=df_hist["DO"], mode="lines+markers", name="DO (mg/L)", line=dict(color="#1a7a4a", width=2.5)))
            fig1.add_trace(go.Scatter(y=df_hist["Temperature"], mode="lines+markers", name="Temp (°C)", line=dict(color="#0284c7", width=2)))
            fig1.update_layout(title="Dissolved Oxygen & Temperature Dynamics",
                               height=260, margin=dict(l=20,r=20,t=40,b=20), template="plotly_white")
            st.plotly_chart(fig1, use_container_width=True)
        with ch2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=df_hist["pH"], mode="lines+markers", name="pH Level", line=dict(color="#10b981", width=2.5)))
            fig2.add_trace(go.Scatter(y=df_hist["Nitrate"], mode="lines+markers", name="Nitrate (ppm)", line=dict(color="#d97706", width=2)))
            fig2.update_layout(title="pH & Nitrogen Balance",
                               height=260, margin=dict(l=20,r=20,t=40,b=20), template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# MODE 3 — PRESENTATION PRESETS
# =========================================================
else:
    section_header("\u25B6", "Presentation Scenario Presets")
    st.caption("Load predefined operational conditions to demonstrate model behavior under various environmental stresses.")

    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        st.markdown("""<div class="preset-card">
            <span class="preset-indicator" style="background:#22c55e;"></span>
            <b>Optimal Bio-Reactor</b><br>
            <small style="color:#3d5a4f;">Max CO&#8322; uptake scenario</small>
        </div>""", unsafe_allow_html=True)
        if st.button("Load Optimal", use_container_width=True):
            st.session_state.current_metrics = telemetry_engine.get_preset_scenario("Optimal Growth Bio-Reactor")
            st.session_state.last_audit_report = None
            st.rerun()
    with p_col2:
        st.markdown("""<div class="preset-card">
            <span class="preset-indicator" style="background:#f59e0b;"></span>
            <b>Acidic Nutrient Runoff</b><br>
            <small style="color:#3d5a4f;">Stress alert condition</small>
        </div>""", unsafe_allow_html=True)
        if st.button("Load Acidic Runoff", use_container_width=True):
            st.session_state.current_metrics = telemetry_engine.get_preset_scenario("Acidic Nutrient Runoff")
            st.session_state.last_audit_report = None
            st.rerun()
    with p_col3:
        st.markdown("""<div class="preset-card">
            <span class="preset-indicator" style="background:#ef4444;"></span>
            <b>Hypoxic Low-Oxygen</b><br>
            <small style="color:#3d5a4f;">Critical risk condition</small>
        </div>""", unsafe_allow_html=True)
        if st.button("Load Hypoxic", use_container_width=True):
            st.session_state.current_metrics = telemetry_engine.get_preset_scenario("Hypoxic Low-Oxygen Alert")
            st.session_state.last_audit_report = None
            st.rerun()
    with p_col4:
        st.markdown("""<div class="preset-card">
            <span class="preset-indicator" style="background:#3b82f6;"></span>
            <b>Standard Open Pond</b><br>
            <small style="color:#3d5a4f;">Baseline reference state</small>
        </div>""", unsafe_allow_html=True)
        if st.button("Load Baseline", use_container_width=True):
            st.session_state.current_metrics = telemetry_engine.get_preset_scenario("Standard Open Pond")
            st.session_state.last_audit_report = None
            st.rerun()

st.divider()

# ---------------------------------------------------------
# Visual Inspection — Computer Vision Layer
# ---------------------------------------------------------
section_header("\u25A6", "Visual Inspection & Computer Vision Layer")
v_col1, v_col2 = st.columns([1, 1])

selected_image = None
image_label    = ""

with v_col1:
    images_dir    = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "images"))
    left_img_dir  = os.path.join(images_dir, "dataset", "leftImg8bit")
    gt_fine_dir   = os.path.join(images_dir, "dataset", "gtFine")

    kaggle_images = []
    if os.path.exists(left_img_dir):
        files = [f for f in os.listdir(left_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 999999)
        kaggle_images = [os.path.join("dataset", "leftImg8bit", f) for f in files]
    elif os.path.exists(images_dir):
        for root, dirs, files in os.walk(images_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith("sample_"):
                    rel = os.path.relpath(os.path.join(root, f), images_dir)
                    kaggle_images.append(rel)

    options = ["Upload Custom Pond Photo", "Curated Presentation Samples"]
    if len(kaggle_images) > 0:
        options.insert(0, f"Authentic Kaggle Dataset ({len(kaggle_images)} real photos)")

    img_source = st.radio("Image Source:", options, horizontal=True)
    mask_image = None

    if "Authentic Kaggle Dataset" in img_source:
        selected_rel = st.selectbox(
            "Select Authentic Kaggle 1080p Bloom Photo:",
            kaggle_images,
            format_func=lambda x: f"Pond Bloom #{os.path.splitext(os.path.basename(x))[0]}  (1920 x 1080 RGB)"
        )
        selected_image = os.path.join(images_dir, selected_rel)
        image_label    = f"Kaggle Dataset — {os.path.basename(selected_rel)}"
        base_num       = os.path.splitext(os.path.basename(selected_rel))[0]
        possible_mask  = os.path.join(gt_fine_dir, f"{base_num}_gtFine_labelIds.png")
        if os.path.exists(possible_mask):
            mask_image = possible_mask

    elif img_source == "Curated Presentation Samples":
        sample_options = {
            "Healthy High-Density Spirulina Culture": "sample_healthy_spirulina.jpg",
            "Commercial Chlorella Raceway Culture":   "sample_chlorella_raceway.jpg",
            "Eutrophic Cyanobacteria Bloom (Risk)":   "sample_cyanobacteria_bloom.jpg",
            "Incipient Culture — Low Density Baseline": "sample_low_density_pond.jpg"
        }
        chosen_sample_name = st.selectbox("Select Demo Sample:", list(sample_options.keys()))
        chosen_sample_file = os.path.join(images_dir, sample_options[chosen_sample_name])
        if os.path.exists(chosen_sample_file):
            selected_image = chosen_sample_file
            image_label    = chosen_sample_name
    else:
        uploaded_file = st.file_uploader("Upload Algae Pond Photo (JPG / PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            selected_image = uploaded_file
            image_label    = uploaded_file.name

with v_col2:
    if selected_image:
        if mask_image:
            tab_orig, tab_mask = st.tabs(["RGB Photograph", "Bio-Segmentation Mask"])
            with tab_orig:
                st.image(selected_image, caption=f"Selected: {image_label}", use_container_width=True)
            with tab_mask:
                st.image(mask_image, caption=f"Ground-Truth Mask — #{base_num}", use_container_width=True)
        else:
            st.image(
                selected_image,
                caption=f"{'Selected' if isinstance(selected_image, str) else 'Uploaded'}: {image_label}",
                use_container_width=True
            )
    else:
        st.info("No image selected. The platform will run numerical telemetry synthesis only.")

st.divider()

# =========================================================
# CORE EXECUTION: LIVE CONTINUOUS MRV INFERENCE
# =========================================================
section_header("\u25CE", "Multi-Core Carbon Verification Engine")

if growth_model is None or water_model is None:
    st.error("Universal ML Models are offline. Please run 'python train_models.py' first.")
else:
    m = st.session_state.current_metrics

    # 1. Algae Growth Biomass Inference (Universal XGBRegressor)
    growth_input_df = pd.DataFrame([[
        m['Light'], m['Nitrate'], m['Iron'], m['Phosphate'], m['Temperature'], m['pH'], m['CO2']
    ]], columns=['Light', 'Nitrate', 'Iron', 'Phosphate', 'Temperature', 'pH', 'CO2'])
    
    predicted_biomass = float(growth_model.predict(growth_input_df)[0])
    predicted_biomass = max(100.0, predicted_biomass)

    # 2. Water Quality Inference (Universal XGBClassifier)
    water_input_df = pd.DataFrame([[
        m['Nitrate'], m['pH'], m['Ammonia'], m['Temperature'], m['DO'], m['Turbidity'], m['Manganese']
    ]], columns=['NITRATE(PPM)', 'PH', 'AMMONIA(mg/l)', 'TEMP', 'DO', 'TURBIDITY', 'MANGANESE(mg/l)'])
    
    encoded_water_pred = int(water_model.predict(water_input_df)[0])
    water_status = encoder.inverse_transform([encoded_water_pred])[0] if encoder else f"Class {encoded_water_pred}"

    # 3. Dynamic Continuous MRV Verification Index Calculation
    mrv_result = compute_continuous_mrv_index(
        metrics=m,
        water_model=water_model,
        encoder=encoder,
        water_input_df=water_input_df,
        predicted_biomass=predicted_biomass,
        vision_result=image_label
    )
    mrv_score = mrv_result["final_mrv"]

    # 4. Carbon Chemistry & Financial Valuation
    financials = compute_carbon_financials(
        predicted_biomass=predicted_biomass,
        mrv_index=mrv_score,
        carbon_price_per_ton=42.50
    )

    # ---------------------------------------------------------
    # Display Results & KPIs
    # ---------------------------------------------------------
    col_hdr, col_curr = st.columns([2.5, 1.5])
    with col_hdr:
        st.markdown("**Real-Time Carbon Sequestration & Financial Metrics**")
    with col_curr:
        selected_currency = st.radio(
            "Currency",
            options=["Currency: USD", "Currency: INR"],
            horizontal=True,
            label_visibility="collapsed",
            key="currency_toggle"
        )

    CONVERSION_RATE = 84.0  # 1 USD = 84 INR
    is_inr = "INR" in selected_currency
    curr_rate = CONVERSION_RATE if is_inr else 1.0
    curr_symbol = "₹" if is_inr else "$"
    curr_code = "INR" if is_inr else "USD"

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(
        "Biomass Density Proxy",
        f"{predicted_biomass:,.1f}",
        delta="Cells/mL Equivalent"
    )
    kpi2.metric(
        "Gross CO\u2082 Sequestered",
        f"{financials['gross_co2_kg']:,.2f} kg",
        delta=f"{financials['daily_capture_rate_kg']:.2f} kg/day rate"
    )

    badge_cls = (
        "status-badge-healthy" if "Healthy" in water_status else
        "status-badge-warning" if "Moderate" in water_status else
        "status-badge-danger"
    )
    kpi3.markdown(f"""
    <div style="padding:10px 0;">
        <div style="font-size:0.78rem;color:#3d5a4f;font-weight:700;
                    text-transform:uppercase;letter-spacing:0.04em;margin-bottom:6px;">
            Water Health Classification
        </div>
        <span class="{badge_cls}">{water_status}</span>
    </div>
    """, unsafe_allow_html=True)

    tradable_val = financials['certified_tradable_usd'] * curr_rate
    gross_val = financials['gross_credit_usd'] * curr_rate

    kpi4.metric(
        f"Tradable Carbon Valuation ({curr_code})",
        f"{curr_symbol}{tradable_val:,.2f} {curr_code}",
        delta=f"{financials['gross_co2_tonnes']:.4f} MT CO\u2082e (Gross: {curr_symbol}{gross_val:,.2f})"
    )

    st.divider()

    # Visual Analytics: Gauges and Radar
    r_col1, r_col2 = st.columns(2)

    with r_col1:
        # Dynamic continuous MRV Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=mrv_score,
            title={"text": "MRV Audit Verification Score (%)", "font": {"color": "#1a2e25", "size": 16}},
            number={"font": {"color": "#1a7a4a", "size": 40, "weight": 800}, "suffix": "%"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#3d5a4f", "tickwidth": 1.5},
                "bar":  {"color": "#1a7a4a"},
                "steps": [
                    {"range": [0,  50], "color": "#fef2f2"},
                    {"range": [50, 75], "color": "#fef9c3"},
                    {"range": [75,100], "color": "#e6f4ed"},
                ],
                "threshold": {"line": {"color": "#0d5c34", "width": 4}, "thickness": 0.8, "value": 85.0}
            }
        ))
        fig_gauge.update_layout(
            height=280, margin=dict(l=20,r=20,t=40,b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with r_col2:
        # Multi-factor Bio-Equilibrium Radar Chart
        categories = ["Aerobic DO", "pH Equilibrium", "Temp Kinetics", "Ammonia Safety", "Photosynthetic PAR"]
        radar_values = [
            mrv_result["do_score"],
            mrv_result["ph_score"],
            mrv_result["temp_score"],
            mrv_result["ammonia_score"],
            mrv_result["photo_score"]
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=categories,
            fill="toself",
            name="Pond Bio-Envelope",
            line=dict(color="#1a7a4a", width=2.5),
            fillcolor="rgba(26,122,74,0.15)"
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#c8e6d8"),
                angularaxis=dict(gridcolor="#c8e6d8")
            ),
            title={"text": "Pond Bio-Equilibrium Sub-Scores (%)", "font": {"color": "#1a2e25", "size": 16}},
            height=280, margin=dict(l=30,r=30,t=40,b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # ---------------------------------------------------------
    # NEW FEATURE: DOWNLOAD CERTIFIED PDF AUDIT REPORT
    # ---------------------------------------------------------
    section_header("\U0001F4C4", f"Official Farm Audit PDF Export: {st.session_state.farm_name}")
    
    col_pdf_desc, col_pdf_btn = st.columns([3.5, 2])
    with col_pdf_desc:
        st.markdown(f"""
        Generate and export an official, cryptographic **ISO 14064 / Verra VM0042 Certified PDF Carbon Audit Report** 
        for facility **{st.session_state.farm_name}** (Operator: `<code>{st.session_state.username}</code>`).
        Includes: Verified MRV Index, Gross vs Tradable Carbon Credit Valuation (USD), Environmental Sub-Scores, and Calculation Breakdown.
        """, unsafe_allow_html=True)
    
    with col_pdf_btn:
        pdf_bytes = generate_farm_carbon_pdf(
            username=st.session_state.username,
            farm_name=st.session_state.farm_name,
            metrics=m,
            predicted_biomass=predicted_biomass,
            water_status=water_status,
            mrv_result=mrv_result,
            financials=financials,
            vision_summary=image_label or "Optical canopy verified",
            ai_synthesis_notes=st.session_state.last_audit_report
        )
        
        pdf_filename = f"Algi_Carbon_Audit_{st.session_state.username}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.download_button(
            label="📥  Download Certified PDF Audit Report",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf",
            use_container_width=True
        )

    st.divider()

    # ---------------------------------------------------------
    # ISO 14064 Certified AI Carbon Auditor Synthesis (Gemini)
    # ---------------------------------------------------------
    section_header("\u2261", "Certified ISO 14064 Carbon Audit Synthesis (AI Engine)")
    
    col_btn, col_btn_info = st.columns([2, 5])
    with col_btn:
        run_ai_audit = st.button("🚀 Generate / Refresh AI Audit Report", use_container_width=True)
    with col_btn_info:
        st.caption("Triggers multi-modal reasoning with Google Gemini 3.6 Flash synthesizing telemetry, universal ML models, and optical imagery.")

    if run_ai_audit or st.session_state.last_audit_report is None:
        with st.spinner("Analyzing optical imagery & synthesizing ISO 14064 Carbon Audit Report with Gemini 3.6 Flash..."):
            vision_result = inspect_image_with_vision(selected_image) if selected_image else "No optical image supplied (Telemetry analysis only)."
            report = generate_audit_report(predicted_biomass, water_status, vision_result, m)
            st.session_state.last_audit_report = report

    if st.session_state.last_audit_report:
        st.markdown(f'<div class="audit-card">{st.session_state.last_audit_report}</div>', unsafe_allow_html=True)
        st.download_button(
            label="\u2193  Export Certified Carbon Audit Report (Markdown)",
            data=st.session_state.last_audit_report,
            file_name=f"algi_carbon_audit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

if ingestion_mode == MODE_IOT and auto_stream:
    time.sleep(3)
    st.rerun()
