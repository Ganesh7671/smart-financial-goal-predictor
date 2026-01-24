import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Load models and preprocessor
xgb_model = joblib.load("model_xgb.pkl")
ann_model = load_model("model_ann.h5")
preprocessor = joblib.load("preprocessor.pkl")

st.set_page_config(page_title="Smart Financial Recommender, Spending Analyzer & Goal Predictor", layout="centered")

# ========================================================================
# SPA STATE MANAGEMENT - Initialize session_state variables
# ========================================================================
if 'current_view' not in st.session_state:
    st.session_state['current_view'] = 'home'  # Default view
if 'form_submitted' not in st.session_state:
    st.session_state['form_submitted'] = False
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = {}
# Simulator slider values persistence
if 'sim_entertainment_val' not in st.session_state:
    st.session_state['sim_entertainment_val'] = 0
if 'sim_eating_out_val' not in st.session_state:
    st.session_state['sim_eating_out_val'] = 0
# ========================================================================
# GOAL PLANNER STATE - Persist goal inputs across navigation
# ========================================================================
if 'goal_name' not in st.session_state:
    st.session_state['goal_name'] = ''
if 'goal_amount' not in st.session_state:
    st.session_state['goal_amount'] = 0
# ========================================================================
# THEME STATE - Persist theme preference across navigation
# ========================================================================
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'light'  # Default to light theme

# Get current theme
is_dark_theme = st.session_state['theme'] == 'dark'

# ========================================================================
# UI VISUAL ENHANCEMENT - Professional FinTech-style styling with Theme Support
# Only CSS changes - NO backend modifications
# ========================================================================

# ========================================================================
# THEME-AWARE CSS - Light and Dark theme with proper text contrast
# ========================================================================
if is_dark_theme:
    # ===== DARK THEME CSS =====
    theme_css = """
    <style>
    /* ===== DARK THEME COLOR PALETTE ===== */
    :root {
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-card: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        --accent-blue: #3b82f6;
        --accent-blue-light: #60a5fa;
        --success-green: #22c55e;
        --warning-amber: #fbbf24;
        --danger-red: #f87171;
        --border-color: #334155;
    }
    
    /* ===== GLOBAL BACKGROUND - DARK ===== */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
    }
    
    /* ===== MAIN CONTENT TEXT VISIBILITY FIX - DARK ===== */
    .stApp, .stApp p, .stApp span, .stApp div {
        color: #f1f5f9 !important;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown span {
        color: #f1f5f9 !important;
    }
    
    /* ===== HEADINGS - DARK THEME ===== */
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }
    h4, h5, h6 {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    
    /* ===== SIDEBAR - DARK ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        border-right: 1px solid #334155;
    }
    
    /* ALL SIDEBAR TEXT - WHITE */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown span,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stButton>button {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        color: #ffffff !important;
    }
    /* Sidebar alerts text */
    [data-testid="stSidebar"] .stSuccess p,
    [data-testid="stSidebar"] .stWarning p,
    [data-testid="stSidebar"] .stAlert p {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] p {
        color: #ffffff !important;
    }

    /* ===== BUTTONS - DARK ===== */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: #ffffff !important;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.4);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 6px 12px -2px rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
    }
    
    /* ===== CARDS & CONTAINERS - DARK ===== */
    .recommend-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        border-left: 4px solid #3b82f6;
        color: #f1f5f9 !important;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .recommend-card h4, .recommend-card p, .recommend-card li {
        color: #f1f5f9 !important;
    }
    
    /* ===== VIEW HEADER - DARK ===== */
    .view-header {
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.2) 0%, transparent 100%) !important;
        border-bottom: 3px solid #3b82f6;
        padding: 16px;
        border-radius: 4px 4px 0 0;
    }
    .view-header h2 {
        color: #f1f5f9 !important;
    }
    
    /* ===== FORM - DARK ===== */
    [data-testid="stForm"] {
        background: #1e293b !important;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 28px;
    }
    
    /* ===== INPUT LABELS - DARK ===== */
    .stNumberInput label, .stTextInput label, .stSelectbox label, .stSlider label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    .stNumberInput label p, [data-testid="stForm"] label p {
        color: #e2e8f0 !important;
    }
    
    /* ===== INPUT FIELDS - DARK ===== */
    .stNumberInput input, .stTextInput input {
        background: #0f172a !important;
        border: 2px solid #334155 !important;
        color: #f1f5f9 !important;
        border-radius: 8px;
    }
    .stNumberInput input:focus {
        border-color: #3b82f6 !important;
    }
    
    /* ===== METRICS - DARK ===== */
    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
    [data-testid="stMetricDelta"] {
        font-weight: 600;
    }
    
    /* ===== TABS - DARK ===== */
    .stTabs [data-baseweb="tab-list"] {
        background: #0f172a;
        border-radius: 12px;
        padding: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa !important;
    }
    .stTabs [aria-selected="true"] {
        background: #3b82f6 !important;
        color: #ffffff !important;
    }
    
    /* ===== ALERTS - DARK ===== */
    .stSuccess {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(34, 197, 94, 0.1) 100%) !important;
        border-left: 4px solid #22c55e;
    }
    .stSuccess p { color: #86efac !important; }
    .stWarning {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.2) 0%, rgba(251, 191, 36, 0.1) 100%) !important;
        border-left: 4px solid #fbbf24;
    }
    .stWarning p { color: #fde047 !important; }
    .stError {
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.2) 0%, rgba(248, 113, 113, 0.1) 100%) !important;
        border-left: 4px solid #f87171;
    }
    .stError p { color: #fca5a5 !important; }
    .stInfo {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(59, 130, 246, 0.1) 100%) !important;
        border-left: 4px solid #3b82f6;
    }
    .stInfo p { color: #93c5fd !important; }
    
    /* ===== PROGRESS BAR - DARK ===== */
    .stProgress > div {
        background: #334155;
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #22c55e 100%);
    }
    
    /* ===== FOOTER - DARK ===== */
    .footer {
        color: #64748b !important;
        text-align: center;
        font-size: 13px;
        padding: 16px;
    }
    .team-credit {
        color: #64748b !important;
        text-align: center;
        font-size: 12px;
        opacity: 0.9;
    }
    
    /* ===== DIVIDER - DARK ===== */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #334155, transparent);
    }
    
    /* ===== SCROLLBAR - DARK ===== */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #1e293b; }
    ::-webkit-scrollbar-thumb { background: #475569; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #64748b; }
    
    /* ===== TOP HEADER BAR - DARK THEME ===== */
    /* White header with subtle shadow for clean separation */
    header[data-testid="stHeader"] {
        background: #ffffff !important;
        border-bottom: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    header[data-testid="stHeader"] * {
        color: #1e293b !important;
    }
    
    /* ===== DEPLOY BUTTON/CONTAINER - DARK THEME ===== */
    /* Styled with deep teal accent for professional FinTech look */
    [data-testid="stHeader"] [data-testid="stToolbar"] {
        background: transparent !important;
    }
    [data-testid="stHeader"] button,
    [data-testid="stHeader"] [data-testid="baseButton-header"] {
        background: #0d9488 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 6px 12px !important;
    }
    [data-testid="stHeader"] button:hover,
    [data-testid="stHeader"] [data-testid="baseButton-header"]:hover {
        background: #0f766e !important;
    }
    /* Deploy menu styling */
    [data-testid="stDeployButton"] {
        background: #0d9488 !important;
        color: #ffffff !important;
        border-radius: 6px !important;
    }
    [data-testid="stDeployButton"] span {
        color: #ffffff !important;
    }
    </style>
    """
else:
    # ===== LIGHT THEME CSS =====
    theme_css = """
    <style>
    /* ===== LIGHT THEME COLOR PALETTE ===== */
    :root {
        --bg-primary: #f8fafc;
        --bg-secondary: #e2e8f0;
        --bg-card: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #475569;
        --text-muted: #64748b;
        --accent-blue: #2563eb;
        --accent-blue-dark: #1a3a5c;
        --success-green: #10b981;
        --warning-amber: #f59e0b;
        --danger-red: #ef4444;
        --border-color: #e2e8f0;
    }
    
    /* ===== GLOBAL BACKGROUND - LIGHT ===== */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
    }
    
    /* ===== MAIN CONTENT TEXT VISIBILITY FIX - LIGHT ===== */
    .stApp p, .stApp span, .stApp div, .stApp li {
        color: #1e293b;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown span {
        color: #1e293b !important;
    }
    
    /* ===== HEADINGS - LIGHT THEME ===== */
    h1, h2, h3 {
        color: #1a3a5c !important;
        font-weight: 700 !important;
    }
    h4, h5, h6 {
        color: #334155 !important;
        font-weight: 600 !important;
    }
    
    /* ===== SIDEBAR - LIGHT ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3a5c 0%, #0f2744 100%) !important;
        border-right: 1px solid #2563eb;
    }
    
    /* ALL SIDEBAR TEXT - WHITE */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown span,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stButton>button {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        color: #ffffff !important;
    }
    /* Sidebar alerts text */
    [data-testid="stSidebar"] .stSuccess p,
    [data-testid="stSidebar"] .stWarning p,
    [data-testid="stSidebar"] .stAlert p {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] p {
        color: #ffffff !important;
    }
    
    /* ===== BUTTONS - LIGHT ===== */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: #ffffff !important;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 6px 12px -2px rgba(37, 99, 235, 0.4);
        transform: translateY(-2px);
    }
    
    /* ===== CARDS & CONTAINERS - LIGHT ===== */
    .recommend-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
        border-left: 4px solid #2563eb;
        color: #1e293b !important;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .recommend-card h4 {
        color: #1a3a5c !important;
    }
    .recommend-card p, .recommend-card li {
        color: #334155 !important;
    }
    
    /* ===== VIEW HEADER - LIGHT ===== */
    .view-header {
        background: linear-gradient(90deg, rgba(37, 99, 235, 0.1) 0%, transparent 100%) !important;
        border-bottom: 3px solid #2563eb;
        padding: 16px;
        border-radius: 4px 4px 0 0;
    }
    .view-header h2 {
        color: #1a3a5c !important;
    }
    
    /* ===== FORM - LIGHT ===== */
    [data-testid="stForm"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 28px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* ===== INPUT LABELS - LIGHT ===== */
    .stNumberInput label, .stTextInput label, .stSelectbox label, .stSlider label {
        color: #1a3a5c !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    .stNumberInput label p, [data-testid="stForm"] label p {
        color: #1a3a5c !important;
        font-weight: 600 !important;
    }
    
    /* ===== INPUT FIELDS - LIGHT ===== */
    .stNumberInput input, .stTextInput input {
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        color: #1e293b !important;
        border-radius: 8px;
    }
    .stNumberInput input:focus {
        border-color: #2563eb !important;
    }
    
    /* ===== METRICS - LIGHT ===== */
    [data-testid="stMetricValue"] {
        color: #1a3a5c !important;
        font-weight: 700;
        font-size: 32px;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 500;
    }
    [data-testid="stMetricDelta"] {
        font-weight: 600;
    }
    
    /* ===== TABS - LIGHT ===== */
    .stTabs [data-baseweb="tab-list"] {
        background: #f1f5f9;
        border-radius: 12px;
        padding: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #64748b !important;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(37, 99, 235, 0.1);
        color: #2563eb !important;
    }
    .stTabs [aria-selected="true"] {
        background: #2563eb !important;
        color: #ffffff !important;
    }
    
    /* ===== ALERTS - LIGHT ===== */
    .stSuccess {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%) !important;
        border-left: 4px solid #10b981;
    }
    .stSuccess p { color: #065f46 !important; }
    .stWarning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%) !important;
        border-left: 4px solid #f59e0b;
    }
    .stWarning p { color: #92400e !important; }
    .stError {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%) !important;
        border-left: 4px solid #ef4444;
    }
    .stError p { color: #991b1b !important; }
    .stInfo {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%) !important;
        border-left: 4px solid #2563eb;
    }
    .stInfo p { color: #1e40af !important; }
    
    /* ===== PROGRESS BAR - LIGHT ===== */
    .stProgress > div {
        background: #e2e8f0;
        border-radius: 10px;
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #2563eb 0%, #10b981 100%);
        border-radius: 10px;
    }
    
    /* ===== FOOTER - LIGHT ===== */
    .footer {
        color: #64748b !important;
        text-align: center;
        font-size: 13px;
        padding: 16px;
    }
    .team-credit {
        color: #64748b !important;
        text-align: center;
        font-size: 12px;
        opacity: 0.8;
    }
    
    /* ===== DIVIDER - LIGHT ===== */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
    }
    
    /* ===== SCROLLBAR - LIGHT ===== */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
    </style>
    """

st.markdown(theme_css, unsafe_allow_html=True)

st.title("Smart Financial Recommender, Spending Analyzer & Goal Predictor")
st.markdown("A smart way to analyze expenses, predict savings, and achieve financial goals.")

# ========================================================================
# SPA NAVIGATION - Sidebar menu for view switching (no page reloads)
# ========================================================================
st.sidebar.markdown("## Navigation")

# ========================================================================
# THEME TOGGLE - Light/Dark mode switch (persists via session_state)
# ========================================================================
st.sidebar.markdown("---")
theme_label = "Dark Mode" if not is_dark_theme else "Light Mode"
theme_icon = "🌙" if not is_dark_theme else "☀️"
if st.sidebar.button(f"{theme_icon} Switch to {theme_label}", key="theme_toggle", use_container_width=True):
    st.session_state['theme'] = 'dark' if st.session_state['theme'] == 'light' else 'light'
    st.rerun()

current_theme_display = "🌙 Dark" if is_dark_theme else "☀️ Light"
st.sidebar.caption(f"Current Theme: {current_theme_display}")
st.sidebar.markdown("---")

# Navigation buttons - update session_state without triggering full page reload
def set_view(view_name):
    st.session_state['current_view'] = view_name

# ========================================================================
# PAGE ORDER FOR NEXT/BACK NAVIGATION
# ========================================================================
PAGE_ORDER = ['home', 'dashboard', 'goal_planner', 'simulator', 'results']

def get_next_page(current):
    """Get next page in sequence"""
    idx = PAGE_ORDER.index(current) if current in PAGE_ORDER else 0
    if idx < len(PAGE_ORDER) - 1:
        return PAGE_ORDER[idx + 1]
    return None

def get_prev_page(current):
    """Get previous page in sequence"""
    idx = PAGE_ORDER.index(current) if current in PAGE_ORDER else 0
    if idx > 0:
        return PAGE_ORDER[idx - 1]
    return None

def render_nav_buttons():
    """Render Back/Next navigation buttons at bottom of each page"""
    current = st.session_state['current_view']
    prev_page = get_prev_page(current)
    next_page = get_next_page(current)
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if prev_page:
            if st.button("< Back", key=f"back_{current}", use_container_width=True):
                set_view(prev_page)
                st.rerun()
    
    with col3:
        if next_page:
            if st.button("Next >", key=f"next_{current}", type="primary", use_container_width=True):
                set_view(next_page)
                st.rerun()

# Navigation menu
nav_options = {
    'home': ('Home / Input', 'home'),
    'dashboard': ('Dashboard / Analysis', 'dashboard'),
    'goal_planner': ('Goal Planner', 'goal_planner'),
    'simulator': ('What-If Simulator', 'simulator'),
    'results': ('Summary / Results', 'results')
}

for key, (label, view) in nav_options.items():
    is_active = st.session_state['current_view'] == view
    button_type = "primary" if is_active else "secondary"
    if st.sidebar.button(label, key=f"nav_{key}", type=button_type, use_container_width=True):
        set_view(view)
        st.rerun()

# Show current view indicator
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current View:** {nav_options.get(st.session_state['current_view'], ('Unknown',))[0]}")

# Show data status
if st.session_state['form_submitted']:
    st.sidebar.success("Data loaded")
else:
    st.sidebar.warning("Enter data on Home page")

# ========================================================================
# LABELS - Used across views
# ========================================================================
labels = [
    "Income", "Age", "Dependents", "Rent", "Loan_Repayment", "Insurance", "Groceries",
    "Transport", "Eating_Out", "Entertainment", "Utilities", "Healthcare", "Education", "Miscellaneous"
]

# ========================================================================
# HELPER FUNCTION - Calculate all derived values (reusable across views)
# Backend logic remains UNCHANGED - only called when needed
# ========================================================================
def calculate_derived_values(user_input):
    """Calculate all derived values from user input - NO backend changes"""
    total_expense = sum(user_input[label] for label in labels[3:])
    disposable_income = user_input["Income"] - total_expense
    desired_savings = disposable_income * 0.3
    desired_savings_percentage = (desired_savings / user_input["Income"]) * 100 if user_input["Income"] > 0 else 0
    
    # Needs vs Wants Categorization
    needs_categories = ["Rent", "Groceries", "Utilities", "Healthcare", "Education", "Insurance", "Transport"]
    wants_categories = ["Entertainment", "Eating_Out", "Miscellaneous"]
    
    needs_total = sum(user_input[cat] for cat in needs_categories)
    wants_total = sum(user_input[cat] for cat in wants_categories)
    
    needs_percentage = (needs_total / user_input["Income"]) * 100 if user_input["Income"] > 0 else 0
    wants_percentage = (wants_total / user_input["Income"]) * 100 if user_input["Income"] > 0 else 0
    savings_percentage = (disposable_income / user_input["Income"]) * 100 if user_input["Income"] > 0 else 0
    
    # Financial Health Score Calculation
    savings_score = min(40, (savings_percentage / 30) * 40) if savings_percentage > 0 else 0
    wants_score = max(0, 25 - (wants_percentage / 30) * 25) if wants_percentage <= 50 else 0
    loan_ratio = (user_input["Loan_Repayment"] / user_input["Income"]) * 100 if user_input["Income"] > 0 else 0
    loan_score = max(0, 20 - (loan_ratio / 36) * 20)
    disposable_ratio = (disposable_income / user_input["Income"]) * 100 if user_input["Income"] > 0 else 0
    disposable_score = min(15, (disposable_ratio / 50) * 15) if disposable_ratio > 0 else 0
    
    financial_health_score = round(savings_score + wants_score + loan_score + disposable_score)
    financial_health_score = max(0, min(100, financial_health_score))
    
    return {
        'total_expense': total_expense,
        'disposable_income': disposable_income,
        'desired_savings': desired_savings,
        'desired_savings_percentage': desired_savings_percentage,
        'needs_categories': needs_categories,
        'wants_categories': wants_categories,
        'needs_total': needs_total,
        'wants_total': wants_total,
        'needs_percentage': needs_percentage,
        'wants_percentage': wants_percentage,
        'savings_percentage': savings_percentage,
        'savings_score': savings_score,
        'wants_score': wants_score,
        'loan_ratio': loan_ratio,
        'loan_score': loan_score,
        'disposable_score': disposable_score,
        'financial_health_score': financial_health_score
    }

# ========================================================================
# VIEW 1: HOME / INPUT - Data entry form
# ========================================================================
if st.session_state['current_view'] == 'home':
    st.markdown("<div class='view-header'><h2>Enter Your Financial Details</h2></div>", unsafe_allow_html=True)
    
    with st.form("input_form"):
        cols = st.columns(2)
        user_input = {}
        for i, label in enumerate(labels):
            with cols[i % 2]:
                # Use stored values if available
                default_val = st.session_state['user_input'].get(label, 0) if st.session_state['form_submitted'] else 0
                user_input[label] = st.number_input(f"{label}", min_value=0, step=100, format="%d", value=default_val)
        submitted = st.form_submit_button("Submit & Analyze", type="primary")
    
    if submitted:
        # Store user input in session_state
        st.session_state['user_input'] = user_input.copy()
        st.session_state['form_submitted'] = True
        # Reset simulator values when new data is submitted
        st.session_state['sim_entertainment_val'] = 0
        st.session_state['sim_eating_out_val'] = 0
        # Navigate to dashboard after submission
        st.session_state['current_view'] = 'dashboard'
        st.rerun()
    
    # Show preview if data exists
    if st.session_state['form_submitted']:
        st.markdown("---")
        st.success("Data already entered. Use navigation to explore analysis or update values above.")
    
    # Navigation buttons
    render_nav_buttons()

# ========================================================================
# VIEW 2: DASHBOARD / ANALYSIS - Visualizations, Predictions, Health Score
# ========================================================================
elif st.session_state['current_view'] == 'dashboard':
    if not st.session_state['form_submitted']:
        st.warning("Please enter your financial details on the Home page first.")
        if st.button("Go to Home"):
            set_view('home')
            st.rerun()
    else:
        user_input = st.session_state['user_input']
        derived = calculate_derived_values(user_input)
        
        # ML Predictions - Backend logic UNCHANGED
        input_df = pd.DataFrame([user_input])
        input_df["Disposable_Income"] = derived['disposable_income']
        input_df["Desired_Savings"] = derived['desired_savings']
        input_df["Desired_Savings_Percentage"] = derived['desired_savings_percentage']
        input_df = input_df[preprocessor.feature_names_in_]
        input_processed = preprocessor.transform(input_df)
        xgb_pred = xgb_model.predict(input_processed)[0]
        ann_pred = np.argmax(ann_model.predict(input_processed), axis=1)[0]
        label_map = {0: "Heavy Spender", 1: "Moderate Spender", 2: "Saver"}
        final_label = label_map[xgb_pred]
        
        st.markdown("<div class='view-header'><h2>Dashboard & Analysis</h2></div>", unsafe_allow_html=True)
        
        # Sub-tabs within dashboard view
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "Visualizations", "Predictions", "Health Score", 
            "Needs vs Wants", "Overspending", "Emergency Fund", "Financial Persona", "Recommendations"
        ])
        
        with tab1:
            st.markdown("### Expense Distribution")
            fig1, ax1 = plt.subplots(figsize=(4.5, 4.5))
            pie_data = [user_input[l] for l in labels[3:]]
            ax1.pie(pie_data, labels=labels[3:], autopct="%1.1f%%", startangle=90)
            ax1.axis("equal")
            st.pyplot(fig1)

            st.markdown("### Category Breakdown")
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            sns.barplot(x=labels[3:], y=pie_data, palette="coolwarm", ax=ax2)
            ax2.set_ylabel("Amount")
            plt.xticks(rotation=45)
            st.pyplot(fig2)

        with tab2:
            st.success(f"### Predicted Category: **{final_label}**")
            st.metric("Disposable Income", f"{derived['disposable_income']:,.0f}")
            st.metric("Desired Savings", f"{derived['desired_savings']:,.0f}")
            st.metric("Desired Savings %", f"{derived['desired_savings_percentage']:.2f}%")

        with tab3:
            st.markdown("### Financial Health Score")
            fhs = derived['financial_health_score']
            if fhs >= 80:
                status_label, status_color, status_emoji = "Excellent", "#28a745", "Green"
            elif fhs >= 60:
                status_label, status_color, status_emoji = "Good", "#ffc107", "Yellow"
            else:
                status_label, status_color, status_emoji = "Needs Improvement", "#dc3545", "Red"
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; border: 3px solid {status_color};">
                    <h1 style="font-size: 72px; color: {status_color}; margin: 0;">{fhs}</h1>
                    <p style="font-size: 24px; margin: 5px 0;">{status_label}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("#### Score Breakdown")
                st.markdown(f"- **Savings Component:** {derived['savings_score']:.1f}/40")
                st.markdown(f"- **Wants Control:** {derived['wants_score']:.1f}/25")
                st.markdown(f"- **Loan Burden:** {derived['loan_score']:.1f}/20")
                st.markdown(f"- **Disposable Income:** {derived['disposable_score']:.1f}/15")
            
            st.progress(fhs / 100)

        with tab4:
            st.markdown("### Needs vs Wants Analysis")
            col1, col2 = st.columns([1, 1])
            with col1:
                fig_donut, ax_donut = plt.subplots(figsize=(5, 5))
                donut_data = [derived['needs_total'], derived['wants_total'], max(0, derived['disposable_income'])]
                donut_labels = [f"Needs\n{derived['needs_total']:,.0f}", f"Wants\n{derived['wants_total']:,.0f}", f"Savings\n{max(0, derived['disposable_income']):,.0f}"]
                donut_colors = ["#3498db", "#e74c3c", "#2ecc71"]
                wedges, texts, autotexts = ax_donut.pie(donut_data, labels=donut_labels, autopct="%1.1f%%", colors=donut_colors, startangle=90, pctdistance=0.75)
                centre_circle = plt.Circle((0, 0), 0.50, fc='white')
                ax_donut.add_patch(centre_circle)
                ax_donut.axis("equal")
                st.pyplot(fig_donut)
            with col2:
                st.metric("Needs", f"{derived['needs_percentage']:.1f}%", delta="Target: <=50%")
                st.metric("Wants", f"{derived['wants_percentage']:.1f}%", delta="Target: <=30%")
                st.metric("Savings", f"{derived['savings_percentage']:.1f}%", delta="Target: >=20%")

        with tab5:
            st.markdown("### Overspending Detector")
            overspending_thresholds = {"Rent": 30, "Entertainment": 10, "Eating_Out": 10, "Wants_Total": 30}
            rent_pct = (user_input["Rent"] / user_input["Income"]) * 100 if user_input["Income"] > 0 else 0
            entertainment_pct = (user_input["Entertainment"] / user_input["Income"]) * 100 if user_input["Income"] > 0 else 0
            eating_out_pct = (user_input["Eating_Out"] / user_input["Income"]) * 100 if user_input["Income"] > 0 else 0
            
            categories = ["Rent", "Entertainment", "Eating Out", "Total Wants"]
            actual_vals = [rent_pct, entertainment_pct, eating_out_pct, derived['wants_percentage']]
            threshold_vals = [30, 10, 10, 30]
            
            fig_thresh, ax_thresh = plt.subplots(figsize=(8, 4))
            x = np.arange(len(categories))
            width = 0.35
            bars1 = ax_thresh.bar(x - width/2, actual_vals, width, label='Your Spending', color=['#e74c3c' if a > t else '#2ecc71' for a, t in zip(actual_vals, threshold_vals)])
            bars2 = ax_thresh.bar(x + width/2, threshold_vals, width, label='Threshold', color='#95a5a6', alpha=0.7)
            ax_thresh.set_ylabel('Percentage of Income (%)')
            ax_thresh.set_xticks(x)
            ax_thresh.set_xticklabels(categories)
            ax_thresh.legend()
            st.pyplot(fig_thresh)

        # ========================================================================
        # TAB 6: EMERGENCY FUND READINESS
        # ========================================================================
        with tab6:
            st.markdown("### Emergency Fund Readiness")
            st.markdown("An emergency fund should cover **6 months** of essential expenses.")
            
            # Calculate emergency fund requirements
            monthly_expenses = derived['total_expense']
            required_emergency_fund = monthly_expenses * 6
            current_savings = max(0, derived['disposable_income'])
            
            # Time to build emergency fund
            months_to_emergency_fund = required_emergency_fund / current_savings if current_savings > 0 else float('inf')
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Required Fund", f"{required_emergency_fund:,.0f}")
            with col2:
                st.metric("Monthly Savings", f"{current_savings:,.0f}")
            with col3:
                if current_savings > 0:
                    st.metric("Months to Build", f"{months_to_emergency_fund:.0f}")
                else:
                    st.metric("Months to Build", "N/A")
            
            st.markdown("---")
            
            # Progress visualization
            st.markdown("#### Emergency Fund Progress")
            if current_savings > 0:
                # Assuming starting from zero, show progress based on monthly savings capacity
                progress_ratio = min(1.0, (current_savings * 12) / required_emergency_fund)
                st.progress(progress_ratio)
                
                if months_to_emergency_fund <= 6:
                    st.success(f"**Excellent!** You can build your emergency fund in {months_to_emergency_fund:.0f} months.")
                elif months_to_emergency_fund <= 12:
                    st.info(f"**Good progress!** You can build your emergency fund in {months_to_emergency_fund:.0f} months (under 1 year).")
                elif months_to_emergency_fund <= 24:
                    st.warning(f"**Moderate timeline.** Building your emergency fund will take {months_to_emergency_fund:.0f} months.")
                else:
                    st.error(f"**Long timeline.** At current savings, building your emergency fund will take {months_to_emergency_fund:.0f} months.")
            else:
                st.progress(0.0)
                st.error("**Warning:** You have no savings capacity. Focus on reducing expenses first.")
            
            # Recommendations
            st.markdown("#### Emergency Fund Tips")
            st.markdown(f"""
            - **Target Amount:** {required_emergency_fund:,.0f} (6 months of expenses)
            - **Monthly Expense Base:** {monthly_expenses:,.0f}
            - Keep emergency funds in a **high-yield savings account**
            - Prioritize building this fund before other investments
            - Automate transfers to your emergency fund each month
            """)

        # ========================================================================
        # TAB 7: FINANCIAL PERSONA
        # ========================================================================
        with tab7:
            st.markdown("### Your Financial Persona")
            
            # Determine persona based on multiple factors
            savings_pct = derived['savings_percentage']
            wants_pct = derived['wants_percentage']
            loan_pct = derived['loan_ratio']
            fhs = derived['financial_health_score']
            
            # Persona classification logic
            if loan_pct > 30:
                persona = "Debt-Heavy User"
                persona_color = "#dc3545"
                persona_desc = "Your loan repayments consume a significant portion of your income. Focus on debt reduction strategies."
                persona_tips = [
                    "Prioritize paying off high-interest debt first",
                    "Consider debt consolidation options",
                    "Avoid taking on new debt",
                    "Build a small emergency fund while paying off debt"
                ]
            elif savings_pct >= 30 and wants_pct <= 20:
                persona = "Smart Saver"
                persona_color = "#28a745"
                persona_desc = "You excel at saving and controlling discretionary spending. Keep up the excellent financial habits!"
                persona_tips = [
                    "Consider investing surplus savings",
                    "Explore tax-advantaged accounts",
                    "Diversify your investment portfolio",
                    "Set stretch savings goals"
                ]
            elif savings_pct >= 20 and wants_pct <= 30:
                persona = "Balanced Planner"
                persona_color = "#17a2b8"
                persona_desc = "You maintain a healthy balance between spending and saving. Good financial discipline!"
                persona_tips = [
                    "Continue maintaining your balanced approach",
                    "Look for small areas to optimize",
                    "Consider increasing savings by 2-5%",
                    "Review subscriptions and recurring expenses"
                ]
            elif wants_pct > 30:
                persona = "Lifestyle Spender"
                persona_color = "#ffc107"
                persona_desc = "You tend to spend more on wants and lifestyle. Consider rebalancing for better financial security."
                persona_tips = [
                    "Review and cut unnecessary subscriptions",
                    "Set a monthly entertainment budget",
                    "Use the 24-hour rule for non-essential purchases",
                    "Track daily expenses to identify patterns"
                ]
            else:
                persona = "Moderate Manager"
                persona_color = "#6c757d"
                persona_desc = "You have average financial habits with room for improvement in both savings and spending."
                persona_tips = [
                    "Set specific savings goals",
                    "Create a detailed monthly budget",
                    "Automate savings transfers",
                    "Review expenses weekly"
                ]
            
            # Display persona
            st.markdown(f"""
            <div style="text-align: center; padding: 30px; background-color: #f8f9fa; border-radius: 15px; border: 4px solid {persona_color};">
                <h1 style="font-size: 36px; color: {persona_color}; margin: 0;">{persona}</h1>
                <p style="font-size: 16px; margin: 15px 0; color: #333;">{persona_desc}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Persona breakdown
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Your Profile")
                st.markdown(f"- **Savings Rate:** {savings_pct:.1f}%")
                st.markdown(f"- **Wants Spending:** {wants_pct:.1f}%")
                st.markdown(f"- **Debt Ratio:** {loan_pct:.1f}%")
                st.markdown(f"- **Health Score:** {fhs}/100")
            
            with col2:
                st.markdown("#### Personalized Tips")
                for tip in persona_tips:
                    st.markdown(f"- {tip}")

        # ========================================================================
        # TAB 8: EXPLAINABLE RECOMMENDATIONS
        # ========================================================================
        with tab8:
            st.markdown("### Explainable Recommendations")
            st.markdown("Here's why we're giving you these specific recommendations:")
            
            st.markdown("---")
            
            # Generate explainable recommendations based on actual data
            recommendations = []
            
            # Check Entertainment
            if entertainment_pct > 10:
                recommendations.append({
                    "category": "Entertainment",
                    "status": "Over Budget",
                    "actual": f"{entertainment_pct:.1f}%",
                    "threshold": "10%",
                    "reason": f"Entertainment spending exceeds {entertainment_pct:.1f}% of income, triggering a reduction recommendation.",
                    "action": f"Reduce entertainment by {user_input['Entertainment'] * 0.5:,.0f}/month to save more."
                })
            
            # Check Eating Out
            if eating_out_pct > 10:
                recommendations.append({
                    "category": "Eating Out",
                    "status": "Over Budget",
                    "actual": f"{eating_out_pct:.1f}%",
                    "threshold": "10%",
                    "reason": f"Eating out spending is {eating_out_pct:.1f}% of income, above the recommended 10%.",
                    "action": f"Reduce eating out by {user_input['Eating_Out'] * 0.5:,.0f}/month by cooking at home more."
                })
            
            # Check Rent
            if rent_pct > 30:
                recommendations.append({
                    "category": "Rent",
                    "status": "Over Budget",
                    "actual": f"{rent_pct:.1f}%",
                    "threshold": "30%",
                    "reason": f"Rent consumes {rent_pct:.1f}% of income, above the recommended 30% limit.",
                    "action": "Consider negotiating rent, finding a roommate, or relocating to reduce housing costs."
                })
            
            # Check Total Wants
            if derived['wants_percentage'] > 30:
                recommendations.append({
                    "category": "Total Wants",
                    "status": "Over Budget",
                    "actual": f"{derived['wants_percentage']:.1f}%",
                    "threshold": "30%",
                    "reason": f"Total discretionary spending is {derived['wants_percentage']:.1f}%, exceeding the 30% guideline.",
                    "action": "Review and prioritize wants. Cut non-essential subscriptions and impulse purchases."
                })
            
            # Check Savings
            if derived['savings_percentage'] < 20:
                recommendations.append({
                    "category": "Savings",
                    "status": "Below Target",
                    "actual": f"{derived['savings_percentage']:.1f}%",
                    "threshold": "20%",
                    "reason": f"Savings rate is {derived['savings_percentage']:.1f}%, below the recommended 20% minimum.",
                    "action": "Automate savings and treat it as a non-negotiable expense."
                })
            
            # Check Loan Burden
            if derived['loan_ratio'] > 36:
                recommendations.append({
                    "category": "Loan Repayment",
                    "status": "High Debt",
                    "actual": f"{derived['loan_ratio']:.1f}%",
                    "threshold": "36%",
                    "reason": f"Debt-to-income ratio is {derived['loan_ratio']:.1f}%, above the healthy limit of 36%.",
                    "action": "Focus on debt reduction. Consider debt consolidation or refinancing."
                })
            
            # Display recommendations
            if recommendations:
                for rec in recommendations:
                    status_color = "#dc3545" if rec["status"] in ["Over Budget", "Below Target", "High Debt"] else "#28a745"
                    st.markdown(f"""
                    <div class='recommend-card' style="margin-bottom: 15px; border-left-color: {status_color};">
                        <h4 style="margin: 0;">{rec['category']} - {rec['status']}</h4>
                        <p><strong>Current:</strong> {rec['actual']} | <strong>Target:</strong> {rec['threshold']}</p>
                        <p><strong>Why:</strong> {rec['reason']}</p>
                        <p><strong>Action:</strong> {rec['action']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("Great job! All your spending categories are within recommended limits.")
            
            st.markdown("---")
            
            # Summary advice based on ML prediction
            st.markdown(f"### Overall Assessment: **{final_label}**")
            if xgb_pred == 0:
                st.markdown("""
                As a **Heavy Spender**, focus on:
                - Setting strict budgets for discretionary categories
                - Using cash envelopes or budget apps
                - Reviewing expenses weekly
                """)
            elif xgb_pred == 1:
                st.markdown("""
                As a **Moderate Spender**, focus on:
                - Fine-tuning your budget for small gains
                - Increasing savings rate gradually
                - Building investment habits
                """)
            else:
                st.markdown("""
                As a **Saver**, focus on:
                - Maintaining your excellent habits
                - Exploring investment options
                - Setting ambitious financial goals
                """)
        
        # Navigation buttons for dashboard
        render_nav_buttons()

# ========================================================================
# VIEW 3: FINANCIAL GOAL PLANNER - Frontend/Recommendation Layer Only
# Uses EXISTING calculated values (income, savings, disposable income)
# NO backend changes - only UI and recommendation logic
# ========================================================================
elif st.session_state['current_view'] == 'goal_planner':
    if not st.session_state['form_submitted']:
        st.warning("Please enter your financial details on the Home page first.")
        if st.button("Go to Home"):
            set_view('home')
            st.rerun()
    else:
        user_input = st.session_state['user_input']
        derived = calculate_derived_values(user_input)
        
        st.markdown("<div class='view-header'><h2>Financial Goal Planner</h2></div>", unsafe_allow_html=True)
        st.markdown("Plan your financial goals and get personalized recommendations to achieve them faster.")
        
        st.markdown("---")
        
        # ========================================================================
        # GOAL INPUT SECTION
        # ========================================================================
        st.markdown("### Set Your Goal")
        
        col1, col2 = st.columns(2)
        with col1:
            goal_name = st.text_input(
                "Goal Description",
                value=st.session_state['goal_name'],
                placeholder="e.g., Buy a bike, Emergency fund, Laptop, Trip",
                key="goal_name_input"
            )
            st.session_state['goal_name'] = goal_name
        
        with col2:
            goal_amount = st.number_input(
                "Goal Amount",
                min_value=0,
                step=1000,
                value=st.session_state['goal_amount'],
                format="%d",
                key="goal_amount_input"
            )
            st.session_state['goal_amount'] = goal_amount
        
        st.markdown("---")
        
        # ========================================================================
        # GOAL ANALYSIS - Using EXISTING derived values only
        # ========================================================================
        if goal_name and goal_amount > 0:
            # ========================================================================
            # CORRECT POLICY FORMULA FOR MONTHLY SAVINGS
            # Monthly_Savings = min(Disposable_Income, Income × Recommended_Saving_Percentage)
            # This ensures no fake savings - affordability + policy compliance
            # ========================================================================
            monthly_income = user_input['Income']
            recommended_saving_pct = 0.20  # 20% recommended saving rate
            policy_based_savings = monthly_income * recommended_saving_pct
            monthly_savings = min(derived['disposable_income'], policy_based_savings) if derived['disposable_income'] > 0 else 0
            
            entertainment_expense = user_input['Entertainment']
            eating_out_expense = user_input['Eating_Out']
            
            st.markdown("### Goal Achievement Summary")
            
            # Display goal summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Goal", goal_name)
            with col2:
                st.metric("Target Amount", f"{goal_amount:,.0f}")
            with col3:
                st.metric("Monthly Savings", f"{monthly_savings:,.0f}")
            
            st.markdown("---")
            
            # ========================================================================
            # TIMELINE CALCULATION - Using ceil for real-world safe rounding
            # ========================================================================
            if monthly_savings > 0:
                months_to_goal = math.ceil(goal_amount / monthly_savings)  # Always round up
                years_to_goal = months_to_goal / 12
                
                # Progress visualization
                st.markdown("### Timeline to Achieve Goal")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    # Visual progress indicator
                    if months_to_goal <= 12:
                        progress_pct = min(1.0, 12 / max(1, months_to_goal))
                        st.progress(progress_pct)
                        st.success(f"**Estimated Time:** {months_to_goal:.1f} months ({years_to_goal:.1f} years)")
                    elif months_to_goal <= 24:
                        progress_pct = min(1.0, 24 / max(1, months_to_goal)) * 0.7
                        st.progress(progress_pct)
                        st.warning(f"**Estimated Time:** {months_to_goal:.1f} months ({years_to_goal:.1f} years)")
                    else:
                        progress_pct = min(1.0, 36 / max(1, months_to_goal)) * 0.4
                        st.progress(progress_pct)
                        st.error(f"**Estimated Time:** {months_to_goal:.1f} months ({years_to_goal:.1f} years)")
                
                with col2:
                    st.metric("Months Required", f"{months_to_goal:.0f}")
                
                st.markdown("---")
                
                # ========================================================================
                # SMART RECOMMENDATIONS (Recommendation Layer - Frontend Only)
                # ========================================================================
                st.markdown("### Smart Recommendations")
                
                # Determine if goal is achievable within reasonable time
                is_achievable_fast = months_to_goal <= 12
                is_achievable_medium = months_to_goal <= 24
                
                if is_achievable_fast:
                    # Goal achievable with current savings
                    st.markdown(f"""
                    <div class='recommend-card'>
                        <h4>Great News! Your Goal is On Track</h4>
                        <p>At your current saving rate of <strong>{monthly_savings:,.0f}</strong> per month, 
                        you can achieve your goal "<strong>{goal_name}</strong>" in approximately 
                        <strong>{months_to_goal:.0f} months</strong>.</p>
                        <p><strong>Keep up the good work!</strong> Continue maintaining your current spending habits.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional tips
                    st.info(f"**Tip:** To reach your goal even faster, consider the suggestions below.")
                    
                else:
                    # Goal needs optimization
                    st.markdown(f"""
                    <div class='recommend-card'>
                        <h4>Goal Optimization Needed</h4>
                        <p>At your current saving rate of <strong>{monthly_savings:,.0f}</strong> per month, 
                        achieving "<strong>{goal_name}</strong>" will take <strong>{months_to_goal:.0f} months</strong>.</p>
                        <p>Here are specific actions to reach your goal faster:</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # ========================================================================
                # SPECIFIC ACTIONABLE RECOMMENDATIONS
                # ========================================================================
                st.markdown("#### Action Plan to Achieve Your Goal Faster")
                
                # Calculate potential savings increases
                potential_entertainment_cut = min(entertainment_expense, entertainment_expense * 0.5)
                potential_eating_out_cut = min(eating_out_expense, eating_out_expense * 0.5)
                
                # Scenario 1: Reduce Entertainment
                if entertainment_expense > 0:
                    new_savings_1 = monthly_savings + potential_entertainment_cut
                    new_months_1 = goal_amount / new_savings_1 if new_savings_1 > 0 else float('inf')
                    months_saved_1 = months_to_goal - new_months_1
                    
                    st.markdown(f"""
                    **Option 1: Reduce Entertainment Spending**
                    - Current Entertainment: {entertainment_expense:,.0f}/month
                    - Suggested Reduction: {potential_entertainment_cut:,.0f}/month (50%)
                    - New Monthly Savings: {new_savings_1:,.0f}
                    - New Timeline: **{new_months_1:.0f} months** (saves {months_saved_1:.0f} months)
                    """)
                
                # Scenario 2: Reduce Eating Out
                if eating_out_expense > 0:
                    new_savings_2 = monthly_savings + potential_eating_out_cut
                    new_months_2 = goal_amount / new_savings_2 if new_savings_2 > 0 else float('inf')
                    months_saved_2 = months_to_goal - new_months_2
                    
                    st.markdown(f"""
                    **Option 2: Reduce Eating Out**
                    - Current Eating Out: {eating_out_expense:,.0f}/month
                    - Suggested Reduction: {potential_eating_out_cut:,.0f}/month (50%)
                    - New Monthly Savings: {new_savings_2:,.0f}
                    - New Timeline: **{new_months_2:.0f} months** (saves {months_saved_2:.0f} months)
                    """)
                
                # Scenario 3: Combined reduction
                if entertainment_expense > 0 and eating_out_expense > 0:
                    combined_cut = potential_entertainment_cut + potential_eating_out_cut
                    new_savings_3 = monthly_savings + combined_cut
                    new_months_3 = goal_amount / new_savings_3 if new_savings_3 > 0 else float('inf')
                    months_saved_3 = months_to_goal - new_months_3
                    
                    st.markdown("---")
                    st.markdown(f"""
                    **Recommended: Combined Approach**
                    - Reduce Entertainment by: {potential_entertainment_cut:,.0f}/month
                    - Reduce Eating Out by: {potential_eating_out_cut:,.0f}/month
                    - Total Additional Savings: {combined_cut:,.0f}/month
                    - New Monthly Savings: **{new_savings_3:,.0f}**
                    - New Timeline: **{new_months_3:.0f} months** (saves {months_saved_3:.0f} months!)
                    """)
                    
                    if new_months_3 <= 12:
                        st.success(f"With these changes, you can achieve your goal in under a year!")
                    elif new_months_3 <= 24:
                        st.info(f"With these changes, you can achieve your goal in under 2 years.")
                
                # ========================================================================
                # GOAL PROGRESS VISUALIZATION
                # ========================================================================
                st.markdown("---")
                st.markdown("### Savings Progress Projection")
                
                # Create projection chart
                max_months = min(int(months_to_goal) + 6, 60)  # Show up to 5 years
                months_range = list(range(1, max_months + 1))
                
                # Current path
                current_path = [monthly_savings * m for m in months_range]
                
                # Optimized path (with combined cuts)
                if entertainment_expense > 0 or eating_out_expense > 0:
                    optimized_savings = monthly_savings + potential_entertainment_cut + potential_eating_out_cut
                    optimized_path = [optimized_savings * m for m in months_range]
                else:
                    optimized_path = current_path
                
                fig_goal, ax_goal = plt.subplots(figsize=(10, 5))
                ax_goal.plot(months_range, current_path, marker='', label='Current Savings Path', color='#3498db', linewidth=2)
                ax_goal.plot(months_range, optimized_path, marker='', label='Optimized Path', color='#2ecc71', linewidth=2, linestyle='--')
                ax_goal.axhline(y=goal_amount, color='#e74c3c', linestyle=':', linewidth=2, label=f'Goal: {goal_amount:,.0f}')
                
                ax_goal.fill_between(months_range, current_path, alpha=0.1, color='#3498db')
                ax_goal.fill_between(months_range, optimized_path, alpha=0.1, color='#2ecc71')
                
                ax_goal.set_xlabel('Months')
                ax_goal.set_ylabel('Cumulative Savings')
                ax_goal.set_title(f'Path to "{goal_name}"')
                ax_goal.legend()
                ax_goal.grid(True, alpha=0.3)
                ax_goal.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
                
                st.pyplot(fig_goal)
                
            else:
                # No savings or negative savings
                st.error("""
                **Warning:** Your current monthly savings is zero or negative. 
                You need to reduce expenses before setting financial goals.
                
                **Immediate Actions:**
                - Review your spending in the Dashboard
                - Use the What-If Simulator to find savings opportunities
                - Focus on reducing discretionary spending (Entertainment, Eating Out)
                """)
        
        else:
            # No goal set yet
            st.info("Enter a goal description and amount above to see your personalized plan and recommendations.")
        
        # Navigation buttons for goal planner
        render_nav_buttons()

# ========================================================================
# VIEW 4: WHAT-IF SIMULATOR - Isolated from main form, uses session_state
# Sliders do NOT trigger navigation or form resets
# ========================================================================
elif st.session_state['current_view'] == 'simulator':
    if not st.session_state['form_submitted']:
        st.warning("Please enter your financial details on the Home page first.")
        if st.button("Go to Home"):
            set_view('home')
            st.rerun()
    else:
        user_input = st.session_state['user_input']
        derived = calculate_derived_values(user_input)
        
        st.markdown("<div class='view-header'><h2>What-If Spending Simulator</h2></div>", unsafe_allow_html=True)
        st.markdown("Simulate how reducing expenses impacts your savings. *Changes here don't affect your actual data.*")
        
        st.markdown("---")
        
        # ========================================================================
        # SIMULATOR SLIDERS - Use session_state for persistence
        # These are OUTSIDE any form to prevent rerun issues
        # ========================================================================
        col1, col2 = st.columns(2)
        with col1:
            sim_entertainment_reduction = st.slider(
                "Reduce Entertainment by",
                min_value=0,
                max_value=max(1, user_input["Entertainment"]),
                value=st.session_state['sim_entertainment_val'],
                step=100,
                key="sim_entertainment_slider"
            )
            # Update session_state on change
            st.session_state['sim_entertainment_val'] = sim_entertainment_reduction
            
        with col2:
            sim_eating_out_reduction = st.slider(
                "Reduce Eating Out by",
                min_value=0,
                max_value=max(1, user_input["Eating_Out"]),
                value=st.session_state['sim_eating_out_val'],
                step=100,
                key="sim_eating_out_slider"
            )
            # Update session_state on change
            st.session_state['sim_eating_out_val'] = sim_eating_out_reduction
        
        # Calculate simulated values (Frontend only - NO backend changes)
        total_monthly_reduction = sim_entertainment_reduction + sim_eating_out_reduction
        sim_disposable_income = derived['disposable_income'] + total_monthly_reduction
        sim_savings = sim_disposable_income * 0.3
        sim_savings_percentage = (sim_savings / user_input["Income"]) * 100 if user_input["Income"] > 0 else 0
        sim_total_savings = sim_disposable_income
        
        st.markdown("### Simulated Impact")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            delta_disp = sim_disposable_income - derived['disposable_income']
            st.metric(
                "Disposable Income",
                f"{sim_disposable_income:,.0f}",
                delta=f"+{delta_disp:,.0f}" if delta_disp > 0 else None
            )
        
        with col2:
            st.metric(
                "Potential Savings",
                f"{sim_total_savings:,.0f}",
                delta=f"+{total_monthly_reduction:,.0f}" if total_monthly_reduction > 0 else None
            )
        
        with col3:
            orig_savings_pct = (derived['disposable_income'] / user_input["Income"]) * 100 if user_input["Income"] > 0 else 0
            new_savings_pct = (sim_disposable_income / user_input["Income"]) * 100 if user_input["Income"] > 0 else 0
            delta_pct = new_savings_pct - orig_savings_pct
            st.metric(
                "Savings %",
                f"{new_savings_pct:.1f}%",
                delta=f"+{delta_pct:.1f}%" if delta_pct > 0 else None
            )
        
        st.markdown("---")
        
        # ========================================================================
        # Yearly Projections
        # ========================================================================
        st.markdown("### Yearly Projections")
        st.markdown("Projected annual values based on current monthly spending.")
        
        annual_expenses = derived['total_expense'] * 12
        annual_savings = derived['disposable_income'] * 12
        annual_wants = derived['wants_total'] * 12
        sim_annual_savings = sim_disposable_income * 12
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Annual Expenses", f"{annual_expenses:,.0f}")
        with col2:
            st.metric("Annual Savings", f"{annual_savings:,.0f}")
        with col3:
            st.metric("Annual Wants", f"{annual_wants:,.0f}")
        
        if total_monthly_reduction > 0:
            st.info(f"Reducing entertainment by {sim_entertainment_reduction:,.0f} and eating out by {sim_eating_out_reduction:,.0f} per month saves **{total_monthly_reduction * 12:,.0f} per year**!")
        
        # 5-year projection chart
        st.markdown("### 5-Year Savings Projection")
        years = [1, 2, 3, 4, 5]
        current_projection = [annual_savings * y for y in years]
        simulated_projection = [sim_annual_savings * y for y in years]
        
        fig_proj, ax_proj = plt.subplots(figsize=(8, 4))
        ax_proj.plot(years, current_projection, marker='o', label='Current Path', color='#3498db', linewidth=2)
        if total_monthly_reduction > 0:
            ax_proj.plot(years, simulated_projection, marker='s', label='With Reductions', color='#2ecc71', linewidth=2, linestyle='--')
        ax_proj.set_xlabel('Years')
        ax_proj.set_ylabel('Cumulative Savings')
        ax_proj.set_title('Projected Savings Growth')
        ax_proj.legend()
        ax_proj.grid(True, alpha=0.3)
        ax_proj.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        st.pyplot(fig_proj)
        
        if total_monthly_reduction > 0:
            five_year_diff = simulated_projection[4] - current_projection[4]
            st.success(f"With the simulated reductions, you could save an additional **{five_year_diff:,.0f}** over 5 years!")
        
        # Navigation buttons for simulator
        render_nav_buttons()

# ========================================================================
# VIEW 5: SUMMARY / RESULTS - Overview of all analysis
# ========================================================================
elif st.session_state['current_view'] == 'results':
    if not st.session_state['form_submitted']:
        st.warning("Please enter your financial details on the Home page first.")
        if st.button("Go to Home"):
            set_view('home')
            st.rerun()
    else:
        user_input = st.session_state['user_input']
        derived = calculate_derived_values(user_input)
        
        # ML Predictions
        input_df = pd.DataFrame([user_input])
        input_df["Disposable_Income"] = derived['disposable_income']
        input_df["Desired_Savings"] = derived['desired_savings']
        input_df["Desired_Savings_Percentage"] = derived['desired_savings_percentage']
        input_df = input_df[preprocessor.feature_names_in_]
        input_processed = preprocessor.transform(input_df)
        xgb_pred = xgb_model.predict(input_processed)[0]
        label_map = {0: "Heavy Spender", 1: "Moderate Spender", 2: "Saver"}
        final_label = label_map[xgb_pred]
        
        st.markdown("<div class='view-header'><h2>Summary / Results</h2></div>", unsafe_allow_html=True)
        
        # Quick Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Income", f"{user_input['Income']:,.0f}")
        with col2:
            st.metric("Expenses", f"{derived['total_expense']:,.0f}")
        with col3:
            st.metric("Savings", f"{derived['disposable_income']:,.0f}")
        with col4:
            st.metric("Health Score", f"{derived['financial_health_score']}/100")
        
        st.markdown("---")
        
        # Category
        fhs = derived['financial_health_score']
        if fhs >= 80:
            st.success(f"**Spending Category:** {final_label} | **Financial Health:** Excellent")
        elif fhs >= 60:
            st.warning(f"**Spending Category:** {final_label} | **Financial Health:** Good")
        else:
            st.error(f"**Spending Category:** {final_label} | **Financial Health:** Needs Improvement")
        
        st.markdown("---")
        
        # Key Insights
        st.markdown("### Key Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 50/30/20 Rule Status")
            needs_ok = derived['needs_percentage'] <= 50
            wants_ok = derived['wants_percentage'] <= 30
            savings_ok = derived['savings_percentage'] >= 20
            st.markdown(f"- Needs (<=50%): {'Pass' if needs_ok else 'Fail'} - {derived['needs_percentage']:.1f}%")
            st.markdown(f"- Wants (<=30%): {'Pass' if wants_ok else 'Fail'} - {derived['wants_percentage']:.1f}%")
            st.markdown(f"- Savings (>=20%): {'Pass' if savings_ok else 'Fail'} - {derived['savings_percentage']:.1f}%")
        
        with col2:
            st.markdown("#### Annual Projections")
            st.markdown(f"- Annual Expenses: {derived['total_expense'] * 12:,.0f}")
            st.markdown(f"- Annual Savings: {derived['disposable_income'] * 12:,.0f}")
            st.markdown(f"- 5-Year Savings: {derived['disposable_income'] * 12 * 5:,.0f}")
        
        st.markdown("---")
        
        # Actions
        st.markdown("### Next Steps")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Update Data", use_container_width=True):
                set_view('home')
                st.rerun()
        with col2:
            if st.button("View Analysis", use_container_width=True):
                set_view('dashboard')
                st.rerun()
        with col3:
            if st.button("Try Simulator", use_container_width=True):
                set_view('simulator')
                st.rerun()
        
        # Navigation buttons for results
        render_nav_buttons()

# ========================================================================
# FOOTER & TEAM CREDIT
# ========================================================================
st.markdown("---")
st.markdown("<p class='footer'>Smart Financial Recommender, Spending Analyzer & Goal Predictor</p>", unsafe_allow_html=True)
st.markdown("<p class='team-credit'>Developed by Team Rovers</p>", unsafe_allow_html=True)
