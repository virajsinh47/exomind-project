import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
import sys
import os

# Dynamically add both the parent directory and nested directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
nested_dir = os.path.join(parent_dir, "exomind-project")
sys.path.append(parent_dir)
sys.path.append(nested_dir)

from data_pipeline.data_pipeline import get_clean_lightcurve
from model_training import predict_exoplanet
from analytics import calculate_transit_stats

# ==========================================
# EXOMIND — SPACE COMMAND CENTER
# ==========================================

st.set_page_config(page_title="ExoMind Control", layout="wide", page_icon="🪐", initial_sidebar_state="collapsed")

# ==========================================
# CSS: ULTIMATE SPACE THEME
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* =========================================
       1. BASE RESET & DEEP SPACE BACKGROUND
       ========================================= */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif !important;
        background-color: #050810 !important;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 20% 50%, rgba(6, 182, 212, 0.03) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(59, 130, 246, 0.04) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 80%, rgba(14, 165, 233, 0.02) 0%, transparent 60%),
            #050810 !important;
        color: #E2E8F0 !important;
        min-height: 100vh;
    }
    .main { background: transparent !important; }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="collapsedControl"] { display: none !important; }

    /* =========================================
       2. ANIMATED STARFIELD (3 layers)
       ========================================= */
    .stApp::before {
        content: '';
        position: fixed;
        top: -50%; left: -50%;
        width: 300%; height: 300%;
        background-image:
            radial-gradient(1.2px 1.2px at 20px 30px, rgba(255,255,255,0.9), transparent),
            radial-gradient(1px 1px at 80px 120px, rgba(255,255,255,0.5), transparent),
            radial-gradient(1.5px 1.5px at 160px 220px, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 260px 80px, rgba(255,255,255,0.4), transparent),
            radial-gradient(2px 2px at 340px 330px, rgba(34, 211, 238, 0.5), transparent),
            radial-gradient(1.2px 1.2px at 450px 180px, rgba(255,255,255,0.6), transparent),
            radial-gradient(1px 1px at 550px 400px, rgba(255,255,255,0.3), transparent),
            radial-gradient(1.8px 1.8px at 650px 120px, rgba(56, 189, 248, 0.4), transparent),
            radial-gradient(1px 1px at 720px 340px, rgba(255,255,255,0.5), transparent),
            radial-gradient(1.5px 1.5px at 800px 250px, rgba(255,255,255,0.8), transparent);
        background-size: 900px 500px;
        animation: driftStars 180s linear infinite;
        z-index: 0;
        pointer-events: none;
    }
    .stApp::after {
        content: '';
        position: fixed;
        top: -25%; left: -25%;
        width: 200%; height: 200%;
        background-image:
            radial-gradient(2px 2px at 100px 150px, rgba(255,255,255,0.15), transparent),
            radial-gradient(2.5px 2.5px at 300px 350px, rgba(255,255,255,0.12), transparent),
            radial-gradient(3px 3px at 500px 100px, rgba(255,255,255,0.1), transparent),
            radial-gradient(2px 2px at 700px 400px, rgba(255,255,255,0.08), transparent);
        background-size: 800px 500px;
        animation: driftStars 300s linear infinite reverse;
        z-index: 0;
        pointer-events: none;
    }
    @keyframes driftStars {
        from { transform: translateY(0) translateX(0); }
        to   { transform: translateY(-500px) translateX(-300px); }
    }
    section[data-testid="stMain"] > div { position: relative; z-index: 1; }

    /* =========================================
       3. SCANNING LINE (HUD effect)
       ========================================= */
    .scan-line {
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, rgba(6, 182, 212, 0.6) 50%, transparent 100%);
        animation: scanDown 8s ease-in-out infinite;
        z-index: 2;
        pointer-events: none;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.4), 0 0 30px rgba(6, 182, 212, 0.2);
    }
    @keyframes scanDown {
        0%   { top: -2px; opacity: 0; }
        5%   { opacity: 1; }
        95%  { opacity: 1; }
        100% { top: 100vh; opacity: 0; }
    }

    /* =========================================
       4. TYPOGRAPHY
       ========================================= */
    h1 {
        font-weight: 800 !important;
        color: #A5F3FC !important;
        text-shadow:
            0 0 8px rgba(34, 211, 238, 0.5),
            0 0 20px rgba(6, 182, 212, 0.3),
            0 0 40px rgba(6, 182, 212, 0.15);
        text-align: center;
        margin-bottom: 0px !important;
        padding-top: 2rem !important;
        font-size: 3.2rem !important;
        letter-spacing: -0.5px;
        animation: titlePulse 4s ease-in-out infinite;
    }
    @keyframes titlePulse {
        0%, 100% { text-shadow: 0 0 8px rgba(34,211,238,0.5), 0 0 20px rgba(6,182,212,0.3); }
        50%      { text-shadow: 0 0 12px rgba(34,211,238,0.7), 0 0 30px rgba(6,182,212,0.5), 0 0 50px rgba(6,182,212,0.2); }
    }
    .subtitle {
        text-align: center;
        color: #94A3B8;
        font-size: 1.1rem;
        margin-top: 2px;
        margin-bottom: 40px;
        font-weight: 400;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    /* Universal text color */
    label, p, span, div { color: #E2E8F0 !important; }

    /* =========================================
       5. GLASSMORPHISM COMMAND CARD
       Uses st.container(border=True) target
       ========================================= */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(8, 15, 30, 0.65) !important;
        backdrop-filter: blur(24px) saturate(1.3) !important;
        -webkit-backdrop-filter: blur(24px) saturate(1.3) !important;
        border: 1px solid rgba(6, 182, 212, 0.25) !important;
        border-radius: 20px !important;
        box-shadow:
            0 0 1px rgba(6, 182, 212, 0.4),
            0 0 25px rgba(6, 182, 212, 0.08),
            0 25px 50px rgba(0, 0, 0, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
        max-width: 720px !important;
        margin: 0 auto !important;
        padding: 45px 55px !important;
        position: relative;
        overflow: hidden;
    }
    /* Corner HUD brackets */
    [data-testid="stVerticalBlockBorderWrapper"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 50px; height: 50px;
        border-top: 2px solid rgba(6, 182, 212, 0.6);
        border-left: 2px solid rgba(6, 182, 212, 0.6);
        border-radius: 20px 0 0 0;
        pointer-events: none;
        z-index: 3;
    }
    [data-testid="stVerticalBlockBorderWrapper"]::after {
        content: '';
        position: absolute;
        bottom: 0; right: 0;
        width: 50px; height: 50px;
        border-bottom: 2px solid rgba(6, 182, 212, 0.6);
        border-right: 2px solid rgba(6, 182, 212, 0.6);
        border-radius: 0 0 20px 0;
        pointer-events: none;
        z-index: 3;
    }

    /* Center ALL content inside the card */
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
        align-items: center !important;
        text-align: center !important;
    }
    /* Force ALL element containers inside card to full width */
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stElementContainer"] {
        width: 100% !important;
    }

    /* Section header */
    .section-header {
        color: #F1F5F9 !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 25px !important;
        text-align: center !important;
        padding-bottom: 15px;
        border-bottom: 1px solid rgba(6, 182, 212, 0.15);
        width: 100%;
    }

    /* =========================================
       6. CUSTOM LABELS (Cyan Monospace)
       ========================================= */
    .field-label {
        color: #22D3EE !important;
        font-family: 'Fira Code', monospace !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        margin-bottom: 8px !important;
        margin-top: 20px !important;
        text-align: center !important;
        width: 100%;
    }

    /* =========================================
       7. RADIO BUTTONS (CYAN SEGMENTED TOGGLE)
       ========================================= */
    /* Force the radio's stElementContainer + all parents to full width & centered */
    .stRadio {
        width: 100% !important;
    }
    .stRadio > div {
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
    }
    .stRadio > div > div {
        width: auto !important;
    }
    div[data-baseweb="radio"] {
        display: flex !important;
        justify-content: center !important;
        gap: 0px !important;
        width: 100% !important;
    }
    /* Direct radiogroup target */
    [role="radiogroup"] {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }
    div[data-baseweb="radio"] > div {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(6, 182, 212, 0.25) !important;
        padding: 12px 30px !important;
        cursor: pointer !important;
        transition: all 0.25s ease !important;
    }
    div[data-baseweb="radio"] > div:first-child {
        border-radius: 10px 0 0 10px !important;
        border-right: none !important;
    }
    div[data-baseweb="radio"] > div:last-child {
        border-radius: 0 10px 10px 0 !important;
    }
    div[data-baseweb="radio"] > div:hover {
        background: rgba(6, 182, 212, 0.1) !important;
        border-color: rgba(6, 182, 212, 0.4) !important;
    }
    /* Radio text */
    div[data-baseweb="radio"] label span {
        color: #CBD5E1 !important;
        font-family: 'Fira Code', monospace !important;
        font-size: 0.9rem !important;
    }
    /* Radio dot → cyan */
    div[data-baseweb="radio"] label > div:first-child {
        border-color: rgba(6, 182, 212, 0.5) !important;
    }
    div[data-baseweb="radio"] label > div:first-child > div {
        background-color: #06B6D4 !important;
    }

    /* =========================================
       8. TEXT INPUT
       ========================================= */
    [data-testid="stVerticalBlockBorderWrapper"] .stTextInput {
        width: 100% !important;
    }
    .stTextInput>div>div>input {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid rgba(6, 182, 212, 0.25) !important;
        color: #67E8F9 !important;
        border-radius: 10px !important;
        font-family: 'Fira Code', monospace !important;
        text-align: center;
        font-size: 1.15rem !important;
        font-weight: 500 !important;
        padding: 16px !important;
        letter-spacing: 2px !important;
        transition: all 0.3s ease !important;
    }
    .stTextInput>div>div>input:focus {
        border-color: #06B6D4 !important;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.2), 0 0 30px rgba(6, 182, 212, 0.1) !important;
    }

    /* =========================================
       9. FILE UPLOADER
       ========================================= */
    [data-testid="stVerticalBlockBorderWrapper"] .stFileUploader {
        width: 100% !important;
    }
    /* Target the main dropzone container */
    [data-testid="stFileUploader"] > section,
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px dashed rgba(6, 182, 212, 0.35) !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploader"] > section:hover,
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #06B6D4 !important;
        background-color: rgba(6, 182, 212, 0.05) !important;
        box-shadow: 0 0 20px rgba(6, 182, 212, 0.1) !important;
    }
    /* Style the Browse Files button inside */
    [data-testid="stFileUploader"] button {
        background-color: rgba(6, 182, 212, 0.15) !important;
        border: 1px solid rgba(6, 182, 212, 0.5) !important;
        color: #A5F3FC !important;
        border-radius: 6px !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: rgba(6, 182, 212, 0.3) !important;
        color: #FFFFFF !important;
    }
    /* Force all text inside to cyan/slate */
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] label {
        color: #A5F3FC !important;
    }
    /* Ensure icons match */
    [data-testid="stFileUploader"] svg {
        fill: #A5F3FC !important;
        color: #A5F3FC !important;
    }
    
    /* Style the uploaded file preview items */
    [data-testid="stUploadedFile"],
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploader"] ul > li,
    [data-testid="stFileUploadDropzone"] ul > li,
    [data-testid="stFileUploader"] section + div > div {
        background: rgba(15, 23, 42, 0.95) !important;
        background-color: rgba(15, 23, 42, 0.95) !important;
        border: 1px solid rgba(6, 182, 212, 0.5) !important;
        border-radius: 12px !important;
    }
    
    [data-testid="stUploadedFile"] div,
    [data-testid="stUploadedFile"] span,
    [data-testid="stFileUploaderFile"] div,
    [data-testid="stFileUploaderFile"] span,
    [data-testid="stFileUploader"] ul > li div,
    [data-testid="stFileUploader"] ul > li span,
    [data-testid="stFileUploader"] section + div > div div,
    [data-testid="stFileUploader"] section + div > div span {
        background: transparent !important;
        background-color: transparent !important;
        color: #A5F3FC !important;
    }

    /* The delete 'x' button */
    [data-testid="stUploadedFile"] button,
    [data-testid="stFileUploaderFile"] button,
    [data-testid="stFileUploader"] ul > li button,
    [data-testid="stFileUploader"] section + div > div button {
        background: rgba(6, 182, 212, 0.1) !important;
        background-color: rgba(6, 182, 212, 0.1) !important;
    }

    /* The left file icon background */
    [data-testid="stUploadedFile"] > div > div:first-child,
    [data-testid="stFileUploaderFile"] > div > div:first-child,
    [data-testid="stFileUploader"] ul > li > div > div:first-child,
    [data-testid="stFileUploader"] section + div > div > div > div:first-child {
        background: rgba(6, 182, 212, 0.15) !important;
        background-color: rgba(6, 182, 212, 0.15) !important;
        border-radius: 6px !important;
    }

    /* =========================================
       10. CTA BUTTON — CENTERED, CYAN (NO PURPLE)
       ========================================= */
    /* Force button container + all parents to center */
    .stButton {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] .stButton {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }
    .stButton > div {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0891B2 0%, #0284C7 50%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-family: 'Fira Code', monospace !important;
        font-size: 0.95rem !important;
        letter-spacing: 3px !important;
        text-transform: uppercase !important;
        width: 280px !important;
        min-width: 280px !important;
        max-width: 280px !important;
        height: 54px !important;
        white-space: nowrap !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 0 15px rgba(8, 145, 178, 0.3), 0 8px 25px rgba(0, 0, 0, 0.4) !important;
        margin: 15px auto 0 auto !important;
        display: block !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 0 25px rgba(8, 145, 178, 0.5), 0 12px 30px rgba(0, 0, 0, 0.5) !important;
        border-color: rgba(6, 182, 212, 0.6) !important;
    }
    .stButton>button:active {
        transform: translateY(0px) !important;
    }

    /* =========================================
       11. METRIC CARDS
       ========================================= */
    div[data-testid="metric-container"] {
        background: rgba(8, 12, 24, 0.8) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(6, 182, 212, 0.12) !important;
        border-top: 2px solid #06B6D4 !important;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        border-color: rgba(6, 182, 212, 0.35) !important;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5), 0 0 15px rgba(6, 182, 212, 0.1);
    }
    div[data-testid="metric-container"] label {
        color: #22D3EE !important;
        font-family: 'Fira Code', monospace !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #F1F5F9 !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        font-family: 'Fira Code', monospace !important;
    }

    /* =========================================
       12. ALERTS
       ========================================= */
    .stAlert {
        background: rgba(8, 12, 24, 0.85) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px !important;
    }
    div[data-testid="stAlert"] > div { color: #E2E8F0 !important; }

    /* =========================================
       13. DIVIDER
       ========================================= */
    hr {
        border-color: rgba(6, 182, 212, 0.12) !important;
        margin: 30px 0 !important;
    }

    /* =========================================
       14. SPINNER
       ========================================= */
    .stSpinner > div > div { border-top-color: #06B6D4 !important; }
    .stSpinner > div > span {
        color: #94A3B8 !important;
        font-family: 'Fira Code', monospace !important;
    }

    /* =========================================
       15. STANDBY MESSAGE
       ========================================= */
    .standby-msg {
        text-align: center;
        color: #475569 !important;
        margin-top: 80px;
        font-family: 'Fira Code', monospace;
        font-size: 0.95rem;
        animation: standbyPulse 3s ease-in-out infinite;
    }
    .standby-msg .bracket { color: #0E7490 !important; }
    @keyframes standbyPulse {
        0%, 100% { opacity: 0.5; }
        50%      { opacity: 1; }
    }

    /* =========================================
       16. ANIMATED BACKGROUND LOGO (EXOMIND)
       ========================================= */
    .bg-logo-container {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 800px;
        height: 800px;
        z-index: 1;
        opacity: 0.25;
        pointer-events: none;
        animation: floatLogo 12s ease-in-out infinite;
    }
    @keyframes floatLogo {
        0%, 100% { transform: translate(-50%, -50%); }
        50% { transform: translate(-50%, -52%); }
    }
</style>
<!-- Scanning line overlay -->
<div class="scan-line"></div>
""", unsafe_allow_html=True)

# SVG with internal CSS to bypass Streamlit's DOMPurify which strips IDs
logo_svg = """<svg viewBox="0 0 800 800" xmlns="http://www.w3.org/2000/svg">
    <style>
        .circuit-line {
            stroke: #ff9900;
            stroke-width: 2.5;
            fill: none;
            stroke-dasharray: 15 15;
            animation: dataFlow 20s linear infinite;
        }
        @keyframes dataFlow {
            from { stroke-dashoffset: 1000; }
            to { stroke-dashoffset: 0; }
        }
        .ring-pulse {
            stroke: #00bfff;
            stroke-width: 4;
            fill: none;
            animation: ringPulse 4s ease-in-out infinite;
        }
        @keyframes ringPulse {
            0%, 100% { filter: drop-shadow(0 0 10px #00bfff); opacity: 0.8; }
            50% { filter: drop-shadow(0 0 25px #00bfff); opacity: 1; }
        }
        .sun-pulse {
            animation: sunPulse 6s ease-in-out infinite;
        }
        @keyframes sunPulse {
            0%, 100% { filter: drop-shadow(0 0 40px #ff6600); opacity: 0.8; }
            50% { filter: drop-shadow(0 0 70px #ff9900); opacity: 1; }
        }
    </style>
    <defs>
        <radialGradient id="sunGrad" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="#ffcc00" />
            <stop offset="70%" stop-color="#ff6600" />
            <stop offset="100%" stop-color="#cc3300" />
        </radialGradient>
        <radialGradient id="planetGrad" cx="50%" cy="30%" r="50%">
            <stop offset="0%" stop-color="#33ccff" />
            <stop offset="60%" stop-color="#0066cc" />
            <stop offset="100%" stop-color="#000033" />
        </radialGradient>
    </defs>
    <circle cx="400" cy="350" r="180" fill="url(#sunGrad)" class="sun-pulse" />
    <circle cx="400" cy="350" r="220" stroke="#ff9900" stroke-width="1" fill="none" opacity="0.3" stroke-dasharray="5 5" />
    <g opacity="0.8">
        <path d="M 50 350 L 220 350 M 100 320 L 220 320 M 80 380 L 220 380 M 120 290 L 220 290 M 120 410 L 220 410" class="circuit-line" />
        <circle cx="50" cy="350" r="5" fill="#ff9900" />
        <circle cx="100" cy="320" r="4" fill="#ff9900" />
        <circle cx="80" cy="380" r="4" fill="#ff9900" />
        <circle cx="120" cy="290" r="3" fill="#ff9900" />
        <circle cx="120" cy="410" r="3" fill="#ff9900" />
        <path d="M 750 350 L 580 350 M 700 320 L 580 320 M 720 380 L 580 380 M 680 290 L 580 290 M 680 410 L 580 410" class="circuit-line" />
        <circle cx="750" cy="350" r="5" fill="#ff9900" />
        <circle cx="700" cy="320" r="4" fill="#ff9900" />
        <circle cx="720" cy="380" r="4" fill="#ff9900" />
        <circle cx="680" cy="290" r="3" fill="#ff9900" />
        <circle cx="680" cy="410" r="3" fill="#ff9900" />
        <path d="M 330 200 L 330 350 M 365 140 L 365 350 M 400 100 L 400 350 M 435 140 L 435 350 M 470 200 L 470 350" class="circuit-line" />
        <circle cx="330" cy="200" r="4" fill="#ff9900" />
        <circle cx="365" cy="140" r="4" fill="#ff9900" />
        <circle cx="400" cy="100" r="5" fill="#ff9900" />
        <circle cx="435" cy="140" r="4" fill="#ff9900" />
        <circle cx="470" cy="200" r="4" fill="#ff9900" />
    </g>
    <circle cx="400" cy="480" r="110" fill="url(#planetGrad)" filter="drop-shadow(0 0 20px #00bfff)" />
    <ellipse cx="400" cy="480" rx="240" ry="40" class="ring-pulse" transform="rotate(-12 400 480)" />
    <path d="M 300 450 Q 400 480 500 450" stroke="#000033" stroke-width="4" fill="none" opacity="0.5" />
    <path d="M 295 480 Q 400 510 505 480" stroke="#000033" stroke-width="6" fill="none" opacity="0.6" />
    <path d="M 310 520 Q 400 550 490 520" stroke="#000033" stroke-width="3" fill="none" opacity="0.4" />
</svg>"""

import base64
b64_logo = base64.b64encode(logo_svg.encode("utf-8")).decode("utf-8")

st.markdown(f"""
<div class="bg-logo-container">
    <img src="data:image/svg+xml;base64,{b64_logo}" width="800" height="800" />
</div>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown("<h1>ExoMind System</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Bharatiya Antariksh Hackathon &nbsp;·&nbsp; AI Exoplanet Detection</div>", unsafe_allow_html=True)

# ==========================================
# ==========================================
st.write("")

# Use st.container(border=True) for a reliable CSS target
with st.container(border=True):
    st.markdown("<div class='section-header'>🛰️ Mission Parameters</div>", unsafe_allow_html=True)

    # Radio toggle — force centered, horizontal, with gap
    st.markdown("""<style>
        .stRadio { display:flex !important; justify-content:center !important; width:100% !important; }
        .stRadio > div { display:flex !important; justify-content:center !important; width:100% !important; }
        .stRadio > div > div { display:flex !important; justify-content:center !important; }
        .stRadio div[role="radiogroup"] { display:flex !important; flex-direction:row !important; justify-content:center !important; width:100% !important; gap:15px !important; }
        .stRadio [data-baseweb="radio"] { display:flex !important; flex-direction:row !important; justify-content:center !important; width:100% !important; gap:15px !important; }
        .stElementContainer:has(.stRadio) { display:flex !important; justify-content:center !important; width:100% !important; }
    </style>""", unsafe_allow_html=True)
    data_source = st.radio("Telemetry Link", ["Live NASA API", "Local File Upload"], horizontal=True, label_visibility="collapsed")

    # Conditional input
    if data_source == "Live NASA API":
        st.markdown("<div class='field-label'>Target Identification (TIC ID)</div>", unsafe_allow_html=True)
        tic_id = st.text_input("Target TIC ID", value="TIC 279741379", label_visibility="collapsed")
        uploaded_file = None
    else:
        st.markdown("<div class='field-label'>Upload Light Curve Payload</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload FITS/CSV Payload", type=["fits", "csv"], label_visibility="collapsed")
        tic_id = "Uploaded File"

    st.write("")
    # Button — centered using columns
    b_spacer1, b_center, b_spacer2 = st.columns([1, 1, 1])
    with b_center:
        analyze_btn = st.button("🚀  INITIATE  SCAN", use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ==========================================
# CORE LOGIC & VISUALIZATION (UNTOUCHED)
# ==========================================
if analyze_btn:
    import streamlit.components.v1 as components
    components.html("""
        <script>
            setTimeout(() => {
                const target = window.parent.document.getElementById('loader-target');
                if (target) {
                    target.scrollIntoView({behavior: 'smooth', block: 'center'});
                }
            }, 150);
        </script>
    """, height=0)

    loader_placeholder = st.empty()
    loader_html = """
    <div id="loader-target" style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 80px 0; z-index: 10;">
        <div style="position: relative; width: 120px; height: 120px;">
            <div style="position: absolute; top: 50%; left: 50%; width: 50px; height: 50px; background: radial-gradient(circle at 30% 30%, #33ccff, #0066cc); border-radius: 50%; transform: translate(-50%, -50%); box-shadow: 0 0 20px #00bfff;"></div>
            <div style="position: absolute; top: 50%; left: 50%; width: 120px; height: 40px; border: 4px solid #ff9900; border-radius: 50%; transform: translate(-50%, -50%) rotate(20deg); border-top-color: transparent; border-bottom-color: transparent; animation: spinLoader 1.2s linear infinite;"></div>
            <div style="position: absolute; top: 50%; left: 50%; width: 100px; height: 100px; border: 2px dashed #A5F3FC; border-radius: 50%; transform: translate(-50%, -50%); animation: spinLoaderReverse 4s linear infinite; opacity: 0.5;"></div>
        </div>
        <div style="margin-top: 30px; font-family: 'Fira Code', monospace; color: #A5F3FC; font-size: 1.1rem; letter-spacing: 3px; animation: pulseText 1.5s infinite; text-transform: uppercase;">
            {text}
        </div>
        <style>
            @keyframes spinLoader { 0% { transform: translate(-50%, -50%) rotate(0deg); } 100% { transform: translate(-50%, -50%) rotate(360deg); } }
            @keyframes spinLoaderReverse { 0% { transform: translate(-50%, -50%) rotate(360deg); } 100% { transform: translate(-50%, -50%) rotate(0deg); } }
            @keyframes pulseText { 0%, 100% { opacity: 0.5; } 50% { opacity: 1; text-shadow: 0 0 15px #A5F3FC; } }
        </style>
    </div>
    """

    time_array, flux_array = np.array([]), np.array([])

    if data_source == "Live NASA API":
        loader_placeholder.markdown(loader_html.replace("{text}", "DOWNLOADING MAST TELEMETRY..."), unsafe_allow_html=True)
        time_array, flux_array = get_clean_lightcurve(tic_id)
        loader_placeholder.empty()
    else:
        if uploaded_file is not None:
            loader_placeholder.markdown(loader_html.replace("{text}", "PROCESSING LOCAL PAYLOAD..."), unsafe_allow_html=True)
            try:
                import pandas as pd
                import tempfile
                import lightkurve as lk

                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    if df.shape[1] < 2:
                        raise ValueError("Commander, the CSV payload must contain at least two columns (time and flux).")
                    
                    # Force conversion to numeric, ignoring text/headers
                    time_data = pd.to_numeric(df.iloc[:, 0], errors='coerce')
                    flux_data = pd.to_numeric(df.iloc[:, 1], errors='coerce')
                    
                    # Filter out any rows that had invalid text or missing data
                    valid_mask = ~time_data.isna() & ~flux_data.isna()
                    time_array = time_data[valid_mask].values
                    flux_array = flux_data[valid_mask].values
                    
                    if len(flux_array) < 50:
                        raise ValueError("Payload rejected: Not enough valid numeric data points found in the CSV.")
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    lc = lk.read(tmp_path)
                    clean_lc = lc.remove_nans().remove_outliers().flatten(window_length=101)
                    time_array = clean_lc.time.value
                    flux_array = clean_lc.flux.value
                    os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Error processing payload: {e}")
            loader_placeholder.empty()
        else:
            st.error("Commander, please upload a payload first.")

    if len(flux_array) == 0:
        st.error(f"Could not lock onto telemetry for {tic_id}. Please try another ID.")
    else:
        # Run AI
        loader_placeholder.markdown(loader_html.replace("{text}", "ROUTING DATA THROUGH NEURAL NETWORK..."), unsafe_allow_html=True)
        target_length = 20000
        if len(flux_array) > target_length:
            flux_for_ai = flux_array[:target_length]
        elif len(flux_array) < target_length:
            median_flux = np.median(flux_array)
            padding = np.full(target_length - len(flux_array), median_flux)
            flux_for_ai = np.concatenate([flux_array, padding])
        else:
            flux_for_ai = flux_array

        ai_result = predict_exoplanet(flux_for_ai)
        
        # Calculate Math
        loader_placeholder.markdown(loader_html.replace("{text}", "CALCULATING ORBITAL MECHANICS..."), unsafe_allow_html=True)
        stats = calculate_transit_stats(time_array, flux_array)
        loader_placeholder.empty()

        # ---- INTELLIGENCE REPORT ----
        st.markdown("<div class='section-header' style='border-bottom: none;'>📊 Intelligence Report</div>", unsafe_allow_html=True)

        if ai_result["is_planet"]:
            st.success(f"🌟 **VERIFIED:** {ai_result['class_label']} Detected! (Confidence: {ai_result['confidence']*100:.1f}%)")
        else:
            st.warning(f"⚠️ **NEGATIVE:** {ai_result['class_label']} (Confidence: {ai_result['confidence']*100:.1f}%)")

        st.write("")

        # Metric Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Orbital Period", f"{stats['orbital_period_days']:.4f} d")
        col2.metric("Transit Duration", f"{stats['transit_duration_days'] * 24:.2f} h")
        col3.metric("Transit Depth", f"{stats['transit_depth']:.4f}")
        col4.metric("Signal-to-Noise", f"{stats['snr']:.1f}")
        col5.metric("Data Points", f"{len(flux_array)}")

        st.write("")

        # ---- PLOTLY CHART ----
        st.markdown(f"<div class='section-header' style='border-bottom: none;'>📈 Light Curve Telemetry: {tic_id}</div>", unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_array,
            y=flux_array,
            mode='markers',
            marker=dict(size=3, color='#06B6D4', opacity=0.85),
            name='Flux Data'
        ))
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Time (Days)",
            yaxis_title="Normalized Flux",
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="x unified",
            font=dict(family="Fira Code, monospace", color="#94A3B8"),
            xaxis=dict(gridcolor="rgba(6, 182, 212, 0.06)", zerolinecolor="rgba(6,182,212,0.1)"),
            yaxis=dict(gridcolor="rgba(6, 182, 212, 0.06)", zerolinecolor="rgba(6,182,212,0.1)")
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown("""
        <div class='standby-msg'>
            <span class='bracket'>[</span> SYSTEM STANDBY <span class='bracket'>]</span><br>
            Awaiting Commander input — Initialize scan to begin.
        </div>
    """, unsafe_allow_html=True)

