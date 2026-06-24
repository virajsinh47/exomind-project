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
# 🚀 EXO-MIND: ULTIMATE SPACE THEME UX
# ==========================================

st.set_page_config(page_title="ExoMind Control", layout="wide", page_icon="🪐", initial_sidebar_state="collapsed")

# 1. UI/UX PRO MAX CSS INJECTION (SPACE THEME)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* BASE THEME */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .stApp, .main, [data-testid="stHeader"] {
        background-color: transparent !important;
        color: #F8FAFC !important;
    }
    
    [data-testid="collapsedControl"] { display: none !important; }

    /* 🌌 ANIMATED GALAXY BACKGROUND */
    body::before {
        content: '';
        position: fixed;
        top: -50%; left: -50%; width: 200%; height: 200%;
        background-color: #030614;
        background-image: 
            radial-gradient(2px 2px at 40px 60px, #ffffff, transparent),
            radial-gradient(2.5px 2.5px at 150px 250px, rgba(255,255,255,0.8), transparent),
            radial-gradient(1.5px 1.5px at 300px 100px, rgba(255,255,255,0.6), transparent),
            radial-gradient(3px 3px at 400px 400px, rgba(100, 200, 255, 0.9), transparent),
            radial-gradient(2.5px 2.5px at 600px 200px, #ffffff, transparent),
            radial-gradient(1.5px 1.5px at 800px 700px, rgba(255,255,255,0.5), transparent),
            radial-gradient(3px 3px at 900px 100px, rgba(255,255,255,0.8), transparent);
        background-size: 1000px 1000px;
        animation: rotateSpace 250s linear infinite;
        z-index: -2;
        opacity: 0.9;
    }

    body::after {
        content: '';
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        background: 
            radial-gradient(circle at 20% 30%, rgba(59, 130, 246, 0.12) 0%, transparent 40%),
            radial-gradient(circle at 80% 70%, rgba(139, 92, 246, 0.12) 0%, transparent 50%);
        z-index: -1;
        animation: pulseNebula 12s ease-in-out infinite alternate;
        pointer-events: none;
    }

    @keyframes rotateSpace {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes pulseNebula {
        0% { opacity: 0.6; transform: scale(1); }
        100% { opacity: 1; transform: scale(1.1); }
    }

    /* TYPOGRAPHY */
    h1 {
        font-weight: 800 !important;
        background: linear-gradient(to right, #FFFFFF, #93C5FD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px !important;
        padding-top: 2rem !important;
        font-size: 3.5rem !important;
        letter-spacing: -1px;
    }
    h2, h3 {
        font-weight: 600 !important;
        color: #F8FAFC !important;
    }
    .subtitle {
        text-align: center;
        color: #94A3B8;
        font-size: 1.2rem;
        margin-top: -5px;
        margin-bottom: 40px;
        font-weight: 400;
        letter-spacing: 1px;
    }

    /* UNIVERSAL TEXT COLOR FIX */
    label, p, span, div {
        color: #E2E8F0 !important;
    }

    /* 🎛️ GLASSMORPHISM CONTROL DECK CARD */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] > div[data-testid="stVerticalBlock"] {
        background: rgba(10, 15, 30, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-top: 1px solid rgba(139, 92, 246, 0.4); /* Purple rim light */
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255,255,255,0.1);
        max-width: 800px;
        margin: 0 auto;
    }

    /* FORM INPUTS */
    div[data-baseweb="radio"] {
        gap: 15px;
        display: flex;
        justify-content: center;
        margin-bottom: 15px;
    }
    div[data-baseweb="radio"] > div {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 12px 24px;
        border-radius: 12px;
        transition: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    div[data-baseweb="radio"] > div:hover {
        border-color: #60A5FA;
        background: rgba(30, 58, 138, 0.4);
        transform: translateY(-2px);
    }

    .stTextInput>div>div>input {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(59, 130, 246, 0.4) !important;
        color: #60A5FA !important;
        border-radius: 12px !important;
        font-family: 'Fira Code', monospace !important;
        text-align: center;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        padding: 15px !important;
        transition: all 0.3s ease !important;
    }
    .stTextInput>div>div>input:focus {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.3) !important;
    }

    /* 🔥 FIXING THE UGLY FILE UPLOADER */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(15, 23, 42, 0.5) !important;
        border: 2px dashed rgba(59, 130, 246, 0.4) !important;
        border-radius: 16px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #8B5CF6 !important;
        background-color: rgba(30, 58, 138, 0.3) !important;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.2) !important;
    }
    [data-testid="stFileUploadDropzone"] div, [data-testid="stFileUploadDropzone"] span {
        color: #93C5FD !important;
    }

    /* 🚀 PRIMARY LAUNCH CTA BUTTON */
    .stButton>button {
        background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 30px !important; /* Sleek pill shape */
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        width: 100% !important;
        height: 60px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 10px 25px rgba(124, 58, 237, 0.4) !important;
        margin-top: 20px !important; 
    }
    .stButton>button:hover {
        transform: scale(1.02) translateY(-2px) !important;
        box-shadow: 0 15px 35px rgba(124, 58, 237, 0.6) !important;
    }

    /* DATA METRICS (Monospace digits) */
    div[data-testid="metric-container"] {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-top: 3px solid #8B5CF6;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-color: rgba(139, 92, 246, 0.5);
    }
    div[data-testid="metric-container"] label {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="metric-container"] div {
        background: linear-gradient(to right, #93C5FD, #C4B5FD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        font-family: 'Fira Code', monospace !important; 
    }

    /* ALERTS */
    .stAlert {
        background: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(10px);
        border: 1px solid #3B82F6 !important;
        border-radius: 12px !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown("<h1>ExoMind System</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Bharatiya Antariksh Hackathon - AI Exoplanet Detection</div>", unsafe_allow_html=True)

# ==========================================
# CENTRALIZED CONTROL DECK (VERTICAL STACK)
# ==========================================
st.write("") # Top spacing
col_spacer1, col_center, col_spacer2 = st.columns([1, 4, 1])

with col_center:
    st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>🛰️ Mission Parameters</h3>", unsafe_allow_html=True)
    
    # Fully vertical stack for perfect alignment
    data_source = st.radio("Telemetry Link", ["Live NASA API", "Local File Upload"], horizontal=True, label_visibility="collapsed")
    
    st.write("") # spacing
    
    tic_id = "Uploaded File"
    uploaded_file = None
    if data_source == "Live NASA API":
        tic_id = st.text_input("Target TIC ID", value="TIC 279741379", label_visibility="collapsed")
    else:
        uploaded_file = st.file_uploader("Upload FITS/CSV Payload", type=["fits", "csv"], label_visibility="collapsed")

    st.write("") # spacing
    analyze_btn = st.button("🚀 INITIATE SCAN")

st.markdown("<br><hr style='border-color: rgba(59, 130, 246, 0.15);'><br>", unsafe_allow_html=True)


# ==========================================
# CORE LOGIC & VISUALIZATION
# ==========================================
if analyze_btn:
    time_array, flux_array = np.array([]), np.array([])
    
    if data_source == "Live NASA API":
        with st.spinner("Establishing uplink to NASA MAST... Downloading telemetry..."):
            time_array, flux_array = get_clean_lightcurve(tic_id)
    else:
        if uploaded_file is not None:
            with st.spinner("Processing local payload..."):
                try:
                    import pandas as pd
                    import tempfile
                    import lightkurve as lk
                    
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        time_array = df.iloc[:, 0].values
                        flux_array = df.iloc[:, 1].values
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
        else:
            st.error("Commander, please upload a payload first.")
        
    if len(flux_array) == 0:
        st.error(f"Could not lock onto telemetry for {tic_id}. Aborting.")
    else:
        # Run AI
        with st.spinner("Routing data through PyTorch Neural Network..."):
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
        with st.spinner("Calculating Orbital Mechanics (BLS)..."):
            stats = calculate_transit_stats(time_array, flux_array)

        # ----------------------------------------------------
        # UI: Metrics Dashboard
        # ----------------------------------------------------
        st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>📊 Intelligence Report</h3>", unsafe_allow_html=True)
        
        if ai_result["is_planet"]:
            st.success(f"🌟 **VERIFIED:** {ai_result['class_label']} Detected! (Confidence: {ai_result['confidence']*100:.1f}%)")
        else:
            st.warning(f"⚠️ **NEGATIVE:** {ai_result['class_label']} (Confidence: {ai_result['confidence']*100:.1f}%)")

        st.write("") 

        # Display the 5 math numbers
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Orbital Period", f"{stats['orbital_period_days']:.4f} d")
        col2.metric("Transit Duration", f"{stats['transit_duration_days'] * 24:.2f} h")
        col3.metric("Transit Depth", f"{stats['transit_depth']:.4f}")
        col4.metric("Signal-to-Noise", f"{stats['snr']:.1f}")
        col5.metric("Data Points", f"{len(flux_array)}")

        st.write("") 

        # ----------------------------------------------------
        # UI: Beautiful Plotly Graph
        # ----------------------------------------------------
        st.markdown(f"<h3 style='text-align: center;'>📈 Light Curve Telemetry: {tic_id}</h3>", unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_array, 
            y=flux_array, 
            mode='markers',
            marker=dict(size=3, color='#8B5CF6', opacity=0.8),
            name='Flux Data'
        ))
        
        # Transparent dark mode plot matching OLED theme
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Time (Days)",
            yaxis_title="Normalized Flux",
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="x unified",
            font=dict(family="Fira Code, monospace", color="#94A3B8")
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown("<div style='text-align: center; color: #64748B; margin-top: 80px; font-weight: 500;'>Awaiting Commander input... Initialize scan to begin.</div>", unsafe_allow_html=True)
