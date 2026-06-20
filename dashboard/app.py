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

# Import the real functions from your team!
from data_pipeline.data_pipeline import get_clean_lightcurve
from model_training import predict_exoplanet
from analytics import calculate_transit_stats

# ==========================================
# 🚀 THE STREAMLIT UI (PRO MAX UX DESIGN)
# ==========================================

st.set_page_config(page_title="ExoMind Control", layout="wide", page_icon="🪐", initial_sidebar_state="collapsed")

# 1. UI/UX PRO MAX CSS INJECTION
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

    /* BASE THEME - OLED DARK MODE */
    html, body, [class*="css"] {
        font-family: 'Fira Sans', sans-serif !important;
    }
    
    .stApp, .main, [data-testid="stHeader"] {
        background-color: transparent !important;
        color: #F8FAFC !important;
    }
    
    [data-testid="collapsedControl"] { display: none !important; }

    /* BACKGROUND - SLOW CINEMATIC PARALLAX */
    body::before {
        content: '';
        position: fixed;
        top: 0; left: 0; width: 200%; height: 200%;
        background-color: #050A15; /* Deep OLED Navy */
        background-image: 
            radial-gradient(1px 1px at 50px 50px, rgba(255,255,255,0.8), rgba(0,0,0,0)),
            radial-gradient(1.5px 1.5px at 150px 250px, rgba(255,255,255,0.6), rgba(0,0,0,0)),
            radial-gradient(1px 1px at 300px 100px, rgba(255,255,255,0.5), rgba(0,0,0,0)),
            radial-gradient(2px 2px at 400px 400px, rgba(59, 130, 246, 0.4), rgba(0,0,0,0)),
            radial-gradient(1px 1px at 600px 200px, rgba(255,255,255,0.7), rgba(0,0,0,0));
        background-repeat: repeat;
        background-size: 800px 800px;
        animation: slowDrift 200s linear infinite;
        z-index: -1;
    }
    @keyframes slowDrift {
        from { transform: translate(0, 0); }
        to { transform: translate(-400px, -400px); }
    }

    /* TYPOGRAPHY */
    h1 {
        font-weight: 700 !important;
        color: #F8FAFC !important;
        text-align: center;
        margin-bottom: 0px !important;
        padding-top: 1.5rem !important;
        font-size: 2.5rem !important;
    }
    h2, h3 {
        font-weight: 600 !important;
        color: #F8FAFC !important;
    }
    .subtitle {
        text-align: center;
        color: #94A3B8;
        font-size: 1.1rem;
        margin-top: -5px;
        margin-bottom: 30px;
        font-weight: 400;
    }

    /* UNIVERSAL TEXT COLOR FIX (Solves the invisible text issue) */
    label, p, span, div {
        color: #E2E8F0 !important;
    }

    /* CONTROL DECK CARD */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] > div[data-testid="stVerticalBlock"] {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(59, 130, 246, 0.2);
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        max-width: 900px;
        margin: 0 auto;
    }

    /* FORM INPUTS */
    div[data-baseweb="radio"] {
        gap: 20px;
    }
    div[data-baseweb="radio"] > div {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 10px 20px;
        border-radius: 8px;
        transition: 0.2s ease;
    }
    div[data-baseweb="radio"] > div:hover {
        border-color: #3B82F6;
    }

    .stTextInput>div>div>input, .stFileUploader>div>div {
        background-color: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        color: #F8FAFC !important;
        border-radius: 8px !important;
        font-family: 'Fira Code', monospace !important;
    }
    .stTextInput>div>div>input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 1px #3B82F6 !important;
    }

    /* PRIMARY CTA BUTTON */
    .stButton>button {
        background-color: #2563EB !important; /* Blue-600 */
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.5px !important;
        width: 100% !important;
        height: 100% !important;
        min-height: 50px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
        margin-top: 28px !important; /* Aligns button with inputs */
    }
    .stButton>button:hover {
        background-color: #1D4ED8 !important; /* Blue-700 */
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
    }

    /* DATA METRICS (Monospace digits) */
    div[data-testid="metric-container"] {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-top: 3px solid #3B82F6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="metric-container"] label {
        color: #94A3B8 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="metric-container"] div {
        color: #F8FAFC !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        font-family: 'Fira Code', monospace !important; /* Data styling */
    }

    /* ALERTS */
    .stAlert {
        background: rgba(30, 41, 59, 0.9) !important;
        border: 1px solid #3B82F6 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown("<h1>ExoMind System</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Bharatiya Antariksh Hackathon - AI Exoplanet Detection</div>", unsafe_allow_html=True)

# ==========================================
# CENTRALIZED CONTROL DECK
# ==========================================
# Centered Container for better visual flow
st.write("") # Top spacing
col_spacer1, col_center, col_spacer2 = st.columns([1, 6, 1])

with col_center:
    st.markdown("### 📡 Mission Parameters")
    
    # Form layout
    form_col1, form_col2 = st.columns([1, 1])
    
    with form_col1:
        data_source = st.radio("Data Source", ["Live NASA API", "Local File Upload"], horizontal=True)
        
    with form_col2:
        tic_id = "Uploaded File"
        uploaded_file = None
        if data_source == "Live NASA API":
            tic_id = st.text_input("Target TIC ID", value="TIC 279741379")
        else:
            uploaded_file = st.file_uploader("Upload Light Curve", type=["fits", "csv"])

    analyze_btn = st.button("🚀 Deploy AI Pipeline")

st.markdown("<br><hr style='border-color: rgba(59, 130, 246, 0.2);'><br>", unsafe_allow_html=True)


# ==========================================
# CORE LOGIC & VISUALIZATION
# ==========================================
if analyze_btn:
    time_array, flux_array = np.array([]), np.array([])
    
    if data_source == "Live NASA API":
        with st.spinner("Establishing uplink to NASA MAST... Downloading data..."):
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
                    st.error(f"Error processing file: {e}")
        else:
            st.error("Please upload a file first.")
        
    if len(flux_array) == 0:
        st.error(f"Could not find or download data for {tic_id}. Please try another ID.")
    else:
        # Run AI
        with st.spinner("Routing data to PyTorch Neural Network..."):
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
        st.markdown("### 📊 Intelligence Report")
        
        if ai_result["is_planet"]:
            st.success(f"🌟 **VERDICT:** {ai_result['class_label']} Detected! (Confidence: {ai_result['confidence']*100:.1f}%)")
        else:
            st.warning(f"⚠️ **VERDICT:** {ai_result['class_label']} (Confidence: {ai_result['confidence']*100:.1f}%)")

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
        st.markdown(f"### 📈 Light Curve Telemetry: {tic_id}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_array, 
            y=flux_array, 
            mode='markers',
            marker=dict(size=3, color='#3B82F6', opacity=0.8),
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
