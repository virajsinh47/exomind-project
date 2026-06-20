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
# 🚀 THE STREAMLIT UI (PROFESSIONAL THEME)
# ==========================================

st.set_page_config(page_title="ExoMind Control", layout="wide", page_icon="🪐", initial_sidebar_state="collapsed")

# 1. PROFESSIONAL CINEMATIC DEEP SPACE CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');

    /* Global Font & Transparent App Background */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif !important;
    }
    .stApp, .main, [data-testid="stHeader"] {
        background-color: transparent !important;
        color: #F8FAFC;
    }
    
    /* Hide sidebar toggle */
    [data-testid="collapsedControl"] { display: none !important; }

    /* Deep Space Parallax Background */
    /* This creates a slow, elegant starfield without breaking markdown or layouts */
    body::before {
        content: '';
        position: fixed;
        top: 0; left: 0; width: 200%; height: 200%;
        background-color: #030712;
        background-image: 
            radial-gradient(1px 1px at 50px 50px, #ffffff, rgba(0,0,0,0)),
            radial-gradient(1.5px 1.5px at 150px 250px, rgba(255,255,255,0.8), rgba(0,0,0,0)),
            radial-gradient(1px 1px at 300px 100px, rgba(255,255,255,0.6), rgba(0,0,0,0)),
            radial-gradient(2px 2px at 400px 400px, rgba(100, 200, 255, 0.9), rgba(0,0,0,0)),
            radial-gradient(1px 1px at 600px 200px, #ffffff, rgba(0,0,0,0));
        background-repeat: repeat;
        background-size: 800px 800px;
        animation: slowDrift 200s linear infinite;
        opacity: 0.85;
        z-index: -1;
    }
    @keyframes slowDrift {
        from { transform: translate(0, 0); }
        to { transform: translate(-400px, -400px); }
    }

    /* Clean Typography */
    h1 {
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        color: #FFFFFF !important;
        text-align: center;
        margin-bottom: 0px !important;
        padding-top: 1rem !important;
    }
    h3 {
        font-weight: 600 !important;
        color: #F8FAFC !important;
    }
    .subtitle {
        text-align: center;
        color: #94A3B8;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 40px;
        font-weight: 400;
        letter-spacing: 0.5px;
    }

    /* Streamlit Input Styling (Glassmorphism) */
    div[data-baseweb="radio"] {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.2);
        padding: 10px 15px;
        border-radius: 8px;
    }
    .stTextInput>div>div>input {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        color: #F8FAFC !important;
        border-radius: 6px;
    }
    .stTextInput>div>div>input:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 1px #38BDF8 !important;
    }
    label {
        color: #CBD5E1 !important;
        font-weight: 500 !important;
    }

    /* Enterprise Button */
    .stButton>button {
        background-color: #0284C7;
        color: white !important;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        letter-spacing: 0.5px;
        width: 100%;
        height: 100%;
        min-height: 44px;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #0369A1;
        color: white !important;
    }

    /* Professional Metrics */
    div[data-testid="metric-container"] {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-top: 3px solid #38BDF8;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    div[data-testid="metric-container"] label {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="metric-container"] div {
        color: #F8FAFC !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }

    /* Alerts */
    .stAlert {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid #38BDF8 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown("<h1>🪐 ExoMind System</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Bharatiya Antariksh Hackathon - AI Exoplanet Detection</div>", unsafe_allow_html=True)

# ==========================================
# CENTRALIZED CONTROL DECK (Native Streamlit)
# ==========================================
# We use native columns without wrapping them in HTML divs to prevent layout breaking.
st.markdown("### 📡 Mission Parameters")

col1, col2, col3 = st.columns([1, 1.5, 1])

with col1:
    data_source = st.radio("Data Source", ["Live NASA API", "Local File Upload"], horizontal=True)

tic_id = "Uploaded File"
uploaded_file = None

with col2:
    if data_source == "Live NASA API":
        tic_id = st.text_input("Target TIC ID", value="TIC 279741379")
    else:
        uploaded_file = st.file_uploader("Upload Light Curve", type=["fits", "csv"])

with col3:
    st.write("") # Alignment spacing
    st.write("")
    analyze_btn = st.button("🚀 Deploy AI Pipeline")

st.markdown("---")


# ==========================================
# CORE LOGIC & VISUALIZATION
# ==========================================
if analyze_btn:
    time_array, flux_array = np.array([]), np.array([])
    
    if data_source == "Live NASA API":
        # 1. Fetch Data
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
        # 2. Run AI
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
            
        # 3. Calculate Math
        with st.spinner("Calculating Orbital Mechanics (BLS)..."):
            stats = calculate_transit_stats(time_array, flux_array)

        # ----------------------------------------------------
        # UI: Metrics Dashboard
        # ----------------------------------------------------
        st.markdown("### 📊 Intelligence Report")
        
        # Highlight the AI's final decision
        if ai_result["is_planet"]:
            st.success(f"🌟 **VERDICT:** {ai_result['class_label']} Detected! (Confidence: {ai_result['confidence']*100:.1f}%)")
        else:
            st.warning(f"⚠️ **VERDICT:** {ai_result['class_label']} (Confidence: {ai_result['confidence']*100:.1f}%)")

        st.write("") # Spacing

        # Display the 5 math numbers in a clean row
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Orbital Period", f"{stats['orbital_period_days']:.4f} Days")
        col2.metric("Transit Duration", f"{stats['transit_duration_days'] * 24:.2f} Hrs")
        col3.metric("Transit Depth", f"{stats['transit_depth']:.4f}")
        col4.metric("Signal-to-Noise", f"{stats['snr']:.1f}")
        col5.metric("Data Points", f"{len(flux_array)}")

        st.write("") # Spacing

        # ----------------------------------------------------
        # UI: Beautiful Plotly Graph
        # ----------------------------------------------------
        st.markdown(f"### 📈 Light Curve Telemetry: {tic_id}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_array, 
            y=flux_array, 
            mode='markers',
            marker=dict(size=3, color='#38BDF8', opacity=0.7),
            name='Flux Data'
        ))
        
        # Sleek transparent dark mode plot
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Time (Days)",
            yaxis_title="Normalized Flux",
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown("<div style='text-align: center; color: #64748B; margin-top: 50px;'>Awaiting Commander input... Initialize scan to begin.</div>", unsafe_allow_html=True)
