import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
import sys
import os

# Add the nested exomind-project directory to the Python path
# This allows us to import the files created by the rest of the team.
sys.path.append(r"D:\exomind-project\exomind-project")

# Import the real functions from your team!
from data_pipeline.data_pipeline import get_clean_lightcurve
from model_training import predict_exoplanet
from analytics import calculate_transit_stats

# ==========================================
# 🚀 THE STREAMLIT UI (YOUR JOB)
# ==========================================

# Set up the webpage
st.set_page_config(page_title="ExoMind Dashboard", layout="wide", page_icon="🪐")

# Inject Custom CSS for Premium Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=Orbitron:wght@500;700;900&display=swap');

    /* Global Font & Animated Star Background */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    .stApp {
        background-color: #050810;
        background-image: 
            radial-gradient(1px 1px at 20px 30px, #ffffff, rgba(0,0,0,0)),
            radial-gradient(1px 1px at 40px 70px, rgba(255,255,255,0.8), rgba(0,0,0,0)),
            radial-gradient(2px 2px at 90px 40px, #ffffff, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 160px 120px, rgba(255,255,255,0.6), rgba(0,0,0,0));
        background-repeat: repeat;
        background-size: 200px 200px;
        animation: stars 100s linear infinite;
        color: #E2E8F0;
    }

    @keyframes stars {
        0% { background-position: 0 0; }
        100% { background-position: -1000px 1000px; }
    }

    /* Sci-Fi Titles */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        color: #00F0FF !important;
        text-shadow: 0 0 15px rgba(0, 240, 255, 0.4);
        letter-spacing: 1px;
    }

    /* Glassmorphism Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(5, 8, 16, 0.8) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(0, 240, 255, 0.2);
    }
    
    /* Glowing Sci-Fi Metrics */
    div[data-testid="metric-container"] {
        background: rgba(10, 15, 30, 0.6);
        border: 1px solid rgba(0, 240, 255, 0.2);
        border-left: 4px solid #00F0FF;
        padding: 5% 5% 5% 10%;
        border-radius: 8px;
        box-shadow: 0 4px 15px 0 rgba(0, 240, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 8px 30px 0 rgba(0, 240, 255, 0.4);
        border: 1px solid rgba(0, 240, 255, 0.8);
        border-left: 4px solid #00F0FF;
    }
    
    div[data-testid="metric-container"] label {
        color: #94A3B8 !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 0.9rem !important;
        letter-spacing: 1.5px;
    }

    div[data-testid="metric-container"] div {
        color: #FFFFFF !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.4);
    }

    /* Cyberpunk Button */
    .stButton>button {
        background: linear-gradient(90deg, #050810 0%, #00F0FF 100%);
        background-size: 200% auto;
        color: #ffffff !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 900;
        letter-spacing: 2px;
        border: 1px solid #00F0FF;
        border-radius: 5px;
        transition: 0.5s;
        text-transform: uppercase;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.3);
    }
    
    .stButton>button:hover {
        background-position: right center;
        box-shadow: 0 0 30px rgba(0, 240, 255, 0.8);
        transform: scale(1.05);
        color: #000000 !important;
    }

    /* Neon Input Field */
    .stTextInput>div>div>input {
        background-color: rgba(0, 240, 255, 0.05) !important;
        border: 1px solid rgba(0, 240, 255, 0.4) !important;
        color: #00F0FF !important;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: bold;
        font-size: 1.1rem;
        border-radius: 5px;
        transition: 0.3s;
    }
    
    .stTextInput>div>div>input:focus {
        box-shadow: 0 0 20px rgba(0, 240, 255, 0.6) !important;
        border: 1px solid #00F0FF !important;
    }

    /* Alerts / Notifications */
    .stAlert {
        background: rgba(10, 15, 30, 0.8) !important;
        border: 1px solid #00F0FF !important;
        border-radius: 8px !important;
        backdrop-filter: blur(10px);
        color: white !important;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Hide Header */
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("🪐 ExoMind: AI-Enabled Exoplanet Detection")
st.markdown("**Bharatiya Antariksh Hackathon - Challenge 07**")

# Sidebar for User Input
st.sidebar.header("Data Ingestion")
data_source = st.sidebar.radio("Data Source", ["Live NASA API", "Local File Upload"])

tic_id = "Uploaded File"
uploaded_file = None

if data_source == "Live NASA API":
    tic_id = st.sidebar.text_input("Enter a Star's TIC ID", value="TIC 279741379")
else:
    uploaded_file = st.sidebar.file_uploader("Upload .fits or .csv light curve", type=["fits", "csv"])

analyze_btn = st.sidebar.button("Run AI Analysis")

if analyze_btn:
    time_array, flux_array = np.array([]), np.array([])
    
    if data_source == "Live NASA API":
        # 1. Fetch Data (Calls Member 1)
        with st.spinner("Downloading real light curve from NASA MAST..."):
            time_array, flux_array = get_clean_lightcurve(tic_id)
    else:
        if uploaded_file is not None:
            with st.spinner("Processing local file..."):
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
        # 2. Run AI (Calls Member 2)
        with st.spinner("Running PyTorch 1D-CNN Inference..."):
            # --- FIX FOR API MISMATCH ---
            # PyTorch model expects exactly 20000 points.
            # We will pad with the median or truncate the array.
            target_length = 20000
            if len(flux_array) > target_length:
                # Truncate
                flux_for_ai = flux_array[:target_length]
            elif len(flux_array) < target_length:
                # Pad with median flux to avoid artificial dips
                median_flux = np.median(flux_array)
                padding = np.full(target_length - len(flux_array), median_flux)
                flux_for_ai = np.concatenate([flux_array, padding])
            else:
                flux_for_ai = flux_array

            ai_result = predict_exoplanet(flux_for_ai)
            
        # 3. Calculate Math (Calls Member 3)
        with st.spinner("Calculating Transit Parameters (Box Least Squares)..."):
            stats = calculate_transit_stats(time_array, flux_array)

        # ----------------------------------------------------
        # UI: Beautiful Plotly Graph
        # ----------------------------------------------------
        st.subheader(f"Light Curve Analysis for {tic_id}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_array, 
            y=flux_array, 
            mode='markers',
            marker=dict(size=4, color='#00f2fe', opacity=0.7),
            name='Flux Data'
        ))
        
        # Make it look like a sleek, dark-mode space dashboard
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

        # ----------------------------------------------------
        # UI: Metrics Dashboard
        # ----------------------------------------------------
        st.markdown("### AI Classification & Analytics")
        
        # Highlight the AI's final decision
        if ai_result["is_planet"]:
            st.success(f"🌟 **AI VERDICT:** {ai_result['class_label']} Detected! (Confidence: {ai_result['confidence']*100:.1f}%)")
        else:
            st.warning(f"⚠️ **AI VERDICT:** {ai_result['class_label']} (Confidence: {ai_result['confidence']*100:.1f}%)")

        # Display the 5 math numbers in a clean row
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Orbital Period", f"{stats['orbital_period_days']:.4f} Days")
        col2.metric("Transit Duration", f"{stats['transit_duration_days'] * 24:.2f} Hours")
        col3.metric("Transit Depth", f"{stats['transit_depth']:.4f}")
        col4.metric("Signal-to-Noise", f"{stats['snr']:.1f}")
        col5.metric("Data Points", f"{len(flux_array)}")

else:
    st.info("👈 Enter a TIC ID in the sidebar and click 'Run AI Analysis' to test the live pipeline.")
