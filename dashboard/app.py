import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
import sys
import os

# Dynamically add both the parent directory and nested directory to Python path
# so it works on ANY computer (Mac, Windows, Linux) and from GitHub clones
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
# 🚀 THE STREAMLIT UI (YOUR JOB)
# ==========================================

# Set up the webpage
st.set_page_config(page_title="ExoMind Command Center", layout="wide", page_icon="🪐", initial_sidebar_state="collapsed")

# 1. INJECT 3D HYPERSPACE WARP BACKGROUND VIA IFRAME
hyperspace_html = """
<!DOCTYPE html>
<html>
<head>
<style>
    body, html { margin: 0; padding: 0; width: 100%; height: 100%; background-color: #050810; overflow: hidden; }
    canvas { display: block; }
</style>
</head>
<body>
<canvas id="starfield"></canvas>
<script>
    const canvas = document.getElementById('starfield');
    const ctx = canvas.getContext('2d');
    let w, h;
    const stars = [];
    const numStars = 800;
    const speed = 7; // Warp speed

    function resize() {
        w = canvas.width = window.innerWidth;
        h = canvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    class Star {
        constructor() {
            this.x = (Math.random() - 0.5) * w * 2;
            this.y = (Math.random() - 0.5) * h * 2;
            this.z = Math.random() * w;
            this.pz = this.z;
        }
        update() {
            this.z -= speed;
            if (this.z < 1) {
                this.x = (Math.random() - 0.5) * w * 2;
                this.y = (Math.random() - 0.5) * h * 2;
                this.z = w;
                this.pz = this.z;
            }
        }
        draw() {
            const sx = (this.x / this.z) * w + w / 2;
            const sy = (this.y / this.z) * h + h / 2;
            const r = (1 - this.z / w) * 3;
            
            ctx.beginPath();
            ctx.arc(sx, sy, r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0, 240, 255, ${1 - this.z/w})`;
            ctx.fill();
            this.pz = this.z;
        }
    }

    for(let i=0; i<numStars; i++) stars.push(new Star());

    function animate() {
        ctx.fillStyle = 'rgba(5, 8, 16, 0.3)'; // Trailing motion blur effect
        ctx.fillRect(0, 0, w, h);
        stars.forEach(s => { s.update(); s.draw(); });
        requestAnimationFrame(animate);
    }
    animate();
</script>
</body>
</html>
"""

import base64
b64_html = base64.b64encode(hyperspace_html.encode('utf-8')).decode('utf-8')

st.markdown(f"""
    <iframe src="data:text/html;base64,{b64_html}" 
            style="position:fixed; top:0; left:0; width:100vw; height:100vh; z-index:-999; border:none; pointer-events:none;">
    </iframe>
""", unsafe_allow_html=True)


# 2. INJECT CUSTOM CSS FOR TRANSPARENCY AND NEON STYLING
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=Orbitron:wght@500;700;900&display=swap');

    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    /* Make Streamlit's default backgrounds completely transparent */
    .stApp, .main, [data-testid="stHeader"] {
        background: transparent !important;
        color: #E2E8F0;
    }

    /* Completely hide the sidebar toggle button just in case */
    [data-testid="collapsedControl"] { display: none !important; }

    /* Sci-Fi Titles */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        color: #00F0FF !important;
        text-shadow: 0 0 20px rgba(0, 240, 255, 0.6);
        letter-spacing: 1.5px;
        text-align: center;
    }
    
    /* Control Deck Container */
    .control-deck {
        background: rgba(5, 8, 16, 0.6);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 240, 255, 0.3);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 0 30px rgba(0, 240, 255, 0.1);
        margin-bottom: 30px;
    }

    /* Glowing Sci-Fi Metrics */
    div[data-testid="metric-container"] {
        background: rgba(10, 15, 30, 0.5);
        border: 1px solid rgba(0, 240, 255, 0.2);
        border-bottom: 4px solid #00F0FF;
        padding: 5% 5% 5% 10%;
        border-radius: 8px;
        box-shadow: 0 4px 15px 0 rgba(0, 240, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        text-align: center;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 8px 30px 0 rgba(0, 240, 255, 0.4);
        border: 1px solid rgba(0, 240, 255, 0.8);
        border-bottom: 4px solid #00F0FF;
    }
    
    div[data-testid="metric-container"] label {
        color: #94A3B8 !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 0.9rem !important;
        letter-spacing: 1.5px;
    }

    div[data-testid="metric-container"] div {
        color: #FFFFFF !important;
        font-size: 1.8rem !important;
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
        width: 100%;
        height: 100%;
        min-height: 60px;
    }
    
    .stButton>button:hover {
        background-position: right center;
        box-shadow: 0 0 30px rgba(0, 240, 255, 0.8);
        transform: scale(1.02);
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
    
    /* Radio Buttons */
    div[role="radiogroup"] {
        background: rgba(0,0,0,0.4);
        padding: 10px;
        border-radius: 8px;
        border: 1px solid rgba(0, 240, 255, 0.2);
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

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================
st.markdown("<h1>🪐 ExoMind Command Center</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 1.2rem; margin-bottom: 30px;'>Bharatiya Antariksh Hackathon - AI Exoplanet Detection System</p>", unsafe_allow_html=True)


# ==========================================
# CENTRALIZED CONTROL DECK (Replaces Sidebar)
# ==========================================
st.markdown('<div class="control-deck">', unsafe_allow_html=True)
st.markdown("<h3>📡 INITIATE SCAN</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.5, 2, 1.5])

with col1:
    st.markdown("<span style='color: #00F0FF; font-weight: bold;'>1. Select Data Source</span>", unsafe_allow_html=True)
    data_source = st.radio("", ["Live NASA API", "Local File Upload"], horizontal=True, label_visibility="collapsed")

tic_id = "Uploaded File"
uploaded_file = None

with col2:
    st.markdown("<span style='color: #00F0FF; font-weight: bold;'>2. Provide Target Data</span>", unsafe_allow_html=True)
    if data_source == "Live NASA API":
        tic_id = st.text_input("", value="TIC 279741379", label_visibility="collapsed")
    else:
        uploaded_file = st.file_uploader("", type=["fits", "csv"], label_visibility="collapsed")

with col3:
    st.markdown("<span style='color: #00F0FF; font-weight: bold;'>3. Execute Pipeline</span>", unsafe_allow_html=True)
    analyze_btn = st.button("🚀 DEPLOY AI")

st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# CORE LOGIC & VISUALIZATION
# ==========================================
if analyze_btn:
    time_array, flux_array = np.array([]), np.array([])
    
    if data_source == "Live NASA API":
        # 1. Fetch Data (Calls Member 1)
        with st.spinner("Establishing uplink to NASA MAST... Downloading light curve..."):
            time_array, flux_array = get_clean_lightcurve(tic_id)
    else:
        if uploaded_file is not None:
            with st.spinner("Processing local encrypted payload..."):
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
        with st.spinner("Routing data to PyTorch 1D-CNN Matrix..."):
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
            
        # 3. Calculate Math (Calls Member 3)
        with st.spinner("Calculating Orbital Mechanics (Box Least Squares)..."):
            stats = calculate_transit_stats(time_array, flux_array)

        st.markdown("---")

        # ----------------------------------------------------
        # UI: Metrics Dashboard (HUD)
        # ----------------------------------------------------
        st.markdown("<h3>INTELLIGENCE REPORT</h3>", unsafe_allow_html=True)
        
        # Highlight the AI's final decision
        if ai_result["is_planet"]:
            st.success(f"🌟 **AI VERDICT:** {ai_result['class_label']} Detected! (Neural Confidence: {ai_result['confidence']*100:.1f}%)")
        else:
            st.warning(f"⚠️ **AI VERDICT:** {ai_result['class_label']} (Neural Confidence: {ai_result['confidence']*100:.1f}%)")

        st.write("") # Spacing

        # Display the 5 math numbers in a clean row
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Orbital Period", f"{stats['orbital_period_days']:.4f} Days")
        col2.metric("Transit Duration", f"{stats['transit_duration_days'] * 24:.2f} Hours")
        col3.metric("Transit Depth", f"{stats['transit_depth']:.4f}")
        col4.metric("Signal-to-Noise", f"{stats['snr']:.1f}")
        col5.metric("Data Points", f"{len(flux_array)}")

        st.write("") # Spacing

        # ----------------------------------------------------
        # UI: Beautiful Plotly Graph
        # ----------------------------------------------------
        st.markdown(f"<h3>Light Curve Telemetry: {tic_id}</h3>", unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_array, 
            y=flux_array, 
            mode='markers',
            marker=dict(size=4, color='#00f2fe', opacity=0.8),
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
    st.markdown("<div style='text-align: center; color: #94A3B8; margin-top: 50px;'>Awaiting Commander input... Initialize scan to begin.</div>", unsafe_allow_html=True)
