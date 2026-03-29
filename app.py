import streamlit as st
import cv2
import PIL.Image as Image
import numpy as np
from vision_guard import run_vision_guard

st.set_page_config(page_title="VisionGuard AI", page_icon="🚗", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    .metric-card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #30363d;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚗 VisionGuard: AI-Powered Driver Monitoring System")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Project Overview")
    st.write("""
    VisionGuard is a real-time computer vision system designed to prevent accidents caused by driver fatigue and distraction. 
    It uses MediaPipe Face Mesh to analyze facial landmarks and detect:
    - **Drowsiness**: Prolonged eye closure (EAR).
    - **Yawning**: Signs of physical fatigue (MAR).
    - **Distraction**: Looking away from the road (Head Pose).
    """)
    
    st.subheader("Key Metrics")
    st.markdown("""
    - **EAR (Eye Aspect Ratio)**: < 0.23 indicates closed eyes.
    - **MAR (Mouth Aspect Ratio)**: > 0.6 indicates yawning.
    - **Yaw/Pitch**: Measures head rotation relative to the camera.
    """)

    if st.button("🚀 Launch Live Demo"):
        st.info("Live Demo launching in a separate window. Press 'ESC' to close.")
        run_vision_guard()

with col2:
    st.header("Technical Explanation")
    
    tabs = st.tabs(["Eye Aspect Ratio", "Mouth Aspect Ratio", "Head Pose"])
    
    with tabs[0]:
        st.image("https://user-images.githubusercontent.com/38150419/54070054-d84a7e80-424a-11e9-864b-76f57356263b.png", caption="EAR Calculation Logic")
        st.code("""
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """, language="python")

    with tabs[1]:
        st.write("MAR detects the opening of the mouth. If the value exceeds a threshold for a sustained period, a 'Yawning' alert is triggered.")
        
    with tabs[2]:
        st.write("Using 3D-to-2D point mapping via SolvePnP, we estimate the Euler angles (Pitch, Yaw, Roll) to determine where the driver is looking.")

st.markdown("---")
st.caption("Developed by Hardik | BYOP Capstone Project")
