# VisionGuard: AI-Powered Driver Monitoring System
### BYOP Capstone Project Report

**Date**: March 29, 2026  
**Author**: Hardik  
**Repo**: [https://github.com/hardik0903/CV_proj.git](https://github.com/hardik0903/CV_proj.git)

---

## 1. Problem Statement
Driver drowsiness and distraction are major causes of road accidents worldwide. Thousands of lives are lost every year due to momentary lapses in attention or falling asleep at the wheel. The goal of this project is to develop a robust, real-time monitoring system that can detect signs of fatigue and distraction using standard webcam hardware.

## 2. Approach & Methodology

### Core Technologies
- **MediaPipe Face Mesh**: Used for high-speed facial landmark extraction (468 points).
- **OpenCV**: Used for real-time video processing and visual indicators.
- **Python**: The backend for implementing mathematical models for fatigue.

### Detection Algorithms
1. **Drowsiness (EAR)**: The Eye Aspect Ratio (EAR) measures the distance between vertical and horizontal eye landmarks. A value below 0.23 for more than 15 consecutive frames triggers an alert.
2. **Yawning (MAR)**: The Mouth Aspect Ratio (MAR) detects significant mouth opening. Consistent high MAR indicates yawning.
3. **Distraction (Head Pose)**: By mapping 2D facial landmarks to a generic 3D face model and using `solvePnP`, we calculate the Pitch and Yaw of the head. If the driver looks away from the "road" (center of the screen) for more than a set angle, a distraction warning is issued.

## 3. Key Decisions
- **MediaPipe vs Dlib**: I chose MediaPipe because it is significantly faster on CPUs and provides more landmarks (468) compared to Dlib's 68, allowing for more precise pose estimation.
- **Threshold Calibration**: I chose to use dynamic thresholds based on standard research papers (EAR ~0.2-0.25) but added a "consecutive frames" check to reduce false positives from blinking.

## 4. Challenges Faced
- **Variable Lighting**: Landmark detection accuracy decreases in low-light conditions. I addressed this by using MediaPipe's robust detection model and keeping the camera gain high.
- **Head Pose Stability**: Initial pose estimation was jumpy. I smoothed the output by averaging values over a few frames and using a 3D generic model for stability.

## 5. Learnings
- **Real-time Performance**: I learned how to optimize OpenCV pipelines to maintain a high framerate (>20 FPS) while running heavy landing-mesh processing.
- **Mathematical CV**: Implementing the `solvePnP` algorithm gave me deep insights into how 2D images are projected from 3D space.
- **User Experience**: Building the Streamlit dashboard taught me how to present a technical CV project as a premium, user-friendly product.

## 6. Conclusion
VisionGuard proves that standard hardware can be leveraged to build life-saving safety systems. Future improvements could include integrating IR cameras for night vision and using 1D CNNs for sequence-based drowsiness prediction.
