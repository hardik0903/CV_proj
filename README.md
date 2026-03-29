# VisionGuard: AI-Powered Driver Monitoring System

**VisionGuard** is a real-time Computer Vision solution designed to enhance road safety by monitoring driver attentiveness. It uses advanced facial landmark analysis to detect signs of drowsiness, distraction, and fatigue.

![VisionGuard Demo](https://via.placeholder.com/800x400.png?text=VisionGuard+Driver+Monitoring+System+Demo)

## 🚀 Features
- **Real-time Drowsiness Detection**: Monitors Eye Aspect Ratio (EAR) and alerts the driver after sustained eye closure.
- **Yawn Detection**: Analyzes Mouth Aspect Ratio (MAR) to detect physical signs of fatigue.
- **Distraction Monitoring**: Estimates head pose (Yaw and Pitch) to ensure the driver is looking at the road.
- **Aesthetic Overlay**: Professional, high-contrast visual indicators for live feedback.
- **Interactive Dashboard**: A Streamlit interface for easy demonstration and technical insights.

## 🛠️ Tech Stack
- **Language**: Python 3.9+
- **Computer Vision**: OpenCV, MediaPipe (Face Mesh)
- **Math & Logic**: NumPy, SciPy
- **UI Framework**: Streamlit

## 📦 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/hardik0903/CV_proj.git
   cd CV_proj
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard**
   ```bash
   streamlit run app.py
   ```

4. **Launch Live Demo Directly**
   ```bash
   python vision_guard.py
   ```

## 📐 How it Works
1. **Face Mesh**: MediaPipe extracts 468 facial landmarks in 3D.
2. **EAR calculation**: Computes the ratio between the vertical and horizontal Eye distance.
3. **MAR calculation**: Computes the ratio of the inner mouth opening.
4. **SolvePnP**: Estimates head orientation (Yaw, Pitch, Roll) using a 3D generic facial model.

## 📄 License
MIT License - Feel free to use and improve for road safety research!
