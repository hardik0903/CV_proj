import cv2
import mediapipe as mp
import time
import numpy as np
from utils import calculate_ear, calculate_mar, get_head_pose, ALERT_COLOR, BOX_COLOR, TEXT_COLOR, FONT

# Thresholds
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 15
MOUTH_AR_THRESH = 0.6
DISTRACTION_YAW_THRESH = 35
DISTRACTION_PITCH_THRESH = 25

# MediaPipe Initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark Indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 13, 14] # corners, top, bottom
POSE_POINTS = [1, 152, 33, 263, 61, 291] # Nose, Chin, L Eye corner, R Eye corner, L Mouth, R Mouth

def run_vision_guard(source=0):
    cap = cv2.VideoCapture(source)
    counter = 0
    alarm_on = False
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        status_text = "Status: Attentive"
        current_color = BOX_COLOR

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Extract coordinates
                left_eye_pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_EYE]
                right_eye_pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_EYE]
                mouth_pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in MOUTH]
                pose_pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in POSE_POINTS]

                # Calculations
                left_ear = calculate_ear(left_eye_pts)
                right_ear = calculate_ear(right_eye_pts)
                ear = (left_ear + right_ear) / 2.0
                mar = calculate_mar(mouth_pts)
                pitch, yaw, roll = get_head_pose(pose_pts, w, h)

                # Drowsiness Logic
                if ear < EYE_AR_THRESH:
                    counter += 1
                    if counter >= EYE_AR_CONSEC_FRAMES:
                        status_text = "WARNING: DROWSY!"
                        current_color = ALERT_COLOR
                else:
                    counter = 0

                # Yawn Logic
                if mar > MOUTH_AR_THRESH:
                    status_text = "WARNING: YAWNING!"
                    current_color = ALERT_COLOR

                # Distraction Logic
                if abs(yaw) > DISTRACTION_YAW_THRESH or abs(pitch) > DISTRACTION_PITCH_THRESH:
                    status_text = "WARNING: DISTRACTED!"
                    current_color = ALERT_COLOR

                # Visualizations
                cv2.putText(image, f"EAR: {ear:.2f}", (10, 30), FONT, 0.7, TEXT_COLOR, 2)
                cv2.putText(image, f"MAR: {mar:.2f}", (10, 60), FONT, 0.7, TEXT_COLOR, 2)
                cv2.putText(image, f"Yaw: {yaw:.1f}", (10, 90), FONT, 0.7, TEXT_COLOR, 2)
                
                # Draw status box
                cv2.rectangle(image, (0, h-50), (w, h), current_color, -1)
                cv2.putText(image, status_text, (w//2 - 100, h-15), FONT, 1, (0,0,0), 2)

        cv2.imshow('VisionGuard - Driver Monitoring', image)

        if cv2.waitKey(5) & 0xFF == 27: # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vision_guard()
