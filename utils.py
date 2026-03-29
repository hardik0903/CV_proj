import cv2
import numpy as np
from scipy.spatial import distance as dist

def calculate_ear(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR)
    eye_landmarks: list of 6 (x, y) coordinates for an eye
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    """
    # Vertical distances
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    # Horizontal distance
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_mar(mouth_landmarks):
    """
    Calculate Mouth Aspect Ratio (MAR) to detect yawning
    mouth_landmarks: [left, right, top_inner, bottom_inner]
    """
    # Horizontal distance
    C = dist.euclidean(mouth_landmarks[0], mouth_landmarks[1])
    # Vertical distance (inner mouth)
    V = dist.euclidean(mouth_landmarks[2], mouth_landmarks[3])
    
    mar = V / C
    return mar

def get_head_pose(shape, img_w, img_h):
    """
    Estimate Head Pose (Yaw, Pitch, Roll) using solvePnP
    Requires specific landmark indices from Face Mesh.
    Indices: 1 (Nose), 152 (Chin), 33 (L Eye Outer), 263 (R Eye Outer), 
             61 (L Mouth), 291 (R Mouth)
    """
    # 3D model points (generic facial model)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # 2D image points from face mesh
    image_points = np.array([
        shape[0],     # Nose tip
        shape[1],     # Chin
        shape[2],     # Left eye outer
        shape[3],     # Right eye outer
        shape[4],     # Left mouth corner
        shape[5]      # Right mouth corner
    ], dtype="double")

    # Camera internals
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    # Convert rotation vector to Euler Angles
    rmat, _ = cv2.Rodrigues(rotation_vector)
    _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(np.hstack((rmat, translation_vector)))
    
    pitch, yaw, roll = angles.flatten()
    
    return pitch, yaw, roll

# Visualization constants
BOX_COLOR = (0, 255, 0)
ALERT_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
