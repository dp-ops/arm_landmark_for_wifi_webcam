import cv2  

import numpy as np
import mediapipe as mp
import os
import time 

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

script_dir = os.path.dirname(os.path.abspath(__file__))
pose_model_path = os.path.join(script_dir, 'tasks/pose_landmarker_lite.task') 
hand_model_path = os.path.join(script_dir, 'tasks/hand_landmarker.task')

def draw_landmarks_on_image(rgb_image, pose_result, hand_result):
    annotated_image = np.copy(rgb_image)

    # Draw Pose
    if pose_result and pose_result.pose_landmarks:
        pose_style = drawing_styles.get_default_pose_landmarks_style()
        pose_connections = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
        
        for pose_landmarks in pose_result.pose_landmarks:
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=pose_style,
                connection_drawing_spec=pose_connections)

    # Draw Hands
    if hand_result and hand_result.hand_landmarks:
        hand_landmark_style = drawing_styles.get_default_hand_landmarks_style()
        hand_connection_style = drawing_styles.get_default_hand_connections_style()
        
        for hand_landmarks in hand_result.hand_landmarks:
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=vision.HandLandmarksConnections.HAND_CONNECTIONS,
                landmark_drawing_spec=hand_landmark_style,
                connection_drawing_spec=hand_connection_style)

    return annotated_image

# --- UPDATED: Options with Lower Confidence and VIDEO Mode ---
pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=pose_model_path),
    running_mode=vision.RunningMode.VIDEO, # Enables temporal tracking
    min_pose_detection_confidence=0.3,     # Helps find the body initially without face
    min_pose_presence_confidence=0.3,      # Helps maintain the body without face
    min_tracking_confidence=0.3,           # Helps track fast arm movements
    output_segmentation_masks=False)

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=hand_model_path),
    running_mode=vision.RunningMode.VIDEO, 
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
    num_hands=2)

cap = cv2.VideoCapture(0)

# VIDEO mode requires strictly increasing timestamps. 
# We grab a start time to calculate milliseconds later.
start_time = time.time()

# --- Live Feed Loop ---
with vision.PoseLandmarker.create_from_options(pose_options) as pose_detector, \
     vision.HandLandmarker.create_from_options(hand_options) as hand_detector:
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Calculate timestamp in milliseconds
        timestamp_ms = int((time.time() - start_time) * 1000)

        # UPDATED: Use detect_for_video() instead of detect()
        pose_result = pose_detector.detect_for_video(mp_image, timestamp_ms)
        hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)

        annotated_image = draw_landmarks_on_image(rgb_frame, pose_result, hand_result)

        display_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        
        cv2.imshow('MediaPipe Full Arm & Hand Tracking', display_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()