import cv2
import requests
import numpy as np
import mediapipe as mp
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

stream_url = 'http://192.168.1.3/stream' 

script_dir = os.path.dirname(os.path.abspath(__file__))
pose_model_path = os.path.join(script_dir, 'tasks/pose_landmarker_lite.task')
hand_model_path = os.path.join(script_dir, 'tasks/hand_landmarker.task')

# Pose Landmarker Lite is smaller, not as good as full. The problem both need to
# find the face to work. 

pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=pose_model_path),
    output_segmentation_masks=False)

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=hand_model_path),
    num_hands=2)

# --- Function coppied from repo ---
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

def watch_stream():
    """
    Connects to the ESP32 MJPEG stream, decodes each frame, 
    applies MediaPipe tracking, and displays it.
    """
    # 1. Initialize models BEFORE starting the stream loop
    with vision.PoseLandmarker.create_from_options(pose_options) as pose_detector, \
         vision.HandLandmarker.create_from_options(hand_options) as hand_detector:
        
        try:
            r = requests.get(stream_url, stream=True, timeout=10)
            r.raise_for_status() 
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to stream: {e}")
            return

        print("Connected to stream. Press 'q' in the window to quit.")
        
        bytes_data = bytes()
        
        for chunk in r.iter_content(chunk_size=1024):
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8') 
            b = bytes_data.find(b'\xff\xd9') 
            
            if a != -1 and b != -1:
                jpg_raw = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                
                img = cv2.imdecode(np.frombuffer(jpg_raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if img is not None:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    # =========================================================
                    # MEDIAPIPE PROCESSING BLOCK
                    # Convert BGR to RGB and wrap in MediaPipe Image object
                    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                    # Run detections
                    pose_result = pose_detector.detect(mp_image)
                    hand_result = hand_detector.detect(mp_image)

                    # Draw landmarks on the RGB frame
                    annotated_image = draw_landmarks_on_image(rgb_frame, pose_result, hand_result)

                    # Convert back to BGR so OpenCV displays colors correctly
                    img = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                    # =========================================================

                    cv2.imshow('ESP32-CAM AI Stream', img)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Quitting stream.")
                        break
        
        cv2.destroyAllWindows()
        r.close()

if __name__ == '__main__':
    watch_stream()