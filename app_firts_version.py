import cv2
import mediapipe as mp
import numpy as np

# 1. Base Setup for the New MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Configure the hand detector to read from a live video stream
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7
)

# 2. Predefine Your Target Positions (Normalized X, Y coordinates between 0.0 and 1.0)
TARGET_POSITIONS = [
    {"name": "Position 1 (Top-Left)", "x": 0.3, "y": 0.3, "color": (255, 0, 0)},     # Blue
    {"name": "Position 2 (Bottom-Right)", "x": 0.7, "y": 0.7, "color": (0, 255, 255)} # Yellow
]

DISTANCE_THRESHOLD = 0.08  # Spatial tolerance radius
REQUIRED_HOLD_FRAMES = 15  # Must hold for ~0.5 seconds at each spot

current_target_index = 0   # Track checkpoints
hold_counter = 0
task_status = "IN PROGRESS"
status_color = (0, 165, 255) # Orange

cap = cv2.VideoCapture(0)

# Initialize the modern landmarker inside a context manager
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Convert to RGB and wrap into MediaPipe's Image object
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # New API requires tracking specific timestamps in milliseconds
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if frame_timestamp_ms == 0:
            frame_timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

        # Run inference
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        
        # Draw target circles on screen
        for idx, target in enumerate(TARGET_POSITIONS):
            tx, ty = int(target["x"] * w), int(target["y"] * h)
            thickness = -1 if idx < current_target_index else 2
            cv2.circle(frame, (tx, ty), 25, target["color"], thickness)
            cv2.putText(frame, f"P{idx+1}", (tx-10, ty+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Process spatial logic if hand landmarks are found
        if results.hand_landmarks and current_target_index < len(TARGET_POSITIONS):
            # Track landmark index 8 (INDEX_FINGER_TIP)
            index_tip = results.hand_landmarks[0][8]
            
            # Fetch target location details
            active_target = TARGET_POSITIONS[current_target_index]
            distance = np.sqrt((index_tip.x - active_target["x"])**2 + (index_tip.y - active_target["y"])**2)
            
            # Draw tracking marker on your finger
            fx, fy = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(frame, (fx, fy), 8, (0, 255, 0), -1)

            if distance < DISTANCE_THRESHOLD:
                hold_counter += 1
                if hold_counter >= REQUIRED_HOLD_FRAMES:
                    print(f"Cleared: {active_target['name']}!")
                    current_target_index += 1
                    hold_counter = 0
            else:
                hold_counter = max(0, hold_counter - 1)

        if current_target_index >= len(TARGET_POSITIONS):
            task_status = "TASK CORRECT"
            status_color = (0, 255, 0)
        
        # UI Overlays
        cv2.putText(frame, f"Status: {task_status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        if current_target_index < len(TARGET_POSITIONS):
            cv2.putText(frame, f"Target: {TARGET_POSITIONS[current_target_index]['name']}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame, f"Hold Progress: {hold_counter}/{REQUIRED_HOLD_FRAMES}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow("Multi-Position Spatial Validation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
