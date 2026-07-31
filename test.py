import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os

if not os.path.exists("calibration_data.json"):
    print("[ERROR] No calibration file found! Please run 'calibrate.py' first.")
    exit()

with open("calibration_data.json", "r") as f:
    saved_positions = json.load(f)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7
)

DISTANCE_THRESHOLD = 0.08  
REQUIRED_HOLD_FRAMES = 15  
TOTAL_ALLOWED_TIME = 60    

app_state = "TEST_P1" 
hold_counter = 0
time_taken = 0
start_test_time = time.time()

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if frame_timestamp_ms == 0:
            frame_timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        
        finger_x, finger_y = None, None
        
        # FIXED LOGIC: Clean nested tracking data processing
        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            first_hand = results.hand_landmarks[0]
            if len(first_hand) > 8:
                index_tip = first_hand[8]
                finger_x, finger_y = index_tip.x, index_tip.y
                cv2.circle(frame, (int(finger_x * w), int(finger_y * h)), 12, (0, 255, 0), -1)

        elapsed_time = time.time() - start_test_time
        remaining_time = max(0, int(TOTAL_ALLOWED_TIME - elapsed_time))
        
        if remaining_time <= 0:
            app_state = "TIMEOUT"
            break

        target_idx = 0 if app_state == "TEST_P1" else 1
        active_target = saved_positions[target_idx]
        
        tx, ty = int(active_target["x"] * w), int(active_target["y"] * h)
        cv2.circle(frame, (tx, ty), 30, (0, 0, 255), 2) 
        
        if finger_x is not None and finger_y is not None:
            distance = np.sqrt((finger_x - active_target["x"])**2 + (finger_y - active_target["y"])**2)
            
            if distance < DISTANCE_THRESHOLD:
                hold_counter += 1
                if hold_counter >= REQUIRED_HOLD_FRAMES:
                    hold_counter = 0
                    if app_state == "TEST_P1":
                        app_state = "TEST_P2"
                    elif app_state == "TEST_P2":
                        time_taken = time.time() - start_test_time
                        app_state = "SUCCESS"
                        break
            else:
                hold_counter = max(0, hold_counter - 1)

        cv2.putText(frame, f"TIME LEFT: {remaining_time}s", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Targeting: Position {target_idx + 1}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Hold Progress: {hold_counter}/{REQUIRED_HOLD_FRAMES}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Testing Mode", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*40)
if app_state == "SUCCESS":
    print("STATUS: TASK CORRECT")
    print(f"Time Taken to complete: {time_taken:.2f} seconds")
else:
    print("STATUS: TASK WRONG")
    print("Reason: Timeout. 60 seconds exceeded.")
print("="*40)
