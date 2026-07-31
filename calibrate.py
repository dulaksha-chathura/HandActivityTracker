import cv2
import mediapipe as mp
import json

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

app_state = "CALIBRATE_P1"
saved_positions = [] 
cap = cv2.VideoCapture(0)

print("Starting Calibration... Move your index finger and press 'C' to save a position.")

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
        
        # FIXED LOGIC: Correctly checking the nested list structures for the modern API
        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            first_hand = results.hand_landmarks[0] # Get the first hand detected
            if len(first_hand) > 8:
                index_tip = first_hand[8] # Landmark 8 is INDEX_FINGER_TIP
                finger_x, finger_y = index_tip.x, index_tip.y
                cv2.circle(frame, (int(finger_x * w), int(finger_y * h)), 12, (0, 255, 0), -1)

        # Draw already saved calibration targets
        for idx, pos in enumerate(saved_positions):
            cv2.circle(frame, (int(pos["x"] * w), int(pos["y"] * h)), 20, (255, 165, 0), -1)

        if app_state == "CALIBRATE_P1":
            status_text = "Setup Position 1. Press 'C' to save."
        elif app_state == "CALIBRATE_P2":
            status_text = "Setup Position 2. Press 'C' to save."

        cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        cv2.imshow("Calibration Mode", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and finger_x is not None and finger_y is not None:
            if app_state == "CALIBRATE_P1":
                saved_positions.append({"x": finger_x, "y": finger_y})
                app_state = "CALIBRATE_P2"
                print("→ Position 1 Saved!")
            elif app_state == "CALIBRATE_P2":
                saved_positions.append({"x": finger_x, "y": finger_y})
                print("→ Position 2 Saved!")
                
                with open("calibration_data.json", "w") as f:
                    json.dump(saved_positions, f)
                print("\n[SUCCESS] calibration_data.json file created successfully!")
                break

cap.release()
cv2.destroyAllWindows()
