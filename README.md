# Spatial Task Quality Assurance & Hand Activity Tracker Engine 

A real-time, client-side computer vision application designed to map, track, and verify hand spatial movements. Built using **MediaPipe Tasks API**, **JavaScript/HTML5 Canvas**, and **Python OpenCV**, this tool calibrates spatial coordinates to verify precision tasks with dwell-time detection.

 **Live Web Demo:** [https://dulaksha-chathura.github.io/HandActivityTracker/](https://dulaksha-chathura.github.io/HandActivityTracker/)

---

## Key Features

* **Real-time Fingertip Tracking:** Tracks index finger tip (Landmark 8) position with visual UI overlays.
* **Two-Point Spatial Calibration:** Allows users to set up custom target points (`Position 1` & `Position 2`).
* **Dwell Verification Engine:** Requires continuous target holding (`15 frames`) within a distance threshold (`0.08 normalized units`) to complete tasks.
* **Live HUD & Timers:** Real-time countdown timer (60s limit), status telemetry, and hold progress indicators.
* **Audio Feedback:** Synthesized audio cues for position saves, successful completions, and failure timeouts.
* **Dual Implementation:** Includes both a **Web-native (JS/WASM)** interface and equivalent **Python/OpenCV** scripts.

---

## 🛠️ Project Structure

```text
├── index.html               # Main Web App (HTML5, JS, MediaPipe Tasks WASM)
├── calibrate.py             # Python script for saving calibration data
├── test.py                  # Python script for running spatial task verification
├── calibration_data.json    # Generated target points from Python calibration
└── hand_landmarker.task     # MediaPipe Hand Landmarker model file (for Python)

# ==============================================================================
# GETTING STARTED WITH HAND ACTIVITY TRACKER
# ==============================================================================

# ------------------------------------------------------------------------------
# OPTION 1: WEB VERSION (RECOMMENDED)
# ------------------------------------------------------------------------------
# Simply open the live web application in any modern web browser with webcam access:
#  https://dulaksha-chathura.github.io/HandActivityTracker/

# If you downloaded the source files and want to run the web version locally,
# serve index.html via an HTTP server (due to browser CORS restrictions):
python -m http.server 8000
# Open http://localhost:8000 in your browser.


# ------------------------------------------------------------------------------
# OPTION 2: PYTHON VERSION
# ------------------------------------------------------------------------------

# Step 1: Install prerequisites
pip install opencv-python mediapipe numpy

# Step 2: Download Model File
# Download 'hand_landmarker.task' from MediaPipe and place it in the root directory.

# Step 3: Run Calibration
python calibrate.py
# -> Move your index finger to Position 1 and press 'C' to save.
# -> Move your index finger to Position 2 and press 'C' to save.
# -> This outputs 'calibration_data.json'.

# Step 4: Run Verification Test
python test.py
# -> Target the red circle representing Position 1 until progress completes.
# -> Target Position 2 to complete the task before the 60-second timer expires.
