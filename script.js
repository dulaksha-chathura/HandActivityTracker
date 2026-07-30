const video = document.getElementById("webcam");
const canvas = document.getElementById("output_canvas");
const ctx = canvas.getContext("2d");
const hud = document.getElementById("hud");
const btnAction = document.getElementById("btnAction");
const videoContainer = document.getElementById("videoContainer");
const report = document.getElementById("report");

let handLandmarker = undefined;
let runningMode = "VIDEO";
let appState = "LOADING"; // LOADING, READY, CALIBRATE_P1, CALIBRATE_P2, TEST_P1, TEST_P2, FINISHED
let fingerX = null, fingerY = null;
let savedPositions = [];

// Performance Tuning Thresholds
const DISTANCE_THRESHOLD = 0.12; 
const REQUIRED_HOLD_FRAMES = 8;
const TOTAL_ALLOWED_TIME = 60;

let holdCounter = 0;
let startTime = 0;
let timerInterval = null;
let streamRef = null;

// Initialize MediaPipe Hand Landmarker
async function initializeHandLandmarker() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://jsdelivr.net"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://googleapis.com",
            delegate: "GPU"
        },
        runningMode: runningMode,
        numHands: 1
    });
    appState = "READY";
    hud.innerText = "Status: Ready to Calibrate";
    btnAction.disabled = false;
}
initializeHandLandmarker();

// Handle Button Click Actions (State machine coordinator)
btnAction.addEventListener("click", async () => {
    if (appState === "READY") {
        savedPositions = [];
        report.style.display = "none";
        appState = "CALIBRATE_P1";
        btnAction.innerText = "Save Position 1";
        videoContainer.style.display = "block";
        await startCamera();
    } else if (appState === "CALIBRATE_P1" && fingerX !== null) {
        savedPositions.push({ x: fingerX, y: fingerY, color: "#3b82f6" });
        appState = "CALIBRATE_P2";
        btnAction.innerText = "Save Position 2";
    } else if (appState === "CALIBRATE_P2" && fingerX !== null) {
        savedPositions.push({ x: fingerX, y: fingerY, color: "#eab308" });
        appState = "TEST_P1";
        btnAction.style.display = "none";
        holdCounter = 0;
        startTime = Date.now();
        startTimer();
    }
});

async function startCamera() {
    const constraints = { video: { width: 640, height: 480, facingMode: "user" } };
    streamRef = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = streamRef;
    video.addEventListener("loadeddata", predictWebcam);
}

function stopCamera() {
    if (streamRef) {
        streamRef.getTracks().forEach(track => track.stop());
    }
    videoContainer.style.display = "none";
    btnAction.style.display = "inline-block";
    btnAction.innerText = "Restart Test";
    appState = "READY";
}

function startTimer() {
    timerInterval = setInterval(() => {
        let elapsed = (Date.now() - startTime) / 1000;
        let remaining = Math.max(0, Math.ceil(TOTAL_ALLOWED_TIME - elapsed));
        
        if (appState === "TEST_P1" || appState === "TEST_P2") {
            hud.innerText = `TEST ACTIVE | Time Left: ${remaining}s`;
        }

        if (remaining <= 0 && (appState === "TEST_P1" || appState === "TEST_P2")) {
            endTask(false, 0);
        }
    }, 200);
}

function endTask(isSuccess, timeTaken) {
    clearInterval(timerInterval);
    stopCamera();
    
    report.style.display = "block";
    if (isSuccess) {
        hud.innerText = "Status: Completed!";
        report.className = "success";
        report.innerHTML = `<strong>STATUS: TASK CORRECT</strong><br>Time Taken: ${timeTaken.toFixed(2)} seconds!`;
    } else {
        hud.innerText = "Status: Failed!";
        report.className = "failed";
        report.innerHTML = `<strong>STATUS: TASK WRONG</strong><br>Reason: 60 Seconds Timeout Exceeded.`;
    }
}

let lastVideoTime = -1;
async function predictWebcam() {
    if (video.currentTime === lastVideoTime || appState === "READY") {
        if (appState !== "READY") window.requestAnimationFrame(predictWebcam);
        return;
    }
    lastVideoTime = video.currentTime;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let startTimeMs = performance.now();
    const detections = handLandmarker.detectForVideo(video, startTimeMs);

    fingerX = null; fingerY = null;

    // Parse Hand Tracking Results
    if (detections.handLandmarks && detections.handLandmarks.length > 0) {
        const landmarks = detections.handLandmarks[0];
        const indexTip = landmarks[8]; // Landmark 8 = Index Finger Tip
        fingerX = indexTip.x;
        fingerY = indexTip.y;

        // Draw green tracking dot on screen
        ctx.beginPath();
        ctx.arc(fingerX * canvas.width, fingerY * canvas.height, 10, 0, 2 * Math.PI);
        ctx.fillStyle = "#22c55e";
        ctx.fill();
    }

    // Render Saved Calibration Points
    savedPositions.forEach((pos, idx) => {
        ctx.beginPath();
        ctx.arc(pos.x * canvas.width, pos.y * canvas.height, 25, 0, 2 * Math.PI);
        ctx.lineWidth = 4;
        ctx.strokeStyle = pos.color;
        ctx.stroke();
        ctx.fillStyle = "white";
        ctx.font = "16px Arial";
        ctx.fillText(`P${idx+1}`, pos.x * canvas.width - 10, pos.y * canvas.height + 5);
    });

    // Handle Calibration State UI Contexts
    if (appState === "CALIBRATE_P1") hud.innerText = "Action: Position your finger for Target 1 and tap button.";
    if (appState === "CALIBRATE_P2") hud.innerText = "Action: Position your finger for Target 2 and tap button.";

    // Run Testing Sequential Validation Loops
    if (appState === "TEST_P1" || appState === "TEST_P2") {
        const targetIdx = appState === "TEST_P1" ? 0 : 1;
        const activeTarget = savedPositions[targetIdx];

        // Draw an outer glowing ring highlighting the current active checkpoint
        ctx.beginPath();
        ctx.arc(activeTarget.x * canvas.width, activeTarget.y * canvas.height, 35, 0, 2 * Math.PI);
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#ef4444";
        ctx.stroke();

        if (fingerX !== null) {
            let distance = Math.sqrt(Math.pow(fingerX - activeTarget.x, 2) + Math.pow(fingerY - activeTarget.y, 2));
            
            if (distance < DISTANCE_THRESHOLD) {
                holdCounter++;
                if (holdCounter >= REQUIRED_HOLD_FRAMES) {
                    holdCounter = 0;
                    if (appState === "TEST_P1") {
                        appState = "TEST_P2";
                    } else if (appState === "TEST_P2") {
                        let finalDuration = (Date.now() - startTime) / 1000;
                        endTask(true, finalDuration);
                        return; // Exit animation frame loop sequence cleanly
                    }
                }
            } else {
                holdCounter = Math.max(0, holdCounter - 1);
            }
        }
        
        // Render progress track feedback directly on top of the layout
        ctx.fillStyle = "white";
        ctx.font = "18px Arial";
        ctx.fillText(`Hold Progress: ${holdCounter}/${REQUIRED_HOLD_FRAMES}`, 20, canvas.height - 20);
    }

    if (appState !== "READY") {
        window.requestAnimationFrame(predictWebcam);
    }
}
