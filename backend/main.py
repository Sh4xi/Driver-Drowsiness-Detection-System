"""
Driver Drowsiness Detection System
===================================
Real-time drowsiness detection using MediaPipe Face Mesh and OpenCV.
Detects eye closure (EAR) and yawning (MAR) to alert drowsy drivers.

Author: Your Name
License: MIT
"""

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template
from flask_cors import CORS
import time
from datetime import datetime
import threading
from collections import deque

# ============================================================================
# Flask Application Setup
# ============================================================================

app = Flask(__name__, template_folder='../frontend')
CORS(app)

print("✅ Imports loaded successfully")

# ============================================================================
# MediaPipe Face Mesh - Lazy Loading
# ============================================================================

mp_face_mesh = None
face_mesh = None
face_mesh_lock = threading.Lock()

def initialize_mediapipe():
    """Initialize MediaPipe Face Mesh on first use to prevent startup delay."""
    global mp_face_mesh, face_mesh
    
    with face_mesh_lock:
        if face_mesh is None:
            print("🔄 Initializing MediaPipe Face Mesh...")
            try:
                import mediapipe as mp
                mp_face_mesh = mp.solutions.face_mesh
                face_mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7
                )
                print("✅ MediaPipe initialized successfully!")
                return True
            except Exception as e:
                print(f"❌ MediaPipe initialization failed: {e}")
                return False
        return True

# ============================================================================
# Configuration & Thresholds
# ============================================================================

# Facial landmark indices for eyes and mouth
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 0, 17, 269, 405, 181, 314]

# Detection thresholds
EAR_THRESHOLD = 0.23           # Eye Aspect Ratio (lower = more closed)
MAR_THRESHOLD = 1.20           # Mouth Aspect Ratio (higher = more open)
CLOSED_FRAMES_THRESHOLD = 25   # Frames needed for drowsiness alert (~0.8s at 30 FPS)
YAWN_FRAMES_THRESHOLD = 20     # Frames needed for yawn detection (~0.6s at 30 FPS)

# Smoothing and performance
SMOOTHING_WINDOW = 3           # Moving average window size

# Debug mode - prints real-time EAR/MAR values to terminal
CALIBRATION_MODE = False

# ============================================================================
# System State Management
# ============================================================================

class SystemState:
    """Tracks all system metrics and detection states."""
    
    def __init__(self):
        # Core metrics
        self.ear = 0.0
        self.mar = 0.0
        self.is_drowsy = False
        self.is_yawning = False
        self.face_detected = False
        
        # Counters
        self.closed_frames = 0
        self.yawn_frames = 0
        self.drowsy_count = 0
        self.yawn_count = 0
        self.alert_count = 0
        
        # Session tracking
        self.session_start = datetime.now()
        self.last_alert_time = 0
        
        # Performance metrics
        self.fps = 0
        self.brightness = 0
        self.confidence = 0
        
        # Smoothing buffers
        self.ear_buffer = deque(maxlen=SMOOTHING_WINDOW)
        self.mar_buffer = deque(maxlen=SMOOTHING_WINDOW)

state = SystemState()

# ============================================================================
# Camera Management
# ============================================================================

camera_lock = threading.Lock()
camera = None
camera_initialized = False

def get_camera():
    """Initialize camera with DirectShow on Windows for better performance."""
    global camera, camera_initialized
    
    with camera_lock:
        if not camera_initialized:
            print("🎥 Initializing camera...")
            try:
                # Try DirectShow first (Windows)
                camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                time.sleep(0.3)
                
                if not camera.isOpened():
                    print("⚠️  DirectShow failed, trying default...")
                    camera = cv2.VideoCapture(0)
                    time.sleep(0.3)
                
                if camera.isOpened():
                    # Configure camera settings
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                    
                    print("✅ Camera initialized successfully!")
                    camera_initialized = True
                else:
                    print("❌ Failed to open camera")
                    camera = None
            except Exception as e:
                print(f"❌ Camera error: {e}")
                camera = None
                
        return camera

# ============================================================================
# Detection Algorithms
# ============================================================================

def euclidean_distance(point_a, point_b):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(point_a - point_b)

def calculate_EAR(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR).
    EAR decreases when eye closes.
    """
    # Vertical distances
    vertical_1 = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    vertical_2 = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    
    # Horizontal distance
    horizontal = euclidean_distance(eye_landmarks[0], eye_landmarks[3])
    
    # EAR formula
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def calculate_MAR(mouth_landmarks):
    """
    Calculate Mouth Aspect Ratio (MAR).
    MAR increases when mouth opens (yawning).
    """
    # Vertical distances
    vertical_1 = euclidean_distance(mouth_landmarks[1], mouth_landmarks[7])
    vertical_2 = euclidean_distance(mouth_landmarks[2], mouth_landmarks[6])
    vertical_3 = euclidean_distance(mouth_landmarks[3], mouth_landmarks[5])
    
    # Horizontal distance
    horizontal = euclidean_distance(mouth_landmarks[0], mouth_landmarks[4])
    
    # MAR formula
    mar = (vertical_1 + vertical_2 + vertical_3) / (3.0 * horizontal)
    return mar

def smooth_value(buffer, new_value):
    """Apply moving average smoothing to reduce noise."""
    buffer.append(new_value)
    return np.mean(buffer)

def calculate_brightness(frame):
    """Calculate average brightness of frame for lighting assessment."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return int(np.mean(gray))

def calculate_confidence(face_landmarks, frame_width, frame_height, brightness):
    """
    Calculate detection confidence based on multiple factors:
    - Face size and position
    - Lighting conditions
    - Landmark quality
    """
    # Extract face boundaries
    x_coords = [int(landmark.x * frame_width) for landmark in face_landmarks.landmark]
    y_coords = [int(landmark.y * frame_height) for landmark in face_landmarks.landmark]
    
    face_width = max(x_coords) - min(x_coords)
    face_height = max(y_coords) - min(y_coords)
    
    # Factor 1: Face size (optimal: 20-80% of frame width)
    size_ratio = face_width / frame_width
    if 0.2 <= size_ratio <= 0.8:
        size_confidence = 100
    elif size_ratio < 0.2:
        size_confidence = min(100, int((size_ratio / 0.2) * 100))
    else:
        size_confidence = max(50, int((1.0 - (size_ratio - 0.8)) * 100))
    
    # Factor 2: Face position (centered is optimal)
    face_center_x = (min(x_coords) + max(x_coords)) / 2
    face_center_y = (min(y_coords) + max(y_coords)) / 2
    center_offset_x = abs(face_center_x - frame_width/2) / (frame_width/2)
    center_offset_y = abs(face_center_y - frame_height/2) / (frame_height/2)
    position_confidence = int(100 * (1 - (center_offset_x + center_offset_y) / 2))
    
    # Factor 3: Lighting conditions (optimal: 80-200)
    if 80 <= brightness <= 200:
        lighting_confidence = 100
    elif brightness < 80:
        lighting_confidence = int((brightness / 80) * 100)
    else:
        lighting_confidence = max(60, int((255 - brightness) / 55 * 100))
    
    # Factor 4: MediaPipe landmark quality
    landmark_confidence = 95
    
    # Weighted average
    confidence = int(
        size_confidence * 0.30 +
        position_confidence * 0.20 +
        lighting_confidence * 0.25 +
        landmark_confidence * 0.25
    )
    
    return confidence, (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

# ============================================================================
# Frame Processing
# ============================================================================

def process_frame(frame):
    """
    Process video frame for drowsiness detection.
    Returns annotated frame with detection overlays.
    """
    global face_mesh
    
    height, width = frame.shape[:2]
    state.brightness = calculate_brightness(frame)
    
    # Initialize MediaPipe on first frame
    if face_mesh is None:
        if not initialize_mediapipe():
            cv2.putText(frame, "Face Detection Unavailable", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    try:
        result = face_mesh.process(rgb_frame)
    except Exception as e:
        print(f"⚠️  Face processing error: {e}")
        state.face_detected = False
        return frame
    
    if result.multi_face_landmarks:
        state.face_detected = True
        
        for face_landmarks in result.multi_face_landmarks:
            
            # Extract eye landmarks
            left_eye = np.array([[int(face_landmarks.landmark[i].x * width),
                                 int(face_landmarks.landmark[i].y * height)]
                                for i in LEFT_EYE])
            
            right_eye = np.array([[int(face_landmarks.landmark[i].x * width),
                                  int(face_landmarks.landmark[i].y * height)]
                                 for i in RIGHT_EYE])
            
            # Extract mouth landmarks
            mouth = np.array([[int(face_landmarks.landmark[i].x * width),
                              int(face_landmarks.landmark[i].y * height)]
                             for i in MOUTH])
            
            # Calculate ratios
            left_ear = calculate_EAR(left_eye)
            right_ear = calculate_EAR(right_eye)
            raw_ear = (left_ear + right_ear) / 2.0
            raw_mar = calculate_MAR(mouth)
            
            # Apply smoothing
            state.ear = smooth_value(state.ear_buffer, raw_ear)
            state.mar = smooth_value(state.mar_buffer, raw_mar)
            
            # Calculate confidence and face bounds
            state.confidence, face_bounds = calculate_confidence(
                face_landmarks, width, height, state.brightness
            )
            x_min, y_min, x_max, y_max = face_bounds
            
            # Add padding to face box
            x_min = max(0, x_min - 30)
            y_min = max(0, y_min - 50)
            x_max = min(width, x_max + 30)
            y_max = min(height, y_max + 30)
            
            # Draw face bounding box
            box_color = (0, 255, 0) if not state.is_drowsy else (0, 0, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 3)
            
            # Drowsiness detection
            if state.ear < EAR_THRESHOLD:
                state.closed_frames += 1
                
                if state.closed_frames > CLOSED_FRAMES_THRESHOLD:
                    if not state.is_drowsy:
                        state.drowsy_count += 1
                        state.alert_count += 1
                        state.is_drowsy = True
                        state.last_alert_time = time.time()
                        print(f"🚨 DROWSINESS ALERT! EAR: {state.ear:.3f}")
            else:
                # Fast recovery when eyes open
                if state.is_drowsy and state.closed_frames > 0:
                    state.closed_frames = max(0, state.closed_frames - 10)
                    if state.closed_frames < 5:
                        state.closed_frames = 0
                        state.is_drowsy = False
                        print("✅ Eyes open - Alert cleared!")
                else:
                    state.closed_frames = max(0, state.closed_frames - 2)
            
            # Yawn detection
            if state.mar > MAR_THRESHOLD:
                state.yawn_frames += 1
                if state.yawn_frames > YAWN_FRAMES_THRESHOLD:
                    if not state.is_yawning:
                        state.yawn_count += 1
                        print(f"🥱 YAWN DETECTED! MAR: {state.mar:.3f}")
                    state.is_yawning = True
            else:
                # Fast recovery when mouth closes
                if state.is_yawning and state.yawn_frames > 0:
                    state.yawn_frames = max(0, state.yawn_frames - 8)
                    if state.yawn_frames < 3:
                        state.yawn_frames = 0
                        state.is_yawning = False
                else:
                    state.yawn_frames = max(0, state.yawn_frames - 2)
            
            # Draw alert overlay if drowsy
            if state.is_drowsy:
                draw_alert_overlay(frame, width, height)
            
            # Draw yawn indicator
            if state.is_yawning:
                cv2.putText(frame, "YAWNING", (50, height - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            
            # Debug mode - print calibration values
            if CALIBRATION_MODE:
                print_calibration_info(raw_ear, raw_mar)
                
    else:
        state.face_detected = False
        state.ear = 0.0
        state.mar = 0.0
        
        # Gradual counter reset when face not detected
        state.closed_frames = max(0, state.closed_frames - 1)
        state.yawn_frames = max(0, state.yawn_frames - 1)
    
    return frame

def draw_alert_overlay(frame, width, height):
    """Draw drowsiness alert overlay on frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Main alert text
    alert_text = "DROWSINESS DETECTED"
    (text_width, text_height), baseline = cv2.getTextSize(
        alert_text, font, 1.2, 3
    )
    
    text_x = (width - text_width) // 2
    text_y = 80
    padding = 15
    
    # Draw background
    cv2.rectangle(frame,
                 (text_x - padding, text_y - text_height - padding),
                 (text_x + text_width + padding, text_y + baseline + padding),
                 (0, 0, 0), -1)
    
    # Draw border
    cv2.rectangle(frame,
                 (text_x - padding, text_y - text_height - padding),
                 (text_x + text_width + padding, text_y + baseline + padding),
                 (0, 0, 255), 3)
    
    # Draw text
    cv2.putText(frame, alert_text, (text_x, text_y),
               font, 1.2, (0, 0, 255), 3)
    
    # Secondary warning
    break_text = "TAKE A BREAK!"
    (break_width, break_height), _ = cv2.getTextSize(break_text, font, 0.9, 2)
    break_x = (width - break_width) // 2
    break_y = text_y + 60
    
    cv2.rectangle(frame,
                 (break_x - padding, break_y - break_height - padding),
                 (break_x + break_width + padding, break_y + baseline + padding),
                 (0, 0, 0), -1)
    
    cv2.rectangle(frame,
                 (break_x - padding, break_y - break_height - padding),
                 (break_x + break_width + padding, break_y + baseline + padding),
                 (255, 255, 0), 3)
    
    cv2.putText(frame, break_text, (break_x, break_y),
               font, 0.9, (255, 255, 0), 2)
    
    # Red overlay effect
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

def print_calibration_info(raw_ear, raw_mar):
    """Print real-time calibration information for tuning thresholds."""
    drowsy_indicator = "🔴 DROWSY!" if state.ear < EAR_THRESHOLD else "🟢 AWAKE"
    yawn_indicator = "🟡 YAWN!" if state.mar > MAR_THRESHOLD else "⚪ NORMAL"
    
    print(f"\n{'='*60}")
    print(f"👁️  EAR: {state.ear:.3f} (Raw: {raw_ear:.3f}) | "
          f"Threshold: {EAR_THRESHOLD:.3f} {drowsy_indicator}")
    print(f"   Closed frames: {state.closed_frames}/{CLOSED_FRAMES_THRESHOLD}")
    print(f"👄 MAR: {state.mar:.3f} (Raw: {raw_mar:.3f}) | "
          f"Threshold: {MAR_THRESHOLD:.3f} {yawn_indicator}")
    print(f"   Yawn frames: {state.yawn_frames}/{YAWN_FRAMES_THRESHOLD}")
    print(f"{'='*60}\n")

# ============================================================================
# Video Streaming
# ============================================================================

def generate_frames():
    """Generate video frames for streaming to web interface."""
    cam = get_camera()
    
    if cam is None:
        # Send error frame if camera unavailable
        while True:
            error_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(error_frame, "CAMERA NOT AVAILABLE", (400, 360),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(error_frame, "Check camera permissions", (380, 420),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', error_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
        return
    
    print("📹 Starting video stream...")
    prev_time = time.time()
    
    try:
        while True:
            with camera_lock:
                success, frame = cam.read()
            
            if not success:
                time.sleep(0.01)
                continue
            
            # Calculate FPS
            current_time = time.time()
            state.fps = int(1 / (current_time - prev_time)) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            # Process frame for detection
            frame = process_frame(frame)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    except GeneratorExit:
        print("📹 Video stream closed")
    except Exception as e:
        print(f"❌ Error in video stream: {e}")

# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def index():
    """Serve main dashboard."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """API endpoint for current system status."""
    session_duration = (datetime.now() - state.session_start).total_seconds()
    
    return jsonify({
        'ear': round(state.ear, 3),
        'mar': round(state.mar, 3),
        'is_drowsy': state.is_drowsy,
        'is_yawning': state.is_yawning,
        'face_detected': state.face_detected,
        'drowsy_count': state.drowsy_count,
        'yawn_count': state.yawn_count,
        'alert_count': state.alert_count,
        'fps': state.fps,
        'brightness': state.brightness,
        'confidence': state.confidence,
        'session_duration': int(session_duration),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/emergency', methods=['POST'])
def send_emergency():
    """API endpoint for emergency alert."""
    print("🚨 EMERGENCY ALERT TRIGGERED!")
    print(f"Time: {datetime.now()}")
    print(f"Drowsy Count: {state.drowsy_count}")
    
    state.alert_count += 1
    
    return jsonify({
        'success': True,
        'message': 'Emergency alert sent',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/export', methods=['GET'])
def export_data():
    """API endpoint for exporting session data."""
    session_duration = (datetime.now() - state.session_start).total_seconds()
    
    return jsonify({
        'session_start': state.session_start.isoformat(),
        'session_duration': session_duration,
        'drowsy_events': state.drowsy_count,
        'yawns': state.yawn_count,
        'alerts_sent': state.alert_count,
        'export_time': datetime.now().isoformat()
    })

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🚗 DRIVER DROWSINESS DETECTION SYSTEM")
    print("=" * 60)
    print("✅ Flask server starting...")
    print("🌐 Server: http://localhost:5000")
    if CALIBRATION_MODE:
        print("📊 Calibration mode ENABLED - Check terminal for values")
    print("=" * 60)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        if camera is not None:
            camera.release()
            print("📷 Camera released")
    except Exception as e:
        print(f"❌ Server error: {e}")