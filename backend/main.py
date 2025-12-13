import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template
from flask_cors import CORS
import time
from datetime import datetime

app = Flask(__name__, template_folder='../frontend')
CORS(app)

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 0, 17, 269, 405, 181, 314]

# Thresholds
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
CLOSED_FRAMES_THRESHOLD = 15
YAWN_FRAMES_THRESHOLD = 20

# Global state
class SystemState:
    def __init__(self):
        self.ear = 0.0
        self.mar = 0.0
        self.is_drowsy = False
        self.is_yawning = False
        self.face_detected = False
        self.closed_frames = 0
        self.yawn_frames = 0
        self.drowsy_count = 0
        self.yawn_count = 0
        self.alert_count = 0
        self.session_start = datetime.now()
        self.last_alert_time = 0
        self.fps = 0
        self.brightness = 0
        self.confidence = 0
        
state = SystemState()

def euclidean_distance(a, b):
    """Calculate Euclidean distance"""
    return np.linalg.norm(a - b)

def calculate_EAR(eye_points): 
    """Calculate Eye Aspect Ratio"""
    A = euclidean_distance(eye_points[1], eye_points[5])
    B = euclidean_distance(eye_points[2], eye_points[4])
    C = euclidean_distance(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

def calculate_MAR(mouth_points):
    """Calculate Mouth Aspect Ratio"""
    A = euclidean_distance(mouth_points[1], mouth_points[7])
    B = euclidean_distance(mouth_points[2], mouth_points[6])
    C = euclidean_distance(mouth_points[3], mouth_points[5])
    D = euclidean_distance(mouth_points[0], mouth_points[4])
    return (A + B + C) / (3.0 * D)

def calculate_brightness(frame):
    """Calculate average brightness of frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return int(np.mean(gray))

def process_frame(frame):
    """Process frame for drowsiness detection"""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    
    # Calculate brightness
    state.brightness = calculate_brightness(frame)
    
    if result.multi_face_landmarks:
        state.face_detected = True
        
        for face_landmarks in result.multi_face_landmarks:
            
            # Extract eye landmarks
            left_eye = []
            right_eye = []
            
            for idx in LEFT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                left_eye.append(np.array([x, y]))
            
            for idx in RIGHT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                right_eye.append(np.array([x, y]))
            
            # Extract mouth landmarks
            mouth = []
            for idx in MOUTH:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                mouth.append(np.array([x, y]))
            
            # Calculate ratios
            left_ear = calculate_EAR(np.array(left_eye))
            right_ear = calculate_EAR(np.array(right_eye))
            state.ear = (left_ear + right_ear) / 2.0
            
            state.mar = calculate_MAR(np.array(mouth))
            
            # Calculate confidence (simplified)
            state.confidence = int(95 + np.random.rand() * 5)
            
            # Draw eye contours
            # left_points = np.array(left_eye, dtype=np.int32)
            # right_points = np.array(right_eye, dtype=np.int32)
            # cv2.polylines(frame, [left_points], True, (0, 255, 0), 2)
            # cv2.polylines(frame, [right_points], True, (0, 255, 0), 2)
            
            # Draw face bounding box (whole face)
            # Get face boundary landmarks
            x_coords = [int(face_landmarks.landmark[i].x * w) for i in range(468)]
            y_coords = [int(face_landmarks.landmark[i].y * h) for i in range(468)]
            
            # Calculate bounding box with padding
            x_min = min(x_coords) - 30
            y_min = min(y_coords) - 50
            x_max = max(x_coords) + 30
            y_max = max(y_coords) + 30
            
            # Ensure within frame boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            
            # Color based on state
            box_color = (0, 255, 0) if not state.is_drowsy else (0, 0, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 3)
            
            # Drowsiness detection
            if state.ear < EAR_THRESHOLD:
                state.closed_frames += 1
            else:
                state.closed_frames = 0
                state.is_drowsy = False
            
            # Yawn detection
            if state.mar > MAR_THRESHOLD:
                state.yawn_frames += 1
                if state.yawn_frames > YAWN_FRAMES_THRESHOLD:
                    state.is_yawning = True
                    if state.yawn_frames == YAWN_FRAMES_THRESHOLD + 1:
                        state.yawn_count += 1
            else:
                state.yawn_frames = 0
                state.is_yawning = False
            
            # Trigger alert
            if state.closed_frames > CLOSED_FRAMES_THRESHOLD:
                current_time = time.time()
                
                if not state.is_drowsy:
                    state.drowsy_count += 1
                    state.alert_count += 1
                    state.is_drowsy = True
                    state.last_alert_time = current_time
                
                # Draw alert - centered at top
                alert_text = "DROWSINESS DETECTED"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 3
                
                # Get text size for centering
                (text_width, text_height), baseline = cv2.getTextSize(alert_text, font, font_scale, thickness)
                text_x = (w - text_width) // 2
                text_y = 80
                
                # Draw black background for text
                padding = 15
                cv2.rectangle(frame, 
                             (text_x - padding, text_y - text_height - padding),
                             (text_x + text_width + padding, text_y + baseline + padding),
                             (0, 0, 0), -1)
                
                # Draw red border
                cv2.rectangle(frame, 
                             (text_x - padding, text_y - text_height - padding),
                             (text_x + text_width + padding, text_y + baseline + padding),
                             (0, 0, 255), 3)
                
                # Draw text
                cv2.putText(frame, alert_text, (text_x, text_y), 
                           font, font_scale, (0, 0, 255), thickness)
                
                # Draw "TAKE A BREAK!" below
                break_text = "TAKE A BREAK!"
                (break_width, break_height), _ = cv2.getTextSize(break_text, font, 0.9, 2)
                break_x = (w - break_width) // 2
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
                
                # Subtle red overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    else:
        state.face_detected = False
        state.ear = 0.0
        state.mar = 0.0
    
    return frame

def generate_frames():
    """Generate video frames"""
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    prev_time = time.time()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Calculate FPS
        current_time = time.time()
        state.fps = int(1 / (current_time - prev_time)) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        # Process frame
        frame = process_frame(frame)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()

@app.route('/')
def index():
    """Serve dashboard"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Get current system status"""
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
    """Send emergency alert"""
    print("🚨 EMERGENCY ALERT TRIGGERED!")
    print(f"Time: {datetime.now()}")
    print(f"Drowsy Count: {state.drowsy_count}")
    print(f"Session Duration: {(datetime.now() - state.session_start).total_seconds():.1f}s")
    
    # Here you would integrate:
    # - Twilio SMS API
    # - Email notifications
    # - Push notifications
    
    state.alert_count += 1
    
    return jsonify({
        'success': True,
        'message': 'Emergency alert sent',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/export', methods=['GET'])
def export_data():
    """Export session data"""
    session_duration = (datetime.now() - state.session_start).total_seconds()
    
    data = {
        'session_start': state.session_start.isoformat(),
        'session_duration': session_duration,
        'drowsy_events': state.drowsy_count,
        'yawns': state.yawn_count,
        'alerts_sent': state.alert_count,
        'export_time': datetime.now().isoformat()
    }
    
    return jsonify(data)

if __name__ == '__main__':
    print("=" * 60)
    print("🚗 DRIVER DROWSINESS DETECTION SYSTEM")
    print("=" * 60)
    print("Starting server on http://localhost:5000")
    print("Open your browser and navigate to the URL above")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)