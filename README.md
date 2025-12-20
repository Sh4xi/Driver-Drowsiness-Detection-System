# 🚗 Driver Drowsiness Detection System

A real-time drowsiness detection system using computer vision and machine learning to prevent accidents caused by driver fatigue. The system monitors eye closure (EAR - Eye Aspect Ratio) and yawning (MAR - Mouth Aspect Ratio) to detect drowsiness and alert drivers immediately.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-3.0.0-green.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.8.1-red.svg)
![MediaPipe](https://img.shields.io/badge/mediapipe-0.10.9-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

- **Real-time Face Detection**: Uses MediaPipe Face Mesh for accurate facial landmark detection
- **Drowsiness Detection**: Monitors Eye Aspect Ratio (EAR) to detect eye closure
- **Yawn Detection**: Tracks Mouth Aspect Ratio (MAR) to identify yawning
- **Visual & Audio Alerts**: Immediate on-screen alerts with audio warnings
- **Session Statistics**: Tracks drowsy events, yawns, and session duration
- **GPS Integration**: Browser-based location tracking
- **Emergency Alert System**: One-click emergency notification
- **Data Export**: Export session data as JSON for analysis
- **Responsive UI**: Modern, dark-themed dashboard with real-time metrics

## 🎯 How It Works

### Eye Aspect Ratio (EAR)
The system calculates the ratio between eye height and width. When eyes close, EAR drops below threshold, triggering drowsiness detection.

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

### Mouth Aspect Ratio (MAR)
Monitors mouth opening to detect yawning, an early sign of fatigue.

```
MAR = (||p2 - p8|| + ||p3 - p7|| + ||p4 - p6||) / (3 * ||p1 - p5||)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam
- Modern web browser (Chrome, Firefox, Edge)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sh4xi/drowsiness-detection-system.git
   cd drowsiness-detection-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   cd backend
   python main.py
   ```

5. **Open your browser**
   ```
   http://localhost:5000
   ```

## 📁 Project Structure

```
drowsiness-detection-system/
├── backend/
│   └── main.py              # Main Flask application
├── frontend/
│   └── index.html           # Web dashboard
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── LICENSE                 # MIT License
└── .gitignore             # Git ignore rules
```

## ⚙️ Configuration

Adjust detection sensitivity by modifying thresholds in `backend/main.py`:

```python
# Detection thresholds (line 56-59)
EAR_THRESHOLD = 0.23           # Lower = more sensitive to eye closure
MAR_THRESHOLD = 1.20           # Higher = less sensitive to mouth opening
CLOSED_FRAMES_THRESHOLD = 25   # Frames before drowsiness alert
YAWN_FRAMES_THRESHOLD = 20     # Frames before yawn detection
```

### Calibration Mode

Enable calibration to find optimal thresholds for your setup:

```python
CALIBRATION_MODE = True  # Line 63
```

Run the system and check terminal output for real-time EAR/MAR values.

## 📊 System Requirements

### Minimum
- Processor: Intel i3 or equivalent
- RAM: 4GB
- Webcam: 720p @ 15 FPS
- OS: Windows 10, macOS 10.14+, Ubuntu 18.04+

### Recommended
- Processor: Intel i5 or equivalent
- RAM: 8GB
- Webcam: 1080p @ 30 FPS
- Good lighting conditions

## 🎨 Dashboard Features

- **Live Video Feed**: Real-time camera stream with face detection overlay
- **Metrics Display**: EAR and MAR values with visual indicators
- **System Info**: Confidence, brightness, FPS, face detection status
- **GPS Location**: Real-time location tracking
- **Session Statistics**: Drowsy events, yawns, alerts, session time
- **Activity Log**: Chronological event logging
- **Emergency Button**: Quick access to emergency contacts

## 🔧 Troubleshooting

### Camera not detected
- Check camera permissions in system settings
- Ensure no other application is using the camera
- Try different USB port or restart computer

### Low FPS
- Reduce frame resolution in code (line 107)
- Close other resource-intensive applications
- Ensure good lighting (reduces processing time)

### False drowsiness alerts
- Increase `EAR_THRESHOLD` (try 0.25, 0.27)
- Increase `CLOSED_FRAMES_THRESHOLD` (try 30, 35)
- Improve lighting conditions
- Enable calibration mode to find your baseline

### Not detecting drowsiness
- Decrease `EAR_THRESHOLD` (try 0.21, 0.19)
- Decrease `CLOSED_FRAMES_THRESHOLD` (try 20, 15)
- Ensure face is properly centered and visible

## 📈 Performance

- **Detection Latency**: < 100ms
- **Alert Response Time**: ~0.8 seconds (configurable)
- **Average FPS**: 25-30 FPS on recommended hardware
- **CPU Usage**: 15-30% on modern processors

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) - Face mesh detection
- [OpenCV](https://opencv.org/) - Computer vision processing
- [Flask](https://flask.palletsprojects.com/) - Web framework
- EAR algorithm based on: Soukupová and Čech (2016)
- Developed with assistance from AI pair programming tools

## 📧 Contact

Ignacio Tabug III - [@2ez4thirdy](https://www.facebook.com/2ez4thirdy/) - ignaciotabug36@gmail.com

Project Link: [https://github.com/Sh4xi/Driver-Drowsiness-Detection-System](https://github.com/Sh4xi/Driver-Drowsiness-Detection-System)

## 🌟 Star History

If you find this project helpful, please consider giving it a star ⭐

---

**⚠️ Disclaimer**: This system is intended as an aid and should not be solely relied upon for driver safety. Always drive responsibly and take breaks when feeling tired.
