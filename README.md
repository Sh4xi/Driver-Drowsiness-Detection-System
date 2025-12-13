# 🚗 Driver Drowsiness Detection System

An intelligent real-time driver monitoring system that detects drowsiness and fatigue using computer vision and machine learning to prevent road accidents.

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Overview

This project implements a comprehensive driver drowsiness detection system using MediaPipe's Face Mesh for real-time facial landmark detection. The system monitors eye movements, yawning patterns, and provides instant alerts to prevent fatigue-related accidents.

## ✨ Features

- **Real-time Face Detection** - Uses MediaPipe Face Mesh with 468 facial landmarks
- **Eye Aspect Ratio (EAR)** - Monitors eye closure patterns to detect drowsiness
- **Mouth Aspect Ratio (MAR)** - Detects yawning as an indicator of fatigue
- **Live Video Feed** - Real-time video processing with OpenCV
- **Professional Dashboard** - Modern web-based interface with Flask backend
- **Visual & Audio Alerts** - Immediate warnings when drowsiness is detected
- **Session Statistics** - Tracks drowsy events, yawns, and session duration
- **GPS Integration** - Real-time location tracking (browser-based geolocation)
- **Data Export** - Export session data in JSON format
- **Emergency Alert System** - Quick alert button for critical situations

## 🛠️ Tech Stack

**Backend:**
- Python 3.12
- OpenCV - Video processing
- MediaPipe - Face mesh detection
- Flask - Web server
- NumPy - Numerical computations

**Frontend:**
- HTML5/CSS3/JavaScript
- Real-time API integration
- Responsive design
- Modern UI/UX

## 📊 Detection Methodology

### Eye Aspect Ratio (EAR)
```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```
- Threshold: < 0.25 (indicates closed eyes)
- Alert trigger: 15+ consecutive frames (~0.5 seconds)

### Mouth Aspect Ratio (MAR)
```
MAR = (||p2 - p8|| + ||p3 - p7|| + ||p4 - p6||) / (3 * ||p1 - p5||)
```
- Threshold: > 0.6 (indicates yawning)
- Detection: 20+ consecutive frames

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Webcam
- Modern web browser (Chrome, Firefox, Edge)

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/Sh4xi/drowsiness-detection-system.git
cd drowsiness-detection-system
```

2. **Create virtual environment (optional but recommended)**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
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

## 📦 Dependencies
```txt
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.23.0
flask>=3.0.0
flask-cors>=4.0.0
```

## 🎮 Usage

1. **Start the system** - Run `python main.py` from the backend folder
2. **Allow camera access** - Grant permission when prompted by browser
3. **Monitor the dashboard** - View real-time metrics and statistics
4. **Receive alerts** - System will alert when drowsiness is detected
5. **Export data** - Download session reports for analysis

## 🔧 Configuration

Adjust detection sensitivity in `backend/main.py`:
```python
EAR_THRESHOLD = 0.25          # Eye closure threshold
MAR_THRESHOLD = 0.6           # Yawn detection threshold
CLOSED_FRAMES_THRESHOLD = 15  # Frames before alert
YAWN_FRAMES_THRESHOLD = 20    # Frames for yawn detection
```


## 🌟 Future Enhancements

- [ ] SMS/Email alerts via Twilio integration
- [ ] Database storage (PostgreSQL/MongoDB)
- [ ] Head pose estimation for distraction detection
- [ ] Night vision support with IR camera
- [ ] Mobile app integration
- [ ] Cloud deployment (AWS/Azure)
- [ ] Multi-language support
- [ ] Driver behavior analytics dashboard

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Ignacio Tabug**
- GitHub: [@Sh4xi](https://github.com/Sh4xi)
- LinkedIn: [thirdtabug](https://linkedin.com/in/thirdtabug)
- Email: ignaciotabug36@gmail.com

## 🙏 Acknowledgments

- MediaPipe team for the Face Mesh model
- OpenCV community for computer vision tools
- Flask framework developers

## 📧 Contact

For questions or suggestions, please open an issue or contact me directly.

---

⭐ If you found this project helpful, please give it a star!

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/drowsiness-detection-system?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/drowsiness-detection-system?style=social)
```

**Note:** Replace `yourusername` with your actual GitHub username!

---

## 📄 **2. LICENSE Content**

**Create file:** `LICENSE` (no extension) in root folder

**Paste this:**
```
MIT License

Copyright (c) 2024 Ignacio Tabug

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## ✅ **Summary - Create These 2 Files:**

### **File 1: README.md**
- Location: Root folder
- Content: Full markdown documentation (copy from above)

### **File 2: LICENSE**
- Location: Root folder
- Content: MIT License text (copy from above)

---

## 🎯 **After Creating Both Files:**

Your structure will be:
```
drowsiness-detection-system/
├── .venv/
├── backend/
│   └── main.py
├── frontend/
│   └── index.html
├── .gitignore          ✅ DONE
├── README.md           ✅ CREATE THIS NOW
├── LICENSE             ✅ CREATE THIS NOW
└── requirements.txt    ✅ DONE