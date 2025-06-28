# 🔥 Fire Detection System using AI

This project is an AI-based fire detection system that uses deep learning and computer vision to identify fire in real-time through video input. It triggers an alarm when fire is detected.

## 🚀 Features

- Real-time fire detection using OpenCV
- Custom-trained CNN model for fire classification
- Alarm system triggered on fire detection
- Video input via webcam or video files
- Easy to use and extend

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Virtual Environment for dependency management

## 📁 Project Structure

fire-detection-ai/
│
├── src/ # Source code
│ ├── train.py # Model training script
│ ├── predict.py # Prediction on static images
│ └── realtime_detect.py # Real-time fire detection via webcam/video
│
├── dataset/ # Image dataset (ignored in Git)
├── venv/ # Python virtual environment (ignored in Git)
├── alarm.mp3 # Alarm sound file
├── test_images/ # Test images folder (optional)
├── output.avi # Output video file (ignored in Git)
├── .gitignore # Files/folders to exclude from Git
└── README.md # Project documentation

bash
Copy code

## 🧪 How to Use

### 1. 🔧 Setup Environment

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # On Windows
# or
source venv/bin/activate     # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
🧠 Train the Model
```bash
python src/train.py
🔍 Run Detection on an Image
```bash
python src/predict.py --image path/to/image.jpg
📹 Real-Time Detection via Webcam
```bash
python src/realtime_detect.py
📌 To-Do
 Improve model accuracy

 Add email/SMS alert system

 Deploy using Flask + Docker

 Build a web dashboard for alerts

🤝 Contributor
Sneha Pasam
