# 🔥 Fire Detection System using AI

A real-time fire detection system using deep learning and OpenCV. The system is capable of classifying images as "fire" or "no fire" and can perform real-time fire detection through a webcam feed.

---

## 🚀 Features

- 🧠 Train a CNN model to classify fire vs. no-fire images
- 🖼️ Predict fire from uploaded images
- 🎥 Real-time fire detection using webcam (OpenCV)
- 🔔 Alarm sound when fire is detected (optional)

---

## 📁 Project Structure

fire-detection-ai/
├── dataset/ # Local training data (excluded from repo)
│ ├── fire/
│ └── no_fire/
├── src/
│ ├── train.py # Script to train the CNN model
│ ├── predict.py # Script to predict fire from image
│ ├── realtime_detect.py # Script for webcam-based real-time detection
│ └── utils.py # Helper functions (if any)
├── alarm.mp3 # Alarm sound (played when fire is detected)
├── requirements.txt # List of required Python packages
├── .gitignore # Ignoring large dataset & unnecessary files
└── README.md


---

## 🧪 Setup & Usage

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

### 2️⃣ Train the model
```bash
python src/train.py
Ensure your dataset is in dataset/fire/ and dataset/no_fire/.

### 3️⃣ Predict fire in an image
```bash
python src/predict.py path/to/image.jpg





