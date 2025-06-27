import cv2
import tensorflow as tf
import numpy as np
from playsound import playsound
import threading
import os

# Only play alarm once per fire event
alarm_playing = False

def play_alarm():
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True
        playsound("alarm.mp3")  # make sure alarm.mp3 is in the same directory or give full path
        alarm_playing = False

def run_webcam_detection():
    global alarm_playing

    model = tf.keras.models.load_model('models/fire_model.h5')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (128, 128))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 128, 128, 3))

        prediction = model.predict(reshaped)[0][0]
        label = "Fire" if prediction > 0.5 else "No Fire"
        color = (0, 0, 255) if label == "Fire" else (0, 255, 0)

        # 🔔 Trigger alarm in a new thread
        if label == "Fire" and not alarm_playing:
            threading.Thread(target=play_alarm, daemon=True).start()

        cv2.putText(frame, f"{label} ({prediction:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Fire Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
