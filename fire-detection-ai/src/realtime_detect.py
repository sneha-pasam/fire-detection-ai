import cv2
import numpy as np
from keras.models import load_model
from playsound import playsound
import threading
import time

# Load model
model = load_model('models/fire_model.h5')

# Alarm control
alarm_playing = False
last_alarm_time = 0
cooldown = 5  # seconds

def play_alarm():
    global alarm_playing
    playsound('alarm.mp3')
    alarm_playing = False

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    resized = cv2.resize(frame, (128, 128))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 128, 128, 3))

    # Predict
    prediction = model(reshaped, training=False).numpy()[0][0]
    label = "Fire" if prediction > 0.5 else "No Fire"
    color = (0, 0, 255) if label == "Fire" else (0, 255, 0)

    # Display prediction
    cv2.putText(frame, f"{label} ({prediction:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Fire Detection", frame)

    # Play alarm with cooldown
    if label == "Fire" and not alarm_playing:
        current_time = time.time()
        if current_time - last_alarm_time > cooldown:
            alarm_playing = True
            last_alarm_time = current_time
            threading.Thread(target=play_alarm, daemon=True).start()

    # Optional: Debug print shapes with 'd'
    if cv2.waitKey(1) & 0xFF == ord('d'):
        print("Model input shape:", model.input_shape)
        print("Input given:", reshaped.shape)
    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
