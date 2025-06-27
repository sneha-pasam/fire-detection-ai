import cv2
import numpy as np

def load_and_preprocess_image(path, target_size=(64, 64)):
    img = cv2.imread(path)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img
