from src.train import train_model
from src.predict import run_webcam_detection

if __name__ == "__main__":
    choice = input("Enter 'train' to train model or 'detect' to run fire detection: ").lower()

    if choice == "train":
        train_model()
    elif choice == "detect":
        run_webcam_detection()
    else:
        print("Invalid input. Please enter 'train' or 'detect'.")
