import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
import os
import time

# -------------------------------
# Load model and labels
# -------------------------------
print("🔄 Loading model...")
model = load_model("emotion_detection_model.h5")
emotion_labels = ["angry", "disgust", "fear", "happy"]  # Update based on your dataset
print("✅ Model loaded successfully!\n")


# -------------------------------
# Predict emotion from image
# -------------------------------
def predict_emotion_from_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Could not read image.")
        return

    img_resized = cv2.resize(img, (48, 48))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=(0, -1))

    prediction = model.predict(img_input, verbose=0)
    label = emotion_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    print(f"🧠 Predicted Emotion: {label} ({confidence:.2f}%)")

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(img_color, f"{label} ({confidence:.1f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("🖼️ Image Emotion Detection", img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------------------
# Automatically capture once from webcam
# -------------------------------
def auto_capture_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("🎥 Webcam started! Capturing automatically in 3 seconds...")
    time.sleep(3)

    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to capture image.")
        cap.release()
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        print("⚠️ No face detected. Try again.")
    else:
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))

        prediction = model.predict(roi, verbose=0)
        label = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        print(f"🧠 Predicted Emotion: {label} ({confidence:.2f}%)")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({confidence:.1f}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("📸 Emotion Captured", frame)
        cv2.waitKey(2000)

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Webcam closed. Goodbye!")


# -------------------------------
# Browse image and predict
# -------------------------------
def browse_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    root.destroy()

    if file_path:
        predict_emotion_from_image(file_path)
    else:
        print("⚠️ No image selected.")


# -------------------------------
# Main menu
# -------------------------------
if __name__ == "__main__":
    print("\nChoose an option:")
    print("1️⃣  Automatically Capture Emotion via Webcam")
    print("2️⃣  Detect Emotion from an Uploaded Image\n")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        auto_capture_from_webcam()
    elif choice == "2":
        browse_image()
    else:
        print("❌ Invalid choice. Please enter 1 or 2.")
