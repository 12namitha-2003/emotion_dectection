import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# -------------------------------
# Load model and define labels
# -------------------------------
print("🔄 Loading model...")
model = load_model("emotion_detection_model.h5")
emotion_labels = ["angry", "disgust", "fear", "happy"]  # Change if your model has more classes
print("✅ Model loaded successfully!\n")

# -------------------------------
# Function to predict emotion from an image
# -------------------------------
def predict_emotion_from_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Could not read image.")
        return

    # Preprocess
    img_resized = cv2.resize(img, (48, 48))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=(0, -1))

    # Predict
    prediction = model.predict(img_input, verbose=0)
    label = emotion_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    print(f"🧠 Predicted Emotion: {label} ({confidence:.2f}%)")

    # Show image with label
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(img_color, f"{label} ({confidence:.1f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("🖼️ Image Emotion Detection", img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------------------
# Function for live webcam prediction
# -------------------------------
def live_emotion_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("🎥 Webcam started! Press 'q' to quit.\n")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray / 255.0
            roi = np.expand_dims(roi, axis=(0, -1))

            prediction = model.predict(roi, verbose=0)
            label = emotion_labels[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("😊 Live Emotion Detection",
