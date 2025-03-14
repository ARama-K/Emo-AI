import cv2
import numpy as np
import tensorflow as tf
import time

# Load the trained emotion detection model
model = tf.keras.models.load_model('emotion_detection_model.keras')

# Load OpenCV's pre-trained face det
# ector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels    
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

start_time = time.time()
duration = 140  # Test duration in seconds

while time.time() - start_time < duration:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.reshape(1, 48, 48, 1) / 255.0  # Preprocess for the model

        # Predict the emotion
        emotion_probs = model.predict(roi_gray)
        predicted_emotion = np.argmax(emotion_probs)
        emotion_text = emotion_labels[predicted_emotion]

        # Draw face rectangle and emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow("Video Capture Test", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Test completed successfully.")
cap.release()
cv2.destroyAllWindows()