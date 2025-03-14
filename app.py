from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import google.generativeai as genai
import time
import threading

app = Flask(__name__)

# ========== Load Emotion Detection Model ==========
model = load_model("emotion_detection_model.keras")

# ========== Global Variables ==========
current_emotion = "Neutral"
latest_frame = None  # Store latest frame globally for processing

# ========== Gemini API Setup ==========
genai.configure(api_key="AIzaSyCDH6ao_WM8oVtakf4IVIx6iOPrnVVBXqM")  # Replace with your actual key
gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')

# ========== Emotion Mapping ==========
def get_emotion_label(index):
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return emotions[index]

# ========== Real-time Emotion Detection ==========
def emotion_detection_loop():
    global current_emotion, latest_frame

    while True:
        if latest_frame is not None:
            gray = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(gray, (48, 48)).reshape(1, 48, 48, 1) / 255.0
            prediction = model.predict(face)
            current_emotion = get_emotion_label(np.argmax(prediction))
        time.sleep(1)

# Start the emotion detection in a background thread
threading.Thread(target=emotion_detection_loop, daemon=True).start()

# ========== Video Streaming ==========
def generate_frames():
    global latest_frame
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        latest_frame = frame.copy()

        cv2.putText(frame, f"Emotion: {current_emotion}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# ========== Chatbot Response ==========
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")

    system_prompt = f"""
    You are a friendly, empathetic chatbot. The user is currently experiencing the emotion: {current_emotion}.
    Respond to the user message in a way that acknowledges their current emotion.
    Be supportive if they are sad or angry, cheerful if they are happy, and calming if they are fearful.
    """

    try:
        response = gemini_model.generate_content(system_prompt + "\nUser: " + user_message + "\nAssistant:")
        bot_response = response.text.strip()
    except Exception as e:
        bot_response = "Sorry, I'm having trouble thinking right now."

    return jsonify({"response": bot_response})

# ========== Routes ==========
@app.route('/')
def index():
    return render_template('index.html', emotion=current_emotion)

@app.route('/current_emotion')
def current_emotion_endpoint():
    return jsonify({"emotion": current_emotion})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)