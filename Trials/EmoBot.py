import threading
import tkinter as tk
from tkinter import scrolledtext
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import requests

# Load your trained model
model = load_model("emotion_detection_model.keras")
emotion_label = "Neutral"  # Default emotion (global variable)

# ========================== EMOTION DETECTION ==========================

def emotion_detection():
    global emotion_label
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (48, 48)).reshape(1, 48, 48, 1) / 255.0

        prediction = model.predict(face)
        emotion_label = get_emotion_label(np.argmax(prediction))

        cv2.putText(frame, emotion_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_emotion_label(index):
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return emotions[index]

# ========================== LLM INTEGRATION ==========================

def get_emotion_aware_response(user_message, emotion):
    system_prompt = f"""
    You are a friendly, empathetic chatbot. The user is currently experiencing the emotion: {emotion}.
    Respond to the user message in a way that acknowledges their current emotion.
    Be supportive if they are sad or angry, cheerful if they are happy, and calming if they are fearful.
    """

    payload = {
        "model": "mistral",  # Replace with "llama2" if using LLaMA 2
        "prompt": f"System: {system_prompt}\nUser: {user_message}\nAssistant:",
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "I'm not sure how to respond to that.")
        else:
            return "Sorry, I'm having trouble thinking right now."
    except requests.RequestException:
        return "I can't connect to my brain (LLM server). Please check if Ollama is running."

# ========================== CHATBOT GUI ==========================

class ChatBotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot")
        self.root.geometry("400x500")

        self.chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', height=20, width=50)
        self.chat_area.pack(pady=10)

        self.entry = tk.Entry(root, width=50)
        self.entry.pack(pady=5)

        self.send_button = tk.Button(root, text="Send", command=self.send_message)
        self.send_button.pack()

        self.update_emotion_label()

    def send_message(self):
        user_message = self.entry.get()
        self.display_message(f"You: {user_message}")
        bot_response = self.get_bot_response(user_message)
        self.display_message(f"Bot: {bot_response}")
        self.entry.delete(0, tk.END)

    def display_message(self, message):
        self.chat_area.configure(state='normal')
        self.chat_area.insert(tk.END, message + "\n")
        self.chat_area.configure(state='disabled')
        self.chat_area.yview(tk.END)

    def get_bot_response(self, user_message):
        global emotion_label
        return get_emotion_aware_response(user_message, emotion_label)

    def update_emotion_label(self):
        self.root.title(f"Chatbot - Current Emotion: {emotion_label}")
        self.root.after(1000, self.update_emotion_label)  # Update every second

# ========================== START GUI + EMOTION DETECTION ==========================

if __name__ == "__main__":
    threading.Thread(target=emotion_detection, daemon=True).start()

    root = tk.Tk()
    app = ChatBotApp(root)
    root.mainloop()
