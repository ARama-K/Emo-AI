import google.generativeai as genai

# Configure Gemini API key (replace with your actual API key)
genai.configure(api_key="AIzaSyCDH6ao_WM8oVtakf4IVIx6iOPrnVVBXqM")

def get_gemini_response(user_message, emotion):
    """Send user message + detected emotion context to Gemini."""
    prompt = f"User's detected emotion: {emotion}\nUser: {user_message}\nAssistant:"

    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)

    return response.text