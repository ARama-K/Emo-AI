import openai

openai.api_key = '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'  # Replace with your actual API key

conversation = [
    {"role": "system", "content": "You are a friendly chatbot that adjusts its tone based on user emotion."}
]

def get_chatbot_response(user_message, emotion):
    global conversation

    # Add the user message + emotion to the conversation
    conversation.append({"role": "user", "content": f"My emotion is {emotion}. {user_message}"})

    # Call GPT-like API
    response = openai.ChatCompletion.create(
        model="gpt-4",  # or gpt-3.5-turbo if you want cheaper
        messages=conversation
    )

    chatbot_message = response['choices'][0]['message']['content']

    # Store chatbot response in conversation history
    conversation.append({"role": "assistant", "content": chatbot_message})

    return chatbot_message
