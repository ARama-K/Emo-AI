import google.generativeai as genai

genai.configure(api_key="AIzaSyCDH6ao_WM8oVtakf4IVIx6iOPrnVVBXqM")

models = genai.list_models()

for model in models:
    print(model.name)
