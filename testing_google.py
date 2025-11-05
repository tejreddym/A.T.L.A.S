import google.generativeai as genai

genai.configure(api_key="my_api")  # Replace with your actual Gemini API key

model = genai.GenerativeModel("gemma-3-12b-it")

response = model.generate_content("Whon is the cm of AP")

print(response.text)
