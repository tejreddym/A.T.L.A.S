# In vision_logic.py

import gradio as gr
import os
import google.generativeai as genai
from PIL import Image

# --- Configure the Google AI API Key ---
try:
    # Get the API key from the environment variables
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Google AI API key configured successfully.")
except Exception as e:
    print(f"Error configuring Google AI API key: {e}")
    # Handle the case where the key is not found or invalid
    GOOGLE_API_KEY = None

# --- Main function to analyze an image ---
async def analyze_image(image_to_analyze: Image.Image, prompt_text: str):
    """
    Analyzes an image with a text prompt using the Gemini Pro Vision model.

    Args:
        image_to_analyze: A PIL Image object of the image to analyze.
        prompt_text: The user's question about the image.

    Returns:
        A string containing the model's response.
    """
    if not GOOGLE_API_KEY:
        return "Error: Google AI API key is not configured. Please add it to your .env file."
    
    if image_to_analyze is None:
        return "Error: Please upload an image first."
        
    if not prompt_text:
        return "Error: Please enter a question or prompt about the image."

    try:
        gr.Info("A.T.L.A.S. is looking at the image...")
        
        # Initialize the specific model we want to use
        vision_model = genai.GenerativeModel('gemini-pro-vision')
        
        # The API expects a list containing the prompt and the image
        # We use the async version of the call to work well with our app
        response = await vision_model.generate_content_async([prompt_text, image_to_analyze])
        
        gr.Info("Analysis complete.")
        # Return the text part of the model's response
        return response.text

    except Exception as e:
        error_message = f"An error occurred while analyzing the image: {e}"
        print(error_message)
        gr.Error(error_message)
        return error_message