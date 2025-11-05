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
    GOOGLE_API_KEY = None

# --- Main function to analyze an image ---
async def analyze_image(image_to_analyze: Image.Image, prompt_text: str):
    """
    Analyzes an image with a text prompt using the Gemma 3 Vision model.
    """
    if not GOOGLE_API_KEY:
        return "Error: Google AI API key is not configured. Please add it to your .env file."
    
    if image_to_analyze is None:
        return "Error: Please upload an image first."
        
    if not prompt_text:
        return "Error: Please enter a question or prompt about the image."

    try:
        gr.Info("A.T.L.A.S. is looking at the image with its Gemma 3 eyes...")
        
        # --- THIS IS THE CHANGE ---
        # Initialize the Gemma 3 model you selected.
        vision_model = genai.GenerativeModel('gemma-3-27b-it')
        
        # The API expects a list containing the prompt and the image
        response = await vision_model.generate_content_async([prompt_text, image_to_analyze])
        
        gr.Info("Analysis complete.")
        return response.text

    except Exception as e:
        error_message = f"An error occurred while analyzing the image: {e}"
        print(error_message)
        gr.Error(error_message)
        return error_message
