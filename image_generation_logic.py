# In image_generation_logic.py

import gradio as gr
import os
import httpx
from PIL import Image
import io
from datetime import datetime
from gradio_client import Client # <-- New Import
from dotenv import load_dotenv

# Ensure environment variables from .env are loaded when this module is imported.
load_dotenv()

# --- Load API Token and Define API URLs ---
GENERAL_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"


def _get_hf_token() -> tuple:
    """Return a tuple (env_var_name, token) for Hugging Face token from common env var names.
    Returns (None, '') if not found.
    """
    candidates = [
        "HUGGINGFACEHUB_API_TOKEN",
        "HUGGING_FACE_API_TOKEN",
        "HUGGINGFACE_API_TOKEN",
        "HUGGING_FACE_TOKEN",
        "HF_API_TOKEN",
    ]
    for name in candidates:
        val = os.getenv(name)
        if val:
            # Detect placeholder tokens like '*****' which are not real credentials
            v = val.strip().strip('"').strip("'")
            if v and set(v) == {'*'}:
                print(f"Found placeholder Hugging Face token in env var {name}. Please replace with a valid token.")
                return None, ""
            return name, v
    return None, ""


def _build_hf_headers() -> dict:
    name, token = _get_hf_token()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


# --- Function for the GENERAL "Image Generation" tab (Unchanged) ---
async def generate_image(prompt: str):
    gr.Info("Sending prompt to Stable Diffusion XL...")
    image, status = await _query_huggingface_api(prompt, GENERAL_API_URL)
    return image, status


# --- NEW Function for the ANIME "Image Generation" tab ---
def generate_anime_image(prompt: str):
    """
    Connects to a public Gradio Space to generate an anime image.
    """
    gr.Info("Connecting to Heartsync/Anime Gradio Space...")
    try:
        client = Client("Heartsync/NSFW-Uncensored")
        # We use the parameters you provided, inserting the user's prompt
        result = client.predict(
                prompt=prompt,
                negative_prompt="text, talk bubble, low quality, watermark, signature",
                seed=0,
                randomize_seed=True,
                width=1024,
                height=1024,
                guidance_scale=7,
                num_inference_steps=28,
                api_name="/infer"
        )
        
        # The result from gradio_client is a filepath to the downloaded image
        print(f"Gradio client returned filepath: {result}")
        gr.Info("Image received!")
        # The gr.Image component can directly display the image from this filepath
        return result, "Image generation complete."

    except Exception as e:
        error_message = f"An error occurred while using the Gradio Client: {e}"
        print(error_message)
        gr.Error(error_message)
        return None, error_message


# --- Helper function for the general image generator ---
async def _query_huggingface_api(prompt: str, api_url: str):
    name, token = _get_hf_token()
    if not token:
        msg = (
            "Hugging Face API token not found. Set HUGGINGFACEHUB_API_TOKEN or HUGGING_FACE_API_TOKEN in your .env."
        )
        gr.Warning(msg)
        return None, msg
    payload = {"inputs": prompt}
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(api_url, headers=_build_hf_headers(), json=payload)
        if response.status_code != 200:
            # If the API returns JSON with an error message, include that text
            try:
                body = response.json()
            except Exception:
                body = response.text
            # Include which env var supplied the token (masked) for diagnostics
            token_info = name if name else 'none'
            masked = None
            try:
                masked = f"{token_info} (len={len(token)})"
            except Exception:
                masked = token_info
            error_message = f"Error from API: {response.status_code} - {body} | token_from={masked}"
            print(error_message)
            gr.Error(error_message)
            return None, error_message
        # Sometimes the response may be JSON (e.g., an error) even with 200; handle gracefully
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body = response.json()
            except Exception:
                body = response.text
            token_info = name if name else 'none'
            masked = None
            try:
                masked = f"{token_info} (len={len(token)})"
            except Exception:
                masked = token_info
            error_message = f"Received JSON response from HF inference endpoint: {body} | token_from={masked}"
            print(error_message)
            gr.Error(error_message)
            return None, error_message

        image = Image.open(io.BytesIO(response.content))
        gr.Info("Image received!")
        return image, "Image generation complete."
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(error_message)
        gr.Error(error_message)
        return None, error_message


# --- Function for the AGENT TOOL (Unchanged) ---
def generate_image_tool_func(prompt: str) -> str:
    output_dir = "generated_images"
    os.makedirs(output_dir, exist_ok=True)
    name, token = _get_hf_token()
    if not token:
        return "Failed to generate image: Hugging Face API token not found. Set HUGGINGFACEHUB_API_TOKEN or HUGGING_FACE_API_TOKEN in your environment."
    with httpx.Client(timeout=120.0) as client:
        response = client.post(GENERAL_API_URL, headers=_build_hf_headers(), json={"inputs": prompt})
    if response.status_code == 200:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"atlas_image_{timestamp}.png")
        try:
            image = Image.open(io.BytesIO(response.content))
        except Exception as e:
            # If content is JSON, include it in the error
            try:
                body = response.json()
            except Exception:
                body = response.text
            token_info = name if name else 'none'
            masked = None
            try:
                masked = f"{token_info} (len={len(token)})"
            except Exception:
                masked = token_info
            return f"Failed to parse image from response: {e}. Response body: {body} | token_from={masked}"
        image.save(filepath)
        print(f"Image saved to {filepath}")
        return f"Successfully generated and saved image to the file '{filepath}'."
    else:
        # include json body if present
        try:
            body = response.json()
        except Exception:
            body = response.text
        token_info = name if name else 'none'
        masked = None
        try:
            masked = f"{token_info} (len={len(token)})"
        except Exception:
            masked = token_info
        return f"Failed to generate image. API responded with status {response.status_code}: {body} | token_from={masked}"