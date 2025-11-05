import os
from typing import Optional

def create_groq_chat(model_name: Optional[str] = None, temperature: float = 0.2, **kwargs):
    """Create a ChatGroq instance using an environment override or the provided model_name.

    Returns the ChatGroq instance, or None if initialization fails. Caller should handle None.
    """
    try:
        from langchain_groq import ChatGroq
    except Exception as e:
        print(f"Groq library not installed or unavailable: {e}")
        return None

    model = model_name or os.getenv("GROQ_MODEL") or os.getenv("GROQ_DEFAULT_MODEL")
    if not model:
        # Keep existing default if no override provided; caller may pass a default model_name
        model = None

    # Allow api key override if present
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY_JUDGE")

    try:
        if model:
            return ChatGroq(temperature=temperature, model_name=model, api_key=api_key, **kwargs)
        else:
            return ChatGroq(temperature=temperature, api_key=api_key, **kwargs)
    except Exception as e:
        # Provide actionable message about model deprecation or bad configuration
        print(f"Failed to initialize Groq Chat model (model={model}): {e}")
        print("Set the environment variable GROQ_MODEL to a supported model name or update your code to use a supported model.")
        return None
