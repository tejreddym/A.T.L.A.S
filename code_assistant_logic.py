# code_assistant_logic.py

import os
import asyncio
import gradio as gr
from typing import Optional, Tuple

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

# --- Load API Key ---
from dotenv import load_dotenv
load_dotenv() # Ensure .env is loaded here too

# --- LLM Setup for Code Assistant ---
try:
    # Use a powerful Groq LLM for code generation/assistance
    # Llama 3 70B is generally good for coding tasks
    CODE_ASSISTANT_LLM_KEY = os.getenv("GROQ_API_KEY_JUDGE") # Using a powerful key
    if not CODE_ASSISTANT_LLM_KEY:
        CODE_ASSISTANT_LLM_KEY = os.getenv("GROQ_API_KEY") # Fallback
    if not CODE_ASSISTANT_LLM_KEY:
        raise ValueError("GROQ_API_KEY or GROQ_API_KEY_JUDGE not found in .env for code assistant.")
    code_assistant_llm = ChatGroq(temperature=0.1, model_name="llama3-70b-8192", api_key=CODE_ASSISTANT_LLM_KEY)
except ValueError as e:
    print(f"Error loading Groq API key for code assistant LLM: {e}")
    code_assistant_llm = None


# --- Core Code Assistant Logic ---
async def generate_code_response(user_prompt: str, programming_language: str) -> Tuple[str, str]:
    """
    Generates code, explanations, or debugging help based on the user's prompt and language.
    """
    gr.Info(f"AI Code Assistant is thinking in {programming_language}...")
    response_content = ""
    status_message = "Generating code response."

    if code_assistant_llm is None:
        status_message = "Error: Code Assistant LLM not initialized (API key missing or invalid)."
        gr.Error(status_message)
        return "", status_message
    
    if not user_prompt:
        status_message = "Please provide a prompt for the code assistant."
        gr.Warning(status_message)
        return "", status_message

    system_message_content = (
        f"You are an expert AI programming assistant specializing in {programming_language}. "
        "Your task is to respond to user queries related to code. This includes generating code, "
        "explaining code, debugging code, or providing best practices. "
        "Always provide complete and correct code snippets where applicable, enclosed in Markdown code blocks. "
        "For explanations or debugging, be concise and clear. "
        "If generating code, include necessary imports and comments. "
        "Do not include conversational filler like 'Here is your code:' or 'I can help with that.' "
        "Just provide the direct response."
    )

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message_content),
        HumanMessage(content=f"User's request in {programming_language}:\n{user_prompt}")
    ])

    code_chain = prompt_template | code_assistant_llm | StrOutputParser()

    try:
        response_content = await code_chain.ainvoke({"user_prompt": user_prompt, "programming_language": programming_language})
        status_message = "Code response generated successfully!"
        gr.Info(status_message)
    except Exception as e:
        status_message = f"Error generating code response: {e}"
        gr.Error(status_message)
        print(f"Code Assistant Error: {e}")
        response_content = f"Failed to generate a code response due to an internal error: {e}"

    return response_content, status_message

# --- Example Usage for direct testing (Optional) ---
if __name__ == "__main__":
    async def run_test_code_assistant():
        print("\n--- Running Code Assistant Test (Python - Sum of two numbers) ---")
        code, status = await generate_code_response(
            user_prompt="Write a Python function to add two numbers and return their sum.",
            programming_language="Python"
        )
        print(f"Status: {status}")
        print(f"Generated Code:\n{code}")
        print("-" * 30)

        print("\n--- Running Code Assistant Test (JavaScript - Explain promise) ---")
        code, status = await generate_code_response(
            user_prompt="Explain what a JavaScript Promise is and provide a simple example.",
            programming_language="JavaScript"
        )
        print(f"Status: {status}")
        print(f"Generated Code:\n{code}")
        print("-" * 30)

    asyncio.run(run_test_code_assistant())