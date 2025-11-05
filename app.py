import gradio as gr
import torch
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from groq_utils import create_groq_chat
from dotenv import load_dotenv
import os
import shutil
from langchain_community.vectorstores import FAISS
import asyncio
import pandas as pd
import json

from typing import List, Dict, Tuple, Optional, AsyncGenerator, Any

# --- Import our logic from the other files ---
from summarizer_logic import summarize_with_local_model, summarize_with_groq
from doc_qa_logic import create_knowledge_base, answer_conversational
from web_research_logic import run_web_researcher
from combat_logic import DebateAgent, run_debate, get_ai_judgement
from image_generation_logic import generate_image, generate_anime_image

# API Management imports
from cryptography.fernet import Fernet
from typing import Dict, Optional

# Imports for Chat With My Web
from web_kb_logic import create_web_knowledge_base, answer_web_kb_conversational
# Imports for Meeting Prep Assistant
from meeting_prep_logic import prepare_for_meeting
# Imports for Code Assistant
from code_assistant_logic import generate_code_response

# NEW IMPORT for Data Analysis
from data_analysis_logic import load_and_preview_data, analyze_data_with_llm # NEW IMPORT


# --- Load the API Key from .env file ---
load_dotenv()

# Normalize Hugging Face environment variables: some code/libs expect
# HUGGINGFACEHUB_API_TOKEN while older .env may use HUGGING_FACE_API_TOKEN or HUGGINGFACE_API_TOKEN.
hf_token_candidates = [
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGING_FACE_API_TOKEN",
    "HUGGINGFACE_API_TOKEN",
    "HUGGING_FACE_TOKEN",
    "HF_API_TOKEN",
]
found = None
for name in hf_token_candidates:
    val = os.getenv(name)
    if val:
        found = (name, val)
        break
if found:
    # Ensure canonical env var is set for libraries that expect HUGGINGFACEHUB_API_TOKEN
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = found[1]
        print(f"Set HUGGINGFACEHUB_API_TOKEN from {found[0]}.")
else:
    print("Warning: No Hugging Face API token found in environment. If you plan to use Hugging Face Hub endpoints, set HUGGINGFACEHUB_API_TOKEN in .env.")

# Set environment variables to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer parallelism warnings
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"  # Set matplotlib cache directory

# Initialize encryption for API keys
def get_or_create_encryption_key():
    key = os.getenv('ENCRYPTION_KEY')
    if not key:
        key = Fernet.generate_key().decode()
        with open('.env', 'a') as f:
            f.write(f'\nENCRYPTION_KEY={key}')
    return key.encode()

fernet = Fernet(get_or_create_encryption_key())

def encrypt_api_key(api_key: str) -> bytes:
    return fernet.encrypt(api_key.encode())

def decrypt_api_key(encrypted_key: bytes) -> str:
    return fernet.decrypt(encrypted_key).decode()

def load_api_keys() -> Dict[str, str]:
    api_keys = {}
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    if key != 'ENCRYPTION_KEY':
                        api_keys[key] = value.strip('"')
    return api_keys

def save_api_keys(api_keys: Dict[str, str]) -> None:
    try:
        encryption_key = os.getenv('ENCRYPTION_KEY', '')
        # Create backup of current .env file
        if os.path.exists('.env'):
            import shutil
            shutil.copy2('.env', '.env.backup')
        
        with open('.env', 'w') as f:
            f.write(f'ENCRYPTION_KEY={encryption_key}\n')
            for key, value in api_keys.items():
                if key != 'ENCRYPTION_KEY':
                    # Sanitize the value to prevent injection
                    safe_value = value.replace('"', '\"').strip()
                    f.write(f'{key}="{safe_value}"\n')
    except Exception as e:
        # Restore from backup if save failed
        if os.path.exists('.env.backup'):
            shutil.copy2('.env.backup', '.env')
        raise Exception(f"Failed to save API keys: {str(e)}")

def update_api_key(key_name: str, new_value: str) -> str:
    if not key_name or not new_value:
        return "Error: API key name and value are required"
    if key_name == 'ENCRYPTION_KEY':
        return "Error: Cannot modify encryption key"
    try:
        api_keys = load_api_keys()
        api_keys[key_name] = new_value.strip()
        save_api_keys(api_keys)
        return f"Successfully updated {key_name}"
    except Exception as e:
        return f"Error updating API key: {str(e)}"

def toggle_api_key_visibility(key_value: str, is_visible: bool) -> str:
    return key_value if is_visible else '*' * len(key_value)

# --- Define constants and load models ---
FAISS_INDEX_PATH = "my_faiss_index"

# Determine device for PyTorch operations
if torch.backends.mps.is_available():
    device = "mps"
    print("M2/M3 Chip detected. Using Apple's Metal Performance Senders (MPS).")
else:
    device = "cpu"
    print("MPS not available. Using CPU.")

# Load Local Summarization Model
print("Loading Local Summarization model...")
local_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
print("Local Summarization model loaded.")

# Load Embedding Model
print("Loading Embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': device}
)
print("Embedding model loaded.")

# Load persisted knowledge base if it exists
persisted_knowledge_base = None
if os.path.exists(FAISS_INDEX_PATH):
    print("Found existing knowledge base. Loading...")
    try:
        persisted_knowledge_base = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
        print("Knowledge base loaded successfully.")
    except Exception as e:
        print(f"Error loading existing knowledge base: {e}")
else:
    print("No existing knowledge base found.")

# --- Define Combat Agents (Global Instances) ---
ALL_DEBATE_AGENTS = {
    "Arthur (Logical Male)": DebateAgent("Arthur", "Male", "Logical"),
    "Fiona (Emphatic Female)": DebateAgent("Fiona", "Female", "Emphatic"),
    "Brad (Aggressive Male)": DebateAgent("Brad", "Male", "Aggressive"),
    "Clara (Analytical Female)": DebateAgent("Clara", "Female", "Analytical"),
    "Zenith (Neutral Mediator)": DebateAgent("Zenith", "Neutral", "Conciliatory")
}
DEBATE_AGENT_NAMES = list(ALL_DEBATE_AGENTS.keys())


# --- Dynamic Agent Dropdown Logic ---
def update_agent_dropdown_choices(
    team1_selected_names: List[str], 
    team2_selected_names: List[str]
) -> Tuple[gr.Dropdown, gr.Dropdown]:
    
    all_names_set = set(DEBATE_AGENT_NAMES)
    
    team2_selected_set = set(team2_selected_names)
    available_for_team1 = sorted(list(all_names_set - team2_selected_set))
    
    team1_selected_set = set(team1_selected_names)
    available_for_team2 = sorted(list(all_names_set - team1_selected_set))
    
    return gr.Dropdown(choices=available_for_team1, value=team1_selected_names), \
           gr.Dropdown(choices=available_for_team2, value=team2_selected_names)


# --- Dispatcher and Logic Functions for existing tabs ---

async def chat_dispatcher(question: str, chat_history: List[Dict], knowledge_base: Optional[FAISS]) -> Tuple[str, List[Dict]]:
    llm = create_groq_chat(model_name="llama-3.1-8b-instant", temperature=0.2)
    if llm is None:
        return "Error: Groq LLM initialization failed. Set GROQ_MODEL environment variable to a supported model.", chat_history
    return await answer_conversational(llm, knowledge_base, question, chat_history)

async def summarizer_dispatcher(path_choice: str, url: str) -> str:
    if not url: return "Please enter a URL."
    if path_choice == "tejreddym Cloud Engine":
        return await summarize_with_groq(url)
    elif path_choice == "Local & Private":
        return summarize_with_local_model(url, local_summarizer)
    else:
        return "Error: Invalid path selected."

# Researcher Dispatcher now only uses Groq and returns a report
async def researcher_dispatcher_with_llm_choice(llm_choice: str, tool_choice: str, query: str) -> str: # llm_choice is currently ignored for Groq only
    if not query: return "Please enter a query."
    
    # Force Groq Cloud for researcher as Google AI had issues.
    main_research_llm = create_groq_chat(model_name="llama-3.3-70b-versatile", temperature=0)
    
    if not main_research_llm:
        return "Error: Failed to initialize Groq LLM for researcher. Set GROQ_MODEL to a supported model."

    return await run_web_researcher(main_research_llm, query, tool_choice)


def clear_saved_kb() -> Tuple[None, None, str, List]:
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            shutil.rmtree(FAISS_INDEX_PATH)
            gr.Info("Saved knowledge base has been cleared.")
            return None, None, "Saved knowledge base cleared. Upload new files.", []
        except Exception as e:
            gr.Error(f"Error clearing knowledge base: {e}")
            return None, None, f"Error: {e}", []
    else:
        gr.Info("No saved knowledge base to clear.")
        return None, None, "No saved knowledge base to clear.", []

# --- Combat Arena Logic Dispatchers ---

current_debate_task: Optional[asyncio.Task] = None

async def start_debate_dispatcher(
    topic: str,
    team1_selected_agents: List[str],
    team2_selected_agents: List[str],
    max_turns_per_side: int,
    judge_option: str,
    chatbot_history: List[Dict],
    judge_verdict_output_state: str
) -> AsyncGenerator[Tuple[List[Dict], str, gr.Button, gr.Button], None]:
    
    global current_debate_task
    
    if current_debate_task and not current_debate_task.done():
        gr.Warning("A debate is already running. Please stop it first.")
        yield chatbot_history, judge_verdict_output_state, gr.Button(interactive=True), gr.Button(interactive=False) 
        return

    if not topic:
        gr.Warning("Please provide a debate topic.")
        yield chatbot_history, judge_verdict_output_state, gr.Button(interactive=True), gr.Button(interactive=False)
        return

    if not team1_selected_agents or not team2_selected_agents:
        gr.Warning("Please select at least one agent for each team.")
        yield chatbot_history, judge_verdict_output_state, gr.Button(interactive=True), gr.Button(interactive=False)
        return
    
    common_agents = set(team1_selected_agents).intersection(set(team2_selected_agents))
    if common_agents:
        gr.Error(f"Error: Agent(s) '{', '.join(common_agents)}' cannot be in both teams. Please re-select.")
        yield chatbot_history, judge_verdict_output_state, gr.Button(interactive=True), gr.Button(interactive=False)
        return


    chatbot_history = []
    judge_verdict_output_state = ""
    gr.Info("Starting debate...")

    yield chatbot_history, judge_verdict_output_state, gr.Button(interactive=False), gr.Button(interactive=True) 

    try:
        team1_agents_instances = [ALL_DEBATE_AGENTS[name] for name in team1_selected_agents]
        team2_agents_instances = [ALL_DEBATE_AGENTS[name] for name in team2_selected_agents]
        
        judge_at_end = (judge_option == "AI Judged")

        debate_history, final_verdict = await run_debate(
            topic=topic,
            team1_agents=team1_agents_instances,
            team2_agents=team2_agents_instances,
            max_turns_per_side=max_turns_per_side,
            judge_at_end=judge_at_end
        )
        
        gr_chatbot_messages = []
        team1_names_set = set(name for name in team1_selected_agents)

        for entry in debate_history:
            role = "user" if entry["name"] in team1_names_set else "assistant"
            gr_chatbot_messages.append({"role": role, "content": f"**{entry['name']} ({entry['team']})**: {entry['content']}"})

        gr.Info("Debate finished.")
        yield gr_chatbot_messages, final_verdict, gr.Button(interactive=True), gr.Button(interactive=False)

    except asyncio.CancelledError:
        gr.Warning("Debate cancelled by user.")
        yield chatbot_history, "Debate cancelled by user.", gr.Button(interactive=True), gr.Button(interactive=False)
    except Exception as e:
        gr.Error(f"Error during debate: {e}")
        yield chatbot_history, f"Error: {e}", gr.Button(interactive=True), gr.Button(interactive=False)


async def stop_debate_dispatcher() -> Tuple[str, gr.Button, gr.Button]:
    global current_debate_task
    if current_debate_task and not current_debate_task.done():
        current_debate_task.cancel()
        gr.Warning("Debate forcefully stopped.")
        await asyncio.to_thread(asyncio.sleep, 0.1) 
        return "Debate stopped by user.", gr.Button(interactive=True), gr.Button(interactive=False)
    else:
        return "", gr.Button(interactive=True), gr.Button(interactive=False)


# --- Web KB Dispatchers ---

async def create_web_kb_dispatcher(urls_input: str, web_kb_status_output: str) -> Tuple[Optional[FAISS], str, gr.Textbox, gr.Button]:
    gr.Info("Creating web knowledge base...")
    kb, status = await create_web_knowledge_base(urls_input, embedding_model)
    chat_input_interactive = True if kb else False
    return kb, status, gr.Textbox(interactive=chat_input_interactive), gr.Button(interactive=chat_input_interactive)


async def web_chat_dispatcher(question: str, web_chat_history: List[Dict], web_knowledge_base: Optional[FAISS]) -> Tuple[str, List[Dict]]:
    llm = ChatGroq(temperature=0.2, model_name="llama-3.1-8b-instant") # Using 8B model for chat
    return await answer_web_kb_conversational(llm, web_knowledge_base, question, web_chat_history)


# --- Meeting Prep Dispatcher ---
async def generate_meeting_prep_dispatcher(
    meeting_topic: str,
    urls_input: str,
    uploaded_files: List[Any] # Gradio File objects from gr.File
) -> Tuple[str, str]: # Returns Markdown notes and status message
    gr.Info("Preparing meeting notes...")
    prep_notes, status = await prepare_for_meeting(
        meeting_topic, urls_input, uploaded_files, embedding_model
    )
    return prep_notes, status

# --- Code Assistant Dispatcher ---
async def generate_code_dispatcher(user_prompt: str, programming_language: str) -> Tuple[str, str]:
    gr.Info(f"Requesting code from AI for {programming_language}...")
    code_output, status = await generate_code_response(user_prompt, programming_language)
    return code_output, status


# --- Data Analysis Dispatcher ---
async def data_analysis_dispatcher(
    dataframe_state: pd.DataFrame, # Pass the DataFrame state
    user_query: str
) -> Tuple[str, str]: # Returns analysis output markdown and status
    gr.Info("AI is analyzing your data...")
    analysis_output, status = await analyze_data_with_llm(dataframe_state, user_query)
    return analysis_output, status


# --- GRADIO UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# A.T.L.A.S.")
    gr.Markdown("*(Advanced Team Language & Analytical System)*")

    # --- Tab: Chat With My Docs ---
    with gr.Tab("Chat With My Docs"):
        uploaded_files_state = gr.State([])
        knowledge_base_state = gr.State(persisted_knowledge_base)
        
        gr.Markdown("## Step 1: Manage Knowledge Base")
        with gr.Row():
            file_upload = gr.File(label="Upload documents to create or update knowledge base", file_count="multiple", scale=3)
            with gr.Column(scale=1):
                create_kb_button = gr.Button("Create / Update KB")
                clear_kb_button = gr.Button("Clear Saved KB", variant="stop")
        initial_kb_status = "Knowledge base loaded from disk." if persisted_knowledge_base else "No knowledge base loaded."
        kb_status = gr.Textbox(label="Knowledge Base Status", value=initial_kb_status, interactive=False)
        
        gr.Markdown("---")
        gr.Markdown("## Step 2: Have a Conversation")
        gr.Markdown("This chat uses the **tejreddym Cloud Engine**.")
        
        chatbot = gr.Chatbot(label="Conversation", height=400, render=True, type="messages")
        
        with gr.Row():
            chat_input = gr.Textbox(label="Your message", placeholder="Ask a follow-up question...", scale=4)
            send_button = gr.Button("Send", scale=1)
            clear_chat_button = gr.Button("Clear Chat", scale=1)
            
        file_upload.upload(lambda files: files, inputs=file_upload, outputs=uploaded_files_state)
        create_kb_button.click(fn=lambda files: create_knowledge_base(files, embedding_model), inputs=uploaded_files_state, outputs=[knowledge_base_state, kb_status])
        clear_kb_button.click(fn=clear_saved_kb, inputs=None, outputs=[knowledge_base_state, file_upload, kb_status, chatbot])
        
        send_button.click(fn=chat_dispatcher, inputs=[chat_input, chatbot, knowledge_base_state], outputs=[chat_input, chatbot])
        chat_input.submit(fn=chat_dispatcher, inputs=[chat_input, chatbot, knowledge_base_state], outputs=[chat_input, chatbot])
        
        clear_chat_button.click(lambda: [], None, chatbot, queue=False)

    # --- Tab: Summarize Webpage ---
    with gr.Tab("Summarize Webpage"):
        gr.Markdown("## Summarize a Webpage")
        path_selector_summary = gr.Radio(["Local & Private", "tejreddym Cloud Engine"], label="Choose Summarization Engine", value="Local & Private")
        url_input = gr.Textbox(label="Enter a URL")
        summary_output = gr.Textbox(label="Summary", lines=5, interactive=False)
        summarize_button = gr.Button("Summarize")
        summarize_button.click(fn=summarizer_dispatcher, inputs=[path_selector_summary, url_input], outputs=summary_output)

    # --- Tab: Web Researcher ---
    with gr.Tab("Web Researcher"):
        gr.Markdown("## Proactive Web Researcher")
        gr.Markdown("Ask a question and choose a tool, or let the agent decide with 'Auto' mode.")
        
        # LLM choice for Web Researcher
        researcher_llm_choice = gr.Radio(
            ["Groq Cloud (Llama3-70B)", "Google AI (Gemma 3 12B)"],
            label="Choose Researcher's Brain (LLM)",
            value="Groq Cloud (Llama3-70B)"
        )

        tool_selector = gr.Dropdown(["Auto", "Wikipedia", "Smart Search", "Calculator", "ArXiv Search", "Current Time", "Image Generator"], label="Choose Tool", value="Auto")
        with gr.Row():
            research_question = gr.Textbox(label="Your Question", scale=3)
            research_button = gr.Button("Start Research", scale=1)
        research_answer = gr.Markdown("### Answer")
        
        research_button.click(
            fn=researcher_dispatcher_with_llm_choice,
            inputs=[researcher_llm_choice, tool_selector, research_question],
            outputs=research_answer
        )

    # --- Tab: Image Generation ---
    with gr.Tab("Image Generation"):
        gr.Markdown("## Generate Images with AI")
        gr.Markdown("Create images from text prompts using different models.")

        with gr.Tab("General Image Generation"):
            gr.Markdown("### Stability AI (General Purpose)")
            general_image_prompt = gr.Textbox(label="Prompt", placeholder="A futuristic city at sunset")
            general_image_output = gr.Image(label="Generated Image", type="pil", height=512)
            general_image_status = gr.Textbox(label="Status", interactive=False)
            generate_general_image_button = gr.Button("Generate General Image")
            generate_general_image_button.click(
                fn=generate_image,
                inputs=[general_image_prompt],
                outputs=[general_image_output, general_image_status]
            )
        
        with gr.Tab("Anime Image Generation"):
            gr.Markdown("### Heartsync/Anime (via Gradio Client)")
            anime_image_prompt = gr.Textbox(label="Prompt", placeholder="1girl, solo, masterpiece, best quality, blue hair")
            anime_image_output = gr.Image(label="Generated Anime Image", type="filepath", height=512)
            anime_image_status = gr.Textbox(label="Status", interactive=False)
            generate_anime_image_button = gr.Button("Generate Anime Image")
            generate_anime_image_button.click(
                fn=generate_anime_image,
                inputs=[anime_image_prompt],
                outputs=[anime_image_output, anime_image_status]
            )

    # --- Tab: Agents Combat Arena (Modified) ---
    with gr.Tab("Agents Combat Arena"):
        gr.Markdown("## Agents Combat Arena")
        gr.Markdown("Set up a debate between AI agents and see whose argument prevails!")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Debate Setup")
                debate_topic_input = gr.Textbox(
                    label="Debate Topic",
                    placeholder="e.g., Is AI a threat or a boon to humanity?",
                    lines=2
                )
                
                with gr.Row():
                    # Team 1 Dropdown
                    team1_agents_dropdown = gr.Dropdown(
                        label="Select Agents for Team 1",
                        choices=DEBATE_AGENT_NAMES,
                        multiselect=True,
                        value=[DEBATE_AGENT_NAMES[0]] # Default to first agent
                    )
                    # Team 2 Dropdown
                    team2_agents_dropdown = gr.Dropdown(
                        label="Select Agents for Team 2",
                        choices=DEBATE_AGENT_NAMES,
                        multiselect=True,
                        value=[DEBATE_AGENT_NAMES[1]] # Default to second agent
                    )
                
                max_turns_per_side_slider = gr.Slider(
                    label="Max Turns per Side (each agent)",
                    minimum=1, maximum=5, value=2, step=1, interactive=True
                )

                judge_option_radio = gr.Radio(
                    label="Judge Debate",
                    choices=["User Judged", "AI Judged"],
                    value="AI Judged"
                )

                with gr.Row():
                    start_debate_button = gr.Button("Start Debate", variant="primary")
                    stop_debate_button = gr.Button("Stop Debate", variant="stop", interactive=False)
                    clear_arena_button = gr.Button("Clear Arena", variant="secondary")

            with gr.Column(scale=3):
                gr.Markdown("### Live Debate")
                debate_chatbot = gr.Chatbot(
                    label="Debate Log",
                    height=500,
                    type="messages",
                    show_copy_button=True
                )
                gr.Markdown("### Verdict")
                judge_verdict_output = gr.Markdown("No debate has concluded yet.")
        
        # --- Wire up Dynamic Agent Dropdown Choices ---
        team1_agents_dropdown.change(
            fn=update_agent_dropdown_choices,
            inputs=[team1_agents_dropdown, team2_agents_dropdown],
            outputs=[team1_agents_dropdown, team2_agents_dropdown]
        )
        team2_agents_dropdown.change(
            fn=update_agent_dropdown_choices,
            inputs=[team1_agents_dropdown, team2_agents_dropdown],
            outputs=[team1_agents_dropdown, team2_agents_dropdown]
        )

        start_debate_button.click(
            fn=start_debate_dispatcher,
            inputs=[
                debate_topic_input,
                team1_agents_dropdown,
                team2_agents_dropdown,
                max_turns_per_side_slider,
                judge_option_radio,
                debate_chatbot,
                judge_verdict_output
            ],
            outputs=[
                debate_chatbot,
                judge_verdict_output,
                start_debate_button,
                stop_debate_button
            ]
        )
        
        stop_debate_button.click(
            fn=stop_debate_dispatcher,
            inputs=[],
            outputs=[
                judge_verdict_output,
                start_debate_button,
                stop_debate_button
            ]
        )

        clear_arena_button.click(
            lambda: ([], "", gr.Button(interactive=True), gr.Button(interactive=False)),
            outputs=[debate_chatbot, judge_verdict_output, start_debate_button, stop_debate_button]
        )

    # --- Tab: Chat With My Web ---
    with gr.Tab("Chat With My Web"):
        gr.Markdown("## Enhanced Web & Document Knowledge Base")
        gr.Markdown("🚀 **New Features:** Now supports PDFs, HTML pages, text files, and JSON!")
        gr.Markdown("Provide URLs to web pages, PDF documents, or text files. A.T.L.A.S. will extract and process content to create a searchable knowledge base.")
        
        with gr.Row():
            with gr.Column(scale=2):
                web_urls_input = gr.Textbox(
                    label="📎 Web Links & Document URLs (one per line)", 
                    placeholder="""Paste URLs here, examples:
🌐 Web pages: https://en.wikipedia.org/wiki/Artificial_intelligence
📄 PDF documents: https://example.com/research-paper.pdf
📝 Text files: https://raw.githubusercontent.com/user/repo/main/README.md
📊 JSON data: https://api.example.com/data.json""", 
                    lines=6
                )
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Supported Content Types")
                gr.Markdown("""
                ✅ **HTML Web Pages**  
                ✅ **PDF Documents** (NEW!)  
                ✅ **Plain Text Files**  
                ✅ **JSON Data** (NEW!)  
                ✅ **Markdown Files**  
                
                **Limits:** 15 URLs, 50MB per file
                """)
        
        # States for web knowledge base and chat history
        web_knowledge_base_state = gr.State(None)
        web_chat_history_state = gr.State([])

        with gr.Row():
            create_web_kb_button = gr.Button("🔨 Create Enhanced Knowledge Base", variant="primary", size="lg")
            
        web_kb_status = gr.Textbox(
            label="📊 Knowledge Base Status", 
            interactive=False, 
            value="No knowledge base loaded. Enter URLs above and click 'Create Enhanced Knowledge Base'.",
            lines=3
        )
            
        gr.Markdown("---")
        gr.Markdown("## 💬 Chat with Your Knowledge Base")
        gr.Markdown("Ask questions about the content from your URLs. The AI will search across all processed documents and provide sourced answers.")
        
        web_chatbot = gr.Chatbot(
            label="Enhanced Web Knowledge Chat", 
            height=450, 
            render=True, 
            type="messages",
            show_copy_button=True,
            placeholder="Your conversation will appear here once you create a knowledge base..."
        )
        
        with gr.Row():
            web_chat_input = gr.Textbox(
                label="💬 Your question", 
                placeholder="Ask about the content from your URLs (e.g., 'What are the main findings in the research paper?')", 
                scale=4, 
                interactive=False
            )
            web_send_button = gr.Button("🚀 Send", scale=1, interactive=False, variant="primary")
            web_clear_chat_button = gr.Button("🗑️ Clear Chat", scale=1)
        
        # Example questions
        gr.Markdown("### 💡 Example Questions You Can Ask:")
        with gr.Row():
            gr.Markdown("""
            - "Summarize the main points from the PDF"
            - "What are the key findings in the research?"
            - "Compare information from different sources"
            - "What does source X say about topic Y?"
            """)
        
        # --- Wire up Enhanced Web KB Creation ---
        create_web_kb_button.click(
            fn=create_web_kb_dispatcher,
            inputs=[web_urls_input, web_kb_status],
            outputs=[web_knowledge_base_state, web_kb_status, web_chat_input, web_send_button]
        )
        
        # --- Wire up Enhanced Web KB Chat ---
        web_send_button.click(
            fn=web_chat_dispatcher,
            inputs=[web_chat_input, web_chatbot, web_knowledge_base_state],
            outputs=[web_chat_input, web_chatbot]
        )
        web_chat_input.submit(
            fn=web_chat_dispatcher,
            inputs=[web_chat_input, web_chatbot, web_knowledge_base_state],
            outputs=[web_chat_input, web_chatbot]
        )
        
        # --- Clear Web KB Chat ---
        web_clear_chat_button.click(
            lambda: ([], "Knowledge base still loaded. Enter new URLs to create a fresh knowledge base.", gr.Textbox(interactive=False), gr.Button(interactive=False)),
            None, 
            [web_chatbot, web_kb_status, web_chat_input, web_send_button], 
            queue=False
        )

    # --- NEW Tab: Meeting Prep Assistant ---
    with gr.Tab("Meeting Prep Assistant"):
        gr.Markdown("## Prepare for Your Meetings with AI Assistance")
        gr.Markdown("Provide your meeting topic, relevant web links, and local documents. A.T.L.A.S. will synthesize key talking points, questions, and summaries.")

        with gr.Column():
            meeting_topic_input = gr.Textbox(label="Meeting Topic / Goal", placeholder="e.g., 'Q3 Sales Strategy Review'", lines=2)
            
            meeting_urls_input = gr.Textbox(label="Relevant Web Links (one per line)", placeholder="e.g., latest industry reports, competitor news", lines=3)
            
            meeting_files_upload = gr.File(label="Relevant Local Documents (PDF, TXT, DOCX)", file_count="multiple", type="filepath")
            
            generate_prep_button = gr.Button("Generate Meeting Prep Notes", variant="primary")
            
            meeting_prep_status = gr.Textbox(label="Status", interactive=False, lines=2)
            meeting_prep_output = gr.Markdown("### Meeting Preparation Notes\n(Notes will appear here)", elem_id="meeting_prep_output") # Using Markdown for rich text

        # --- Wire up Meeting Prep Dispatcher ---
        generate_prep_button.click(
            fn=generate_meeting_prep_dispatcher,
            inputs=[
                meeting_topic_input,
                meeting_urls_input,
                meeting_files_upload
            ],
            outputs=[meeting_prep_output, meeting_prep_status]
        )

    # --- NEW Tab: AI-Powered Code Assistant ---
    with gr.Tab("Code Assistant"):
        gr.Markdown("## AI-Powered Code Assistant")
        gr.Markdown("Ask A.T.L.A.S. to generate code, explain snippets, or debug issues in various programming languages.")

        with gr.Column():
            code_prompt_input = gr.Textbox(
                label="Your Coding Request",
                placeholder="e.g., 'Write a Python function to perform a quick sort on a list.' or 'Explain this JavaScript Promise example.'",
                lines=5
            )
            programming_language_dropdown = gr.Dropdown(
                label="Programming Language",
                choices=["Python", "JavaScript", "Java", "C++", "Go", "Rust", "SQL", "Bash", "HTML/CSS"],
                value="Python"
            )
            
            generate_code_button = gr.Button("Generate Code / Explanation", variant="primary")
            
            code_assistant_status = gr.Textbox(label="Status", interactive=False, lines=1)
            code_output_display = gr.Markdown("### Generated Code / Explanation\n(Output will appear here)", elem_id="code_output_display")

        # --- Wire up Code Assistant Dispatcher ---
        generate_code_button.click(
            fn=generate_code_dispatcher,
            inputs=[
                code_prompt_input,
                programming_language_dropdown
            ],
            outputs=[code_output_display, code_assistant_status]
        )

    # --- NEW Tab: API Management ---
    with gr.Tab("API Management"):
        gr.Markdown("## API Key Management")
        gr.Markdown("View and manage your API keys securely. Keys are encrypted before storage.")
        
        api_keys = load_api_keys()
        with gr.Column():
            api_keys_container = gr.Column()
            for key_name, key_value in api_keys.items():
                if key_name != 'ENCRYPTION_KEY':
                    with gr.Row():
                        key_name_text = gr.Text(value=key_name, label="API Key Name", interactive=False)
                        key_value_text = gr.Text(value=toggle_api_key_visibility(key_value, False), label="API Key Value", interactive=True)
                        show_button = gr.Button("👁️", scale=1)
                        update_button = gr.Button("Update", scale=2)
                        delete_button = gr.Button("Delete", scale=2, variant="stop")
                        
                        # Status message for this row
                        status_message = gr.Text(value="", label="Status", visible=True)
                        
                        # Wire up visibility toggle (simplified)
                        def toggle_visibility(current_value):
                            if '*' in current_value:
                                return key_value  # Show actual value
                            else:
                                return toggle_api_key_visibility(key_value, False)  # Hide value
                        
                        show_button.click(
                            fn=toggle_visibility,
                            inputs=[key_value_text],
                            outputs=key_value_text
                        )
                        
                        # Wire up update functionality (simplified)
                        update_button.click(
                            fn=update_api_key,
                            inputs=[key_name_text, key_value_text],
                            outputs=status_message
                        )
                        
                        # Wire up delete functionality (simplified)
                        def delete_key(key_name: str) -> str:
                            try:
                                api_keys = load_api_keys()
                                if key_name in api_keys:
                                    del api_keys[key_name]
                                    save_api_keys(api_keys)
                                    return f"Successfully deleted {key_name}"
                                else:
                                    return f"Key {key_name} not found"
                            except Exception as e:
                                return f"Error deleting key: {str(e)}"
                        
                        delete_button.click(
                            fn=delete_key,
                            inputs=key_name_text,
                            outputs=status_message
                        )
            
            gr.Markdown("---")
            gr.Markdown("### Add New API Key")
            with gr.Column():
                new_key_name = gr.Text(label="New API Key Name", placeholder="Enter new API key name", max_lines=1)
                new_key_value = gr.Text(label="New API Key Value", placeholder="Enter new API key value", max_lines=1, type="password")
                add_status = gr.Text(label="Add Status", visible=True)
                add_button = gr.Button("Add New API Key", variant="primary")
            
            def validate_and_add_key(name: str, value: str) -> Tuple[str, str, str]:
                if not name or not value:
                    return "Error: Both name and value are required", name, value
                if name == 'ENCRYPTION_KEY':
                    return "Error: Cannot add encryption key", name, value
                if name in load_api_keys():
                    return "Error: Key name already exists", name, value
                
                result = update_api_key(name, value)
                if "Successfully" in result:
                    return result, "", ""  # Clear inputs on success
                else:
                    return result, name, value  # Keep inputs on error
            
            # Wire up new key addition (simplified)
            add_button.click(
                fn=validate_and_add_key,
                inputs=[new_key_name, new_key_value],
                outputs=[add_status, new_key_name, new_key_value]
            )

    # --- NEW Tab: Data Analyst (ADD THIS BLOCK) ---
    with gr.Tab("Data Analyst"):
        gr.Markdown("## AI-Powered Data Analysis")
        gr.Markdown("Upload a CSV or Excel file, then ask A.T.L.A.S. to analyze your tabular data using natural language queries.")

        with gr.Column():
            data_file_upload = gr.File(label="Upload CSV or Excel File", file_count="single", type="filepath", file_types=[".csv", ".xls", ".xlsx"])
            
            data_load_status = gr.Textbox(label="File Load Status", interactive=False, lines=1)
            data_preview_display = gr.Markdown("### Data Preview\n(Upload a file to see preview)", elem_id="data_preview_display")

            # State to hold the DataFrame after loading
            dataframe_state = gr.State(None)

            gr.Markdown("---")
            gr.Markdown("### Query Your Data")
            
            data_query_input = gr.Textbox(
                label="Your Data Analysis Query",
                placeholder="e.g., 'What is the total sales for products in the 'North' region?' or 'Summarize the average quantity sold per product category.'",
                lines=3,
                interactive=False # Initially disabled
            )
            
            analyze_data_button = gr.Button("Analyze Data", variant="primary", interactive=False) # Initially disabled
            
            data_analysis_status = gr.Textbox(label="Analysis Status", interactive=False, lines=1)
            data_analysis_output = gr.Markdown("### AI Analysis Results\n(Results will appear here)", elem_id="data_analysis_output")

        # --- Wire up Data Loading and Preview ---
        data_file_upload.upload(
            fn=load_and_preview_data,
            inputs=data_file_upload,
            outputs=[dataframe_state, data_preview_display, data_load_status, data_query_input]
        )

        # Wire up Data Analysis Query ---
        analyze_data_button.click(
            fn=data_analysis_dispatcher,
            inputs=[
                dataframe_state,
                data_query_input
            ],
            outputs=[data_analysis_output, data_analysis_status]
        )
        data_query_input.submit( # Allow pressing Enter in textbox
            fn=data_analysis_dispatcher,
            inputs=[
                dataframe_state,
                data_query_input
            ],
            outputs=[data_analysis_output, data_analysis_status]
        )


# This line launches the user interface
demo.launch()