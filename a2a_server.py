# In a2a_server.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# Load environment variables (for Groq/Tavily/HF keys)
from dotenv import load_dotenv
load_dotenv()

# Import the logic we want to expose as skills
# We import the _tool_func because it's designed to be called by another program
from image_generation_logic import generate_image_tool_func
from web_research_logic import run_web_researcher
from summarizer_logic import summarize_with_groq
from langchain_groq import ChatGroq

# --- Define the structure of A2A requests using Pydantic ---
class A2AParams(BaseModel):
    query: Optional[str] = None
    url: Optional[str] = None

class A2ARequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: A2AParams
    id: int

# --- Create the FastAPI application ---
app = FastAPI(
    title="A.T.L.A.S. A2A Server",
    description="Exposing the skills of A.T.L.A.S. to other AI agents.",
)

# --- Define the API "Skills" Endpoint ---
@app.post("/rpc")
async def agent_rpc_endpoint(request: A2ARequest):
    """
    This single endpoint handles all incoming requests from other agents.
    """
    print(f"A2A Server received request for method: {request.method}")
    method = request.method
    params = request.params
    request_id = request.id

    try:
        if method == "research_topic":
            if not params.query:
                raise ValueError("'query' parameter is required.")
            llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
            result = await run_web_researcher(llm, params.query, tool_choice="Auto")
            return {"jsonrpc": "2.0", "result": {"answer": result}, "id": request_id}

        elif method == "summarize_webpage":
            if not params.url:
                raise ValueError("'url' parameter is required.")
            result = await summarize_with_groq(params.url)
            return {"jsonrpc": "2.0", "result": {"summary": result}, "id": request_id}
            
        # --- THIS IS THE NEW SKILL ---
        elif method == "generate_image":
            if not params.query:
                raise ValueError("'query' parameter is required for the image prompt.")
            
            # FastAPI runs synchronous functions in a threadpool automatically
            status_message = generate_image_tool_func(params.query)
            
            return {"jsonrpc": "2.0", "result": {"status": status_message}, "id": request_id}
            
        else:
            # If the method is not recognized
            return {"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}, "id": request_id}

    except Exception as e:
        # Generic error handler
        return {"jsonrpc": "2.0", "error": {"code": -32602, "message": str(e)}, "id": request_id}


@app.get("/")
def read_root():
    return {"message": "A.T.L.A.S. A2A Server is running. Send POST requests to /rpc."}