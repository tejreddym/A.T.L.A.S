# meeting_prep_logic.py

import os
import asyncio
import gradio as gr
from typing import List, Dict, Tuple, Optional, Any

# Import knowledge base creation and conversational answer logic
from doc_qa_logic import create_knowledge_base # For local files
from web_kb_logic import create_web_knowledge_base # For web links

# For LLM synthesis
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

# For embeddings (passed from app.py)
# from langchain_huggingface import HuggingFaceEmbeddings # Not needed here, passed in

# --- Core Meeting Preparation Logic ---

async def prepare_for_meeting(
    meeting_topic: str,
    urls_input: str, # Newline separated URLs
    uploaded_files: List[Any], # Gradio File objects
    embedding_model # Passed from app.py
) -> Tuple[str, str]: # Returns Markdown notes and status message
    
    gr.Info("Starting meeting preparation...")
    prep_notes = ""
    status_message = "Preparation initiated."
    
    # Ensure LLM for synthesis is available
    try:
        # Use a powerful Groq LLM for synthesis
        synthesis_llm = ChatGroq(temperature=0.1, model_name="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY_JUDGE") or os.getenv("GROQ_API_KEY"))
    except Exception as e:
        status_message = f"Error: LLM for synthesis not initialized. Check API key. {e}"
        gr.Error(status_message)
        return "", status_message

    if not meeting_topic:
        status_message = "Please provide a meeting topic."
        gr.Warning(status_message)
        return "", status_message

    all_retrieved_content = []

    # 1. Process Web Links
    web_knowledge_base = None
    if urls_input:
        gr.Info("Processing web links...")
        # Create web KB (reusing web_kb_logic)
        web_knowledge_base, web_kb_status_msg = await create_web_knowledge_base(urls_input, embedding_model)
        gr.Info(f"Web KB Status: {web_kb_status_msg}")
        if web_knowledge_base:
            # Query web KB for top relevant documents related to meeting topic
            try:
                web_docs = web_knowledge_base.as_retriever(search_kwargs={"k": 5}).get_relevant_documents(meeting_topic)
                for doc in web_docs:
                    all_retrieved_content.append(f"Source (Web): {doc.metadata.get('source', 'N/A')}\nContent:\n{doc.page_content}\n---")
                gr.Info(f"Retrieved {len(web_docs)} relevant web documents.")
            except Exception as e:
                gr.Warning(f"Error querying web KB: {e}")
        else:
            gr.Warning("Could not create web knowledge base from provided links.")

    # 2. Process Local Documents
    doc_knowledge_base = None
    if uploaded_files:
        gr.Info("Processing local documents...")
        # Create local KB (reusing doc_qa_logic)
        doc_knowledge_base, doc_kb_status_msg = create_knowledge_base(uploaded_files, embedding_model)
        gr.Info(f"Doc KB Status: {doc_kb_status_msg}")
        if doc_knowledge_base:
            # Query local KB for top relevant documents
            try:
                doc_docs = doc_knowledge_base.as_retriever(search_kwargs={"k": 5}).get_relevant_documents(meeting_topic)
                for doc in doc_docs:
                    all_retrieved_content.append(f"Source (Doc): {os.path.basename(doc.metadata.get('source', 'N/A'))}\nContent:\n{doc.page_content}\n---")
                gr.Info(f"Retrieved {len(doc_docs)} relevant local documents.")
            except Exception as e:
                gr.Warning(f"Error querying local KB: {e}")
        else:
            gr.Warning("Could not create local knowledge base from uploaded files.")

    if not all_retrieved_content:
        status_message = "No relevant content found from provided links or documents."
        gr.Warning(status_message)
        return "No information gathered to prepare for the meeting.", status_message

    # 3. Synthesize Meeting Prep Notes
    gr.Info("Synthesizing meeting preparation notes...")
    
    combined_content = "\n\n".join(all_retrieved_content)
    
    # Ensure combined_content is not too long for LLM context window
    MAX_LLM_CONTEXT_CHARS = 30000 # ~30KB, adjust based on model context window
    if len(combined_content) > MAX_LLM_CONTEXT_CHARS:
        gr.Warning(f"Combined content is very large ({len(combined_content)} chars). Truncating for LLM.")
        combined_content = combined_content[:MAX_LLM_CONTEXT_CHARS]

    prep_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are a highly efficient meeting preparation assistant. "
            "Your task is to synthesize the provided background information and meeting topic "
            "into concise, actionable meeting preparation notes. "
            "The notes should be formatted in Markdown and include the following sections:\n"
            "1. **Meeting Topic Overview:** A brief summary of the main subject.\n"
            "2. **Key Talking Points:** Essential information or arguments to present.\n"
            "3. **Key Questions to Ask:** Important questions to raise or get answers for.\n"
            "4. **Actionable Insights/Next Steps:** Any immediate actions suggested by the content."
            "Ensure all points are directly supported by the provided content. Be concise."
        ),
        HumanMessage(content=f"Meeting Topic: {meeting_topic}\n\nBackground Information:\n{combined_content}\n\nGenerate Meeting Preparation Notes:")
    ])

    try:
        prep_chain = prep_prompt_template | synthesis_llm | StrOutputParser()
        prep_notes = await prep_chain.ainvoke({"meeting_topic": meeting_topic, "combined_content": combined_content})
        status_message = "Meeting preparation notes generated successfully!"
        gr.Info(status_message)
    except Exception as e:
        status_message = f"Error synthesizing meeting notes: {e}"
        gr.Error(status_message)
        print(f"Meeting Prep Synthesis Error: {e}")
        prep_notes = "Failed to generate meeting preparation notes due to an internal error."

    return prep_notes, status_message