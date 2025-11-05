# In doc_qa_logic.py

import gradio as gr
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
# Docx2txtLoader is optional (requires external `docx2txt` package). Make the import
# graceful so the module can still be imported when the package is not installed.
try:
    from langchain_community.document_loaders import Docx2txtLoader
    DOCX_LOADER_AVAILABLE = True
except Exception:
    Docx2txtLoader = None
    DOCX_LOADER_AVAILABLE = False
    print("Warning: 'docx2txt' or its LangChain loader is not available. .docx files will be skipped. Install with 'pip install docx2txt' to enable DOCX support.")
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage # <-- New Import

FAISS_INDEX_PATH = "my_faiss_index"

# This function is updated to handle the new chatbot message format
async def answer_conversational(llm, knowledge_base, question, chat_history):
    if knowledge_base is None:
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": "Please create a knowledge base first before asking questions."})
        return "", chat_history

    if not question:
        # Return an empty string for the input and the unchanged history
        return "", chat_history

    # Set up memory. LangChain's memory works with HumanMessage and AIMessage objects
    memory = ConversationBufferWindowMemory(
        memory_key='chat_history', return_messages=True, k=5
    )
    # Load past messages from the chat history, converting them to the correct type
    for message in chat_history:
        if message["role"] == "user":
            memory.chat_memory.add_user_message(message["content"])
        else:
            memory.chat_memory.add_ai_message(message["content"])

    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=knowledge_base.as_retriever(), memory=memory
    )
    
    gr.Info("Thinking...")
    result = await conversational_chain.ainvoke({"question": question})
    answer = result["answer"]
    
    # Append the new interaction to the history
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})
    
    # Return an empty string to clear the input, and the updated history
    return "", chat_history


def create_knowledge_base(files, embedding_model):
    # This function remains unchanged
    if not files:
        gr.Warning("No files uploaded. Please upload documents first.")
        return None, "Please upload documents first."
    try:
        all_docs = []
        gr.Info("Reading documents...")
        for file in files:
            file_path = file.name
            if not os.path.exists(file_path):
                gr.Warning(f"File path does not exist: {file_path}. Skipping.")
                continue
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_ext == '.docx':
                if not DOCX_LOADER_AVAILABLE:
                    gr.Warning(f"Skipping .docx file because the required loader is not available: {file_path}. Install 'docx2txt' to enable .docx support.")
                    continue
                try:
                    loader = Docx2txtLoader(file_path)
                except ModuleNotFoundError:
                    gr.Warning(f"Skipping .docx file because the 'docx2txt' package is not installed: {file_path}. Install 'docx2txt' to enable .docx support.")
                    continue
                except Exception as e:
                    gr.Warning(f"Skipping .docx file due to error initializing DOCX loader for {file_path}: {e}")
                    continue
            elif file_ext == '.txt':
                loader = TextLoader(file_path)
            else:
                gr.Warning(f"Unsupported file type: {file_ext}. Skipping.")
                continue
            all_docs.extend(loader.load())
        if not all_docs:
            return None, "No supported documents were found or processed."
        gr.Info("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(all_docs)
        gr.Info("Creating knowledge base...")
        knowledge_base = FAISS.from_documents(split_docs, embedding_model)
        knowledge_base.save_local(FAISS_INDEX_PATH)
        gr.Info("Knowledge base created and saved to disk!")
        return knowledge_base, f"Knowledge base created from {len(files)} file(s) and saved."
    except Exception as e:
        print(f"An error occurred in create_knowledge_base: {e}")
        gr.Error(f"Error creating knowledge base: {e}")
        return None, f"An error occurred: {e}"