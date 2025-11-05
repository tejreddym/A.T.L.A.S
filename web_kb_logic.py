# web_kb_logic.py

import os
import asyncio
import httpx # For making asynchronous HTTP requests
from bs4 import BeautifulSoup # For parsing HTML
import gradio as gr
from typing import List, Dict, Tuple, Optional
import PyPDF2
import io
import tempfile
from urllib.parse import urlparse, urljoin
import mimetypes
import traceback  # For detailed error logging

from langchain_community.vectorstores import FAISS # For vector store
from langchain.text_splitter import RecursiveCharacterTextSplitter # For text splitting
from langchain_core.documents import Document # For creating documents

# Re-use answer_conversational logic from doc_qa_logic, or re-implement here for self-containment
# For simplicity, we will copy the core logic here.
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
import re


# --- URL validation and preprocessing ---
def _validate_and_clean_urls(url_list: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Validates and cleans a list of URLs, providing feedback on issues.
    Returns: (valid_urls, issues_dict)
    """
    valid_urls = []
    issues = {}
    
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    for i, url in enumerate(url_list, 1):
        url = url.strip()
        
        # Skip empty URLs
        if not url:
            continue
            
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            if url.startswith('www.'):
                url = 'https://' + url
            elif '.' in url:
                url = 'https://' + url
            else:
                issues[f"URL {i}"] = f"Invalid URL format: {url}"
                continue
        
        # Validate URL format
        if url_pattern.match(url):
            valid_urls.append(url)
        else:
            issues[f"URL {i}"] = f"Invalid URL format: {url}"
    
    return valid_urls, issues


# --- Enhanced content type detection helper ---
async def _detect_content_type_from_url(url: str) -> str:
    """Enhanced content type detection from URL extension, handling query parameters."""
    # Remove query parameters and fragments for extension detection
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    clean_path = parsed_url.path.lower()
    
    if clean_path.endswith('.pdf'):
        return 'application/pdf'
    elif clean_path.endswith(('.txt', '.md', '.rst')):
        return 'text/plain'
    elif clean_path.endswith('.json'):
        return 'application/json'
    elif clean_path.endswith(('.html', '.htm')):
        return 'text/html'
    else:
        return 'unknown'


# --- Content inspection helper ---
def _inspect_content_for_type(content_bytes: bytes, url: str) -> str:
    """Inspect the actual content to determine type if headers are unreliable."""
    # Check for PDF magic number
    if content_bytes.startswith(b'%PDF'):
        return 'application/pdf'
    
    # Try to decode as text and check for HTML
    try:
        text_content = content_bytes.decode('utf-8', errors='ignore')[:1000]  # First 1000 chars
        if any(tag in text_content.lower() for tag in ['<html', '<head', '<body', '<!doctype html']):
            return 'text/html'
        elif text_content.strip().startswith(('{', '[')):
            return 'application/json'
        else:
            return 'text/plain'
    except:
        return 'unknown'


# --- Enhanced function to detect and process different content types ---
async def _fetch_and_process_content(url: str) -> Optional[Tuple[str, str]]:
    """
    Fetches content from a URL and returns cleaned text with content type.
    Supports HTML pages, PDFs, and plain text files.
    Returns: (cleaned_text, content_type) or None if failed
    """
    try:
        gr.Info(f"Analyzing content from: {url}")
        
        # Set up headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0, headers=headers) as client:
            # First, try a HEAD request to check content type without downloading everything
            try:
                head_response = await client.head(url)
                content_type = head_response.headers.get('content-type', '').lower()
                content_length = head_response.headers.get('content-length')
                
                # Check file size (limit to 50MB)
                if content_length and int(content_length) > 50 * 1024 * 1024:
                    gr.Warning(f"File too large ({int(content_length)/(1024*1024):.1f}MB). Skipping: {url}")
                    return None
                    
            except:
                # If HEAD fails, we'll proceed with GET and check content type then
                content_type = ""
            
            # Now get the actual content
            response = await client.get(url)
            response.raise_for_status()
            
            # Enhanced content type detection
            server_content_type = response.headers.get('content-type', '').lower()
            url_based_type = await _detect_content_type_from_url(url)
            
            # Inspect actual content if server type is unreliable or missing
            content_based_type = _inspect_content_for_type(response.content, url)
            
            # Priority: URL extension > content inspection > server header
            if url_based_type != 'unknown':
                final_content_type = url_based_type
            elif content_based_type != 'unknown':
                final_content_type = content_based_type
            else:
                final_content_type = server_content_type
            
            # Console debug logging
            print(f"[WEB_KB DEBUG] URL: {url}")
            print(f"[WEB_KB DEBUG] Content type detection - Server: '{server_content_type}', URL: '{url_based_type}', Content: '{content_based_type}', Final: '{final_content_type}'")
            
            gr.Info(f"Content type detection - Server: {server_content_type}, URL: {url_based_type}, Content: {content_based_type}, Final: {final_content_type}")
            
            # Process content based on detected type
            if final_content_type == 'application/pdf' or 'application/pdf' in server_content_type:
                return await _process_pdf_content(response.content, url)
            elif final_content_type == 'text/html' or 'text/html' in server_content_type:
                return await _process_html_content(response.text, url)
            elif final_content_type == 'text/plain':
                return response.text.strip(), "text/plain"
            elif final_content_type == 'application/json':
                # Try to extract readable content from JSON
                try:
                    import json
                    json_data = json.loads(response.text)
                    # Convert JSON to readable text
                    text_content = json.dumps(json_data, indent=2, ensure_ascii=False)
                    return text_content, "application/json"
                except:
                    return response.text, "application/json"
            else:
                # Try to process as text anyway, but warn user
                gr.Warning(f"Unknown content type '{final_content_type}' (server: '{server_content_type}') for {url}. Attempting to process as text.")
                try:
                    return response.text[:10000], f"unknown ({final_content_type})"  # Limit to first 10k chars
                except:
                    gr.Warning(f"Could not process content from {url}")
                    return None
                    
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP Error {e.response.status_code} fetching {url}: {e}"
        print(f"[WEB_KB ERROR] {error_msg}")
        print(f"[WEB_KB ERROR] Response headers: {dict(e.response.headers)}")
        gr.Error(error_msg)
        return None
    except httpx.RequestError as e:
        error_msg = f"Network Error fetching {url}: {e}"
        print(f"[WEB_KB ERROR] {error_msg}")
        gr.Error(error_msg)
        return None
    except Exception as e:
        error_msg = f"Unexpected error processing {url}: {e}"
        print(f"[WEB_KB ERROR] {error_msg}")
        print(f"[WEB_KB ERROR] Full traceback: {traceback.format_exc()}")
        gr.Error(error_msg)
        return None


async def _process_pdf_content(pdf_bytes: bytes, url: str) -> Optional[Tuple[str, str]]:
    """Extract text from PDF bytes."""
    try:
        gr.Info(f"Extracting text from PDF: {url}")
        
        # Create a file-like object from bytes
        pdf_file = io.BytesIO(pdf_bytes)
        
        # Use PyPDF2 to extract text
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        if len(pdf_reader.pages) == 0:
            gr.Warning(f"PDF appears to be empty: {url}")
            return None
            
        extracted_text = []
        page_count = min(len(pdf_reader.pages), 100)  # Limit to first 100 pages
        
        for page_num in range(page_count):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text.strip():
                extracted_text.append(f"--- Page {page_num + 1} ---\n{text.strip()}")
        
        if not extracted_text:
            gr.Warning(f"Could not extract readable text from PDF: {url}")
            return None
            
        full_text = "\n\n".join(extracted_text)
        gr.Info(f"Successfully extracted {len(full_text)} characters from PDF ({page_count} pages)")
        
        return full_text, "application/pdf"
        
    except Exception as e:
        error_msg = f"Error processing PDF from {url}: {e}"
        print(f"[WEB_KB ERROR] {error_msg}")
        print(f"[WEB_KB ERROR] PDF processing traceback: {traceback.format_exc()}")
        gr.Error(error_msg)
        return None


async def _process_html_content(html_text: str, url: str) -> Optional[Tuple[str, str]]:
    """Extract and clean text from HTML content."""
    try:
        soup = BeautifulSoup(html_text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", 
                           "form", "button", "iframe", "img", "svg", "noscript", 
                           "meta", "link", "title"]):
            element.extract()
        
        # Try to find main content areas first
        main_content = None
        for selector in ['main', 'article', '.content', '.main-content', '#content', '#main']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if main_content:
            clean_text = main_content.get_text(separator="\n", strip=True)
        else:
            # Fall back to body content
            body = soup.find('body')
            if body:
                clean_text = body.get_text(separator="\n", strip=True)
            else:
                clean_text = soup.get_text(separator="\n", strip=True)
        
        # Clean up the text
        lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
        clean_text = '\n'.join(lines)
        
        if len(clean_text) < 100:  # Very short content
            gr.Warning(f"Very little content extracted from {url} ({len(clean_text)} chars)")
            return None
            
        gr.Info(f"Successfully extracted {len(clean_text)} characters from HTML")
        return clean_text, "text/html"
        
    except Exception as e:
        error_msg = f"Error processing HTML from {url}: {e}"
        print(f"[WEB_KB ERROR] {error_msg}")
        print(f"[WEB_KB ERROR] HTML processing traceback: {traceback.format_exc()}")
        gr.Error(error_msg)
        return None
    except Exception as e:
        gr.Error(f"An unexpected error occurred while fetching {url}: {e}")
        return None


# --- Enhanced main function to create knowledge base from URLs ---
async def create_web_knowledge_base(urls: str, embedding_model) -> Tuple[Optional[FAISS], str]:
    """
    Fetches content from a newline-separated string of URLs,
    splits them, creates embeddings, and builds a FAISS vector store.
    Now supports HTML pages, PDFs, text files, and more!
    """
    if not urls:
        gr.Warning("No URLs provided. Please enter links to create a knowledge base.")
        return None, "Please enter URLs first."

    url_list = [url.strip() for url in urls.split('\n') if url.strip()]
    
    if not url_list:
        gr.Warning("No valid URLs found after parsing input.")
        return None, "No valid URLs found."
    
    # Validate and clean URLs
    valid_urls, url_issues = _validate_and_clean_urls(url_list)
    
    if url_issues:
        issue_text = "\n".join([f"⚠️ {issue}: {msg}" for issue, msg in url_issues.items()])
        gr.Warning(f"URL validation issues found:\n{issue_text}")
    
    if not valid_urls:
        return None, "No valid URLs found after validation. Please check your URLs and try again."
    
    url_list = valid_urls  # Use validated URLs

    # --- Configuration for URL Limit ---
    MAX_URLS = 15 # Increased limit since we support more content types
    if len(url_list) > MAX_URLS:
        gr.Warning(f"Too many URLs provided. Processing first {MAX_URLS} only.")
        url_list = url_list[:MAX_URLS]
    # --- End URL Limit Config ---

    all_docs = []
    processed_count = 0
    failed_count = 0
    content_types_processed = set()

    gr.Info(f"Starting to process {len(url_list)} URL(s)... This may take a few minutes for PDFs.")
    
    # Process URLs with progress tracking
    for i, url in enumerate(url_list, 1):
        gr.Info(f"Processing {i}/{len(url_list)}: {url}")
        
        result = await _fetch_and_process_content(url)
        
        if result:
            cleaned_text, content_type = result
            content_types_processed.add(content_type)
            
            # Create document with enhanced metadata
            metadata = {
                'source': url,
                'content_type': content_type,
                'length': len(cleaned_text),
                'processed_at': str(asyncio.get_event_loop().time())
            }
            
            all_docs.append(Document(page_content=cleaned_text, metadata=metadata))
            processed_count += 1
            gr.Info(f"✅ Successfully processed {url} ({content_type})")
        else:
            failed_count += 1
            gr.Warning(f"❌ Failed to process: {url}")
    
    if not all_docs:
        status_message = f"Failed to fetch content from any of the provided URLs. {failed_count} failed."
        gr.Error(status_message)
        return None, status_message

    gr.Info(f"Successfully processed {processed_count} URL(s). Content types: {', '.join(content_types_processed)}")
    gr.Info("Splitting content into chunks for optimal search...")
    
    # Enhanced text splitting with different strategies based on content type
    chunks_per_type = {}
    all_split_docs = []
    
    for doc in all_docs:
        content_type = doc.metadata.get('content_type', 'unknown')
        
        # Choose chunk size based on content type
        if content_type == 'application/pdf':
            # PDFs often have more structured content, can use larger chunks
            chunk_size = 1500
            chunk_overlap = 300
        elif content_type == 'text/html':
            # HTML content, standard chunking
            chunk_size = 1000
            chunk_overlap = 200
        else:
            # Plain text and other formats
            chunk_size = 800
            chunk_overlap = 150
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        split_docs = text_splitter.split_documents([doc])
        all_split_docs.extend(split_docs)
        
        # Track chunks per content type
        chunks_per_type[content_type] = chunks_per_type.get(content_type, 0) + len(split_docs)
    
    if not all_split_docs:
        status_message = "No text chunks generated after splitting. Content might be too small or empty."
        gr.Error(status_message)
        return None, status_message

    chunk_summary = ", ".join([f"{count} {ctype} chunks" for ctype, count in chunks_per_type.items()])
    gr.Info(f"Creating enhanced knowledge base from {len(all_split_docs)} total chunks ({chunk_summary})...")
    
    try:
        web_knowledge_base = FAISS.from_documents(all_split_docs, embedding_model)
        status_message = (f"🎉 Enhanced web knowledge base created successfully!\n"
                         f"📊 Processed: {processed_count} URLs ({failed_count} failed)\n"
                         f"📄 Content types: {', '.join(content_types_processed)}\n"
                         f"🔍 Search chunks: {len(all_split_docs)}")
        gr.Info(status_message)
        return web_knowledge_base, status_message
    except Exception as e:
        status_message = f"Error creating web knowledge base: {e}"
        print(f"[WEB_KB ERROR] FAISS creation failed: {e}")
        print(f"[WEB_KB ERROR] FAISS creation traceback: {traceback.format_exc()}")
        print(f"[WEB_KB ERROR] Number of documents: {len(all_split_docs)}")
        print(f"[WEB_KB ERROR] Document sample: {all_split_docs[0] if all_split_docs else 'No documents'}")
        gr.Error(status_message)
        return None, status_message


# --- Enhanced function to answer questions from web knowledge base ---
async def answer_web_kb_conversational(llm, knowledge_base, question, chat_history):
    """
    Answers a question using the enhanced web knowledge base with better context awareness.
    """
    if knowledge_base is None:
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": "Please create a web knowledge base first by entering URLs above."})
        return "", chat_history

    if not question:
        return "", chat_history

    gr.Info("🔍 Searching web knowledge base...")
    try:
        # Enhanced retriever with better search parameters
        retriever = knowledge_base.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 8,  # Retrieve more relevant chunks
            }
        )
        
        # Retrieve relevant documents
        relevant_docs = await retriever.ainvoke(question)
        
        if not relevant_docs:
            answer = "I couldn't find relevant information in the knowledge base to answer your question. Please try rephrasing or ensure the content covers your topic."
        else:
            # Prepare context from retrieved documents
            context_parts = []
            sources = set()
            content_types = set()
            
            for i, doc in enumerate(relevant_docs[:6]):  # Use top 6 most relevant
                context_parts.append(f"Document {i+1}:\n{doc.page_content}\n")
                
                if 'source' in doc.metadata:
                    sources.add(doc.metadata['source'])
                if 'content_type' in doc.metadata:
                    content_types.add(doc.metadata['content_type'])
            
            context = "\n".join(context_parts)
            
            # Build conversation history context
            history_context = ""
            if len(chat_history) > 0:
                recent_history = chat_history[-6:]  # Last 3 exchanges
                history_parts = []
                for msg in recent_history:
                    role = "Human" if msg["role"] == "user" else "Assistant"
                    history_parts.append(f"{role}: {msg['content']}")
                history_context = "\n".join(history_parts)
            
            # Create comprehensive prompt
            prompt = f"""You are an AI assistant helping to answer questions based on web content and documents. 

Context from knowledge base:
{context}

Previous conversation:
{history_context}

Current question: {question}

Please provide a comprehensive answer based on the context provided above. If the context doesn't contain enough information to fully answer the question, say so clearly. Always be accurate and cite information from the context when possible."""

            # Get response from LLM
            response = await llm.ainvoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Add source information to the answer
            if sources:
                source_info = "\n\n📚 **Sources consulted:**\n"
                for i, source in enumerate(list(sources)[:3], 1):  # Show top 3 sources
                    source_info += f"{i}. {source}\n"
                
                if len(sources) > 3:
                    source_info += f"... and {len(sources) - 3} more sources"
                
                # Add content type info if available
                if content_types:
                    type_info = ", ".join(content_types)
                    source_info += f"\n📄 **Content types:** {type_info}"
                
                answer += source_info
        
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})
        
        return "", chat_history
        
    except Exception as e:
        error_msg = f"Error answering web KB question: {e}"
        print(f"[WEB_KB ERROR] Chat error: {e}")
        print(f"[WEB_KB ERROR] Chat traceback: {traceback.format_exc()}")
        print(f"[WEB_KB ERROR] Question: {question}")
        print(f"[WEB_KB ERROR] KB status: {knowledge_base is not None}")
        gr.Error(error_msg)
        error_message = f"I encountered an error while searching the knowledge base: {str(e)}. Please try again or recreate the knowledge base."
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": error_message})
        return "", chat_history