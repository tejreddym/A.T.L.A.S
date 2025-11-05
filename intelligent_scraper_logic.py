# In intelligent_scraper_logic.py, find and replace the entire intelligent_scraper_dispatcher function

import os
import asyncio
import pandas as pd
import json
import re
import gradio as gr # Needed for gr.Info/Warning/Error

from typing import List, Dict, Tuple, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults # For search tool

# Import the browser-based scraper
from auth_scraper_logic import scrape_with_browser # Assuming scrape_with_browser is fully implemented and working

# --- Global LLM and Tool Instances for Intelligent Scraper ---
# Load environment variables (for Groq API key)
from dotenv import load_dotenv
load_dotenv()

try:
    ORCHESTRATION_LLM_KEY = os.getenv("GROQ_API_KEY_T2_A1")
    if not ORCHESTRATION_LLM_KEY:
        ORCHESTRATION_LLM_KEY = os.getenv("GROQ_API_KEY")
    if not ORCHESTRATION_LLM_KEY:
        raise ValueError("GROQ_API_KEY or GROQ_API_KEY_T2_A1 not found in .env for orchestration LLM.")
    orchestration_llm = ChatGroq(temperature=0.1, model_name="llama-3.1-8b-instant", api_key=ORCHESTRATION_LLM_KEY)
except ValueError as e:
    print(f"Error loading Groq API key for orchestration LLM: {e}")
    orchestration_llm = None

# Search Tool (Tavily)
tavily_tool = TavilySearchResults(max_results=5) # Max 5 search results to avoid too many scrapes


# --- Export Helper Functions (Moved from app.py) ---
GENERATED_FILES_DIR = "generated_scrapes"
os.makedirs(GENERATED_FILES_DIR, exist_ok=True)

def export_to_excel(data: List[Dict], filename: str) -> str:
    if not data: return ""
    try:
        df = pd.DataFrame(data)
        filepath = os.path.join(GENERATED_FILES_DIR, f"{filename}.xlsx")
        df.to_excel(filepath, index=False)
        return filepath
    except Exception as e:
        gr.Error(f"Error exporting to Excel: {e}")
        return ""

def export_to_csv(data: List[Dict], filename: str) -> str:
    if not data: return ""
    try:
        df = pd.DataFrame(data)
        filepath = os.path.join(GENERATED_FILES_DIR, f"{filename}.csv")
        df.to_csv(filepath, index=False)
        return filepath
    except Exception as e:
        gr.Error(f"Error exporting to CSV: {e}")
        return ""

def export_to_json(data: List[Dict], filename: str) -> str:
    if not data: return ""
    try:
        filepath = os.path.join(GENERATED_FILES_DIR, f"{filename}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return filepath
    except Exception as e:
        gr.Error(f"Error exporting to JSON: {e}")
        return ""

def export_to_txt(data: List[Dict], filename: str) -> str:
    """Exports data to a plain text file (simple representation)."""
    if not data: return ""
    try:
        filepath = os.path.join(GENERATED_FILES_DIR, f"{filename}.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return filepath
    except Exception as e:
        gr.Error(f"Error exporting to Text: {e}")
        return ""


# --- Intelligent Scraper Dispatcher ---

async def intelligent_scraper_dispatcher(
    main_topic_prompt: str,
    export_format: str
) -> Tuple[str, Optional[str]]:
    
    gr.Info("Starting intelligent scraping process...")
    consolidated_extracted_data = []
    status_message = ""
    output_filepath = None

    if not orchestration_llm:
        gr.Error("Orchestration LLM is not initialized (API key missing or invalid). Cannot perform intelligent scraping.")
        return "Error: Orchestration LLM not ready.", None
    
    if not main_topic_prompt:
        gr.Warning("Please provide a main topic/prompt for the intelligent scraper.")
        return "Please provide a prompt.", None

    try:
        # Step 1: Use LLM to formulate search queries and extraction goals
        gr.Info("AI is formulating search queries and extraction goals...")
        query_and_extraction_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(
                "You are a web research assistant. Given a main request, formulate precise search queries "
                "to find relevant web pages, and identify the specific data to be extracted from each page. "
                "Return a JSON array of objects, where each object has a 'search_query' and a 'data_extraction_prompt'. "
                "For example: [{'search_query': 'latest AI news in India', 'data_extraction_prompt': 'Extract headlines and first paragraph of each news article.'}]. "
                "Focus on the extraction prompt being specific for the data you expect to find on the search result pages."
                "Limit to 1-2 search queries unless explicitly asked for more. Respond ONLY with the JSON."
            ),
            HumanMessage(content=f"Main request: {main_topic_prompt}\n\nJSON Output:")
        ])
        
        llm_response_orchestration = await orchestration_llm.ainvoke(
            query_and_extraction_prompt_template.format_messages(main_topic_prompt=main_topic_prompt)
        )
        raw_orchestration_output = llm_response_orchestration.content.strip()

        # Robust JSON parsing for orchestration LLM output
        orchestration_json_match = re.search(r'\[.*\]|\{.*\}', raw_orchestration_output, re.DOTALL)
        orchestration_plan = []
        if orchestration_json_match:
            try:
                potential_json_str = orchestration_json_match.group(0)
                if potential_json_str.startswith("```json"):
                    potential_json_str = potential_json_str.replace("```json\n", "", 1)
                if potential_json_str.endswith("```"):
                    potential_json_str = potential_json_str.rsplit("```", 1)[0]
                
                orchestration_plan = json.loads(potential_json_str)
                if not isinstance(orchestration_plan, list):
                    orchestration_plan = [orchestration_plan] if isinstance(orchestration_plan, dict) else []
            except json.JSONDecodeError:
                gr.Warning(f"Orchestration LLM returned malformed JSON plan. Raw: {raw_orchestration_output[:200]}...")
                return "Error: Could not parse AI's search plan. Please refine your prompt.", None
        
        if not orchestration_plan:
            gr.Warning("AI could not formulate a search plan from your prompt.")
            return "AI could not formulate a search plan. Try a more direct prompt.", None
        
        # Step 2: Execute search and scrape each relevant URL
        gr.Info(f"AI formulated {len(orchestration_plan)} search queries. Starting web search and scraping...")
        
        for item in orchestration_plan:
            search_query = item.get("search_query")
            data_extraction_prompt_for_page = item.get("data_extraction_prompt")
            
            if not search_query or not data_extraction_prompt_for_page:
                gr.Warning(f"Skipping malformed plan item: {item}")
                continue

            gr.Info(f"Searching for: '{search_query}'...")
            search_results_list = await asyncio.to_thread(tavily_tool.run, search_query) # Renamed variable

            # --- DEBUGGING ADDITION ---
            print(f"--- DEBUG: Tavily Search Results Type: {type(search_results_list)} ---")
            print(f"--- DEBUG: Tavily Search Results (first 500 chars):\n{str(search_results_list)[:500]} ---")
            # --- END DEBUGGING ADDITION ---

            # --- FIX: Extract URLs directly from the list of dicts ---
            urls_to_scrape = []
            if isinstance(search_results_list, list):
                for result_dict in search_results_list:
                    if isinstance(result_dict, dict) and 'url' in result_dict:
                        urls_to_scrape.append(result_dict['url'])
            # --- END FIX ---
            
            if not urls_to_scrape:
                gr.Warning(f"No URLs found for search query: '{search_query}'.")
                continue
            
            gr.Info(f"Found {len(urls_to_scrape)} URLs. Scraping pages...")
            
            for url in urls_to_scrape:
                gr.Info(f"Scraping {url}...")
                page_extracted_data, page_status = await scrape_with_browser(
                    target_url=url,
                    login_required=False, # We assume public web for now for automated search
                    data_extraction_prompt=data_extraction_prompt_for_page
                )
                if page_extracted_data:
                    consolidated_extracted_data.extend(page_extracted_data)
                    gr.Info(f"Extracted {len(page_extracted_data)} items from {url}.")
                else:
                    gr.Warning(f"No data extracted from {url}. Status: {page_status}")
        
        if not consolidated_extracted_data:
            status_message = "No data was extracted from any of the searched pages."
            return status_message, None

        # Step 3: Export the consolidated data
        base_filename = f"intelligent_scrape_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        if export_format == "Excel":
            output_filepath = export_to_excel(consolidated_extracted_data, base_filename)
        elif export_format == "CSV":
            output_filepath = export_to_csv(consolidated_extracted_data, base_filename)
        elif export_format == "JSON":
            output_filepath = export_to_json(consolidated_extracted_data, base_filename)
        elif export_format == "Plain Text":
            output_filepath = export_to_txt(consolidated_extracted_data, base_filename)
        
        if output_filepath:
            status_message = f"Intelligent scraping complete. Data exported to {os.path.basename(output_filepath)}."
            gr.Info(status_message)
        else:
            status_message = "Intelligent scraping completed, but failed to export data."

    except Exception as e:
        status_message = f"An unhandled error occurred during intelligent scraping: {e}"
        gr.Error(status_message)
        print(f"Intelligent Scraper Error: {e}") # Print full error for debugging

    return status_message, output_filepath