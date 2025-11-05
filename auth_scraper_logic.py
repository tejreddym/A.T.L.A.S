# In auth_scraper_logic.py, replace the entire scrape_with_browser function

import asyncio
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import gradio as gr
from typing import List, Dict, Tuple, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json
import re
import httpx

# --- LLM Setup for Extraction ---
try:
    EXTRACTION_GROQ_API_KEY = os.getenv("GROQ_API_KEY_JUDGE") # Use a powerful model for extraction
    if not EXTRACTION_GROQ_API_KEY:
        EXTRACTION_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not EXTRACTION_GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY or GROQ_API_KEY_JUDGE not found in .env")
    extraction_llm = ChatGroq(temperature=0.01, model_name="llama3-70b-8192", api_key=EXTRACTION_GROQ_API_KEY)
except ValueError as e:
    print(f"Error loading Groq API key for extraction LLM: {e}")
    extraction_llm = None

# --- Helper Functions for Selenium Setup ---
def get_chrome_options():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    return options

def initialize_driver():
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=get_chrome_options())
    driver.implicitly_wait(10)
    return driver

# --- Core Universal Browser-Based Scraping Logic ---

async def scrape_with_browser(
    target_url: str,
    login_required: bool = False,
    username: Optional[str] = None,
    password: Optional[str] = None,
    username_field_id: Optional[str] = None,
    password_field_id: Optional[str] = None,
    login_button_id: Optional[str] = None,
    data_extraction_prompt: str = ""
) -> Tuple[List[Dict], str]:
    
    driver = None
    extracted_data = []
    status_message = "Scraping process initiated."
    
    if extraction_llm is None:
        status_message = "Error: LLM for extraction is not initialized (API key missing or invalid)."
        gr.Error(status_message)
        return extracted_data, status_message

    try:
        # Check Content-Type header before launching browser for non-HTML
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                head_response = await client.head(target_url, follow_redirects=True)
                content_type = head_response.headers.get('content-type', '').lower()
                
                if 'application/pdf' in content_type:
                    status_message = f"Skipping PDF document (unsupported): {target_url}"
                    gr.Warning(status_message)
                    return [], status_message
                elif not 'text/html' in content_type and not 'application/xhtml+xml' in content_type:
                    status_message = f"Skipping non-HTML content type ({content_type}): {target_url}"
                    gr.Warning(status_message)
                    return [], status_message
        except Exception as e:
            gr.Warning(f"Failed to get content-type header for {target_url}: {e}. Proceeding with browser.")

        driver = await asyncio.to_thread(initialize_driver)

        if login_required:
            gr.Info(f"Attempting to log in to: {target_url}")
            await asyncio.to_thread(driver.get, target_url)
            
            username_input = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.ID, username_field_id))
            )
            await asyncio.to_thread(username_input.send_keys, username)
            
            password_input = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.ID, password_field_id))
            )
            await asyncio.to_thread(password_input.send_keys, password)
            
            login_button = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.ID, login_button_id))
            )
            await asyncio.to_thread(login_button.click)
            gr.Info("Login button clicked. Waiting for redirection...")

            try:
                WebDriverWait(driver, 20).until(EC.url_changes(target_url))
            except Exception:
                if "login" in driver.current_url.lower() or "error" in driver.page_source.lower():
                    status_message = "Login failed: Incorrect credentials or unexpected page structure/redirection."
                    gr.Error(status_message)
                    return extracted_data, status_message

            gr.Info(f"Successfully logged in and navigated to: {driver.current_url}")
            
        else: # Non-authenticated scraping
            gr.Info(f"Navigating to (non-authenticated): {target_url}")
            await asyncio.to_thread(driver.get, target_url)

        await asyncio.to_thread(WebDriverWait(driver, 20).until, EC.presence_of_element_located((By.TAG_NAME, "body")))
        await asyncio.sleep(5) # Give more time for JavaScript to render
        
        page_source = await asyncio.to_thread(lambda: driver.page_source)

        print(f"--- DEBUG: Current URL: {driver.current_url} ---")
        print(f"--- DEBUG: Length of page_source: {len(page_source)} ---")
        
        if len(page_source) < 500:
            print(f"--- DEBUG: Page source is very small/potentially empty. First 500 chars:\n{page_source[:500]} ---")
            status_message = "Warning: Page source is very small or empty. Website might be blocking or content not loaded."
            gr.Warning(status_message)
            return extracted_data, status_message 
        
        gr.Info("Pre-processing HTML and attempting AI analysis...")
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # Programmatic Title Extraction (remains)
        if "title of the page" in data_extraction_prompt.lower():
            title_tag = soup.find('title')
            if title_tag and title_tag.string:
                extracted_data.append({"page_title": title_tag.string.strip()})
                status_message = f"Programmatically extracted page title from {target_url}."
                gr.Info(status_message)
                return extracted_data, status_message # Return early if successful

        # --- NEW: Hybrid Extraction - Targeted HTML + LLM per item ---
        # Define common HTML element types and class/id patterns for *individual item* containers
        # These are common patterns for news articles, product listings, search results, etc.
        item_selectors = [
            # News/Blog articles: semantic tags + common class/id patterns
            {'tag': ['article', 'section'], 'class_id_re': r'article|post|story|news-item|blog-post', 'attrs': {'role': 'article'}},
            # Product listings/Search results
            {'tag': ['div', 'li'], 'class_id_re': r'product|item|listing|result|card|media-item|s-result-item', 'attrs': {'data-asin': True}},
            # Generic repeating content that looks like a block
            {'tag': ['div', 'li'], 'class_id_re': r'item|card|entry|row|column', 'min_text_len': 50}, # Min text length to avoid tiny divs
            # Links that represent a main item (e.g., news headlines that are direct links to articles)
            {'tag': 'a', 'class_id_re': r'headline|title|link', 'has_href': True}, 
        ]
        
        identified_item_containers = []

        for selector in item_selectors:
            tag = selector.get('tag')
            class_id_re = selector.get('class_id_re')
            min_text_len = selector.get('min_text_len', 0)
            has_href = selector.get('has_href', False)
            attrs = selector.get('attrs', {})

            if class_id_re:
                attrs_re = {**attrs, **{'class': re.compile(class_id_re, re.I), 'id': re.compile(class_id_re, re.I)}}
            else:
                attrs_re = attrs
            
            found_elements = soup.find_all(tag, attrs=attrs_re)
            
            for el in found_elements:
                # Basic sanity checks to avoid noisy or irrelevant elements
                if el.get_text(strip=True) and \
                   (len(el.get_text(strip=True)) > min_text_len or (has_href and el.get('href'))):
                    # Exclude common noisy elements that might match generic 'div' or 'item'
                    if not any(cls in el.get('class', []) for cls in ['footer', 'header', 'nav', 'sidebar', 'advertisement', 'ad-unit', 'widget', 'skip-link']):
                        if el not in identified_item_containers: # Avoid duplicates
                            identified_item_containers.append(el)

        # Fallback if no specific item containers are found: use the main content area as a single large item
        if not identified_item_containers:
            gr.Warning("No specific repeating item containers found. Attempting extraction from detected main content area as a single item.")
            
            main_content_element = soup.find('article') or soup.find('main') or soup.find('div', role='main')
            if not main_content_element:
                main_content_element = soup.find('div', id=re.compile(r'main-content|content-body|article-body', re.I)) or \
                                       soup.find('div', class_=re.compile(r'main-content|article-content|body-content', re.I)) or \
                                       soup.find('section', id=re.compile(r'main-section|article-section', re.I))
            if not main_content_element:
                main_content_element = soup.find('body')

            if main_content_element:
                identified_item_containers = [main_content_element] # Treat main content as a single item
            else:
                gr.Warning("No main content element or body tag found. Using entire page source as a single item for LLM extraction.")
                identified_item_containers = [BeautifulSoup(page_source, 'html.parser')] # Re-parse to treat as a soup object
        
        gr.Info(f"Identified {len(identified_item_containers)} potential item containers for detailed LLM extraction.")
        
        # Define the extraction prompt template for individual content chunks
        extraction_template = ChatPromptTemplate.from_messages([
            SystemMessage(
                "You are a highly efficient and strict data extractor. "
                "Your task is to extract very specific information from the provided text content snippet "
                "based on the user's explicit extraction prompt. "
                "**Your output MUST be ONLY a VALID JSON array of objects, AND NOTHING ELSE.** "
                "Each object should contain the requested data points as key-value pairs. "
                "If a data point cannot be found, omit its key or set its value to null. "
                "If absolutely no data can be extracted based on the prompt, you MUST return `[]` (an empty JSON array). "
                "**DO NOT include any conversational text, explanations, or markdown fences outside the JSON array.** "
                "Begin your response directly with `[` or `{`."
            ),
            HumanMessage(content="Text Content Snippet:\n```text\n{text_content_snippet}\n```\n\nExtraction Prompt: {extraction_prompt}\n\nJSON Output:")
        ])

        total_extracted_items = 0
        for i, container_soup_obj in enumerate(identified_item_containers):
            # Extract clean text from the current container's HTML.
            # Make a copy to extract text without modifying soup's original structure for other containers.
            snippet_soup_copy = BeautifulSoup(str(container_soup_obj), 'html.parser') 
            
            # Remove common noisy tags from this individual snippet
            for script_or_style_or_noise in snippet_soup_copy(["script", "style", "nav", "footer", "header", "aside", "form", "button", "iframe", "img", "svg", "noscript"]):
                script_or_style_or_noise.extract()
            
            clean_text_content_for_llm = snippet_soup_copy.get_text(separator="\n", strip=True)

            if not clean_text_content_for_llm.strip():
                print(f"--- DEBUG: Skipping empty text content from container {i+1}. ---")
                continue

            # Limit snippet size for LLM if it's still too large
            if len(clean_text_content_for_llm) > 15000: # Max ~15k chars for a manageable snippet
                gr.Warning(f"Snippet {i+1} is very large ({len(clean_text_content_for_llm)} chars). Truncating for LLM.")
                clean_text_content_for_llm = clean_text_content_for_llm[:15000]

            print(f"--- DEBUG: Sending snippet {i+1} (Length: {len(clean_text_content_for_llm)}) to LLM. ---")
            print(f"--- DEBUG: Snippet Content (first 500 chars):\n{clean_text_content_for_llm[:500]}... ---")

            messages = extraction_template.format_messages(
                text_content_snippet=clean_text_content_for_llm,
                extraction_prompt=data_extraction_prompt
            )
            
            llm_response = await extraction_llm.ainvoke(messages)
            raw_llm_output = llm_response.content
            
            print(f"--- DEBUG: Raw LLM Output for snippet {i+1} (Length: {len(raw_llm_output)}):\n{raw_llm_output[:1000]}... ---")

            json_match = re.search(r'\[.*\]|\{.*\}', raw_llm_output, re.DOTALL)
            if json_match:
                potential_json_str = json_match.group(0)
                if potential_json_str.startswith("```json"):
                    potential_json_str = potential_json_str.replace("```json\n", "", 1)
                if potential_json_str.endswith("```"):
                    potential_json_str = potential_json_str.rsplit("```", 1)[0]
                raw_llm_output = potential_json_str.strip()
            else:
                raw_llm_output = raw_llm_output.strip()
            
            try:
                extracted_from_snippet = json.loads(raw_llm_output)
                if not isinstance(extracted_from_snippet, list):
                    extracted_from_snippet = [extracted_from_snippet] if isinstance(extracted_from_snippet, dict) else []
                
                if extracted_from_snippet:
                    extracted_data.extend(extracted_from_snippet)
                    total_extracted_items += len(extracted_from_snippet)
                    print(f"--- DEBUG: Extracted {len(extracted_from_snippet)} items from snippet {i+1}. ---")

            except json.JSONDecodeError as e:
                gr.Warning(f"AI extraction failed or returned malformed JSON from snippet {i+1}: {e}. Output: {raw_llm_output[:200]}...")
                extracted_data.append({"raw_llm_output_error": raw_llm_output, "original_prompt": data_extraction_prompt, "json_error": str(e), "source_snippet": clean_text_content_for_llm[:500]})
            except Exception as e:
                gr.Warning(f"An error processing snippet {i+1} output: {e}. Output: {raw_llm_output[:200]}...")
                extracted_data.append({"raw_llm_output_error": raw_llm_output, "original_prompt": data_extraction_prompt, "error": str(e), "source_snippet": clean_text_content_for_llm[:500]})

        if not extracted_data:
            status_message = "No specific data found by AI based on prompt after checking all content areas."
        else:
            status_message = f"Successfully extracted {total_extracted_items} items by AI from {target_url}."


    except Exception as e:
        status_message = f"An error occurred during browser scraping: {e}"
        gr.Error(status_message)
        print(f"Browser Scrape Error: {e}")
        extracted_data = []
    finally:
        if driver:
            gr.Info("Closing browser...")
            await asyncio.to_thread(driver.quit)
            gr.Info("Browser closed.")
    
    return extracted_data, status_message

# --- Example Usage for direct testing (Optional) ---
if __name__ == "__main__":
    TEST_TARGET_URL_NON_AUTH = "https://www.theguardian.com/uk" # A news site that often has 'article' tags
    
    async def run_test_scrape():
        print("\n--- Running Non-Authenticated Scrape Test ---")
        data, status = await scrape_with_browser(
            target_url=TEST_TARGET_URL_NON_AUTH,
            login_required=False,
            data_extraction_prompt="Extract the titles and URLs of the main news articles on the homepage. Keys: 'title', 'url'."
        )
        print(f"Status: {status}")
        print("Extracted Data (first 2 entries):")
        for entry in data[:2]:
            print(f"  Entry: {entry}")
        print("-" * 30)

    asyncio.run(run_test_scrape())