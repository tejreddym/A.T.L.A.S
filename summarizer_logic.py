# In summarizer_logic.py
import requests
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# This function is fine as synchronous, as it uses a local pipeline directly
def summarize_with_local_model(url, summarizer_pipeline):
    # ... no changes to the inside of this function ...
    print(f"Fetching content for local summarizer: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    article_text = ' '.join([p.get_text() for p in paragraphs])
    if len(article_text) < 200:
        return "Could not find enough text on the page to summarize."
    print("Summarizing text with local model...")
    summary = summarizer_pipeline(article_text[:4096], max_length=150, min_length=50, do_sample=False)
    print("Local summary complete.")
    return summary[0]['summary_text']

async def summarize_with_groq(url): #<-- async
    # ... no changes to the inside of this function ...
    print(f"Fetching content for Groq summarizer: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    article_text = ' '.join([p.get_text() for p in paragraphs])
    if len(article_text) < 200:
        return "Could not find enough text on the page to summarize."
    print("Summarizing text with Groq API...")
    prompt = ChatPromptTemplate.from_template(
        "You are an expert summarizer. Provide a concise, clear summary of the following article text in about 150 words.\n\n"
        "Article Text:\n---\n{text}"
    )
    groq_llm = ChatGroq(temperature=0.2, model_name="llama-3.1-8b-instant")
    chain = prompt | groq_llm
    result = await chain.ainvoke({"text": article_text[:8000]}) #<-- await and ainvoke
    print("Groq summary complete.")
    return result.content