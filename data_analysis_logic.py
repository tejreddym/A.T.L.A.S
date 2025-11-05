# data_analysis_logic.py (CORRECTED VERSION)

import os
import asyncio
import pandas as pd
import gradio as gr
from typing import List, Dict, Tuple, Optional, Any # Ensure Any and Optional are imported

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

# --- Load API Key ---
from dotenv import load_dotenv
load_dotenv() # Ensure .env is loaded here too

# --- LLM Setup for Data Analysis ---
try:
    # Use a powerful Groq LLM for data analysis
    DATA_ANALYSIS_LLM_KEY = os.getenv("GROQ_API_KEY_JUDGE") # Use a powerful key
    if not DATA_ANALYSIS_LLM_KEY:
        DATA_ANALYSIS_LLM_KEY = os.getenv("GROQ_API_KEY") # Fallback
    if not DATA_ANALYSIS_LLM_KEY:
        raise ValueError("GROQ_API_KEY or GROQ_API_KEY_JUDGE not found in .env for data analysis.")
    data_analysis_llm = ChatGroq(temperature=0.1, model_name="llama3-70b-8192", api_key=DATA_ANALYSIS_LLM_KEY)
except ValueError as e:
    print(f"Error loading Groq API key for data analysis LLM: {e}")
    data_analysis_llm = None


# --- Core Data Analysis Logic ---

# Corrected return type: now returns a Gradio component for interactivity
async def load_and_preview_data(file_obj: gr.File) -> Tuple[Optional[pd.DataFrame], str, str, gr.Textbox]: 
    """
    Loads data from an uploaded CSV/Excel file, returns a DataFrame, its preview, status,
    and a Gradio Textbox component indicating if the query input should be interactive.
    """
    if file_obj is None:
        return None, "", "Please upload a CSV or Excel file.", gr.Textbox(interactive=False) # Return Textbox component

    filepath = file_obj.name
    filename = os.path.basename(filepath)
    df = None
    status_message = ""
    preview_markdown = ""
    
    try:
        gr.Info(f"Loading data from {filename}...")
        if filepath.endswith('.csv'):
            df = await asyncio.to_thread(pd.read_csv, filepath)
        elif filepath.endswith(('.xls', '.xlsx')):
            df = await asyncio.to_thread(pd.read_excel, filepath)
        else:
            status_message = "Unsupported file type. Please upload a CSV or Excel file (.csv, .xls, .xlsx)."
            gr.Error(status_message)
            return None, "", status_message, gr.Textbox(interactive=False) # Return Textbox component
        
        if df.empty:
            status_message = "Uploaded file is empty or could not be parsed into a DataFrame."
            gr.Warning(status_message)
            return None, "", status_message, gr.Textbox(interactive=False) # Return Textbox component

        status_message = f"Successfully loaded {len(df)} rows and {len(df.columns)} columns from {filename}."
        gr.Info(status_message)
        
        preview_markdown = "### Data Preview (First 5 Rows)\n"
        preview_markdown += df.head().to_markdown(index=False)
        preview_markdown += "\n\n### Data Info\n"
        preview_markdown += f"- Rows: {len(df)}\n"
        preview_markdown += f"- Columns: {len(df.columns)}\n"
        preview_markdown += f"- Column Names: {', '.join(df.columns)}"

        return df, preview_markdown, status_message, gr.Textbox(interactive=True) # Return Textbox component
    
    except pd.errors.EmptyDataError:
        status_message = f"Error: CSV file {filename} is empty."
        gr.Error(status_message)
        return None, "", status_message, gr.Textbox(interactive=False)
    except FileNotFoundError:
        status_message = f"Error: File not found at {filepath}."
        gr.Error(status_message)
        return None, "", status_message, gr.Textbox(interactive=False)
    except Exception as e:
        status_message = f"Error loading file {filename}: {e}"
        gr.Error(status_message)
        print(f"Data Load Error: {e}")
        return None, "", status_message, gr.Textbox(interactive=False)


async def analyze_data_with_llm(dataframe: pd.DataFrame, user_query: str) -> Tuple[str, str]:
    """
    Analyzes the DataFrame based on user's query using an LLM.
    Returns: analysis_output_markdown, status_message
    """
    analysis_output = ""
    status_message = "Analyzing data with AI..."

    if data_analysis_llm is None:
        status_message = "Error: Data Analysis LLM not initialized (API key missing or invalid)."
        gr.Error(status_message)
        return "", status_message
    
    if dataframe is None or dataframe.empty:
        status_message = "No data loaded for analysis. Please upload a file first."
        gr.Warning(status_message)
        return "", status_message

    if not user_query:
        status_message = "Please provide a question or instruction for data analysis."
        gr.Warning(status_message)
        return "", status_message

    # Provide DataFrame schema and first few rows as context to the LLM
    df_info = dataframe.info(buf=None, verbose=False, show_counts=False) # Get concise info
    df_head_markdown = dataframe.head(5).to_markdown(index=False) # Get first 5 rows
    df_columns = ", ".join(dataframe.columns)
    
    context_for_llm = (
        f"You are a skilled data analyst. Analyze the provided tabular data (CSV/Excel) "
        f"based on the user's query. The data has {len(dataframe)} rows and columns: {df_columns}. "
        f"Here is a preview of the data:\n\n```csv\n{df_head_markdown}\n```\n\n"
        "Your response should directly answer the query by providing insights, trends, "
        "or specific values extracted from the data. If statistical analysis or aggregation "
        "is required, perform it conceptually or describe how to get it. "
        "You can also generate Python Pandas code snippets if the user explicitly asks for code or if it significantly "
        "helps illustrate the solution. Always explain your findings in natural language. "
        "If providing code, enclose it in markdown code blocks. "
        "Do NOT execute any code. Focus on insight generation."
    )

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=context_for_llm),
        HumanMessage(content=f"User's data analysis query: {user_query}")
    ])

    analysis_chain = prompt_template | data_analysis_llm | StrOutputParser()

    try:
        analysis_output = await analysis_chain.ainvoke({"user_query": user_query, "dataframe": dataframe, "dataframe_head_markdown": df_head_markdown})
        status_message = "Data analysis complete!"
        gr.Info(status_message)
    except Exception as e:
        status_message = f"Error performing data analysis: {e}"
        gr.Error(status_message)
        print(f"Data Analysis Error: {e}")
        analysis_output = f"Failed to perform analysis due to an internal error: {e}"

    return analysis_output, status_message

# --- Example Usage for direct testing (Optional) ---
if __name__ == "__main__":
    async def run_test_data_analysis():
        # Create a dummy CSV file for testing
        dummy_data = {
            'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Mouse'],
            'Region': ['North', 'South', 'East', 'North', 'West'],
            'Sales': [1200, 50, 75, 300, 60],
            'Quantity': [1, 5, 2, 1, 3]
        }
        test_df = pd.DataFrame(dummy_data)
        test_csv_path = "test_sales_data.csv"
        test_df.to_csv(test_csv_path, index=False)
        
        # Mocking gr.File object
        class MockGradioFile:
            def __init__(self, name):
                self.name = name
        
        mock_file_obj = MockGradioFile(test_csv_path)

        print("\n--- Running Data Load and Preview Test ---")
        df_loaded, preview, status, interactive_bool = await load_and_preview_data(mock_file_obj)
        print(f"Load Status: {status}")
        print(f"Preview:\n{preview}")
        print(f"Interactive state: {interactive_bool}")
        print("-" * 30)

        if df_loaded is not None and not df_loaded.empty and interactive_bool:
            print("\n--- Running Data Analysis Test (Total Sales) ---")
            analysis, status_an = await analyze_data_with_llm(df_loaded, "What is the total sales amount?")
            print(f"Analysis Status: {status_an}")
            print(f"Analysis Output:\n{analysis}")
            print("-" * 30)

            print("\n--- Running Data Analysis Test (Sales by Region) ---")
            analysis_region, status_region = await analyze_data_with_llm(df_loaded, "Summarize sales by region.")
            print(f"Analysis Status: {status_region}")
            print(f"Analysis Output:\n{analysis_region}")
            print("-" * 30)
        
        # Clean up dummy file
        if os.path.exists(test_csv_path):
            os.remove(test_csv_path)

    asyncio.run(run_test_data_analysis())