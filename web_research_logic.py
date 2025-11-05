# In web_research_logic.py

from __future__ import annotations

import gradio as gr
from datetime import datetime
import re
import numexpr
import os

from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from groq_utils import create_groq_chat
# Optional Google AI Studio integration. Wrap in try/except so the module can still be
# imported when the optional dependency is not installed in the environment.
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # For Google AI Studio API
    GOOGLE_GENAI_AVAILABLE = True
except Exception:
    ChatGoogleGenerativeAI = None
    GOOGLE_GENAI_AVAILABLE = False
    print("Warning: 'langchain_google_genai' package is not installed. Google AI features will be disabled. Install it with 'pip install langchain-google-genai' to enable.")
from langchain_core.prompts import ChatPromptTemplate # From langchain_core.prompts
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage # From langchain_core.messages
from langchain_core.output_parsers import StrOutputParser # For simple string output
from typing import List, Dict, Tuple, Any # For type hints


# --- Import our image generation tool function (remains) ---
# Ensure this file exists and contains generate_image_tool_func
try:
    from image_generation_logic import generate_image_tool_func
except ImportError:
    print("Warning: image_generation_logic.py or generate_image_tool_func not found. ImageGenerator tool will not work.")
    def generate_image_tool_func(*args, **kwargs): return "Image generation tool not available."


# --- Load API Key (Google AI) ---
from dotenv import load_dotenv
load_dotenv() # Ensure .env is loaded here too

# --- Tool Definitions ---

def run_safe_calculator(query: str) -> str:
    """Safely evaluates mathematical expressions."""
    # Ensure this LLM is properly configured (e.g., in .env GROQ_API_KEY)
    llm = create_groq_chat(model_name="llama-3.1-8b-instant", temperature=0)
    prompt = f"Convert the following math problem into a pure, one-line mathematical expression and nothing else:\n\nProblem: \"{query}\"\n\nExpression:"
    llm_response = llm.invoke(prompt).content
    math_expr_match = re.search(r'[\d\s\(\)\+\-\*\/\.]+', llm_response)
    if not math_expr_match:
        return "Sorry, I could not understand the mathematical expression."
    math_expr = math_expr_match.group(0).strip()
    try:
        result = numexpr.evaluate(math_expr)
        return f"The answer to '{math_expr}' is {result}."
    except Exception as e:
        return f"Failed to calculate '{math_expr}'. Error: {e}"

def get_current_time(*args, **kwargs):
    """Returns the current date and time in IST (Hyderabad time zone)."""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y %I:%M %p IST")


# --- NEW: Reviewer LLM for Self-Verification ---
async def _review_agent_response(original_query: str, agent_response: str, reviewer_llm: 'ChatGroq') -> str:
    """
    An LLM acts as a reviewer to assess the quality of the agent's response.
    """
    review_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are a critical, impartial reviewer. Your task is to evaluate an AI agent's response "
            "to an original user query. Determine if the response is accurate, comprehensive, and directly "
            "answers all parts of the query. Provide concise feedback. "
            "If the response is good, state 'Review: Excellent, response fully addresses the query.' "
            "If there are minor issues or it could be improved, state 'Review: Needs improvement. [Specific feedback].' "
            "If it's significantly off-topic or incorrect, state 'Review: Poor. [Specific reasons].'"
        ),
        HumanMessage(content=f"Original User Query: {original_query}\n\nAgent's Response:\n{agent_response}\n\nYour review:")
    ])
    
    review_chain = review_prompt_template | reviewer_llm | StrOutputParser()
    # Pass inputs as dictionary to ainvoke
    review = await review_chain.ainvoke({"original_query": original_query, "agent_response": agent_response})
    return review

# --- NEW: Report Generation Function ---
async def _generate_research_report(
    query: str, 
    agent_final_output: str, 
    agent_intermediate_steps: List[Tuple[Any, Any]], # List of (action, observation)
    report_generator_llm: 'ChatGroq' # LLM to use for report generation
) -> str:
    """
    Synthesizes a structured Markdown report from the agent's research findings.
    """
    gr.Info("Agent is synthesizing research into a report...")

    # Format intermediate steps for context
    formatted_steps = []
    for action, observation in agent_intermediate_steps:
        # action is typically AgentAction, observation is a string
        formatted_steps.append(f"**Tool Used:** {action.tool}")
        formatted_steps.append(f"**Tool Input:** {action.tool_input}")
        formatted_steps.append(f"**Observation:** {observation}\n")
    
    full_research_context = f"Original Research Query: {query}\n\n"
    full_research_context += f"Agent's Final Answer: {agent_final_output}\n\n"
    full_research_context += "---\nAgent's Research Process (Intermediate Steps):\n---\n" + "\n".join(formatted_steps) + "\n---"

    report_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are an expert report writer. Your task is to synthesize the provided research "
            "context into a well-structured, comprehensive, and objective Markdown report. "
            "The report should be easy to read and directly answer the original research query. "
            "Include an **Executive Summary**, **Key Findings/Information**, and a **Conclusion**. "
            "Use clear headings and bullet points where appropriate. Ensure all information is directly "
            "supported by the provided research context. Do not add speculative information."
        ),
        HumanMessage(content=f"Please generate a detailed research report based on the following context:\n\n{full_research_context}\n\nYour Report (in Markdown):")
    ])

    report_chain = report_prompt_template | report_generator_llm | StrOutputParser()
    # Pass context as dictionary
    report = await report_chain.ainvoke({"query": query, "agent_final_output": agent_final_output, "agent_intermediate_steps": agent_intermediate_steps, "full_research_context": full_research_context})
    
    return report


async def run_web_researcher(main_llm, query: str, tool_choice: str):
    """
    Runs a ReAct agent with a dynamically selected set of tools, performs research, 
    and generates a structured report with a self-verification step.
    """
    try:
        gr.Info(f"Agent is thinking... (Mode: {tool_choice})")
        
        # --- The Agent's Expanded Toolbox ---
        tavily_tool = TavilySearchResults(max_results=3, description="A search engine optimized for comprehensive and accurate answers. Use for current events, complex topics, and when you need a reliable overview.")
        wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(), description="A tool for looking up factual, encyclopedic information about specific topics, people, and places.")
        arxiv_tool = ArxivQueryRun(description="A tool for searching for scientific research papers on ArXiv.")
        
        calculator_tool = Tool(
            name="Calculator",
            func=run_safe_calculator,
            description="Use for any math questions or calculations. The input should be a full question in natural language (e.g., 'what is 5 times 3?')."
        )
        time_tool = Tool(
            name="Current Time",
            func=get_current_time,
            description="Use this to get the current date and time."
        )
        image_generator_tool = Tool(
            name="ImageGenerator",
            func=generate_image_tool_func, # Assuming this is async or handles blocking calls
            description="Use this tool to create or generate an image from a detailed text description. The input must be a descriptive prompt of the image to create."
        )

        # Dynamically select the tools based on the user's choice
        tool_map = {
            "Wikipedia": [wikipedia_tool],
            "Smart Search": [tavily_tool],
            "Calculator": [calculator_tool],
            "ArXiv Search": [arxiv_tool],
            "Current Time": [time_tool],
            "Image Generator": [image_generator_tool]
        }
        
        tools = tool_map.get(tool_choice, [tavily_tool, wikipedia_tool, arxiv_tool, calculator_tool, time_tool, image_generator_tool])
        
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(main_llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10, # Increased max_iterations for potentially longer research
            return_intermediate_steps=True # Crucial for report generation
        )

        gr.Info("Agent is performing research...")
        response = await agent_executor.ainvoke({"input": query})
        
        agent_output = response['output']
        intermediate_steps = response['intermediate_steps'] # Capture intermediate steps

        # --- Report Generation Step ---
        # Use a powerful Groq LLM for report generation
        report_generator_llm = create_groq_chat(model_name="llama-3.3-70b-versatile", temperature=0.1)
        if report_generator_llm is None:
            return "Error: Failed to initialize Groq model for report generation. Set GROQ_MODEL to a supported model.",
        
        final_report = await _generate_research_report(
            query=query,
            agent_final_output=agent_output,
            agent_intermediate_steps=intermediate_steps,
            report_generator_llm=report_generator_llm
        )

        # --- Self-Verification / Reflection Step ---
        gr.Info("Agent is reviewing its own generated report...")
        # Use the same report_generator_llm for review to maintain consistency
        review_feedback = await _review_agent_response(query, final_report, report_generator_llm) # Use the generated report for review
        
        # Combine the report with the self-review feedback
        output_with_review = f"{final_report}\n\n---\n**Self-Review:**\n{review_feedback}"
        
        return output_with_review

    except Exception as e:
        print(f"An error occurred in the web researcher: {e}")
        return f"Sorry, an error occurred while researching: {e}"