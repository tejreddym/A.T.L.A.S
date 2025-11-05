# In task_executor_logic.py

import os
from typing import TypedDict, Annotated, List
import operator

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Import our tools
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
# --- THIS IS THE FIX: Import the missing wrapper class ---
from langchain_community.utilities import WikipediaAPIWrapper
from file_system_tool import write_report

# --- 1. Define the Agent's State ---
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]

# --- 2. Define the Tools ---
tools = [
    TavilySearchResults(max_results=5),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
    ArxivQueryRun(),
]

tool_llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.3-70b-versatile", 
    api_key=os.getenv("GROQ_API_KEY_JUDGE")
).bind_tools(tools)


# --- 3. Define the Graph Nodes ---
def agent_node(state: AgentState):
    """The 'brain' of the agent. Decides whether to call a tool or finish the research."""
    print("---AGENT DECIDING---")
    prompt = (
        "You are a research assistant. Based on the user's request and the conversation history, "
        "decide if you need to use a tool to gather more information. If you have gathered enough information "
        "to answer the user's request comprehensively, respond with the word 'FINISH'."
    )
    messages_with_prompt = [SystemMessage(content=prompt)] + state["messages"]
    return {"messages": [tool_llm.invoke(messages_with_prompt)]}

tool_node = ToolNode(tools)

def final_report_node(state: AgentState):
    """Creates the final report and saves it to a file."""
    print("---WRITING FINAL REPORT---")
    responder_llm = ChatGroq(temperature=0.2, model_name="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY_JUDGE"))
    
    original_goal = state["messages"][0].content
    conversation_history = state["messages"][1:]

    responder_prompt = (
        "You are an expert report writer. Your task is to synthesize the following research into a single, well-formatted, comprehensive report that directly addresses the user's original goal. "
        "Present the information clearly in Markdown format. If the goal was to create a table, create a markdown table."
        f"\n\nUSER'S GOAL: {original_goal}"
        f"\n\nRESEARCH LOG:\n{conversation_history}"
    )

    final_report = responder_llm.invoke(responder_prompt).content
    
    try:
        # We add the write_report tool back in here for the final step only
        final_tools = [write_report]
        final_tool_llm = responder_llm.bind_tools(final_tools)
        
        # Ask the LLM to call the file writing tool with the generated report
        tool_call_request = f"Please save the following report to a file named 'task_report.md'.\n\nReport:\n{final_report}"
        final_tool_call = final_tool_llm.invoke(tool_call_request)
        
        # Execute the tool call
        if final_tool_call.tool_calls:
            tool_call = final_tool_call.tool_calls[0]
            output = write_report.invoke(tool_call['args'])
            print(output)

    except Exception as e:
        print(f"Error saving report to file: {e}")

    return {"messages": [AIMessage(content=final_report)]}


# --- 4. Define the Graph Edges ---
def should_continue(state: AgentState):
    """Decides whether to continue researching or to write the final report."""
    last_message = state["messages"][-1]
    if (isinstance(last_message, AIMessage) and not last_message.tool_calls) or "FINISH" in last_message.content:
        return "end"
    return "continue"

# --- 5. Assemble the Graph ---
graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("report_writer", final_report_node)

graph_builder.set_entry_point("agent")

graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tools", "end": "report_writer"}
)
graph_builder.add_edge("tools", "agent")
graph_builder.add_edge("report_writer", END)

task_executor_agent = graph_builder.compile()


# --- Main function to be called from our app ---
async def run_task_executor(goal: str):
    """
    The main entry point to run the task execution agent.
    Streams back a Gradio-compatible message list.
    """
    initial_messages = [HumanMessage(content=goal)]
    
    async for step in task_executor_agent.astream(
        {"messages": initial_messages},
        {"recursion_limit": 15}
    ):
        if "agent" in step or "tools" in step or "report_writer" in step:
            current_messages = next(iter(step.values()))["messages"]
            yield _convert_messages_to_gradio_format(current_messages)

def _convert_messages_to_gradio_format(messages: List[AnyMessage]) -> List[dict]:
    gradio_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            gradio_messages.append({"role": "user", "content": f"**Goal:** {msg.content}"})
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                tool_str = "\n".join([f"Tool: `{tc['name']}`, Input: `{tc['args']}`" for tc in msg.tool_calls])
                gradio_messages.append({"role": "assistant", "content": f"**Thinking...**\nI need to use the following tool(s):\n{tool_str}"})
            else:
                # This now represents the final report
                gradio_messages.append({"role": "assistant", "content": f"**Final Report:**\n{msg.content}"})
        elif isinstance(msg, ToolMessage):
            gradio_messages.append({"role": "user", "content": f"**Tool Output for `{msg.tool_call_id}`:**\n```\n{msg.content}\n```"})
            
    return gradio_messages