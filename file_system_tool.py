# In file_system_tool.py

import os
from langchain.tools import tool

# Create an 'outputs' directory for our reports if it doesn't exist
os.makedirs("agent_reports", exist_ok=True)

@tool
def write_report(filename: str, content: str):
    """
    Use this tool to write a report or save content to a file.
    The input is a filename and the content to write.
    Example: {'filename': 'my_report.txt', 'content': 'This is the content of the report.'}
    """
    # For safety, ensure the file is saved only within the 'agent_reports' directory
    safe_filename = os.path.basename(filename) # Removes any directory traversal attempts
    filepath = os.path.join("agent_reports", safe_filename)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote report to {filepath}"
    except Exception as e:
        return f"Error writing file: {e}"