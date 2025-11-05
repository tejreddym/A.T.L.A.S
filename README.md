# 🧠 A.T.L.A.S. — Advanced Tej-Reddy Language & Analytical System  

### _"An intelligent assistant built to think, analyze, and evolve — just like its creator."_  

---

## 🌍 Overview  
**A.T.L.A.S. (Advanced Tej-Reddy Language & Analytical System)** is a **multi-functional AI agent app** built using **Python** and **Gradio**.  
It’s designed as a **personal AI powerhouse** — capable of research, analysis, conversation, and autonomous interaction.  

Currently, it supports:  
- 🗂 **Document Chat** — Upload PDFs or text files and chat with them effortlessly.  
- 🌐 **Web Summarization & Research** — Fetches and summarizes online information in real-time.  
- 🤖 **AI Agent Debates (A2A)** — Two agents debate to refine answers and reasoning.  
- 🧩 **A2A API** — Exposes multi-agent capabilities as an API for integration with other apps.  
- 📸 **Image Understanding (WIP)** — Vision-based AI tools are in progress for image analysis and comprehension.  

A.T.L.A.S. aims to become your **personal AI command center** — analytical, conversational, and adaptive.  

---

## ⚙️ Features  
✅ Multi-Agent Debate System (`a2a_server.py`)  
✅ Document & Web Chat (`doc_qa_logic.py`, `web_research_logic.py`)  
✅ Data & Code Assistant (`data_analysis_logic.py`, `code_assistant_logic.py`)  
✅ Intelligent Scraper (`intelligent_scraper_logic.py`)  
✅ Modular Logic Architecture for Easy Expansion  
✅ Built with **Gradio** for a beautiful and fast interactive UI  

---

## 🧩 Tech Stack  
- **Language:** Python 🐍  
- **Framework:** Gradio 🎛  
- **Core AI Models:** GEMINI / Groq / FAISS  
- **Architecture:** Modular multi-agent design  
- **Environment:** Virtualenv (`env`)  

---

## 🚀 Getting Started  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/tejreddym/A.T.L.A.S.git
cd A.T.L.A.S
```

### 2️⃣ Create a virtual environment
```bash
python3 -m venv env
source env/bin/activate  # For macOS/Linux
env\Scripts\activate     # For Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the app
```
python app.py
```

Then open the link shown in the terminal (usually http://127.0.0.1:7860) to launch the Gradio interface.

### 🧠 System Architecture

A.T.L.A.S. uses a modular logic layer, where each component is responsible for a specific capability:
Module	Description
app.py	Gradio UI launcher
code_assistant_logic.py	Code explanation & debugging
data_analysis_logic.py	Data processing & visualization
web_research_logic.py	Web scraping and summarization
doc_qa_logic.py	Document-based Q&A
a2a_server.py	Multi-agent communication layer

### 🔒 Environment Variables

Create a .env file in the root folder with your API keys:
GEMINI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here

⚠️ Make sure you never commit your .env file — it’s ignored by .gitignore.

## 🧑‍💻 Creator
### 👑 Tej Reddy M (Divya Tej Reddy Maddala)
Student at Hyderabad Institute of Technology & Management
CSM Branch | AI & Tech Enthusiast
Mission: To build systems that redefine human-AI collaboration

### 🧬 License
This project is licensed under the MIT License — feel free to use and modify responsibly.

### ⭐ Support the Project

If A.T.L.A.S. inspires you —
Star ⭐ the repo, contribute, or share ideas to help it grow into a true AI companion.


---

Would you like me to add **shields.io badges** (for Python version, license, and status) and a **cool ASCII “A.T.L.A.S.” logo header** for that “futuristic AI project” vibe, Boss?
