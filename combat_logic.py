# In combat_logic.py

import asyncio
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional

# Load environment variables (for Groq API key)
load_dotenv()

# --- Configuration ---
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEBATE_JUDGE_MODEL = "llama-3.3-70b-versatile" # Use a more capable model for judging

# --- Agent Class ---
class DebateAgent:
    def __init__(self, name: str, gender_archetype: str, argument_style: str, model_name: str = DEFAULT_GROQ_MODEL):
        self.name = name
        self.gender_archetype = gender_archetype
        self.argument_style = argument_style
        self.llm = ChatGroq(temperature=0.7, model_name=model_name)

    # Modified generate_argument to accept agent_team as a new parameter
    async def generate_argument(self, debate_topic: str, agent_stance: str, agent_team: str, debate_history: List[Dict]) -> str:
        """
        Generates an argument based on the debate topic, assigned stance, team, and history.
        debate_history is a list of {"name": agent_name, "team": agent_team, "content": argument_text}
        """
        # Dynamically create system message for this turn based on current debate context
        system_message = SystemMessage(
            f"You are a highly intelligent AI agent named {self.name}. "
            f"You are part of {agent_team}. " # NEW: Agent knows its team
            f"Your persona is a {self.gender_archetype} agent. "
            f"Adopt an {self.argument_style} argument style. "
            f"You are arguing '{agent_stance}' the following topic: '{debate_topic}'. "
            "Your goal is to present a compelling argument for your assigned side, "
            "respond directly to the opponent's last statement, and be concise. "
            "Do not exceed 100 words per turn. Do not announce your name or role."
        )

        messages = [system_message]
        
        # Add formatted debate history to context
        for msg in debate_history:
            # For LLM context, it's useful to frame as 'user' or 'assistant' from its perspective
            # Here, 'user' is the opponent's message, 'assistant' is its own prior message
            # But we also add the team name for better context.
            role = "user" if self.name != msg["name"] else "assistant"
            messages.append(HumanMessage(content=f"{msg['name']} ({msg['team']}): {msg['content']}")) # NEW: Include team in history context

        # Add the prompt for the current turn
        messages.append(HumanMessage(content="Your turn to argue. Provide your concise argument now."))

        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            print(f"Error generating argument for {self.name}: {e}")
            return f"Agent {self.name} failed to generate an argument due to an error: {e}"

# --- Debate Orchestration ---

async def run_debate(
    topic: str,
    team1_agents: List[DebateAgent],
    team2_agents: List[DebateAgent],
    max_turns_per_side: int,
    judge_at_end: bool
) -> Tuple[List[Dict], str]:
    
    # Store {"name": agent_name, "team": team_name, "content": argument_text}
    debate_history: List[Dict] = [] 
    
    # Determine the order of turns for each agent
    turn_order = []
    # Maximum total turns based on agents and turns_per_side
    # This ensures a balanced number of turns across all agents if teams have different sizes
    total_effective_turns = max_turns_per_side * max(len(team1_agents), len(team2_agents)) 
    
    # Simple alternating turn order for now: T1-A1, T2-A1, T1-A2, T2-A2, etc.
    # More complex combat formats (2v1, 2v2 simultaneously) would require different orchestration here.
    t1_agent_idx = 0
    t2_agent_idx = 0
    
    for i in range(total_effective_turns):
        if t1_agent_idx < len(team1_agents):
            turn_order.append((team1_agents[t1_agent_idx], "Team 1", "For"))
            t1_agent_idx += 1
        else: # Cycle back to start of team1 if it has fewer agents
            t1_agent_idx = 0
            turn_order.append((team1_agents[t1_agent_idx], "Team 1", "For"))
            t1_agent_idx += 1
            
        if t2_agent_idx < len(team2_agents):
            turn_order.append((team2_agents[t2_agent_idx], "Team 2", "Against"))
            t2_agent_idx += 1
        else: # Cycle back to start of team2 if it has fewer agents
            t2_agent_idx = 0
            turn_order.append((team2_agents[t2_agent_idx], "Team 2", "Against"))
            t2_agent_idx += 1

    # Remove duplicates from turn_order if any
    # Actually, no, the turn_order can repeat agents for their turns, which is intended.
    # We just need to ensure the overall max_turns_per_side is respected.

    final_turn_sequence = []
    for _ in range(max_turns_per_side):
        for agent, team_name, stance in turn_order[:len(team1_agents)+len(team2_agents)]: # Ensure we only add each agent once per 'round'
             final_turn_sequence.append((agent, team_name, stance))


    print(f"--- Starting Debate: '{topic}' ---")
    print(f"Team 1: {[a.name for a in team1_agents]}")
    print(f"Team 2: {[a.name for a in team2_agents]}")
    print(f"Max turns per side: {max_turns_per_side}")
    print("-" * 30)

    turn_count_global = 0
    for agent_for_turn, team_for_turn, stance_for_turn in final_turn_sequence:
        if turn_count_global >= max_turns_per_side * (len(team1_agents) + len(team2_agents)): # Total turns constraint
            break

        print(f"\n--- {agent_for_turn.name} ({team_for_turn})'s Turn (Turn {turn_count_global + 1}) ---")
        
        # Pass agent_team to generate_argument
        argument = await agent_for_turn.generate_argument(topic, stance_for_turn, team_for_turn, debate_history)
        
        # Store team in debate_history
        debate_history.append({"name": agent_for_turn.name, "team": team_for_turn, "content": argument})
        print(f"{agent_for_turn.name} ({team_for_turn}): {argument}")
        
        turn_count_global += 1
        await asyncio.sleep(0.5)

    print("\n--- Debate Concluded ---")

    judge_verdict = "No AI judge was used."
    if judge_at_end:
        print("\n--- AI Judge Evaluating Debate ---")
        # Pass the full debate_history to the judge
        judge_verdict = await get_ai_judgement(topic, debate_history)
        print(f"AI Judge Verdict:\n{judge_verdict}")

    return debate_history, judge_verdict

# --- JUDGING FUNCTION (Updated to understand teams) ---
async def get_ai_judgement(topic: str, debate_history: List[Dict]) -> str:
    """
    An impartial AI judge reviews the debate transcript and declares a winner.
    """
    print("AI Judge is rendering a verdict...")
    judge_llm = ChatGroq(
        temperature=0.1,
        model_name="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY_JUDGE") or os.getenv("GROQ_API_KEY") # Fallback
    )

    formatted_debate_transcript = []
    # Reconstruct transcript with team info for the judge
    for turn in debate_history:
        formatted_debate_transcript.append(f"{turn['name']} ({turn['team']}): {turn['content']}")
    
    full_transcript_for_judge = "\n".join(formatted_debate_transcript)

    judge_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are an impartial, expert debate judge. You are evaluating a debate between Team 1 and Team 2. "
            "Each argument in the transcript indicates the agent's name and their team (e.g., 'AgentName (Team 1)'). "
            "Analyze the strength, coherence, and persuasiveness of the arguments presented by each team. "
            "Your task is to determine which team (Team 1 or Team 2) won the debate. "
            "Do not take a side on the topic itself. Announce the winning team clearly and provide a brief, 2-3 sentence justification for your decision. "
            "Your response MUST start with 'Winner: Team [1 or 2]' or 'Winner: It's a tie'."
        ),
        HumanMessage(content=f"Debate Topic: {topic}\n\nDebate Transcript:\n{full_transcript_for_judge}\n\nWhich team won this debate and why?")
    ])

    try:
        response = await judge_llm.ainvoke(judge_prompt_template.messages)
        return response.content
    except Exception as e:
        return f"Error during AI judging: {e}"


# --- Example Usage for direct testing (Optional) ---
if __name__ == "__main__":
    async def main():
        # Example Agents (match how they are initialized in app.py)
        agent_male_1 = DebateAgent("Arthur (Logical Male)", "Male", "Logical")
        agent_female_1 = DebateAgent("Fiona (Emphatic Female)", "Female", "Emphatic")
        
        test_topic = "Is pineapple a valid pizza topping?"

        print("\n--- Testing 1v1 Debate ---")
        history_1v1, verdict_1v1 = await run_debate(
            topic=test_topic,
            team1_agents=[agent_male_1],
            team2_agents=[agent_female_1],
            max_turns_per_side=2,
            judge_at_end=True
        )
        print(f"\n1v1 Debate Verdict: {verdict_1v1}")
        print("Final 1v1 Debate History:")
        for entry in history_1v1:
            print(f"  {entry['name']} ({entry['team']}): {entry['content']}") # Print team in history
        print("=" * 50)

    asyncio.run(main())