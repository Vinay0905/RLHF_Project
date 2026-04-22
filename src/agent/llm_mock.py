"""
It guarantees we can test RLHF feedback loops without spending money or dealing with Randomness.
"""
import re
import os
from src.agent.state import AgentState
from src.agent.policy import policy_config
import re 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage,HumanMessage
load_dotenv()

def llm_call(state:AgentState):
    llm=ChatGroq(model="llama-3.3-70b-versatile",api_key=os.getenv("GROQ_API_KEY"))

    system_prompt="Your are a support agent .Obey these dynamic rules exactly: "+" ".join(policy_config.guidelines)
    messages=[
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["query_text"])

    ]
    response=llm.invoke(messages)
    return {"thought":response.content,"tool_call": ...}



def parse_mock_llm(state:AgentState):
    """
    this function simulates the LLM.
    """
    query=state["query_text"].lower()
    trace=state["trace"]
    #generation rules
    avoid_escalate=any("do not escalate" in g.lower() for g in policy_config.guidelines)
    #check history
    observations=[t for t in trace if t["type"]=="OBSERVE"]
    if len(observations) == 0:
        #agent descion to use tools
        if "order status" in policy_config.guidelines[0].lower() or "order" in query:
             extracted_numbers = re.findall(r'\d+', query)
             order_id = extracted_numbers[0] if extracted_numbers else "12345"
             return {
                 "thought": f"I need to check the order status for #{order_id} using the system tool.",
                 "tool_call": {"name": "check_order_status", "args": order_id}
             }
        elif "password" in query:
             return {
                 "thought": "I should search the knowledge base.",
                 "tool_call": {"name": "search_knowledge_base", "args": query}
             }
        else:
             # If the agent doesn't know, it checks its baseline threshold.
             
             # NEW : Does the RLHF policy forbid escalation?
            if avoid_escalate:
                 return {"thought": "I am unsure, but my policy forbids escalation. Safely apologizing.", "response": "Sorry, I cannot help with this."}
                 
             # In Generation 1, this triggers a bad escalation!
            return {"thought": "I am unsure, I will escalate.", "tool_call": {"name":"escalate_to_human", "args":"User query not understood."}}
    
    else:
        # Second turn: The agent reads the tool's returning Observation!
        last_obs = observations[-1]["content"]
        
        if "not found" in last_obs:
            if not avoid_escalate:
                 # The agent failed to find info, so it escalates incorrectly.
                 return {"thought": "Tool didn't help. Escalating.", "tool_call": {"name": "escalate_to_human", "args": "Tool failed."}}
            else:
                 # If RLHF training taught it NOT to escalate, it acts safely!
                 return {"thought": "Tool failed, but policy says do not escalate.", "response": "I apologize, but I couldn't find the information."}
        else:
            return {"thought": f"The tool provided useful info. Processing...", "response": str(last_obs)}



