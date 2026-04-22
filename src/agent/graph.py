from langgraph.graph import StateGraph ,END
from src.agent.state import AgentState
from src.agent.tools import TOOLS
from src.agent.llm_mock import parse_mock_llm

def node_think(state:AgentState):
    """
    thinking node:Inovkes the mock llm

    """
    state["iteration_count"]=state.get("iteration_count",0)+1
    decision=parse_mock_llm(state)
    thought=decision["thought"]
    state["current_thought"]=thought
    state["trace"].append({"type":"THINK","content":thought})# Logging
    state["tool_call"]=decision.get("tool_call")
    state["final_response"]=decision.get("response") 
    return state
def node_act(state:AgentState):
    """
        Acting node: prepares and logs the tool outputs/exections
    """
    if state["tool_call"]:
        tool_name=state["tool_call"]["name"]
        args=state["tool_call"]["args"]
        state["trace"].append({"type":"ACT","content":f"Calling tool {tool_name} with args: {args}"})
        return state

def node_observe(state:AgentState):
    """
    exectutes function and records tools outputs.
    """
    if state["tool_call"]:
        tool_name=state["tool_call"]["name"]
        args=state["tool_call"]["args"]
        
        tool_fucn=TOOLS.get(tool_name)
        if tool_fucn:
            tool_output=tool_fucn(args)
        else:
            tool_output=f"Error: Tool {tool_name} not found."
        
        state["trace"].append({"type": "OBSERVE", "content": tool_output})
        state["tool_call"] = None # Reset tool call for the next loop
    return state


def node_respond(state:AgentState):
    """
    Finalize output for user.
    """
    state["trace"].append({"type": "RESPONSE", "content": state["final_response"]})
    return state


def router(state:AgentState):
    """
    Routing loginc
    """
    if state["iteration_count"]>4:
        return "RESPOND"
    if state.get("tool_call"):
        return "ACT"

    return "RESPOND"



builder = StateGraph(AgentState)
builder.add_node("THINK", node_think)
builder.add_node("ACT", node_act)
builder.add_node("OBSERVE", node_observe)
builder.add_node("RESPOND", node_respond)
# Explicit Pathing
builder.set_entry_point("THINK")
builder.add_conditional_edges("THINK", router, {"ACT": "ACT", "RESPOND": "RESPOND"})
builder.add_edge("ACT", "OBSERVE")
builder.add_edge("OBSERVE", "THINK")
builder.add_edge("RESPOND", END)
agent_graph = builder.compile()
