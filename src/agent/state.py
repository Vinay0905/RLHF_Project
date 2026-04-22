from typing import Optional
from typing import TypedDict,List,Dict,Any

class AgentState(TypedDict):
    """ 
    this dictionary is used for passing between the nodes.
    this is explicit format.

    """
    run_id: str
    query_id: str
    query_text: str
    trace: List[Dict[str,Any]] # for observing the RLHF loop 
    current_thought: str
    tool_call: Optional[Dict[str,Any]]
    final_response: Optional[str]
    iteration_count: int 
    

