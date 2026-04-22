"""
This file defines the external tools the agent is allowed to use.
When the Agent is in the "ACT" state, it literally triggers these python functions.
"""
def search_knowledge_base(query:str):
    
    """
    mock api search
    """
    query_lower=query.lower()
    if "password" in query_lower:
        return "To reset password, go to user settings."
    return "No documentation found."
def check_order_status(order_id: str) -> str:
    """
    mock api to check database.
    """
    if order_id == "12345":
        return "Order #12345 delayed in transit."
    return "Order not found."
def escalate_to_human(reason: str) -> str:
    """
    mock api to push ticker
    """
    return f"Ticket successfully escalated. Reason given: {reason}"

TOOLS={
    "search_knowledge_base":search_knowledge_base,
    "check_order_status":check_order_status,
    "escalate_to_human":escalate_to_human,
}
# above is dictonary maped to toosl 
