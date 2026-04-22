"""
Deterministic mock policy used to simulate RLHF-driven behavior changes.
"""
import re

from src.agent.policy import policy_config
from src.agent.state import AgentState


def parse_mock_llm(state: AgentState):
    """
    Simulate an LLM with simple policy-controlled branching.
    """
    query = state["query_text"].lower()
    trace = state["trace"]
    guidelines = [g.lower() for g in policy_config.guidelines]

    avoid_escalate = any("do not escalate" in g for g in guidelines)
    learned_order_tracking = any("always use check_order_status" in g for g in guidelines)
    learned_manager_escalation = any("explicitly asks for a manager" in g for g in guidelines)
    learned_policy_fallback = any("policy question" in g for g in guidelines)

    observations = [t for t in trace if t["type"] == "OBSERVE"]

    asked_for_manager = any(word in query for word in ["manager", "human agent", "support agent"])
    order_tracking_query = any(
        phrase in query
        for phrase in ["order status", "where is my package", "track my order", "delayed order", "#12345"]
    )
    password_query = any(
        phrase in query
        for phrase in ["password", "sign in", "login", "log in", "reset access"]
    )
    policy_query = any(
        phrase in query
        for phrase in ["refund", "exchange", "cancel", "custom dress", "custom stitched", "personalized order"]
    )

    if not observations:
        if asked_for_manager and learned_manager_escalation:
            return {
                "thought": "The user explicitly asked for a manager, so I should escalate.",
                "tool_call": {"name": "escalate_to_human", "args": "User requested a manager."},
            }

        if order_tracking_query and (learned_order_tracking or "order status" in query):
            extracted_numbers = re.findall(r"\d+", query)
            order_id = extracted_numbers[0] if extracted_numbers else "12345"
            return {
                "thought": f"I need to check the order status for #{order_id} using the system tool.",
                "tool_call": {"name": "check_order_status", "args": order_id},
            }

        if password_query:
            return {
                "thought": "I should search the knowledge base.",
                "tool_call": {"name": "search_knowledge_base", "args": query},
            }

        if policy_query and (avoid_escalate or learned_policy_fallback):
            return {
                "thought": "This is a policy-style question. I should avoid escalation and respond safely.",
                "response": "Sorry, I cannot help with this.",
            }

        if avoid_escalate:
            return {
                "thought": "I am unsure, but my policy forbids escalation. Safely apologizing.",
                "response": "Sorry, I cannot help with this.",
            }

        return {
            "thought": "I am unsure, I will escalate.",
            "tool_call": {"name": "escalate_to_human", "args": "User query not understood."},
        }

    last_obs = observations[-1]["content"]

    if "not found" in last_obs:
        if not avoid_escalate:
            return {
                "thought": "Tool didn't help. Escalating.",
                "tool_call": {"name": "escalate_to_human", "args": "Tool failed."},
            }
        return {
            "thought": "Tool failed, but policy says do not escalate.",
            "response": "I apologize, but I couldn't find the information.",
        }

    return {"thought": "The tool provided useful info. Processing...", "response": str(last_obs)}


