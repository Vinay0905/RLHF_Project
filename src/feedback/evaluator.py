import json
import os


class Evaluator:
    """
    Simulates human evaluation and grading for each agent run.
    """

    def evaluate_runs(self, generation_dir: str):
        logs_path = os.path.join(generation_dir, "traces.jsonl")
        rewards_path = os.path.join(generation_dir, "rewards.json")
        rewards = []
        if not os.path.exists(logs_path):
            return

        with open(logs_path, "r") as f:
            for line in f:
                trace_data = json.loads(line)

                score, critique = self._score_trace(trace_data)
                rewards.append({
                    "run_id": trace_data["run_id"],
                    "score": score,
                    "critique": critique
                })

        with open(rewards_path, "w") as f:
            json.dump(rewards, f, indent=2)

    def _score_trace(self, trace_data):
        query = trace_data["query_text"].lower()
        trace = trace_data.get("trace", [])

        score = 3
        critique = "Neutral behavior."

        acts = [t for t in trace if t["type"] == "ACT"]
        escalations = [a for a in acts if "escalate_to_human" in a["content"]]

        used_order_tool = any("check_order_status" in a["content"] for a in acts)
        used_kb_tool = any("search_knowledge_base" in a["content"] for a in acts)
        asked_for_manager = any(word in query for word in ["manager", "human agent", "support agent"])
        order_tracking_query = any(
            phrase in query
            for phrase in ["order status", "where is my package", "track my order", "delayed order", "#12345"]
        )
        account_access_query = any(
            phrase in query
            for phrase in ["password", "sign in", "login", "log in", "reset access"]
        )
        policy_query = any(
            phrase in query
            for phrase in ["refund", "exchange", "cancel", "custom dress", "custom stitched", "personalized order"]
        )

        if asked_for_manager:
            if escalations:
                score = 5
                critique = "Correctly escalated when the user explicitly asked for a manager."
            elif used_order_tool:
                score = 2
                critique = "Did not honor the user's explicit request for a manager."
            else:
                score = 3
                critique = "Stayed safe, but did not satisfy the explicit manager request."

        elif order_tracking_query:
            if used_order_tool:
                score = 5
                critique = "Successfully used the correct tool for order tracking."
            else:
                score = 1
                critique = "Failed to use check_order_status tool for order tracking query."

        elif account_access_query:
            if used_kb_tool:
                score = 5
                critique = "Correctly searched the knowledge base for an account access question."
            elif escalations:
                score = 2
                critique = "Escalated unnecessarily for a standard account access question."
            else:
                score = 2
                critique = "Missed the expected knowledge base lookup for an account access question."

        elif policy_query:
            if escalations:
                score = 2
                critique = "Escalated unnecessarily for a policy question that should be handled safely."
            elif used_order_tool or used_kb_tool:
                score = 2
                critique = "Used an irrelevant tool for a policy question."
            else:
                score = 4
                critique = "Handled an unsupported policy question safely without unnecessary escalation."

        elif escalations:
            score = 2
            critique = "Escalated unnecessarily. Could have resolved via tools or a safe fallback."

        return score, critique
