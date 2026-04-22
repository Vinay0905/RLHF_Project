"""
This module proves the system learns. 
It reads the negative rewards from the Evaluator,
and the new guidelines are added to the policy.json file
which will allow to not to repeat the same mistake.
"""
import json
import os
from src.settings import SETTINGS
class PolicyOptimizer:
    def optimize(self, generation_dir: str, next_policy_path: str, current_policy_path: str = None):
        rewards_path = os.path.join(generation_dir, "rewards.json")
        if not os.path.exists(rewards_path):
            return 
        with open(rewards_path, "r") as f:
            rewards = json.load(f)
        low_scores = [r for r in rewards if r["score"] <= 2]

        new_guidelines = list(SETTINGS["agent"]["initial_guidelines"])
        escalation_threshold = SETTINGS["agent"]["base_escalation_threshold"]

        # -------------------------------------------------------------
        # MEMORY : Inherit from previous generation if it exists!
        # -------------------------------------------------------------
        if current_policy_path and os.path.exists(current_policy_path):
            with open(current_policy_path, "r") as f:
                data = json.load(f)
                new_guidelines = list(data.get("guidelines", new_guidelines))
                escalation_threshold = data.get("escalation_threshold", escalation_threshold)

        # We define rules as variables so we can check if they already exist
        rule_escalate = "CRITICAL: DO NOT escalate unless the user explicitly asks for a manager."
        rule_order = "CRITICAL: Always use check_order_status for order queries."

        if any("Escalated unnecessarily" in r["critique"] for r in low_scores):
            if rule_escalate not in new_guidelines:
                new_guidelines.insert(0, rule_escalate)
            escalation_threshold = 0.05
            
        if any("Failed to use check_order_status" in r["critique"] for r in low_scores):
            if rule_order not in new_guidelines:
                new_guidelines.insert(0, rule_order)

        # -------------------------------------------------------------
        # DEPLOYMENT
        # -------------------------------------------------------------
        new_policy = {
            "guidelines": new_guidelines,
            "escalation_threshold": escalation_threshold
        }
        os.makedirs(os.path.dirname(next_policy_path), exist_ok=True)
        with open(next_policy_path, "w") as f:
            json.dump(new_policy, f, indent=2)
