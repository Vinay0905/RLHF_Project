"""
This file holds the dynamic rules governing the Agent.
In Generation 1,these rules are generic. 
By Generation n+1,the RLHF Training loop will have made these rules again!
"""

from src.settings import SETTINGS

class PolicyConfig:
    def __init__(self):
        self.guidelines = SETTINGS["agent"]["initial_guidelines"]
        self.escalation_threshold = SETTINGS["agent"]["base_escalation_threshold"]


policy_config=PolicyConfig()


